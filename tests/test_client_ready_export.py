"""Client-readiness regression tests (2026-07-02).

* Issue A: the export must NOT delete position-valid epochs. A missing
  velocity sigma (the epoch-weighted smoother path does not propagate
  ``sd_v*``) marks the row ``vel_trusted=0`` -- it never drops the row.
  An honest horizontal 2-sigma over the 6 m bar is KEPT and flagged
  ``pos_within_bar=0``. On day14 dodge190336 this restores the client CSV
  from ~477 rows back to ~1412 (input epoch count minus robust-filter
  repairs), with honesty preserved via the flag columns.

* Issue B: the CLI (``scripts/run_pipeline_from_raw.py``) must ship the
  sync player (media + path + Motion sensor trust panel) like the GUI does,
  wired with the same inputs, and must never crash the run if the build
  fails.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from data_pipeline.parsers import PosRow
from data_pipeline.stages.user_export import export_trajectory


def _read(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    return list(rdr), rdr.fieldnames


def _smoothed_like_row(utc, sd_xy, quality=2, lat=32.06, lon=34.79):
    """A row shaped like the CLI's epoch-weighted smoothed output:
    position + position sigma present, velocity sigma ABSENT (NaN)."""
    return PosRow(
        utc_s=utc, lat_deg=lat, lon_deg=lon, h_m=55.0, quality=quality,
        ns=20, vn=1.0, ve=0.5, vu=0.0,
        sd_n=sd_xy, sd_e=sd_xy, sd_u=sd_xy * 2,
        # sd_vn / sd_ve / sd_vu left at their NaN defaults.
    )


def test_missing_velocity_sigma_never_drops_position_rows(tmp_path):
    """Issue A core: rows with valid position sigma but NO velocity sigma
    must all ship (marked vel_trusted=0), not be suppressed."""
    rows = [_smoothed_like_row(1000.0 + i, 0.3) for i in range(50)]
    assert all(not math.isfinite(r.sd_vn) for r in rows)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False)
    assert res.n_input_rows == 50
    assert res.n_rows == 50          # nothing deleted
    assert res.n_dropped_rows == 0
    assert res.n_vel_untrusted == 50  # honestly marked
    data, cols = _read(res.csv_path)
    assert "vel_trusted" in cols and "pos_within_bar" in cols
    assert all(d["vel_trusted"] == "0" for d in data)


def test_over_bar_epochs_ship_with_flag_not_deleted(tmp_path):
    """Issue A day14 shape: honest (inflated) sigma marginally above the
    6 m bar must not delete the epoch -- it ships with pos_within_bar=0."""
    rows = [_smoothed_like_row(1000.0 + i, 0.3) for i in range(40)]
    for i in range(10, 30):
        rows[i] = _smoothed_like_row(1000.0 + i, 4.0)  # 2-sigma well over 6 m
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False)
    assert res.n_rows == 40
    assert res.n_dropped_rows == 0
    assert res.n_flagged_over_bar == 20
    assert len(res.flagged_sections) == 1
    sec = res.flagged_sections[0]
    assert sec.n_epochs == 20 and sec.reason == "horizontal"
    data, _ = _read(res.csv_path)
    n_flag0 = sum(1 for d in data if d["pos_within_bar"] == "0")
    assert n_flag0 == 20
    # Honesty preserved: the flagged rows still carry their large 2-sigma.
    worst = max(float(d["err_horiz_2sigma_m"]) for d in data
                if d["pos_within_bar"] == "0")
    assert worst > 6.0


def test_summary_text_reports_flags_and_coverage(tmp_path):
    rows = [_smoothed_like_row(1000.0 + i, 0.3) for i in range(20)]
    rows[5] = _smoothed_like_row(1005.0, 4.0)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False)
    txt = res.summary_text()
    assert "over-bar" in txt
    assert "vel untrusted" in txt
    assert "coverage" in txt


def test_hard_drop_mode_still_available(tmp_path):
    """Legacy behaviour stays reachable for callers that want the old
    delete-over-bar export."""
    rows = [_smoothed_like_row(1000.0 + i, 0.1, quality=1) for i in range(20)]
    for i in range(8, 12):
        rows[i] = _smoothed_like_row(1000.0 + i, 4.0, quality=1)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False,
                            hard_drop_over_bar=True)
    assert res.n_dropped_rows == 4
    assert res.n_rows == 16


# ---------------------------------------------------------------------------
# PP5 honesty: a HARD-DROPPED filter run must leave a visible gap marker.
# ---------------------------------------------------------------------------

_LAT0, _LON0, _H0 = 32.06, 34.80, 47.0
_MLAT = 111_320.0
_MLON = 111_320.0 * math.cos(math.radians(_LAT0))


def _drive_row(t, east_m, h=_H0):
    """1 Hz eastward drive row with full position + velocity sigmas."""
    return PosRow(
        utc_s=float(t),
        lat_deg=_LAT0, lon_deg=_LON0 + east_m / _MLON, h_m=float(h),
        quality=2, ns=12,
        sd_n=0.4, sd_e=0.4, sd_u=0.8,
        vn=0.0, ve=13.0, vu=0.0,
        sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
    )


def test_hard_dropped_run_flags_surviving_neighbours_with_gap(tmp_path):
    """A run long enough to be hard-DROPPED (not repaired) by the winning
    filter must leave gap=1 on the surviving rows bracketing the hole.

    Regression: the filter sets gap=True only on the dropped epochs, whose
    utc_s never appear among the surviving rows -- so previously NO written
    row carried gap=1 and the client silently bridged the hole (PP5
    violation).
    """
    n = 120
    rows = [_drive_row(1000.0 + i, east_m=13.0 * i) for i in range(n)]
    # 20-epoch altitude blow-up (+100 m vs the session median): way past the
    # winning preset's alt_above_median_m=40 and max_repair_epochs=10 /
    # max_repair_seconds=12 -> the whole run is hard-dropped.
    for i in range(40, 60):
        rows[i] = _drive_row(1000.0 + i, east_m=13.0 * i, h=_H0 + 100.0)

    res = export_trajectory(rows, tmp_path / "t.csv",
                            suppress_inaccurate=False)  # filter default ON
    assert res.n_filter_dropped >= 10, "test premise: the run must be DROPPED"
    assert res.n_filter_repaired == 0

    data, cols = _read(res.csv_path)
    assert "gap" in cols
    times = [float(d["gpstime"]) for d in data]

    # The hole is real: none of the spiked epochs survived.
    hole_lo, hole_hi = 1000.0 + 40, 1000.0 + 59
    ls = times[0] - 1000.0  # epoch offset offset baked into reference time
    assert all(not (hole_lo + ls - 0.5 <= t <= hole_hi + ls + 0.5)
               for t in times)

    # Find the hole in the surviving series and require BOTH bracketing
    # surviving rows to carry the gap marker -- the hole must be visible,
    # never silently bridgeable.
    hole_edges = [i for i in range(len(times) - 1)
                  if times[i + 1] - times[i] > 5.0]
    assert len(hole_edges) == 1, "exactly one hole expected"
    i = hole_edges[0]
    assert data[i]["gap"] == "1", "surviving row BEFORE the hole not flagged"
    assert data[i + 1]["gap"] == "1", "surviving row AFTER the hole not flagged"
    # And the flags are specific: far-away clean rows stay gap=0.
    assert data[0]["gap"] == "0"
    assert data[-1]["gap"] == "0"


# ---------------------------------------------------------------------------
# Issue B: sync player wired into the CLI
# ---------------------------------------------------------------------------

class _FakeRaw:
    """Duck-typed RawInputs stand-in for the CLI sync-player step."""

    def __init__(self, with_video=True):
        self.recording_mp4 = Path("v.mp4") if with_video else None
        self.recording_txt = Path("rec.txt")
        self.sensors_txt = Path("sensors.txt")
        self.measurements_txt = Path("meas.txt")
        self.capture_meta_json = Path("capture_meta.json")
        self.audio_anchor_txt = Path("audio_anchor.txt")
        self.video_anchor_txt = Path("video_anchor.txt")
        self.audio_wav = Path("audio.wav")


def test_cli_sync_player_step_wires_gui_inputs(monkeypatch, tmp_path):
    from scripts.run_pipeline_from_raw import build_sync_player_step
    from data_pipeline.stages import viewers

    calls = {}

    def fake_build(**kw):
        calls.update(kw)

    monkeypatch.setattr(viewers, "build_sync_player", fake_build)
    raw = _FakeRaw(with_video=True)
    ok = build_sync_player_step(
        raw=raw, pos_path=Path("rover.pos"),
        frame_times_csv=Path("ft.csv"), stat_path=Path("rover.pos.stat"),
        out_html=tmp_path / "sync_player.html", log=lambda *_: None,
    )
    assert ok is True
    # Same wiring the GUI uses (gui.py::_run_sync_player).
    assert calls["video"] == raw.recording_mp4
    assert calls["sensors_txt"] == raw.sensors_txt
    assert calls["wav"] == raw.audio_wav
    assert calls["audio_anchor"] == raw.audio_anchor_txt
    assert calls["capture_meta"] == raw.capture_meta_json
    assert calls["video_anchor"] == raw.video_anchor_txt
    assert calls["recording_map"] == raw.recording_txt
    assert calls["data_log"] == raw.measurements_txt
    assert calls["out_html"] == tmp_path / "sync_player.html"


def test_cli_sync_player_step_skips_without_video(monkeypatch, tmp_path):
    from scripts.run_pipeline_from_raw import build_sync_player_step
    from data_pipeline.stages import viewers

    def boom(**kw):
        raise AssertionError("must not be called without video")

    monkeypatch.setattr(viewers, "build_sync_player", boom)
    ok = build_sync_player_step(
        raw=_FakeRaw(with_video=False), pos_path=Path("rover.pos"),
        frame_times_csv=Path("ft.csv"), stat_path=None,
        out_html=tmp_path / "sync_player.html", log=lambda *_: None,
    )
    assert ok is False


def test_sync_player_uses_time_anchor_fallback(monkeypatch, tmp_path):
    """day14-style sessions ship a 0-byte recording_*.txt; build_sync_player
    must go through time_sync.fit_time_anchor_with_fallback (session map +
    measurements) exactly like coordinate output does, not bare fit_time_anchor."""
    import pytest as _pytest
    from data_pipeline import time_sync
    from data_pipeline.stages import viewers

    rec = tmp_path / "recording_x.txt"
    rec.write_text("", encoding="utf-8")  # the day14 0-byte case
    meas = tmp_path / "measurements_x.txt"
    meas.write_text("# fake\n", encoding="utf-8")

    seen = {}

    def sentinel(recording_path, measurements_path=None, **kw):
        seen["rec"] = Path(recording_path)
        seen["meas"] = measurements_path
        raise RuntimeError("sentinel-stop")

    monkeypatch.setattr(time_sync, "fit_time_anchor_with_fallback", sentinel)
    with _pytest.raises(RuntimeError, match="sentinel-stop"):
        viewers.build_sync_player(
            video=tmp_path / "v.mp4",
            pos_file=tmp_path / "rover.pos",
            frame_times_csv=tmp_path / "ft.csv",
            recording_map=rec,
            out_html=tmp_path / "sync.html",
            data_log=meas,
        )
    assert seen["rec"] == rec
    assert seen["meas"] == meas  # measurements wired as the fallback source


def test_cli_sync_player_step_never_crashes_the_run(monkeypatch, tmp_path):
    from scripts.run_pipeline_from_raw import build_sync_player_step
    from data_pipeline.stages import viewers

    def boom(**kw):
        raise RuntimeError("synthetic viewer failure")

    monkeypatch.setattr(viewers, "build_sync_player", boom)
    logs = []
    ok = build_sync_player_step(
        raw=_FakeRaw(with_video=True), pos_path=Path("rover.pos"),
        frame_times_csv=Path("ft.csv"), stat_path=None,
        out_html=tmp_path / "sync_player.html", log=logs.append,
    )
    assert ok is False
    assert any("WARN" in str(ln) for ln in logs)
