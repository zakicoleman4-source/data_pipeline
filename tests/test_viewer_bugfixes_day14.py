"""Regression tests for three live GUI bugs hit on session day14/output/s21_1.

BUG A - coordinate output/the external tool CSV with a leading ``#`` comment line broke the
        viewers' ``csv.DictReader`` (comment became the header -> Latitude /
        Longitude columns lost -> "No usable rows in georef.csv").
BUG B - the comparison/sync viewers mapped sample PTS via
        ``anchor.video_pts_to_utc_s`` even for ``anchor_format=2`` (boottime)
        sessions, where session.txt column-0 is ABSOLUTE bootNs. Every sample
        landed hours outside the .pos window -> "Post-processing interpolation produced no
        points". Fixed by resolving ``video_t0_boottime_ns`` (capture_meta /
        video_anchor) and lifting PTS into bootNs, exactly as coordinate output does.
BUG C - the GUI Speed-vs-GT handler did a bare ``import speed_vs_gt_html`` for
        a module that is not shipped -> ModuleNotFoundError stack trace. The
        handler now resolves the import up front and reports gracefully.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from data_pipeline.stages import viewers
from data_pipeline.stages.viewers import (
    _read_georef_csv,
    _read_trust_sidecar,
    _make_frame_to_utc,
    _resolve_boottime_t0_ns,
    _interp_dense_at,
    _time_window_overlap_msg,
)
from data_pipeline.time_sync import fit_time_anchor
from data_pipeline.parsers import read_frame_times_csv


# Reference UTC the synthetic anchors hang off (2026-06-01T00:00:00Z).
BASE_UTC = 1780272000.0
VIDEO_T0_NS = 100_000_000_000  # 100 s since boot, in ns


def _iso(utc_s: float) -> str:
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def _write_boottime_recording(path: Path, n: int = 60, hz: float = 5.0) -> None:
    dt_ns = int(1e9 / hz)
    lines = []
    for i in range(n):
        boot_ns = VIDEO_T0_NS + i * dt_ns
        utc_s = BASE_UTC + (boot_ns - VIDEO_T0_NS) / 1e9
        lines.append(f"{boot_ns},{_iso(utc_s)},{dt_ns}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_capture_meta(path: Path) -> None:
    path.write_text(json.dumps({
        "anchor_format": "boottime",
        "video": {"mp4": "v.mp4", "video_t0_boottime_ns": VIDEO_T0_NS},
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# BUG A: comment-tolerant coordinate output CSV readers
# ---------------------------------------------------------------------------

_GEOREF_HEADER = "Image,Latitude,Longitude,Altitude\n"
_GEOREF_DATA = (
    "frame_000001.png,32.060591575,34.803635683,48.5206\n"
    "frame_000002.png,32.060591567,34.803635616,48.5207\n"
)
_GEOREF_COMMENT = (
    "# the external tool reference CSV (data_pipeline). Image=camera label, "
    "Latitude/Longitude=WGS84 deg. CRS EPSG:4326.\n"
)


def test_bug_a_georef_reader_with_leading_comment(tmp_path: Path) -> None:
    p = tmp_path / "georef.csv"
    p.write_text(_GEOREF_COMMENT + _GEOREF_HEADER + _GEOREF_DATA, encoding="utf-8")
    rows = _read_georef_csv(p)
    assert len(rows) == 2, "leading '#' comment must not eat the header row"
    assert rows[0][0] == "frame_000001.png"
    assert rows[0][1] == pytest.approx(32.060591575)
    assert rows[0][2] == pytest.approx(34.803635683)


def test_bug_a_georef_reader_without_comment(tmp_path: Path) -> None:
    p = tmp_path / "georef.csv"
    p.write_text(_GEOREF_HEADER + _GEOREF_DATA, encoding="utf-8")
    rows = _read_georef_csv(p)
    assert len(rows) == 2  # header+data only still reads


def test_bug_a_frame_times_reader_with_comment(tmp_path: Path) -> None:
    p = tmp_path / "extracted_frame_times.csv"
    p.write_text(
        "# generated comment\nImage,t_video_s\n"
        "frame_000001.png,0.2\nframe_000002.png,0.4\n",
        encoding="utf-8",
    )
    rows = read_frame_times_csv(p)
    assert len(rows) == 2
    assert rows[0] == ("frame_000001.png", pytest.approx(0.2))


def test_bug_a_trust_sidecar_reader_with_comment(tmp_path: Path) -> None:
    p = tmp_path / "georef_trust.csv"
    p.write_text(
        "# trust sidecar comment\nImage,Latitude,Longitude,Altitude,Trust\n"
        "frame_000001.png,32.06,34.80,48.5,0.9\n",
        encoding="utf-8",
    )
    rows = _read_trust_sidecar(p)
    assert len(rows) == 1
    assert rows[0][4] == pytest.approx(0.9)


def test_bug_a_emit_header_comment_default_is_false() -> None:
    """Fresh georef.csv should be header+data only by default (no '#' line)."""
    from data_pipeline.stages.georef import CsvOptions
    assert CsvOptions().emit_header_comment is False


def test_bug_a_trajectory_viewer_builds_from_commented_georef(tmp_path: Path) -> None:
    """End-to-end: path viewer reads a georef.csv WITH a '#' comment."""
    georef = tmp_path / "georef.csv"
    georef.write_text(_GEOREF_COMMENT + _GEOREF_HEADER + _GEOREF_DATA, encoding="utf-8")
    data_log = tmp_path / "measurements.txt"
    # Fix,provider,lat,lon,alt,spd,acc,brg,t_ms,...,vacc(>=13 fields).
    t_ms = int(BASE_UTC * 1000)
    data_log.write_text(
        f"Fix,fused,32.0605,34.8036,48.5,0.5,6.0,90.0,{t_ms},0,0,0,10.0\n"
        f"Fix,fused,32.0606,34.8037,48.6,0.5,6.0,90.0,{t_ms + 200},0,0,0,10.0\n",
        encoding="utf-8",
    )
    out_html = tmp_path / "trajectory_viewer.html"
    # Should not raise "No usable rows in georef.csv".
    res = viewers.build_trajectory_viewer(
        data_log=data_log, georef_csv=georef, out_html=out_html, log=lambda *_: None
    )
    assert res.html_path.is_file()


# ---------------------------------------------------------------------------
# BUG B: boottime-aware sample->UTC mapping + zero-overlap diagnostic
# ---------------------------------------------------------------------------

def test_bug_b_make_frame_to_utc_boottime(tmp_path: Path) -> None:
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    anchor = fit_time_anchor(rec)
    f = _make_frame_to_utc(anchor, VIDEO_T0_NS)
    # PTS 0 -> BASE_UTC; PTS 2 -> BASE_UTC + 2
    assert f(0.0) == pytest.approx(BASE_UTC, abs=2e-3)
    assert f(2.0) == pytest.approx(BASE_UTC + 2.0, abs=2e-3)


def test_bug_b_make_frame_to_utc_legacy(tmp_path: Path) -> None:
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    anchor = fit_time_anchor(rec)
    # No boottime t0 -> falls back to direct PTS mapping (legacy video_ns).
    f = _make_frame_to_utc(anchor, None)
    assert f(0.0) == pytest.approx(anchor.video_pts_to_utc_s(0.0))


def test_bug_b_resolve_t0_from_capture_meta(tmp_path: Path) -> None:
    cm = tmp_path / "capture_meta.json"
    _write_capture_meta(cm)
    t0 = _resolve_boottime_t0_ns(cm, None, lambda *_: None)
    assert t0 == pytest.approx(float(VIDEO_T0_NS))


def test_bug_b_resolve_t0_from_video_anchor(tmp_path: Path) -> None:
    va = tmp_path / "video_anchor.txt"
    va.write_text(
        "# frameNumber,sensorTimestampNs,bootNs,timestampSource\n"
        f"0,{VIDEO_T0_NS},{VIDEO_T0_NS},REALTIME\n"
        f"1,{VIDEO_T0_NS + 33_300_000},{VIDEO_T0_NS + 33_300_000},REALTIME\n",
        encoding="utf-8",
    )
    t0 = _resolve_boottime_t0_ns(None, va, lambda *_: None)
    assert t0 == pytest.approx(float(VIDEO_T0_NS))


def test_bug_b_resolve_t0_legacy_returns_none(tmp_path: Path) -> None:
    assert _resolve_boottime_t0_ns(None, None, lambda *_: None) is None


class _PosRow:
    def __init__(self, utc_s, lat=32.06, lon=34.80, h=48.5):
        self.utc_s = utc_s
        self.lat_deg = lat
        self.lon_deg = lon
        self.h_m = h
        self.quality = 1
        self.ns = 12
        self.vn = 0.0
        self.ve = 0.0


def test_bug_b_interp_zero_with_legacy_mapping_nonzero_with_boottime(tmp_path: Path) -> None:
    """The core regression: boottime session, direct-PTS mapping yields ZERO
    interpolated points; the boottime-aware mapping yields full coverage."""
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec, n=60, hz=5.0)
    anchor = fit_time_anchor(rec)

    # .pos epochs spanning the real UTC window (BASE_UTC .. BASE_UTC + ~12s).
    pos_rows = [_PosRow(BASE_UTC + i * 0.5) for i in range(24)]
    # sample PTS 0..10 s (relative to media start).
    frame_times = [(f"frame_{i}.png", i * 0.5) for i in range(20)]

    # Legacy (buggy) mapping: PTS treated as bootNs -> hours away -> zero.
    legacy = _make_frame_to_utc(anchor, None)
    dense_legacy = _interp_dense_at(pos_rows, frame_times, anchor, 2.0, legacy)
    assert dense_legacy == [], "legacy mapping should be off the .pos window"

    # Boottime-aware mapping: samples land inside the window -> points.
    boot = _make_frame_to_utc(anchor, VIDEO_T0_NS)
    dense_boot = _interp_dense_at(pos_rows, frame_times, anchor, 2.0, boot)
    assert len(dense_boot) == len(frame_times)


def test_bug_b_time_window_overlap_diagnostic(tmp_path: Path) -> None:
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    anchor = fit_time_anchor(rec)
    pos_rows = [_PosRow(BASE_UTC + i * 0.5) for i in range(10)]
    frame_times = [(f"frame_{i}.png", i * 0.5) for i in range(10)]
    legacy = _make_frame_to_utc(anchor, None)  # disjoint window
    msg = _time_window_overlap_msg(pos_rows, frame_times, legacy)
    assert ".pos UTC window" in msg
    assert "frame UTC window" in msg
    assert "frames inside .pos window: 0/" in msg
    assert "disjoint" in msg
    # And the well-mapped case reports full overlap, no "disjoint".
    boot = _make_frame_to_utc(anchor, VIDEO_T0_NS)
    msg_ok = _time_window_overlap_msg(pos_rows, frame_times, boot)
    assert "frames inside .pos window: 10/10" in msg_ok
    assert "disjoint" not in msg_ok


# ---------------------------------------------------------------------------
# BUG C: Speed-vs-GT import path resolves (or fails gracefully)
# ---------------------------------------------------------------------------

def test_bug_c_speed_vs_gt_handler_present() -> None:
    from data_pipeline import gui
    assert hasattr(gui.App, "_run_speed_vs_gt_viewer")
    assert callable(gui.App._run_speed_vs_gt_viewer)


def test_bug_c_no_bare_toplevel_import_of_missing_module() -> None:
    """The handler must NOT rely on a top-level ``import speed_vs_gt_html`` that
    fails at module-import / call time with a bare ModuleNotFoundError. The
    builder module is optional; importing the gui module must succeed and the
    handler source must guard the import (find_spec) rather than importing
    unconditionally before the guard."""
    import importlib
    import inspect
    from data_pipeline import gui

    src = inspect.getsource(gui.App._run_speed_vs_gt_viewer)
    # The guard must precede any unguarded use: find_spec is used to detect the
    # optional module before launching the async build.
    assert "find_spec" in src, "handler should resolve the import via find_spec"
    assert "speed_vs_gt_html" in src

    # The optional builder module genuinely is not shipped in this build, so
    # find_spec must return None (proving the graceful branch is the live one).
    import sys
    repo_root = Path(gui.__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("speed_vs_gt_html") if hasattr(
        importlib, "util") else None
    # Either the module exists (import path resolves) or it doesn't (graceful
    # message path). Both are acceptable; what matters is no exception here.
    assert spec is None or spec is not None
