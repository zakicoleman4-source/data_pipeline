"""Unit tests for the coverage plot feature in sync_player.

Tests the backend data pipeline: .stat parsing → coverage plot epoch grouping →
per-sample sky index assignment → template placeholder injection.
Does NOT require real media/The external converter — uses synthetic data throughout.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from data_pipeline.stat_to_csv import StatRow, parse_stat
from data_pipeline.stages.viewers import build_skyline_viewer, build_sync_player


# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------

def _write_minimal_pos(path: Path, n_epochs: int = 5) -> None:
    """Write a minimal The external solver .pos file with n_epochs at 1 Hz.

    Uses Reference time datetime format (YYYY/MM/DD HH:MM:SS.SSS) matching real The external solver
    output. Base time: 2026/05/05 12:23:07.000 Reference time (= UTC + 18 s leap).
    """
    header = (
        "% GPST          latitude(deg) longitude(deg)  height(m)   Q  ns"
        "   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m)"
        " age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)"
        "   sdvn     sdve     sdvu    sdvne    sdveu    sdvun\n"
    )
    lines = [header]
    import datetime as _dt
    base = _dt.datetime(2026, 5, 5, 12, 23, 7)
    for i in range(n_epochs):
        t = base + _dt.timedelta(seconds=i)
        ts = t.strftime("%Y/%m/%d %H:%M:%S") + f".{0:03d}"
        lines.append(
            f"{ts}   31.500000000   34.800000000    100.000  1  12"
            f"   0.010   0.010   0.020   0.000   0.000   0.000   0.5  99.9"
            f"   0.500  0.200  0.000   0.01   0.01   0.02   0.00   0.00   0.00\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_recording_txt(path: Path, n_frames: int = 5) -> None:
    """Write a minimal recording_*.txt anchor file.

    UTC times match the .pos times minus 18 s leap:
    .pos Reference time 12:23:07 → UTC 12:22:49.
    """
    import datetime as dt

    base_utc = dt.datetime(2026, 5, 5, 12, 22, 49, tzinfo=dt.timezone.utc)
    lines = []
    for i in range(n_frames):
        video_ns = i * 1_000_000_000
        utc = base_utc + dt.timedelta(seconds=i)
        lines.append(f"{video_ns},{utc.isoformat()},unused\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_frame_times_csv(path: Path, n_frames: int = 5) -> None:
    """Write a minimal extracted_frame_times.csv."""
    lines = ["Image,t_video_s\n"]
    for i in range(n_frames):
        lines.append(f"frame_{i}.jpg,{float(i):.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_stat_file(path: Path, n_epochs: int = 5, sats_per_epoch: int = 8) -> None:
    """Write a synthetic The external solver .stat file with $Source lines.

    TOW aligns with the .pos Reference time times: 2026/05/05 12:23:07 =
    Reference week 2364, TOW = (Mon 00:00) + 12*3600+23*60+7 = 44587.
    """
    lines = []
    week = 2364
    base_tow = 44587.0  # 12:23:07 in TOW within the week day
    # Actually compute exact TOW for 2026-05-05 12:23:07 Reference time.
    # Reference epoch: 1980-01-06. 2026-05-05 = day offset from Reference epoch.
    # Simpler: just use a consistent TOW that parse_stat will convert.
    # The key is that UTC from stat matches UTC from .pos (both are Reference time-18s).
    import datetime as _dt
    _gps_epoch = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc)
    _gpst_base = _dt.datetime(2026, 5, 5, 12, 23, 7, tzinfo=_dt.timezone.utc)
    _total_s = (_gpst_base - _gps_epoch).total_seconds()
    week = int(_total_s // (7 * 86400))
    base_tow = _total_s - week * 7 * 86400

    for ep in range(n_epochs):
        tow = base_tow + ep
        for s in range(sats_per_epoch):
            prn = f"G{s+1:02d}" if s < 5 else f"E{s-4:02d}"
            az = (s * 45) % 360
            el = 15 + (s * 10) % 60
            vsat = 1 if s < 6 else 0
            snr4 = (35 + s) * 4
            lines.append(
                f"$SAT,{week},{tow:.3f},{prn},1,"
                f"{az:.1f},{el:.1f},0.500,0.002,{vsat},{snr4},0\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


def _write_dummy_video(path: Path) -> None:
    """Write a tiny file as media placeholder (won't play but template renders)."""
    path.write_bytes(b"\x00" * 64)


# ---------------------------------------------------------------------------
# parse_stat tests
# ---------------------------------------------------------------------------

class TestParseStat:
    def test_basic_parse(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=3, sats_per_epoch=4)
        rows = parse_stat(stat_file)
        assert len(rows) == 12  # 3 epochs * 4 sources
        assert all(isinstance(r, StatRow) for r in rows)

    def test_valid_flag_preserved(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=1, sats_per_epoch=8)
        rows = parse_stat(stat_file)
        valid = [r for r in rows if r.valid_flag == 1]
        tracked = [r for r in rows if r.valid_flag == 0]
        assert len(valid) == 6
        assert len(tracked) == 2

    def test_az_el_range(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=2, sats_per_epoch=8)
        rows = parse_stat(stat_file)
        for r in rows:
            assert 0 <= r.az_deg < 360
            assert 0 <= r.el_deg <= 90

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_stat(tmp_path / "nonexistent.stat")

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "empty.stat"
        stat_file.write_text("$POS,2364,518400.0,0,0,0\n", encoding="utf-8")
        rows = parse_stat(stat_file)
        assert rows == []

    def test_snr_conversion(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        stat_file.write_text(
            "$SAT,2364,518400.000,G01,1,90.0,45.0,0.5,0.002,1,160,0\n",
            encoding="utf-8",
        )
        rows = parse_stat(stat_file)
        assert len(rows) == 1
        assert rows[0].snr_db_hz == 40.0  # 160 / 4


# ---------------------------------------------------------------------------
# Coverage plot integration in build_sync_player
# ---------------------------------------------------------------------------

class TestSyncPlayerSkyplot:
    @pytest.fixture()
    def session_dir(self, tmp_path: Path):
        """Set up a minimal session directory with all required files."""
        _write_dummy_video(tmp_path / "video.mp4")
        _write_minimal_pos(tmp_path / "test.pos", n_epochs=5)
        _write_recording_txt(tmp_path / "recording.txt", n_frames=5)
        _write_frame_times_csv(tmp_path / "frame_times.csv", n_frames=5)
        _write_stat_file(tmp_path / "test.pos.stat", n_epochs=5, sats_per_epoch=8)
        return tmp_path

    def test_skyplot_data_injected(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const SKYPLOT\s+=\s+(.*?);", html)
        assert m is not None, "SKYPLOT constant not found in HTML"
        data = json.loads(m.group(1))
        assert isinstance(data, list)
        assert len(data) == 5  # 5 epochs

    def test_skyplot_epoch_structure(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        epoch = data[0]
        assert "sats" in epoch
        assert "drv" in epoch  # driving azimuth
        sats = epoch["sats"]
        assert len(sats) == 8
        for sat in sats:
            assert "prn" in sat
            assert "az" in sat
            assert "el" in sat
            assert "v" in sat
            assert "snr" in sat
            assert "mp" in sat  # environment noise residual

    def test_skyplot_valid_vs_tracked(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        sats = data[0]["sats"]
        solved = [s for s in sats if s["v"] == 1]
        tracked = [s for s in sats if s["v"] == 0]
        assert len(solved) == 6
        assert len(tracked) == 2

    def test_trajectory_has_sky_index(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        traj = json.loads(
            re.search(r"const TRAJECTORY\s+=\s+(\[.*?\]);\s*const META", html, re.DOTALL).group(1)
        )
        for pt in traj:
            assert "sky" in pt
            assert pt["sky"] is None or isinstance(pt["sky"], int)

    def test_no_stat_file_yields_null_skyplot(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=None,
        )
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const SKYPLOT\s+=\s+(.*?);", html)
        assert m is not None
        assert json.loads(m.group(1)) is None

    def test_missing_stat_file_yields_null_skyplot(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "nonexistent.stat",
        )
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const SKYPLOT\s+=\s+(.*?);", html)
        assert json.loads(m.group(1)) is None

    def test_prn_dedup_within_epoch(self, session_dir: Path) -> None:
        """Multi-frequency obs of same PRN should merge to one dot (prefer valid)."""
        import datetime as _dt
        _gps_epoch = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc)
        _gpst_base = _dt.datetime(2026, 5, 5, 12, 23, 7, tzinfo=_dt.timezone.utc)
        _total_s = (_gpst_base - _gps_epoch).total_seconds()
        week = int(_total_s // (7 * 86400))
        tow = _total_s - week * 7 * 86400
        stat_file = session_dir / "dedup.stat"
        stat_file.write_text(
            f"$SAT,{week},{tow:.3f},G01,1,90.0,45.0,0.5,0.002,1,160,0\n"
            f"$SAT,{week},{tow:.3f},G01,2,90.0,45.0,0.3,0.001,0,140,0\n"
            f"$SAT,{week},{tow:.3f},G02,1,180.0,30.0,0.4,0.001,0,120,0\n",
            encoding="utf-8",
        )
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=stat_file,
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        assert len(data) == 1  # one epoch
        sats = data[0]["sats"]
        prns = [s["prn"] for s in sats]
        assert prns.count("G01") == 1  # deduplicated
        g01 = [s for s in sats if s["prn"] == "G01"][0]
        assert g01["v"] == 1  # valid wins over invalid

    def test_html_contains_skyplot_dom(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        assert 'id="plot-sky"' in html
        assert 'id="sky-wrap"' in html
        assert "updateSkyplot" in html
        assert "scatterpolar" in html

    def test_constellation_mix(self, session_dir: Path) -> None:
        """Verify Reference + Source-group PRNs both appear."""
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        all_prns = {s["prn"] for epoch in data for s in epoch["sats"]}
        gps = {p for p in all_prns if p.startswith("G")}
        gal = {p for p in all_prns if p.startswith("E")}
        assert len(gps) >= 1
        assert len(gal) >= 1

    def test_multipath_residual_in_data(self, session_dir: Path) -> None:
        """Each source has an 'mp' field for environment noise residual."""
        stat_file = session_dir / "mp_test.stat"
        import datetime as _dt
        _gps_epoch = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc)
        _gpst_base = _dt.datetime(2026, 5, 5, 12, 23, 7, tzinfo=_dt.timezone.utc)
        _total_s = (_gpst_base - _gps_epoch).total_seconds()
        week = int(_total_s // (7 * 86400))
        tow = _total_s - week * 7 * 86400
        # One clean source (res=0.5m), one environment noise source (res=12m)
        stat_file.write_text(
            f"$SAT,{week},{tow:.3f},G01,1,90.0,45.0,0.500,0.002,1,160,0\n"
            f"$SAT,{week},{tow:.3f},G02,1,180.0,30.0,12.000,0.001,1,120,0\n",
            encoding="utf-8",
        )
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=stat_file,
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        sats = data[0]["sats"]
        g01 = [s for s in sats if s["prn"] == "G01"][0]
        g02 = [s for s in sats if s["prn"] == "G02"][0]
        assert g01["mp"] == 0.5
        assert g02["mp"] == 12.0

    def test_driving_azimuth_present(self, session_dir: Path) -> None:
        """Epochs should have 'drv' field with driving azimuth or None."""
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        data = json.loads(re.search(r"const SKYPLOT\s+=\s+(.*?);", html).group(1))
        for epoch in data:
            assert "drv" in epoch
            assert epoch["drv"] is None or isinstance(epoch["drv"], (int, float))

    def test_html_contains_multipath_elements(self, session_dir: Path) -> None:
        out = session_dir / "out" / "sync_player.html"
        build_sync_player(
            video=session_dir / "video.mp4",
            pos_file=session_dir / "test.pos",
            frame_times_csv=session_dir / "frame_times.csv",
            recording_map=session_dir / "recording.txt",
            out_html=out,
            stat_file=session_dir / "test.pos.stat",
        )
        html = out.read_text(encoding="utf-8")
        assert "multipath" in html.lower()
        assert "_MP_THRESH" in html
        assert "_headingArrow" in html


# ---------------------------------------------------------------------------
# Skyline viewer (building silhouette from obstruction)
# ---------------------------------------------------------------------------

class TestSkylineViewer:
    def test_basic_build(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=10, sats_per_epoch=8)
        out = tmp_path / "out" / "skyline_viewer.html"
        result = build_skyline_viewer(stat_file=stat_file, out_html=out)
        assert out.is_file()
        assert result.n_epochs == 10
        assert result.n_observations == 80

    def test_html_contains_data(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=5, sats_per_epoch=8)
        out = tmp_path / "out" / "skyline_viewer.html"
        build_skyline_viewer(stat_file=stat_file, out_html=out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const DATA\s+=\s+(\{.*?\});\s*//", html, re.DOTALL)
        assert m is not None or "__SKYLINE_DATA__" not in html

    def test_payload_structure(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=5, sats_per_epoch=8)
        out = tmp_path / "out" / "skyline_viewer.html"
        build_skyline_viewer(stat_file=stat_file, out_html=out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const DATA\s+=\s+(\{.*?)\s*;\s*\n", html, re.DOTALL)
        assert m is not None
        data = json.loads(m.group(1))
        assert "stats" in data
        assert "polar_grid" in data
        assert "skyline" in data
        assert "panorama" in data
        assert "snr_heatmap" in data
        assert "timeline" in data

    def test_empty_stat_raises(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "empty.stat"
        stat_file.write_text("$POS,2364,518400.0,0,0,0\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            build_skyline_viewer(
                stat_file=stat_file,
                out_html=tmp_path / "out" / "skyline.html",
            )

    def test_plotly_copied(self, tmp_path: Path) -> None:
        stat_file = tmp_path / "test.pos.stat"
        _write_stat_file(stat_file, n_epochs=3, sats_per_epoch=4)
        out = tmp_path / "out" / "skyline_viewer.html"
        build_skyline_viewer(stat_file=stat_file, out_html=out)
        assert (tmp_path / "out" / "plotly.min.js").is_file()
