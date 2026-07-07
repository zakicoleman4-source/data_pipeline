"""Tests for the boottime-anchored capture format support.

Covers capture_meta parsing, the TimeAnchor boottime alias, the
``_load_frames`` boottime branch, the empty-session guard, and the
parse_imu boottime timebase.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from data_pipeline.capture_meta import parse_capture_meta
from data_pipeline.errors import PipelineError
from data_pipeline.parsers import parse_imu
from data_pipeline.time_sync import fit_time_anchor
from data_pipeline.stages.georef import _load_frames


# A reference UTC second the synthetic anchors hang off (2026-06-01T00:00:00Z).
BASE_UTC = 1780272000.0
# video_t0 boottime: 100 s since boot, in ns.
VIDEO_T0_NS = 100_000_000_000


def _iso(utc_s: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def _write_boottime_recording(path: Path, n: int = 50, hz: float = 5.0) -> None:
    """recording_*.txt: boottime_ns, utc_iso, interval_ns. 1:1 boottime->UTC."""
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
        "audio": {"timebase": "boottime", "sample_rate": 48000},
        "video": {
            "mp4": "video_123.mp4",
            "video_t0_boottime_ns": VIDEO_T0_NS,
            "timestamp_source": "boottime",
            "dropped_frames": 0,
        },
        "clock": {"mono_to_boot_offset_ns": 0},
    }), encoding="utf-8")


def test_parse_capture_meta(tmp_path: Path) -> None:
    cm_path = tmp_path / "capture_meta.json"
    _write_capture_meta(cm_path)
    cm = parse_capture_meta(cm_path)
    assert cm.video_t0_boottime_ns == VIDEO_T0_NS
    assert cm.video_name == "video_123.mp4"
    assert cm.timestamp_source == "boottime"
    assert cm.is_boottime is True


def test_parse_capture_meta_sparse(tmp_path: Path) -> None:
    p = tmp_path / "capture_meta.json"
    p.write_text("{}", encoding="utf-8")
    cm = parse_capture_meta(p)
    assert cm.video_t0_boottime_ns is None
    assert cm.is_boottime is True  # defaults to boottime when not contradicted


def test_boottime_anchor_maps_to_utc(tmp_path: Path) -> None:
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    anchor = fit_time_anchor(rec)
    # boottime_to_utc_s of the t0 boottime should equal BASE_UTC.
    assert anchor.boottime_to_utc_s(VIDEO_T0_NS) == pytest.approx(BASE_UTC, abs=1e-3)
    # one second of boottime later -> one second of UTC later.
    assert anchor.boottime_to_utc_s(VIDEO_T0_NS + 1e9) == pytest.approx(
        BASE_UTC + 1.0, abs=1e-3
    )


def test_load_frames_boottime_branch(tmp_path: Path) -> None:
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    cm = tmp_path / "capture_meta.json"
    _write_capture_meta(cm)
    ftc = tmp_path / "extracted_frame_times.csv"
    # samples at PTS 0.0, 1.0, 2.0 s (seconds since media start)
    ftc.write_text(
        "Image,t_video_s\nframe_0.000000.png,0.0\n"
        "frame_1.000000.png,1.0\nframe_2.000000.png,2.0\n",
        encoding="utf-8",
    )
    frames, anchor = _load_frames(ftc, rec, lambda *_a: None, capture_meta=cm)
    assert len(frames) == 3
    # sample UTC = boottime_to_utc(video_t0 + pts*1e9) = BASE_UTC + pts
    assert frames[0].utc_s == pytest.approx(BASE_UTC + 0.0, abs=2e-3)
    assert frames[1].utc_s == pytest.approx(BASE_UTC + 1.0, abs=2e-3)
    assert frames[2].utc_s == pytest.approx(BASE_UTC + 2.0, abs=2e-3)


def test_boottime_branch_with_monotonic_timestamp_source(tmp_path: Path) -> None:
    """timestamp_source='monotonic' must STILL take the boottime branch when
    video_t0_boottime_ns is present (regression: gating must not depend on the
    timestamp_source string)."""
    rec = tmp_path / "recording_x.txt"
    _write_boottime_recording(rec)
    cm = tmp_path / "capture_meta.json"
    cm.write_text(json.dumps({
        "video": {
            "mp4": "v.mp4",
            "video_t0_boottime_ns": VIDEO_T0_NS,
            "timestamp_source": "monotonic",   # legal, non-"boottime"
        },
    }), encoding="utf-8")
    ftc = tmp_path / "extracted_frame_times.csv"
    ftc.write_text("Image,t_video_s\na.png,0.0\nb.png,2.0\n", encoding="utf-8")
    frames, _ = _load_frames(ftc, rec, lambda *_a: None, capture_meta=cm)
    # must be ~BASE_UTC + pts, NOT the garbage a legacy mapping would give
    assert frames[0].utc_s == pytest.approx(BASE_UTC, abs=2e-3)
    assert frames[1].utc_s == pytest.approx(BASE_UTC + 2.0, abs=2e-3)


def test_legacy_branch_unchanged(tmp_path: Path) -> None:
    """Without capture_meta the old media-PTS mapping is used."""
    # Legacy session: video_ns, utc. video_ns 0 == BASE_UTC.
    rec = tmp_path / "recording_legacy.txt"
    lines = []
    for i in range(40):
        vns = i * 200_000_000  # 5 Hz in ns
        lines.append(f"{vns},{_iso(BASE_UTC + vns / 1e9)},{200_000_000}")
    rec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ftc = tmp_path / "extracted_frame_times.csv"
    ftc.write_text("Image,t_video_s\na.png,0.0\nb.png,1.0\n", encoding="utf-8")
    frames, _ = _load_frames(ftc, rec, lambda *_a: None)  # no capture_meta
    assert frames[0].utc_s == pytest.approx(BASE_UTC, abs=2e-3)
    assert frames[1].utc_s == pytest.approx(BASE_UTC + 1.0, abs=2e-3)


def test_empty_recording_raises(tmp_path: Path) -> None:
    rec = tmp_path / "recording_empty.txt"
    rec.write_text("", encoding="utf-8")
    with pytest.raises(PipelineError) as ei:
        fit_time_anchor(rec)
    assert ei.value.code == "E-PP-305"


def test_parse_imu_gps_seconds(tmp_path: Path) -> None:
    """sensors_*.txt column 0 is Reference-epoch seconds in both formats:
    utc = gps_s + GPS_EPOCH(315964800) - leap(18)."""
    sens = tmp_path / "sensors_gps.txt"
    gps_s = 1.43e9  # Reference-epoch scale
    sens.write_text(f"{gps_s:.3f},0.01,0.02,0.03,0.1,0.2,9.81\n", encoding="utf-8")
    imu = parse_imu(sens)
    assert len(imu) == 1
    # gps_s + 315964800 - 18 -> a 2025-ish UTC, far from the raw reference value
    assert imu[0].utc_s == pytest.approx(gps_s + 315964800 - 18, abs=1.0)
    assert imu[0].gx == pytest.approx(0.01)
    assert imu[0].az == pytest.approx(9.81)
