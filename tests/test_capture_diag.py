"""Tests for data_pipeline.capture_diag and the capture_diag_viewer stage.

All inputs are synthetic. We verify, without depending on a real the probe tool
binary or a real session:

* the probe tool JSON parsing -> resolution / fps / duration / size, and MB-per-minute;
* focal length absent -> ``None`` + "unavailable" (never fabricated);
* focal length present in a tag or capture_meta -> surfaced with its source;
* video_anchor parsing + sample-period OLS recovers an injected media rate;
* a known stream + Signal anchor pair recovers stream offset/drift vs Signal;
* media<->Signal drift equals the Signal bridge drift (shared boot clock);
* cut math: head/tail/total/pct_kept against a coverage window;
* the local measurements Fix-row boot->UTC fallback;
* the viewer writes a self-contained HTML carrying the computed numbers.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.capture_diag import (
    CaptureDiag,
    boot_utc_pairs_from_fix_rows,
    compute_capture_diag,
    compute_trim,
    extract_focal_length,
    parse_ffprobe_json,
    parse_video_anchor,
    video_frame_period_ns,
)
from data_pipeline.stages.capture_diag_viewer import build_capture_diag_viewer
from data_pipeline.time_sync import TimeAnchor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_wav(path: Path, n_frames: int, rate: int) -> None:
    """Write a tiny silent 16-bit mono WAV of n_frames at rate."""
    samples = (np.zeros(n_frames) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples)


def _ffprobe_json(
    *, width=1920, height=1080, avg="30/1", duration="120.0",
    size="125829120", fmt_tags=None, stream_tags=None,
) -> str:
    return json.dumps({
        "streams": [{
            "codec_type": "video",
            "width": width,
            "height": height,
            "avg_frame_rate": avg,
            "r_frame_rate": avg,
            "tags": stream_tags or {},
        }],
        "format": {
            "duration": duration,
            "size": size,
            "tags": fmt_tags or {},
        },
    })


# -----------------------------------------------------------------------------
# the probe tool parsing + data rate
# -----------------------------------------------------------------------------


def test_ffprobe_parse_basic():
    probe = parse_ffprobe_json(_ffprobe_json())
    assert probe.width == 1920
    assert probe.height == 1080
    assert probe.fps == pytest.approx(30.0)
    assert probe.duration_s == pytest.approx(120.0)
    assert probe.file_size_bytes == 125829120


def test_fps_fraction_parsing():
    # The real demo session reports a non-trivial rational sample rate.
    probe = parse_ffprobe_json(_ffprobe_json(avg="104868000/3504319"))
    assert probe.fps == pytest.approx(29.925, abs=0.01)


def test_mb_per_minute():
    # 120 MB exactly over 2 minutes -> 60 MB/min.
    size = 120 * 1024 * 1024
    text = _ffprobe_json(duration="120.0", size=str(size))
    probe = parse_ffprobe_json(text, file_size_bytes=size)
    mb = probe.file_size_bytes / (1024 * 1024)
    mb_per_min = mb / (probe.duration_s / 60.0)
    assert mb_per_min == pytest.approx(60.0)


# -----------------------------------------------------------------------------
# Focal length
# -----------------------------------------------------------------------------


def test_focal_absent_is_unavailable():
    probe = parse_ffprobe_json(_ffprobe_json())  # device-style, no focal tag
    focal, source, notes = extract_focal_length(probe, capture_meta={"video": {}})
    assert focal is None
    assert source is None
    assert any("unavailable" in n for n in notes)


def test_focal_from_ffprobe_tag():
    probe = parse_ffprobe_json(
        _ffprobe_json(stream_tags={"focal_length": "4.25"})
    )
    focal, source, _ = extract_focal_length(probe, None)
    assert focal == pytest.approx(4.25)
    assert "ffprobe" in source and "focal_length" in source


def test_focal_from_capture_meta():
    probe = parse_ffprobe_json(_ffprobe_json())
    meta = {"camera": {"focal_length_35mm": 27.0}}
    focal, source, _ = extract_focal_length(probe, meta)
    assert focal == pytest.approx(27.0)
    assert "capture_meta" in source


def test_focal_never_fabricated_on_nonnumeric_tag():
    probe = parse_ffprobe_json(
        _ffprobe_json(stream_tags={"lens": "Wide Camera"})
    )
    focal, source, notes = extract_focal_length(probe, None)
    assert focal is None
    assert source is None


# -----------------------------------------------------------------------------
# Media anchor + sample period
# -----------------------------------------------------------------------------


def test_video_anchor_parse_and_period(tmp_path):
    # 30 fps -> 33_333_333 ns/sample, starting boot at 2_000_000_000_000.
    period = 33_333_333
    boot0 = 2_000_000_000_000
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(100):
        boot = boot0 + i * period
        lines.append(f"{i},{boot},{boot},REALTIME")
    p = tmp_path / "recording_x.video_anchor.txt"
    p.write_text("\n".join(lines), encoding="utf-8")

    pairs = parse_video_anchor(p)
    assert len(pairs) == 100
    assert pairs[0] == (0.0, float(boot0))
    est = video_frame_period_ns(pairs)
    assert est == pytest.approx(period, rel=1e-9)


# -----------------------------------------------------------------------------
# Cut math
# -----------------------------------------------------------------------------


def test_trim_window_inside_video():
    # Media [100, 200] s; coverage [110, 190] -> head 10, tail 10, total 20.
    head, tail, total, pct = compute_trim(
        video_t0_utc_s=100.0,
        video_end_utc_s=200.0,
        coverage_start_utc_s=110.0,
        coverage_end_utc_s=190.0,
    )
    assert head == pytest.approx(10.0)
    assert tail == pytest.approx(10.0)
    assert total == pytest.approx(20.0)
    assert pct == pytest.approx(80.0)


def test_trim_full_coverage():
    head, tail, total, pct = compute_trim(
        video_t0_utc_s=0.0, video_end_utc_s=100.0,
        coverage_start_utc_s=-5.0, coverage_end_utc_s=105.0,
    )
    assert head == 0.0 and tail == 0.0 and total == 0.0
    assert pct == pytest.approx(100.0)


def test_trim_no_overlap():
    head, tail, total, pct = compute_trim(
        video_t0_utc_s=0.0, video_end_utc_s=100.0,
        coverage_start_utc_s=200.0, coverage_end_utc_s=300.0,
    )
    assert total == pytest.approx(100.0)
    assert pct == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# Fix-row boot->UTC fallback
# -----------------------------------------------------------------------------


def test_fix_row_fallback(tmp_path):
    # boottime_ns = 2e12 + i*1e9 ; UTC = 1.78e12 ms-derived seconds + i.
    lines = [
        "# Fix,Provider,Lat,Lon,Alt,Speed,Acc,Bearing,UnixTimeMillis,SpeedAcc,"
        "BearingAcc,elapsedRealtimeNanos,VertAcc,Mock,Used,VertSpeedAcc,SolType"
    ]
    base_boot = 2_000_000_000_000
    base_ms = 1_782_662_207_000
    for i in range(10):
        boot = base_boot + i * 1_000_000_000
        ms = base_ms + i * 1000
        lines.append(
            f"Fix,gps,32.0,34.8,70.0,0.0,1.7,196.3,{ms},1.3,179.9,{boot},11.8,0,,1.3,"
        )
    p = tmp_path / "measurements_x.txt"
    p.write_text("\n".join(lines), encoding="utf-8")

    pairs = boot_utc_pairs_from_fix_rows(p)
    assert len(pairs) == 10
    # First pair: boot, utc_s.
    assert pairs[0][0] == float(base_boot)
    assert pairs[0][1] == pytest.approx(base_ms / 1e3)


# -----------------------------------------------------------------------------
# Full synthetic session: stream/media offset + drift recovery
# -----------------------------------------------------------------------------


def _build_synthetic_session(tmp_path: Path, *, audio_drift_ppm: float):
    """Create a session folder with stream + media + measurements Fix rows.

    Signal boot->UTC: utc_s = 1_000_000.0 + boot_ns * 1e-9 (slope 1e-9 -> 0 ppm).
    Stream anchor: nominal 48 kHz with an injected sample-clock drift.
    Media t0 boottime = stream boot0 + 100 ms (so stream starts 100 ms after media).
    """
    rate = 48000
    boot0 = 5_000_000_000_000  # 5000 s after boot

    # --- measurements Fix rows give the Signal boot->UTC bridge (slope 1e-9) ---
    fix_lines = [
        "# Fix,Provider,Lat,Lon,Alt,Speed,Acc,Bearing,UnixTimeMillis,SpeedAcc,"
        "BearingAcc,elapsedRealtimeNanos,VertAcc,Mock,Used,VertSpeedAcc,SolType"
    ]
    # Cover a generous UTC window around the media so cut has overlap.
    for i in range(60):
        boot = boot0 + i * 1_000_000_000
        utc_s = 1_000_000.0 + boot * 1e-9
        ms = int(round(utc_s * 1000))
        fix_lines.append(
            f"Fix,gps,32.0,34.8,70.0,0.0,1.7,0.0,{ms},1.0,1.0,{boot},1.0,0,,1.0,"
        )
    (tmp_path / "measurements_s.txt").write_text("\n".join(fix_lines), encoding="utf-8")

    # --- stream WAV (silent, ~3 s) -------------------------------------------
    n_frames = rate * 3
    _write_wav(tmp_path / "audio_s.wav", n_frames, rate)

    # --- stream anchor: boot_ns = boot0 + ns_per_frame * sample ---------------
    # Inject drift: effective rate = nominal * (1 + ppm/1e6).
    eff_rate = rate * (1.0 + audio_drift_ppm / 1e6)
    ns_per_frame = 1e9 / eff_rate
    a_lines = []
    for frame in range(0, n_frames, rate):  # one anchor per second
        boot = boot0 + ns_per_frame * frame
        a_lines.append(f"{frame},{int(round(boot))}")
    (tmp_path / "audio_anchor_s.txt").write_text("\n".join(a_lines), encoding="utf-8")

    # --- media anchor: 30 fps, t0 = boot0 + 100 ms (media starts BEFORE stream)
    video_t0 = boot0 - 100_000_000  # media starts 100 ms before stream sample 0
    v_period = 33_333_333
    v_lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(90):  # 3 s at 30 fps
        boot = video_t0 + i * v_period
        v_lines.append(f"{i},{boot},{boot},REALTIME")
    (tmp_path / "recording_s.video_anchor.txt").write_text(
        "\n".join(v_lines), encoding="utf-8"
    )

    # --- capture_meta.json --------------------------------------------------
    meta = {
        "anchor_format": 2,
        "audio": {"wav": "audio_s.wav", "anchor": "audio_anchor_s.txt",
                  "sample_rate": rate, "timebase": "boottime"},
        "video": {"mp4": "recording_s.mp4", "video_t0_boottime_ns": video_t0,
                  "timestamp_source": "REALTIME"},
    }
    (tmp_path / "capture_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # --- empty session.txt + required sensors/measurements stubs ----------
    (tmp_path / "recording_s.txt").write_text("", encoding="utf-8")
    (tmp_path / "sensors_s.txt").write_text("", encoding="utf-8")

    return {
        "video_t0": video_t0,
        "boot0": boot0,
        "rate": rate,
        "ns_per_frame": ns_per_frame,
    }


def test_full_session_sync_recovery(tmp_path):
    info = _build_synthetic_session(tmp_path, audio_drift_ppm=-3.0)

    # No container file -> the probe tool step is skipped gracefully (notes mention it).
    diag = compute_capture_diag(session_dir=tmp_path)

    # Stream device drift recovered.
    assert diag.audio_gnss_drift_ppm == pytest.approx(-3.0, abs=0.2)

    # Media<->Signal drift == Signal bridge drift (slope 1e-9 -> ~0 ppm).
    assert diag.video_gnss_drift_ppm == pytest.approx(0.0, abs=0.5)

    # Stream starts 100 ms AFTER media (video_t0 = boot0 - 100ms; stream sample 0
    # at boot0). offset = audio_utc - video_utc = +100 ms.
    assert diag.audio_gnss_offset_ms == pytest.approx(100.0, abs=2.0)
    assert diag.video_gnss_offset_ms == pytest.approx(0.0, abs=1e-6)

    # to_dict is JSON-clean.
    json.dumps(diag.to_dict())


def test_full_session_trim(tmp_path):
    _build_synthetic_session(tmp_path, audio_drift_ppm=0.0)
    diag = compute_capture_diag(session_dir=tmp_path)
    # No container file, but the video_anchor span gives a media window and the Fix rows
    # give coverage, so cut is still computed (anchor-span fallback).
    assert diag.total_trim_s is not None
    assert diag.head_trim_s is not None and diag.head_trim_s >= 0.0
    assert diag.tail_trim_s is not None and diag.tail_trim_s >= 0.0
    # Coverage (Fix rows: boot0..boot0+59s) starts AT stream sample 0 but the
    # media begins 100 ms earlier, so a small head cut is expected.
    assert diag.head_trim_s == pytest.approx(0.1, abs=0.01)
    assert 0.0 <= diag.pct_kept <= 100.0


# -----------------------------------------------------------------------------
# Segment clips: the segment's own video_anchor t0 must WIN over capture_meta
# -----------------------------------------------------------------------------


def _write_video_anchor(path: Path, boots) -> None:
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i, b in enumerate(boots):
        lines.append(f"{i},{int(b)},{int(b)},REALTIME")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_chop_video_anchor_t0_wins_over_capture_meta(tmp_path):
    """For a cut clip, ``chop_video_anchor`` min(bootNs) is the media t0.

    The parent ``capture_meta.video_t0_boottime_ns`` is the ORIGINAL full
    session's sample 0; using it for a segment reports a stream<->media offset
    off by however far into the session the segment starts (live day15 bug:
    -288.8 ms reported vs ~-212490 ms correct). The synthetic segment here starts
    5 s into the session, so the correct offset is ~-5000 ms, not the
    parent's +100 ms.
    """
    info = _build_synthetic_session(tmp_path, audio_drift_ppm=0.0)
    boot0 = info["boot0"]
    v_period = 33_333_333

    # Segment starts 5 s into the session. The first DATA row is deliberately
    # NOT the minimum bootNs: min() must win over row [0].
    chop_t0 = boot0 + 5_000_000_000
    boots = [chop_t0 + i * v_period for i in range(60)]
    boots[0], boots[1] = boots[1], boots[0]
    chop = tmp_path / "chop_x.video_anchor.txt"
    _write_video_anchor(chop, boots)

    base = compute_capture_diag(session_dir=tmp_path)
    chopd = compute_capture_diag(session_dir=tmp_path, chop_video_anchor=chop)

    # Non-segment path unchanged: parent capture_meta t0 (stream 100 ms after media).
    assert base.audio_gnss_offset_ms == pytest.approx(100.0, abs=2.0)

    # Segment path: stream sample 0 (boot0) sits 5 s BEFORE the segment's sample 0.
    # A wrong parent t0 would report +100 ms; a wrong row-[0] t0 would report
    # -5033.3 ms (one sample period off). Both fail this tolerance.
    assert chopd.audio_gnss_offset_ms == pytest.approx(-5000.0, abs=2.0)
    assert chopd.video_gnss_offset_ms == pytest.approx(0.0, abs=1e-6)

    # Cut runs on the Segment's own span (fully inside the Fix-row coverage).
    assert chopd.head_trim_s == pytest.approx(0.0, abs=0.01)
    assert chopd.pct_kept == pytest.approx(100.0, abs=0.5)

    # The override is surfaced to the user.
    assert any("chop" in n for n in chopd.notes)


def test_non_chop_without_capture_meta_uses_video_anchor_min(tmp_path):
    """No capture_meta t0 -> t0 = min(bootNs) of the session video_anchor.

    Rows are written out of order so min() != row [0]; the offset must
    reflect the true (minimum) sample-0 boottime.
    """
    info = _build_synthetic_session(tmp_path, audio_drift_ppm=0.0)
    # Strip the t0 from capture_meta (legacy-ish manifest).
    meta_path = tmp_path / "capture_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["video"].pop("video_t0_boottime_ns", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Rewrite the session video_anchor with the first row NOT the minimum.
    video_t0 = info["video_t0"]
    v_period = 33_333_333
    boots = [video_t0 + i * v_period for i in range(90)]
    boots[0], boots[1] = boots[1], boots[0]
    _write_video_anchor(tmp_path / "recording_s.video_anchor.txt", boots)

    diag = compute_capture_diag(session_dir=tmp_path)
    # Same number as the capture_meta-driven session: stream 100 ms after media.
    assert diag.audio_gnss_offset_ms == pytest.approx(100.0, abs=2.0)


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------


def test_viewer_writes_selfcontained_html(tmp_path):
    diag = CaptureDiag(
        audio_gnss_offset_ms=-12.5,
        audio_gnss_drift_ppm=-2.0,
        video_gnss_offset_ms=0.0,
        video_gnss_drift_ppm=3.4,
        head_trim_s=0.15,
        tail_trim_s=0.22,
        total_trim_s=0.37,
        pct_kept=99.9,
        width=1920,
        height=1080,
        fps=29.93,
        duration_s=389.4,
        file_size_bytes=695_033_000,
        mb_per_min=102.1,
        focal_length=None,
        focal_source="unavailable",
        notes=["focal length unavailable: device mp4 exposes no focal_length tag."],
    )
    out = tmp_path / "capture_diag.html"
    res = build_capture_diag_viewer(diag=diag, out_html=out)
    assert out.is_file()
    assert (tmp_path / "plotly.min.js").is_file()  # vendored next to it
    html = out.read_text(encoding="utf-8")
    assert "1920x1080" in html
    assert "MB/min" in html
    assert "unavailable" in html          # focal length surfaced as unavailable
    assert "99.9" in html                 # kept percentage
    assert "window.CAPTURE_DIAG" in html  # embedded data
    # The embedded JSON is parseable.
    assert res.diag.resolution == "1920x1080"


def test_viewer_handles_all_missing(tmp_path):
    # Everything unavailable must still render without raising.
    diag = CaptureDiag(notes=["nothing available"])
    out = tmp_path / "cd.html"
    build_capture_diag_viewer(diag=diag, out_html=out)
    html = out.read_text(encoding="utf-8")
    assert "unavailable" in html
