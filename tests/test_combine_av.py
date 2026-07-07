"""Pure-planning tests for data_pipeline.combine_av (no the external converter required).

A synthetic session is built in tmp_path: a real WAV header (stdlib wave),
a linear audio_anchor_*.txt, capture_meta.json, a per-sample video_anchor.txt,
a stub recording_*.container file and one chop_*/ cut slice. All assertions are
against plan_mux / audio_start_boottime_ns — nothing external runs.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from data_pipeline.combine_av import (
    ClipInfo,
    audio_start_boottime_ns,
    clip_first_frame_boot_ns,
    discover_videos,
    plan_mux,
)

NOMINAL_FS = 48000.0
AUDIO_BOOT0_NS = 104_000_000_000_000  # boottime of WAV sample 0
WAV_SECONDS = 60.0


def _write_wav(path: Path, seconds: float = WAV_SECONDS, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def _write_audio_anchor(path: Path, boot0_ns: float, true_fs: float,
                        n: int = 30) -> None:
    """Exact linear (sample, bootNs) pairs: boot = boot0 + 1e9/true_fs * sample."""
    ns_per_frame = 1e9 / true_fs
    lines = []
    for i in range(n):
        frame = i * 48000
        lines.append(f"{frame},{int(round(boot0_ns + ns_per_frame * frame))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_video_anchor(path: Path, t0_boot_ns: int, n: int = 20,
                        frame0: int = 0) -> None:
    """Per-sample anchor: frameNumber,sensorTimestampNs,bootNs,timestampSource."""
    rows = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(n):
        boot = t0_boot_ns + i * 33_333_333
        rows.append(f"{frame0 + i},{boot},{boot},REALTIME")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_session(
    tmp_path: Path,
    *,
    true_fs: float = NOMINAL_FS,
    video_t0_offset_s: float = 2.5,
    chop_start_offset_s: float = 10.0,
    chop_len_s: float = 5.0,
    chop_anchor_extra_ns: int = 3_000,
) -> Path:
    """Synthesize a session dir. Media sample-0 boot = audio_boot0 + offset."""
    s = tmp_path / "20240101_120000_000"
    s.mkdir()
    ts = "20240101_120000_000"

    # Core Signal logs so RawInputs.from_folder resolves the session.
    (s / f"measurements_{ts}.txt").write_text("# stub\n", encoding="utf-8")
    (s / f"recording_{ts}.txt").write_text("# stub\n", encoding="utf-8")
    (s / f"sensors_{ts}.txt").write_text("# stub\n", encoding="utf-8")

    _write_wav(s / f"audio_{ts}.wav")
    _write_audio_anchor(s / f"audio_anchor_{ts}.txt", AUDIO_BOOT0_NS, true_fs)

    video_t0 = int(AUDIO_BOOT0_NS + video_t0_offset_s * 1e9)
    (s / "capture_meta.json").write_text(
        json.dumps(
            {
                "anchor_format": 2,
                "audio": {
                    "wav": f"audio_{ts}.wav",
                    "anchor": f"audio_anchor_{ts}.txt",
                    "sample_rate": 48000,
                    "channels": 1,
                    "timebase": "boottime",
                },
                "video": {
                    "mp4": f"recording_{ts}.mp4",
                    "video_t0_boottime_ns": video_t0,
                },
            }
        ),
        encoding="utf-8",
    )

    (s / f"recording_{ts}.mp4").write_bytes(b"")
    _write_video_anchor(s / f"recording_{ts}.video_anchor.txt", video_t0)

    # One cut segment slice. Its media anchor's min bootNs deliberately
    # differs from chop_meta start_boottime_ns by chop_anchor_extra_ns so the
    # tests can prove the anchor (not the meta) wins.
    chop_ts = "20240101_120010_000"
    chop_dir = s / f"chop_{chop_ts}"
    chop_dir.mkdir()
    chop_start = int(AUDIO_BOOT0_NS + chop_start_offset_s * 1e9)
    chop_end = int(chop_start + chop_len_s * 1e9)
    (chop_dir / f"chop_{chop_ts}.chop_meta.json").write_text(
        json.dumps(
            {
                "schema": "chop_meta/1",
                "source_mp4": f"recording_{ts}.mp4",
                "source_capture_meta": "capture_meta.json",
                "source_audio_wav": f"audio_{ts}.wav",
                "source_audio_anchor": f"audio_anchor_{ts}.txt",
                "source_gnss": f"recording_{ts}.txt",
                "video_t0_boottime_ns": video_t0,
                "start_boottime_ns": chop_start,
                "end_boottime_ns": chop_end,
                "chopped_pts_rebased_to_zero": True,
                "audio_sample_rate": 48000,
            }
        ),
        encoding="utf-8",
    )
    (chop_dir / f"chop_{chop_ts}.mp4").write_bytes(b"")
    _write_video_anchor(
        chop_dir / f"chop_{chop_ts}.video_anchor.txt",
        chop_start + chop_anchor_extra_ns,
        frame0=300,
    )
    return s


# -----------------------------------------------------------------------------
# stream start + discovery
# -----------------------------------------------------------------------------


def test_audio_start_boottime_ns_recovers_anchor_intercept(tmp_path):
    s = _make_session(tmp_path)
    got = audio_start_boottime_ns(s)
    assert got == pytest.approx(AUDIO_BOOT0_NS, abs=1.0)


def test_discover_lists_full_then_trim(tmp_path):
    s = _make_session(tmp_path)
    items = discover_videos(s)
    assert [it.kind for it in items] == ["full", "trim"]
    assert items[0].mp4.name.startswith("recording_")
    assert items[1].mp4.name.startswith("chop_")
    assert items[1].duration_s == pytest.approx(5.0)


# -----------------------------------------------------------------------------
# FULL clip planning
# -----------------------------------------------------------------------------


def test_full_clip_seek_matches_video_t0_offset(tmp_path):
    s = _make_session(tmp_path, video_t0_offset_s=2.5)
    plan = plan_mux(s, "full")
    assert plan.clip_kind == "full"
    assert plan.audio_seek_s == pytest.approx(2.5, abs=1e-6)
    assert plan.clip_boot_ns == pytest.approx(AUDIO_BOOT0_NS + 2.5e9, abs=1.0)
    # seek >= 0 -> -ss before the WAV input, no -itsoffset.
    cmd = plan.ffmpeg_cmd
    assert "-itsoffset" not in cmd
    i = cmd.index("-ss")
    assert cmd[i + 1] == "2.500000"
    assert cmd[i + 2] == "-i" and cmd[i + 3] == str(plan.wav_path)
    # -ss comes BEFORE the stream -i (input seeking), after the media -i.
    assert cmd.index("-i") < i  # first -i is the media
    assert "-shortest" in cmd and "copy" in cmd and "aac" in cmd


def test_full_clip_falls_back_to_capture_meta_when_no_anchor(tmp_path):
    s = _make_session(tmp_path, video_t0_offset_s=2.5)
    anchor = next(s.glob("recording_*.video_anchor.txt"))
    anchor.unlink()
    plan = plan_mux(s, "full")
    assert plan.audio_seek_s == pytest.approx(2.5, abs=1e-6)


def test_negative_seek_uses_itsoffset_and_warns(tmp_path):
    # Media starts 1 s BEFORE the stream -> stream must be delayed.
    s = _make_session(tmp_path, video_t0_offset_s=-1.0)
    plan = plan_mux(s, "full")
    assert plan.audio_seek_s == pytest.approx(-1.0, abs=1e-6)
    cmd = plan.ffmpeg_cmd
    assert "-ss" not in cmd
    i = cmd.index("-itsoffset")
    assert cmd[i + 1] == "1.000000"
    assert any("silent" in w for w in plan.warnings)


def test_small_negative_seek_does_not_warn(tmp_path):
    s = _make_session(tmp_path, video_t0_offset_s=-0.2)
    plan = plan_mux(s, "full")
    assert "-itsoffset" in plan.ffmpeg_cmd
    assert not any("head will be silent" in w for w in plan.warnings)


# -----------------------------------------------------------------------------
# drift / atempo
# -----------------------------------------------------------------------------


def test_no_drift_no_atempo(tmp_path):
    s = _make_session(tmp_path, true_fs=NOMINAL_FS)
    plan = plan_mux(s, "full")
    assert abs(plan.ppm) < 0.5
    assert plan.atempo is None
    assert "-af" not in plan.ffmpeg_cmd


def test_drift_beyond_half_ppm_gets_atempo(tmp_path):
    true_fs = NOMINAL_FS * (1 + 20e-6)  # +20 ppm crystal
    s = _make_session(tmp_path, true_fs=true_fs)
    plan = plan_mux(s, "full")
    assert plan.ppm == pytest.approx(20.0, abs=0.5)
    assert plan.true_fs == pytest.approx(true_fs, rel=1e-9)
    assert plan.atempo == pytest.approx(true_fs / NOMINAL_FS, rel=1e-9)
    i = plan.ffmpeg_cmd.index("-af")
    assert plan.ffmpeg_cmd[i + 1] == f"atempo={true_fs / NOMINAL_FS:.9f}"


def test_sub_threshold_drift_no_atempo(tmp_path):
    true_fs = NOMINAL_FS * (1 + 0.3e-6)  # +0.3 ppm: below the 0.5 threshold
    s = _make_session(tmp_path, true_fs=true_fs)
    plan = plan_mux(s, "full")
    assert plan.atempo is None
    assert "-af" not in plan.ffmpeg_cmd


def test_no_rate_disables_atempo_even_with_drift(tmp_path):
    s = _make_session(tmp_path, true_fs=NOMINAL_FS * (1 + 20e-6))
    plan = plan_mux(s, "full", no_rate=True)
    assert abs(plan.ppm) > 0.5
    assert plan.atempo is None
    assert "-af" not in plan.ffmpeg_cmd


# -----------------------------------------------------------------------------
# Cut (segment) clip planning
# -----------------------------------------------------------------------------


def test_chop_uses_its_own_anchor_min_boot_not_meta_start(tmp_path):
    extra = 3_000  # anchor min bootNs is 3 us after chop_meta start_boottime_ns
    s = _make_session(tmp_path, chop_start_offset_s=10.0,
                      chop_anchor_extra_ns=extra)
    plan = plan_mux(s, 2)  # menu order: 1=full, 2=cut
    assert plan.clip_kind == "trim"
    expected_boot = AUDIO_BOOT0_NS + 10.0e9 + extra
    assert plan.clip_boot_ns == pytest.approx(expected_boot, abs=1.0)
    assert plan.audio_seek_s == pytest.approx(
        (expected_boot - AUDIO_BOOT0_NS) / 1e9, abs=1e-9
    )
    assert plan.clip_path.name.startswith("chop_")
    # WAV comes from the parent session, seek positive -> -ss path.
    assert "-ss" in plan.ffmpeg_cmd
    assert plan.clip_duration_s == pytest.approx(5.0)


def test_chop_falls_back_to_meta_start_when_anchor_unreadable(tmp_path):
    s = _make_session(tmp_path, chop_start_offset_s=10.0,
                      chop_anchor_extra_ns=3_000)
    chop_anchor = next(s.glob("chop_*/*.video_anchor.txt"))
    chop_anchor.write_text("# header only, no rows\n", encoding="utf-8")
    plan = plan_mux(s, 2)
    # Falls back to chop_meta start_boottime_ns (no +3 us anchor offset).
    assert plan.clip_boot_ns == pytest.approx(AUDIO_BOOT0_NS + 10.0e9, abs=1.0)


def test_clip_first_frame_boot_ns_trim_prefers_anchor(tmp_path):
    s = _make_session(tmp_path, chop_anchor_extra_ns=7_000)
    trim = discover_videos(s)[1]
    got = clip_first_frame_boot_ns(trim)
    assert got == pytest.approx(AUDIO_BOOT0_NS + 10.0e9 + 7_000, abs=1.0)


# -----------------------------------------------------------------------------
# selection, output path, robustness
# -----------------------------------------------------------------------------


def test_select_full_by_keyword_and_number_agree(tmp_path):
    s = _make_session(tmp_path)
    assert plan_mux(s, "full").clip_path == plan_mux(s, 1).clip_path


def test_default_out_path_is_combined_basename_in_session(tmp_path):
    s = _make_session(tmp_path)
    plan = plan_mux(s, "full")
    assert plan.out_path.parent == s
    assert plan.out_path.name == f"combined_{plan.clip_path.stem}.mp4"
    assert plan.ffmpeg_cmd[-1] == str(plan.out_path)


def test_explicit_out_path(tmp_path):
    s = _make_session(tmp_path)
    out = tmp_path / "muxed.mp4"
    plan = plan_mux(s, "full", out=out)
    assert plan.out_path == out
    assert plan.ffmpeg_cmd[-1] == str(out)


def test_invalid_selection_raises(tmp_path):
    s = _make_session(tmp_path)
    with pytest.raises(ValueError):
        plan_mux(s, 99)
    with pytest.raises(ValueError):
        plan_mux(s, "nope")
    with pytest.raises(ValueError):
        plan_mux(s)  # two clips, no selection


def test_robust_fit_ignores_late_anchor_outliers(tmp_path):
    """One-sided +50 ms outliers must not bias audio_start (robust MAD fit)."""
    s = _make_session(tmp_path)
    anchor_path = next(s.glob("audio_anchor_*.txt"))
    ns_per_frame = 1e9 / NOMINAL_FS
    lines = []
    for i in range(60):
        frame = i * 48000
        boot = AUDIO_BOOT0_NS + ns_per_frame * frame
        if i % 15 == 7:  # a few late (positive-only) writes, +50 ms
            boot += 50e6
        lines.append(f"{frame},{int(round(boot))}")
    anchor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    got = audio_start_boottime_ns(s)
    # A plain OLS fit would sit ~ms off; the robust fit stays sub-100 us.
    assert got == pytest.approx(AUDIO_BOOT0_NS, abs=1e5)


def test_wav_duration_reported_from_header(tmp_path):
    s = _make_session(tmp_path)
    plan = plan_mux(s, "full")
    assert plan.wav_duration_s == pytest.approx(WAV_SECONDS, rel=1e-4)
