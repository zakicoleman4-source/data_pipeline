"""Unit tests for the offset-aware stream/media mux (``mux_audio_into_mp4``).

Verifies the external converter command construction only -- no the external converter execution, no I/O
beyond building a Path. Sign convention under test matches
``audio_sync.compute_sync_stats``:

    audio_video_offset_ms = (audio_start_utc - video_start_utc) * 1000

Positive offset -> stream started AFTER media (stream lags) -> the stream input
must be delayed, i.e. a positive ``-itsoffset`` on the stream input.
Negative offset -> stream started BEFORE media (stream leads) -> the stream
input's leading edge must be cut, i.e. a negative ``-itsoffset``.
"""

from pathlib import Path

from data_pipeline.stages.viewers import _build_mux_cmd


VIDEO = Path("C:/fake/recording_test.mp4")
WAV = Path("C:/fake/audio_test.wav")
OUT = Path("C:/fake/recording_test_av.mp4")


def _itsoffset_value(cmd: list) -> str:
    assert "-itsoffset" in cmd, f"-itsoffset missing from cmd: {cmd}"
    idx = cmd.index("-itsoffset")
    return cmd[idx + 1]


def test_zero_offset_omits_itsoffset():
    cmd = _build_mux_cmd("ffmpeg", VIDEO, WAV, OUT, offset_ms=0.0)
    assert "-itsoffset" not in cmd


def test_positive_offset_delays_audio_input():
    # stream started 250 ms AFTER media -> delay stream playback by +0.25s.
    cmd = _build_mux_cmd("ffmpeg", VIDEO, WAV, OUT, offset_ms=250.0)
    val = float(_itsoffset_value(cmd))
    assert val > 0
    assert abs(val - 0.25) < 1e-9


def test_negative_offset_trims_audio_input():
    # stream started 250 ms BEFORE media (the day14 dodge case) -> cut the
    # leading 0.25s of stream -> negative -itsoffset.
    cmd = _build_mux_cmd("ffmpeg", VIDEO, WAV, OUT, offset_ms=-250.0)
    val = float(_itsoffset_value(cmd))
    assert val < 0
    assert abs(val - (-0.25)) < 1e-9


def test_itsoffset_immediately_precedes_audio_input():
    cmd = _build_mux_cmd("ffmpeg", VIDEO, WAV, OUT, offset_ms=-250.0)
    idx = cmd.index("-itsoffset")
    # -itsoffset <val> -i <wav>  -- applies only to the stream input, and the
    # media input (first -i) must be untouched / unshifted.
    assert cmd[idx + 2] == "-i"
    assert cmd[idx + 3] == str(WAV.resolve())
    # The media input must appear before -itsoffset and have no offset.
    video_i_idx = cmd.index("-i")
    assert video_i_idx < idx
    assert cmd[video_i_idx + 1] == str(VIDEO.resolve())


def test_cmd_keeps_video_copy_and_aac_audio():
    cmd = _build_mux_cmd("ffmpeg", VIDEO, WAV, OUT, offset_ms=100.0)
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-shortest" in cmd
    assert "-map" in cmd


def test_mux_audio_into_mp4_passes_offset_to_cmd(monkeypatch, tmp_path):
    """End-to-end through mux_audio_into_mp4 (subprocess.run monkeypatched)."""
    import data_pipeline.stages.viewers as viewers_mod

    captured = {}

    class _FakeCompleted:
        returncode = 0

    def _fake_run(cmd, check=True, stdout=None, stderr=None):
        captured["cmd"] = cmd
        return _FakeCompleted()

    monkeypatch.setattr(
        "data_pipeline.ffmpeg_paths.resolve_ffmpeg", lambda: "ffmpeg"
    )
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    video = tmp_path / "recording_test.mp4"
    wav = tmp_path / "audio_test.wav"
    video.write_bytes(b"\x00")
    wav.write_bytes(b"\x00")
    out_mp4 = tmp_path / "recording_test_av.mp4"

    result = viewers_mod.mux_audio_into_mp4(
        video, wav, out_mp4, offset_ms=-250.0,
    )
    assert result == out_mp4.resolve()
    cmd = captured["cmd"]
    val = float(_itsoffset_value(cmd))
    assert abs(val - (-0.25)) < 1e-9
