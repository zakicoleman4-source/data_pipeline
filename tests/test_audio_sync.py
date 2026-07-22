"""Tests for data_pipeline.audio_sync.

All inputs are synthetic (no real session WAV/audio_anchor ships in the repo).
We verify:

* a known tone yields a feature map peak at the right frequency bin;
* a known stream anchor maps stream sample 0 to the correct UTC;
* an injected stream sample-clock drift is recovered in ppm within tolerance;
* WAV decoding handles 16/24/32-bit PCM and multi-channel down-mix;
* sync stats expose the stream<->media offset and the to_dict is JSON-clean.
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.audio_sync import (
    AudioAnchor,
    analyze_audio,
    compute_spectrogram,
    compute_sync_stats,
    fit_audio_anchor,
    parse_audio_anchor,
    read_wav,
)
from data_pipeline.time_sync import TimeAnchor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_wav(path: Path, samples: np.ndarray, rate: int, sampwidth: int = 2,
               n_channels: int = 1) -> None:
    """Write float samples in [-1,1] to a PCM WAV at the given width."""
    if sampwidth == 2:
        ints = np.clip(samples, -1, 1) * 32767.0
        data = ints.astype("<i2").tobytes()
    elif sampwidth == 4:
        ints = np.clip(samples, -1, 1) * 2147483647.0
        data = ints.astype("<i4").tobytes()
    elif sampwidth == 1:
        ints = (np.clip(samples, -1, 1) * 127.0 + 128.0).astype(np.uint8)
        data = ints.tobytes()
    else:
        raise ValueError("test helper supports 1/2/4-byte widths")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(data)


def _boot_anchor_identity(boot0_ns: float, utc0_s: float) -> TimeAnchor:
    """A boot->UTC anchor with slope exactly 1e-9 (no Signal drift).

    boottime_to_utc_s(x_ns) = utc0_s + 1e-9 * (x_ns - boot0_ns)
    """
    return TimeAnchor(
        slope=1e-9,
        xmean=boot0_ns,
        ymean=utc0_s,
        n=1000,
        rmse_s=0.001,
        max_abs_s=0.003,
        sxx_ns2=1e24,
    )


# -----------------------------------------------------------------------------
# WAV reading
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("sampwidth", [1, 2, 4])
def test_read_wav_roundtrip_amplitude(tmp_path, sampwidth):
    rate = 48000
    t = np.arange(rate) / rate
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t)
    p = tmp_path / "audio.wav"
    _write_wav(p, sig, rate, sampwidth=sampwidth)
    a = read_wav(p)
    assert a.sample_rate == rate
    assert a.n_frames == rate
    assert abs(a.duration_s - 1.0) < 1e-6
    # Peak amplitude should be near 0.5 (quantisation tolerant for 8-bit).
    tol = 0.05 if sampwidth > 1 else 0.02
    assert abs(np.max(np.abs(a.samples)) - 0.5) < tol


def test_read_wav_multichannel_downmix(tmp_path):
    rate = 8000
    t = np.arange(rate) / rate
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = -left  # opposite phase -> averages to ~0
    inter = np.empty(2 * rate)
    inter[0::2] = left
    inter[1::2] = right
    p = tmp_path / "stereo.wav"
    _write_wav(p, inter, rate, sampwidth=2, n_channels=2)
    a = read_wav(p)
    assert a.n_frames == rate
    assert np.max(np.abs(a.samples)) < 1e-2  # cancelled to near silence


def test_read_wav_empty(tmp_path):
    p = tmp_path / "empty.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(b"")
    a = read_wav(p)
    assert a.n_frames == 0
    assert a.samples.size == 0


# -----------------------------------------------------------------------------
# Feature map
# -----------------------------------------------------------------------------


def test_spectrogram_peak_at_tone_frequency(tmp_path):
    rate = 48000
    dur = 2.0
    tone_hz = 3000.0
    t = np.arange(int(rate * dur)) / rate
    sig = 0.8 * np.sin(2 * np.pi * tone_hz * t)
    p = tmp_path / "tone.wav"
    _write_wav(p, sig, rate)
    a = read_wav(p)
    spec = compute_spectrogram(a, max_time_bins=200, max_freq_bins=128)
    assert spec.power_db.ndim == 2
    assert spec.power_db.shape[1] <= 200
    assert spec.power_db.shape[0] <= 128
    # The frequency bin with the largest mean power should be near tone_hz.
    mean_per_freq = spec.power_db.mean(axis=1)
    peak_bin = int(np.argmax(mean_per_freq))
    peak_freq = spec.freqs_hz[peak_bin]
    # Allow one decimated freq-bin width of error.
    bin_width = spec.freqs_hz[1] - spec.freqs_hz[0]
    assert abs(peak_freq - tone_hz) <= 2 * bin_width


def test_spectrogram_utc_axis_uses_anchor(tmp_path):
    rate = 48000
    t = np.arange(rate) / rate
    sig = 0.3 * np.sin(2 * np.pi * 1000 * t)
    p = tmp_path / "a.wav"
    _write_wav(p, sig, rate)
    a = read_wav(p)
    boot0 = 5_000_000_000_000.0  # 5000 s of uptime, ns
    utc0 = 1_700_000_000.0
    aud_anchor = AudioAnchor(
        boot0_ns=boot0, ns_per_frame=1e9 / rate, n=2, nominal_rate_hz=rate
    )
    boot_anchor = _boot_anchor_identity(boot0, utc0)
    spec = compute_spectrogram(
        a, audio_anchor=aud_anchor, boot_anchor=boot_anchor, max_time_bins=50
    )
    assert spec.utc_s is not None
    # First time bin is centred at n_fft/2 samples -> a small positive offset.
    expected0 = utc0 + (spec.t_audio_s[0])
    assert abs(spec.utc_s[0] - expected0) < 1e-3
    # to_dict must be JSON-serialisable.
    json.dumps(spec.to_dict())


# -----------------------------------------------------------------------------
# Stream anchor parsing + fitting
# -----------------------------------------------------------------------------


def test_parse_audio_anchor_tolerates_header_and_comments(tmp_path):
    p = tmp_path / "audio_anchor.txt"
    p.write_text(
        "audioFrame,bootNs\n"
        "# comment line\n"
        "0,1000000000000\n"
        "\n"
        "48000,1001000000000\n",
        encoding="utf-8",
    )
    pairs = parse_audio_anchor(p)
    assert pairs == [(0.0, 1000000000000.0), (48000.0, 1001000000000.0)]


def test_fit_audio_anchor_maps_frame_zero_to_correct_utc():
    rate = 48000
    boot0 = 1_000_000_000_000.0
    # exactly 1 s per 48000 samples -> no drift
    pairs = [(0.0, boot0), (48000.0, boot0 + 1e9)]
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate)
    assert abs(anchor.frame_to_boot_ns(0.0) - boot0) < 1.0
    assert abs(anchor.rate_drift_ppm) < 1.0


def test_fit_audio_anchor_recovers_injected_drift():
    rate = 48000
    boot0 = 2_000_000_000_000.0
    # Inject a +200 ppm *rate* drift: the stream device samples slightly FASTER
    # than nominal, so each sample corresponds to slightly LESS boot time.
    # effective_rate = nominal*(1+200e-6)  ->  ns_per_frame = 1e9/effective_rate.
    drift_ppm = 200.0
    effective_rate = rate * (1.0 + drift_ppm / 1e6)
    ns_per_frame_true = 1e9 / effective_rate
    frames = [0.0, 24000.0, 48000.0, 96000.0]
    pairs = [(f, boot0 + ns_per_frame_true * f) for f in frames]
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate)
    assert abs(anchor.rate_drift_ppm - drift_ppm) < 1.0


def test_fit_audio_anchor_rejects_one_sided_scheduling_outliers():
    """Late anchor writes are ONE-SIDED (+30..+80 ms, never negative) and
    bias a plain OLS fit; the robust fit must stay on the quiet majority.

    Mirrors the real day14 session 20260628_190336_677 where 25/1413 late
    anchors dragged the plain fit +2.4 ms (RMSE 8.6 ms plain vs 0.25 ms
    robust).
    """
    rate = 48000
    boot0 = 2_500_000_000_000.0
    ns_per_frame = 1e9 / rate  # exact nominal, no drift
    pairs = []
    for i in range(200):
        f = i * float(rate)
        b = boot0 + ns_per_frame * f
        if i % 20 == 5:  # 10 outliers, ALL +50 ms late (one-sided)
            b += 50e6
        pairs.append((f, b))

    anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate)
    assert anchor.n_rejected >= 8
    # Intercept must sit on the clean line, not be dragged by the outliers
    # (plain OLS here is ~+2.5 ms off).
    assert abs(anchor.frame_to_boot_ns(0.0) - boot0) < 0.5e6  # < 0.5 ms
    assert abs(anchor.rate_drift_ppm) < 0.5
    assert anchor.rmse_ns < 1e6  # residual RMSE < 1 ms after rejection

    # Non-robust fit keeps every row (locks the opt-out).
    plain = fit_audio_anchor(pairs, nominal_rate_hz=rate, robust=False)
    assert plain.n_rejected == 0
    assert plain.n == 200
    assert abs(plain.frame_to_boot_ns(0.0) - boot0) > 1e6  # visibly biased


def test_fit_audio_anchor_robust_noop_on_clean_data():
    """MAD rejection must not alter a clean anchor set."""
    rate = 48000
    boot0 = 1_000_000_000_000.0
    pairs = [(i * float(rate), boot0 + i * 1e9) for i in range(50)]
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate)
    assert anchor.n == 50
    assert anchor.n_rejected == 0
    assert abs(anchor.frame_to_boot_ns(0.0) - boot0) < 1.0


def test_fit_audio_anchor_single_pair_falls_back_to_nominal():
    rate = 48000
    boot0 = 3_000_000_000_000.0
    anchor = fit_audio_anchor([(100.0, boot0)], nominal_rate_hz=rate)
    assert anchor.n == 1
    assert abs(anchor.effective_rate_hz - rate) < 1e-3
    # sample 0 boot = boot0 - ns_per_frame*100
    assert abs(anchor.frame_to_boot_ns(100.0) - boot0) < 1.0


def test_fit_audio_anchor_empty_raises():
    with pytest.raises(ValueError):
        fit_audio_anchor([], nominal_rate_hz=48000)


# -----------------------------------------------------------------------------
# Sync stats
# -----------------------------------------------------------------------------


def test_sync_stats_audio_video_offset(tmp_path):
    rate = 48000
    boot0 = 10_000_000_000_000.0
    utc0 = 1_700_000_500.0
    boot_anchor = _boot_anchor_identity(boot0, utc0)
    # Stream sample 0 at boot0; media t0 100 ms LATER on the boot clock.
    audio_anchor = AudioAnchor(
        boot0_ns=boot0, ns_per_frame=1e9 / rate, n=3, nominal_rate_hz=rate
    )
    video_t0_ns = boot0 + 100e6  # +100 ms
    from data_pipeline.audio_sync import AudioData
    audio = AudioData(np.zeros(rate), rate, rate, 1.0)
    stats = compute_sync_stats(
        audio=audio,
        audio_anchor=audio_anchor,
        boot_anchor=boot_anchor,
        video_t0_boottime_ns=video_t0_ns,
    )
    # stream starts 100 ms BEFORE media -> offset = stream - media = -100 ms.
    assert stats.audio_video_offset_ms is not None
    assert abs(stats.audio_video_offset_ms - (-100.0)) < 0.5
    assert abs(stats.audio_start_utc_s - utc0) < 1e-3
    json.dumps(stats.to_dict())


def test_sync_stats_flags_discontinuity():
    rate = 48000
    boot0 = 4_000_000_000_000.0
    boot_anchor = _boot_anchor_identity(boot0, 1_700_000_000.0)
    ns_per_frame = 1e9 / rate
    # Clean pairs plus one with a 200 ms jump.
    pairs = [
        (0.0, boot0),
        (48000.0, boot0 + 1e9),
        (96000.0, boot0 + 2e9 + 200e6),  # +200 ms discontinuity
    ]
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate)
    from data_pipeline.audio_sync import AudioData
    audio = AudioData(np.zeros(rate), rate, rate, 1.0)
    stats = compute_sync_stats(
        audio=audio,
        audio_anchor=anchor,
        boot_anchor=boot_anchor,
        audio_anchor_pairs=pairs,
        discontinuity_ms=50.0,
    )
    assert stats.audio_discontinuities >= 1
    assert stats.audio_max_residual_ms > 50.0


# -----------------------------------------------------------------------------
# End-to-end analyze_audio
# -----------------------------------------------------------------------------


def test_analyze_audio_end_to_end(tmp_path):
    rate = 48000
    dur = 1.5
    tone_hz = 5000.0
    t = np.arange(int(rate * dur)) / rate
    sig = 0.6 * np.sin(2 * np.pi * tone_hz * t)
    wav = tmp_path / "audio_session.wav"
    _write_wav(wav, sig, rate)

    boot0 = 7_000_000_000_000.0
    utc0 = 1_700_111_111.0
    anchor_txt = tmp_path / "audio_anchor_session.txt"
    anchor_txt.write_text(
        "audioFrame,bootNs\n"
        f"0,{int(boot0)}\n"
        f"{rate},{int(boot0 + 1e9)}\n"
        f"{2 * rate},{int(boot0 + 2e9)}\n",
        encoding="utf-8",
    )
    boot_anchor = _boot_anchor_identity(boot0, utc0)

    res = analyze_audio(
        wav=wav,
        audio_anchor=anchor_txt,
        boot_anchor=boot_anchor,
        video_t0_boottime_ns=boot0,  # media coincides with stream start
        max_time_bins=120,
        max_freq_bins=128,
    )
    assert res.audio.sample_rate == rate
    assert res.spectrogram.utc_s is not None
    assert abs(res.stats.audio_video_offset_ms) < 1.0
    # tone peak check
    mean_per_freq = res.spectrogram.power_db.mean(axis=1)
    peak_freq = res.spectrogram.freqs_hz[int(np.argmax(mean_per_freq))]
    bin_width = res.spectrogram.freqs_hz[1] - res.spectrogram.freqs_hz[0]
    assert abs(peak_freq - tone_hz) <= 2 * bin_width
    # full payload is JSON-clean
    json.dumps(res.to_dict())


# -----------------------------------------------------------------------------
# Integration: build_sync_player with synthetic stream embeds feature map+stats
# -----------------------------------------------------------------------------


def _write_minimal_pos(path: Path, n_epochs: int = 6) -> None:
    import datetime as _dt
    header = (
        "% GPST          latitude(deg) longitude(deg)  height(m)   Q  ns"
        "   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m)"
        " age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)"
        "   sdvn     sdve     sdvu    sdvne    sdveu    sdvun\n"
    )
    lines = [header]
    base = _dt.datetime(2026, 5, 5, 12, 23, 7)
    for i in range(n_epochs):
        t = base + _dt.timedelta(seconds=i)
        ts = t.strftime("%Y/%m/%d %H:%M:%S") + ".000"
        lines.append(
            f"{ts}   31.500000000   34.800000000    100.000  1  12"
            f"   0.010   0.010   0.020   0.000   0.000   0.000   0.5  99.9"
            f"   0.500  0.200  0.000   0.01   0.01   0.02   0.00   0.00   0.00\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_recording_txt(path: Path, n_frames: int = 6):
    """recording_*.txt in video_ns dialect; returns base UTC seconds."""
    import datetime as _dt
    base_utc = _dt.datetime(2026, 5, 5, 12, 22, 49, tzinfo=_dt.timezone.utc)
    lines = []
    for i in range(n_frames):
        video_ns = i * 1_000_000_000
        utc = base_utc + _dt.timedelta(seconds=i)
        lines.append(f"{video_ns},{utc.isoformat()},unused\n")
    path.write_text("".join(lines), encoding="utf-8")
    return base_utc.timestamp()


def _write_frame_times_csv(path: Path, n_frames: int = 6) -> None:
    lines = ["Image,t_video_s\n"]
    for i in range(n_frames):
        lines.append(f"frame_{i}.jpg,{float(i):.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_build_sync_player_embeds_audio(tmp_path):
    from data_pipeline.stages.viewers import build_sync_player

    _write_minimal_pos(tmp_path / "test.pos", n_epochs=6)
    _write_recording_txt(tmp_path / "recording.txt", n_frames=6)
    _write_frame_times_csv(tmp_path / "frame_times.csv", n_frames=6)

    # WAV: 3 s tone at 4 kHz, 48 kHz mono.
    rate = 48000
    dur = 3.0
    t = np.arange(int(rate * dur)) / rate
    sig = 0.5 * np.sin(2 * np.pi * 4000 * t)
    wav = tmp_path / "audio.wav"
    _write_wav(wav, sig, rate)

    # Stream anchor: sample -> "boot" expressed in the session's video_ns domain
    # (boottime_to_utc_s aliases video_ns_to_utc_s). Stream sample 0 at video_ns 0
    # so stream start coincides with media pts 0.
    anchor_txt = tmp_path / "audio_anchor.txt"
    anchor_txt.write_text(
        "audioFrame,bootNs\n"
        "0,0\n"
        f"{rate},1000000000\n"
        f"{2 * rate},2000000000\n",
        encoding="utf-8",
    )

    # capture_meta supplies video_t0_boottime_ns so the stream<->media offset can
    # be computed. Media pts 0 at boot 0, stream sample 0 at boot 0 -> offset ~0.
    cmeta = tmp_path / "capture_meta.json"
    cmeta.write_text(
        json.dumps({
            "anchor_format": 2,
            "video": {"mp4": "video.mp4", "video_t0_boottime_ns": 0,
                      "timestamp_source": "boottime"},
            "audio": {"timebase": "boottime"},
        }),
        encoding="utf-8",
    )

    out_html = tmp_path / "sync_player.html"
    res = build_sync_player(
        video=tmp_path / "video.mp4",   # need not exist for HTML build
        pos_file=tmp_path / "test.pos",
        frame_times_csv=tmp_path / "frame_times.csv",
        recording_map=tmp_path / "recording.txt",
        out_html=out_html,
        wav=wav,
        audio_anchor=anchor_txt,
        video_anchor=None,
        capture_meta=cmeta,
        show_spectrogram=True,
        mux_audio=False,
    )

    assert out_html.is_file()
    html = out_html.read_text(encoding="utf-8")
    # Placeholders filled.
    for ph in ("__AUDIO_SRC__", "__SPECTRO__", "__SYNC_STATS__",
               "__VIDEO_UTC_AFFINE__"):
        assert ph not in html, f"placeholder {ph} not substituted"
    # Stream src + feature map + stats present in the result + HTML.
    assert res.audio_src == "audio.wav"
    assert res.spectrogram_bins is not None
    assert res.sync_stats is not None
    assert "audio.wav" in html
    assert '"power_db"' in html
    # Feature map size is bounded for embedding.
    n_time, n_freq = res.spectrogram_bins
    assert n_time <= 400 and n_freq <= 128
    # Sync stats expose the stream<->media offset (≈0 here) + stream drift.
    assert "audio_video_offset_ms" in res.sync_stats
    assert abs(res.sync_stats["audio_video_offset_ms"]) < 5.0


def test_build_sync_player_without_audio_is_unchanged(tmp_path):
    """No wav -> stream fields null, placeholders still substituted."""
    from data_pipeline.stages.viewers import build_sync_player

    _write_minimal_pos(tmp_path / "test.pos", n_epochs=6)
    _write_recording_txt(tmp_path / "recording.txt", n_frames=6)
    _write_frame_times_csv(tmp_path / "frame_times.csv", n_frames=6)

    out_html = tmp_path / "sync_player.html"
    res = build_sync_player(
        video=tmp_path / "video.mp4",
        pos_file=tmp_path / "test.pos",
        frame_times_csv=tmp_path / "frame_times.csv",
        recording_map=tmp_path / "recording.txt",
        out_html=out_html,
    )
    html = out_html.read_text(encoding="utf-8")
    assert "__AUDIO_SRC__" not in html
    assert "const AUDIO_SRC  = null;" in html
    assert "const SPECTRO    = null;" in html
    assert res.audio_src is None
    assert res.sync_stats is None
    # Offset token substituted (0 without stream).
    assert "__AUDIO_OFFSET_MS__" not in html
    assert "const AUDIO_OFFSET_MS = 0.0;" in html


# -----------------------------------------------------------------------------
# Live-player stream offset (sync_player.html JS)
# -----------------------------------------------------------------------------


def _build_player_with_av_offset(tmp_path, *, audio_boot0_ns: int,
                                 mux_audio: bool = False, monkeypatch=None):
    """Build a sync player whose stream starts ``audio_boot0_ns`` after media
    pts 0 (session anchor is 1:1 video_ns->UTC, video_t0_boottime_ns=0)."""
    from data_pipeline.stages.viewers import build_sync_player

    _write_minimal_pos(tmp_path / "test.pos", n_epochs=6)
    _write_recording_txt(tmp_path / "recording.txt", n_frames=6)
    _write_frame_times_csv(tmp_path / "frame_times.csv", n_frames=6)

    rate = 48000
    t = np.arange(rate) / rate
    wav = tmp_path / "audio.wav"
    _write_wav(wav, 0.5 * np.sin(2 * np.pi * 1000 * t), rate)

    anchor_txt = tmp_path / "audio_anchor.txt"
    anchor_txt.write_text(
        "audioFrame,bootNs\n"
        f"0,{audio_boot0_ns}\n"
        f"{rate},{audio_boot0_ns + 1_000_000_000}\n"
        f"{2 * rate},{audio_boot0_ns + 2_000_000_000}\n",
        encoding="utf-8",
    )
    cmeta = tmp_path / "capture_meta.json"
    cmeta.write_text(json.dumps({
        "anchor_format": 2,
        "video": {"mp4": "video.mp4", "video_t0_boottime_ns": 0,
                  "timestamp_source": "boottime"},
        "audio": {"timebase": "boottime"},
    }), encoding="utf-8")

    out_html = tmp_path / "sync_player.html"
    res = build_sync_player(
        video=tmp_path / "video.mp4",
        pos_file=tmp_path / "test.pos",
        frame_times_csv=tmp_path / "frame_times.csv",
        recording_map=tmp_path / "recording.txt",
        out_html=out_html,
        wav=wav,
        audio_anchor=anchor_txt,
        capture_meta=cmeta,
        show_spectrogram=False,
        mux_audio=mux_audio,
    )
    return res, out_html.read_text(encoding="utf-8")


def test_sync_player_sidecar_wav_embeds_residual_offset(tmp_path):
    """Side-car <stream> (no mux): the JS must receive the measured offset.

    Stream starts 0.5 s AFTER media (boot 500 ms with a 1:1 anchor), so
    audio_video_offset_ms ~ +500 and the live player must seek
    stream.currentTime = media.currentTime - 0.5.
    """
    res, html = _build_player_with_av_offset(
        tmp_path, audio_boot0_ns=500_000_000, mux_audio=False,
    )
    assert res.sync_stats is not None
    off = res.sync_stats["audio_video_offset_ms"]
    assert abs(off - 500.0) < 5.0
    assert "__AUDIO_OFFSET_MS__" not in html
    import re
    m = re.search(r"const AUDIO_OFFSET_MS = (-?[\d.]+);", html)
    assert m, "AUDIO_OFFSET_MS constant missing from rendered HTML"
    assert abs(float(m.group(1)) - 500.0) < 5.0
    # The JS applies it with the correct sign (stream lags -> subtract).
    assert "video.currentTime - AUDIO_OFFSET_MS / 1000" in html


def test_sync_player_muxed_audio_offset_is_zero(tmp_path, monkeypatch):
    """Muxed AV container file: -itsoffset already bakes the offset into the file, so
    the JS residual must be 0 (double-applying would desync by 2x)."""
    import data_pipeline.stages.viewers as viewers_mod

    captured = {}

    def _fake_mux(video, wav, out_mp4, *, offset_ms=0.0, log=None):
        captured["offset_ms"] = offset_ms
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"\x00")
        return out_mp4.resolve()

    monkeypatch.setattr(viewers_mod, "mux_audio_into_mp4", _fake_mux)

    res, html = _build_player_with_av_offset(
        tmp_path, audio_boot0_ns=500_000_000, mux_audio=True,
    )
    # The full measured offset went into the mux command...
    assert abs(captured["offset_ms"] - 500.0) < 5.0
    assert res.av_mux_path is not None
    # ...so the live player must NOT re-apply it.
    assert "const AUDIO_OFFSET_MS = 0.0;" in html
