"""Stream timeline, multi-clock sync statistics, and feature map.

This module turns the source app's stream sidecars into something the offline
viewers can consume:

* the WAV (``audio_*.wav``, 48 kHz PCM mono) is read with the stdlib ``wave``
  module so we depend on nothing beyond NumPy;
* the stream anchor (``audio_anchor_*.txt`` -> stream-sample -> CLOCK_BOOTTIME) is
  mapped to absolute UTC through the *same* boot->UTC :class:`TimeAnchor` the
  rest of the pipeline already fits from ``recording_*.txt``;
* a small, embeddable log-power STFT feature map is computed with its time axis
  expressed in UTC so the sync player can align a playhead to the path;
* SYNC/DRIFT statistics are derived across the three clocks (stream anchor,
  media anchor, Signal anchor) so the user can see the residual offset, the
  relative drift in ppm, anchor RMSE, and any clock discontinuities.

Everything here is read-only on the inputs. The boot->UTC anchor and the
capture manifest are *imported* from their existing modules — this module never
edits them.
"""

from __future__ import annotations

import datetime as dt
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .time_sync import TimeAnchor

# -----------------------------------------------------------------------------
# Stream anchor file
# -----------------------------------------------------------------------------
#
# audio_anchor_*.txt maps a stream *sample index* (sample position, counted from
# the start of the WAV) to an absolute CLOCK_BOOTTIME in nanoseconds. The file
# is comma-separated, tolerant of a header line and of '#'-comments:
#
#     audioFrame,bootNs[,...]
#     0,123456789012345
#     48000,123456789012345 + ~1e9
#
# Two anchors are enough to derive the stream sample clock's relationship to
# BOOTTIME; more anchors let us measure the stream device's own drift.


@dataclass(frozen=True)
class AudioAnchor:
    """Affine map stream-sample -> CLOCK_BOOTTIME (ns), fit by OLS.

        boot_ns(sample) = boot0_ns + ns_per_frame * sample

    ``ns_per_frame`` is ~ 1e9 / declared_sample_rate; the difference between the
    fitted rate and the nominal rate is the stream device's clock drift.
    """

    boot0_ns: float
    ns_per_frame: float
    n: int
    nominal_rate_hz: float
    rmse_ns: float = 0.0
    max_abs_ns: float = 0.0
    # Anchor rows rejected by the robust (MAD) outlier pass. Stream anchor
    # writes are scheduled on a best-effort thread, so late writes produce
    # one-sided positive boot_ns outliers (+30..+80 ms observed); a plain OLS
    # fit is dragged several ms off the quiet-majority line by them.
    n_rejected: int = 0

    @property
    def effective_rate_hz(self) -> float:
        """Sample rate implied by the anchor fit (vs the WAV header nominal)."""
        if self.ns_per_frame <= 0:
            return float("nan")
        return 1e9 / self.ns_per_frame

    @property
    def rate_drift_ppm(self) -> float:
        """Stream sample-clock drift vs the nominal WAV rate, in ppm."""
        if self.nominal_rate_hz <= 0 or not math.isfinite(self.effective_rate_hz):
            return float("nan")
        return (self.effective_rate_hz / self.nominal_rate_hz - 1.0) * 1e6

    def frame_to_boot_ns(self, frame: float) -> float:
        return self.boot0_ns + self.ns_per_frame * frame


def parse_audio_anchor(path: Path) -> List[Tuple[float, float]]:
    """Read ``(audio_frame, boot_ns)`` pairs from an ``audio_anchor_*.txt``.

    Tolerates a header row, blank lines and ``#`` comments. Rows whose first
    two comma-separated fields are not numeric are skipped.
    """
    pairs: List[Tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                frame = float(parts[0])
                boot_ns = float(int(parts[1])) if "." not in parts[1] else float(parts[1])
            except (ValueError, IndexError):
                continue
            pairs.append((frame, boot_ns))
    return pairs


def fit_audio_anchor(
    pairs: List[Tuple[float, float]],
    *,
    nominal_rate_hz: float,
    robust: bool = True,
    mad_threshold: float = 5.0,
    max_iter: int = 3,
) -> AudioAnchor:
    """OLS fit of ``boot_ns = boot0 + ns_per_frame * sample``.

    With ``robust=True`` (default), anchors whose residuals exceed
    ``mad_threshold`` * MAD are iteratively removed -- mirroring
    ``time_sync.fit_time_anchor_from_pairs``. This matters because the stream
    anchor writer's outliers are ONE-SIDED (late scheduling => boot_ns tens of
    ms too large, never too small), so unlike zero-mean jitter they BIAS a
    plain OLS fit: on a real 24-min session, 25 late anchors out of 1413
    pulled the fitted line +2.4 ms off the quiet majority (RMSE 8.6 ms
    plain vs 0.25 ms robust).

    With a single pair we fall back to the nominal sample rate for the slope
    (we still have an absolute boot offset). With zero pairs this raises.
    """
    if not pairs:
        raise ValueError("audio anchor has no usable (frame, boot_ns) rows")

    frames = np.array([p[0] for p in pairs], dtype=np.float64)
    boots = np.array([p[1] for p in pairs], dtype=np.float64)
    n0 = len(pairs)

    if n0 == 1 or float(np.ptp(frames)) == 0.0:
        ns_per_frame = 1e9 / nominal_rate_hz if nominal_rate_hz > 0 else 0.0
        boot0 = float(boots[0] - ns_per_frame * frames[0])
        return AudioAnchor(
            boot0_ns=boot0,
            ns_per_frame=ns_per_frame,
            n=n0,
            nominal_rate_hz=nominal_rate_hz,
        )

    def _ols(f: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
        # Fit about the means for numerical stability (samples/boots are large).
        fmean = float(f.mean())
        bmean = float(b.mean())
        df = f - fmean
        db = b - bmean
        sxx = float((df * df).sum())
        slope = float((df * db).sum() / sxx) if sxx > 0 else 0.0
        boot0 = bmean - slope * fmean  # value at sample = 0
        resid = b - (boot0 + slope * f)
        return boot0, slope, resid

    boot0, slope, resid = _ols(frames, boots)

    n_rejected = 0
    if robust:
        for _ in range(max_iter):
            mad = float(np.median(np.abs(resid)))
            if mad <= 0.0:
                break
            keep = np.abs(resid) <= mad_threshold * mad
            n_keep = int(keep.sum())
            # Bail rather than refit a degenerate set (need >= 2 distinct
            # sample values for the slope).
            if n_keep < 2 or n_keep == len(frames):
                break
            frames = frames[keep]
            boots = boots[keep]
            if float(np.ptp(frames)) == 0.0:
                break
            boot0, slope, resid = _ols(frames, boots)
        n_rejected = n0 - len(frames)

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    max_abs = float(np.max(np.abs(resid)))

    return AudioAnchor(
        boot0_ns=float(boot0),
        ns_per_frame=slope,
        n=len(frames),
        nominal_rate_hz=nominal_rate_hz,
        rmse_ns=rmse,
        max_abs_ns=max_abs,
        n_rejected=n_rejected,
    )


# -----------------------------------------------------------------------------
# WAV reading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioData:
    """Decoded mono stream: float samples in [-1, 1] plus the sample rate."""

    samples: np.ndarray   # 1-D float64
    sample_rate: int
    n_frames: int
    duration_s: float


def read_wav(path: Path) -> AudioData:
    """Read a PCM WAV (any channel count, 8/16/24/32-bit) into mono float64.

    Multi-channel stream is averaged to mono. Uses only the stdlib ``wave``
    module so there is no extra dependency.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_frames == 0 or not raw:
        return AudioData(np.zeros(0, dtype=np.float64), rate, 0, 0.0)

    if sampwidth == 1:
        # 8-bit PCM is unsigned, centred at 128.
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 3:
        # 24-bit packed little-endian; widen to int32 with sign extension.
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float64) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    n_frames = int(len(data))
    duration_s = n_frames / float(rate) if rate else 0.0
    return AudioData(
        samples=data.astype(np.float64, copy=False),
        sample_rate=int(rate),
        n_frames=n_frames,
        duration_s=duration_s,
    )


# -----------------------------------------------------------------------------
# Feature map
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrogram:
    """Compact log-power STFT for embedding in a viewer.

    ``power_db`` is ``[n_freq, n_time]`` log-power (dB), normalised so the peak
    is 0 dB. ``freqs_hz`` indexes rows; ``utc_s`` indexes columns (absolute UTC
    so the viewer can align a playhead to the path). ``t_audio_s`` is the
    same axis relative to the start of the WAV (handy when no anchor exists).
    """

    power_db: np.ndarray      # [n_freq, n_time]
    freqs_hz: np.ndarray      # [n_freq]
    t_audio_s: np.ndarray     # [n_time], seconds from WAV start
    utc_s: Optional[np.ndarray]  # [n_time] absolute UTC, or None if unmapped
    sample_rate: int
    n_fft: int
    hop: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "power_db": [[round(float(v), 2) for v in row] for row in self.power_db],
            "freqs_hz": [round(float(v), 2) for v in self.freqs_hz],
            "t_audio_s": [round(float(v), 4) for v in self.t_audio_s],
            "utc_s": (
                [round(float(v), 4) for v in self.utc_s]
                if self.utc_s is not None
                else None
            ),
            "sample_rate": int(self.sample_rate),
            "n_fft": int(self.n_fft),
            "hop": int(self.hop),
            "n_freq": int(self.power_db.shape[0]),
            "n_time": int(self.power_db.shape[1]),
        }


def compute_spectrogram(
    audio: AudioData,
    *,
    audio_anchor: Optional[AudioAnchor] = None,
    boot_anchor: Optional[TimeAnchor] = None,
    max_time_bins: int = 400,
    max_freq_bins: int = 128,
    n_fft: int = 1024,
    floor_db: float = -90.0,
) -> Spectrogram:
    """Compute a log-power STFT decimated to <= ``max_time_bins`` x ``max_freq_bins``.

    A Hann-windowed STFT is computed, then the time and frequency axes are
    reduced (by max-pooling power) to keep the embedded payload small. When both
    ``audio_anchor`` and ``boot_anchor`` are supplied each time bin is also
    tagged with absolute UTC.
    """
    x = audio.samples
    rate = audio.sample_rate
    if x.size == 0 or rate <= 0:
        return Spectrogram(
            power_db=np.zeros((0, 0)),
            freqs_hz=np.zeros(0),
            t_audio_s=np.zeros(0),
            utc_s=None,
            sample_rate=rate,
            n_fft=n_fft,
            hop=n_fft // 4,
        )

    n_fft = int(min(n_fft, _next_pow2(max(8, x.size))))
    # Choose a hop so the (downsampled) STFT lands near max_time_bins columns.
    approx_frames = max(1, x.size // max(1, n_fft // 4))
    decim_t = max(1, math.ceil(approx_frames / max_time_bins))
    hop = max(1, (n_fft // 4) * decim_t)

    window = np.hanning(n_fft).astype(np.float64)
    win_norm = float(np.sum(window ** 2)) or 1.0

    starts = list(range(0, max(1, x.size - n_fft + 1), hop))
    if not starts:
        starts = [0]
    n_time = len(starts)
    n_freq_full = n_fft // 2 + 1

    spec_power = np.empty((n_freq_full, n_time), dtype=np.float64)
    for j, s in enumerate(starts):
        seg = x[s : s + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        spec = np.fft.rfft(seg * window)
        spec_power[:, j] = (np.abs(spec) ** 2) / win_norm

    freqs_full = np.fft.rfftfreq(n_fft, d=1.0 / rate)

    # Decimate the frequency axis by max-pooling so peaks survive.
    if n_freq_full > max_freq_bins:
        decim_f = math.ceil(n_freq_full / max_freq_bins)
        n_freq = math.ceil(n_freq_full / decim_f)
        pooled = np.full((n_freq, n_time), floor_db)
        freqs = np.zeros(n_freq)
        for fi in range(n_freq):
            lo = fi * decim_f
            hi = min(lo + decim_f, n_freq_full)
            pooled_block = spec_power[lo:hi, :].max(axis=0)
            pooled[fi, :] = pooled_block
            freqs[fi] = float(freqs_full[lo:hi].mean())
        spec_power = pooled
        freqs_hz = freqs
    else:
        freqs_hz = freqs_full

    # Log power, normalised so the global peak is 0 dB.
    peak = float(spec_power.max()) or 1.0
    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(np.maximum(spec_power, 1e-20) / peak)
    power_db = np.maximum(power_db, floor_db)

    t_audio_s = np.array([(s + n_fft / 2.0) / rate for s in starts], dtype=np.float64)

    utc_s: Optional[np.ndarray] = None
    if audio_anchor is not None and boot_anchor is not None:
        centre_frames = np.array([s + n_fft / 2.0 for s in starts], dtype=np.float64)
        boot_ns = np.array(
            [audio_anchor.frame_to_boot_ns(f) for f in centre_frames]
        )
        utc_s = np.array([boot_anchor.boottime_to_utc_s(b) for b in boot_ns])

    return Spectrogram(
        power_db=power_db,
        freqs_hz=freqs_hz,
        t_audio_s=t_audio_s,
        utc_s=utc_s,
        sample_rate=rate,
        n_fft=n_fft,
        hop=hop,
    )


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1)).bit_length()


# -----------------------------------------------------------------------------
# Sync / drift statistics across the three clocks
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncStats:
    """Cross-clock synchronisation diagnostics.

    Clocks involved:

    * Stream  — stream sample clock, mapped sample -> boot -> UTC.
    * Media  — media PTS clock, mapped pts -> boot -> UTC (shared boot anchor).
    * Signal   — the boot->UTC :class:`TimeAnchor` itself (the Signal time bridge).

    Offsets are reported at the session start (UTC of stream sample 0 vs UTC of
    media pts 0). Drift is the relative rate error of the two device clocks.
    """

    # Stream device clock.
    audio_rate_nominal_hz: float
    audio_rate_effective_hz: float
    audio_drift_ppm: float
    audio_anchor_n: int
    audio_anchor_rmse_ms: float

    # Signal / boot->UTC bridge.
    gnss_anchor_n: int
    gnss_drift_ppm: float
    gnss_anchor_rmse_ms: float
    gnss_fit_uncertainty_ms: float

    # Stream start vs media start (and vs Signal reference).
    audio_start_utc_s: Optional[float] = None
    video_start_utc_s: Optional[float] = None
    audio_video_offset_ms: Optional[float] = None
    audio_video_drift_ppm: Optional[float] = None

    # Discontinuities: anchor residuals exceeding a threshold (count + worst).
    audio_discontinuities: int = 0
    audio_max_residual_ms: float = 0.0
    gnss_discontinuities: int = 0

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _r(v: Optional[float], nd: int = 3) -> Optional[float]:
            return None if v is None or not math.isfinite(v) else round(float(v), nd)

        return {
            "audio_rate_nominal_hz": _r(self.audio_rate_nominal_hz, 1),
            "audio_rate_effective_hz": _r(self.audio_rate_effective_hz, 3),
            "audio_drift_ppm": _r(self.audio_drift_ppm, 2),
            "audio_anchor_n": int(self.audio_anchor_n),
            "audio_anchor_rmse_ms": _r(self.audio_anchor_rmse_ms, 3),
            "gnss_anchor_n": int(self.gnss_anchor_n),
            "gnss_drift_ppm": _r(self.gnss_drift_ppm, 2),
            "gnss_anchor_rmse_ms": _r(self.gnss_anchor_rmse_ms, 3),
            "gnss_fit_uncertainty_ms": _r(self.gnss_fit_uncertainty_ms, 3),
            "audio_start_utc_s": _r(self.audio_start_utc_s, 4),
            "video_start_utc_s": _r(self.video_start_utc_s, 4),
            "audio_video_offset_ms": _r(self.audio_video_offset_ms, 2),
            "audio_video_drift_ppm": _r(self.audio_video_drift_ppm, 2),
            "audio_discontinuities": int(self.audio_discontinuities),
            "audio_max_residual_ms": _r(self.audio_max_residual_ms, 3),
            "gnss_discontinuities": int(self.gnss_discontinuities),
            "notes": list(self.notes),
        }


def compute_sync_stats(
    *,
    audio: AudioData,
    audio_anchor: AudioAnchor,
    boot_anchor: TimeAnchor,
    audio_anchor_pairs: Optional[List[Tuple[float, float]]] = None,
    video_t0_boottime_ns: Optional[float] = None,
    discontinuity_ms: float = 50.0,
) -> SyncStats:
    """Compute sync/drift statistics across stream, media and Signal clocks.

    ``boot_anchor`` is the Signal-derived boot->UTC :class:`TimeAnchor` (fit from
    ``recording_*.txt``). ``video_t0_boottime_ns`` (from capture_meta or the
    per-sample ``video_anchor.txt``) lets us place the media timeline; when it is
    absent the stream<->media offset is reported as ``None`` but the stream and
    Signal diagnostics are still produced.
    """
    notes: List[str] = []

    # Stream start UTC: stream sample 0 -> boot -> UTC.
    audio_boot0_ns = audio_anchor.frame_to_boot_ns(0.0)
    audio_start_utc = boot_anchor.boottime_to_utc_s(audio_boot0_ns)

    # Stream device drift (effective vs nominal sample rate).
    audio_drift_ppm = audio_anchor.rate_drift_ppm

    # Discontinuities in the stream anchor (large residuals vs the linear fit).
    audio_disc = 0
    audio_max_resid_ms = 0.0
    if audio_anchor_pairs:
        for frame, boot_ns in audio_anchor_pairs:
            resid_ns = boot_ns - audio_anchor.frame_to_boot_ns(frame)
            resid_ms = abs(resid_ns) / 1e6
            audio_max_resid_ms = max(audio_max_resid_ms, resid_ms)
            if resid_ms > discontinuity_ms:
                audio_disc += 1

    # Media start UTC (shared boot clock).
    video_start_utc: Optional[float] = None
    audio_video_offset_ms: Optional[float] = None
    if video_t0_boottime_ns is not None:
        video_start_utc = boot_anchor.boottime_to_utc_s(float(video_t0_boottime_ns))
        audio_video_offset_ms = (audio_start_utc - video_start_utc) * 1e3
    else:
        notes.append(
            "video_t0_boottime_ns not provided; audio<->video offset unavailable."
        )

    # Stream<->media relative drift: both ride the same boot clock, so the only
    # relative drift is the stream device's own deviation from the (boot-rate)
    # media clock. The media PTS clock is assumed locked to boot, so the
    # stream<->media drift equals the stream device drift.
    audio_video_drift_ppm: Optional[float] = (
        audio_drift_ppm if math.isfinite(audio_drift_ppm) else None
    )

    if audio_anchor.n < 2:
        notes.append(
            "audio anchor has <2 rows; audio drift falls back to nominal rate."
        )

    return SyncStats(
        audio_rate_nominal_hz=audio_anchor.nominal_rate_hz,
        audio_rate_effective_hz=audio_anchor.effective_rate_hz,
        audio_drift_ppm=audio_drift_ppm,
        audio_anchor_n=audio_anchor.n,
        audio_anchor_rmse_ms=audio_anchor.rmse_ns / 1e6,
        gnss_anchor_n=boot_anchor.n,
        gnss_drift_ppm=boot_anchor.drift_ppm,
        gnss_anchor_rmse_ms=boot_anchor.rmse_s * 1e3,
        gnss_fit_uncertainty_ms=boot_anchor.fit_uncertainty_s * 1e3,
        audio_start_utc_s=audio_start_utc,
        video_start_utc_s=video_start_utc,
        audio_video_offset_ms=audio_video_offset_ms,
        audio_video_drift_ppm=audio_video_drift_ppm,
        audio_discontinuities=audio_disc,
        audio_max_residual_ms=audio_max_resid_ms,
        gnss_discontinuities=0,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Convenience: load + analyse a session's stream in one call
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioSyncResult:
    """Everything the viewer needs about a session's stream."""

    audio: AudioData
    audio_anchor: AudioAnchor
    spectrogram: Spectrogram
    stats: SyncStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.audio.sample_rate,
            "duration_s": round(self.audio.duration_s, 4),
            "spectrogram": self.spectrogram.to_dict(),
            "stats": self.stats.to_dict(),
        }


def analyze_audio(
    *,
    wav: Path,
    audio_anchor: Path,
    boot_anchor: TimeAnchor,
    video_t0_boottime_ns: Optional[float] = None,
    max_time_bins: int = 400,
    max_freq_bins: int = 128,
) -> AudioSyncResult:
    """Read a session's WAV + stream anchor and produce feature map + sync stats.

    ``boot_anchor`` is the Signal boot->UTC :class:`TimeAnchor`, fit elsewhere
    (e.g. ``time_sync.fit_time_anchor(recording_map)``). This function does not
    read ``recording_*.txt`` itself so the caller controls how the bridge is
    built.
    """
    audio_data = read_wav(Path(wav))
    pairs = parse_audio_anchor(Path(audio_anchor))
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=float(audio_data.sample_rate))
    spec = compute_spectrogram(
        audio_data,
        audio_anchor=anchor,
        boot_anchor=boot_anchor,
        max_time_bins=max_time_bins,
        max_freq_bins=max_freq_bins,
    )
    stats = compute_sync_stats(
        audio=audio_data,
        audio_anchor=anchor,
        boot_anchor=boot_anchor,
        audio_anchor_pairs=pairs,
        video_t0_boottime_ns=video_t0_boottime_ns,
    )
    return AudioSyncResult(
        audio=audio_data,
        audio_anchor=anchor,
        spectrogram=spec,
        stats=stats,
    )
