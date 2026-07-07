"""Stage 3: build the Coordinate output reference CSV from Post-processing + extracted sample times.

For each extracted sample we:

1. compute its UTC instant from ``recording_*.txt`` and the sample's PTS;
2. interpolate the surrounding Post-processing ``.pos`` rows to that instant;
3. optionally smooth the resulting path with a configurable profile;
4. derive yaw from the smoothed path (heading-from-velocity) and
   borrow pitch/roll from the device Motion sensor (smoothed, decimated to 10 Hz).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as _np

from ..fused_bend import FusedBendOptions, bend_fused_to_ppk
from ..geo import ecef_to_enu, heading_from_latlon, llh_to_ecef
from ..imu_gnss_fusion import AttitudeSample as _AttSample
from ..imu_gnss_fusion import fuse as _imu_gnss_fuse
from ..kalman_smoother import smooth_trajectory_with_rts
from ..parsers import (
    Orient,
    PosRow,
    decimate_orientation,
    detect_static_periods,
    gravity_pitch_roll_from_static,
    interp_orient,
    interp_pos,
    interp_pos_with_velocity,
    parse_imu,
    parse_orientation,
    parse_data_fix,
    parse_rtkpos,
    read_frame_times_csv,
)
from ..pipeline import LogFn, make_logger
from ..smoothing import (
    estimate_rate_hz,
    gaussian_smooth,
    gaussian_smooth_adaptive_bw,
    gaussian_smooth_circular_deg,
    gaussian_smooth_weighted,
)
from ..cv_rts import gate_then_cv
from ..ns_sigma import (
    AdaptiveBwParams, NsSigmaParams, ns_is_informative,
    sigma_samples_from_ns, weights_from_ns,
)
from ..time_sync import TimeAnchor, fit_time_anchor


SmoothingProfile = Literal[
    "none", "gentle", "car", "aggressive", "custom", "fused-bent"
]


# (xy_sigma_seconds, z_sigma_seconds)
_PROFILE_PRESETS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "gentle": (0.5, 2.0),
    "car": (2.0, 10.0),
    "aggressive": (5.0, 20.0),
}


@dataclass
class CsvOptions:
    """How to build the Coordinate output CSV.

    The default schema is **lat/lon only** (no Altitude / AccuracyZ columns)
    because source-grade Z is rarely accurate enough to help coordinate tagging
    convergence and frequently *hurts* it. Set ``include_altitude=True`` to
    emit the Altitude/AccuracyZ columns from the (smoothed) Post-processing path.
    """

    smoothing: SmoothingProfile = "car"
    custom_xy_sigma_s: float = 2.0
    custom_z_sigma_s: float = 10.0
    # New in refactor: use RTS Recursive smoother instead of Gaussian smoothing.
    # RTS is superior for vehicle coordinate tagging (better noise/bias trade-off).
    use_rts_smoother: bool = False
    use_gravity_orientation: bool = False
    use_imu_fusion: bool = False

    # When True, the non-fused-bent Gaussian smoothing path uses per-epoch
    # bandwidth driven by Post-processing source count (``ns``): low-ns epochs get
    # wide windows (denoise environment noise), high-ns epochs get narrow windows
    # (preserve detail). Empirically -1.9 % hRMSE on reference session.
    # ``xy_sigma_s`` / ``z_sigma_s`` are ignored when this is on.
    use_ns_adaptive_smoothing: bool = False

    # When True, run The factor library factor graph optimization over (Post-processing + Motion sensor) before
    # sample interpolation. Replaces ``pos_rows`` with the FGO-smoothed
    # path. Requires the factor library (conda install -c conda-forge the factor library) AND
    # sensors_txt to be available. On reference session FGO ~matches uniform
    # Gaussian (limited by Post-processing Q=2 float bias, not noise). See
    # data_pipeline.fgo.run_fgo and memory/session_rtklib_conf_sweep.md for
    # how Layer-0 fix-rate work unlocks FGO improvements.
    use_fgo_smoothing: bool = False

    # ``fused-bent`` profile tuning. Takes the device fused-location track
    # (dense, Motion sensor-blended, smooth but meters-off) and warps its shape onto
    # the Post-processing anchor cloud within a trust band: full Gaussian weight inside
    # 1 sigma, hard reject beyond reject_k * sigma.
    # Device-grade Post-processing calibration: 3 m horiz 1-sigma, 15 m vert 1-sigma,
    # 30 m worst-case epoch jumps. reject_k=10 puts the hard cutoff at
    # the stated 30 m jump ceiling; the Gaussian still attenuates 2-9 sigma
    # smoothly.
    fused_bend_xy_sigma_m: float = 3.0
    fused_bend_z_sigma_m: float = 15.0
    fused_bend_reject_k: float = 10.0
    fused_bend_time_smooth_s: float = 5.0
    # Car non-holonomic constraint applied to each Post-processing anchor before bend.
    # Lateral residual sigma matches per-epoch Post-processing noise (3 m); a 30 m
    # environment noise jump lands at 10 sigma -> w~0 even without explicit cutoff.
    fused_bend_car_lateral_sigma_m: float = 3.0
    fused_bend_car_smooth_s: float = 3.0
    fused_bend_car_min_speed_mps: float = 0.5

    # Shape source for fused-bent. "device" reads ``Fix,...`` rows from the
    # data log (works only when the log includes a fused-location provider
    # or at least dense GPS_PROVIDER fixes). "ekf" runs a 6-state Motion sensor/Signal
    # EKF on sensors_*.txt + Post-processing and uses its dense output as the bend
    # shape -- best when the device never logged FLP rows but did log Motion sensor.
    fused_bend_shape_source: str = "device"  # "device", "ekf", "ekf2", "cv", "ctrv"
    # Process-noise standard deviation (m/s^2) of the EKF's linear sensor input.
    # Higher -> EKF trusts Signal more / Motion sensor less.
    fused_bend_ekf_accel_noise_std: float = 0.5
    # 9-state EKF (ekf2) Rate-signal-velocity measurement noise.
    # User spec: 2 m/s @ 2 sigma = 1 m/s @ 1 sigma horizontal.
    fused_bend_ekf_vel_h_mps: float = 1.0
    fused_bend_ekf_vel_z_mps: float = 2.0
    # 9-state EKF (ekf2) linear sensor bias random-walk stddev (m/s^2 / sqrt(s)).
    # Small -> bias estimate held nearly constant; larger -> tracks
    # temperature drift but accepts more variance.
    fused_bend_ekf_bias_rw_std: float = 0.001
    # 9-state EKF (ekf2) ZUPT threshold + measurement noise.
    # When Rate-signal speed < threshold, inject v=0 update with tight sigma
    # so the bias state absorbs whatever drift accumulated.
    fused_bend_ekf_zupt_speed_mps: float = 0.3
    fused_bend_ekf_zupt_sigma_mps: float = 0.05

    include_altitude: bool = False
    # Override the profile's Z smoothing sigma independently of the XY profile.
    # Post-processing vertical noise is 3-5x worse than horizontal; use None to keep profile default.
    z_sigma_override_s: Optional[float] = None

    # Explicit user-facing altitude-smoothing opt-in.
    #
    #   None  (default) -> legacy behaviour: the vertical (Z) channel is
    #                      smoothed using the profile / z_sigma_override_s. This
    #                      preserves the exact output of every existing caller.
    #   False           -> altitude smoothing is OFF: Z is passed through raw
    #                      (no Gaussian on the vertical channel) regardless of
    #                      the profile. Device Z is noisy; opt in deliberately.
    #   True            -> altitude smoothing is ON, using
    #                      ``altitude_smooth_sigma_s`` when set, else
    #                      ``z_sigma_override_s``, else the profile's Z sigma.
    #
    # ``smooth_altitude`` is independent of ``include_altitude``: you can smooth
    # the Z used for velocity/derived columns even when the Altitude column is
    # not emitted, and vice-versa.
    smooth_altitude: Optional[bool] = None
    # Gaussian sigma (seconds) for altitude smoothing when ``smooth_altitude``
    # is True. None -> fall back to ``z_sigma_override_s`` / profile Z sigma.
    altitude_smooth_sigma_s: Optional[float] = None

    add_ypr: bool = True
    yaw_pitch_roll_sigma_s: float = 3.0
    decimate_orient_hz: float = 10.0
    # Constant pitch/roll written when no Motion sensor orientation is available.
    # Forward-facing dashcam: pitch_prior_deg=0, roll_prior_deg=0 (source horizontal).
    # Leave as NaN to omit the column when sensors have no data.
    pitch_prior_deg: float = float("nan")
    roll_prior_deg: float = float("nan")

    accuracy_x_m: float = 0.10
    accuracy_y_m: float = 0.10
    accuracy_z_m: float = 0.30
    accuracy_yaw_deg: float = 10.0
    accuracy_pitch_deg: float = 5.0
    accuracy_roll_deg: float = 5.0

    max_interp_gap_s: float = 2.0

    # --- the external tool import-contract controls -------------------------------
    # the external tool's reference importer matches each CSV row to a source by its
    # *label*. On the client's the external tool the label is the FULL filename
    # INCLUDING the extension (e.g. ``frame_000001.png``), so the ``Image``
    # column MUST carry the extension to join. We therefore default to
    # ``label_strip_ext=False`` and emit the exact on-disk filename.
    #
    # Sample filenames are dot-free except the extension (``frame_000001.png``),
    # so the full filename is the safe label form regardless of whether a given
    # the external tool build keeps or strips the extension: with the extension kept it
    # matches exactly; the only previously-broken case (a decimal PTS in the
    # name collapsing on the first dot) no longer exists.
    #
    # Set ``label_strip_ext=True`` only for a project explicitly imported with
    # "omit file extension in label" — then the bare stem ``frame_000001`` is
    # emitted instead.
    label_strip_ext: bool = False
    # Emit a leading comment header line documenting column order + units.
    # the external tool skips lines starting with the configured comment char ('#'),
    # so this makes the file self-describing without breaking the importer.
    #
    # Default is False: the offline HTML viewers read georef.csv with
    # ``csv.DictReader``; a leading '#' comment would otherwise be parsed as
    # the header row (Track-4 regression -> "No usable rows in georef.csv").
    # The viewers now also skip '#' lines, so either default reads correctly,
    # but a header+data-only file is the safe shape. Set True to re-emit the
    # self-describing comment for tooling that wants it.
    emit_header_comment: bool = False

    # Manual media-to-Signal clock offset (seconds). Positive = media was
    # lagging Signal (shifts sample UTCs forward). Applied ON TOP of any
    # auto-detected offset from Motion sensor-Post-processing cross-correlation.
    # Set to 0.0 to rely only on auto-detection. Set to None to disable
    # both auto-detection and manual offset.
    video_offset_s: float = 0.0

    def sigmas(self) -> tuple[float, float]:
        if self.smoothing == "custom":
            xy_s, z_s = max(0.0, self.custom_xy_sigma_s), max(0.0, self.custom_z_sigma_s)
        elif self.smoothing == "fused-bent":
            # The bend replaces Gaussian smoothing entirely; report 0/0 so
            # downstream logs and the CsvResult don't claim a Gaussian was
            # applied when it wasn't.
            xy_s, z_s = 0.0, 0.0
        else:
            xy_s, z_s = _PROFILE_PRESETS.get(self.smoothing, (0.0, 0.0))
        if self.z_sigma_override_s is not None and self.z_sigma_override_s >= 0:
            z_s = float(self.z_sigma_override_s)

        # Explicit altitude-smoothing opt-in (independent of XY).
        if self.smooth_altitude is True:
            # Force ON: prefer the dedicated sigma, then the override, then
            # whatever the profile resolved to above.
            if (self.altitude_smooth_sigma_s is not None
                    and self.altitude_smooth_sigma_s >= 0):
                z_s = float(self.altitude_smooth_sigma_s)
            # else: keep z_s (override or profile default) as the ON sigma.
        elif self.smooth_altitude is False:
            # Force OFF: pass the vertical channel through raw.
            z_s = 0.0
        # smooth_altitude is None -> legacy behaviour: leave z_s as resolved.

        return xy_s, z_s


@dataclass(frozen=True)
class _Frame:
    image: str
    t_video_s: float
    utc_s: float


@dataclass(frozen=True)
class CsvResult:
    csv_path: Path
    n_frames: int
    n_with_position: int
    n_with_orientation: int
    smoothing: SmoothingProfile
    xy_sigma_s: float
    z_sigma_s: float
    time_anchor: TimeAnchor


def _derive_effective_fps(frames: list[_Frame]) -> float:
    """Derive effective sample rate from the median time difference between samples.

    This is more robust than using a user-supplied fps parameter, as it accounts
    for the actual the external converter pipeline's sample dropping/decimation. For example, if
    the source is 29.97 fps and the external converter decimates by K=5, the effective fps is
    29.97/5 = 5.994 Hz, not exactly 6 Hz.

    Returns:
        Effective fps (samples per second) computed from median sample interval.
    """
    if len(frames) < 2:
        return 1.0  # Fallback for pathological case.

    # Compute inter-sample intervals in seconds.
    intervals_s: list[float] = []
    for i in range(1, len(frames)):
        dt_s = frames[i].t_video_s - frames[i - 1].t_video_s
        if dt_s > 0:  # Skip any backward jumps.
            intervals_s.append(dt_s)

    if not intervals_s:
        return 1.0

    # Sort and take the median to be robust to outliers.
    intervals_s.sort()
    median_interval_s = intervals_s[len(intervals_s) // 2]

    # FPS is the reciprocal of the median interval.
    effective_fps = 1.0 / median_interval_s if median_interval_s > 0 else 1.0
    return effective_fps


def _first_boottime_ns_from_video_anchor(path: Path) -> Optional[float]:
    """Read the earliest ``bootNs`` from a per-sample ``video_anchor.txt``.

    Format (header + rows): ``frameNumber,sensorTimestampNs(raw),bootNs,timestampSource``.
    Returns the minimum bootNs across data rows, or ``None`` if unreadable /
    empty. Used as a precise fallback for the media-PTS t0 when capture_meta
    does not carry a usable ``video_t0_boottime_ns``.
    """
    try:
        boots: list[float] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    boots.append(float(int(parts[2])))
                except ValueError:
                    continue
        if not boots:
            return None
        return min(boots)
    except OSError:
        return None


def _probe_video_start_time_s(video_path: Path, log: LogFn) -> Optional[float]:
    """Return the container ``start_time`` of ``video_path`` in seconds.

    DIAGNOSTIC ONLY. Segment clips are extracted WITHOUT ``-copyts``, so the external converter
    already subtracts the container start_time before showinfo — the
    extracted ``t_video_s`` start at ~0 regardless of this value, and it must
    NEVER be added to sample times. Tolerant by design — returns ``None`` when
    the probe tool or the container file is missing, when start_time is "N/A"/unparseable, or
    when the value is implausibly large for a container start_time (a real
    one is a few tens of ms; anything bigger is likely start_pts ticks or a
    broken container and cannot be trusted).
    """
    try:
        if video_path is None or not Path(video_path).is_file():
            return None
    except OSError:
        return None
    try:
        from ..ffmpeg_paths import resolve_ffprobe
        ffprobe = resolve_ffprobe()
    except Exception:
        log("[csv] chop guard: ffprobe not found; skipping start_time probe")
        return None
    import subprocess
    try:
        p = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=start_time",
             "-of", "default=nk=1:nw=1", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        log(f"[csv] chop guard: ffprobe failed ({e}); skipping start_time probe")
        return None
    if p.returncode != 0:
        log("[csv] chop guard: ffprobe returned an error; skipping start_time probe")
        return None
    # Only stream=start_time is requested (never start_pts: that value is in
    # timebase TICKS, not seconds, and misreading it as seconds would inject
    # minutes of error). Parse the single bare value.
    value: float | None = None
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            value = float(line)
        except ValueError:
            continue
        break
    if value is None:
        return None
    # Sanity bound: a real container start_time is tiny (ms-scale). A large
    # magnitude means the value is not a trustworthy start_time (e.g. raw
    # ticks or a corrupt header) — refuse it rather than propagate it.
    if abs(value) > 5.0:
        log(f"[csv] chop guard: ffprobe start_time {value:+.6f} s is "
            "implausibly large for a container start_time; ignoring it")
        return None
    return value


def _load_frames(
    frame_times_csv: Path, recording_map: Path, log: LogFn,
    *,
    imu_rows: list | None = None,
    pos_rows: list | None = None,
    video_offset_s: float = 0.0,
    capture_meta: Path | None = None,
    video_anchor: Path | None = None,
    measurements_txt: Path | None = None,
    chop_video_anchor: Path | None = None,
    video_path: Path | None = None,
) -> tuple[list[_Frame], TimeAnchor]:
    """Load each sample's source timestamp and convert it to UTC via the fit.

    Supported session layouts are handled transparently: when a manifest with a
    timeline offset is supplied, sample timestamps are shifted into the timing
    file's reference before conversion; otherwise they are converted directly.
    When ``imu_rows`` and ``pos_rows`` are provided, a residual systematic offset
    is estimated and applied. ``video_offset_s`` is added on top.

    When ``recording_map`` is empty/missing but ``measurements_txt`` is given,
    the boot->UTC anchor is derived from the measurements Signal clock +
    ChipsetElapsedRealtimeNanos (see time_sync.fit_time_anchor_with_fallback).

    ``chop_video_anchor``: for a cut ("segment") clip, the segment's own
    per-sample video_anchor.txt. When given, the sample-0 boottime t0 is the
    minimum bootNs of that file and OVERRIDES capture_meta's
    ``video_t0_boottime_ns`` (which is the original full-session sample-0 boot
    and would map rebased-to-zero segment PTS ~minutes early). Segment extraction
    runs WITHOUT ``-copyts``, so the extracted ``t_video_s`` already start at
    ~0; a sanity check warns (never "corrects") if they unexpectedly do not.
    ``video_path`` is the (segment) container file, probed with the probe tool for diagnostics
    only. Raises ``ValueError`` when ``chop_video_anchor`` is supplied but
    unreadable/empty — falling back to the full-session t0 would silently map
    every segment sample minutes early.
    """
    from ..time_sync import fit_time_anchor_with_fallback
    anchor, anchor_source = fit_time_anchor_with_fallback(
        recording_map, measurements_txt,
        imu_rows=imu_rows, pos_rows=pos_rows,
    )
    if anchor_source == "measurements-fallback":
        log("[csv] recording_*.txt empty/missing -> time anchor recovered "
            "from measurements_*.txt (GNSS clock + ChipsetElapsedRealtimeNanos)")
    elif anchor_source == "measurements-fix-fallback":
        log("[csv] recording_*.txt empty/missing -> time anchor recovered "
            "from measurements_*.txt Fix rows (UnixTimeMillis + "
            "elapsedRealtimeNanos)")
        log("[csv] WARN: the Fix-row bridge includes the GNSS fix delivery "
            "latency -- absolute frame UTC is typically ~0.10-0.15 s EARLY "
            "on this device class (measured -107..-140 ms on SM-S901B "
            "sessions with ground truth). Frame positions inherit that "
            "along-track bias; audio<->video relative sync is unaffected.")

    # --- Cut-clip ("segment") handling -----------------------------------
    # A segment container file has its PTS rebased to zero, so the ORIGINAL session sample-0
    # boottime carried by capture_meta (video_t0_boottime_ns) must NOT be used
    # as t0 — that would map every sample to the start of the full session.
    # The segment's true sample-0 CLOCK_BOOTTIME is the minimum bootNs logged in
    # the segment's own video_anchor.txt. All bootNs are raw CLOCK_BOOTTIME; no
    # mono_to_boot offset is ever applied.
    is_chop = chop_video_anchor is not None
    chop_t0_ns: float | None = None
    if is_chop:
        chop_t0_ns = _first_boottime_ns_from_video_anchor(chop_video_anchor)
        if chop_t0_ns is None:
            # A known segment with an unreadable anchor must FAIL, not fall back:
            # capture_meta's video_t0_boottime_ns is the ORIGINAL session
            # sample-0 boot, and using it would silently map every segment sample
            # to the start of the full session (minutes early).
            raise ValueError(
                f"Chop video anchor {Path(chop_video_anchor)} is "
                "unreadable/empty; cannot recover the chop frame-0 boottime. "
                "Refusing to fall back to the full-session t0, which would "
                "map every chop frame minutes early. Re-export the chop or "
                "restore its video_anchor.txt."
            )

    # When the manifest carries a timeline offset, shift sample timestamps into
    # the timing file's reference; otherwise convert directly.
    boottime_t0_ns: float | None = None
    if is_chop:
        boottime_t0_ns = chop_t0_ns
        log("[csv] chop clip: t0 from chop video_anchor min bootNs"
            + (", capture_meta t0 overridden" if capture_meta is not None
               else ""))
    elif capture_meta is not None:
        try:
            from ..capture_meta import parse_capture_meta
            cm = parse_capture_meta(capture_meta)
            if cm.video_t0_boottime_ns is not None:
                boottime_t0_ns = float(cm.video_t0_boottime_ns)
                log("[csv] manifest timeline offset applied")
            else:
                log("[csv] manifest present without a timeline offset; "
                    "using direct timestamp mapping")
        except FileNotFoundError:
            pass
        except Exception as e:  # tolerant: bad manifest must not kill the run
            log(f"[csv] manifest parse failed ({e}); using direct mapping")

    # When the manifest lacks a usable timeline offset but a per-sample
    # video_anchor.txt is present (new format), recover the boottime t0 from the
    # earliest bootNs it records. This strictly improves the otherwise-broken
    # case without touching captures that already have a manifest offset.
    if boottime_t0_ns is None and video_anchor is not None:
        t0 = _first_boottime_ns_from_video_anchor(video_anchor)
        if t0 is not None:
            boottime_t0_ns = t0
            log(f"[csv] timeline offset recovered from {Path(video_anchor).name}")
    total_offset = anchor.clock_offset_s + video_offset_s
    log(
        f"[csv] time anchor: n={anchor.n} (rejected {anchor.n_rejected} outliers) "
        f"drift={anchor.drift_ppm:+.2f} ppm; "
        f"per-anchor jitter rmse={anchor.rmse_s * 1e3:.2f} ms "
        f"max-abs={anchor.max_abs_s * 1e3:.2f} ms; "
        f"fit uncertainty ~{anchor.fit_uncertainty_s * 1e3:.3f} ms; "
        f"cubic-vs-linear rmse improvement {anchor.cubic_rmse_improvement_s * 1e3:.2f} ms"
    )
    if anchor.clock_offset_s != 0.0:
        log(f"[csv] auto-detected clock offset: {anchor.clock_offset_s * 1e3:+.1f} ms "
            f"(confidence={anchor.clock_offset_confidence:.1f})")
    if video_offset_s != 0.0:
        log(f"[csv] manual video offset: {video_offset_s * 1e3:+.1f} ms")
    if abs(total_offset) > 0.001:
        log(f"[csv] total video-GNSS offset applied: {total_offset * 1e3:+.1f} ms")
    # Apply manual offset by shifting all sample UTCs.
    manual_shift = video_offset_s
    out: list[_Frame] = []
    frame_rows = read_frame_times_csv(frame_times_csv)
    # Sanity check (segment only): extraction runs WITHOUT -copyts, so the external converter
    # already rebases PTS before showinfo — the first extracted t_video_s
    # must be ~0. If it is not (PTS unexpectedly preserved / an edit list
    # survived), warn loudly but do NOT "correct" anything: adding the
    # container start_time here would shift every segment sample late.
    if is_chop and frame_rows:
        first_t_video_s = frame_rows[0][1]
        if abs(first_t_video_s) > 0.5:
            start_time_s = (
                _probe_video_start_time_s(video_path, log)
                if video_path is not None else None
            )
            extra = (
                f"; container start_time={start_time_s:+.6f} s"
                if start_time_s is not None else ""
            )
            log(f"[csv] WARN: chop clip first t_video_s = "
                f"{first_t_video_s:+.3f} s (expected ~0 — PTS may have been "
                f"preserved or an edit list is present{extra}). Frame timing "
                "may be offset by that amount; no automatic correction is "
                "applied.")
    for image, t_video_s in frame_rows:
        if boottime_t0_ns is not None:
            utc_s = anchor.boottime_to_utc_s(
                boottime_t0_ns + t_video_s * 1e9
            ) + manual_shift
        else:
            utc_s = anchor.video_pts_to_utc_s(t_video_s) + manual_shift
        out.append(_Frame(image, t_video_s, utc_s))
    return out, anchor


def _interp_from_rows(
    frames: list[_Frame], rows: list[PosRow], max_gap_s: float
) -> tuple[list[float], list[float], list[float], list[bool]]:
    times = [r.utc_s for r in rows]
    n = len(frames)
    lat_out: list[float] = [float("nan")] * n
    lon_out: list[float] = [float("nan")] * n
    h_out: list[float] = [float("nan")] * n
    has: list[bool] = [False] * n
    for i, f in enumerate(frames):
        llh = interp_pos(rows, f.utc_s, max_gap_s, times=times)
        if llh is None:
            continue
        lat_out[i], lon_out[i], h_out[i] = llh
        has[i] = True
    return lat_out, lon_out, h_out, has


def _ns_per_frame(frames: list[_Frame], rows: list[PosRow]) -> list[float]:
    """Per-sample ns (sources in solution) from nearest Post-processing row.

    Uses numpy.searchsorted for vectorised nearest-neighbour lookup.
    """
    if not rows or not frames:
        return [float("nan")] * len(frames)
    times = _np.array([r.utc_s for r in rows])
    ns_arr = _np.array([float(r.ns) for r in rows])
    ftimes = _np.array([f.utc_s for f in frames])
    idx = _np.searchsorted(times, ftimes, side="left")
    idx = _np.clip(idx, 1, len(rows) - 1)
    left_dt = ftimes - times[idx - 1]
    right_dt = times[idx] - ftimes
    use_left = left_dt <= right_dt
    result = _np.where(use_left, ns_arr[idx - 1], ns_arr[idx])
    return result.tolist()



def _rate_hz_from_frames(
    frames: Optional[list[_Frame]], fallback_fps: float,
) -> float:
    """Return effective samples-per-second from sample timestamps.

    Used to convert ``sigma_seconds → sigma_samples`` for any smoothing
    pass that operates on per-sample arrays. Adaptive selection produces
    irregular spacing where ``fallback_fps`` (the user's requested fps)
    is meaningless — the median inter-sample Δt is the right thing to
    use. Falls back to ``fallback_fps`` when there are < 2 samples.
    """
    if frames is None or len(frames) < 2:
        return max(1.0, fallback_fps)
    dts = _np.diff([f.utc_s for f in frames])
    dts = dts[dts > 1e-9]
    if dts.size == 0:
        return max(1.0, fallback_fps)
    return 1.0 / float(_np.median(dts))


def _smooth_trajectory(
    lat: list[float],
    lon: list[float],
    h: list[float],
    fps: float,
    xy_sigma_s: float,
    z_sigma_s: float,
    frames: Optional[list[_Frame]] = None,
    use_rts: bool = False,
    ns_per_sample: Optional[Sequence[float]] = None,
) -> tuple[list[float], list[float], list[float]]:
    """Smooth path in metric Local-frame space, not lat/lon space.

    This removes the latitude bias where 1° lon ≠ 1° lat in meters.
    The smoothing is performed in local Local-frame coordinates about the path's
    initial reference point, preserving isotropic smoothing in metric space.

    If use_rts=True and samples are provided, use RTS Recursive smoother (superior
    for vehicle motion). Otherwise fall back to Gaussian smoothing.

    The sigma_seconds → sigma_samples conversion uses the actual median
    inter-sample spacing (from ``samples``) when available, with ``fps`` only
    as a fallback. For adaptive-rate selection (where samples are spaced
    non-uniformly by Rate-signal distance, not time) the nominal ``fps`` is
    not the true sampling rate and using it would over- or under-smooth.
    """
    if use_rts and frames is not None and len(frames) > 0:
        # Use RTS Recursive smoother.
        times_s = [f.utc_s for f in frames]
        lat_smooth, lon_smooth, h_smooth = smooth_trajectory_with_rts(
            lat, lon, h, times_s,
            measurement_variance=0.01 ** 2,
            process_noise_std=0.01
        )
        return lat_smooth, lon_smooth, h_smooth
    if xy_sigma_s <= 0 and z_sigma_s <= 0:
        return list(lat), list(lon), list(h)

    # Find reference point (first finite coordinate).
    ref_lat, ref_lon, ref_h = None, None, None
    for la, lo, he in zip(lat, lon, h):
        if math.isfinite(la) and math.isfinite(lo) and math.isfinite(he):
            ref_lat, ref_lon, ref_h = la, lo, he
            break

    if ref_lat is None:
        # No finite points; return as-is.
        return list(lat), list(lon), list(h)

    # Convert path to Local-frame coordinates about reference point (vectorised).
    ref_llh = (ref_lat, ref_lon, ref_h)
    lat_a = _np.asarray(lat, dtype=float)
    lon_a = _np.asarray(lon, dtype=float)
    h_a = _np.asarray(h, dtype=float)
    finite = _np.isfinite(lat_a) & _np.isfinite(lon_a) & _np.isfinite(h_a)
    es = _np.full(len(lat), float("nan"))
    ns = _np.full(len(lat), float("nan"))
    us = _np.full(len(lat), float("nan"))
    for i in _np.nonzero(finite)[0]:
        x, y, z = llh_to_ecef(lat_a[i], lon_a[i], h_a[i])
        es[i], ns[i], us[i] = ecef_to_enu(x, y, z, ref_llh)
    es = es.tolist()
    ns = ns.tolist()
    us = us.tolist()

    # Smooth in Local-frame space.
    # Derive samples-per-second from the actual sample timestamps when
    # available (adaptive selection picks samples at irregular intervals;
    # the nominal ``fps`` is irrelevant there). Fall back to fps for
    # fixed-rate pipelines that don't pass ``samples`` through.
    if frames is not None and len(frames) >= 2:
        dts = _np.diff([f.utc_s for f in frames])
        # Filter out non-positive deltas defensively.
        dts = dts[dts > 1e-9]
        median_dt = float(_np.median(dts)) if dts.size else 1.0 / max(fps, 1e-9)
        rate_hz = 1.0 / median_dt
    else:
        rate_hz = max(1.0, fps)
    sigma_xy = max(0.0, xy_sigma_s) * rate_hz
    sigma_z = max(0.0, z_sigma_s) * rate_hz

    if ns_per_sample is not None and len(ns_per_sample) == len(es):
        # Adaptive per-epoch bandwidth: kernel sigma in samples scales with
        # Post-processing source count. Low ns -> wide window (denoise environment noise);
        # high ns -> narrow window (preserve detail). Empirically -1.9%
        # hRMSE on reference session vs uniform Gaussian at xy_sigma=2s, z_sigma=10s.
        bw = AdaptiveBwParams()
        ns_arr = _np.asarray(ns_per_sample, dtype=float)
        sig_h = sigma_samples_from_ns(ns_arr, rate_hz, bw, axis="h")
        sig_v = sigma_samples_from_ns(ns_arr, rate_hz, bw, axis="v")
        es_smooth = gaussian_smooth_adaptive_bw(es, sig_h.tolist())
        ns_smooth = gaussian_smooth_adaptive_bw(ns, sig_h.tolist())
        us_smooth = gaussian_smooth_adaptive_bw(us, sig_v.tolist())
    else:
        es_smooth = gaussian_smooth(es, sigma_xy) if sigma_xy > 0 else es
        ns_smooth = gaussian_smooth(ns, sigma_xy) if sigma_xy > 0 else ns
        us_smooth = gaussian_smooth(us, sigma_z) if sigma_z > 0 else us

    # Convert smoothed Local-frame back to lat/lon/h.
    lat_smooth: list[float] = []
    lon_smooth: list[float] = []
    h_smooth: list[float] = []

    rx, ry, rz = llh_to_ecef(*ref_llh)
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)

    for e, n, u in zip(es_smooth, ns_smooth, us_smooth):
        if not (math.isfinite(e) and math.isfinite(n) and math.isfinite(u)):
            lat_smooth.append(float("nan"))
            lon_smooth.append(float("nan"))
            h_smooth.append(float("nan"))
        else:
            # Inverse Local-frame to Cartesian XYZ: x,y,z = ref + R * (e,n,u)
            x = rx + (-so * e - sl * co * n + cl * co * u)
            y = ry + (co * e - sl * so * n + cl * so * u)
            z = rz + (cl * n + sl * u)
            # Cartesian XYZ to LLH (iterative Helix method, simplified).
            # For now, use a simplified closed-form approximation.
            from ..geo import _A, _E2
            p = math.sqrt(x * x + y * y)
            lat_iter = math.atan2(z, p * (1 - _E2))
            for _ in range(3):  # 3 iterations converges to sub-mm.
                n_rad = _A / math.sqrt(1 - _E2 * math.sin(lat_iter) ** 2)
                lat_iter = math.atan2(z + _E2 * n_rad * math.sin(lat_iter), p)
            lon_final = math.atan2(y, x)
            h_final = p / math.cos(lat_iter) - _A / math.sqrt(1 - _E2 * math.sin(lat_iter) ** 2)
            lat_smooth.append(math.degrees(lat_iter))
            lon_smooth.append(math.degrees(lon_final))
            h_smooth.append(h_final)

    return lat_smooth, lon_smooth, h_smooth


def _yaw_from_trajectory(
    lat: list[float], lon: list[float],
    times: Optional[list[float]] = None,
    window_s: float = 0.4,
    window: int = 2,
) -> list[float]:
    """Compass heading at every sample, computed from a small look-around window.

    With ``times`` supplied, the window is **time-based**: the look-around
    indices are the first samples ≥ ``window_s`` either side of ``i``. This
    is required for adaptive-rate sample selection, where a fixed sample
    window collapses to ~0 s in dense regions and stretches to many seconds
    in sparse ones — both extremes corrupt the finite-difference heading.

    Without ``times`` (legacy callers) the window falls back to a fixed
    ``window``-sample look-around suitable for constant-FPS streams.
    """
    n = len(lat)
    out: list[float] = []
    use_time = times is not None and len(times) == n and n > 1
    from bisect import bisect_left as _bl_, bisect_right as _br_
    for i in range(n):
        if use_time:
            t_i = times[i]
            # First index with times[a] <= t_i - window_s, else 0.
            a = _br_(times, t_i - window_s) - 1
            if a < 0:
                a = 0
            # First index with times[b] >= t_i + window_s, else n-1.
            b = _bl_(times, t_i + window_s)
            if b >= n:
                b = n - 1
            # heading_from_latlon(a, b) is the bearing at the midpoint of
            # (t_a, t_b); we assign it to sample i. If the two sides of the
            # bracket are wildly asymmetric in time (e.g. dense samples
            # before t_i, sparse after) the midpoint drifts off t_i and
            # the yaw becomes a biased estimate of the vehicle's heading
            # at sample i. Refuse the estimate when the asymmetry exceeds
            # half a window; the velocity-derived yaw will fill in via
            # _merge_yaw_streams.
            t_mid = (times[a] + times[b]) / 2.0
            if abs(t_mid - t_i) > window_s / 2.0:
                out.append(float("nan"))
                continue
        else:
            a = max(0, i - window)
            b = min(n - 1, i + window)
        if not (math.isfinite(lat[a]) and math.isfinite(lat[b])):
            out.append(float("nan"))
            continue
        out.append(heading_from_latlon(lat[a], lon[a], lat[b], lon[b]))
    return out


def _yaw_from_velocity(
    frames: list["_Frame"],
    pos_rows: list[PosRow],
    max_gap_s: float,
    min_speed_mps: float = 0.5,
) -> list[float]:
    """Per-sample compass heading from the Post-processing solver's North/East velocity.

    Velocity columns in The external solver ``.pos`` are derived from carrier Rate-signal and
    are therefore both lower-noise and physically uncorrelated with the
    position-finite-difference noise that ``_yaw_from_trajectory`` is exposed
    to. Returns NaN where the speed falls under ``min_speed_mps`` or where
    velocity columns are absent (NaN in the parsed row).
    """
    if not pos_rows:
        return [float("nan")] * len(frames)
    times = [r.utc_s for r in pos_rows]
    out: list[float] = []
    speed_sq_min = min_speed_mps * min_speed_mps
    for f in frames:
        v = interp_pos_with_velocity(pos_rows, f.utc_s, max_gap_s, times=times)
        if v is None:
            out.append(float("nan"))
            continue
        _, _, _, vn, ve, _ = v
        if not (math.isfinite(vn) and math.isfinite(ve)):
            out.append(float("nan"))
            continue
        if vn * vn + ve * ve < speed_sq_min:
            out.append(float("nan"))
            continue
        h = math.degrees(math.atan2(ve, vn))
        out.append(h + 360.0 if h < 0 else h)
    return out


def _merge_yaw_streams(
    yaw_velocity: list[float], yaw_traj: list[float]
) -> list[float]:
    """Prefer velocity-derived yaw; fall back to path-derived where NaN."""
    n = len(yaw_traj)
    out: list[float] = [float("nan")] * n
    for i in range(n):
        v = yaw_velocity[i] if i < len(yaw_velocity) else float("nan")
        if math.isfinite(v):
            out[i] = v
        else:
            t = yaw_traj[i]
            if math.isfinite(t):
                out[i] = t
    return out


def _pitch_roll_for_frames(
    frames: list[_Frame],
    data_log: Path,
    yaw_pitch_roll_sigma_s: float,
    decimate_hz: float,
    max_gap_s: float,
    log: LogFn,
) -> tuple[list[float], list[float]]:
    rows_full = parse_orientation(data_log)
    if not rows_full:
        log("[csv] no OrientationDeg lines found; pitch/roll columns left blank.")
        return [float("nan")] * len(frames), [float("nan")] * len(frames)
    rows = decimate_orientation(rows_full, decimate_hz)
    log(
        f"[csv] orientation: {len(rows_full)} -> {len(rows)} samples "
        f"(decimated to ~{decimate_hz:g} Hz)"
    )

    if yaw_pitch_roll_sigma_s > 0 and len(rows) > 1:
        rate = estimate_rate_hz([r.utc_s for r in rows])
        sigma = max(1.0, yaw_pitch_roll_sigma_s * rate)
        smooth_pitch = gaussian_smooth([r.pitch for r in rows], sigma)
        smooth_roll = gaussian_smooth([r.roll for r in rows], sigma)
        smooth_yaw = gaussian_smooth_circular_deg([r.yaw for r in rows], sigma)
        smoothed = [
            type(r)(
                utc_s=r.utc_s, yaw=smooth_yaw[i], roll=smooth_roll[i],
                pitch=smooth_pitch[i], cal=r.cal
            )
            for i, r in enumerate(rows)
        ]
        log(
            f"[csv] orientation smoothed: sigma={yaw_pitch_roll_sigma_s:g}s "
            f"({sigma:.1f} samples)"
        )
    else:
        smoothed = rows

    pitches = [float("nan")] * len(frames)
    rolls = [float("nan")] * len(frames)
    smoothed_times = [r.utc_s for r in smoothed]
    for i, f in enumerate(frames):
        o = interp_orient(smoothed, f.utc_s, max_gap_s, times=smoothed_times)
        if o is None:
            continue
        pitches[i] = o.pitch
        rolls[i] = o.roll
    return pitches, rolls


def _apply_gravity_corrections(
    frames: "list[_Frame]",
    pitches: list[float],
    rolls: list[float],
    pos_rows: list[PosRow],
    sensors_txt: Path,
    log: "LogFn",
) -> tuple[list[float], list[float]]:
    """Override pitch/roll with gravity-derived values during static stops.

    During stops the linear sensor measures only gravity → absolute pitch/roll
    with no rate sensor drift. This produces reference-quality orientation for samples
    captured while the vehicle is stationary (e.g. at intersections).
    """
    try:
        imu_rows = parse_imu(sensors_txt)
    except Exception as e:
        log(f"[csv] gravity orientation: could not parse {sensors_txt.name}: {e}")
        return pitches, rolls

    static_periods = detect_static_periods(pos_rows)
    anchors = gravity_pitch_roll_from_static(imu_rows, static_periods)

    if not anchors:
        log("[csv] gravity orientation: no static periods found in PPK velocity data")
        return pitches, rolls

    n_replaced = 0
    frame_times = [f.utc_s for f in frames]
    from bisect import bisect_left as _bl
    for t_start, t_end in static_periods:
        i0 = _bl(frame_times, t_start)
        if i0 >= len(frames):
            continue
        # Find the gravity anchor closest to this period's midpoint.
        mid = (t_start + t_end) / 2.0
        best = min(anchors, key=lambda a: abs(a[0] - mid))
        _, grav_pitch, grav_roll = best
        for i in range(i0, len(frames)):
            if frame_times[i] > t_end:
                break
            pitches[i] = grav_pitch
            rolls[i] = grav_roll
            n_replaced += 1

    log(
        f"[csv] gravity orientation: {len(static_periods)} static periods, "
        f"{len(anchors)} gravity anchors, {n_replaced} frames corrected"
    )
    return pitches, rolls


# ─── Motion sensor fusion: source-sample YPR from Complementary-update quaternion ──────────────────────

def _qrot_b2w(q: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
    """Rotate body vector v to world (Local-frame) sample via quaternion q [w,x,y,z]."""
    w, x, y, z = q
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return _np.array([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ])


def _detect_camera_up_body(mount_pitch_deg: float, mount_roll_deg: float) -> "np.ndarray":
    """Source up vector in body sample, auto-detected from mount orientation.

    The linear sensor at rest measures specific force = R^T * [0,0,+g] (Local-frame),
    which points toward world UP in body sample.  For a horizontal dashcam,
    camera_up ≈ world UP ≈ specific force direction.

    specific_force_body_hat = (-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll))
    Dominant axis determines mounting type; snap to nearest axis unit vector.

      |ax| dominant  landscape (device on its side)
      |ay| dominant  portrait on steep windshield
      |az| dominant  portrait on shallow windshield / near-flat mount
    """
    p = math.radians(mount_pitch_deg)
    r = math.radians(mount_roll_deg)
    # Specific force direction in body (= world UP in body sample)
    sf_x = -math.sin(p)
    sf_y =  math.cos(p) * math.sin(r)
    sf_z =  math.cos(p) * math.cos(r)   # positive (not negative)
    sf = _np.array([abs(sf_x), abs(sf_y), abs(sf_z)])
    dom = int(sf.argmax())
    if dom == 0:
        return _np.array([math.copysign(1.0, sf_x), 0.0, 0.0])
    elif dom == 1:
        return _np.array([0.0, math.copysign(1.0, sf_y), 0.0])
    else:
        return _np.array([0.0, 0.0, math.copysign(1.0, sf_z)])


def _apply_image_rotation_cw(camera_up: "np.ndarray", degrees_cw: int) -> "np.ndarray":
    """Rotate camera_up_body by image rotation.

    Each 90° CW image rotation: new source top = old source left = −(old right).
    camera_right = cross(camera_up, −camera_fwd) in right-hand source sample.
    """
    fwd = _np.array([0.0, 0.0, -1.0])
    up = camera_up.astype(float).copy()
    for _ in range((degrees_cw // 90) % 4):
        right = _np.cross(up, -fwd)
        right /= float(_np.linalg.norm(right)) + 1e-12
        up = -right
    return up


def _read_image_wh(path: "Path") -> "tuple[int,int] | None":
    """Return (width, height) from a JPEG or PNG without Pillow."""
    import struct
    try:
        with open(path, "rb") as f:
            header = f.read(131072)  # 128 KB covers most JPEG SOF markers
        if header[:2] == b"\xff\xd8":  # JPEG
            i = 2
            while i < len(header) - 8:
                if header[i] != 0xFF:
                    i += 1
                    continue
                m = header[i + 1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
                    h, w = struct.unpack(">HH", header[i + 5: i + 9])
                    return w, h
                if m in (0xD8, 0xD9):
                    break
                seg_len = struct.unpack(">H", header[i + 2: i + 4])[0]
                i += 2 + seg_len
        elif header[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
            w, h = struct.unpack(">II", header[16:24])
            return w, h
    except Exception:
        pass
    return None


def _auto_image_rotation_cw(frames: list, cam_up_body: "np.ndarray") -> int:
    """Infer image_rotation_cw_deg from first sample dimensions vs mount type.

    Logic:
      - Detect device mounting from dominant specific-force axis in cam_up_body.
      - Read first readable sample's (width, height).
      - Landscape device + landscape image  → 0°  (no rotation applied)
      - Landscape device + portrait image   → 90° CW  (cam_up=[+X]→[0,1,0])
                                          or 270° CW  (cam_up=[-X]→[0,1,0])
      - Portrait device  + portrait image   → 0°
      - Portrait device  + landscape image  → 90° CW
    Returns 0 if detection fails (safe default).
    """
    # Find first image file that exists
    img_path = None
    for f in frames[:30]:
        p = Path(f.image)
        if p.exists():
            img_path = p
            break
    if img_path is None:
        return 0
    dims = _read_image_wh(img_path)
    if dims is None:
        return 0
    w, h = dims
    img_is_landscape = w > h

    dom = int(_np.abs(cam_up_body).argmax())
    device_is_landscape = (dom == 0)   # dominant X axis = landscape mount

    if device_is_landscape and img_is_landscape:
        return 0
    if device_is_landscape and not img_is_landscape:
        # landscape device → portrait image: 90° CW if cam_up=+X, 270° CW if -X
        return 90 if cam_up_body[0] > 0 else 270
    if not device_is_landscape and not img_is_landscape:
        return 0
    # portrait device → landscape image: 90° CW
    return 90


def _slerp_q(q0: "np.ndarray", q1: "np.ndarray", t: float) -> "np.ndarray":
    dot = float(_np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / float(_np.linalg.norm(out))
    theta = math.acos(min(dot, 1.0))
    s = math.sin(theta)
    return (math.sin((1.0 - t) * theta) / s) * q0 + (math.sin(t * theta) / s) * q1


def _interp_q_at(
    fused_att: list,
    t: float,
    max_gap_s: float,
    fa_times: "np.ndarray",
) -> "Optional[np.ndarray]":
    """SLERP interpolation of body→world quaternion from fused_att at time t."""
    from bisect import bisect_left as _bl
    if not fused_att:
        return None
    i = _bl(fa_times, t)
    if i == 0:
        s = fused_att[0]
        return s.q if s.q is not None and abs(s.utc_s - t) <= max_gap_s else None
    if i >= len(fused_att):
        s = fused_att[-1]
        return s.q if s.q is not None and abs(t - s.utc_s) <= max_gap_s else None
    s0, s1 = fused_att[i - 1], fused_att[i]
    if s0.q is None or s1.q is None:
        return None
    gap = s1.utc_s - s0.utc_s
    if gap > max_gap_s:
        return None
    alpha = (t - s0.utc_s) / max(gap, 1e-9)
    return _slerp_q(_np.array(s0.q), _np.array(s1.q), alpha)


def _camera_ypr_from_q(
    q: "np.ndarray",
    cam_fwd_body: "np.ndarray",
    cam_up_body: "np.ndarray",
) -> tuple[float, float, float]:
    """Coordinate output YPR from body→Local-frame quaternion and source orientation in body sample.

    Returns (yaw_deg, pitch_deg, roll_deg).
    Yaw: heading from North CW.  Pitch: + = nose up.  Roll: + = right side down.
    """
    fwd_w = _qrot_b2w(q, cam_fwd_body)
    up_w  = _qrot_b2w(q, cam_up_body)

    yaw = math.degrees(math.atan2(float(fwd_w[0]), float(fwd_w[1]))) % 360.0

    fh = math.sqrt(float(fwd_w[0])**2 + float(fwd_w[1])**2)
    pitch = math.degrees(math.atan2(float(fwd_w[2]), fh))

    fwd_n = fwd_w / (float(_np.linalg.norm(fwd_w)) + 1e-9)
    world_up = _np.array([0.0, 0.0, 1.0])
    wu_perp = world_up - float(_np.dot(world_up, fwd_n)) * fwd_n
    cu_perp = up_w     - float(_np.dot(up_w,     fwd_n)) * fwd_n
    nwu = float(_np.linalg.norm(wu_perp))
    ncu = float(_np.linalg.norm(cu_perp))
    if nwu > 1e-6 and ncu > 1e-6:
        wu_n = wu_perp / nwu
        cu_n = cu_perp / ncu
        cos_r = float(_np.clip(_np.dot(wu_n, cu_n), -1.0, 1.0))
        cross = _np.cross(wu_n, cu_n)
        sin_r = float(_np.dot(cross, fwd_n))
        roll = math.degrees(math.atan2(sin_r, cos_r))
    else:
        roll = 0.0

    return yaw, pitch, roll


def _build_ekf_shape(
    pos_rows: list[PosRow],
    sensors_txt: Path,
    options: "CsvOptions",
    log: LogFn,
) -> list:
    """Run Complementary-update + position EKF on sensors_*.txt + Post-processing; return DataFix-like rows.

    Variant is picked by ``options.fused_bend_shape_source``:

    * ``"ekf"``  -- 6-state pos+vel EKF (legacy ``run_position_ekf``).
    * ``"ekf2"`` -- 9-state EKF with online linear sensor-bias estimation +
                    Rate-signal-velocity update + ZUPT during stops
                    (``run_position_ekf_v2``).

    Output rows carry ``provider="ekf_ins"`` so :func:`bend_fused_to_ppk`
    picks them up via its provider filter.
    """
    from ..imu_gnss_fusion import (
        run_mahony,
        run_position_ekf,
        run_position_ekf_v2,
    )
    from ..parsers import DataFix as _DataFix
    from ..parsers import parse_imu

    imu_rows = parse_imu(sensors_txt)
    if not imu_rows:
        log("[fusion] sensors_*.txt parsed 0 IMU rows; falling back to device fixes")
        return []
    log(f"[fusion] IMU rows: {len(imu_rows)} "
        f"({imu_rows[0].utc_s:.1f}-{imu_rows[-1].utc_s:.1f})")
    _, quaternions = run_mahony(imu_rows, pos_rows)
    log(f"[fusion] Mahony quaternions: {len(quaternions)}")

    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    if options.fused_bend_shape_source == "ekf2":
        ekf_rows = run_position_ekf_v2(
            imu_rows, pos_rows, quaternions, ref_llh,
            accel_noise_std=options.fused_bend_ekf_accel_noise_std,
            bias_random_walk_std=options.fused_bend_ekf_bias_rw_std,
            sigma_pos_h_m=options.fused_bend_xy_sigma_m,
            sigma_pos_z_m=options.fused_bend_z_sigma_m,
            sigma_vel_h_mps=options.fused_bend_ekf_vel_h_mps,
            sigma_vel_z_mps=options.fused_bend_ekf_vel_z_mps,
            zupt_speed_mps=options.fused_bend_ekf_zupt_speed_mps,
            sigma_zupt_mps=options.fused_bend_ekf_zupt_sigma_mps,
        )
        log(f"[fusion] EKF-v2 position rows: {len(ekf_rows)} "
            f"(accel_noise={options.fused_bend_ekf_accel_noise_std:g}, "
            f"bias_rw={options.fused_bend_ekf_bias_rw_std:g}, "
            f"vel_R=({options.fused_bend_ekf_vel_h_mps:g},"
            f"{options.fused_bend_ekf_vel_z_mps:g}) m/s, "
            f"ZUPT@{options.fused_bend_ekf_zupt_speed_mps:g} m/s)")
    else:
        ekf_rows = run_position_ekf(
            imu_rows, pos_rows, quaternions, ref_llh,
            accel_noise_std=options.fused_bend_ekf_accel_noise_std,
        )
        log(f"[fusion] EKF position rows: {len(ekf_rows)} "
            f"(accel_noise_std={options.fused_bend_ekf_accel_noise_std:g} m/s^2)")

    fixes = [
        _DataFix(
            utc_s=r.utc_s,
            provider="ekf_ins",
            lat=r.lat_deg, lon=r.lon_deg, h=r.h_m,
            h_acc=1.0, v_acc=2.0,
        )
        for r in ekf_rows
    ]
    return fixes


def _build_kalman_shape(
    pos_rows: list[PosRow],
    sensors_txt: Optional[Path],
    frames: list[_Frame],
    options: "CsvOptions",
    log: LogFn,
) -> list:
    """Run a textbook Recursive filter (CV or CTRV) on Post-processing pos+Rate-signal, return
    DataFix-like dense rows sampled at sample UTC instants.

    No linear sensor integration (which is what poisons the Motion sensor EKFs on device-
    grade sensors). CTRV optionally consumes rate sensor yaw rate.
    """
    from ..kalman_simple import run_ctrv_ekf, run_cv_kf
    from ..parsers import DataFix as _DataFix
    from ..parsers import parse_imu

    src = options.fused_bend_shape_source
    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    out_t = [f.utc_s for f in frames]

    if src == "cv":
        rows = run_cv_kf(
            pos_rows, ref_llh, out_times=out_t,
            accel_noise_std=options.fused_bend_ekf_accel_noise_std,
            sigma_pos_h_m=options.fused_bend_xy_sigma_m,
            sigma_vel_h_mps=options.fused_bend_ekf_vel_h_mps,
        )
        log(f"[fusion] CV-KF: {len(rows)} dense rows "
            f"(accel_noise={options.fused_bend_ekf_accel_noise_std:g} m/s^2, "
            f"sigma_pos={options.fused_bend_xy_sigma_m:g} m, "
            f"sigma_vel={options.fused_bend_ekf_vel_h_mps:g} m/s)")
    else:  # ctrv
        imu_rows = None
        if sensors_txt is not None:
            try:
                imu_rows = parse_imu(sensors_txt)
                log(f"[fusion] CTRV: ingesting {len(imu_rows)} gyro samples")
            except Exception as e:
                log(f"[fusion] CTRV: could not parse {sensors_txt.name}: {e}")
                imu_rows = None
        rows = run_ctrv_ekf(
            pos_rows, ref_llh, out_times=out_t, imu_rows=imu_rows,
            accel_noise_std=options.fused_bend_ekf_accel_noise_std,
            sigma_pos_h_m=options.fused_bend_xy_sigma_m,
            sigma_vel_h_mps=options.fused_bend_ekf_vel_h_mps,
        )
        log(f"[fusion] CTRV-EKF: {len(rows)} dense rows "
            f"(accel_noise={options.fused_bend_ekf_accel_noise_std:g} m/s^2)")

    fixes = [
        _DataFix(
            utc_s=r.utc_s, provider="ekf_ins",
            lat=r.lat_deg, lon=r.lon_deg, h=r.h_m,
            h_acc=1.0, v_acc=2.0,
        )
        for r in rows
    ]
    return fixes


def _write_fused_bend_trust_sidecar(
    out_csv: Path,
    frames: list[_Frame],
    lat: list[float],
    lon: list[float],
    height: list[float],
    trust: list[float],
    log: LogFn,
) -> Path:
    """Write ``<out_csv_stem>_trust.csv`` with one row per sample:

        Image, Latitude, Longitude, Altitude, Trust

    Trust is the per-sample [0, 1] Post-processing-anchor-trust score from
    :func:`bend_fused_to_ppk` (1.0 = bent fully onto Post-processing, 0.0 = pure FLP).
    Consumed by :func:`stages.viewers.build_trust_viewer`.
    """
    sidecar = out_csv.with_name(out_csv.stem + "_trust.csv")
    with sidecar.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image", "Latitude", "Longitude", "Altitude", "Trust"])
        for fr, la, lo, hh, t in zip(frames, lat, lon, height, trust):
            if not (math.isfinite(la) and math.isfinite(lo)):
                continue
            h_out = hh if math.isfinite(hh) else 0.0
            w.writerow([fr.image, f"{la:.9f}", f"{lo:.9f}",
                        f"{h_out:.3f}", f"{t:.4f}"])
    log(f"[csv] wrote trust sidecar {sidecar}")
    return sidecar


def run(
    *,
    frame_times_csv: Path,
    recording_map: Path,
    pos_file: Path,
    data_log: Path,
    sensors_txt: Optional[Path] = None,
    out_csv: Path,
    fps: float | None = None,
    options: Optional[CsvOptions] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    video_path: Optional[Path] = None,
    log: Optional[LogFn] = None,
) -> CsvResult:
    log_ = make_logger(log)
    options = options or CsvOptions()
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Parse raw Post-processing rows (used for yaw-from-velocity and as fusion input).
    pos_rows = parse_rtkpos(pos_file)
    if not pos_rows:
        raise RuntimeError(f"No PPK rows parsed from {pos_file}")

    # Optional Motion sensor/Signal fusion: Complementary-update attitude only (position unchanged).
    # sensors_*.txt column 0 is Reference-epoch seconds in both capture formats, so
    # parse_imu maps it to UTC directly — no boottime bridge needed here (that
    # bridge applies only to the media/session timeline, handled below).
    fused_att: list[_AttSample] = []
    imu_rows_fuse: list = []
    if sensors_txt is not None:
        try:
            imu_rows_fuse = parse_imu(sensors_txt)
        except Exception as _e:
            log_(f"[csv] IMU parse failed: {_e}")
    if options.use_imu_fusion and imu_rows_fuse:
        try:
            _, fused_att = _imu_gnss_fuse(imu_rows_fuse, pos_rows, log=log_)
        except Exception as _e:
            log_(f"[fusion] attitude fusion failed, continuing without: {_e}")

    # Load samples with clock offset detection (needs Motion sensor + Post-processing).
    frames, anchor = _load_frames(
        frame_times_csv, recording_map, log_,
        imu_rows=imu_rows_fuse or None,
        pos_rows=pos_rows,
        video_offset_s=options.video_offset_s,
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        measurements_txt=data_log,
        chop_video_anchor=chop_video_anchor,
        video_path=video_path,
    )
    if not frames:
        raise RuntimeError(f"No frames found in {frame_times_csv}")
    log_(f"[csv] loaded {len(frames)} frame timestamps")

    # Derive effective FPS from sample timings if not supplied.
    if fps is None:
        fps = _derive_effective_fps(frames)
        log_(f"[csv] effective fps derived from frame timings: {fps:.3f} Hz")
    else:
        derived_fps = _derive_effective_fps(frames)
        log_(
            f"[csv] fps: user supplied {fps:.3f} Hz, "
            f"derived from timings {derived_fps:.3f} Hz"
        )

    # Optional FGO smoothing: The factor library factor graph over Post-processing + Motion sensor. Replaces
    # ``pos_rows`` with the smoothed path before sample interpolation.
    if options.use_fgo_smoothing:
        if sensors_txt is None:
            log_("[csv] use_fgo_smoothing requested but no sensors_txt; skipping FGO")
        else:
            try:
                from ..fgo import FgoOptions, run_fgo
                imu_for_fgo = imu_rows_fuse or parse_imu(sensors_txt)
                fgo_res = run_fgo(pos_rows, imu_for_fgo, options=FgoOptions(), log=log_)
                # Replace pos_rows with FGO-smoothed (keep all other fields per epoch)
                fgo_rows: list[PosRow] = []
                for i, r in enumerate(pos_rows):
                    if not (math.isfinite(fgo_res.lat_deg[i])
                            and math.isfinite(fgo_res.lon_deg[i])
                            and math.isfinite(fgo_res.h_m[i])):
                        fgo_rows.append(r); continue
                    fgo_rows.append(PosRow(
                        utc_s=r.utc_s,
                        lat_deg=fgo_res.lat_deg[i],
                        lon_deg=fgo_res.lon_deg[i],
                        h_m=fgo_res.h_m[i],
                        quality=r.quality, ns=r.ns,
                        sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                        ve=r.ve, vn=r.vn, vu=r.vu,
                    ))
                pos_rows = fgo_rows
                log_(f"[csv] FGO smoothing applied: {fgo_res.n_factors} factors, "
                     f"converged={fgo_res.converged}")
            except Exception as _e:
                log_(f"[csv] FGO smoothing failed, continuing with raw PPK: {_e}")

    lat, lon, h, has_pos = _interp_from_rows(frames, pos_rows, options.max_interp_gap_s)
    n_pos = sum(has_pos)
    log_(f"[csv] interpolated PPK at {n_pos}/{len(frames)} frames")

    xy_sigma, z_sigma = options.sigmas()
    if options.smoothing == "fused-bent":
        src = options.fused_bend_shape_source
        if src in ("ekf", "ekf2") and sensors_txt is not None:
            fused_fixes = _build_ekf_shape(pos_rows, sensors_txt, options, log_)
            if not fused_fixes:
                log_("[csv] EKF produced no rows; falling back to device Fix shape")
                fused_fixes = parse_data_fix(data_log)
        elif src in ("cv", "ctrv"):
            fused_fixes = _build_kalman_shape(
                pos_rows, sensors_txt, frames, options, log_,
            )
            if not fused_fixes:
                log_("[csv] Kalman produced no rows; falling back to device Fix shape")
                fused_fixes = parse_data_fix(data_log)
        else:
            if src in ("ekf", "ekf2") and sensors_txt is None:
                log_("[csv] shape_source=ekf* requested but no sensors_txt path; "
                     "using device Fix shape")
            fused_fixes = parse_data_fix(data_log)
        bend_opts = FusedBendOptions(
            xy_sigma_m=options.fused_bend_xy_sigma_m,
            z_sigma_m=options.fused_bend_z_sigma_m,
            reject_k=options.fused_bend_reject_k,
            time_smooth_s=options.fused_bend_time_smooth_s,
            max_gap_s=options.max_interp_gap_s,
            car_lateral_sigma_m=options.fused_bend_car_lateral_sigma_m,
            car_smooth_s=options.fused_bend_car_smooth_s,
            car_min_speed_mps=options.fused_bend_car_min_speed_mps,
        )
        lat_s, lon_s, h_s, has_pos, trust_arr, bend_res = bend_fused_to_ppk(
            fused_fixes, pos_rows, [f.utc_s for f in frames],
            options=bend_opts,
        )
        n_pos = sum(has_pos)
        _write_fused_bend_trust_sidecar(
            out_csv, frames, lat_s, lon_s, h_s, trust_arr, log_,
        )
        log_(
            f"[csv] fused-bent: fused n={bend_res.n_fused} ppk n={bend_res.n_ppk} "
            f"anchors used={bend_res.n_anchors_used} rejected={bend_res.n_anchors_rejected} "
            f"(xy_sigma={bend_opts.xy_sigma_m:g} m, z_sigma={bend_opts.z_sigma_m:g} m, "
            f"reject>{bend_opts.reject_k:g}-sigma, time_smooth={bend_opts.time_smooth_s:g}s); "
            f"horiz residual median={bend_res.median_residual_m:.2f} m "
            f"p95={bend_res.p95_residual_m:.2f} m; "
            f"car constraint (sigma_lat={bend_opts.car_lateral_sigma_m:g} m, "
            f"smooth={bend_opts.car_smooth_s:g}s, min-speed={bend_opts.car_min_speed_mps:g} m/s): "
            f"flagged {bend_res.n_car_flagged} epochs, "
            f"lateral median={bend_res.median_lateral_m:.2f} m "
            f"p95={bend_res.p95_lateral_m:.2f} m; "
            f"bent positions for {n_pos}/{len(frames)} frames"
        )
    else:
        ns_per_frame = (
            _ns_per_frame(frames, pos_rows)
            if options.use_ns_adaptive_smoothing else None
        )
        lat_s, lon_s, h_s = _smooth_trajectory(
            lat, lon, h, fps, xy_sigma, z_sigma,
            frames=frames, use_rts=options.use_rts_smoother,
            ns_per_sample=ns_per_frame,
        )
        if options.use_rts_smoother:
            log_(
                f"[csv] trajectory smoothed with RTS Kalman smoother "
                f"(process_noise_std=0.01 m/s²)"
            )
        else:
            log_(
                f"[csv] smoothing profile={options.smoothing} "
                f"(xy_sigma={xy_sigma:g}s, z_sigma={z_sigma:g}s)"
            )

    yaws: list[float] = []
    pitches: list[float] = []
    rolls: list[float] = []
    if options.add_ypr:
        if fused_att and options.use_imu_fusion:
            # Compute mount-orientation reference from median gravity anchor.
            # Complementary-update outputs body-sample angles; the device mounting offset
            # (e.g. landscape right-side-up gives body-pitch ≈ -85°) must be
            # removed so the output is in Coordinate output's source sample (pitch=0 = horizontal).
            _sp_fuse = detect_static_periods(pos_rows)
            _anchors_fuse = (gravity_pitch_roll_from_static(imu_rows_fuse, _sp_fuse)
                             if imu_rows_fuse else [])
            if _anchors_fuse:
                _sorted_p = sorted(a[1] for a in _anchors_fuse)
                _sorted_r = sorted(a[2] for a in _anchors_fuse)
                _mount_pitch = _sorted_p[len(_sorted_p) // 2]   # median
                _mount_roll  = _sorted_r[len(_sorted_r) // 2]
            else:
                _mount_pitch, _mount_roll = 0.0, 0.0
            log_(f"[fusion] mount ref: pitch={_mount_pitch:.1f}° roll={_mount_roll:.1f}°"
                 f"  ({len(_anchors_fuse)} gravity anchors)")

            # Yaw from Signal velocity — avoids the body-sample yaw offset
            # caused by the device's landscape mounting orientation.
            yaws_vel = _yaw_from_velocity(frames, pos_rows, options.max_interp_gap_s)
            yaws_traj = _yaw_from_trajectory(
                lat_s, lon_s, times=[f.utc_s for f in frames]
            )
            _n_vel_fuse = sum(1 for v in yaws_vel if math.isfinite(v))
            yaws = _merge_yaw_streams(yaws_vel, yaws_traj)
            log_(f"[csv] yaw: {_n_vel_fuse}/{len(frames)} from PPK velocity, "
                 f"rest from trajectory finite-difference")
            _rate_hz = _rate_hz_from_frames(frames, fps)
            if xy_sigma > 0:
                yaws = gaussian_smooth_circular_deg(
                    yaws, max(1.0, xy_sigma * _rate_hz))

            # Source-sample YPR from full Complementary-update quaternion.
            # Detects mount orientation (landscape/portrait/etc.) from median gravity
            # direction; auto-detects image rotation from first sample dimensions.
            _cam_up = _detect_camera_up_body(_mount_pitch, _mount_roll)
            _img_rot = _auto_image_rotation_cw(frames, _cam_up)
            _cam_up = _apply_image_rotation_cw(_cam_up, _img_rot)
            _cam_fwd = _np.array([0.0, 0.0, -1.0])
            _mounting = ("landscape" if abs(math.sin(math.radians(_mount_pitch))) > 0.7
                         else "portrait")
            log_(f"[fusion] mounting={_mounting}  camera_up_body={_cam_up.tolist()}"
                 f"  image_rot={_img_rot}° (auto)")

            _fa_times = _np.array([s.utc_s for s in fused_att])
            pitches = [float("nan")] * len(frames)
            rolls   = [float("nan")] * len(frames)
            _pit_prior = (options.pitch_prior_deg if math.isfinite(options.pitch_prior_deg)
                          else 0.0)
            _rol_prior = (options.roll_prior_deg if math.isfinite(options.roll_prior_deg)
                          else 0.0)
            _MAX_PITCH_DEV = 25.0  # reject pre-mount / large disturbance samples
            _MAX_ROLL_DEV  = 30.0
            for i, f in enumerate(frames):
                _q = _interp_q_at(fused_att, f.utc_s, options.max_interp_gap_s, _fa_times)
                if _q is not None:
                    _, _pitch_cam, _roll_cam = _camera_ypr_from_q(_q, _cam_fwd, _cam_up)
                    if abs(_pitch_cam - _pit_prior) <= _MAX_PITCH_DEV:
                        pitches[i] = _pitch_cam
                    if abs(_roll_cam - _rol_prior) <= _MAX_ROLL_DEV:
                        rolls[i] = _roll_cam
            # Smooth at sample rate. Use actual inter-sample Δt for adaptive
            # selection where nominal fps is meaningless.
            if options.yaw_pitch_roll_sigma_s > 0:
                _sigma_fr = max(1.0,
                                options.yaw_pitch_roll_sigma_s
                                * _rate_hz_from_frames(frames, fps))
                pitches = gaussian_smooth(pitches, _sigma_fr)
                rolls   = gaussian_smooth(rolls,   _sigma_fr)
            n_fused_att = sum(1 for p in pitches if math.isfinite(p))
            n_fused_roll = sum(1 for r in rolls if math.isfinite(r))
            log_(f"[fusion] pitch={n_fused_att}/{len(frames)} frames  "
                 f"roll={n_fused_roll}/{len(frames)} frames (from quaternion)")
        else:
            # Prefer Rate-signal-derived velocity heading (low noise, decoupled from
            # the position-path smoothing); fall back to path-derived
            # heading where velocity is missing (NaN cols) or below threshold.
            yaws_vel = _yaw_from_velocity(
                frames, pos_rows, options.max_interp_gap_s
            )
            yaws_traj = _yaw_from_trajectory(
                lat_s, lon_s, times=[f.utc_s for f in frames]
            )
            n_vel = sum(1 for v in yaws_vel if math.isfinite(v))
            yaws = _merge_yaw_streams(yaws_vel, yaws_traj)
            log_(
                f"[csv] yaw: {n_vel}/{len(frames)} from PPK velocity, "
                f"rest from trajectory finite-difference"
            )
            if xy_sigma > 0:
                yaws = gaussian_smooth_circular_deg(
                    yaws, max(1.0, xy_sigma * _rate_hz_from_frames(frames, fps)))
            pitches, rolls = _pitch_roll_for_frames(
                frames=frames,
                data_log=data_log,
                yaw_pitch_roll_sigma_s=options.yaw_pitch_roll_sigma_s,
                decimate_hz=options.decimate_orient_hz,
                max_gap_s=options.max_interp_gap_s,
                log=log_,
            )
            if options.use_gravity_orientation and sensors_txt is not None:
                pitches, rolls = _apply_gravity_corrections(
                    frames=frames,
                    pitches=pitches,
                    rolls=rolls,
                    pos_rows=pos_rows,
                    sensors_txt=sensors_txt,
                    log=log_,
                )

    # Per-sample speed/velocity columns.
    # Rate-signal Local-frame components come from Post-processing fine measurements rate (device angle irrelevant).
    # ve = East m/s, vn = North m/s, vu = Up m/s (vu not used for horizontal speed).
    # Horizontal azimuth = atan2(ve, vn) degrees clockwise from North.
    _pos_vtimes = [r.utc_s for r in pos_rows]
    doppler_speeds: list[float] = [float("nan")] * len(frames)
    doppler_ve: list[float] = [float("nan")] * len(frames)
    doppler_vn: list[float] = [float("nan")] * len(frames)
    doppler_vu: list[float] = [float("nan")] * len(frames)
    coords_speeds: list[float] = [float("nan")] * len(frames)

    for _i, _f in enumerate(frames):
        _v = interp_pos_with_velocity(pos_rows, _f.utc_s, options.max_interp_gap_s,
                                      times=_pos_vtimes)
        if _v is not None:
            _, _, _, _vn, _ve, _vu = _v
            if math.isfinite(_vn) and math.isfinite(_ve):
                doppler_speeds[_i] = math.sqrt(_vn * _vn + _ve * _ve)
                doppler_ve[_i] = _ve
                doppler_vn[_i] = _vn
            if math.isfinite(_vu):
                doppler_vu[_i] = _vu

    _ref_llh_sp: Optional[tuple[float, float, float]] = None
    for _la, _lo, _he in zip(lat_s, lon_s, h_s):
        if math.isfinite(_la) and math.isfinite(_lo) and math.isfinite(_he):
            _ref_llh_sp = (_la, _lo, _he)
            break
    if _ref_llh_sp is not None:
        _enu_xy: list[Optional[tuple[float, float]]] = []
        for _la, _lo, _he in zip(lat_s, lon_s, h_s):
            if math.isfinite(_la) and math.isfinite(_lo) and math.isfinite(_he):
                _x, _y, _z = llh_to_ecef(_la, _lo, _he)
                _e, _n, _ = ecef_to_enu(_x, _y, _z, _ref_llh_sp)
                _enu_xy.append((_e, _n))
            else:
                _enu_xy.append(None)
        for _i in range(len(frames)):
            if not has_pos[_i] or _enu_xy[_i] is None:
                continue
            _a, _b = _i - 1, _i + 1
            if 0 <= _a and _b < len(frames) and _enu_xy[_a] is not None and _enu_xy[_b] is not None:
                _dt = frames[_b].utc_s - frames[_a].utc_s
            elif _b < len(frames) and _enu_xy[_b] is not None:
                _a, _dt = _i, frames[_b].utc_s - frames[_i].utc_s
            elif 0 <= _a and _enu_xy[_a] is not None:
                _b, _dt = _i, frames[_i].utc_s - frames[_a].utc_s
            else:
                continue
            if _dt <= 0:
                continue
            _de = _enu_xy[_b][0] - _enu_xy[_a][0]
            _dn = _enu_xy[_b][1] - _enu_xy[_a][1]
            coords_speeds[_i] = math.sqrt(_de * _de + _dn * _dn) / _dt

    headers: list[str] = ["Image", "Latitude", "Longitude"]
    if options.include_altitude:
        headers.append("Altitude")
    headers += ["AccuracyX", "AccuracyY"]
    if options.include_altitude:
        headers.append("AccuracyZ")
    if options.add_ypr:
        headers += [
            "Yaw",
            "Pitch",
            "Roll",
            "AccuracyYaw",
            "AccuracyPitch",
            "AccuracyRoll",
        ]
    # Local-frame Rate-signal components: ve=East, vn=North, vu=Up (all m/s, The standard datum local Local-frame).
    # Azimuth = atan2(DopplerVe, DopplerVn) degrees clockwise from North.
    headers += ["DopplerVe_mps", "DopplerVn_mps", "DopplerVu_mps",
                "DopplerSpeed_mps", "CoordsSpeed_mps"]

    n_orient = 0
    with out_csv.open("w", newline="", encoding="utf-8") as g:
        if options.emit_header_comment:
            # Leading '#' comment lines are ignored by the external tool's reference
            # CSV importer; they make the file self-describing. Image label is
            # written WITHOUT extension to match source.label (see below).
            g.write(
                "# the external tool reference CSV (client_pipeline). "
                "Image=camera label (no extension), Latitude/Longitude=WGS84 deg, "
                "Altitude=m (optional), Accuracy*=1-sigma (m / deg), "
                "Yaw/Pitch/Roll=deg, Doppler*=m/s. CRS EPSG:4326.\n"
            )
        w = csv.writer(g)
        w.writerow(headers)
        for i, f in enumerate(frames):
            if not has_pos[i]:
                continue
            # the external tool matches reference rows to sources by label, which is
            # the filename WITHOUT extension by default. Strip it so the join
            # succeeds; otherwise 0 sources get coordinates ("format broken").
            image_label = Path(f.image).stem if options.label_strip_ext else f.image
            row: list[str] = [
                image_label,
                f"{lat_s[i]:.9f}",
                f"{lon_s[i]:.9f}",
            ]
            if options.include_altitude:
                row.append(f"{h_s[i]:.4f}")
            row += [
                f"{options.accuracy_x_m:.3f}",
                f"{options.accuracy_y_m:.3f}",
            ]
            if options.include_altitude:
                row.append(f"{options.accuracy_z_m:.3f}")
            if options.add_ypr:
                yaw_v = yaws[i] if i < len(yaws) else float("nan")
                pit_v = pitches[i] if i < len(pitches) else float("nan")
                rol_v = rolls[i] if i < len(rolls) else float("nan")
                # Apply constant prior when Motion sensor has no data for this sample.
                if not math.isfinite(pit_v) and math.isfinite(options.pitch_prior_deg):
                    pit_v = options.pitch_prior_deg
                if not math.isfinite(rol_v) and math.isfinite(options.roll_prior_deg):
                    rol_v = options.roll_prior_deg
                if math.isfinite(pit_v) and math.isfinite(rol_v):
                    n_orient += 1
                row += [
                    "" if not math.isfinite(yaw_v) else f"{yaw_v:.3f}",
                    "" if not math.isfinite(pit_v) else f"{pit_v:.3f}",
                    "" if not math.isfinite(rol_v) else f"{rol_v:.3f}",
                    f"{options.accuracy_yaw_deg:.3f}" if math.isfinite(yaw_v) else "",
                    f"{options.accuracy_pitch_deg:.3f}" if math.isfinite(pit_v) else "",
                    f"{options.accuracy_roll_deg:.3f}" if math.isfinite(rol_v) else "",
                ]
            dve = doppler_ve[i]; dvn = doppler_vn[i]; dvu = doppler_vu[i]
            dsp = doppler_speeds[i];  csp = coords_speeds[i]
            row += [
                "" if not math.isfinite(dve) else f"{dve:.4f}",
                "" if not math.isfinite(dvn) else f"{dvn:.4f}",
                "" if not math.isfinite(dvu) else f"{dvu:.4f}",
                "" if not math.isfinite(dsp) else f"{dsp:.4f}",
                "" if not math.isfinite(csp) else f"{csp:.4f}",
            ]
            w.writerow(row)

    log_(
        f"[csv] wrote {out_csv} (frames={len(frames)} pos={n_pos} orient={n_orient})"
    )
    return CsvResult(
        csv_path=out_csv,
        n_frames=len(frames),
        n_with_position=n_pos,
        n_with_orientation=n_orient,
        smoothing=options.smoothing,
        xy_sigma_s=xy_sigma,
        z_sigma_s=z_sigma,
        time_anchor=anchor,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-times-csv", required=True, type=Path)
    ap.add_argument("--recording-map", required=True, type=Path)
    ap.add_argument("--pos", required=True, type=Path)
    ap.add_argument("--data-log", required=True, type=Path)
    ap.add_argument("--capture-meta", type=Path, default=None,
                    help="capture_meta.json for boottime-anchored captures "
                         "(omit for the legacy video-PTS format)")
    ap.add_argument("--chop-video-anchor", type=Path, default=None,
                    help="trimmed ('chop') clip's own *.video_anchor.txt — "
                         "REQUIRED when georeferencing a chop: its min "
                         "bootNs overrides the parent capture_meta "
                         "video_t0_boottime_ns (chop PTS are rebased to 0)")
    ap.add_argument("--video-path", type=Path, default=None,
                    help="the (chop) .mp4 the frames came from; probed with "
                         "ffprobe for chop start-time diagnostics only")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument(
        "--smoothing",
        choices=list(_PROFILE_PRESETS) + ["custom", "fused-bent"],
        default="car",
    )
    ap.add_argument("--xy-sigma-s", type=float, default=2.0)
    ap.add_argument("--z-sigma-s", type=float, default=10.0)
    ap.add_argument(
        "--fused-bend-xy-sigma-m", type=float, default=3.0,
        help="fused-bent profile: horizontal 1-sigma trust band (m). "
             "Device PPK ~3 m default.",
    )
    ap.add_argument(
        "--fused-bend-z-sigma-m", type=float, default=15.0,
        help="fused-bent profile: vertical 1-sigma trust band (m). "
             "Device PPK ~15 m default.",
    )
    ap.add_argument(
        "--fused-bend-reject-k", type=float, default=10.0,
        help="fused-bent profile: hard-reject anchors with |residual| > "
             "k * sigma. Default 10 -> 30 m horiz cutoff (worst-case jump).",
    )
    ap.add_argument(
        "--fused-bend-time-smooth-s", type=float, default=5.0,
        help="fused-bent profile: Gaussian time-kernel width for residuals (s).",
    )
    ap.add_argument(
        "--fused-bend-car-lateral-sigma-m", type=float, default=3.0,
        help="fused-bent profile: 1-sigma band for lateral (perpendicular) "
             "residual under car non-holonomic constraint (m). Default 3 m "
             "tolerates per-epoch PPK noise; 30 m jumps land at 10-sigma.",
    )
    ap.add_argument(
        "--fused-bend-car-smooth-s", type=float, default=3.0,
        help="fused-bent profile: smoothing window for car-constraint "
             "reference path (s).",
    )
    ap.add_argument(
        "--fused-bend-car-min-speed-mps", type=float, default=0.5,
        help="fused-bent profile: below this speed the car constraint "
             "is inactive (m/s).",
    )
    ap.add_argument(
        "--fused-bend-shape-source",
        choices=["device", "ekf", "ekf2", "cv", "ctrv"],
        default="device",
        help="fused-bent profile: where the dense shape comes from. "
             "'device' = Fix lines from data log; "
             "'ekf'  = 6-state IMU/GNSS EKF on sensors_*.txt + PPK; "
             "'ekf2' = 9-state EKF with bias estimation + Doppler + ZUPT; "
             "'cv'   = 4-state constant-velocity Kalman (no IMU); "
             "'ctrv' = 5-state CTRV EKF (Udacity SDC standard).",
    )
    ap.add_argument(
        "--fused-bend-ekf-accel-noise-std", type=float, default=0.5,
        help="fused-bent profile: EKF accel process-noise stddev (m/s^2).",
    )
    ap.add_argument(
        "--fused-bend-ekf-vel-h-mps", type=float, default=1.0,
        help="ekf2: Doppler-velocity horiz 1-sigma (m/s). User spec: 1.0.",
    )
    ap.add_argument(
        "--fused-bend-ekf-vel-z-mps", type=float, default=2.0,
        help="ekf2: Doppler-velocity vertical 1-sigma (m/s).",
    )
    ap.add_argument(
        "--fused-bend-ekf-bias-rw-std", type=float, default=0.001,
        help="ekf2: accel-bias random-walk stddev (m/s^2 / sqrt(s)).",
    )
    ap.add_argument(
        "--fused-bend-ekf-zupt-speed-mps", type=float, default=0.3,
        help="ekf2: Doppler speed below this triggers a tight v=0 update.",
    )
    ap.add_argument(
        "--fused-bend-ekf-zupt-sigma-mps", type=float, default=0.05,
        help="ekf2: measurement-noise sigma for the ZUPT pseudo-update (m/s).",
    )
    ap.add_argument(
        "--sensors-txt", type=Path, default=None,
        help="Path to sensors_*.txt for IMU/EKF fusion.",
    )
    ap.add_argument("--no-ypr", action="store_true")
    ap.add_argument(
        "--include-altitude",
        action="store_true",
        help="Include Altitude / AccuracyZ columns (default: lat/lon only).",
    )
    ap.add_argument(
        "--smooth-altitude",
        action="store_true",
        help="Opt in to smoothing the altitude (Z) channel. Off by default "
             "because device Z is noisy. Use --altitude-smooth-sigma-s to set "
             "the Gaussian window; otherwise the profile / Z override is used.",
    )
    ap.add_argument(
        "--no-smooth-altitude",
        action="store_true",
        help="Force altitude (Z) smoothing OFF (pass Z through raw), even if a "
             "smoothing profile would otherwise smooth it.",
    )
    ap.add_argument(
        "--altitude-smooth-sigma-s", type=float, default=None,
        help="Gaussian sigma (seconds) for altitude smoothing when "
             "--smooth-altitude is set. Default: use the Z override / profile.",
    )
    ap.add_argument("--accuracy-x-m", type=float, default=0.10)
    ap.add_argument("--accuracy-y-m", type=float, default=0.10)
    ap.add_argument("--accuracy-z-m", type=float, default=0.30)
    args = ap.parse_args()

    options = CsvOptions(
        smoothing=args.smoothing,
        custom_xy_sigma_s=args.xy_sigma_s,
        custom_z_sigma_s=args.z_sigma_s,
        fused_bend_xy_sigma_m=args.fused_bend_xy_sigma_m,
        fused_bend_z_sigma_m=args.fused_bend_z_sigma_m,
        fused_bend_reject_k=args.fused_bend_reject_k,
        fused_bend_time_smooth_s=args.fused_bend_time_smooth_s,
        fused_bend_car_lateral_sigma_m=args.fused_bend_car_lateral_sigma_m,
        fused_bend_car_smooth_s=args.fused_bend_car_smooth_s,
        fused_bend_car_min_speed_mps=args.fused_bend_car_min_speed_mps,
        fused_bend_shape_source=args.fused_bend_shape_source,
        fused_bend_ekf_accel_noise_std=args.fused_bend_ekf_accel_noise_std,
        fused_bend_ekf_vel_h_mps=args.fused_bend_ekf_vel_h_mps,
        fused_bend_ekf_vel_z_mps=args.fused_bend_ekf_vel_z_mps,
        fused_bend_ekf_bias_rw_std=args.fused_bend_ekf_bias_rw_std,
        fused_bend_ekf_zupt_speed_mps=args.fused_bend_ekf_zupt_speed_mps,
        fused_bend_ekf_zupt_sigma_mps=args.fused_bend_ekf_zupt_sigma_mps,
        include_altitude=args.include_altitude,
        smooth_altitude=(
            True if args.smooth_altitude
            else (False if args.no_smooth_altitude else None)
        ),
        altitude_smooth_sigma_s=args.altitude_smooth_sigma_s,
        add_ypr=not args.no_ypr,
        accuracy_x_m=args.accuracy_x_m,
        accuracy_y_m=args.accuracy_y_m,
        accuracy_z_m=args.accuracy_z_m,
    )
    run(
        frame_times_csv=args.frame_times_csv,
        recording_map=args.recording_map,
        pos_file=args.pos,
        data_log=args.data_log,
        sensors_txt=args.sensors_txt,
        out_csv=args.out_csv,
        fps=args.fps,
        options=options,
        capture_meta=args.capture_meta,
        chop_video_anchor=args.chop_video_anchor,
        video_path=args.video_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
