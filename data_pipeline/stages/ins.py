"""Stage: INS-fused path (loose-coupled 9-state EKF + RTS smoother).

Glue layer that wires the EKF in :mod:`data_pipeline.stages.ekf_fusion` into
the same input/output contract as :mod:`data_pipeline.stages.georef`.

What it does:

  1. Parse ``sensors_*.txt`` (Motion sensor) + ``.pos`` (Post-processing) + ``recording_*.txt``
     (anchor map) + ``extracted_frame_times.csv``.
  2. Run forward EKF — feature 1 (dead-reckoning between Post-processing epochs),
     feature 2 (ZUPT static-period clamp), feature 5 (innovation gating —
     opt-in), feature 7 (non-holonomic — opt-in).
  3. Run RTS backwards pass — feature 3 (replaces Gaussian smoothing with
     a physical model that knows ``Δp = v·Δt`` and the linear sensor-bias state).
  4. Sample the smoothed path at every sample UTC.
  5. Write ``georef_ins.csv`` with optional yaw from the smoothed
     attitude track — feature 4 (continuous yaw even when stationary).

Output is a *drop-in replacement* for the coordinate output stage's CSV so the
downstream Coordinate output workflow doesn't care which path produced the rows.
"""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..geo import llh_to_ecef, ecef_to_enu
from ..imu_gnss_fusion import _quat_to_ypr
from ..parsers import (
    parse_imu,
    parse_rtkpos,
    read_frame_times_csv,
)
from ..frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
from ..pipeline import LogFn, make_logger
from ..time_sync import TimeAnchor, fit_time_anchor
from .ekf_fusion import EkfOptions, EkfResult, run_ekf, rts_smooth


@dataclass
class InsOptions:
    """Tunables for the INS stage."""

    # Forward EKF options (delegated). Defaults are gates-off, ZUPT-on.
    ekf: EkfOptions = None  # type: ignore[assignment]
    # Vehicle mode toggles NHC + biases the smoother for road behaviour.
    vehicle_mode: bool = False
    # Emit the Altitude / AccuracyZ columns.
    include_altitude: bool = False
    # Emit Yaw / Pitch / Roll columns from the EKF attitude track.
    add_ypr: bool = True
    # Constant per-axis accuracies written into the CSV (m / deg).
    accuracy_x_m: float = 0.10
    accuracy_y_m: float = 0.10
    accuracy_z_m: float = 0.30
    accuracy_yaw_deg: float = 10.0
    accuracy_pitch_deg: float = 5.0
    accuracy_roll_deg: float = 5.0
    # Reject samples whose UTC sits more than this many seconds outside the
    # smoothed-path time span.
    max_extrap_s: float = 0.5
    # ── Motion model bonus ─────────────────────────────────────────────────────
    # Run The feature library Sparse-feature + 5-point relative-pose on the source media to
    # extract per-sample relative motion. When enabled, ``video_path``
    # must be supplied to ``build_ins_csv``. Post-processing Rate-signal speed scales
    # the unit-norm Motion model direction to a metric Local-frame velocity that becomes
    # an additional KF measurement update between Post-processing epochs.
    use_vio: bool = False
    vio_frame_decim_hz: float = 5.0
    vio_max_features: int = 500
    vio_min_inliers: int = 40
    # Multi-sample Motion model: features tracked across ``vio_track_length`` samples
    # so the 5-point essential-matrix solver gets longer baselines.
    # Calibrated on reference session: track_length=5 cuts calibration residual
    # from 1.19° to 0.90° and improves RTS+Motion model @Post-processing hRMSE 2.930→2.862
    # (and hP50 2.267→2.207, BEATING raw Post-processing at the median for the
    # first time). Set track_length <= 1 to fall back to pairwise Motion model.
    vio_track_length: int = 5

    def __post_init__(self) -> None:
        if self.ekf is None:
            self.ekf = EkfOptions()
        if self.vehicle_mode:
            self.ekf.nhc_enabled = True


@dataclass(frozen=True)
class InsResult:
    csv_path: Path
    n_frames: int
    n_with_position: int
    n_with_orientation: int
    time_anchor: TimeAnchor
    n_pos_updates: int
    n_vel_updates: int
    n_pos_rejected: int
    n_vel_rejected: int
    n_zupt: int
    n_nhc: int
    horiz_rms_residual_m: float
    vert_rms_residual_m: float



def _sample_smoothed_at_frame_utc(
    sm_t: list[float],
    sm_lat: list[float],
    sm_lon: list[float],
    sm_h:   list[float],
    sm_yaw: list[float],
    sm_pitch: list[float],
    sm_roll:  list[float],
    frame_utc: float,
    max_extrap_s: float,
) -> Optional[tuple[float, float, float, float, float, float]]:
    """Linear interpolation of smoothed path at ``frame_utc``.

    Yaw is interpolated as a wrapped circular angle so the short way around
    is always taken; pitch/roll are bounded already.
    """
    n = len(sm_t)
    if n == 0:
        return None
    if frame_utc < sm_t[0] - max_extrap_s or frame_utc > sm_t[-1] + max_extrap_s:
        return None
    i = bisect_left(sm_t, frame_utc)
    if i <= 0:
        i = 1
    if i >= n:
        i = n - 1
    t0, t1 = sm_t[i - 1], sm_t[i]
    alpha = 0.0 if t1 == t0 else max(0.0, min(1.0, (frame_utc - t0) / (t1 - t0)))

    def _lerp(a: float, b: float) -> float:
        return a + alpha * (b - a)

    def _lerp_angle(a: float, b: float) -> float:
        d = ((b - a + 540.0) % 360.0) - 180.0
        return (a + d * alpha) % 360.0

    return (
        _lerp(sm_lat[i - 1],   sm_lat[i]),
        _lerp(sm_lon[i - 1],   sm_lon[i]),
        _lerp(sm_h[i - 1],     sm_h[i]),
        _lerp_angle(sm_yaw[i - 1], sm_yaw[i]),
        _lerp(sm_pitch[i - 1], sm_pitch[i]),
        _lerp(sm_roll[i - 1],  sm_roll[i]),
    )


def build_ins_csv(
    *,
    sensors_txt: Path,
    pos_file: Path,
    recording_map: Path,
    frame_times_csv: Path,
    out_csv: Path,
    options: Optional[InsOptions] = None,
    video_path: Optional[Path] = None,
    log: Optional[LogFn] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
) -> InsResult:
    """End-to-end: Motion sensor+Post-processing → EKF → RTS → per-sample CSV.

    See module docstring for the chain of features the EKF/RTS pair
    implements. Output CSV mirrors the Coordinate output stage's schema so the
    downstream coordinate tagging workflow is interchangeable.

    ``capture_meta`` / ``video_anchor`` / ``chop_video_anchor`` carry the
    boottime-session context (see
    :func:`data_pipeline.frame_time.resolve_video_t0_boottime_ns`). When a
    sample-0 boottime t0 resolves, each sample's PTS is lifted into bootNs
    before hitting the time anchor — without it every sample UTC on a
    boottime/segment session lands ~t0 early and misses the smoothed
    path entirely. Legacy sessions (all three ``None``, or no t0
    recoverable) keep the direct ``video_pts_to_utc_s`` mapping unchanged.
    """
    opts = options or InsOptions()
    log_ = make_logger(log)

    # ── Parse ────────────────────────────────────────────────────────────
    imu = parse_imu(sensors_txt)
    pos = parse_rtkpos(pos_file)
    if not imu:
        raise RuntimeError(
            f"No IMU rows parsed from {sensors_txt}. "
            "Verify the file is a the capture app 'sensors_*.txt' with "
            "UncalAccel/UncalGyro lines, or pass a different sensor file."
        )
    if not pos:
        raise RuntimeError(
            f"No PPK rows parsed from {pos_file}. "
            "Verify the .pos file is a non-empty RTKLIB output (re-run "
            "the PPK stage if it was empty)."
        )
    log_(f"[ins] IMU rows={len(imu)} PPK rows={len(pos)}")

    # ── Optional Motion model (bonus) ─────────────────────────────────────────────
    vio_vels = None
    if opts.use_vio:
        if video_path is None or not video_path.is_file():
            raise RuntimeError(
                "InsOptions.use_vio=True but no valid video_path supplied"
            )
        from ..vio import run_vio, run_vio_multiframe, vio_to_enu_velocities
        log_(f"[ins] VIO on {video_path.name} (decim={opts.vio_frame_decim_hz}Hz "
             f"track_len={opts.vio_track_length})")
        if opts.vio_track_length > 1:
            samples = run_vio_multiframe(
                video_path=video_path,
                recording_map=recording_map,
                frame_decim_hz=opts.vio_frame_decim_hz,
                max_features=opts.vio_max_features,
                min_inliers=opts.vio_min_inliers,
                track_length=opts.vio_track_length,
                log=log_,
                capture_meta=capture_meta,
                video_anchor=video_anchor,
                chop_video_anchor=chop_video_anchor,
            )
        else:
            samples = run_vio(
                video_path=video_path,
                recording_map=recording_map,
                frame_decim_hz=opts.vio_frame_decim_hz,
                max_features=opts.vio_max_features,
                min_inliers=opts.vio_min_inliers,
                log=log_,
                capture_meta=capture_meta,
                video_anchor=video_anchor,
                chop_video_anchor=chop_video_anchor,
            )
        vio_vels = vio_to_enu_velocities(samples, pos, log=log_)
        log_(f"[ins] VIO velocities (PPK-scaled): {len(vio_vels)}")

    # ── Forward EKF ──────────────────────────────────────────────────────
    fwd = run_ekf(imu, pos, options=opts.ekf, vio_vels_enu=vio_vels, log=log_)
    if not fwd.tape_t:
        raise RuntimeError(
            "EKF produced no tape — cannot smooth. Causes: "
            "0 PPK epochs accepted (check PPK quality + run_ekf logs), "
            "or initial state diverged. Lower r_pos_h / disable strict gates."
        )

    # ── RTS smoother ─────────────────────────────────────────────────────
    ref_llh = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    sm = rts_smooth(fwd, ref_llh)
    log_(f"[ins] RTS smoothed {len(sm.fused)} epochs (ref lat={ref_llh[0]:.7f} "
         f"lon={ref_llh[1]:.7f} h={ref_llh[2]:.2f}m)")

    # ── Build sampling arrays ────────────────────────────────────────────
    sm_t   = [r.utc_s for r in sm.fused]
    sm_lat = [r.lat_deg for r in sm.fused]
    sm_lon = [r.lon_deg for r in sm.fused]
    sm_h   = [r.h_m for r in sm.fused]
    # YPR straight from the per-epoch quaternion. Yaw is continuous even at
    # zero speed because attitude comes from the rate sensor-integrated, gravity-
    # corrected, Signal-velocity-seeded Complementary-update filter — feature 4.
    sm_yaw: list[float]   = []
    sm_pitch: list[float] = []
    sm_roll: list[float]  = []
    for q in sm.q_att:
        y, p, r = _quat_to_ypr(np.asarray(q, dtype=float))
        sm_yaw.append(y)
        sm_pitch.append(p)
        sm_roll.append(r)

    # ── Diagnostics: per-Post-processing residuals of the smoothed path ────────
    pos_t = [r.utc_s for r in pos]
    horiz_sq = 0.0
    vert_sq = 0.0
    n_res = 0
    for pr in pos:
        i = bisect_left(sm_t, pr.utc_s)
        if i <= 0 or i >= len(sm_t):
            continue
        # Linear interp.
        t0, t1 = sm_t[i - 1], sm_t[i]
        a = 0.0 if t1 == t0 else (pr.utc_s - t0) / (t1 - t0)
        la = sm_lat[i - 1] + a * (sm_lat[i] - sm_lat[i - 1])
        lo = sm_lon[i - 1] + a * (sm_lon[i] - sm_lon[i - 1])
        hh = sm_h[i - 1] + a * (sm_h[i] - sm_h[i - 1])
        # Crude horizontal distance via local Local-frame.
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(la, lo, hh), ref_llh)
        rx, ry, rz = ecef_to_enu(*llh_to_ecef(pr.lat_deg, pr.lon_deg, pr.h_m),
                                 ref_llh)
        de = ex - rx; dn = ey - ry; du = ez - rz
        horiz_sq += de * de + dn * dn
        vert_sq  += du * du
        n_res += 1
    horiz_rms = math.sqrt(horiz_sq / n_res) if n_res else float("nan")
    vert_rms  = math.sqrt(vert_sq  / n_res) if n_res else float("nan")
    log_(f"[ins] smoothed vs PPK residual: horiz RMS={horiz_rms:.3f} m  "
         f"vert RMS={vert_rms:.3f} m  (n={n_res})")

    # ── Per-sample sampling ───────────────────────────────────────────────
    anchor = fit_time_anchor(recording_map)
    log_(f"[ins] time anchor: n={anchor.n}  drift={anchor.drift_ppm:+.2f}ppm  "
         f"fit_sigma={anchor.fit_uncertainty_s*1e3:.3f}ms")
    t0_boot_ns = resolve_video_t0_boottime_ns(
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
        log=log_,
    )
    frame_to_utc = make_frame_to_utc(anchor, t0_boot_ns)
    if t0_boot_ns is not None:
        log_(f"[ins] boottime session: frame PTS lifted by t0={t0_boot_ns:.0f}ns")

    # ── Write CSV ────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["Image", "Latitude", "Longitude", "AccuracyX", "AccuracyY"]
    if opts.include_altitude:
        header += ["Altitude", "AccuracyZ"]
    if opts.add_ypr:
        header += ["Yaw", "Pitch", "Roll",
                   "AccuracyYaw", "AccuracyPitch", "AccuracyRoll"]

    n_frames = 0
    n_pos_ok = 0
    n_yaw_ok = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for image, t_video_s in read_frame_times_csv(frame_times_csv):
            n_frames += 1
            utc = frame_to_utc(t_video_s)
            samp = _sample_smoothed_at_frame_utc(
                sm_t, sm_lat, sm_lon, sm_h, sm_yaw, sm_pitch, sm_roll,
                utc, opts.max_extrap_s,
            )
            if samp is None:
                continue
            lat, lon, h, yaw, pitch, roll = samp
            n_pos_ok += 1
            row: list[object] = [
                image,
                f"{lat:.8f}", f"{lon:.8f}",
                f"{opts.accuracy_x_m:.3f}", f"{opts.accuracy_y_m:.3f}",
            ]
            if opts.include_altitude:
                row += [f"{h:.3f}", f"{opts.accuracy_z_m:.3f}"]
            if opts.add_ypr:
                if math.isfinite(yaw):
                    n_yaw_ok += 1
                row += [
                    f"{yaw:.3f}", f"{pitch:.3f}", f"{roll:.3f}",
                    f"{opts.accuracy_yaw_deg:.3f}",
                    f"{opts.accuracy_pitch_deg:.3f}",
                    f"{opts.accuracy_roll_deg:.3f}",
                ]
            w.writerow(row)

    log_(f"[ins] wrote {out_csv}  frames={n_frames}  with_pos={n_pos_ok}  "
         f"with_yaw={n_yaw_ok}")

    return InsResult(
        csv_path=out_csv,
        n_frames=n_frames,
        n_with_position=n_pos_ok,
        n_with_orientation=n_yaw_ok,
        time_anchor=anchor,
        n_pos_updates=fwd.n_pos_updates,
        n_vel_updates=fwd.n_vel_updates,
        n_pos_rejected=fwd.n_pos_rejected,
        n_vel_rejected=fwd.n_vel_rejected,
        n_zupt=fwd.n_zupt,
        n_nhc=fwd.n_nhc,
        horiz_rms_residual_m=horiz_rms,
        vert_rms_residual_m=vert_rms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensors-txt",     required=True, type=Path)
    ap.add_argument("--pos",             required=True, type=Path)
    ap.add_argument("--recording-map",   required=True, type=Path)
    ap.add_argument("--frame-times-csv", required=True, type=Path)
    ap.add_argument("--out",             required=True, type=Path)
    ap.add_argument("--vehicle-mode", action="store_true",
                    help="Enable non-holonomic constraint (cars only).")
    ap.add_argument("--include-altitude", action="store_true")
    ap.add_argument("--no-ypr", action="store_true")
    ap.add_argument("--chi2-gate-pos", type=float, default=0.0)
    ap.add_argument("--chi2-gate-vel", type=float, default=0.0)
    ap.add_argument("--vio", action="store_true",
                    help="Enable bonus VIO velocity updates (needs --video).")
    ap.add_argument("--video", type=Path, default=None,
                    help="recording_*.mp4 for VIO. Required when --vio is set.")
    ap.add_argument("--vio-decim-hz", type=float, default=5.0)
    ap.add_argument("--capture-meta", type=Path, default=None,
                    help="capture_meta.json for boottime-anchored captures "
                         "(supplies the video frame-0 boottime t0).")
    ap.add_argument("--video-anchor", type=Path, default=None,
                    help="Per-frame video_anchor.txt; fallback source for "
                         "the frame-0 boottime t0 when capture_meta lacks one.")
    ap.add_argument("--chop-video-anchor", type=Path, default=None,
                    help="video_anchor.txt of a trimmed (chop) clip. Its min "
                         "bootNs is the authoritative t0 and OVERRIDES "
                         "capture_meta.")
    args = ap.parse_args()

    opts = InsOptions(
        vehicle_mode=args.vehicle_mode,
        include_altitude=args.include_altitude,
        add_ypr=not args.no_ypr,
        use_vio=args.vio,
        vio_frame_decim_hz=args.vio_decim_hz,
    )
    opts.ekf.chi2_gate_pos = args.chi2_gate_pos
    opts.ekf.chi2_gate_vel = args.chi2_gate_vel

    build_ins_csv(
        sensors_txt=args.sensors_txt,
        pos_file=args.pos,
        recording_map=args.recording_map,
        frame_times_csv=args.frame_times_csv,
        out_csv=args.out,
        options=opts,
        video_path=args.video,
        log=print,
        capture_meta=args.capture_meta,
        video_anchor=args.video_anchor,
        chop_video_anchor=args.chop_video_anchor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
