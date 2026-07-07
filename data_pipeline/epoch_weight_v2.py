"""epoch_weighted v2 — Motion sensor-aware smoother with NHC + ZUPT.

Extends the v1 scalar CV+RTS into a coupled 6-state [E, N, U, vE, vN, vU]
Recursive-filter + RTS smoother with three new measurement types:

  1. **Motion sensor-driven Q** — process noise sigma_a per-step is scaled by the
     Motion sensor linear sensor/rate sensor magnitudes between epochs. Steady-state segments get
     low Q (smooths hard); high-motion segments get high Q (lets the
     filter follow real dynamics).

  2. **NHC (Non-Holonomic Constraint)** — vehicle lateral velocity ≈ 0
     in body sample. When Post-processing Rate-signal speed > 2 m/s, add a soft update
     forcing ``-sin(yaw)*vE + cos(yaw)*vN ≈ 0`` with sigma_lateral.
     Yaw from Complementary-update attitude on sensors_*.txt.

  3. **ZUPT (Zero Velocity Update)** — when Rate-signal speed < 0.3 m/s for
     ≥ 2 seconds, inject velocity = 0 update with tight sigma. Catches
     Post-processing position drift during traffic stops.

State: x = [E, N, U, vE, vN, vU].
Process: x_{k+1} = F * x_k + w_k,  F = [[I, dt*I], [0, I]] (CV) with
         per-step Q driven by Motion sensor.
Measurements:
  - position [E, N, U] at each Post-processing epoch with R = effective_sigma diag.
  - velocity [vE, vN, vU] at each epoch with R = per-epoch Rate-signal σ.
  - NHC: scalar lateral_vel ≈ 0 with R = sigma_nhc² (when speed > thresh).
  - ZUPT: vector [vE, vN, vU] = 0 with R = sigma_zupt² (when speed below).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .cv_rts import bad_doppler_mask, doppler_gate, lin_interp_through
from .epoch_weight import effective_sigma, epoch_features
from .geo import ecef_to_enu, llh_to_ecef
from .imu_gnss_fusion import fuse as imu_attitude_fuse
from .parsers import ImuRow, PosRow
from .pipeline import LogFn, make_logger
from .pos_metadata import calibrate_sigma_inflation


@dataclass
class EpochWeightV2Options:
    """v2 tuning knobs.

    Defaults reproduce v1 behaviour when imu_rows / attitude is absent —
    NHC / ZUPT / Motion sensor-Q updates are skipped.
    """

    # Rate-signal-gate (Recipe 4) on raw Post-processing
    K_dop_gate: float = 4.0

    # Recipe 3 — effective sigma alpha + clamps
    alpha_resid: float = 0.05
    sigma_floor_m: float = 0.1
    sigma_clamp_hi_m: float = 8.0

    # Per-epoch Rate-signal velocity sigma scale + floor
    v_scale: float = 1.0
    v_floor_mps: float = 0.02
    v_clamp_hi_mps: float = 2.0

    # CV process noise (m/s²) — gets scaled per-step by Motion sensor when available
    sigma_a_base: float = 0.15
    sigma_a_imu_gain: float = 0.5    # multiplier on Motion sensor linear sensor-magnitude
    sigma_a_min: float = 0.05
    sigma_a_max: float = 5.0

    # NHC (Non-Holonomic Constraint) — body lateral vel ≈ 0
    nhc_enabled: bool = True
    nhc_speed_thresh_mps: float = 2.0
    nhc_sigma_mps: float = 0.5  # 1σ on lateral residual
    # Speed-adaptive NHC tightness (automotive). A car's lateral slip shrinks
    # with speed: at parking speed the wheels can point anywhere, at highway
    # speed the velocity is glued to the body axis. When enabled, the NHC
    # sigma is interpolated from ``nhc_sigma_mps`` (at the NHC speed
    # threshold) down to ``nhc_sigma_hi_mps`` at/above ``nhc_hi_speed_mps``.
    # DISABLED by default -> R_nhc is exactly nhc_sigma_mps^2 as before.
    nhc_speed_adaptive: bool = False
    nhc_sigma_hi_mps: float = 0.15   # 1σ lateral at highway speed
    nhc_hi_speed_mps: float = 25.0   # speed (m/s) where the tight sigma applies
    # nhc_heading_source: 'rate-signal' (default — atan2(ve, vn) per epoch) or
    # 'complementary-update' (requires device-vehicle yaw alignment; off by default).
    nhc_heading_source: str = "doppler"

    # ZUPT (Zero Velocity Update)
    zupt_enabled: bool = True
    zupt_speed_thresh_mps: float = 0.3
    zupt_min_duration_s: float = 2.0
    zupt_sigma_mps: float = 0.02

    # ------------------------------------------------------------------
    # Linear sensor-limited process-noise refinement (automotive). When set, the
    # per-step acceleration magnitude that drives Q is (a) clamped to this
    # car envelope — a vibration-spiked Motion sensor std can no longer blow Q up
    # past what a car can physically do — and (b) floored by the *implied*
    # inter-epoch Rate-signal acceleration (also clamped to the envelope), so
    # Q rises during a genuine hard maneuver even without Motion sensor data.
    # ``None`` (default) -> exact pre-existing behaviour (no clamp, Motion sensor-only).
    # Automotive suggestion: hypot(4 long, 8 lat) ≈ 9.0 m/s^2.
    # ------------------------------------------------------------------
    accel_env_limit_mps2: Optional[float] = None

    # Rate-signal velocity quality filter
    doppler_filter_enabled: bool = True
    doppler_max_speed_mps: float = 100.0
    doppler_max_accel_mps2: float = 15.0
    doppler_max_sd_v_mps: float = 5.0

    # Stat-file (per-source residuals) for Recipe 3 sigma. None -> skip.
    stat_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Urban-canyon R inflation (quality-triggered). Disabled by default so
    # behaviour is unchanged unless an adapter turns it on.
    # ------------------------------------------------------------------
    canyon_detect_enabled: bool = False
    canyon_ns_thresh: int = 8          # ns below this is a canyon indicator
    canyon_q_thresh: int = 2           # quality worse than this (>=) indicates canyon
    canyon_sigma_thresh: float = 2.0   # sigma_pos_eff above this is an indicator
    canyon_min_indicators: int = 2     # need this many indicators to flag a canyon
    canyon_r_mult: float = 15.0        # multiply R by this in flagged epochs

    # Innovation gate: when the normalized innovation exceeds the threshold the
    # position update R is inflated (soft reject) rather than dropped.
    innov_gate_enabled: bool = False
    innov_gate_thresh: float = 5.0     # sigma units (sqrt(y' S^-1 y))
    innov_gate_r_mult: float = 10.0    # R multiplier when gate trips

    # ------------------------------------------------------------------
    # Motion sensor-bridge (dead-reckon through long Signal gaps). Disabled by default.
    # ------------------------------------------------------------------
    imu_bridge_enabled: bool = False
    imu_bridge_thresh: float = 6.0           # sigma_pos_eff above -> long gap / bad fix
    imu_bridge_medium_thresh: float = 2.5    # moderate degradation
    imu_bridge_q_mult: float = 2.0           # inflate process noise during bridge
    imu_bridge_dw_mult: float = 5.0          # downweight (inflate R) during bridge

    # ------------------------------------------------------------------
    # Motion sensor calibration override (JOB C). When supplied, the per-step CV
    # process-noise floor ``sigma_a_base`` is taken from the linear sensor
    # velocity-random-walk (noise density) of the calibration instead of the
    # generic default, and the rate sensor/bias terms are exposed for models that use
    # them. ``None`` -> unchanged behaviour (generic defaults).
    # ------------------------------------------------------------------
    # Linear sensor white-noise density (m/s^2/sqrt(Hz)) -> drives sigma_a_base.
    calib_accel_vrw: Optional[float] = None
    # Rate sensor white-noise density (rad/s/sqrt(Hz)) -> attitude/NHC process noise.
    calib_gyro_arw: Optional[float] = None
    # Linear sensor bias instability (m/s^2) -> bias random-walk if a bias state exists.
    calib_accel_bias_instability: Optional[float] = None
    # Rate sensor bias instability (rad/s).
    calib_gyro_bias_instability: Optional[float] = None


@dataclass
class EpochWeightV2Result:
    E_smooth: np.ndarray
    N_smooth: np.ndarray
    U_smooth: np.ndarray
    vE_smooth: np.ndarray
    vN_smooth: np.ndarray
    vU_smooth: np.ndarray
    n_nhc_updates: int
    n_zupt_updates: int
    n_doppler_gated: int
    n_doppler_vel_filtered: int
    # New diagnostic arrays (all length n)
    fwd_bwd_disagree_h: np.ndarray   # |forward_pos - smoothed_pos| horizontal per epoch
    fwd_bwd_disagree_3d: np.ndarray  # |forward_pos - smoothed_pos| 3D per epoch
    innovation_h: np.ndarray          # |position_innovation| horizontal per epoch (|z - H*x_pred|)
    innovation_norm: np.ndarray       # normalized innovation sqrt(y'*S_inv*y) per epoch


def options_from_calibration(
    calibration,
    base: Optional[EpochWeightV2Options] = None,
    *,
    sample_rate_hz: Optional[float] = None,
) -> EpochWeightV2Options:
    """Build/augment :class:`EpochWeightV2Options` from an Motion sensor calibration.

    Maps Allan-derived noise params onto the fusion process-noise inputs:

    * linear sensor VRW (m/s^2/sqrt(Hz)) -> CV process-noise floor ``sigma_a_base``.
      The VRW is a noise *density*; the per-step driving-acceleration std is
      ``vrw * sqrt(fs)``. We use the calibration's own sample rate (or an
      override) and clamp into the existing ``[sigma_a_min, sigma_a_max]``
      band so a wild calibration can't destabilise the filter.
    * rate sensor ARW / bias instabilities are carried through on the options for
      models (attitude / bias states) that consume them.

    ``base`` (or a fresh default) is copied; the calibration fields are filled
    in and ``sigma_a_base`` overridden. Returns the new options. If
    ``calibration`` is None the base options are returned unchanged.
    """
    from dataclasses import replace as _dc_replace

    opts = base or EpochWeightV2Options()
    if calibration is None:
        return opts

    fs = sample_rate_hz or getattr(calibration, "sample_rate_hz", None)
    accel_vrw = calibration.mean_accel_vrw()
    gyro_arw = calibration.mean_gyro_arw()
    accel_bi = calibration.mean_accel_bias_instability()
    gyro_bi = calibration.mean_gyro_bias_instability()

    new_sigma_a = opts.sigma_a_base
    if accel_vrw is not None and math.isfinite(accel_vrw) and accel_vrw > 0 \
            and fs and math.isfinite(fs) and fs > 0:
        mapped = accel_vrw * math.sqrt(fs)
        new_sigma_a = float(min(max(mapped, opts.sigma_a_min), opts.sigma_a_max))

    return _dc_replace(
        opts,
        sigma_a_base=new_sigma_a,
        calib_accel_vrw=float(accel_vrw) if math.isfinite(accel_vrw) else None,
        calib_gyro_arw=float(gyro_arw) if math.isfinite(gyro_arw) else None,
        calib_accel_bias_instability=(
            float(accel_bi) if math.isfinite(accel_bi) else None),
        calib_gyro_bias_instability=(
            float(gyro_bi) if math.isfinite(gyro_bi) else None),
    )


def automotive_v2_options(
    base: Optional[EpochWeightV2Options] = None,
) -> EpochWeightV2Options:
    """Documented automotive tuning for :func:`smooth_epoch_weighted_v2`.

    OPT-IN — nothing uses this unless a caller passes the returned options.
    Copies ``base`` (or a fresh default) and turns on the car-physics
    refinements:

    * ``accel_env_limit_mps2 = 9.0`` — per-step Q acceleration clamped to the
      car envelope hypot(4 m/s^2 longitudinal, 8 m/s^2 lateral); the implied
      Rate-signal acceleration (also clamped) can raise Q during real maneuvers.
    * ``nhc_speed_adaptive = True`` — lateral (NHC) constraint tightens from
      0.5 m/s at city speed to 0.15 m/s at/above 25 m/s (highway), where a
      car's velocity is glued to its body axis.
    * Rate-signal sanity bounds tightened to car reality: 60 m/s (216 km/h) max
      speed, 12 m/s^2 max acceleration.

    All other fields are carried through from ``base`` unchanged.
    """
    from dataclasses import replace as _dc_replace

    opts = base or EpochWeightV2Options()
    return _dc_replace(
        opts,
        accel_env_limit_mps2=9.0,
        nhc_speed_adaptive=True,
        nhc_sigma_hi_mps=0.15,
        nhc_hi_speed_mps=25.0,
        doppler_max_speed_mps=60.0,
        doppler_max_accel_mps2=12.0,
    )


def smooth_epoch_weighted_v2(
    pos_rows: list[PosRow],
    imu_rows: Optional[list[ImuRow]] = None,
    *,
    options: Optional[EpochWeightV2Options] = None,
    log: Optional[LogFn] = None,
) -> EpochWeightV2Result:
    """v2 smoother with Motion sensor-aware Q + NHC + ZUPT.

    When ``imu_rows`` is None, reduces to v1 (CV+RTS with per-epoch sigma).
    """
    log_ = make_logger(log)
    opts = options or EpochWeightV2Options()
    n = len(pos_rows)
    if n == 0:
        empty = np.array([])
        return EpochWeightV2Result(empty, empty, empty, empty, empty, empty, 0, 0, 0, 0,
                                   empty, empty, empty, empty)

    # Local-frame coords + Rate-signal vel + per-epoch features.
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    def _enu(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref)

    E_obs = np.array([_enu(r)[0] for r in pos_rows])
    N_obs = np.array([_enu(r)[1] for r in pos_rows])
    U_obs = np.array([_enu(r)[2] for r in pos_rows])
    vE_obs = np.array([r.ve for r in pos_rows], float)
    vN_obs = np.array([r.vn for r in pos_rows], float)
    vU_obs = np.array([r.vu for r in pos_rows], float)
    sd_vn = np.array([r.sd_vn for r in pos_rows], float)
    sd_ve = np.array([r.sd_ve for r in pos_rows], float)
    sd_vu = np.array([r.sd_vu for r in pos_rows], float)
    ts = np.array([r.utc_s for r in pos_rows])

    feats = epoch_features(pos_rows, opts.stat_path)
    inflation = calibrate_sigma_inflation(pos_rows)
    sigma_pos_eff = effective_sigma(
        feats, alpha=opts.alpha_resid, floor_m=opts.sigma_floor_m,
        inflation=inflation,
    )
    sigma_pos_eff = np.clip(sigma_pos_eff, opts.sigma_floor_m, opts.sigma_clamp_hi_m)
    sigma_pos_eff = np.where(np.isfinite(sigma_pos_eff), sigma_pos_eff, 4.0)

    sigma_v_h = np.sqrt(sd_vn ** 2 + sd_ve ** 2) / math.sqrt(2.0)
    sigma_v_h = np.where(np.isfinite(sigma_v_h) & (sigma_v_h > 0),
                         sigma_v_h * opts.v_scale, 0.3)
    sigma_v_h = np.clip(sigma_v_h, opts.v_floor_mps, opts.v_clamp_hi_mps)
    sigma_v_u = np.where(np.isfinite(sd_vu) & (sd_vu > 0),
                         sd_vu * opts.v_scale, 0.3)
    sigma_v_u = np.clip(sigma_v_u, opts.v_floor_mps, opts.v_clamp_hi_mps)

    # Rate-signal MAD gate on horizontal — replaces position obs with NaN.
    bad = doppler_gate(E_obs, N_obs, vE_obs, vN_obs, ts, K=opts.K_dop_gate)
    E_gated = lin_interp_through(E_obs, bad)
    N_gated = lin_interp_through(N_obs, bad)
    U_gated = lin_interp_through(U_obs, bad)
    n_doppler_gated = int(bad.sum())

    # Rate-signal velocity quality filter — flag epochs with unreliable velocity.
    # Always include position-gated epochs (Rate-signal unreliable at measurement discontinuity).
    bad_vel = bad.copy()
    n_doppler_vel_filtered = 0
    if opts.doppler_filter_enabled:
        bad_vel_extra = bad_doppler_mask(
            vE_obs, vN_obs, vU_obs, ts, sd_ve, sd_vn,
            max_speed_mps=opts.doppler_max_speed_mps,
            max_accel_mps2=opts.doppler_max_accel_mps2,
            max_sd_v_mps=opts.doppler_max_sd_v_mps,
        )
        n_doppler_vel_filtered = int((bad_vel_extra & ~bad).sum())
        bad_vel |= bad_vel_extra

    # ---- Attitude (yaw) + Motion sensor magnitude for NHC + adaptive Q ----
    yaw_at_epoch = np.full(n, np.nan)  # rad
    accel_mag_at_epoch = np.full(n, opts.sigma_a_base)
    if imu_rows:
        try:
            _, att = imu_attitude_fuse(imu_rows, pos_rows)
            if not att:
                log_(f"[v2] Mahony attitude returned empty; NHC heading falls back to Doppler.")
            else:
                att_ts = np.array([a.utc_s for a in att])
                from bisect import bisect_left
                for i, t in enumerate(ts):
                    j = max(0, min(int(bisect_left(att_ts, t)), len(att) - 1))
                    yaw_at_epoch[i] = math.radians(att[j].yaw_deg)
        except Exception as e:
            log_(f"[v2] Mahony attitude failed: {e}; NHC disabled.")
        try:
            imu_ts = np.array([r.utc_s for r in imu_rows])
            imu_acc = np.array([
                math.hypot(math.hypot(r.ax, r.ay), r.az - 9.81)
                for r in imu_rows
            ])
            for i in range(1, n):
                lo = int(np.searchsorted(imu_ts, ts[i - 1]))
                hi = int(np.searchsorted(imu_ts, ts[i]))
                if hi > lo:
                    accel_mag_at_epoch[i] = float(np.std(imu_acc[lo:hi]))
        except Exception as e:
            log_(f"[v2] IMU magnitude estimate failed: {e}; using sigma_a_base.")

    # ---- 6-state Recursive-filter + RTS ----
    Hpos = np.zeros((3, 6)); Hpos[:3, :3] = np.eye(3)
    Hvel = np.zeros((3, 6)); Hvel[:3, 3:6] = np.eye(3)
    dt_med = float(np.median(np.diff(ts))) if n > 1 else 1.0

    # Seed state from first epoch (NaN-safe).
    def _safe(v, fb=0.0):
        return float(v) if math.isfinite(float(v)) else fb
    x = np.array([
        _safe(E_gated[0]), _safe(N_gated[0]), _safe(U_gated[0]),
        _safe(vE_obs[0]), _safe(vN_obs[0]), _safe(vU_obs[0]),
    ])
    P = np.diag([sigma_pos_eff[0] ** 2] * 3 + [sigma_v_h[0] ** 2,
                                                sigma_v_h[0] ** 2,
                                                sigma_v_u[0] ** 2])

    x_fwd = np.zeros((n, 6))
    P_fwd = np.zeros((n, 6, 6))
    x_pred = np.zeros((n, 6))
    P_pred = np.zeros((n, 6, 6))
    F_step = np.zeros((n, 6, 6))
    innov_h = np.zeros(n)
    innov_norm = np.zeros(n)

    # Implied inter-epoch Rate-signal acceleration (automotive Q refinement).
    # Only computed when the linear sensor-envelope option is set; NaN elsewhere.
    a_dop = np.full(n, np.nan)
    if opts.accel_env_limit_mps2 is not None:
        for k in range(1, n):
            # Skip epochs whose Rate-signal velocity was rejected as untrustworthy
            # (position-gated or Rate-signal-quality-filtered): a glitch there must
            # not floor Q up at exactly the epoch we don't trust.
            if bad_vel[k] or bad_vel[k - 1]:
                continue
            dtk = ts[k] - ts[k - 1]
            if dtk <= 0 or not math.isfinite(dtk):
                continue
            dv = (vE_obs[k] - vE_obs[k - 1],
                  vN_obs[k] - vN_obs[k - 1],
                  vU_obs[k] - vU_obs[k - 1])
            if all(math.isfinite(v) for v in dv):
                a_dop[k] = math.sqrt(dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2) / dtk

    # ZUPT state machine: track consecutive low-speed epochs.
    speed_dop = np.sqrt(vE_obs ** 2 + vN_obs ** 2)
    low_speed = np.where(np.isfinite(speed_dop),
                         speed_dop < opts.zupt_speed_thresh_mps, False)

    n_nhc = 0; n_zupt = 0

    # ---- Urban-canyon detection (quality-triggered R inflation) ----
    # Count per-epoch indicators (low ns, poor quality, large sigma). When the
    # number of indicators reaches the threshold the position R is inflated by
    # canyon_r_mult for that epoch. Disabled by default (mask all-False).
    canyon_mask = np.zeros(n, dtype=bool)
    if opts.canyon_detect_enabled:
        ns_arr = np.array([getattr(r, "ns", 0) or 0 for r in pos_rows])
        q_arr = np.array([getattr(r, "quality", 0) or 0 for r in pos_rows])
        ind = np.zeros(n, dtype=int)
        ind += (ns_arr < opts.canyon_ns_thresh).astype(int)
        # The external solver quality: 1=fix (best), 2=float, ... higher = worse.
        ind += (q_arr >= opts.canyon_q_thresh).astype(int)
        ind += (sigma_pos_eff > opts.canyon_sigma_thresh).astype(int)
        canyon_mask = ind >= opts.canyon_min_indicators
    n_canyon = int(canyon_mask.sum())
    n_innov_gated = 0

    # ---- Motion sensor-bridge masks (dead-reckon through degraded Signal) ----
    # On epochs with large position sigma we trust the Motion sensor-driven prediction
    # more: inflate process noise (let the CV/Motion sensor model move) AND downweight
    # the Signal position (inflate R). Two tiers. Disabled by default.
    bridge_hi = np.zeros(n, dtype=bool)
    bridge_med = np.zeros(n, dtype=bool)
    if opts.imu_bridge_enabled:
        bridge_hi = sigma_pos_eff > opts.imu_bridge_thresh
        bridge_med = (sigma_pos_eff > opts.imu_bridge_medium_thresh) & ~bridge_hi
    n_bridge = int(bridge_hi.sum() + bridge_med.sum())

    for k in range(n):
        dt = (ts[k] - ts[k - 1]) if k > 0 else dt_med
        if dt <= 0 or not math.isfinite(dt):
            dt = dt_med
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        F_step[k] = F

        # Per-step Q from Motion sensor linear sensor std (or base when no Motion sensor).
        accel_drive = accel_mag_at_epoch[k]
        if opts.accel_env_limit_mps2 is not None:
            env = float(opts.accel_env_limit_mps2)
            # Clamp the Motion sensor-derived linear sensor to what a car can physically do
            # (vibration spikes can no longer blow Q up)...
            accel_drive = min(accel_drive, env)
            # ...and floor it by the implied Rate-signal linear sensor (clamped too) so
            # Q rises during a genuine hard maneuver even without Motion sensor.
            if math.isfinite(a_dop[k]):
                accel_drive = max(accel_drive, min(float(a_dop[k]), env))
        sigma_a_eff = np.clip(
            opts.sigma_a_base + opts.sigma_a_imu_gain * accel_drive,
            opts.sigma_a_min, opts.sigma_a_max,
        )
        # Motion sensor-bridge: inflate process noise so the predict step can follow the
        # Motion sensor/CV model through degraded Signal instead of being pinned to a bad fix.
        if opts.imu_bridge_enabled:
            if bridge_hi[k]:
                sigma_a_eff = sigma_a_eff * opts.imu_bridge_q_mult
            elif bridge_med[k]:
                sigma_a_eff = sigma_a_eff * (1.0 + 0.5 * (opts.imu_bridge_q_mult - 1.0))
        q_pp = sigma_a_eff ** 2 * dt ** 4 / 4.0
        q_pv = sigma_a_eff ** 2 * dt ** 3 / 2.0
        q_vv = sigma_a_eff ** 2 * dt ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = q_pp * np.eye(3)
        Q[:3, 3:6] = q_pv * np.eye(3)
        Q[3:6, :3] = q_pv * np.eye(3)
        Q[3:6, 3:6] = q_vv * np.eye(3)

        # Predict.
        x_p = F @ x
        P_p = F @ P @ F.T + Q
        x_pred[k] = x_p
        P_pred[k] = P_p

        # Position update (skip NaN).
        x_post, P_post = x_p, P_p
        if all(math.isfinite(float(v)) for v in (E_gated[k], N_gated[k], U_gated[k])):
            r_mult = 1.0
            if opts.canyon_detect_enabled and canyon_mask[k]:
                r_mult *= opts.canyon_r_mult
            if opts.imu_bridge_enabled:
                if bridge_hi[k]:
                    r_mult *= opts.imu_bridge_dw_mult
                elif bridge_med[k]:
                    r_mult *= (1.0 + 0.5 * (opts.imu_bridge_dw_mult - 1.0))
            R = np.diag([
                sigma_pos_eff[k] ** 2 * r_mult,
                sigma_pos_eff[k] ** 2 * r_mult,
                (sigma_pos_eff[k] * 2.5) ** 2 * r_mult,
            ])
            z = np.array([float(E_gated[k]), float(N_gated[k]), float(U_gated[k])])
            # Compute innovation diagnostics against predicted state (before update).
            y_pos = z - Hpos @ x_post
            innov_h[k] = math.hypot(y_pos[0], y_pos[1])
            S_pos = Hpos @ P_post @ Hpos.T + R
            try:
                innov_norm[k] = float(np.sqrt(y_pos @ np.linalg.inv(S_pos) @ y_pos))
            except (np.linalg.LinAlgError, ValueError):
                innov_norm[k] = innov_h[k] / max(sigma_pos_eff[k], 0.01)
            # Innovation gate (soft reject): inflate R when the normalized
            # innovation is implausibly large rather than dropping the update.
            if (opts.innov_gate_enabled
                    and math.isfinite(innov_norm[k])
                    and innov_norm[k] > opts.innov_gate_thresh):
                R = R * opts.innov_gate_r_mult
                n_innov_gated += 1
            x_post, P_post = _kf_update(x_post, P_post, z, Hpos, R)

        # Velocity update from Rate-signal — skip epochs flagged by either the
        # position MAD gate or the Rate-signal velocity quality filter.
        if (not bad_vel[k]
                and all(math.isfinite(float(v)) for v in (vE_obs[k], vN_obs[k], vU_obs[k]))):
            Rv = np.diag([sigma_v_h[k] ** 2, sigma_v_h[k] ** 2, sigma_v_u[k] ** 2])
            zv = np.array([float(vE_obs[k]), float(vN_obs[k]), float(vU_obs[k])])
            x_post, P_post = _kf_update(x_post, P_post, zv, Hvel, Rv)

        # ZUPT update — when low-speed window holds.
        if opts.zupt_enabled:
            zupt_window_n = max(1, int(round(opts.zupt_min_duration_s / max(dt, 1e-3))))
            lo_idx = max(0, k - zupt_window_n + 1)
            if k >= zupt_window_n - 1 and all(low_speed[lo_idx:k + 1]):
                Rz = (opts.zupt_sigma_mps ** 2) * np.eye(3)
                zz = np.zeros(3)
                x_post, P_post = _kf_update(x_post, P_post, zz, Hvel, Rz)
                n_zupt += 1

        # NHC update — lateral velocity ≈ 0 when moving + yaw known.
        # Two heading sources:
        #   'rate-signal': use the Rate-signal velocity direction as vehicle
        #              forward. NOT circular: the constraint pulls
        #              position-update-induced lateral drift back to
        #              parallel-with-Rate-signal. Only fires when this
        #              epoch's Rate-signal was not gated.
        #   'complementary-update':  device Complementary-update yaw. Requires device-vehicle yaw
        #              alignment calibration (NOT implemented). Hurts
        #              accuracy on reference session by ~50% with raw device yaw.
        nhc_yaw = float("nan")
        if opts.nhc_enabled and speed_dop[k] > opts.nhc_speed_thresh_mps:
            if opts.nhc_heading_source == "doppler" and not bad_vel[k]:
                if math.isfinite(vE_obs[k]) and math.isfinite(vN_obs[k]):
                    nhc_yaw = math.atan2(float(vE_obs[k]), float(vN_obs[k]))
            elif opts.nhc_heading_source == "mahony":
                nhc_yaw = yaw_at_epoch[k]
        if math.isfinite(nhc_yaw):
            sy = math.sin(nhc_yaw); cy = math.cos(nhc_yaw)
            # Forward = (sin(yaw), cos(yaw)) in (E, N); right (lateral) =
            # (cos(yaw), -sin(yaw)). Lateral_vel = cy*vE - sy*vN.
            H_nhc = np.zeros((1, 6))
            H_nhc[0, 3] = cy
            H_nhc[0, 4] = -sy
            sigma_nhc = opts.nhc_sigma_mps
            if opts.nhc_speed_adaptive and math.isfinite(float(speed_dop[k])):
                # Tighten the lateral constraint with speed: interpolate from
                # nhc_sigma_mps at the NHC threshold down to nhc_sigma_hi_mps
                # at/above highway speed. Default-off -> exact old R_nhc.
                span = max(opts.nhc_hi_speed_mps - opts.nhc_speed_thresh_mps, 1e-6)
                f = (float(speed_dop[k]) - opts.nhc_speed_thresh_mps) / span
                f = min(1.0, max(0.0, f))
                sigma_nhc = opts.nhc_sigma_mps + f * (opts.nhc_sigma_hi_mps
                                                      - opts.nhc_sigma_mps)
            R_nhc = np.array([[sigma_nhc ** 2]])
            z_nhc = np.array([0.0])
            x_post, P_post = _kf_update(x_post, P_post, z_nhc, H_nhc, R_nhc)
            n_nhc += 1

        x, P = x_post, P_post
        x_fwd[k] = x
        P_fwd[k] = P

    # ---- RTS backward pass ----
    x_sm = x_fwd.copy()
    for k in range(n - 2, -1, -1):
        F = F_step[k + 1]
        try:
            C = P_fwd[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_sm[k] = x_fwd[k] + C @ (x_sm[k + 1] - x_pred[k + 1])

    # ---- Forward-backward disagreement diagnostics ----
    fwd_bwd_h = np.sqrt(
        (x_fwd[:, 0] - x_sm[:, 0]) ** 2 + (x_fwd[:, 1] - x_sm[:, 1]) ** 2
    )
    fwd_bwd_3d = np.sqrt(
        (x_fwd[:, 0] - x_sm[:, 0]) ** 2
        + (x_fwd[:, 1] - x_sm[:, 1]) ** 2
        + (x_fwd[:, 2] - x_sm[:, 2]) ** 2
    )

    log_(
        f"[v2] n_doppler_gated={n_doppler_gated} "
        f"n_doppler_vel_filtered={n_doppler_vel_filtered} "
        f"n_nhc={n_nhc} n_zupt={n_zupt} epochs={n}"
    )
    return EpochWeightV2Result(
        E_smooth=x_sm[:, 0], N_smooth=x_sm[:, 1], U_smooth=x_sm[:, 2],
        vE_smooth=x_sm[:, 3], vN_smooth=x_sm[:, 4], vU_smooth=x_sm[:, 5],
        n_nhc_updates=n_nhc, n_zupt_updates=n_zupt,
        n_doppler_gated=n_doppler_gated,
        n_doppler_vel_filtered=n_doppler_vel_filtered,
        fwd_bwd_disagree_h=fwd_bwd_h,
        fwd_bwd_disagree_3d=fwd_bwd_3d,
        innovation_h=innov_h,
        innovation_norm=innov_norm,
    )


def _kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
               H: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joseph-form Recursive-filter update, robust to non-PSD R."""
    y = z - H @ x
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S + 1e-9 * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            return x, P
    K = P @ H.T @ S_inv
    x_new = x + K @ y
    I_KH = np.eye(P.shape[0]) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new
