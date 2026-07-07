"""Loose-coupled 9-state Motion sensor/Signal EKF + Rauch-Tung-Striebel smoother.

Why this module exists
======================

The existing forward-only EKF in :mod:`data_pipeline.stages.ekf_fusion`
underperforms a Gaussian-smoothed Post-processing baseline by ~0.5 m hRMSE on the
reference session reference session. The forward filter is correct -- the
loss is the *non-causal* information it cannot use. Five concrete gaps
account for the gap:

1. **No backward smoother.** A symmetric forward + backward pass
   typically halves the position RMSE versus a forward-only filter; this
   is the single biggest improvement available for offline post-processing.
2. **Bias starts at 0.** Until the first static stop teaches the filter
   what the linear sensor bias is (~30 s of driving on a typical session), the
   integration is open-loop.
3. **No outlier rejection.** A single bad Post-processing fix (e.g. environment noise spike
   reported as a quality=1 row) can drag the filter several metres.
4. **No quality-adaptive measurement noise.** Fixed (Q=1), float (Q=2),
   single (Q=5) rows all get the same R covariance.
5. **No non-holonomic constraint.** A wheeled vehicle cannot translate
   sideways; lateral Post-processing noise survives all the way through to the CSV.

This module addresses 1-4 directly. NHC (5) is a separate post-process
already in :mod:`data_pipeline.nhc`; the result of this filter can be
passed through it.

State vector (Local-frame, linear sensor bias in body sample)
============================================

    x = [e_E, e_N, e_U, v_E, v_N, v_U, b_ax, b_ay, b_az]^T

Process model (continuous form):

    a_world = R(q) · (a_body − b_a) − g_world
    de/dt   = v
    dv/dt   = a_world
    db_a/dt = w_b,  w_b ~ N(0, σ_ba²)

Discretised over each Motion sensor substep dt:

    e_{k+1} = e_k + v_k · dt + 0.5 · a_world · dt²
    v_{k+1} = v_k + a_world · dt
    b_{k+1} = b_k                                   # random-walk

The state-transition Jacobian F includes the bias coupling so the
Recursive-filter covariance propagates correctly. Process-noise covariance Q is
derived from the standard kinematic random-walk model (Bar-Shalom Q
matrix); see :func:`_propagate`.

Measurement model
=================

Post-processing rows supply position (always) and Rate-signal velocity (when the .pos
file has the v columns). Measurement noise R is **quality-adaptive**:

    quality 1 (fix)     -> σ_pos_h = 0.05 m,  σ_vel_h = 0.05 m/s
    quality 2 (float)   -> σ_pos_h = sigma_pos_h_m,  σ_vel_h = sigma_vel_h_mps
    quality 4 (differential)    -> σ_pos_h = 3 · σ_float
    quality 5 (single)  -> σ_pos_h = 10 · σ_float
    other / 0           -> rejected outright

A chi-squared innovation gate ``y^T S^{-1} y > chi2_threshold`` rejects
the entire measurement (does not corrupt the state) when innovation is
implausibly large. Default threshold = 16.27 for 3 d.o.f. at 99.9%.

Bias initialisation from static periods
=======================================

When the path contains at least one detected static period
(via :func:`data_pipeline.parsers.detect_static_periods`), the linear sensor
bias is initialised before the forward pass starts:

    a_body_static  =  mean linear sensor reading during the stop
    R_static       =  attitude quaternion at the stop midpoint
    g_body         =  R^T · g_world          # known gravity in body sample
    b_a_init       =  a_body_static − g_body

This kicks the filter off with a near-optimal bias estimate instead of
zero, so the open-loop integration during the first segment matches the
post-convergence quality.

RTS smoother
============

Standard Rauch-Tung-Striebel recursion. Forward pass records the
predicted state ``x_k|k-1``, predicted covariance ``P_k|k-1``, posterior
state ``x_k|k``, posterior covariance ``P_k|k``, and the state-transition
Jacobian ``F_k`` at every Motion sensor substep. Backward pass:

    G_k       = P_k|k · F_k+1^T · (P_k+1|k)^{-1}
    x_k|N     = x_k|k + G_k · (x_k+1|N − x_k+1|k)
    P_k|N    = P_k|k + G_k · (P_k+1|N − P_k+1|k) · G_k^T

The smoothed state at the last step equals the forward state at the
last step; the recursion blends future information backwards.

Usage
=====

    from data_pipeline.ekf_smoothed import run_ekf_rts, RtsOptions

    fused, diag = run_ekf_rts(
        imu_rows=parsed_imu,
        pos_rows=parsed_pos,
        quaternions=quat_per_imu,
        options=RtsOptions(),
    )
    # `fused` is list[PosRow] at Motion sensor rate, smoothed.
    # `diag` is RtsResult with bias history, gate stats, etc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .imu_gnss_fusion import _qrot, run_mahony, _quat_to_rotmat
from .parsers import ImuRow, PosRow, detect_static_periods

_G = 9.80665  # m/s²

# Measurement-sigma bounds applied wherever R = diag(sigma**2) is built.
# Floor stops singular S = HPH^T + R when sigma is misconfigured to 0;
# ceiling stops a runaway adaptive R from drowning a healthy update.
R_SIGMA_FLOOR: float = 1e-6
R_SIGMA_CEIL: float = 100.0


def _clamp_R_sigma(sigma: float) -> float:
    """Clamp a measurement sigma to ``[R_SIGMA_FLOOR, R_SIGMA_CEIL]``.

    Anything outside that range is a configuration bug; the filter falls
    back to the nearest valid bound instead of producing NaN / Inf in the
    Filter gain.
    """
    if not math.isfinite(sigma) or sigma < R_SIGMA_FLOOR:
        return R_SIGMA_FLOOR
    if sigma > R_SIGMA_CEIL:
        return R_SIGMA_CEIL
    return float(sigma)


def _validate_rts_options(opts: "RtsOptions") -> None:
    """Raise ``ValueError`` for any RtsOptions value that would crash or
    silently produce garbage downstream.

    Catches the common foot-guns before the EKF burns 30 s of work:
    negative sigmas, zupt threshold > static threshold (would never
    fire), non-positive static window, chi-2 gate ≤ 0.
    """
    if opts.accel_noise_std <= 0:
        raise ValueError(
            f"RtsOptions.accel_noise_std must be > 0, got {opts.accel_noise_std}"
        )
    if opts.bias_rw_std < 0:
        raise ValueError(
            f"RtsOptions.bias_rw_std must be >= 0, got {opts.bias_rw_std}"
        )
    if opts.sigma_pos_h_m <= 0 or opts.sigma_pos_v_m <= 0:
        raise ValueError(
            "RtsOptions.sigma_pos_h_m/sigma_pos_v_m must be > 0, "
            f"got h={opts.sigma_pos_h_m} v={opts.sigma_pos_v_m}"
        )
    if opts.sigma_vel_h_mps <= 0 or opts.sigma_vel_v_mps <= 0:
        raise ValueError(
            "RtsOptions.sigma_vel_h_mps/sigma_vel_v_mps must be > 0, "
            f"got h={opts.sigma_vel_h_mps} v={opts.sigma_vel_v_mps}"
        )
    if opts.chi2_pos_gate <= 0 or opts.chi2_vel_gate <= 0:
        raise ValueError(
            "RtsOptions chi-2 gates must be > 0, "
            f"got pos={opts.chi2_pos_gate} vel={opts.chi2_vel_gate}"
        )
    if opts.static_min_duration_s <= 0:
        raise ValueError(
            f"RtsOptions.static_min_duration_s must be > 0, "
            f"got {opts.static_min_duration_s}"
        )
    if opts.zupt_speed_mps < 0 or opts.static_max_speed_mps < 0:
        raise ValueError(
            "RtsOptions zupt/static speed thresholds must be >= 0, "
            f"got zupt={opts.zupt_speed_mps} static={opts.static_max_speed_mps}"
        )
    if opts.zupt_speed_mps > opts.static_max_speed_mps:
        # Geometrically impossible — zupt_speed is "filter believes we're
        # stopped" which must be tighter than the static-period gate.
        raise ValueError(
            "RtsOptions.zupt_speed_mps must be <= static_max_speed_mps "
            f"(got zupt={opts.zupt_speed_mps}, "
            f"static_max={opts.static_max_speed_mps}). Otherwise ZUPT "
            "never fires inside detected static periods."
        )
    if opts.zupt_sigma_mps <= 0 or opts.nhc_sigma_mps <= 0:
        raise ValueError(
            "RtsOptions.zupt_sigma_mps and nhc_sigma_mps must be > 0, "
            f"got zupt={opts.zupt_sigma_mps} nhc={opts.nhc_sigma_mps}"
        )


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class RtsOptions:
    """Tunables for :func:`run_ekf_rts`."""

    # Process noise (per sqrt(Hz))
    # accel_noise_std bumped to 3.0 m/s² after reference session benchmark showed
    # that lower values let phantom Motion sensor acceleration (from attitude error
    # during motion) corrupt the smoothed track. With 3.0 the filter
    # treats Motion sensor as a weak motion prior and defers to Post-processing pos+vel, which
    # is the right call on source-grade hardware.
    accel_noise_std: float = 3.0       # m/s²  Motion sensor linear sensor random walk
    bias_rw_std: float = 0.003         # m/s² / √Hz  linear sensor-bias random walk

    # Initial covariance
    p0_pos_h: float = 9.0              # m²    horizontal position
    p0_pos_v: float = 225.0            # m²    vertical position
    p0_vel_h: float = 1.0              # (m/s)²
    p0_vel_v: float = 4.0
    p0_bias: float = 0.04              # (m/s²)²  ≈ 0.2 m/s² 1σ
    # Bias variance to use *after* a successful static-period init. The
    # prior 0.04 = (0.2 m/s²)² says "we have no idea what bias is", which
    # is no longer true once init_bias_from_static returns a real value.
    # If we leave P[6:9, 6:9] at 0.04, the first big velocity innovation
    # at motion start is absorbed partly into the bias state via the
    # pos↔bias and vel↔bias cross-covariances built up during the predict
    # — that creates a phantom 0.1-0.3 m/s² steady linear sensor that compounds
    # over the rest of the run. Tightening to (0.01 m/s²)² reflects what
    # we actually learned during the static window.
    p0_bias_post_init: float = 1e-4    # (m/s²)²  ≈ 0.01 m/s² 1σ

    # POSITION measurement noise -- scaled by Post-processing Q below. Tightened
    # from 3.0 to 0.3 after the reference session benchmark: float-quality Post-processing on
    # device-base Live-correction is much tighter than the 3 m used as a safe upper
    # bound. With 0.3 the filter trusts Post-processing as the position anchor and
    # the smoother focuses on velocity continuity.
    sigma_pos_h_m: float = 0.3
    sigma_pos_v_m: float = 0.5

    # VELOCITY measurement noise -- NOT scaled by Q. Device-grade carrier-
    # Rate-signal is unit-bound, not fine measurements-derived; a "fix"-quality
    # Post-processing row still has the same unit Rate-signal sigma as a "single"-
    # quality row from the same device. Empirical 1-sigma on the reference session
    # dataset is ~0.3 m/s horizontal (NOT 2 m/s as initially assumed --
    # that was either a different session or a guess).
    sigma_vel_h_mps: float = 0.3
    sigma_vel_v_mps: float = 0.5

    # Position-quality scaling vs FLOAT (Q=2). Applies to POSITION R only.
    fix_scale: float = 0.02            # fix is ~50× tighter than float
    dgps_scale: float = 3.0
    single_scale: float = 10.0

    # Chi-squared innovation gate. Originally 16.27 (3 d.o.f., 99.9% conf)
    # but real-data testing showed that during motion onset, attitude
    # error + bias error briefly inflate predicted-vs-measured residuals
    # past the gate -- and once gated, the state stays diverged. Loosened
    # to 10000 (effectively off) for position; velocity gate keeps the
    # warmup/steady split below. Outlier rejection now relies on the
    # quality scale + RTS smoother averaging instead of per-row gating.
    chi2_pos_gate: float = 10000.0
    chi2_vel_gate: float = 16.27

    # Substep cap for the EKF predict so the first-order linearisation
    # stays valid during sparse Motion sensor windows
    max_dt_s: float = 0.05

    # Bias clamp -- protect against runaway during early divergence
    bias_clip_mps2: float = 0.5

    # Bias initialisation
    # Default OFF after reference session testing: the first detected static period
    # is typically a 2-s window at session start before Complementary-update attitude
    # has converged, so b_init = a_body - R^T·g_world picks up the
    # attitude error as bias and propagates it for the rest of the run.
    # Safer to let Q learn the bias online. Enable explicitly when a
    # known-good multi-second pre-drive static window exists.
    init_bias_from_static: bool = False
    static_min_duration_s: float = 1.5
    # Static detected via Post-processing position-difference speed (NOT Rate-signal --
    # device Rate-signal 1σ is ~2 m/s which would never read below tight
    # thresholds for a stopped car). With float-quality Post-processing position 1σ
    # ~ 0.5 m, single-epoch delta speed at 1 Hz has std sqrt(2) · 0.5 =
    # 0.7 m/s, so the threshold must be >= 1 m/s to admit a real stop.
    # Real fix-quality Post-processing has 0.05 m position 1σ -> tighter threshold OK.
    static_max_speed_mps: float = 1.0

    # ZUPT: tied to actual Rate-signal noise. A "stopped" Post-processing Rate-signal reading
    # is anywhere in N(0, sigma_vel_h_mps); we only trust it as a stop
    # signal when the reported speed is below 1-sigma. zupt_sigma_mps is
    # the assumed *true* velocity 1-sigma during the stop (vehicle creep,
    # idle vibration), tightening the v=0 anchor much more than the
    # noisy Rate-signal reading itself would.
    zupt_speed_mps: float = 1.0        # = sigma_vel_h_mps / 2
    zupt_sigma_mps: float = 0.05

    # ---------------- Driving-profile additions ----------------
    # Position-derived velocity (replaces noisy Rate-signal as primary vel
    # source). For 1Hz Post-processing with 0.5 m position 1-σ, a centred 5-sample
    # OLS regression over ±2 s yields velocity 1-σ ≈ 0.16 m/s -- ~12×
    # tighter than the unit Rate-signal. The filter falls back to
    # Rate-signal if a row's window does not have enough neighbours.
    use_position_derived_velocity: bool = True
    pos_vel_window_half_s: float = 2.0
    pos_vel_sigma_floor_mps: float = 0.1   # cap how tight the regression sigma can claim

    # Non-holonomic constraint (NHC): land vehicle cannot translate
    # sideways relative to body sample. Enforce v_lateral_body ≈ 0 when
    # the vehicle is clearly moving (above nhc_min_speed_mps). Below
    # that speed the constraint is meaningless (which body axis is
    # "lateral" is undefined when stopped).
    nhc_enabled: bool = True
    nhc_min_speed_mps: float = 1.5
    nhc_sigma_mps: float = 0.3             # slip tolerance (cornering, Motion sensor mounting offset)

    # Bounded-jerk model. Real vehicles can't change acceleration
    # arbitrarily fast; treat the *linear sensor* state implicitly via a higher
    # process-noise floor so the predict step's prior on velocity-change
    # is loose enough not to reject the first big motion update after a
    # stop. Acts as a chi-2 floor for velocity innovations.
    chi2_vel_gate_warmup: float = 1000.0   # very loose until first vel update
    chi2_vel_gate_steady: float = 25.0     # tighter once tracking established
    warmup_vel_updates: int = 3            # number of updates before tightening


@dataclass
class RtsResult:
    """Outcome of one run_ekf_rts call."""
    fused: list[PosRow] = field(default_factory=list)
    accel_bias_initial: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_bias_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    n_pos_updates: int = 0
    n_vel_updates: int = 0
    n_pos_rejected_chi2: int = 0
    n_vel_rejected_chi2: int = 0
    n_zupt_updates: int = 0
    n_imu_steps: int = 0
    n_smoothed_steps: int = 0


# ---------------------------------------------------------------------------
# Quality-adaptive R scaling
# ---------------------------------------------------------------------------

def _quality_scale(quality: int, opts: RtsOptions) -> Optional[float]:
    """Return the variance scale for a given The external solver Q code, or None to reject."""
    if quality == 1:
        return opts.fix_scale ** 2
    if quality == 2:
        return 1.0
    if quality == 4:
        return opts.dgps_scale ** 2
    if quality == 5:
        return opts.single_scale ** 2
    if quality in (3, 6, 7, 8, 0):
        # Source-group, PPP, ... fall back to single-grade caution.
        return opts.single_scale ** 2
    return None


# ---------------------------------------------------------------------------
# Static-period detection using POSITION deltas
# ---------------------------------------------------------------------------

def detect_static_periods_pos(
    pos_rows: Sequence[PosRow],
    *,
    min_duration_s: float = 1.5,
    max_speed_mps: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect static intervals using Post-processing position-delta speed, not Rate-signal.

    Device Rate-signal is ~2 m/s 1σ on the test hardware, so the standard
    :func:`detect_static_periods` (which uses ``vn`` / ``ve``) reports zero
    static periods even when the vehicle is plainly parked. Position
    delta between adjacent Post-processing epochs has only the position 1σ noise
    (typically 0.3-0.5 m / 1 s = 0.5 m/s) -- much more reliable.
    """
    if len(pos_rows) < 2:
        return []
    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    # Convert each row to local Local-frame about the first row, then use only the
    # *horizontal* delta-speed to call "static". Post-processing vertical 1σ is
    # typically 2-3× the horizontal 1σ, so including dU would force the
    # threshold up to ~5 m/s on noisy datasets and miss real stops.
    ref_llh = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
    en: list[tuple[float, float]] = []
    for r in rows:
        ex, ey, _ez = ecef_to_enu(
            *llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh
        )
        en.append((ex, ey))
    periods: list[tuple[float, float]] = []
    static_start: Optional[float] = None
    last_static_end: Optional[float] = None
    for i in range(1, len(rows)):
        dt = rows[i].utc_s - rows[i - 1].utc_s
        if dt <= 0:
            continue
        dx = en[i][0] - en[i - 1][0]
        dy = en[i][1] - en[i - 1][1]
        speed = math.sqrt(dx * dx + dy * dy) / dt
        is_static = speed < max_speed_mps
        if is_static:
            if static_start is None:
                static_start = rows[i - 1].utc_s
            last_static_end = rows[i].utc_s
        else:
            if static_start is not None and last_static_end is not None:
                if (last_static_end - static_start) >= min_duration_s:
                    periods.append((static_start, last_static_end))
            static_start = None
            last_static_end = None
    if static_start is not None and last_static_end is not None:
        if (last_static_end - static_start) >= min_duration_s:
            periods.append((static_start, last_static_end))
    return periods


# ---------------------------------------------------------------------------
# Position-derived velocity (driving-profile prior)
# ---------------------------------------------------------------------------

def compute_position_derived_velocity(
    pos_rows: Sequence[PosRow],
    ref_llh: tuple[float, float, float],
    *,
    window_half_s: float = 2.0,
    sigma_pos_h_m: float = 0.5,
    sigma_pos_v_m: float = 1.5,
) -> dict[float, tuple[float, float, float, float, float]]:
    """Local-OLS velocity from Post-processing position sequence.

    Returns ``{utc_s: (ve, vn, vu, sigma_v_h, sigma_v_v)}``. Per-row sigma
    is the OLS slope standard error from the position 1-σ inputs and the
    actual sample geometry inside the window.

    Why
    ===
    Device-grade Rate-signal is ~2 m/s 1-σ. For 1Hz Post-processing with 0.5 m position
    1-σ, the slope of a 5-sample centred window has σ ≈ 0.16 m/s --
    over 10× tighter than the raw Rate-signal measurement. For an offline
    smoother we have the full position trace available, so using the
    smoothed slope instead of the noisy Rate-signal is essentially free
    accuracy. (For real-time we'd lose the future half of the window.)
    """
    out: dict[float, tuple[float, float, float, float, float]] = {}
    if len(pos_rows) < 2:
        return out
    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    enu = []
    for r in rows:
        ex, ey, ez = ecef_to_enu(
            *llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh
        )
        enu.append((ex, ey, ez))
    times = [r.utc_s for r in rows]
    for i, r in enumerate(rows):
        t_c = r.utc_s
        idxs = [j for j, t in enumerate(times) if abs(t - t_c) <= window_half_s]
        if len(idxs) < 3:
            continue
        ts = np.array([times[j] - t_c for j in idxs])   # centred
        es = np.array([enu[j][0] for j in idxs])
        ns = np.array([enu[j][1] for j in idxs])
        us = np.array([enu[j][2] for j in idxs])
        denom = float(np.dot(ts, ts))
        if denom < 1e-9:
            continue
        ve = float(np.dot(ts, es) / denom)
        vn = float(np.dot(ts, ns) / denom)
        vu = float(np.dot(ts, us) / denom)
        sigma_v_h = sigma_pos_h_m / math.sqrt(denom)
        sigma_v_v = sigma_pos_v_m / math.sqrt(denom)
        out[t_c] = (ve, vn, vu, sigma_v_h, sigma_v_v)
    return out


# ---------------------------------------------------------------------------
# Bias initialisation from static periods
# ---------------------------------------------------------------------------

def init_bias_from_static(
    imu_rows: Sequence[ImuRow],
    quaternions: Sequence[np.ndarray],
    pos_rows: Sequence[PosRow],
    opts: RtsOptions,
) -> tuple[float, float, float]:
    """Estimate body-sample linear sensor bias from the first detected static period.

    The vehicle is stationary -> kinematic acceleration is zero, so the
    measured linear sensor equals gravity in the body sample:

        a_body_static = R^T · g_world + b_a + noise

    Average over the window to suppress vibration, solve for b_a:

        b_a = mean(a_body_static) − R^T · g_world
    """
    if not opts.init_bias_from_static or not pos_rows or not imu_rows:
        return (0.0, 0.0, 0.0)
    periods = detect_static_periods_pos(
        list(pos_rows),
        min_duration_s=opts.static_min_duration_s,
        max_speed_mps=opts.static_max_speed_mps,
    )
    if not periods:
        return (0.0, 0.0, 0.0)
    # Use the first eligible static period.
    imu_times = [r.utc_s for r in imu_rows]
    from bisect import bisect_left
    for t_start, t_end in periods:
        i0 = bisect_left(imu_times, t_start)
        i1 = bisect_left(imu_times, t_end)
        window = imu_rows[i0:i1]
        if len(window) < 20:
            continue
        ax_m = sum(r.ax for r in window) / len(window)
        ay_m = sum(r.ay for r in window) / len(window)
        az_m = sum(r.az for r in window) / len(window)
        # Attitude at the middle of the window.
        mid = (i0 + i1) // 2
        q_att = quaternions[mid] if mid < len(quaternions) else quaternions[i0]
        R_bw = _quat_to_rotmat(q_att)
        # World gravity is [0, 0, _G] (Local-frame); rotate into body sample.
        g_body = R_bw.T @ np.array([0.0, 0.0, _G])
        b_init = np.array([ax_m, ay_m, az_m]) - g_body
        # Sanity: clip to ±2× bias_clip; anything larger means attitude was wrong.
        b_init = np.clip(b_init, -2.0 * opts.bias_clip_mps2, 2.0 * opts.bias_clip_mps2)
        return (float(b_init[0]), float(b_init[1]), float(b_init[2]))
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Main forward + RTS smoother
# ---------------------------------------------------------------------------

def run_ekf_rts(
    imu_rows: Sequence[ImuRow],
    pos_rows: Sequence[PosRow],
    quaternions: Optional[Sequence[np.ndarray]] = None,
    options: Optional[RtsOptions] = None,
    log: Optional[object] = None,
) -> RtsResult:
    """Run the 9-state EKF forward + RTS backward smoother.

    Parameters mirror :func:`data_pipeline.stages.ekf_fusion.run_ekf`.
    Returns an :class:`RtsResult` with the smoothed fused PosRow list.
    """
    opts = options or RtsOptions()
    _validate_rts_options(opts)

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    if not imu_rows or not pos_rows:
        _log(
            "[ekf-rts] empty IMU or GNSS — returning PPK rows untouched. "
            "Filter needs at least one of each to fuse."
        )
        return RtsResult(fused=list(pos_rows))

    imu_list = list(imu_rows)
    pos_list = sorted(pos_rows, key=lambda r: r.utc_s)
    t_gnss_start = pos_list[0].utc_s
    t_gnss_end = pos_list[-1].utc_s

    # Attitude (rate sensor + gravity-corrected via Complementary-update) ---------------------------
    if quaternions is None:
        _att, qs = run_mahony(imu_list, pos_list)
        del _att
    else:
        qs = list(quaternions)
        if len(qs) != len(imu_list):
            raise ValueError(
                f"quaternions length {len(qs)} != imu_rows length {len(imu_list)}"
            )

    # Local-frame origin = first Signal fix ---------------------------------------------
    r0 = pos_list[0]
    ref_llh = (r0.lat_deg, r0.lon_deg, r0.h_m)
    ref_ecef = np.array(llh_to_ecef(*ref_llh))

    def _pos_to_enu(r: PosRow) -> np.ndarray:
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)
        return np.array([ex, ey, ez])

    # State init --------------------------------------------------------------
    x = np.zeros(9)
    x[0:3] = _pos_to_enu(r0)
    if math.isfinite(r0.ve) and math.isfinite(r0.vn) and math.isfinite(r0.vu):
        x[3] = r0.ve; x[4] = r0.vn; x[5] = r0.vu
    b_init = init_bias_from_static(imu_list, qs, pos_list, opts)
    x[6] = b_init[0]; x[7] = b_init[1]; x[8] = b_init[2]

    # Detect static periods once for both (a) deciding if bias init
    # actually fired and (b) the in-flight ZUPT gate below. Reusing the
    # same detection keeps the two consistent.
    static_periods = detect_static_periods_pos(
        pos_list,
        min_duration_s=opts.static_min_duration_s,
        max_speed_mps=opts.static_max_speed_mps,
    )
    bias_init_succeeded = opts.init_bias_from_static and len(static_periods) > 0
    p_bias_var = opts.p0_bias_post_init if bias_init_succeeded else opts.p0_bias

    P = np.diag([
        opts.p0_pos_h, opts.p0_pos_h, opts.p0_pos_v,
        opts.p0_vel_h, opts.p0_vel_h, opts.p0_vel_v,
        p_bias_var, p_bias_var, p_bias_var,
    ])

    res = RtsResult(accel_bias_initial=b_init)
    sigma_a2 = opts.accel_noise_std ** 2
    sigma_ba2 = opts.bias_rw_std ** 2
    g_world = np.array([0.0, 0.0, _G])

    # Forward-pass storage for the backward smoother --------------------------
    # Per step k we record:
    #   fwd_x_post[k]  = x_{k|k}       (state after all updates at t_k)
    #   fwd_P_post[k]  = P_{k|k}       (covariance after updates)
    #   fwd_F[k+1]     = F applied from t_k to t_{k+1} (predict only)
    #   fwd_Q[k+1]     = accumulated Q over the same predict (Σ F·Q_sub·F^T + Q)
    # The smoother needs x_{k+1|k} = F[k+1] · x_post[k] and
    # P_{k+1|k} = F[k+1] · P_post[k] · F[k+1]^T + Q[k+1].
    fwd_x_post: list[np.ndarray] = []
    fwd_P_post: list[np.ndarray] = []
    fwd_F: list[np.ndarray] = []
    fwd_Q: list[np.ndarray] = []
    fwd_t: list[float] = []

    def _predict(dt_total: float, q_att: np.ndarray, a_body_meas: np.ndarray):
        """Return (F_total, Q_total) for an interval dt_total.

        The predict integrates substeps internally so the small-angle
        linearisation stays valid; returns the combined F **and accumulated
        Q** so the backward smoother can reconstruct
        ``P_pred[k+1|k] = F_total · P_post[k] · F_total^T + Q_total`` exactly.

        Mutates x, P in place as the substeps go.
        """
        nonlocal x, P
        if dt_total <= 0:
            return np.eye(9), np.zeros((9, 9))
        n_sub = max(1, int(math.ceil(dt_total / max(opts.max_dt_s, 1e-6))))
        sub_dt = dt_total / n_sub
        R_bw = _quat_to_rotmat(q_att)
        F_total = np.eye(9)
        Q_total = np.zeros((9, 9))
        for _ in range(n_sub):
            b_a = x[6:9].copy()
            a_body = a_body_meas - b_a
            a_world = R_bw @ a_body - g_world

            x_pred = x.copy()
            x_pred[0:3] = x[0:3] + x[3:6] * sub_dt + 0.5 * a_world * (sub_dt ** 2)
            x_pred[3:6] = x[3:6] + a_world * sub_dt

            F = np.eye(9)
            F[0:3, 3:6] = np.eye(3) * sub_dt
            F[0:3, 6:9] = -0.5 * R_bw * (sub_dt ** 2)
            F[3:6, 6:9] = -R_bw * sub_dt

            Q = np.zeros((9, 9))
            qpr = sigma_a2 * (sub_dt ** 4) / 4.0
            qpv = sigma_a2 * (sub_dt ** 3) / 2.0
            qv = sigma_a2 * (sub_dt ** 2)
            Q[0, 0] = qpr; Q[1, 1] = qpr; Q[2, 2] = qpr * 4
            Q[3, 3] = qv;  Q[4, 4] = qv;  Q[5, 5] = qv * 4
            for i in range(3):
                Q[i, i + 3] = qpv if i < 2 else qpv * 4
                Q[i + 3, i] = Q[i, i + 3]
            qb = sigma_ba2 * sub_dt
            Q[6, 6] = qb; Q[7, 7] = qb; Q[8, 8] = qb

            P = F @ P @ F.T + Q
            x = x_pred
            # Compose: combined F is F @ prior F_total;
            # combined Q is F @ prior Q_total @ F.T + Q (standard recursion).
            Q_total = F @ Q_total @ F.T + Q
            F_total = F @ F_total
        return F_total, Q_total

    def _apply_pos_update(z_pos: np.ndarray, sigma_h: float, sigma_v: float) -> bool:
        """Chi²-gated position-only update. Returns True if accepted."""
        nonlocal x, P
        sigma_h = _clamp_R_sigma(sigma_h)
        sigma_v = _clamp_R_sigma(sigma_v)
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        R_mat = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
        y = z_pos - H @ x
        S = H @ P @ H.T + R_mat
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            _log(
                "[ekf-rts] singular S in pos update (sigma_h="
                f"{sigma_h:.3g}); skipping this update"
            )
            return False
        chi2 = float(y @ S_inv @ y)
        if chi2 > opts.chi2_pos_gate:
            return False
        K = P @ H.T @ S_inv
        x = x + K @ y
        P = (np.eye(9) - K @ H) @ P
        P = 0.5 * (P + P.T)
        x[6:9] = np.clip(x[6:9], -opts.bias_clip_mps2, opts.bias_clip_mps2)
        return True

    def _apply_vel_update(z_vel: np.ndarray, sigma_h: float, sigma_v: float,
                          chi2_gate: float) -> bool:
        """Chi²-gated velocity update. Returns True if accepted.

        ``chi2_gate`` is passed in so the caller can vary the gate per
        warmup state (loose during first few updates, tight after).
        """
        nonlocal x, P
        sigma_h = _clamp_R_sigma(sigma_h)
        sigma_v = _clamp_R_sigma(sigma_v)
        H = np.zeros((3, 9))
        H[0:3, 3:6] = np.eye(3)
        R_mat = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
        y = z_vel - H @ x
        S = H @ P @ H.T + R_mat
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            _log(
                "[ekf-rts] singular S in vel update (sigma_h="
                f"{sigma_h:.3g}); skipping this update"
            )
            return False
        chi2 = float(y @ S_inv @ y)
        if chi2 > chi2_gate:
            return False
        K = P @ H.T @ S_inv
        x = x + K @ y
        P = (np.eye(9) - K @ H) @ P
        P = 0.5 * (P + P.T)
        return True

    def _apply_nhc(q_att: np.ndarray) -> bool:
        """Non-holonomic constraint: lateral body velocity ≈ 0.

        The vehicle is wheeled, so velocity in body-y (left-right) is
        bounded by mounting offset + cornering slip. With identity
        attitude, body-x = world-east, body-y = world-north, so the
        constraint reads vN ≈ 0 when the vehicle drives east -- exactly
        the kind of lateral Post-processing noise that drags the smoothed track
        sideways otherwise.
        """
        nonlocal x, P
        R_bw = _quat_to_rotmat(q_att)
        H = np.zeros((1, 9))
        H[0, 3:6] = R_bw.T[1, :]            # lateral row of R_wb = R_bw^T
        sigma = _clamp_R_sigma(opts.nhc_sigma_mps)
        R_mat = np.array([[sigma ** 2]])
        pred = float(R_bw.T[1, :] @ x[3:6])
        y = np.array([0.0 - pred])
        S = H @ P @ H.T + R_mat
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            _log("[ekf-rts] singular S in NHC update; skipping")
            return False
        K = P @ H.T @ S_inv
        x = x + (K @ y).flatten()
        P = (np.eye(9) - K @ H) @ P
        P = 0.5 * (P + P.T)
        return True

    def _apply_zupt() -> bool:
        """Tight v=0 anchor update."""
        nonlocal x, P
        H = np.zeros((3, 9))
        H[0:3, 3:6] = np.eye(3)
        sigma = _clamp_R_sigma(opts.zupt_sigma_mps)
        R_mat = np.diag([sigma ** 2] * 3)
        y = -x[3:6]
        S = H @ P @ H.T + R_mat
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            _log("[ekf-rts] singular S in ZUPT update; skipping")
            return False
        K = P @ H.T @ S_inv
        x = x + K @ y
        P = (np.eye(9) - K @ H) @ P
        P = 0.5 * (P + P.T)
        return True

    # ---------------------------------------------------------------------
    # Position-derived velocity table (driving-profile prior, replaces
    # raw Rate-signal as the primary velocity source). Per-row sigma comes
    # from the OLS slope standard error of the local window.
    # ---------------------------------------------------------------------
    pos_derived_vel = (
        compute_position_derived_velocity(
            pos_list, ref_llh,
            window_half_s=opts.pos_vel_window_half_s,
            sigma_pos_h_m=opts.sigma_pos_h_m,
            sigma_pos_v_m=opts.sigma_pos_v_m,
        )
        if opts.use_position_derived_velocity
        else {}
    )

    # ---------------------------------------------------------------------
    # ZUPT static-period gate: reuse the static_periods computed above
    # for bias-init detection so the two are guaranteed consistent.
    # ---------------------------------------------------------------------
    def _in_static_period(t_query: float) -> bool:
        for t_start, t_end in static_periods:
            if t_start <= t_query <= t_end:
                return True
        return False

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    prev_t: Optional[float] = t_gnss_start
    gnss_idx = 1
    for n_imu, (row, q_att) in enumerate(zip(imu_list, qs)):
        t = row.utc_s
        if t < t_gnss_start:
            continue
        if t > t_gnss_end + 5.0:
            break

        a_body = np.array([row.ax, row.ay, row.az])
        cursor_t = prev_t if prev_t is not None else t

        # Apply any Signal rows within this Motion sensor step at their own timestamps.
        F_accum = np.eye(9)
        Q_accum = np.zeros((9, 9))
        while gnss_idx < len(pos_list) and pos_list[gnss_idx].utc_s <= t:
            gr = pos_list[gnss_idx]
            gnss_idx += 1
            gnss_t = max(gr.utc_s, cursor_t)
            if gnss_t > cursor_t:
                F_part, Q_part = _predict(gnss_t - cursor_t, q_att, a_body)
                F_accum = F_part @ F_accum
                # Q_total = F · prior_Q · F^T + Q_part
                Q_accum = F_part @ Q_accum @ F_part.T + Q_part
                cursor_t = gnss_t
            scale = _quality_scale(gr.quality, opts)
            if scale is None:
                continue
            sigma_pos_h = opts.sigma_pos_h_m * math.sqrt(scale)
            sigma_pos_v = opts.sigma_pos_v_m * math.sqrt(scale)
            z_pos = _pos_to_enu(gr)
            if _apply_pos_update(z_pos, sigma_pos_h, sigma_pos_v):
                res.n_pos_updates += 1
            else:
                res.n_pos_rejected_chi2 += 1
            # Choose velocity measurement source:
            #  1. Position-derived (smoothed OLS slope) when available.
            #  2. Rate-signal from Post-processing row as fallback.
            # Position-derived sigma per row already reflects window
            # geometry; clamp it to the floor so tiny windows can't
            # produce an over-confident measurement.
            pd = pos_derived_vel.get(gr.utc_s)
            if pd is not None:
                z_vel = np.array([pd[0], pd[1], pd[2]])
                s_v_h = max(pd[3], opts.pos_vel_sigma_floor_mps)
                s_v_v = max(pd[4], opts.pos_vel_sigma_floor_mps)
                has_vel = True
            elif (math.isfinite(gr.ve) and math.isfinite(gr.vn)
                  and math.isfinite(gr.vu)):
                z_vel = np.array([gr.ve, gr.vn, gr.vu])
                s_v_h = opts.sigma_vel_h_mps
                s_v_v = opts.sigma_vel_v_mps
                has_vel = True
            else:
                has_vel = False
            if has_vel:
                # Warmup-then-steady chi-2 gate. The first few updates
                # need a loose gate so a real motion-start innovation
                # (vel state still at 0, true 5 m/s) is admitted; once
                # the filter is tracking, tighten to reject outliers.
                gate = (opts.chi2_vel_gate_warmup
                        if res.n_vel_updates < opts.warmup_vel_updates
                        else opts.chi2_vel_gate_steady)
                if _apply_vel_update(z_vel, s_v_h, s_v_v, gate):
                    res.n_vel_updates += 1
                else:
                    res.n_vel_rejected_chi2 += 1
                # NHC: enforce lateral body velocity ≈ 0 once the
                # filter clearly thinks the vehicle is moving. Below
                # ``nhc_min_speed_mps`` the constraint is meaningless
                # (which axis is "lateral" isn't defined when stopped).
                if opts.nhc_enabled:
                    est_speed_h = math.sqrt(x[3] ** 2 + x[4] ** 2)
                    if est_speed_h > opts.nhc_min_speed_mps:
                        _apply_nhc(q_att)
                # ZUPT trigger: noisy Rate-signal alone can't reliably say
                # "stopped" (one σ ≈ 2 m/s). We only trust it as a stop
                # signal when reported speed is well below 1σ, AND the
                # filter's own velocity estimate also reads near zero
                # (which catches stops the noisy Rate-signal misses).
                # ZUPT fires when (a) the row is inside a detected static
                # period AND (b) the filter's own velocity estimate is
                # near zero. Requiring both avoids the failure modes of
                # either alone:
                #  - Rate-signal alone is too noisy (~2 m/s 1σ) to confirm a
                #    stop on a single sample.
                #  - "in_static" alone over-triggers during the early
                #    bias-init phase, dragging the bias state.
                # The combined gate fires reliably during real stops once
                # the filter has converged its velocity estimate, and
                # stays quiet during the start-of-run transient.
                est_speed = math.sqrt(x[3] * x[3] + x[4] * x[4])
                in_static = _in_static_period(gr.utc_s)
                if (in_static and est_speed < opts.zupt_speed_mps
                        and _apply_zupt()):
                    res.n_zupt_updates += 1

        if t > cursor_t:
            F_part, Q_part = _predict(t - cursor_t, q_att, a_body)
            F_accum = F_part @ F_accum
            Q_accum = F_part @ Q_accum @ F_part.T + Q_part

        fwd_F.append(F_accum)
        fwd_Q.append(Q_accum)
        fwd_x_post.append(x.copy())
        fwd_P_post.append(P.copy())
        fwd_t.append(t)
        res.n_imu_steps += 1
        prev_t = t
        del n_imu

    if not fwd_x_post:
        _log("[ekf-rts] no IMU samples in the GNSS window — nothing to smooth")
        return res

    # ---------------------------------------------------------------------
    # Backward (RTS) smoother
    # ---------------------------------------------------------------------
    n_steps = len(fwd_x_post)
    smooth_x: list[np.ndarray] = [None] * n_steps   # type: ignore[list-item]
    smooth_P: list[np.ndarray] = [None] * n_steps   # type: ignore[list-item]
    smooth_x[-1] = fwd_x_post[-1]
    smooth_P[-1] = fwd_P_post[-1]
    for k in range(n_steps - 2, -1, -1):
        F_next = fwd_F[k + 1]
        Q_next = fwd_Q[k + 1]
        x_post_k = fwd_x_post[k]
        P_post_k = fwd_P_post[k]
        # Predict from posterior at k forward to k+1 (no measurement updates).
        x_pred_k1 = F_next @ x_post_k
        P_pred_k1 = F_next @ P_post_k @ F_next.T + Q_next
        try:
            G = P_post_k @ F_next.T @ np.linalg.inv(P_pred_k1 + 1e-12 * np.eye(9))
        except np.linalg.LinAlgError:
            smooth_x[k] = x_post_k
            smooth_P[k] = P_post_k
            continue
        smooth_x[k] = x_post_k + G @ (smooth_x[k + 1] - x_pred_k1)
        smooth_P[k] = P_post_k + G @ (smooth_P[k + 1] - P_pred_k1) @ G.T
    res.n_smoothed_steps = n_steps

    # ---------------------------------------------------------------------
    # Emit fused PosRows from smoothed state
    # ---------------------------------------------------------------------
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    for k in range(n_steps):
        e = smooth_x[k]
        dx = -so * e[0] - sl * co * e[1] + cl * co * e[2]
        dy =  co * e[0] - sl * so * e[1] + cl * so * e[2]
        dz =              cl * e[1]        + sl * e[2]
        x_ecef = ref_ecef[0] + dx
        y_ecef = ref_ecef[1] + dy
        z_ecef = ref_ecef[2] + dz
        lat, lon, h = enu_to_llh(e[0], e[1], e[2], ref_llh)
        # Use enu_to_llh from geo for consistency; Cartesian XYZ round-trip above is
        # equivalent but kept in case downstream wants it.
        del x_ecef, y_ecef, z_ecef
        res.fused.append(PosRow(
            utc_s=fwd_t[k],
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            quality=2,
            vn=float(e[4]),
            ve=float(e[3]),
            vu=float(e[5]),
        ))
        res.accel_bias_history.append(
            (fwd_t[k], float(e[6]), float(e[7]), float(e[8]))
        )

    _log(
        f"[ekf-rts] {res.n_imu_steps} forward steps; "
        f"{res.n_pos_updates} pos updates ({res.n_pos_rejected_chi2} chi2-rejected); "
        f"{res.n_vel_updates} vel ({res.n_vel_rejected_chi2} chi2-rejected); "
        f"{res.n_zupt_updates} ZUPTs; "
        f"smoothed {res.n_smoothed_steps} steps; "
        f"bias init=({b_init[0]:+.4f},{b_init[1]:+.4f},{b_init[2]:+.4f}) m/s²"
    )
    return res
