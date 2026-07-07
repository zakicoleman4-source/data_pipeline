"""The external solver-style Signal forward prediction smoother with optional Motion sensor.

Two operating modes selected automatically based on input:

1. **Signal-only** (no ``imu_rows``): 9-state EKF
   ``x = [E, N, U, vE, vN, vU, aE, aN, aU]``
   Acceleration as random walk — The external solver's "unit dynamics" model.
   ``prnaccel`` controls how fast acceleration can change. RTS backward
   pass recovers path through gaps better than forward-only.

2. **Motion sensor mode** (``imu_rows`` supplied): delegates to
   :func:`ekf_smoothed.run_ekf_rts` which runs a 9-state
   ``[pos, vel, accel_bias]`` EKF with Complementary-update attitude, then RTS.
   Device Motion sensor propagates through Signal gaps; bias estimated online.

Both modes: quality-adaptive measurement R, ZUPT at detected stops,
RTS backward smoothing, automatic gap bridging.

Entry point: ``run_dr(pos_rows, imu_rows=None, ...) -> DrResult``
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .parsers import ImuRow, PosRow


_G = 9.80665


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class DrOptions:
    """Tunables for the dead-reckoning smoother.

    Defaults calibrated for device Post-processing (1 Hz, float-quality).
    ``prnaccel`` is the external solver-equivalent parameter: standard deviation
    of acceleration random walk (m/s^2 / sqrt(Hz)).
    """

    # Process noise — acceleration random walk (Signal-only mode).
    # The external solver default is 1 m/s^2 for kinematic; increase for aggressive
    # driving, decrease for smooth highway.
    prnaccel: float = 1.0

    # Vertical process noise multiplier (vertical linear sensor noisier on device).
    prnaccel_v_scale: float = 2.0

    # Position measurement noise (1-sigma, metres).
    sigma_pos_h: float = 0.5
    sigma_pos_v: float = 1.5

    # Velocity measurement noise (1-sigma, m/s).
    sigma_vel_h: float = 0.3
    sigma_vel_v: float = 0.5

    # Quality-adaptive R scaling (relative to float = 1.0).
    fix_r_scale: float = 0.02
    dgps_r_scale: float = 3.0
    single_r_scale: float = 10.0

    # ZUPT: when Post-processing Rate-signal speed < this for >= zupt_min_s seconds.
    zupt_speed_mps: float = 0.5
    zupt_min_s: float = 1.5
    zupt_sigma_mps: float = 0.03

    # Initial covariance.
    p0_pos_h: float = 9.0
    p0_pos_v: float = 100.0
    p0_vel_h: float = 1.0
    p0_vel_v: float = 4.0
    p0_accel: float = 4.0

    # Subsample output: emit one PosRow per Signal epoch (True) or keep
    # the full Motion sensor-rate output when in Motion sensor mode (False).
    output_at_gnss_rate: bool = True


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DrResult:
    """Outcome of one dead-reckoning run."""
    fused: list[PosRow] = field(default_factory=list)
    mode: str = "gnss_only"
    n_input: int = 0
    n_output: int = 0
    n_gaps_bridged: int = 0
    n_zupt: int = 0
    n_pos_updates: int = 0
    n_vel_updates: int = 0
    max_gap_s: float = 0.0


# ---------------------------------------------------------------------------
# Quality-adaptive R
# ---------------------------------------------------------------------------

def _quality_r_scale(quality: int, opts: DrOptions) -> Optional[float]:
    if quality == 1:
        return opts.fix_r_scale ** 2
    if quality == 2:
        return 1.0
    if quality == 4:
        return opts.dgps_r_scale ** 2
    if quality == 5:
        return opts.single_r_scale ** 2
    if quality in (0, 3, 6, 7, 8):
        return opts.single_r_scale ** 2
    return None


# ---------------------------------------------------------------------------
# Static-period detector (position-delta speed, not Rate-signal)
# ---------------------------------------------------------------------------

def _detect_stops(
    pos_rows: Sequence[PosRow],
    ref_llh: tuple[float, float, float],
    max_speed: float = 0.5,
    min_dur: float = 1.5,
) -> list[tuple[float, float]]:
    if len(pos_rows) < 2:
        return []
    en = []
    for r in pos_rows:
        e, n, _u = ecef_to_enu(
            *llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh,
        )
        en.append((e, n))
    periods: list[tuple[float, float]] = []
    start: Optional[float] = None
    end: Optional[float] = None
    for i in range(1, len(pos_rows)):
        dt = pos_rows[i].utc_s - pos_rows[i - 1].utc_s
        if dt <= 0:
            continue
        dx = en[i][0] - en[i - 1][0]
        dy = en[i][1] - en[i - 1][1]
        spd = math.sqrt(dx * dx + dy * dy) / dt
        if spd < max_speed:
            if start is None:
                start = pos_rows[i - 1].utc_s
            end = pos_rows[i].utc_s
        else:
            if start is not None and end is not None and (end - start) >= min_dur:
                periods.append((start, end))
            start = end = None
    if start is not None and end is not None and (end - start) >= min_dur:
        periods.append((start, end))
    return periods


def _in_stop(t: float, stops: list[tuple[float, float]]) -> bool:
    for s, e in stops:
        if s <= t <= e:
            return True
    return False


# ---------------------------------------------------------------------------
# Signal-only 9-state EKF + RTS  (The external solver unit-dynamics equivalent)
# ---------------------------------------------------------------------------

def _run_gnss_only(
    pos_rows: list[PosRow],
    opts: DrOptions,
    log,
) -> DrResult:
    """9-state [pos, vel, linear sensor] EKF with acceleration random walk + RTS.

    This is the external solver "unit dynamics ON" approach ported to Python
    for position-level (.pos) post-processing.
    """
    res = DrResult(mode="gnss_only", n_input=len(pos_rows))
    if len(pos_rows) < 2:
        res.fused = list(pos_rows)
        res.n_output = len(res.fused)
        return res

    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    ref_llh = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
    stops = _detect_stops(rows, ref_llh, max_speed=opts.zupt_speed_mps,
                          min_dur=opts.zupt_min_s)

    def _to_enu(r: PosRow) -> np.ndarray:
        e, n, u = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)
        return np.array([e, n, u])

    # State: [E, N, U, vE, vN, vU, aE, aN, aU]
    x = np.zeros(9)
    x[0:3] = _to_enu(rows[0])
    r0 = rows[0]
    if math.isfinite(r0.ve) and math.isfinite(r0.vn):
        x[3] = r0.ve
        x[4] = r0.vn
        x[5] = r0.vu if math.isfinite(r0.vu) else 0.0

    P = np.diag([
        opts.p0_pos_h, opts.p0_pos_h, opts.p0_pos_v,
        opts.p0_vel_h, opts.p0_vel_h, opts.p0_vel_v,
        opts.p0_accel, opts.p0_accel, opts.p0_accel * opts.prnaccel_v_scale,
    ])

    sigma_a_h = opts.prnaccel
    sigma_a_v = opts.prnaccel * opts.prnaccel_v_scale

    # Forward-pass tape for RTS
    tape_t: list[float] = []
    tape_x_post: list[np.ndarray] = []
    tape_P_post: list[np.ndarray] = []
    tape_F: list[np.ndarray] = []
    tape_Q: list[np.ndarray] = []

    prev_t = rows[0].utc_s

    # Process first epoch as initial state (already set above)
    tape_t.append(prev_t)
    tape_x_post.append(x.copy())
    tape_P_post.append(P.copy())
    tape_F.append(np.eye(9))
    tape_Q.append(np.zeros((9, 9)))

    for k in range(1, len(rows)):
        gr = rows[k]
        dt = gr.utc_s - prev_t
        if dt <= 0:
            continue
        prev_t = gr.utc_s

        if dt > 2.0:
            res.n_gaps_bridged += 1
            res.max_gap_s = max(res.max_gap_s, dt)

        # ── Predict: constant-acceleration model ──
        # pos += vel*dt + 0.5*linear sensor*dt^2
        # vel += linear sensor*dt
        # linear sensor unchanged (random walk)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * (0.5 * dt2)
        F[3:6, 6:9] = np.eye(3) * dt

        x = F @ x

        # Process noise: piecewise-constant jerk model
        # Q derived from sigma_a on acceleration state.
        sa_h2 = sigma_a_h ** 2
        sa_v2 = sigma_a_v ** 2
        Q = np.zeros((9, 9))
        # Position block
        Q[0, 0] = sa_h2 * dt4 / 4.0
        Q[1, 1] = sa_h2 * dt4 / 4.0
        Q[2, 2] = sa_v2 * dt4 / 4.0
        # Velocity block
        Q[3, 3] = sa_h2 * dt2
        Q[4, 4] = sa_h2 * dt2
        Q[5, 5] = sa_v2 * dt2
        # Acceleration block
        Q[6, 6] = sa_h2 * dt
        Q[7, 7] = sa_h2 * dt
        Q[8, 8] = sa_v2 * dt
        # Cross terms (pos-vel, vel-linear sensor)
        pv = sa_h2 * dt3 / 2.0
        pv_v = sa_v2 * dt3 / 2.0
        for i in range(2):
            Q[i, i + 3] = pv
            Q[i + 3, i] = pv
        Q[2, 5] = pv_v
        Q[5, 2] = pv_v

        P = F @ P @ F.T + Q

        # ── Position update ──
        scale = _quality_r_scale(gr.quality, opts)
        if scale is not None:
            z_pos = _to_enu(gr)
            sh = opts.sigma_pos_h * math.sqrt(scale)
            sv = opts.sigma_pos_v * math.sqrt(scale)
            H_p = np.zeros((3, 9))
            H_p[0:3, 0:3] = np.eye(3)
            R_p = np.diag([sh ** 2, sh ** 2, sv ** 2])
            y_p = z_pos - H_p @ x
            S_p = H_p @ P @ H_p.T + R_p
            try:
                K_p = P @ H_p.T @ np.linalg.inv(S_p)
                x = x + K_p @ y_p
                P = (np.eye(9) - K_p @ H_p) @ P
                P = 0.5 * (P + P.T)
                res.n_pos_updates += 1
            except np.linalg.LinAlgError:
                pass

        # ── Velocity update (Rate-signal) ──
        has_vel = (math.isfinite(gr.ve) and math.isfinite(gr.vn)
                   and math.isfinite(gr.vu))
        if has_vel:
            z_v = np.array([gr.ve, gr.vn, gr.vu])
            H_v = np.zeros((3, 9))
            H_v[0:3, 3:6] = np.eye(3)
            R_v = np.diag([opts.sigma_vel_h ** 2, opts.sigma_vel_h ** 2,
                           opts.sigma_vel_v ** 2])
            y_v = z_v - H_v @ x
            S_v = H_v @ P @ H_v.T + R_v
            try:
                K_v = P @ H_v.T @ np.linalg.inv(S_v)
                x = x + K_v @ y_v
                P = (np.eye(9) - K_v @ H_v) @ P
                P = 0.5 * (P + P.T)
                res.n_vel_updates += 1
            except np.linalg.LinAlgError:
                pass

        # ── ZUPT: zero-velocity anchor at detected stops ──
        if _in_stop(gr.utc_s, stops):
            H_z = np.zeros((3, 9))
            H_z[0:3, 3:6] = np.eye(3)
            R_z = np.diag([opts.zupt_sigma_mps ** 2] * 3)
            y_z = -x[3:6]
            S_z = H_z @ P @ H_z.T + R_z
            try:
                K_z = P @ H_z.T @ np.linalg.inv(S_z)
                x = x + K_z @ y_z
                P = (np.eye(9) - K_z @ H_z) @ P
                P = 0.5 * (P + P.T)
                res.n_zupt += 1
            except np.linalg.LinAlgError:
                pass

        tape_t.append(gr.utc_s)
        tape_x_post.append(x.copy())
        tape_P_post.append(P.copy())
        tape_F.append(F)
        tape_Q.append(Q)

    # ── RTS backward smoother ──
    n = len(tape_t)
    if n < 2:
        res.fused = list(pos_rows)
        res.n_output = len(res.fused)
        return res

    sx: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    sP: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    sx[-1] = tape_x_post[-1]
    sP[-1] = tape_P_post[-1]

    for k in range(n - 2, -1, -1):
        F_next = tape_F[k + 1]
        Q_next = tape_Q[k + 1]
        x_post = tape_x_post[k]
        P_post = tape_P_post[k]
        x_pred = F_next @ x_post
        P_pred = F_next @ P_post @ F_next.T + Q_next
        try:
            G = P_post @ F_next.T @ np.linalg.inv(P_pred + 1e-12 * np.eye(9))
        except np.linalg.LinAlgError:
            sx[k] = x_post
            sP[k] = P_post
            continue
        sx[k] = x_post + G @ (sx[k + 1] - x_pred)
        sP[k] = P_post + G @ (sP[k + 1] - P_pred) @ G.T

    # ── Emit smoothed PosRows ──
    for k in range(n):
        e = sx[k]
        lat, lon, h = enu_to_llh(float(e[0]), float(e[1]), float(e[2]), ref_llh)
        res.fused.append(PosRow(
            utc_s=tape_t[k],
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            quality=2,
            vn=float(e[4]),
            ve=float(e[3]),
            vu=float(e[5]),
        ))

    res.n_output = len(res.fused)
    log(f"[dr-gnss] {n} epochs, {res.n_pos_updates} pos updates, "
        f"{res.n_vel_updates} vel updates, {res.n_zupt} ZUPTs, "
        f"{res.n_gaps_bridged} gaps (max {res.max_gap_s:.1f}s)")
    return res


# ---------------------------------------------------------------------------
# Motion sensor mode: delegate to ekf_smoothed with gap-adaptive tuning
# ---------------------------------------------------------------------------

def _run_imu(
    pos_rows: list[PosRow],
    imu_rows: list[ImuRow],
    opts: DrOptions,
    log,
) -> DrResult:
    """9-state [pos, vel, accel_bias] EKF + RTS with device Motion sensor.

    Wraps :func:`ekf_smoothed.run_ekf_rts` with tuning adapted for
    automatic forward prediction through Signal gaps.
    """
    from .ekf_smoothed import RtsOptions, run_ekf_rts

    rts_opts = RtsOptions(
        accel_noise_std=3.0,
        bias_rw_std=0.003,
        sigma_pos_h_m=opts.sigma_pos_h,
        sigma_pos_v_m=opts.sigma_pos_v,
        sigma_vel_h_mps=opts.sigma_vel_h,
        sigma_vel_v_mps=opts.sigma_vel_v,
        fix_scale=math.sqrt(opts.fix_r_scale),
        dgps_scale=math.sqrt(opts.dgps_r_scale),
        single_scale=math.sqrt(opts.single_r_scale),
        zupt_speed_mps=opts.zupt_speed_mps,
        zupt_sigma_mps=opts.zupt_sigma_mps,
        nhc_enabled=True,
        use_position_derived_velocity=True,
    )

    rts_result = run_ekf_rts(imu_rows, pos_rows, options=rts_opts, log=log)

    res = DrResult(
        mode="imu",
        fused=rts_result.fused,
        n_input=len(pos_rows),
        n_output=len(rts_result.fused),
        n_pos_updates=rts_result.n_pos_updates,
        n_vel_updates=rts_result.n_vel_updates,
        n_zupt=rts_result.n_zupt_updates,
    )

    # Count gaps in Post-processing timeline
    sorted_pos = sorted(pos_rows, key=lambda r: r.utc_s)
    for i in range(1, len(sorted_pos)):
        dt = sorted_pos[i].utc_s - sorted_pos[i - 1].utc_s
        if dt > 2.0:
            res.n_gaps_bridged += 1
            res.max_gap_s = max(res.max_gap_s, dt)

    if opts.output_at_gnss_rate and res.fused:
        pos_times = {r.utc_s for r in sorted_pos}
        res.fused = [r for r in res.fused if r.utc_s in pos_times]
        res.n_output = len(res.fused)

    log(f"[dr-imu] {res.n_output} epochs, {res.n_pos_updates} pos, "
        f"{res.n_vel_updates} vel, {res.n_zupt} ZUPT, "
        f"{res.n_gaps_bridged} gaps (max {res.max_gap_s:.1f}s)")
    return res


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dr(
    pos_rows: Sequence[PosRow],
    imu_rows: Optional[Sequence[ImuRow]] = None,
    options: Optional[DrOptions] = None,
    log: Optional[object] = None,
) -> DrResult:
    """Run Signal(-Motion sensor) forward prediction smoother.

    Automatically selects Motion sensor mode when ``imu_rows`` is non-empty,
    otherwise falls back to Signal-only (The external solver-style unit dynamics).
    """
    opts = options or DrOptions()

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)  # type: ignore[operator]

    pos_list = sorted(pos_rows, key=lambda r: r.utc_s)
    if not pos_list:
        return DrResult(n_input=0)

    if imu_rows and len(imu_rows) > 0:
        _log(f"[dr] IMU mode: {len(imu_rows)} IMU + {len(pos_list)} PPK rows")
        return _run_imu(pos_list, list(imu_rows), opts, _log)
    else:
        _log(f"[dr] GNSS-only mode: {len(pos_list)} PPK rows "
             f"(prnaccel={opts.prnaccel})")
        return _run_gnss_only(pos_list, opts, _log)
