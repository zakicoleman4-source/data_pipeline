"""Two textbook Recursive filters for car-on-road Signal smoothing.

These are deliberately minimal, well-known designs that *do not* try to
integrate device linear sensors (which on this hardware adds more error
than it removes). Both produce a dense interpolated path pinned to
Post-processing position+Rate-signal-velocity measurements.

1. CV-KF -- Constant-Velocity Recursive filter (4-state)
    State: [px, py, vx, vy] in Local-frame metres / metres-per-second.
    Process: x' = F x + w, with F encoding pos += v*dt and v constant.
    Process noise driven by an "unknown acceleration" white-noise model
    (Bar-Shalom Q matrix). Classic linear KF; identical structure to
    The feature library's KalmanFilter tracking demo and every textbook example.

2. CTRV-EKF -- Constant-Turn-Rate-and-Velocity (5-state)
    State: [px, py, v, psi, psi_dot].
    Motion model integrates the bicycle-like arc:
        px += (v / psi_dot) * (sin(psi + psi_dot*dt) - sin(psi))
        py += (v / psi_dot) * (-cos(psi + psi_dot*dt) + cos(psi))
        psi += psi_dot * dt
    Falls back to straight-line when |psi_dot| < eps. Process noise is
    on longitudinal linear sensor (nu_a) and yaw acceleration (nu_psi_dd). Used
    in the Udacity Self-Driving Car nanodegree's unscented-KF project.
    Rate sensor yaw rate can optionally update psi_dot directly.

Both filters expose the same shape: input Post-processing rows (positions + Rate-signal
velocities), dense output time grid, return PosRow-style dense list at
the chosen output rate.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .parsers import ImuRow, PosRow


def _enu_to_llh_helpers(ref_llh: tuple[float, float, float]):
    """Return closures that convert PosRow <-> Local-frame about ``ref_llh``."""
    ref_ecef = llh_to_ecef(*ref_llh)
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    from .geo import _A, _E2

    def pos_to_enu(r: PosRow) -> tuple[float, float, float]:
        return ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)

    def enu_to_llh(e: float, n: float, u: float) -> tuple[float, float, float]:
        dx = -so * e - sl * co * n + cl * co * u
        dy =  co * e - sl * so * n + cl * so * u
        dz =                cl * n + sl * u
        x = ref_ecef[0] + dx
        y = ref_ecef[1] + dy
        z = ref_ecef[2] + dz
        p = math.sqrt(x * x + y * y)
        lon = math.atan2(y, x)
        lat = math.atan2(z, p * (1.0 - _E2))
        for _ in range(5):
            sl_ = math.sin(lat)
            n_ = _A / math.sqrt(1.0 - _E2 * sl_ * sl_)
            lat = math.atan2(z + _E2 * n_ * sl_, p)
        sl_ = math.sin(lat)
        n_ = _A / math.sqrt(1.0 - _E2 * sl_ * sl_)
        h = (p / math.cos(lat) - n_) if abs(math.cos(lat)) > 1e-9 \
            else (abs(z) / sl_ - n_ * (1.0 - _E2))
        return math.degrees(lat), math.degrees(lon), h

    return pos_to_enu, enu_to_llh


# ─────────────────────────────────────────────────────────────────────
# 1. Constant-Velocity Recursive filter
# ─────────────────────────────────────────────────────────────────────

def run_cv_kf(
    pos_rows: Sequence[PosRow],
    ref_llh: tuple[float, float, float],
    *,
    out_times: Sequence[float],
    accel_noise_std: float = 0.5,
    sigma_pos_h_m: float = 3.0,
    sigma_vel_h_mps: float = 1.0,
) -> list[PosRow]:
    """4-state CV Recursive-filter: [px, py, vx, vy] in Local-frame, with Rate-signal updates.

    Predicts and updates at ``pos_rows`` epochs (1 Hz typical), then
    samples the smoothed state at every time in ``out_times`` via linear
    interpolation of state vectors. Position and velocity are corrected
    every Post-processing epoch; between epochs the filter coasts on constant
    velocity. Vertical channel is interpolated independently (Post-processing pos
    only, no Rate-signal) because device Z is dominated by Post-processing noise.

    Process-noise model is the "white noise acceleration" formulation
    (Bar-Shalom & Li 1993, eq. 6.3.1-4):
        Q(dt) = sigma_a^2 * [[dt^4/4, dt^3/2],
                              [dt^3/2, dt^2]]
    per axis, with sigma_a = ``accel_noise_std``.
    """
    if not pos_rows:
        return []

    pos_to_enu, enu_to_llh = _enu_to_llh_helpers(ref_llh)
    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    n = len(rows)

    # Output: smoothed state per pos_row epoch, plus z-only series for vertical.
    pos_t = np.array([r.utc_s for r in rows], dtype=np.float64)
    p_enu = np.array([pos_to_enu(r) for r in rows], dtype=np.float64)
    vN = np.array([r.vn for r in rows], dtype=np.float64)
    vE = np.array([r.ve for r in rows], dtype=np.float64)

    # State [px, py, vx, vy], where x=East, y=North.
    x = np.zeros(4)
    x[0] = p_enu[0, 0]
    x[1] = p_enu[0, 1]
    if math.isfinite(vE[0]) and math.isfinite(vN[0]):
        x[2] = vE[0]; x[3] = vN[0]
    P = np.diag([9.0, 9.0, 4.0, 4.0])

    sigma_a2 = accel_noise_std ** 2
    R_pos = np.diag([sigma_pos_h_m ** 2, sigma_pos_h_m ** 2])
    R_vel = np.diag([sigma_vel_h_mps ** 2, sigma_vel_h_mps ** 2])

    smoothed_e = np.empty(n)
    smoothed_n = np.empty(n)
    smoothed_vx = np.empty(n)
    smoothed_vy = np.empty(n)

    prev_t = pos_t[0]
    for i in range(n):
        t = pos_t[i]
        dt = t - prev_t
        prev_t = t

        if dt > 0:
            F = np.eye(4)
            F[0, 2] = dt; F[1, 3] = dt
            x = F @ x
            Q_per = sigma_a2 * np.array([
                [dt ** 4 / 4.0, dt ** 3 / 2.0],
                [dt ** 3 / 2.0, dt ** 2],
            ])
            Q = np.zeros((4, 4))
            Q[0, 0] = Q_per[0, 0]; Q[0, 2] = Q_per[0, 1]
            Q[2, 0] = Q_per[1, 0]; Q[2, 2] = Q_per[1, 1]
            Q[1, 1] = Q_per[0, 0]; Q[1, 3] = Q_per[0, 1]
            Q[3, 1] = Q_per[1, 0]; Q[3, 3] = Q_per[1, 1]
            P = F @ P @ F.T + Q

        # Position update.
        H_p = np.zeros((2, 4)); H_p[0, 0] = 1.0; H_p[1, 1] = 1.0
        z_p = p_enu[i, :2]
        S = H_p @ P @ H_p.T + R_pos
        K = P @ H_p.T @ np.linalg.inv(S)
        x = x + K @ (z_p - H_p @ x)
        P = (np.eye(4) - K @ H_p) @ P

        # Rate-signal velocity update.
        if math.isfinite(vE[i]) and math.isfinite(vN[i]):
            H_v = np.zeros((2, 4)); H_v[0, 2] = 1.0; H_v[1, 3] = 1.0
            z_v = np.array([vE[i], vN[i]])
            S = H_v @ P @ H_v.T + R_vel
            K = P @ H_v.T @ np.linalg.inv(S)
            x = x + K @ (z_v - H_v @ x)
            P = (np.eye(4) - K @ H_v) @ P

        smoothed_e[i] = x[0]; smoothed_n[i] = x[1]
        smoothed_vx[i] = x[2]; smoothed_vy[i] = x[3]

    # Interpolate state at out_times.
    out_t = np.asarray(out_times, dtype=np.float64)
    e_out = np.interp(out_t, pos_t, smoothed_e, left=np.nan, right=np.nan)
    n_out = np.interp(out_t, pos_t, smoothed_n, left=np.nan, right=np.nan)
    u_out = np.interp(out_t, pos_t, p_enu[:, 2], left=np.nan, right=np.nan)
    vx_out = np.interp(out_t, pos_t, smoothed_vx, left=np.nan, right=np.nan)
    vy_out = np.interp(out_t, pos_t, smoothed_vy, left=np.nan, right=np.nan)

    out_rows: list[PosRow] = []
    for i, t in enumerate(out_t):
        if not (math.isfinite(e_out[i]) and math.isfinite(n_out[i])
                and math.isfinite(u_out[i])):
            continue
        lat, lon, h = enu_to_llh(float(e_out[i]), float(n_out[i]), float(u_out[i]))
        out_rows.append(PosRow(
            utc_s=float(t),
            lat_deg=lat, lon_deg=lon, h_m=h,
            quality=2,
            vn=float(vy_out[i]) if math.isfinite(vy_out[i]) else float("nan"),
            ve=float(vx_out[i]) if math.isfinite(vx_out[i]) else float("nan"),
            vu=float("nan"),
        ))
    return out_rows


# ─────────────────────────────────────────────────────────────────────
# 2. CTRV Extended Recursive filter (Constant Turn Rate and Velocity)
# ─────────────────────────────────────────────────────────────────────

def run_ctrv_ekf(
    pos_rows: Sequence[PosRow],
    ref_llh: tuple[float, float, float],
    *,
    out_times: Sequence[float],
    imu_rows: Optional[Sequence[ImuRow]] = None,
    accel_noise_std: float = 1.0,
    yaw_accel_noise_std: float = 0.3,
    sigma_pos_h_m: float = 3.0,
    sigma_vel_h_mps: float = 1.0,
    sigma_yaw_rate_radps: float = 0.05,
) -> list[PosRow]:
    """5-state CTRV EKF: state = [px, py, v, psi, psi_dot] in Local-frame.

    Motion model (continuous arc):
        px' = px + (v/psi_dot)*(sin(psi+psi_dot*dt) - sin(psi))      if |psi_dot|>eps
        py' = py + (v/psi_dot)*(-cos(psi+psi_dot*dt) + cos(psi))
        v'  = v
        psi' = psi + psi_dot*dt
        psi_dot' = psi_dot
    Falls back to straight-line motion when psi_dot ~ 0.

    Signal measurements:
        z_pos = [px, py]                 R from sigma_pos_h_m
        z_v = sqrt(ve^2 + vn^2)          R from sigma_vel_h_mps
        z_psi = atan2(ve, vn)            yaw measurement (when speed > 0.5)
    Optional rate sensor yaw-rate from sensor file gives a direct measurement of
    psi_dot at Motion sensor rate (de-biased via the median yaw rate during stops).
    Vertical channel handled independently (linear interp of Post-processing h).
    """
    if not pos_rows:
        return []
    pos_to_enu, enu_to_llh = _enu_to_llh_helpers(ref_llh)
    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    n = len(rows)
    pos_t = np.array([r.utc_s for r in rows], dtype=np.float64)
    p_enu = np.array([pos_to_enu(r) for r in rows], dtype=np.float64)
    vN = np.array([r.vn for r in rows], dtype=np.float64)
    vE = np.array([r.ve for r in rows], dtype=np.float64)

    # Optional rate sensor yaw-rate stream (around z body axis ≈ z world for level car).
    gyro_t: np.ndarray | None = None
    gyro_z: np.ndarray | None = None
    gyro_bias = 0.0
    if imu_rows:
        gyro_t = np.array([r.utc_s for r in imu_rows], dtype=np.float64)
        gyro_z = np.array([r.gz for r in imu_rows], dtype=np.float64)
        # Estimate rate sensor bias as the median yaw rate when Signal speed is low.
        slow_mask = np.zeros(len(gyro_t), dtype=bool)
        # Mark intervals where bracketing Post-processing speed < 0.3 m/s.
        for i in range(n - 1):
            sp_i = math.hypot(vN[i], vE[i]) if (
                math.isfinite(vN[i]) and math.isfinite(vE[i])
            ) else float("inf")
            sp_j = math.hypot(vN[i + 1], vE[i + 1]) if (
                math.isfinite(vN[i + 1]) and math.isfinite(vE[i + 1])
            ) else float("inf")
            if max(sp_i, sp_j) < 0.3:
                mask = (gyro_t >= pos_t[i]) & (gyro_t < pos_t[i + 1])
                slow_mask |= mask
        if np.any(slow_mask):
            gyro_bias = float(np.median(gyro_z[slow_mask]))
        gyro_z = gyro_z - gyro_bias

    # Initial state.
    yaw0 = 0.0
    if math.isfinite(vE[0]) and math.isfinite(vN[0]):
        sp = math.hypot(vN[0], vE[0])
        if sp > 0.5:
            yaw0 = math.atan2(vE[0], vN[0])
    x = np.array([p_enu[0, 0], p_enu[0, 1],
                  math.hypot(vN[0], vE[0]) if (math.isfinite(vE[0])
                                               and math.isfinite(vN[0])) else 0.0,
                  yaw0,
                  0.0])
    P = np.diag([9.0, 9.0, 4.0, 0.5, 0.25])

    sa = accel_noise_std
    sd = yaw_accel_noise_std

    def _predict(state: np.ndarray, P_mat: np.ndarray, dt: float
                 ) -> tuple[np.ndarray, np.ndarray]:
        px, py, v, psi, psi_dot = state
        eps = 1e-4
        if abs(psi_dot) > eps:
            inv = 1.0 / psi_dot
            sp = math.sin(psi); cp = math.cos(psi)
            sp2 = math.sin(psi + psi_dot * dt)
            cp2 = math.cos(psi + psi_dot * dt)
            px_n = px + v * inv * (sp2 - sp)
            py_n = py + v * inv * (-cp2 + cp)
        else:
            px_n = px + v * math.cos(psi) * dt
            py_n = py + v * math.sin(psi) * dt
            sp2 = math.sin(psi + psi_dot * dt)
            cp2 = math.cos(psi + psi_dot * dt)
        v_n = v
        psi_n = psi + psi_dot * dt
        psi_dot_n = psi_dot
        x_n = np.array([px_n, py_n, v_n, psi_n, psi_dot_n])

        # Jacobian wrt state.
        F = np.eye(5)
        if abs(psi_dot) > eps:
            inv = 1.0 / psi_dot
            F[0, 2] = inv * (sp2 - sp)
            F[0, 3] = v * inv * (cp2 - cp)
            F[0, 4] = (v * dt * cp2) / psi_dot - v * (sp2 - sp) / (psi_dot * psi_dot)
            F[1, 2] = inv * (-cp2 + cp)
            F[1, 3] = v * inv * (sp2 - sp)
            F[1, 4] = (v * dt * sp2) / psi_dot - v * (-cp2 + cp) / (psi_dot * psi_dot)
        else:
            F[0, 2] = math.cos(psi) * dt
            F[0, 3] = -v * math.sin(psi) * dt
            F[1, 2] = math.sin(psi) * dt
            F[1, 3] = v * math.cos(psi) * dt
        F[3, 4] = dt

        # Process noise (process drives v through linear sensor, psi_dot through yaw-linear sensor).
        # G * [nu_a, nu_psi_dd]^T -> state-dim white-noise input.
        G = np.zeros((5, 2))
        c, s = math.cos(psi), math.sin(psi)
        G[0, 0] = 0.5 * dt * dt * c
        G[1, 0] = 0.5 * dt * dt * s
        G[2, 0] = dt
        G[3, 1] = 0.5 * dt * dt
        G[4, 1] = dt
        Q_in = np.diag([sa * sa, sd * sd])
        Q = G @ Q_in @ G.T
        P_n = F @ P_mat @ F.T + Q
        return x_n, P_n

    smoothed = np.empty((n, 5))
    R_pos = np.diag([sigma_pos_h_m ** 2, sigma_pos_h_m ** 2])
    R_vel = np.array([[sigma_vel_h_mps ** 2]])  # 1x1 for scalar v measurement
    R_psi = np.array([[(math.radians(10.0)) ** 2]])  # 10 deg yaw R when moving
    R_psi_dot = np.array([[sigma_yaw_rate_radps ** 2]])

    prev_t = pos_t[0]
    gyro_idx = 0
    for i in range(n):
        t = pos_t[i]
        # Sub-step predict at rate sensor rate so we can ingest psi_dot in-between.
        if gyro_t is not None and gyro_z is not None:
            while gyro_idx < len(gyro_t) and gyro_t[gyro_idx] <= t:
                gt = float(gyro_t[gyro_idx])
                dts = max(0.0, gt - prev_t)
                if dts > 0:
                    x, P = _predict(x, P, dts)
                # Update psi_dot from rate sensor reading.
                H = np.zeros((1, 5)); H[0, 4] = 1.0
                S = H @ P @ H.T + R_psi_dot
                K = P @ H.T @ np.linalg.inv(S)
                innov = np.array([gyro_z[gyro_idx] - x[4]])
                x = x + (K @ innov).flatten()
                P = (np.eye(5) - K @ H) @ P
                prev_t = gt
                gyro_idx += 1

        dt = t - prev_t
        prev_t = t
        if dt > 0:
            x, P = _predict(x, P, dt)

        # Signal position update.
        H = np.zeros((2, 5)); H[0, 0] = 1.0; H[1, 1] = 1.0
        S = H @ P @ H.T + R_pos
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (p_enu[i, :2] - H @ x)
        P = (np.eye(5) - K @ H) @ P

        # Signal Rate-signal velocity-magnitude + heading update.
        if math.isfinite(vE[i]) and math.isfinite(vN[i]):
            sp = math.hypot(vN[i], vE[i])
            # Speed measurement (state v).
            H_v = np.zeros((1, 5)); H_v[0, 2] = 1.0
            S = H_v @ P @ H_v.T + R_vel
            K = P @ H_v.T @ np.linalg.inv(S)
            innov = np.array([sp - x[2]])
            x = x + (K @ innov).flatten()
            P = (np.eye(5) - K @ H_v) @ P
            # Heading measurement when moving (atan2 noisy at low speeds).
            if sp > 0.5:
                z_psi = math.atan2(vE[i], vN[i])
                # Wrap innovation to (-pi, pi].
                innov_psi = (z_psi - x[3] + math.pi) % (2 * math.pi) - math.pi
                H_p = np.zeros((1, 5)); H_p[0, 3] = 1.0
                S = H_p @ P @ H_p.T + R_psi
                K = P @ H_p.T @ np.linalg.inv(S)
                x = x + (K @ np.array([innov_psi])).flatten()
                P = (np.eye(5) - K @ H_p) @ P

        smoothed[i] = x

    # Output samples.
    out_t = np.asarray(out_times, dtype=np.float64)
    e_out = np.interp(out_t, pos_t, smoothed[:, 0], left=np.nan, right=np.nan)
    n_out = np.interp(out_t, pos_t, smoothed[:, 1], left=np.nan, right=np.nan)
    u_out = np.interp(out_t, pos_t, p_enu[:, 2], left=np.nan, right=np.nan)
    v_out = np.interp(out_t, pos_t, smoothed[:, 2], left=np.nan, right=np.nan)
    psi_out = np.interp(out_t, pos_t, smoothed[:, 3], left=np.nan, right=np.nan)

    out_rows: list[PosRow] = []
    for i, t in enumerate(out_t):
        if not (math.isfinite(e_out[i]) and math.isfinite(n_out[i])
                and math.isfinite(u_out[i])):
            continue
        lat, lon, h = enu_to_llh(float(e_out[i]), float(n_out[i]), float(u_out[i]))
        if math.isfinite(v_out[i]) and math.isfinite(psi_out[i]):
            ve = float(v_out[i] * math.sin(psi_out[i]))
            vn = float(v_out[i] * math.cos(psi_out[i]))
        else:
            ve = float("nan"); vn = float("nan")
        out_rows.append(PosRow(
            utc_s=float(t),
            lat_deg=lat, lon_deg=lon, h_m=h,
            quality=2,
            vn=vn, ve=ve, vu=float("nan"),
        ))
    return out_rows
