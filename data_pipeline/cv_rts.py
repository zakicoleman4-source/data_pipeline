"""Constant-velocity Recursive filter + RTS smoother — scalar per-axis.

Driving-profile prior: vehicle position evolves as integrated velocity with
acceleration noise. Beats Gaussian smoothing when Post-processing has biased outliers
that a uniform LPF cannot reject.

State: [position, velocity]. Process noise: ``sigma_a`` (m/s^2) drives the
constant-velocity model. Measurement noise: ``sigma_z`` (m) on position.

Pair with ``doppler_gate`` (Rate-signal vs finite-diff Post-processing MAD-gate) to drop
outlier Post-processing epochs before the Recursive-filter pass.
"""
from __future__ import annotations

import math
import numpy as np


def cv_rts_pv(
    z: np.ndarray,
    v: np.ndarray,
    use_v: np.ndarray,
    dt: float,
    sigma_p: float = 4.0,
    sigma_v: float = 0.3,
    sigma_a: float = 0.2,
) -> np.ndarray:
    """Scalar forward-backward Recursive-filter with **pos + vel** measurements + CV.

    State per axis: ``[position, velocity]``.

    Args:
        z      : measured position series (1D float64).
        v      : measured velocity series (Rate-signal) aligned with z.
        use_v  : per-epoch bool gate -- True if the velocity sample at
                  this index is finite and trustworthy.
        dt     : fixed inter-epoch interval (s).
        sigma_p: position measurement 1-sigma (m).
        sigma_v: velocity measurement 1-sigma (m/s).
        sigma_a: constant-acceleration process-noise 1-sigma (m/s^2).

    Returns:
        Smoothed position series (same length as ``z``).

    Notes:
        Ported from ``scripts/fgo_mirror_cv_rts.cv_rts_pv_local`` -- the
        reference implementation used to publish reference session hRMSE 2.416 m.
        Huber-on-residual options were stripped because the production
        callers (smoothers.py, comparison viewer) don't use them.
    """
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    use_v = np.asarray(use_v, dtype=bool)
    nn = z.size
    if nn == 0:
        return np.array([])
    if v.size != nn or use_v.size != nn:
        raise ValueError(
            f"cv_rts_pv: z / v / use_v length mismatch "
            f"({nn} / {v.size} / {use_v.size})")
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"cv_rts_pv: dt must be > 0 (got {dt})")
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                  [dt ** 3 / 2, dt ** 2]])
    Hp = np.array([[1.0, 0.0]])
    Hv = np.array([[0.0, 1.0]])
    Rp = sigma_p ** 2
    Rv = sigma_v ** 2
    v0 = float(v[0]) if use_v[0] else 0.0
    x = np.array([float(z[0]), v0])
    P = np.diag([Rp, Rv if use_v[0] else 1.0])
    x_fwd = np.zeros((nn, 2)); P_fwd = np.zeros((nn, 2, 2))
    x_pred = np.zeros((nn, 2)); P_pred = np.zeros((nn, 2, 2))
    for k in range(nn):
        # Predict
        x_p = F @ x
        P_p = F @ P @ F.T + Q
        x_pred[k] = x_p
        P_pred[k] = P_p
        # Position update
        S = float((Hp @ P_p @ Hp.T)[0, 0] + Rp)
        innov = float(z[k] - (Hp @ x_p)[0])
        K = (P_p @ Hp.T / S).flatten()
        x = x_p + K * innov
        P = P_p - np.outer(K, Hp @ P_p)
        # Velocity update (gated)
        if bool(use_v[k]):
            S = float((Hv @ P @ Hv.T)[0, 0] + Rv)
            innov_v = float(v[k] - (Hv @ x)[0])
            K = (P @ Hv.T / S).flatten()
            x = x + K * innov_v
            P = P - np.outer(K, Hv @ P)
        x_fwd[k] = x
        P_fwd[k] = P
    # RTS backward
    x_sm = x_fwd.copy()
    for k in range(nn - 2, -1, -1):
        try:
            C = P_fwd[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_sm[k] = x_fwd[k] + C @ (x_sm[k + 1] - x_pred[k + 1])
    return x_sm[:, 0]


def cv_rts(z: np.ndarray, dt: float, sigma_z: float = 2.0, sigma_a: float = 0.5) -> np.ndarray:
    """Forward Recursive-filter + backward RTS on a scalar series at fixed dt.

    NaN samples are skipped (predicted-only update). Returns smoothed series.
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    if n == 0:
        return np.array([])
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"cv_rts: dt must be > 0 (got {dt}).")
    if not (math.isfinite(sigma_z) and sigma_z > 0):
        raise ValueError(f"cv_rts: sigma_z must be > 0 (got {sigma_z}).")
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sigma_a ** 2 * np.array([
        [dt ** 4 / 4, dt ** 3 / 2],
        [dt ** 3 / 2, dt ** 2]])
    R = sigma_z ** 2
    H = np.array([[1.0, 0.0]])
    # Find first finite sample for init; if all NaN, seed at 0 with wide P.
    finite_z = np.isfinite(z)
    if not finite_z.any():
        return np.full(n, np.nan)
    i0 = int(np.argmax(finite_z))  # first True
    x = np.array([float(z[i0]), 0.0])
    P = np.diag([sigma_z ** 2, 1.0])
    x_fwd = np.zeros((n, 2)); P_fwd = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2)); P_pred = np.zeros((n, 2, 2))
    for k in range(n):
        x_p = F @ x; P_p = F @ P @ F.T + Q
        x_pred[k] = x_p; P_pred[k] = P_p
        if finite_z[k]:
            S = float((H @ P_p @ H.T)[0, 0]) + R
            K = (P_p @ H.T / S).flatten()
            innov = float(z[k]) - (H @ x_p)[0]
            x = x_p + K * innov
            P = P_p - np.outer(K, H @ P_p)
        else:
            x = x_p; P = P_p
        x_fwd[k] = x; P_fwd[k] = P
    x_sm = x_fwd.copy()
    for k in range(n - 2, -1, -1):
        try:
            C = P_fwd[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_sm[k] = x_fwd[k] + C @ (x_sm[k + 1] - x_pred[k + 1])
    return x_sm[:, 0]


def bad_doppler_mask(
    ve: np.ndarray, vn: np.ndarray, vu: np.ndarray, ts: np.ndarray,
    sd_ve: np.ndarray, sd_vn: np.ndarray,
    *,
    max_speed_mps: float = 100.0,
    max_accel_mps2: float = 15.0,
    max_sd_v_mps: float = 5.0,
) -> np.ndarray:
    """Flag epochs with unreliable Rate-signal velocity.

    Checks:
      1. NaN / non-finite velocity components
      2. Speed exceeds ``max_speed_mps`` (unrealistic for vehicle)
      3. Epoch-to-epoch acceleration exceeds ``max_accel_mps2``
      4. Reported velocity sigma exceeds ``max_sd_v_mps``

    Returns boolean mask (True = bad Rate-signal, should not trust velocity).
    """
    ve = np.asarray(ve, dtype=np.float64)
    vn = np.asarray(vn, dtype=np.float64)
    vu = np.asarray(vu, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ve)
    bad = np.zeros(n, dtype=bool)

    # 1. Non-finite
    bad |= ~np.isfinite(ve) | ~np.isfinite(vn)

    # 2. Speed exceeds limit
    speed = np.sqrt(ve**2 + vn**2)
    speed_finite = np.where(np.isfinite(speed), speed, 0.0)
    bad |= speed_finite > max_speed_mps

    # 3. Acceleration spike (forward difference)
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt <= 0 or not math.isfinite(dt) or dt > 5.0:
            continue
        if not (math.isfinite(ve[i]) and math.isfinite(ve[i-1]) and
                math.isfinite(vn[i]) and math.isfinite(vn[i-1])):
            continue
        dv = math.hypot(ve[i] - ve[i-1], vn[i] - vn[i-1])
        accel = dv / dt
        if accel > max_accel_mps2:
            bad[i] = True

    # 4. High velocity sigma
    sd_h = np.sqrt(np.asarray(sd_ve, dtype=np.float64)**2 +
                   np.asarray(sd_vn, dtype=np.float64)**2)
    bad |= np.where(np.isfinite(sd_h), sd_h > max_sd_v_mps, False)

    return bad


def doppler_gate(
    E: np.ndarray, N: np.ndarray, ve: np.ndarray, vn: np.ndarray, ts: np.ndarray,
    K: float = 3.0,
) -> np.ndarray:
    """Return a boolean ``bad`` mask where Post-processing finite-diff velocity disagrees
    with Rate-signal velocity by more than ``K`` MAD-scaled sigmas.

    Use to identify Post-processing outlier epochs (environment noise / measurement discontinuity jumps) before
    smoothing. Rate-signal velocity is per-epoch fine measurements-derived and
    independent of Post-processing solution refinement, so disagreement isolates Post-processing
    position outliers reliably.
    """
    E = np.asarray(E, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    if E.size < 2:
        return np.zeros(E.size, dtype=bool)
    ve_ppk = np.gradient(E, ts)
    vn_ppk = np.gradient(N, ts)
    d = np.sqrt((ve_ppk - ve) ** 2 + (vn_ppk - vn) ** 2)
    # nanmedian/nanmean so a few NaN epochs don't poison the whole threshold.
    finite_d = np.isfinite(d)
    if not finite_d.any():
        return np.zeros(d.size, dtype=bool)
    med = float(np.median(d[finite_d]))
    mad = float(np.median(np.abs(d[finite_d] - med)))
    if mad < 1e-9:
        return np.zeros(d.size, dtype=bool)
    thr = med + K * 1.4826 * mad
    # NaN epochs in d itself are not flagged (will be handled by interp/skip
    # downstream); only flag epochs whose disagreement exceeds thr.
    return np.where(finite_d, d > thr, False)


def imu_velocity_gate(
    ve: np.ndarray, vn: np.ndarray, ts: np.ndarray,
    imu_rows, quaternions: np.ndarray,
    *,
    calib_max_sd_v: float = 0.1,
    calib_min_epochs: int = 30,
    gate_sigma: float = 3.0,
    sd_ve: np.ndarray | None = None,
    sd_vn: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag Signal velocity outliers using Motion sensor-calibrated acceleration check.

    Phase 1 — calibrate: during high-quality Signal epochs (low velocity sigma),
    compare Signal delta-v with Motion sensor-integrated delta-v to learn the systematic
    linear sensor bias in Local-frame.

    Phase 2 — gate: at every epoch, compare Signal acceleration with
    bias-corrected Motion sensor acceleration.  Flag epochs where the disagreement
    exceeds ``gate_sigma`` times the calibration residual std.

    Returns boolean mask (True = suspect velocity) and per-epoch
    disagreement scores (0 = agrees, higher = worse).
    """
    from .imu_gnss_fusion import _qrot

    n = len(ve)
    bad = np.zeros(n, dtype=bool)
    scores = np.zeros(n)
    if quaternions is None or len(imu_rows) < 10 or n < 3:
        return bad, scores

    imu_ts = np.array([r.utc_s for r in imu_rows])
    GRAVITY = np.array([0.0, 0.0, 9.81])

    # Integrate Motion sensor linear sensor between consecutive Signal epochs → delta_v in Local-frame
    imu_dv_e = np.zeros(n)
    imu_dv_n = np.zeros(n)
    imu_count = np.zeros(n, dtype=int)

    for k in range(1, n):
        i0 = int(np.searchsorted(imu_ts, ts[k - 1]))
        i1 = int(np.searchsorted(imu_ts, ts[k]))
        if i1 - i0 < 2:
            continue
        dv = np.zeros(3)
        for j in range(i0, min(i1 - 1, len(imu_rows) - 1)):
            dt_imu = imu_ts[j + 1] - imu_ts[j]
            if dt_imu <= 0 or dt_imu > 0.1:
                continue
            if j >= len(quaternions):
                continue
            accel_body = np.array([imu_rows[j].ax, imu_rows[j].ay, imu_rows[j].az])
            accel_enu = _qrot(quaternions[j], accel_body) - GRAVITY
            dv += accel_enu * dt_imu
        imu_dv_e[k] = dv[0]
        imu_dv_n[k] = dv[1]
        imu_count[k] = i1 - i0

    # Signal delta-v between consecutive epochs
    gnss_dv_e = np.zeros(n)
    gnss_dv_n = np.zeros(n)
    for k in range(1, n):
        if math.isfinite(ve[k]) and math.isfinite(ve[k-1]):
            gnss_dv_e[k] = ve[k] - ve[k-1]
        if math.isfinite(vn[k]) and math.isfinite(vn[k-1]):
            gnss_dv_n[k] = vn[k] - vn[k-1]

    # Phase 1: calibrate bias from best epochs
    if sd_ve is not None and sd_vn is not None:
        sd_h = np.sqrt(np.asarray(sd_ve, dtype=np.float64)**2 +
                        np.asarray(sd_vn, dtype=np.float64)**2)
    else:
        sd_h = np.full(n, 999.0)

    calib_mask = (
        (imu_count > 10) &
        np.isfinite(sd_h) & (sd_h < calib_max_sd_v) &
        np.isfinite(gnss_dv_e) & np.isfinite(gnss_dv_n)
    )
    calib_idx = np.where(calib_mask)[0]

    if len(calib_idx) < calib_min_epochs:
        return bad, scores

    residual_e = gnss_dv_e[calib_idx] - imu_dv_e[calib_idx]
    residual_n = gnss_dv_n[calib_idx] - imu_dv_n[calib_idx]
    bias_e = float(np.median(residual_e))
    bias_n = float(np.median(residual_n))
    res_after_e = residual_e - bias_e
    res_after_n = residual_n - bias_n
    std_e = float(np.std(res_after_e))
    std_n = float(np.std(res_after_n))

    if std_e < 1e-6 or std_n < 1e-6:
        return bad, scores

    # Phase 2: score all epochs — normalized disagreement with calibrated Motion sensor
    for k in range(1, n):
        if imu_count[k] < 5:
            continue
        if not (math.isfinite(gnss_dv_e[k]) and math.isfinite(gnss_dv_n[k])):
            continue
        err_e = abs(gnss_dv_e[k] - imu_dv_e[k] - bias_e)
        err_n = abs(gnss_dv_n[k] - imu_dv_n[k] - bias_n)
        scores[k] = math.hypot(err_e / max(std_e, 1e-9), err_n / max(std_n, 1e-9))
        if err_e > gate_sigma * std_e or err_n > gate_sigma * std_n:
            bad[k] = True

    return bad, scores


def lin_interp_through(arr: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace ``bad`` epochs by linear interpolation through neighbouring good ones."""
    a = arr.copy()
    if not bad.any() or (~bad).sum() < 2:
        return a
    idx = np.arange(len(arr))
    a[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
    return a


def gate_then_cv(
    E: np.ndarray, N: np.ndarray, ve: np.ndarray, vn: np.ndarray, ts: np.ndarray,
    K: float = 3.0, sigma_z: float = 2.0, sigma_a: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate-signal-gate Post-processing outliers, then CV+RTS smooth E/N separately.

    Recommended for sessions where The external solver produced Differential-quality (q=4) without
    ns reporting — the ns-adaptive smoother has no signal and a uniform
    Gaussian can't reject the biased outliers. Empirically:
      session 4 session-C  11.9 -> 9.29   -21.95 %
      session 3 s25       9.1 -> 8.51    -6.46 %
      session 5 session-C    5.4 -> 4.88    -9.14 %
      (over 6 ns=0 datasets: mean -7.66 %, worst -0.67 %, wins 6/6)
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        raise ValueError(
            f"gate_then_cv: need >= 2 timestamps to estimate dt, got {ts.size}."
        )
    bad = doppler_gate(E, N, ve, vn, ts, K=K)
    Eg = lin_interp_through(E, bad)
    Ng = lin_interp_through(N, bad)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]  # drop duplicate / out-of-order timestamps
    if diffs.size == 0:
        raise ValueError(
            "gate_then_cv: all timestamps are duplicate / non-monotonic; "
            "cannot estimate dt."
        )
    dt = float(np.median(diffs))
    Es = cv_rts(Eg, dt, sigma_z=sigma_z, sigma_a=sigma_a)
    Ns = cv_rts(Ng, dt, sigma_z=sigma_z, sigma_a=sigma_a)
    return Es, Ns
