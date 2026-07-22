"""Loosely-coupled Motion sensor/Signal fusion for the data_pipeline pipeline.

Two components:
    1. Complementary-update complementary filter — attitude (quaternion) at ~200 Hz
       Rate sensor integration + gravity correction + Signal heading correction.
    2. 6-state EKF (Local-frame pos + vel) — Motion sensor propagation + 1 Hz Signal updates.
       Primary benefit: smooth sub-Hz interpolation; absolute accuracy
       remains Signal-limited.

Entry point: fuse(imu_rows, pos_rows) -> (fused_pos_rows, attitude_samples)
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .parsers import ImuRow, PosRow

_G = 9.80665  # m/s²


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class AttitudeSample:
    """Fused attitude at one Motion sensor timestamp (body→world ZYX convention)."""

    utc_s: float
    yaw_deg: float    # compass heading [0, 360), North=0, East=90
    pitch_deg: float  # positive = nose up
    roll_deg: float   # positive = right side down
    q: Optional[object] = field(default=None, repr=False, compare=False)
    # q: np.ndarray [w,x,y,z] body→world quaternion; None only for legacy callers


# ─── Quaternion helpers ───────────────────────────────────────────────────────

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product [w,x,y,z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body->world 3x3 rotation matrix from a [w,x,y,z] quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by q (body→world: v_world = q * v_body * q*)."""
    pv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, pv), _qconj(q))[1:]


def _qrot_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q^{-1} (world→body)."""
    return _qrot(_qconj(q), v)


def _quat_from_two_vecs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimum-rotation unit quaternion mapping unit vector u → unit vector v."""
    cross = np.cross(u, v)
    dot = float(np.dot(u, v))
    s = float(np.linalg.norm(cross))
    if s < 1e-9:
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # 180° rotation — pick perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = cross / s
    angle = math.atan2(s, dot)
    ha = angle / 2.0
    sa = math.sin(ha)
    return np.array([math.cos(ha), axis[0]*sa, axis[1]*sa, axis[2]*sa])


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ha = angle_rad / 2.0
    s = math.sin(ha)
    return np.array([math.cos(ha), axis[0]*s, axis[1]*s, axis[2]*s])


def _quat_to_ypr(q: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw_deg, pitch_deg, roll_deg) from body→world ZYX quaternion.

    yaw:   [0, 360), North=0, East=90
    pitch: positive = nose-up
    roll:  positive = right side down
    """
    w, x, y, z = q
    sinp = 2.0 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    siny = 2.0 * (w*z + x*y)
    cosy = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.degrees(math.atan2(siny, cosy)) % 360.0

    sinr = 2.0 * (w*x + y*z)
    cosr = 1.0 - 2.0 * (x*x + y*y)
    roll = math.degrees(math.atan2(sinr, cosr))
    return yaw, pitch, roll


# ─── Attitude filter (Complementary-update complementary) ──────────────────────────────────

def _init_attitude(accel_body: np.ndarray) -> np.ndarray:
    """Initialise quaternion from first linear sensor reading.

    The linear sensor with gravity in body sample when stationary equals
    the specific force pointing away from the sphere centre, i.e. it
    reports the direction of world Z (up) in body coordinates.
    Yaw is set to 0 (North) and corrected once the vehicle moves.
    """
    norm = float(np.linalg.norm(accel_body))
    if norm < 1.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a_hat = accel_body / norm
    # q s.t. _qrot_inv(q, [0,0,1]) = a_hat  (a_hat = world Z in body sample)
    q = _quat_from_two_vecs(a_hat, np.array([0.0, 0.0, 1.0]))
    return q / np.linalg.norm(q)


def _snap_pitch_roll_from_gravity(q: np.ndarray, a_body_mean: np.ndarray) -> np.ndarray:
    """Reset pitch/roll from averaged gravity measurement; keep yaw from q.

    Used at static stops to hard-correct pitch/roll drift without disturbing yaw.
    """
    norm_a = float(np.linalg.norm(a_body_mean))
    if norm_a < 1.0:
        return q
    q_grav = _init_attitude(a_body_mean)          # correct pitch/roll, yaw≈arbitrary
    yaw_deg, _, _ = _quat_to_ypr(q)
    grav_yaw_deg, _, _ = _quat_to_ypr(q_grav)
    delta_rad = math.radians(yaw_deg - grav_yaw_deg)
    q_yaw = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), delta_rad)
    q_new = _qmul(q_yaw, q_grav)
    return q_new / float(np.linalg.norm(q_new))


def _mahony_step(
    q: np.ndarray,
    gyro_bias: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    dt: float,
    Kp: float,
    Ki: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Complementary-update filter step (pitch/roll only). Returns (q_new, gyro_bias_new).

    Yaw is handled separately by hard-snapping to Signal Rate-signal heading in
    run_mahony — this step only integrates rate sensor + gravity correction.
    """
    # Gravity correction (pitch/roll): only when linear sensor ≈ 1 g (quasi-static).
    # Skipping during high dynamics prevents false "gravity" from car acceleration.
    norm_a = float(np.linalg.norm(accel))
    gyro_mag = float(np.linalg.norm(gyro))
    if abs(norm_a - _G) < 2.0 and gyro_mag < 2.0:
        a_hat = accel / norm_a
        up_body = _qrot_inv(q, np.array([0.0, 0.0, 1.0]))
        e_grav = np.cross(a_hat, up_body)
    else:
        e_grav = np.zeros(3)

    gyro_bias_new = gyro_bias + Ki * e_grav * dt
    gyro_corr = gyro + Kp * e_grav + gyro_bias_new

    omega = np.array([0.0, gyro_corr[0], gyro_corr[1], gyro_corr[2]])
    q_new = q + 0.5 * _qmul(q, omega) * dt
    q_new /= np.linalg.norm(q_new)
    return q_new, gyro_bias_new


def _snap_yaw_to_heading(q: np.ndarray, heading_rad: float) -> np.ndarray:
    """Hard-set the yaw component of q to heading_rad, keeping pitch/roll."""
    cur_yaw_rad = math.radians(_quat_to_ypr(q)[0])
    delta = (heading_rad - cur_yaw_rad + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]
    q_yaw = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), delta)
    q_new = _qmul(q_yaw, q)
    return q_new / float(np.linalg.norm(q_new))


def run_mahony(
    imu_rows: list[ImuRow],
    pos_rows: list[PosRow],
    Kp: float = 2.0,
    Ki: float = 0.005,
) -> tuple[list[AttitudeSample], list[np.ndarray]]:
    """Run Complementary-update filter on full Motion sensor sequence with Signal-Rate-signal-only yaw.

    Pitch/roll: Complementary-update gravity correction (rate sensor-integrated, gravity-corrected).
    Yaw: hard-snapped to Signal Rate-signal heading at every Motion sensor step while moving
         (speed > 0.5 m/s). When stopped, yaw integrates freely from rate sensor.
    This makes yaw entirely Rate-signal-derived when the vehicle is moving.

    Returns (attitude_samples, quaternions) both parallel to imu_rows.
    """
    from .parsers import detect_static_periods

    if not imu_rows:
        return [], []

    imu_times = [r.utc_s for r in imu_rows]
    pos_times = [r.utc_s for r in pos_rows]

    def _interp_vel(t: float) -> tuple[float, float]:
        # Post-processing rows may be empty if caller passed an empty list; refuse
        # to index then. The Complementary-update loop falls back to rate sensor-only when
        # this returns NaN.
        if not pos_rows:
            return float("nan"), float("nan")
        i = bisect_left(pos_times, t)
        if i == 0:
            r = pos_rows[0]
            return r.vn, r.ve
        if i >= len(pos_rows):
            r = pos_rows[-1]
            return r.vn, r.ve
        r0, r1 = pos_rows[i - 1], pos_rows[i]
        alpha = (t - r0.utc_s) / max(r1.utc_s - r0.utc_s, 1e-9)
        return (r0.vn + alpha * (r1.vn - r0.vn),
                r0.ve + alpha * (r1.ve - r0.ve))

    def _avg_accel_in_window(t_start: float, t_end: float) -> Optional[np.ndarray]:
        i0 = bisect_left(imu_times, t_start)
        i1 = bisect_left(imu_times, t_end)
        window = imu_rows[i0:i1]
        if len(window) < 5:
            return None
        ax = sum(r.ax for r in window) / len(window)
        ay = sum(r.ay for r in window) / len(window)
        az = sum(r.az for r in window) / len(window)
        return np.array([ax, ay, az])

    # Compute static periods from Signal for gravity anchoring
    static_periods = detect_static_periods(pos_rows) if pos_rows else []

    # Build gravity snap schedule: one per static period
    snap_schedule: dict[int, np.ndarray] = {}  # imu_idx → mean linear sensor at that stop
    for sp_start, sp_end in static_periods:
        a_mean = _avg_accel_in_window(sp_start, sp_end)
        if a_mean is None:
            continue
        mid_imu_idx = bisect_left(imu_times, (sp_start + sp_end) / 2.0)
        snap_schedule[mid_imu_idx] = a_mean

    # Choose initialization point: prefer first static stop (device settled)
    # to avoid corrupted attitude from device-mounting motion at session start.
    init_imu_idx = 0
    if static_periods:
        sp_start, sp_end = static_periods[0]
        a_init = _avg_accel_in_window(sp_start, sp_end)
        if a_init is not None and float(np.linalg.norm(a_init)) > 5.0:
            init_imu_idx = bisect_left(imu_times, sp_start)
            q = _init_attitude(a_init)
        else:
            q = _init_attitude(np.array([imu_rows[0].ax, imu_rows[0].ay, imu_rows[0].az]))
    else:
        q = _init_attitude(np.array([imu_rows[0].ax, imu_rows[0].ay, imu_rows[0].az]))

    gyro_bias = np.zeros(3)

    # Bootstrap yaw from first Signal velocity > 1 m/s at or after init point
    init_t = imu_rows[init_imu_idx].utc_s
    for pr in pos_rows:
        if pr.utc_s < init_t:
            continue
        if math.isfinite(pr.vn) and math.isfinite(pr.ve):
            spd = math.sqrt(pr.vn**2 + pr.ve**2)
            if spd > 1.0:
                heading_rad = math.atan2(pr.ve, pr.vn)
                cur_yaw_rad = math.radians(_quat_to_ypr(q)[0])
                # Wrap delta to (-π, π] so the bootstrap rotation always
                # takes the shortest path. Without wrap, a 350° heading
                # vs 10° current yaw would rotate +340° instead of -20°.
                delta = heading_rad - cur_yaw_rad
                delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
                q_yaw = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), delta)
                q = _qmul(q_yaw, q)
                q /= np.linalg.norm(q)
                break

    att_samples: list[AttitudeSample] = []
    quats: list[np.ndarray] = []

    # Pre-fill samples before init point with the initial quaternion
    q_pre = q.copy()
    prev_t = imu_rows[init_imu_idx].utc_s
    for row in imu_rows[:init_imu_idx]:
        yaw, pitch, roll = _quat_to_ypr(q_pre)
        _qc = q_pre.copy()
        att_samples.append(AttitudeSample(row.utc_s, yaw, pitch, roll, q=_qc))
        quats.append(_qc)

    for idx, row in enumerate(imu_rows[init_imu_idx:], start=init_imu_idx):
        # Gravity re-anchor at static stop midpoints
        if idx in snap_schedule:
            q = _snap_pitch_roll_from_gravity(q, snap_schedule[idx])
            gyro_bias = np.zeros(3)  # reset bias after hard correction

        t = row.utc_s
        dt = t - prev_t
        prev_t = t

        if dt > 0:
            accel = np.array([row.ax, row.ay, row.az])
            gyro = np.array([row.gx, row.gy, row.gz])
            # Euler-step quaternion integration is unstable for large dt
            # (~57° per step at 360 deg/s × 0.5 s). Subdivide into substeps
            # of ≤ 0.05 s to keep the small-angle approximation valid no
            # matter how big the upstream Motion sensor gap is. The previous 0.5 s
            # ceiling silently froze attitude during longer sensor gaps,
            # so the substep loop now runs for any positive dt.
            sub_dt_max = 0.05
            n_sub = max(1, int(math.ceil(dt / sub_dt_max)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                q, gyro_bias = _mahony_step(q, gyro_bias, accel, gyro,
                                            sub_dt, Kp, Ki)

            # Yaw: hard-snap to Signal Rate-signal heading when moving.
            # Replaces rate sensor-integrated yaw entirely while speed > 0.5 m/s.
            vn, ve = _interp_vel(t)
            if math.isfinite(vn) and math.isfinite(ve):
                spd = math.sqrt(vn**2 + ve**2)
                if spd > 0.5:
                    q = _snap_yaw_to_heading(q, math.atan2(ve, vn))

        yaw, pitch, roll = _quat_to_ypr(q)
        _qc = q.copy()
        att_samples.append(AttitudeSample(t, yaw, pitch, roll, q=_qc))
        quats.append(_qc)

    return att_samples, quats


# ─── Position EKF ─────────────────────────────────────────────────────────────

def run_position_ekf(
    imu_rows: list[ImuRow],
    pos_rows: list[PosRow],
    quaternions: list[np.ndarray],
    ref_llh: tuple[float, float, float],
    accel_noise_std: float = 0.5,
) -> list[PosRow]:
    """6-state EKF: [de,dn,du,ve,vn,vu] in Local-frame.

    Motion sensor provides the process model (linear sensor integrated in Local-frame).
    Signal provides position + velocity updates at ~1 Hz.
    Returns PosRow entries at Motion sensor rate for the overlapping time window.
    """
    if not imu_rows or not pos_rows:
        return list(pos_rows)

    pos_times = [r.utc_s for r in pos_rows]
    t_gnss_start = pos_times[0]
    t_gnss_end = pos_times[-1]

    # Reference Cartesian XYZ for Local-frame origin
    ref_ecef = llh_to_ecef(*ref_llh)

    def _pos_to_enu(r: PosRow) -> np.ndarray:
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)
        return np.array([ex, ey, ez])

    def _enu_to_llh(enu: np.ndarray) -> tuple[float, float, float]:
        # Approximate inverse: Cartesian XYZ = ref_ECEF + R * local-frame, then Cartesian XYZ→LLH
        rlat = math.radians(ref_llh[0])
        rlon = math.radians(ref_llh[1])
        sl, cl = math.sin(rlat), math.cos(rlat)
        so, co = math.sin(rlon), math.cos(rlon)
        # Local-frame→Cartesian XYZ rotation (transpose of Cartesian XYZ→Local-frame)
        dx = -so*enu[0] - sl*co*enu[1] + cl*co*enu[2]
        dy =  co*enu[0] - sl*so*enu[1] + cl*so*enu[2]
        dz =             cl*enu[1]      + sl*enu[2]
        x = ref_ecef[0] + dx
        y = ref_ecef[1] + dy
        z = ref_ecef[2] + dz
        # Cartesian XYZ→LLH (iterative Bowring)
        p = math.sqrt(x*x + y*y)
        lon = math.atan2(y, x)
        from .geo import _A, _E2
        lat = math.atan2(z, p * (1.0 - _E2))
        for _ in range(5):
            sl_ = math.sin(lat)
            n = _A / math.sqrt(1.0 - _E2 * sl_**2)
            lat = math.atan2(z + _E2 * n * sl_, p)
        sl_ = math.sin(lat)
        n = _A / math.sqrt(1.0 - _E2 * sl_**2)
        h = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-9 else abs(z) / sl_ - n * (1.0 - _E2)
        return math.degrees(lat), math.degrees(lon), h

    # Initialise EKF at first Signal epoch
    r0 = pos_rows[0]
    x = np.zeros(6)
    x[0:3] = _pos_to_enu(r0)
    if math.isfinite(r0.ve) and math.isfinite(r0.vn) and math.isfinite(r0.vu):
        x[3] = r0.ve
        x[4] = r0.vn
        x[5] = r0.vu

    P = np.diag([9.0, 9.0, 225.0, 1.0, 1.0, 4.0])  # generous init

    F6 = np.eye(6)  # will fill dt each step
    gnss_idx = 1    # next Signal row to apply as update

    fused: list[PosRow] = []
    prev_t: Optional[float] = None
    imu_q = {row.utc_s: q for row, q in zip(imu_rows, quaternions)}

    # Process noise per sample: position grows with dt^4/4, velocity with dt^2
    q_vel = accel_noise_std ** 2

    for row in imu_rows:
        t = row.utc_s
        if t < t_gnss_start:
            prev_t = t
            continue
        if t > t_gnss_end + 5.0:
            break

        dt = (t - prev_t) if prev_t is not None else 0.0
        prev_t = t

        # ── Predict ──
        if 0 < dt <= 2.0:
            q_att = imu_q.get(t)
            if q_att is not None:
                # Rotate body linear sensor to Local-frame, subtract gravity
                a_body = np.array([row.ax, row.ay, row.az])
                a_enu = _qrot(q_att, a_body) - np.array([0.0, 0.0, _G])
            else:
                a_enu = np.zeros(3)

            F6[0, 3] = dt; F6[1, 4] = dt; F6[2, 5] = dt
            B = np.zeros((6, 3))
            h2 = 0.5 * dt * dt
            B[0, 0] = h2; B[1, 1] = h2; B[2, 2] = h2
            B[3, 0] = dt; B[4, 1] = dt; B[5, 2] = dt

            x = F6 @ x + B @ a_enu

            # Process noise (simplified diagonal)
            q_pos = q_vel * dt**4 / 4.0
            q_v   = q_vel * dt**2
            Q = np.diag([q_pos, q_pos, q_pos * 4, q_v, q_v, q_v * 4])
            P = F6 @ P @ F6.T + Q

        # ── Update with Signal if a fix is available ──
        while gnss_idx < len(pos_rows) and pos_rows[gnss_idx].utc_s <= t:
            gr = pos_rows[gnss_idx]
            gnss_idx += 1

            z_pos = _pos_to_enu(gr)
            z = np.zeros(6)
            z[0:3] = z_pos
            has_vel = math.isfinite(gr.ve) and math.isfinite(gr.vn) and math.isfinite(gr.vu)
            if has_vel:
                z[3] = gr.ve; z[4] = gr.vn; z[5] = gr.vu

            # Fixed measurement noise (device Post-processing float accuracy)
            r_pos = np.array([9.0, 9.0, 225.0])  # 3 m horiz, 15 m vert
            r_vel = np.array([0.09, 0.09, 0.25]) if has_vel else np.array([1e6, 1e6, 1e6])
            R = np.diag(np.concatenate([r_pos, r_vel]))

            # Recursive-filter update (H = I_6)
            S = P + R
            K = P @ np.linalg.inv(S)
            innov = z - x
            if not has_vel:
                innov[3:] = 0.0
                K[:, 3:] = 0.0
            x = x + K @ innov
            P = (np.eye(6) - K) @ P

        # ── Emit fused PosRow ──
        lat, lon, h = _enu_to_llh(x[0:3])
        fused.append(PosRow(
            utc_s=t,
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            quality=2,
            vn=float(x[4]),
            ve=float(x[3]),
            vu=float(x[5]),
        ))

    return fused if fused else list(pos_rows)


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_position_ekf_v2(
    imu_rows: list[ImuRow],
    pos_rows: list[PosRow],
    quaternions: list[np.ndarray],
    ref_llh: tuple[float, float, float],
    *,
    accel_noise_std: float = 0.5,
    bias_random_walk_std: float = 0.001,
    sigma_pos_h_m: float = 3.0,
    sigma_pos_z_m: float = 15.0,
    sigma_vel_h_mps: float = 1.0,
    sigma_vel_z_mps: float = 2.0,
    zupt_speed_mps: float = 0.3,
    sigma_zupt_mps: float = 0.05,
) -> list[PosRow]:
    """9-state EKF with online linear sensor-bias estimation + ZUPT.

    State (in Local-frame, linear sensor bias in body sample):
        x = [e_E, e_N, e_U, v_E, v_N, v_U, b_ax, b_ay, b_az]^T

    Process model integrates linear sensor after removing estimated bias:
        a_enu(t) = R(q_t) · (a_body(t) − b_a) − g_enu
    Bias is a slow random walk: db_a/dt = w_b,  w_b ~ N(0, sigma_bw²).

    Measurement model at each Post-processing epoch fuses *position* (R_p from
    sigma_pos_*) and *Rate-signal velocity* (R_v from sigma_vel_*). When the
    Rate-signal-measured speed drops below ``zupt_speed_mps`` a tight
    zero-velocity update is injected with ``sigma_zupt_mps`` — that
    forces the linear sensor-bias state to absorb whatever drift accumulated
    during the prior segment, since vehicle linear sensor is structurally zero
    while parked.

    Compared to :func:`run_position_ekf` (6-state), v2 separates Motion sensor
    drift from velocity error: bias is estimated rather than projected
    into velocity noise, so position integration stays stable across
    long Post-processing gaps once a few stops have been observed.
    """
    if not imu_rows or not pos_rows:
        return list(pos_rows)

    pos_times = [r.utc_s for r in pos_rows]
    t_gnss_start = pos_times[0]
    t_gnss_end = pos_times[-1]

    ref_ecef = llh_to_ecef(*ref_llh)
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)

    def _pos_to_enu(r: PosRow) -> np.ndarray:
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)
        return np.array([ex, ey, ez])

    def _enu_to_llh(enu: np.ndarray) -> tuple[float, float, float]:
        dx = -so * enu[0] - sl * co * enu[1] + cl * co * enu[2]
        dy =  co * enu[0] - sl * so * enu[1] + cl * so * enu[2]
        dz =                cl      * enu[1] + sl      * enu[2]
        x = ref_ecef[0] + dx
        y = ref_ecef[1] + dy
        z = ref_ecef[2] + dz
        p = math.sqrt(x * x + y * y)
        lon = math.atan2(y, x)
        from .geo import _A, _E2
        lat = math.atan2(z, p * (1.0 - _E2))
        for _ in range(5):
            sl_ = math.sin(lat)
            n = _A / math.sqrt(1.0 - _E2 * sl_ * sl_)
            lat = math.atan2(z + _E2 * n * sl_, p)
        sl_ = math.sin(lat)
        n = _A / math.sqrt(1.0 - _E2 * sl_ * sl_)
        h = (p / math.cos(lat) - n) if abs(math.cos(lat)) > 1e-9 \
            else (abs(z) / sl_ - n * (1.0 - _E2))
        return math.degrees(lat), math.degrees(lon), h

    r0 = pos_rows[0]
    x = np.zeros(9)
    x[0:3] = _pos_to_enu(r0)
    if math.isfinite(r0.ve) and math.isfinite(r0.vn) and math.isfinite(r0.vu):
        x[3] = r0.ve; x[4] = r0.vn; x[5] = r0.vu
    # Bias states init to zero; converge from data.

    P = np.diag([
        9.0, 9.0, 225.0,
        1.0, 1.0, 4.0,
        0.25, 0.25, 0.25,
    ])

    imu_q = {row.utc_s: q for row, q in zip(imu_rows, quaternions)}
    g_enu = np.array([0.0, 0.0, _G])
    fused: list[PosRow] = []
    prev_t: Optional[float] = None
    gnss_idx = 1
    q_a = accel_noise_std ** 2
    q_b = bias_random_walk_std ** 2

    # Pre-allocate working matrices.
    F = np.eye(9)
    Q = np.zeros((9, 9))
    H_full = np.zeros((6, 9))
    H_full[0:3, 0:3] = np.eye(3)
    H_full[3:6, 3:6] = np.eye(3)
    H_pos = H_full[0:3]
    H_zupt = H_full[3:6]
    R_zupt = np.diag([
        sigma_zupt_mps ** 2, sigma_zupt_mps ** 2, sigma_zupt_mps ** 2,
    ])

    for row in imu_rows:
        t = row.utc_s
        if t < t_gnss_start:
            prev_t = t
            continue
        if t > t_gnss_end + 5.0:
            break

        dt = (t - prev_t) if prev_t is not None else 0.0
        prev_t = t

        # ── Predict ──
        if 0 < dt <= 2.0:
            q_att = imu_q.get(t)
            if q_att is not None:
                # Body→world rotation matrix from quaternion.
                R_bw = _quat_to_rotmat(q_att)
                a_body = np.array([row.ax, row.ay, row.az])
                a_corr = a_body - x[6:9]
                a_enu = R_bw @ a_corr - g_enu
            else:
                R_bw = np.eye(3)
                a_enu = np.zeros(3)

            # State transition F.
            F[:] = np.eye(9)
            F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
            h2 = 0.5 * dt * dt
            # Position depends on bias via -0.5·R·dt² (negative because
            # measured linear sensor is reduced by bias before integrating).
            F[0:3, 6:9] = -h2 * R_bw
            F[3:6, 6:9] = -dt * R_bw

            # Predicted state.
            B_pos = h2 * a_enu
            B_vel = dt * a_enu
            x_new = F @ x
            x_new[0:3] += B_pos - F[0:3, 6:9] @ x[6:9]
            x_new[3:6] += B_vel - F[3:6, 6:9] @ x[6:9]
            x = x_new

            # Process noise: linear sensor drives pos & vel; bias has tiny RW.
            Q[:] = 0.0
            q_pos = q_a * (dt ** 4) / 4.0
            q_vel = q_a * (dt ** 2)
            Q[0, 0] = q_pos; Q[1, 1] = q_pos; Q[2, 2] = q_pos * 4
            Q[3, 3] = q_vel; Q[4, 4] = q_vel; Q[5, 5] = q_vel * 4
            Q[6, 6] = q_b * dt; Q[7, 7] = q_b * dt; Q[8, 8] = q_b * dt
            P = F @ P @ F.T + Q

        # ── Update with Signal if a fix is available ──
        while gnss_idx < len(pos_rows) and pos_rows[gnss_idx].utc_s <= t:
            gr = pos_rows[gnss_idx]
            gnss_idx += 1
            z_pos = _pos_to_enu(gr)
            has_vel = (math.isfinite(gr.ve) and math.isfinite(gr.vn)
                       and math.isfinite(gr.vu))

            if has_vel:
                speed = math.sqrt(gr.vn ** 2 + gr.ve ** 2)
            else:
                speed = float("nan")

            # Position update.
            R_p = np.diag([
                sigma_pos_h_m ** 2, sigma_pos_h_m ** 2, sigma_pos_z_m ** 2,
            ])
            S_p = H_pos @ P @ H_pos.T + R_p
            K_p = P @ H_pos.T @ np.linalg.inv(S_p)
            innov_p = z_pos - x[0:3]
            x = x + K_p @ innov_p
            P = (np.eye(9) - K_p @ H_pos) @ P

            # Velocity update from Rate-signal if available.
            if has_vel:
                z_v = np.array([gr.ve, gr.vn, gr.vu])
                # Looser R when speed is high (signal vel still good) -- here
                # we just use the user-spec sigmas regardless.
                R_v = np.diag([
                    sigma_vel_h_mps ** 2, sigma_vel_h_mps ** 2,
                    sigma_vel_z_mps ** 2,
                ])
                S_v = H_zupt @ P @ H_zupt.T + R_v
                K_v = P @ H_zupt.T @ np.linalg.inv(S_v)
                innov_v = z_v - x[3:6]
                x = x + K_v @ innov_v
                P = (np.eye(9) - K_v @ H_zupt) @ P

                # ZUPT: tight v=0 anchor when Rate-signal says we're stopped.
                if math.isfinite(speed) and speed < zupt_speed_mps:
                    S_z = H_zupt @ P @ H_zupt.T + R_zupt
                    K_z = P @ H_zupt.T @ np.linalg.inv(S_z)
                    innov_z = -x[3:6]
                    x = x + K_z @ innov_z
                    P = (np.eye(9) - K_z @ H_zupt) @ P

        lat, lon, h = _enu_to_llh(x[0:3])
        fused.append(PosRow(
            utc_s=t,
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            quality=2,
            vn=float(x[4]),
            ve=float(x[3]),
            vu=float(x[5]),
        ))

    return fused if fused else list(pos_rows)


def fuse(
    imu_rows: list[ImuRow],
    pos_rows: list[PosRow],
    log: Optional[object] = None,
) -> tuple[list[PosRow], list[AttitudeSample]]:
    """Run Motion sensor/Signal fusion on one session.

    Returns:
        pos_rows:  Original Signal positions unchanged.  Position accuracy is
                   limited by Signal quality; source-grade Motion sensor introduces more
                   error than it removes via gravity-subtraction inaccuracies.
                   The path smoothing in the calling pipeline is sufficient.
        attitude:  AttitudeSample list at Motion sensor rate (~200 Hz) from the Complementary-update
                   complementary filter.  This replaces the rate sensor-derived
                   orientation and gives continuous pitch/roll/yaw with no
                   long-term drift.
    """
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)  # type: ignore[operator]

    if not imu_rows:
        _log("[fusion] no IMU data — skipping attitude fusion")
        return list(pos_rows), []

    if not pos_rows:
        _log("[fusion] no GNSS data — cannot fuse")
        return [], []

    _log(f"[fusion] IMU: {len(imu_rows)} samples "
         f"({imu_rows[0].utc_s:.1f}–{imu_rows[-1].utc_s:.1f})")
    _log(f"[fusion] GNSS: {len(pos_rows)} rows "
         f"({pos_rows[0].utc_s:.1f}–{pos_rows[-1].utc_s:.1f})")

    att_samples, _ = run_mahony(imu_rows, pos_rows)
    _log(f"[fusion] Mahony attitude: {len(att_samples)} samples")

    return list(pos_rows), att_samples
