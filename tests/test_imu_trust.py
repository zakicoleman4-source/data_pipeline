import math
from types import SimpleNamespace

import numpy as np

from data_pipeline.imu_trust import compute_imu_trust
from data_pipeline.parsers import ImuRow


AFFINE = {"a": 0.0, "b": 1.0}  # t_video == utc for tests

# Shared, physically-consistent turn: heading(t) = AMP*sin(W*t), so the
# turn-rate varies over time (non-degenerate correlation) and peaks at
# AMP*W = ~12 deg/s, well above the 3 deg/s gate. The Motion sensor rate sensor-z is exactly
# the derivative of that heading, so pos-derived turn and rate sensor yaw-rate match.
_AMP = 1.0                     # rad (~57 deg heading swing)
_W = 2.0 * math.pi / 30.0      # rad/s, 30 s period


def _pos_turn(t0=0.0, n=61, dt=0.5):
    """Vehicle driving a sinusoidal heading sweep over ~30 s."""
    rows = []
    for i in range(n):
        tt = i * dt
        head = _AMP * math.sin(_W * tt)            # Local-frame heading, rad
        spd = 5.0                                  # m/s, well above the 0.3 gate
        vn = spd * math.cos(head)
        ve = spd * math.sin(head)
        rows.append(SimpleNamespace(utc_s=t0 + tt, vn=vn, ve=ve))
    return rows


def _imu(n=300, dt=0.1, gz_scale=1.0, t0=0.0):
    """Motion sensor flat on a table (gravity on +z); rate sensor-z = d/dt heading (rad/s)."""
    rows = []
    for i in range(n):
        tt = i * dt
        gz = _AMP * _W * math.cos(_W * tt)         # derivative of the heading
        rows.append(ImuRow(
            utc_s=t0 + tt,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=gz * gz_scale,
        ))
    return rows


def test_live_gyro_correlates_and_is_alive():
    out = compute_imu_trust(_imu(), _pos_turn(), AFFINE)
    assert out is not None
    assert out["flags"]["gyro_dead"] is False
    assert out["flags"]["corr"] is not None
    assert out["flags"]["corr"] > 0.8
    assert out["flags"]["sign"] in (+1, -1)
    # strip decimated, arrays aligned
    assert len(out["t_video"]) == len(out["yaw_meas_dps"]) == len(out["accel_mag"])


def test_sign_autoaligned_for_flipped_gyro():
    out = compute_imu_trust(_imu(gz_scale=-1.0), _pos_turn(), AFFINE)
    assert out["flags"]["corr"] > 0.8           # correlation stays high after sign flip
    assert out["flags"]["sign"] == -1


def test_dead_gyro_flagged():
    dead = [ImuRow(utc_s=0.1 * i, ax=0.0, ay=0.0, az=9.81,
                   gx=0.0, gy=0.0, gz=0.0) for i in range(600)]
    out = compute_imu_trust(dead, _pos_turn(), AFFINE)
    assert out["flags"]["gyro_dead"] is True


def test_none_when_no_imu():
    assert compute_imu_trust([], _pos_turn(), AFFINE) is None


def test_box_head_absolute_and_anchored():
    # Box heading sweeps with the turn AND is anchored to the path heading
    # (north-referenced), so early box heading ~ early path heading.
    out = compute_imu_trust(_imu(), _pos_turn(), AFFINE)
    bh = [v for v in out["box_head_deg"] if v is not None]
    th = out["traj_head_deg"]
    assert len(bh) > 2
    assert bh[-1] != bh[0]                       # box actually rotated
    # anchor: at the first sample where both exist, they should be close
    pairs = [(b, t) for b, t in zip(out["box_head_deg"], th)
             if b is not None and t is not None]
    assert pairs
    b0, t0 = pairs[0]
    assert abs(b0 - t0) < 1.0                    # anchored to truth


def _imu_accel(n=400, dt=0.1, fwd_axis=0, brake_at=None):
    """Device flat (gravity +z). A forward linear sensor pulse along device +x (fwd_axis)."""
    rows = []
    for i in range(n):
        tt = i * dt
        # accelerate 0-10 s, coast 10-20 s, brake 20-30 s along device +x
        if tt < 10:
            lon = 1.5
        elif tt < 20:
            lon = 0.0
        else:
            lon = -1.5
        acc = [0.0, 0.0, 9.81]
        acc[fwd_axis] += lon
        rows.append(ImuRow(utc_s=tt, ax=acc[0], ay=acc[1], az=acc[2],
                           gx=0.0, gy=0.0, gz=0.0))
    return rows


def _pos_speed(n=60, dt=0.5):
    """Vehicle speed ramps up then down, straight line (heading due north)."""
    rows = []
    for i in range(n):
        tt = i * dt
        if tt < 10:
            spd = 0.5 * tt            # accelerating
        elif tt < 20:
            spd = 5.0                 # constant
        else:
            spd = max(0.0, 5.0 - 0.5 * (tt - 20))  # braking
        rows.append(SimpleNamespace(utc_s=tt, vn=spd, ve=0.0))
    return rows


def test_mounting_yaw_resolves_forward_accel():
    out = compute_imu_trust(_imu_accel(fwd_axis=0), _pos_speed(), AFFINE)
    assert out["flags"]["mount_resolved"] is True
    assert out["flags"]["mount_conf"] > 0.5
    # forward linear sensor present and positive during the initial acceleration phase
    fwd = out["fwd_accel"]
    tv = out["t_video"]
    early = [f for f, tt in zip(fwd, tv) if f is not None and 2 < tt < 8]
    assert early and sum(early) / len(early) > 0.3


def _pos_turn_coords(t0=0.0, n=61, dt=0.5):
    """Same sinusoidal-heading drive but as lat/lon ONLY (vn/ve absent) -> old format."""
    rows = []
    lat0, lon0 = 32.06, 34.79
    mlat = 111320.0
    mlon = 111320.0 * math.cos(math.radians(lat0))
    e = 0.0; n_m = 0.0; prev = None
    for i in range(n):
        tt = i * dt
        head = _AMP * math.sin(_W * tt)
        spd = 5.0
        vn = spd * math.cos(head); ve = spd * math.sin(head)
        if prev is not None:
            e += ve * dt; n_m += vn * dt
        prev = tt
        rows.append(SimpleNamespace(utc_s=t0 + tt,
                                    lat_deg=lat0 + n_m / mlat,
                                    lon_deg=lon0 + e / mlon,
                                    vn=float("nan"), ve=float("nan")))
    return rows


def test_vel_source_doppler_when_present():
    out = compute_imu_trust(_imu(), _pos_turn(), AFFINE)
    assert out["flags"]["vel_source"] == "doppler"


def test_vel_source_coords_fallback():
    out = compute_imu_trust(_imu(), _pos_turn_coords(), AFFINE)
    assert out["flags"]["vel_source"] == "coords"
    assert out["flags"]["corr"] is not None and out["flags"]["corr"] > 0.6
    th = [v for v in out["traj_head_deg"] if v is not None]
    assert len(th) > 2


def test_vel_source_none_when_no_velocity_or_position():
    rows = [SimpleNamespace(utc_s=0.1 * i, vn=float("nan"), ve=float("nan"))
            for i in range(5)]
    out = compute_imu_trust(_imu(n=50), rows, AFFINE)
    assert out["flags"]["vel_source"] == "none"


def test_verdict_good_for_tracking_gyro():
    out = compute_imu_trust(_imu(), _pos_turn(), AFFINE)
    assert out["flags"]["verdict"] in ("GOOD", "OK")
    assert out["flags"]["drift_max_deg"] is not None
    assert out["flags"]["drift_max_deg"] >= 0.0


def test_verdict_dead_for_flat_gyro():
    dead = [ImuRow(utc_s=0.1 * i, ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0)
            for i in range(600)]
    out = compute_imu_trust(dead, _pos_turn(), AFFINE)
    assert out["flags"]["verdict"] == "DEAD"
