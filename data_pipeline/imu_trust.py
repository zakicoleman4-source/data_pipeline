"""Compute the sync-player 'sensor-trust' strip payload from raw Motion sensor + path.

Pure numpy. Overlays a gravity-projected raw-rate sensor yaw-rate against the
path turn-rate so a user (and a test) can see whether the rate sensor is
trustworthy: on real turns the two track; a flat rate sensor during a turn is dead.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def _lowpass(a: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """One-pole low-pass along axis 0 (rows are time). Isolates gravity from linear sensor."""
    out = np.empty_like(a)
    acc = a[0].copy()
    for i in range(a.shape[0]):
        acc = acc + alpha * (a[i] - acc)
        out[i] = acc
    return out


def _movavg(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average (window k samples). k<2 is a no-op."""
    if k < 2 or x.size < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _effective_velocity(pos_rows, smooth_s):
    """Return (pt, vn, ve, vel_source). Use Rate-signal vn/ve when present, else
    derive from consecutive lat/lon positions (equirectangular Local-frame + d/dt)."""
    pt = np.array([getattr(r, "utc_s", np.nan) for r in pos_rows], dtype=np.float64)
    vn = np.array([getattr(r, "vn", np.nan) for r in pos_rows], dtype=np.float64)
    ve = np.array([getattr(r, "ve", np.nan) for r in pos_rows], dtype=np.float64)
    finite = np.isfinite(vn) & np.isfinite(ve)
    if pt.size >= 3 and float(finite.mean()) >= 0.5:
        return pt, vn, ve, "doppler"
    lat = np.array([getattr(r, "lat_deg", np.nan) for r in pos_rows], dtype=np.float64)
    lon = np.array([getattr(r, "lon_deg", np.nan) for r in pos_rows], dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pt)
    if int(ok.sum()) < 3:
        return pt, vn, ve, "none"
    latf, lonf, ptf = lat[ok], lon[ok], pt[ok]
    lat0, lon0 = float(latf[0]), float(lonf[0])
    mlat = 111320.0
    mlon = 111320.0 * math.cos(math.radians(lat0))
    east = (lonf - lon0) * mlon
    north = (latf - lat0) * mlat
    # Light smoothing only (a couple of samples): differentiating positions
    # already suppresses low-frequency drift, and the caller re-smooths the
    # derived heading with the full turn_smooth_s window before computing
    # turn-rate. Smoothing this raw position trace by the full window first
    # would double-smooth and phase-lag the heading, hurting correlation.
    dt = float(np.median(np.diff(ptf))) if ptf.size > 1 else 1.0
    k = max(1, int(round(min(smooth_s, 2 * dt) / dt))) if dt > 0 else 1
    east = _movavg(east, k)
    north = _movavg(north, k)
    ve_d = np.gradient(east, ptf)
    vn_d = np.gradient(north, ptf)
    return pt, np.interp(pt, ptf, vn_d), np.interp(pt, ptf, ve_d), "coords"


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3:
        return None
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom <= 0:
        return None
    return float((xm * ym).sum() / denom)


def compute_imu_trust(
    imu_rows: Sequence,
    pos_rows: Sequence,
    video_utc_affine: dict,
    *,
    strip_hz: float = 25.0,
    turn_thresh_dps: float = 3.0,
    dead_std_radps: float = 1e-4,
    turn_smooth_s: float = 5.0,
    accel_thresh_mps2: float = 0.3,
    mount_conf_thresh: float = 0.3,
) -> Optional[dict]:
    if imu_rows is None or len(imu_rows) == 0:
        return None

    t = np.array([r.utc_s for r in imu_rows], dtype=np.float64)
    g = np.array([[r.gx, r.gy, r.gz] for r in imu_rows], dtype=np.float64)
    a = np.array([[r.ax, r.ay, r.az] for r in imu_rows], dtype=np.float64)

    order = np.argsort(t)
    t, g, a = t[order], g[order], a[order]

    # gravity-projected yaw-rate: rate sensor . gravity_unit  (mount-agnostic)
    grav = _lowpass(a)
    gn = np.linalg.norm(grav, axis=1, keepdims=True)
    gn[gn < 1e-6] = 1e-6
    gu = grav / gn
    yaw_meas = np.sum(g * gu, axis=1)                 # rad/s
    yaw_meas_dps = np.degrees(yaw_meas)
    accel_mag = np.linalg.norm(a, axis=1)

    gyro_std = float(np.std(g))
    n_unique = int(np.unique(g.round(9)).size)
    gyro_dead = gyro_std < dead_std_radps or n_unique <= 1

    # path turn-rate = d/dt of unwrapped Rate-signal heading (deg/s)
    pt, vn, ve, vel_source = _effective_velocity(pos_rows, turn_smooth_s)
    spd = np.hypot(vn, ve)
    good = spd > 0.3
    turn_on_t = np.full_like(t, np.nan)
    if good.sum() >= 3:
        tp = pt[good]
        head = np.unwrap(np.arctan2(ve[good], vn[good]))       # rad
        # Smooth the heading before differentiating: raw per-epoch Rate-signal
        # heading is noise-dominated, and d/dt amplifies it. Real vehicle turns
        # last seconds, so a few-second average keeps turns and kills the noise.
        dt_pos = float(np.median(np.diff(tp))) if tp.size > 1 else 1.0
        k = max(1, int(round(turn_smooth_s / dt_pos))) if dt_pos > 0 else 1
        head = _movavg(head, k)
        tr = np.degrees(np.gradient(head, tp))                 # deg/s at pos times
        turn_on_t = np.interp(t, tp, tr, left=np.nan, right=np.nan)

    # sign auto-align + correlation over meaningful-turn samples
    mask = np.isfinite(turn_on_t) & (np.abs(turn_on_t) > turn_thresh_dps)
    corr = None
    sign = 1
    if mask.sum() >= 3 and not gyro_dead:
        c = _pearson(yaw_meas_dps[mask], turn_on_t[mask])
        if c is not None:
            sign = -1 if c < 0 else 1
            corr = abs(c)
    yaw_meas_dps = yaw_meas_dps * sign

    # ── Shoebox: ABSOLUTE rate sensor heading (north-referenced) + vehicle linear sensor ─────
    # path heading (deg, compass: 0=N, +=toward E), unwrapped, interp'd
    # onto the Motion sensor clock. This is the truth needle and the anchor for the rate sensor.
    traj_head = np.full_like(t, np.nan)
    if good.sum() >= 2:
        hh = np.degrees(np.unwrap(np.arctan2(ve[good], vn[good])))
        traj_head = np.interp(t, pt[good], hh, left=np.nan, right=np.nan)

    # rate sensor heading: integrate the (sign-aligned) yaw-rate, then anchor the
    # constant of integration to the path heading at the first overlap so
    # the box points at a real compass bearing (drift from truth is then visible).
    dt_arr = np.diff(t, prepend=t[0])
    cumyaw = np.cumsum(yaw_meas_dps * dt_arr)
    h0 = 0.0
    _finite = np.isfinite(traj_head)
    if _finite.any():
        _i0 = int(np.argmax(_finite))
        h0 = float(traj_head[_i0] - cumyaw[_i0])
    box_head = cumyaw + h0

    # heading drift (rate sensor vs path) and a one-word verdict
    _dr = box_head - traj_head
    _dr = (_dr + 180.0) % 360.0 - 180.0          # wrap to [-180, 180]
    _dr = _dr[np.isfinite(_dr)]
    drift_max_deg = float(np.max(np.abs(_dr))) if _dr.size else None

    # Vehicle-sample longitudinal / lateral linear sensor. Needs the device->vehicle
    # mounting yaw, which we estimate. Assume the device is roughly fixed, so a
    # single average gravity direction defines the horizontal sample.
    g0 = gu.mean(axis=0)
    n0 = np.linalg.norm(g0)
    g0 = g0 / n0 if n0 > 1e-6 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0.0, 0.0]) if abs(g0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, g0) * g0
    e1 = e1 / (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(g0, e1)

    # Remove gravity with the CONSTANT mean-gravity vector, not the per-sample
    # low-pass (which would absorb sustained linear sensor/braking over its time
    # constant). Valid under the device-fixed assumption the mounting solve makes.
    gmag0 = float(np.mean(np.linalg.norm(grav, axis=1))) or 9.81
    lin = a - g0 * gmag0                             # gravity removed
    lin_h = lin - (lin @ g0)[:, None] * g0           # horizontal component
    h1 = lin_h @ e1
    h2 = lin_h @ e2

    # path along-track linear sensor (ground truth longitudinal), smoothed
    a_along = np.full_like(t, np.nan)
    if pt.size >= 3:
        sp_all = np.hypot(vn, ve)
        dt_pos = float(np.median(np.diff(pt))) if pt.size > 1 else 1.0
        ks = max(1, int(round(turn_smooth_s / dt_pos))) if dt_pos > 0 else 1
        sp_s = _movavg(sp_all, ks)
        a_along_pos = np.gradient(sp_s, pt)
        a_along = np.interp(t, pt, a_along_pos, left=np.nan, right=np.nan)

    fwd = None
    lat = None
    mount_yaw_deg = None
    mount_conf = 0.0
    mount_resolved = False
    fmask = np.isfinite(a_along) & (np.abs(a_along) > accel_thresh_mps2)
    if fmask.sum() >= 10:
        c1 = float(np.sum(h1[fmask] * a_along[fmask]))
        c2 = float(np.sum(h2[fmask] * a_along[fmask]))
        dn = math.hypot(c1, c2)
        if dn > 1e-9:
            fdir = np.array([c1, c2]) / dn           # forward dir in (e1,e2)
            fwd = h1 * fdir[0] + h2 * fdir[1]
            lat = -h1 * fdir[1] + h2 * fdir[0]
            cc = _pearson(fwd[fmask], a_along[fmask])
            mount_conf = 0.0 if cc is None else abs(cc)
            mount_yaw_deg = math.degrees(math.atan2(fdir[1], fdir[0]))
            mount_resolved = mount_conf >= mount_conf_thresh
    if not mount_resolved:
        fwd = None
        lat = None
    else:
        # light smoothing (~0.3 s) for a precise, readable g-force trace
        dt_imu = float(np.median(np.diff(t))) if t.size > 1 else 0.01
        ka = max(1, int(round(0.3 / dt_imu))) if dt_imu > 0 else 1
        fwd = _movavg(fwd, ka)
        lat = _movavg(lat, ka)

    # decimate to ~strip_hz
    if t.size > 1:
        dt_med = float(np.median(np.diff(t))) or (1.0 / strip_hz)
        step = max(1, int(round((1.0 / strip_hz) / dt_med)))
    else:
        step = 1
    idx = np.arange(0, t.size, step)

    a0 = float(video_utc_affine["a"])
    b0 = float(video_utc_affine["b"]) or 1.0
    t_video = (t[idx] - a0) / b0

    def _rl(x, nd=4):
        return [None if not math.isfinite(v) else round(float(v), nd) for v in x[idx]]

    # Note: keep this ASCII — it is logged to the console, and Windows code
    # pages (e.g. cp1255) raise UnicodeEncodeError on non-ASCII, which would
    # otherwise abort the whole strip.
    if gyro_dead:
        note = f"gyro DEAD (std={gyro_std:.2e} rad/s) - sensor not trustworthy"
    elif corr is None:
        note = "no clear turns to correlate against"
    else:
        note = f"gyro-vs-turn corr {corr:.2f} (sign {'+' if sign > 0 else '-'})"

    if vel_source == "coords" and not gyro_dead:
        note = note + " - heading from positions"

    if gyro_dead:
        verdict = "DEAD"
    elif corr is not None and corr >= 0.6 and (drift_max_deg is not None and drift_max_deg <= 10.0):
        verdict = "GOOD"
    elif corr is not None and corr >= 0.3:
        verdict = "OK"
    else:
        verdict = "WEAK"

    return {
        "t_video": [round(float(v), 4) for v in t_video],
        "yaw_meas_dps": _rl(yaw_meas_dps),
        "turn_traj_dps": _rl(turn_on_t),
        "accel_mag": _rl(accel_mag),
        "fwd_accel": (None if fwd is None else _rl(fwd)),
        "lat_accel": (None if lat is None else _rl(lat)),
        "box_head_deg": _rl(box_head),      # absolute rate sensor heading (compass deg)
        "traj_head_deg": _rl(traj_head),    # path truth heading (compass deg)
        "flags": {
            "gyro_dead": bool(gyro_dead),
            "gyro_std_radps": round(gyro_std, 8),
            "corr": (None if corr is None else round(corr, 3)),
            "sign": int(sign),
            "mount_yaw_deg": (None if mount_yaw_deg is None else round(mount_yaw_deg, 1)),
            "mount_conf": round(mount_conf, 3),
            "mount_resolved": bool(mount_resolved),
            "vel_source": vel_source,
            "drift_max_deg": (None if drift_max_deg is None else round(drift_max_deg, 1)),
            "verdict": verdict,
        },
        "note": note,
    }
