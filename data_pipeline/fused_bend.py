"""Bend the device's fused-location path onto Post-processing anchor points.

The platform FusedLocationProvider stream (``Fix,fused,...`` rows in
the source app's measurements log) is dense, Motion sensor-blended, and visually
smooth -- it absorbs short Signal dropouts gracefully -- but it can drift
several meters off the true track in absolute terms. Post-processing is sub-cm
accurate at every 1 Hz epoch but jittery between epochs and prone to
the occasional bad-fix outlier.

This module fuses the two: take the fused shape and warp it so it
passes within a user-defined trust band of every Post-processing anchor.

Algorithm
---------
1. Resample fused track at each Post-processing epoch (linear in Local-frame).
2. Compute residual ``r_i = ppk_enu_i - fused_enu_i`` at each epoch.
3. Outlier-resistant per-anchor weight (Huber-style):

       w_i = exp(-|r|^2 / (2 sigma^2))    if |r| <= reject_k * sigma
           = 0                             otherwise

   Defaults: ``xy_sigma_m=3.0``, ``z_sigma_m=10.0``, ``reject_k=2.0``
   so a residual <= 3 m horizontally gets full Gaussian trust, 6 m
   gets hard-rejected (caller's "3 m @ 1 sigma, 6 m @ 2 sigma" rule).

4. Nadaraya-Watson smoothing in time gives a per-time correction
   field ``c(t) = sum(w_i K(t - t_i) r_i) / sum(w_i K(t - t_i))``
   with Gaussian time kernel ``K`` of width ``time_smooth_s``.

5. For every query time: ``pos(t) = interp_fused(t) + c(t)``.

The result keeps the fused track's dense Motion sensor shape but slides it onto
the Post-processing anchor cloud, hard-rejecting individual Post-processing epochs that fall
outside the trust band (likely bad fixes, environment noise, or moments where
fused itself glitched).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .parsers import DataFix, PosRow
from .smoothing import gaussian_smooth


@dataclass
class FusedBendOptions:
    """Tuning knobs for :func:`bend_fused_to_ppk`.

    Defaults are calibrated for **source-grade Post-processing**:

    * 3 m horizontal accuracy at 1 sigma, 6 m at 2 sigma
    * 30 m worst-case epoch jumps (environment noise / measurement discontinuity)
    * 15 m vertical accuracy at 1 sigma

    Both FLP and device Post-processing have similar absolute accuracy under this
    profile, so the bend is less "warp FLP onto truth" and more "average
    two independent estimates while cancelling FLP's slow bias drift".
    The reject_k=10 ratio gives a hard cutoff at the stated 30 m horizontal
    jump ceiling; the Gaussian kernel itself further attenuates outliers
    smoothly between 6 m (2 sigma, w=0.14) and 30 m (10 sigma, w~0).
    """

    xy_sigma_m: float = 1.5
    """Horizontal 1-sigma trust band on residuals (meters). Tightened
    from 3.0 after GT sweep on reference session (survey-base PPK): a tighter band
    gives sharper anchor-vs-FLP discrimination, downweighting multipath
    spikes more aggressively. Raise (3-6 m) for worse PPK quality."""

    z_sigma_m: float = 10.0
    """Vertical 1-sigma trust band on residuals (meters). Device PPK
    vertical is typically 3-5x worse than horizontal."""

    reject_k: float = 6.0
    """Hard-reject any anchor with |residual| > ``reject_k * sigma``.
    Tightened from 10.0 — together with xy_sigma_m=1.5 this hard-cuts
    at 9 m horizontal, which matches the empirical PPK spike threshold
    on reference session (worst-case multipath ~25 m needs reject; healthy float
    epochs at 3-5 m stay)."""

    time_smooth_s: float = 20.0
    """Gaussian time-kernel width for residual smoothing (seconds).
    Raised from 5.0 after GT sweep: 20 s window provides ~sqrt(20)=4.5×
    noise reduction on the correction field while still tracking 2-3
    minute trajectory turns. Combined with the tighter xy_sigma + the
    tighter car-lateral gate this halves the residual hRMSE on reference session
    vs the original 3 s / 3 m / k=10 defaults."""

    max_gap_s: float = 2.0
    """Max distance (seconds) from a query time to the nearest fused sample
    before the bent output is marked missing."""

    provider_filter: tuple[str, ...] = (
        "fused", "FUSED", "FUSED_LOCATION_PROVIDER", "fused_location",
        "ekf_ins",
    )
    """Provider names accepted as fused-location rows. Match is case-
    insensitive. If no Fix line matches, all Fix rows are used as a
    fallback (some loggers tag the provider differently)."""

    car_lateral_sigma_m: float = 1.0
    """Non-holonomic car constraint: per-epoch weight drops with
    Gaussian sigma when the PPK position residual perpendicular to the
    direction of travel exceeds this. A car can't move sideways, so a
    PPK epoch sitting off-path lateral to neighbours = jumpy/multipath/
    cycle-slip. 3 m matches device-PPK per-epoch noise (so normal float
    scatter is tolerated); 30 m jumps register at 10 sigma -> w~0; 0
    disables the test."""

    car_smooth_s: float = 3.0
    """Gaussian time-window used to derive the smooth reference path
    against which lateral residuals are measured (seconds)."""

    car_min_speed_mps: float = 0.5
    """Below this ground speed the heading is ill-defined; the car
    constraint contributes a neutral weight (1.0) instead of a noisy one."""


@dataclass(frozen=True)
class FusedBendResult:
    """Diagnostics returned alongside the bent path."""

    n_fused: int
    n_ppk: int
    n_anchors_used: int
    n_anchors_rejected: int
    median_residual_m: float
    p95_residual_m: float
    n_car_flagged: int = 0
    median_lateral_m: float = float("nan")
    p95_lateral_m: float = float("nan")


def _filter_fused(
    fixes: Sequence[DataFix], providers: tuple[str, ...]
) -> list[DataFix]:
    keep = {p.lower() for p in providers}
    matched = [f for f in fixes if f.provider and f.provider.lower() in keep]
    return matched if matched else list(fixes)


def _car_jumpyness_weight(
    p_t: np.ndarray,
    pE: np.ndarray,
    pN: np.ndarray,
    vN: np.ndarray,
    vE: np.ndarray,
    *,
    sigma_m: float,
    smooth_s: float,
    min_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch weight from the car non-holonomic constraint.

    A vehicle on wheels cannot translate sideways; its velocity is tangent
    to the path. Each Post-processing epoch's position should lie close to a smooth
    path through its neighbours. The component of the raw-minus-smoothed
    residual perpendicular to the local heading is the "jumpyness" of
    that epoch -- environment noise, measurement discontinuity, and brief float excursions all
    manifest as sudden sideways offsets.

    Returns ``(weight, lateral_m)`` -- the per-epoch multiplicative weight
    and the signed lateral residual (m) used to compute it. Lateral is
    NaN where the test was skipped (slow speed, too-short series).
    """
    n = int(len(p_t))
    if n < 3 or sigma_m <= 0:
        return np.ones(n), np.full(n, np.nan)

    dts = np.diff(p_t)
    dts = dts[dts > 1e-9]
    if dts.size == 0:
        return np.ones(n), np.full(n, np.nan)
    median_dt = float(np.median(dts))
    sigma_samples = max(1.0, smooth_s / max(median_dt, 1e-9))

    pE_s = np.asarray(gaussian_smooth(pE.tolist(), sigma_samples))
    pN_s = np.asarray(gaussian_smooth(pN.tolist(), sigma_samples))

    dE = pE - pE_s
    dN = pN - pN_s

    win = max(1, int(round(1.0 / max(median_dt, 1e-9))))  # ~1 s chord.
    idx = np.arange(n)
    il = np.clip(idx - win, 0, n - 1)
    ir = np.clip(idx + win, 0, n - 1)
    dt_chord = p_t[ir] - p_t[il]
    dt_chord = np.where(dt_chord > 1e-9, dt_chord, 1.0)
    fwd_E = (pE_s[ir] - pE_s[il]) / dt_chord
    fwd_N = (pN_s[ir] - pN_s[il]) / dt_chord
    speed_est = np.sqrt(fwd_E * fwd_E + fwd_N * fwd_N)

    has_v = np.isfinite(vN) & np.isfinite(vE)
    speed_dop = np.sqrt(
        np.where(has_v, vN, 0.0) ** 2 + np.where(has_v, vE, 0.0) ** 2
    )

    fwd_norm = np.where(speed_est > 1e-6, speed_est, 1.0)
    uE = fwd_E / fwd_norm
    uN = fwd_N / fwd_norm

    # Perpendicular to (uE, uN) is (uN, -uE); project residual onto it.
    lateral = dE * uN - dN * uE

    speed_gate = np.where(has_v, speed_dop, speed_est)
    active = speed_gate >= min_speed_mps

    w = np.where(
        active,
        np.exp(-(lateral / max(sigma_m, 1e-6)) ** 2 / 2.0),
        1.0,
    )
    lateral_reported = np.where(active, lateral, np.nan)
    return w, lateral_reported


def bend_fused_to_ppk(
    fused: Sequence[DataFix],
    ppk: Sequence[PosRow],
    query_times_utc_s: Sequence[float],
    *,
    options: FusedBendOptions | None = None,
) -> tuple[list[float], list[float], list[float], list[bool], list[float], FusedBendResult]:
    """Return per-query-time (lat, lon, h, has_pos, trust, diagnostics).

    Samples whose query time has no fused sample within ``max_gap_s`` end
    up with ``has_pos=False`` and NaN coordinates. Samples with a fused
    sample but no usable Post-processing anchor within ``4 * time_smooth_s`` fall
    back to the unbent fused position (still flagged ``has_pos=True``).

    ``trust`` is a per-query-time score in [0, 1]: 1.0 means every Post-processing
    anchor near this time was fully weighted (clean Post-processing, output = bent
    onto Post-processing); 0.0 means all nearby anchors were downweighted to zero
    (output = pure FLP shape). Useful for diagnostic colouring.
    """
    options = options or FusedBendOptions()
    n_q = len(query_times_utc_s)
    lat_out = [float("nan")] * n_q
    lon_out = [float("nan")] * n_q
    h_out = [float("nan")] * n_q
    has = [False] * n_q
    trust_out = [0.0] * n_q

    fused = _filter_fused(fused, options.provider_filter)
    if not fused or not ppk or n_q == 0:
        return lat_out, lon_out, h_out, has, trust_out, FusedBendResult(
            n_fused=len(fused), n_ppk=len(ppk),
            n_anchors_used=0, n_anchors_rejected=0,
            median_residual_m=float("nan"), p95_residual_m=float("nan"),
            n_car_flagged=0,
            median_lateral_m=float("nan"), p95_lateral_m=float("nan"),
        )

    ref: tuple[float, float, float] | None = None
    for f in fused:
        if (math.isfinite(f.lat) and math.isfinite(f.lon)
                and math.isfinite(f.h)):
            ref = (f.lat, f.lon, f.h)
            break
    if ref is None:
        return lat_out, lon_out, h_out, has, trust_out, FusedBendResult(
            n_fused=len(fused), n_ppk=len(ppk),
            n_anchors_used=0, n_anchors_rejected=0,
            median_residual_m=float("nan"), p95_residual_m=float("nan"),
            n_car_flagged=0,
            median_lateral_m=float("nan"), p95_lateral_m=float("nan"),
        )

    fused_sorted = sorted(fused, key=lambda r: r.utc_s)
    f_t = np.array([r.utc_s for r in fused_sorted], dtype=np.float64)
    fE = np.empty(len(fused_sorted)); fN = np.empty(len(fused_sorted))
    fU = np.empty(len(fused_sorted))
    for i, r in enumerate(fused_sorted):
        h_use = r.h if math.isfinite(r.h) else ref[2]
        x, y, z = llh_to_ecef(r.lat, r.lon, h_use)
        e, n, u = ecef_to_enu(x, y, z, ref)
        fE[i], fN[i], fU[i] = e, n, u

    ppk_sorted = sorted(ppk, key=lambda r: r.utc_s)
    p_t = np.array([r.utc_s for r in ppk_sorted], dtype=np.float64)
    pE = np.empty(len(ppk_sorted)); pN = np.empty(len(ppk_sorted))
    pU = np.empty(len(ppk_sorted))
    for i, r in enumerate(ppk_sorted):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, n, u = ecef_to_enu(x, y, z, ref)
        pE[i], pN[i], pU[i] = e, n, u
    vN_arr = np.array([r.vn for r in ppk_sorted], dtype=np.float64)
    vE_arr = np.array([r.ve for r in ppk_sorted], dtype=np.float64)

    car_w, lateral_m = _car_jumpyness_weight(
        p_t, pE, pN, vN_arr, vE_arr,
        sigma_m=options.car_lateral_sigma_m,
        smooth_s=options.car_smooth_s,
        min_speed_mps=options.car_min_speed_mps,
    )

    fE_at_p = np.interp(p_t, f_t, fE, left=np.nan, right=np.nan)
    fN_at_p = np.interp(p_t, f_t, fN, left=np.nan, right=np.nan)
    fU_at_p = np.interp(p_t, f_t, fU, left=np.nan, right=np.nan)

    idx = np.searchsorted(f_t, p_t)
    il = np.clip(idx - 1, 0, len(f_t) - 1)
    ir = np.clip(idx, 0, len(f_t) - 1)
    gap_p = np.minimum(np.abs(p_t - f_t[il]), np.abs(f_t[ir] - p_t))
    bad_p = gap_p > options.max_gap_s
    fE_at_p[bad_p] = np.nan
    fN_at_p[bad_p] = np.nan
    fU_at_p[bad_p] = np.nan

    rE = pE - fE_at_p
    rN = pN - fN_at_p
    rU = pU - fU_at_p
    horiz = np.sqrt(rE * rE + rN * rN)
    vert = np.abs(rU)

    sigma_h = max(1e-3, options.xy_sigma_m)
    sigma_v = max(1e-3, options.z_sigma_m)
    reject_h = options.reject_k * sigma_h
    reject_v = options.reject_k * sigma_v

    finite_h = np.isfinite(horiz)
    finite_v = np.isfinite(vert)
    w_h = np.where(finite_h & (horiz <= reject_h),
                   np.exp(-(horiz / sigma_h) ** 2 / 2.0), 0.0)
    w_v = np.where(finite_v & (vert <= reject_v),
                   np.exp(-(vert / sigma_v) ** 2 / 2.0), 0.0)
    # Combine Post-processing-shape-vs-fused weight with the car kinematic weight.
    # The car gate is a strict multiplicative AND: an anchor that jumps
    # sideways relative to its neighbours is downweighted regardless of
    # how well it agrees with FLP.
    w_h = w_h * car_w
    w_v = w_v * car_w
    rE_safe = np.where(np.isfinite(rE), rE, 0.0)
    rN_safe = np.where(np.isfinite(rN), rN, 0.0)
    rU_safe = np.where(np.isfinite(rU), rU, 0.0)

    n_used = int(np.sum(w_h > 0.0))
    n_reject = int(np.sum(finite_h & (w_h == 0.0)))
    med_res = float(np.median(horiz[finite_h])) if np.any(finite_h) else float("nan")
    p95_res = (float(np.percentile(horiz[finite_h], 95))
               if np.any(finite_h) else float("nan"))
    finite_lat = np.isfinite(lateral_m)
    n_car_flagged = int(np.sum(finite_lat & (car_w < 0.5)))
    if np.any(finite_lat):
        med_lat = float(np.median(np.abs(lateral_m[finite_lat])))
        p95_lat = float(np.percentile(np.abs(lateral_m[finite_lat]), 95))
    else:
        med_lat = float("nan")
        p95_lat = float("nan")

    tau = max(1e-3, options.time_smooth_s)
    win = 4.0 * tau

    q_t = np.asarray(query_times_utc_s, dtype=np.float64)
    cE_out = np.full(n_q, np.nan)
    cN_out = np.full(n_q, np.nan)
    cU_out = np.full(n_q, np.nan)
    trust_arr = np.zeros(n_q)
    lo_q = np.searchsorted(p_t, q_t - win, side="left")
    hi_q = np.searchsorted(p_t, q_t + win, side="right")
    for i in range(n_q):
        a, b = int(lo_q[i]), int(hi_q[i])
        if b <= a:
            continue
        dt_ = q_t[i] - p_t[a:b]
        k = np.exp(-(dt_ / tau) ** 2 / 2.0)
        wh = w_h[a:b] * k
        wv = w_v[a:b] * k
        sh = float(wh.sum())
        sv = float(wv.sum())
        if sh > 1e-9:
            cE_out[i] = float((wh * rE_safe[a:b]).sum() / sh)
            cN_out[i] = float((wh * rN_safe[a:b]).sum() / sh)
        if sv > 1e-9:
            cU_out[i] = float((wv * rU_safe[a:b]).sum() / sv)
        # Trust = fraction of nearby anchor kernel mass that survived weighting.
        # k.sum() is the upper bound (all anchors trusted); wh.sum() is the
        # actual horizontal trusted mass. Ratio in [0, 1] is what gets plotted.
        ks = float(k.sum())
        if ks > 1e-9:
            trust_arr[i] = min(1.0, sh / ks)

    qE = np.interp(q_t, f_t, fE, left=np.nan, right=np.nan)
    qN = np.interp(q_t, f_t, fN, left=np.nan, right=np.nan)
    qU = np.interp(q_t, f_t, fU, left=np.nan, right=np.nan)
    idx_q = np.searchsorted(f_t, q_t)
    il_q = np.clip(idx_q - 1, 0, len(f_t) - 1)
    ir_q = np.clip(idx_q, 0, len(f_t) - 1)
    gap_q = np.minimum(np.abs(q_t - f_t[il_q]), np.abs(f_t[ir_q] - q_t))
    bad_q = gap_q > options.max_gap_s
    qE[bad_q] = np.nan
    qN[bad_q] = np.nan
    qU[bad_q] = np.nan

    cE = np.where(np.isfinite(cE_out), cE_out, 0.0)
    cN = np.where(np.isfinite(cN_out), cN_out, 0.0)
    cU = np.where(np.isfinite(cU_out), cU_out, 0.0)

    bentE = qE + cE
    bentN = qN + cN
    bentU = qU + cU

    for i in range(n_q):
        if not (math.isfinite(bentE[i]) and math.isfinite(bentN[i])
                and math.isfinite(bentU[i])):
            continue
        la, lo, hh = enu_to_llh(
            float(bentE[i]), float(bentN[i]), float(bentU[i]), ref
        )
        lat_out[i] = la
        lon_out[i] = lo
        h_out[i] = hh
        has[i] = True
        trust_out[i] = float(trust_arr[i])

    return lat_out, lon_out, h_out, has, trust_out, FusedBendResult(
        n_fused=len(fused),
        n_ppk=len(ppk),
        n_anchors_used=n_used,
        n_anchors_rejected=n_reject,
        median_residual_m=med_res,
        p95_residual_m=p95_res,
        n_car_flagged=n_car_flagged,
        median_lateral_m=med_lat,
        p95_lateral_m=p95_lat,
    )
