"""Non-holonomic constraint (NHC) for vehicle Signal post-processing.

A wheeled vehicle on a planar surface cannot translate sideways: its
velocity vector is constrained to lie along its heading (long-axis). Post-processing
solutions on a moving car routinely violate this — environment noise, measurement discontinuity, and float ambiguities all manifest as **lateral** position
wobble + lateral velocity components that the vehicle physically cannot
produce. NHC enforces the constraint as a post-process, killing that
lateral noise while leaving the (real) longitudinal motion alone.

Pipeline
--------
1. Heading provider θ(t)
   * preferred source: Motion sensor yaw from
     :func:`data_pipeline.imu_gnss_fusion.fuse` (Complementary-update complementary
     filter, ~200 Hz, magnetometer-disciplined long-term)
   * fallback: Post-processing Rate-signal velocity heading ``atan2(ve, vn)`` when speed
     exceeds the configured floor (Rate-signal heading is ill-defined
     statically)
   * last-ditch: coords-delta heading from successive Local-frame positions

2. Per-epoch gate
   * skip when ground speed < ``min_speed_mps`` (heading undefined)
   * skip when |yaw rate| > ``max_yaw_rate_dps`` (NHC weakens during
     hard turns; the Motion sensor has its own lag during transient yaw, and the
     coords-Δ heading lags the body sample by half a sample)
   * skip when no heading source resolves at that UTC

3. Velocity NHC. Decompose (vn, ve) into longitudinal + lateral in the
   body sample::

       v_long =  vn cos θ + ve sin θ
       v_lat  = -vn sin θ + ve cos θ

   Shrink ``v_lat`` to a multiplicative residual (default 0 = full kill)
   and recompose. Writes back to the PosRow's velocity columns.

4. Position NHC. The position residual from a locally-smoothed reference
   track is decomposed the same way; the lateral component is shrunk to
   zero so the corrected position lies on the smoothed line at the raw
   epoch's longitudinal offset. This is what "kills lateral Post-processing noise"
   visually in Export format / viewers.

The heading source is *only* needed for these decompositions; the
smoothing reference is computed independently with a short Gaussian
(default 3 s in Local-frame metric space, not lat/lon, so the kernel width is
constant in metres at any latitude).

Limits
------
* On heavy slides (rally / off-road) the NHC assumption breaks and the
  output is *worse* than the raw Post-processing. The default ``max_yaw_rate_dps =
  30°/s`` disables NHC when yaw rate spikes, but the user should still
  inspect the lateral residual histogram in :class:`NhcResult` before
  consuming.
* Pure-pedestrian sessions violate NHC constantly (walkers swing
  laterally). Disable for non-vehicle data.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .parsers import DataFix, PosRow
from .smoothing import gaussian_smooth


# ----------------------------------------------------------------------
# Options + result types
# ----------------------------------------------------------------------


@dataclass
class NhcOptions:
    """Tuning knobs for :func:`apply_nhc`.

    Defaults are calibrated for **car-grade device Post-processing**: kill lateral
    residuals fully, gate by 2 m/s minimum speed, disable above 30°/s
    yaw rate (typical lane-change is ~15°/s; tight U-turn is ~45°/s).
    """

    enabled: bool = True

    # Speed floor below which heading is too noisy to trust (m/s).
    min_speed_mps: float = 2.0

    # Yaw-rate ceiling above which NHC is disabled this epoch (deg/s).
    max_yaw_rate_dps: float = 30.0

    # Multiplicative shrink applied to the lateral velocity component
    # (0.0 = hard kill, 1.0 = no change). Floats between 0 and 1 give a
    # soft prior.
    lateral_shrink_velocity: float = 0.0

    # Multiplicative shrink applied to the lateral position residual
    # (0.0 = snap to smoothed reference line, 1.0 = no change).
    lateral_shrink_position: float = 0.0

    # Reference-path smoothing window (seconds). Translated to samples
    # via the median Post-processing epoch dt. 3 s ≈ one car-length at 30 km/h.
    position_smooth_s: float = 3.0

    # Heading source preference:
    #   "auto"    -- Motion sensor if provided else Rate-signal else coords-Δ
    #   "motion sensor"     -- only Motion sensor (fail-fast if no attitude samples)
    #   "rate-signal" -- only Rate-signal from PosRow.vn,ve
    #   "coords"  -- only Local-frame coord-derivative
    heading_source: str = "auto"


@dataclass(frozen=True)
class NhcResult:
    """Outcome of one :func:`apply_nhc` pass.

    ``rows_out`` is the corrected PosRow list (same length, same UTC,
    same Q/ns columns; only lat/lon/h and vn/ve/vu may change).

    ``lat_resid_before_m`` / ``lat_resid_after_m`` are RMS of the
    lateral position residual against the smoothed reference, in
    metres — the headline metric for "how much wobble did NHC kill".
    """

    rows_out: list[PosRow]
    n_in: int
    n_modified: int
    n_skipped_slow: int
    n_skipped_turning: int
    n_skipped_no_heading: int
    lat_resid_before_m: float
    lat_resid_after_m: float
    long_resid_before_m: float
    long_resid_after_m: float
    heading_source_used: str
    summary: str


# ----------------------------------------------------------------------
# Heading providers
# ----------------------------------------------------------------------


def _build_heading_imu(
    attitude_samples: Sequence[object],
) -> tuple[list[float], list[float]]:
    """Return (utc_s, yaw_deg) arrays from Motion sensor attitude samples."""
    if not attitude_samples:
        return [], []
    t = [float(s.utc_s) for s in attitude_samples]        # type: ignore[attr-defined]
    y = [float(s.yaw_deg) for s in attitude_samples]      # type: ignore[attr-defined]
    return t, y


def _build_heading_doppler(
    rows: Sequence[PosRow], min_speed: float,
) -> tuple[list[float], list[float]]:
    """Heading from Post-processing Rate-signal velocity. Returns (utc, deg) at epochs
    with sufficient speed; gaps left for the caller's interpolator to
    skip."""
    t, y = [], []
    for r in rows:
        if not (math.isfinite(r.vn) and math.isfinite(r.ve)):
            continue
        s = math.hypot(r.vn, r.ve)
        if s < min_speed:
            continue
        h = math.degrees(math.atan2(r.ve, r.vn)) % 360.0
        t.append(r.utc_s)
        y.append(h)
    return t, y


def _build_heading_coords(
    rows: Sequence[PosRow], ref_llh: tuple[float, float, float], min_speed: float,
) -> tuple[list[float], list[float]]:
    """Heading from Local-frame coord derivative. Centred finite difference so
    the heading is referenced to t_i, not t_i − Δt/2."""
    n = len(rows)
    e = np.empty(n); nn = np.empty(n)
    for i, r in enumerate(rows):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        ee, n2, _ = ecef_to_enu(x, y, z, ref_llh)
        e[i] = ee; nn[i] = n2
    out_t: list[float] = []
    out_h: list[float] = []
    for i in range(1, n - 1):
        dt = rows[i + 1].utc_s - rows[i - 1].utc_s
        if dt <= 0 or dt > 4.0:
            continue
        de = e[i + 1] - e[i - 1]
        dn = nn[i + 1] - nn[i - 1]
        spd = math.hypot(de, dn) / dt
        if spd < min_speed:
            continue
        h = math.degrees(math.atan2(de, dn)) % 360.0
        out_t.append(rows[i].utc_s)
        out_h.append(h)
    return out_t, out_h


def _unwrap_deg(values: list[float]) -> list[float]:
    """Unwrap a heading-in-degrees series so linear interpolation crosses
    the 0/360 seam smoothly (e.g. 359 → 1 becomes 359 → 361)."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        prev = out[-1]
        d = v - prev
        while d > 180.0:
            v -= 360.0
            d = v - prev
        while d < -180.0:
            v += 360.0
            d = v - prev
        out.append(v)
    return out


class _HeadingLookup:
    """Linear interpolator over a sorted (t, heading_deg_unwrapped) series.

    Returns the heading in degrees mod 360 at any query UTC. Refuses to
    extrapolate beyond the first / last sample by more than 1 s.
    """

    def __init__(self, t: list[float], deg: list[float]):
        if len(t) != len(deg):
            raise ValueError("t and deg must be same length")
        if len(t) >= 2 and not all(t[i] <= t[i + 1] for i in range(len(t) - 1)):
            pairs = sorted(zip(t, deg))
            t = [p[0] for p in pairs]
            deg = [p[1] for p in pairs]
        self._t = t
        self._h = _unwrap_deg(deg)
        self._range = (t[0], t[-1]) if t else (0.0, 0.0)

    def __bool__(self) -> bool:
        return len(self._t) >= 2

    def at(self, utc: float) -> Optional[float]:
        if len(self._t) < 2:
            return None
        if utc < self._range[0] - 1.0 or utc > self._range[1] + 1.0:
            return None
        j = bisect_left(self._t, utc)
        if j <= 0:
            return self._h[0] % 360.0
        if j >= len(self._t):
            return self._h[-1] % 360.0
        t0, t1 = self._t[j - 1], self._t[j]
        h0, h1 = self._h[j - 1], self._h[j]
        if t1 <= t0:
            return h0 % 360.0
        u = (utc - t0) / (t1 - t0)
        return (h0 + u * (h1 - h0)) % 360.0


# ----------------------------------------------------------------------
# Reference path smoothing (Local-frame metric space)
# ----------------------------------------------------------------------


def _smooth_enu(
    rows: Sequence[PosRow], ref_llh: tuple[float, float, float],
    smooth_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (e, n, u) smoothed in Local-frame about ``ref_llh``."""
    n = len(rows)
    e_raw = np.empty(n); n_raw = np.empty(n); u_raw = np.empty(n)
    for i, r in enumerate(rows):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        ee, nn, uu = ecef_to_enu(x, y, z, ref_llh)
        e_raw[i] = ee; n_raw[i] = nn; u_raw[i] = uu

    ts = [r.utc_s for r in rows]
    dts = [ts[i + 1] - ts[i] for i in range(n - 1) if ts[i + 1] - ts[i] > 1e-6]
    if not dts or smooth_s <= 0:
        return e_raw, n_raw, u_raw
    median_dt = sorted(dts)[len(dts) // 2]
    sigma_samples = max(1.0, smooth_s / max(median_dt, 1e-9))

    e_s = np.asarray(gaussian_smooth(e_raw.tolist(), sigma_samples))
    n_s = np.asarray(gaussian_smooth(n_raw.tolist(), sigma_samples))
    u_s = np.asarray(gaussian_smooth(u_raw.tolist(), sigma_samples))
    return e_s, n_s, u_s


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------


def apply_nhc(
    pos_rows: Sequence[PosRow],
    attitude_samples: Optional[Sequence[object]] = None,
    *,
    options: Optional[NhcOptions] = None,
    log: Optional[object] = None,
) -> NhcResult:
    """Apply the non-holonomic constraint to ``pos_rows``.

    Pure function: returns a new ``PosRow`` list, never mutates the
    input. Pass ``attitude_samples`` (from
    :func:`data_pipeline.imu_gnss_fusion.fuse`) to use Motion sensor yaw as the
    heading source; otherwise the function falls back to Rate-signal / coords
    according to ``options.heading_source``.
    """
    options = options or NhcOptions()
    rows = list(pos_rows)
    n = len(rows)

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)  # type: ignore[operator]

    if n == 0:
        return NhcResult(
            rows_out=[], n_in=0, n_modified=0, n_skipped_slow=0,
            n_skipped_turning=0, n_skipped_no_heading=0,
            lat_resid_before_m=0.0, lat_resid_after_m=0.0,
            long_resid_before_m=0.0, long_resid_after_m=0.0,
            heading_source_used="empty", summary="no rows",
        )

    if not options.enabled:
        return NhcResult(
            rows_out=rows, n_in=n, n_modified=0, n_skipped_slow=0,
            n_skipped_turning=0, n_skipped_no_heading=0,
            lat_resid_before_m=0.0, lat_resid_after_m=0.0,
            long_resid_before_m=0.0, long_resid_after_m=0.0,
            heading_source_used="disabled",
            summary="NHC disabled (pass-through)",
        )

    rows_sorted = sorted(rows, key=lambda r: r.utc_s)
    ref_llh = (rows_sorted[0].lat_deg, rows_sorted[0].lon_deg, rows_sorted[0].h_m)

    # ── heading provider ──────────────────────────────────────────────
    src = options.heading_source.lower()
    lookup: _HeadingLookup = _HeadingLookup([], [])
    used = "none"

    def _try(name: str, t: list[float], h: list[float]) -> bool:
        nonlocal lookup, used
        if len(t) >= 2:
            lookup = _HeadingLookup(t, h)
            used = name
            return True
        return False

    if src == "imu" or src == "auto":
        t_i, h_i = _build_heading_imu(attitude_samples or [])
        if _try("imu", t_i, h_i):
            pass
        elif src == "imu":
            _log("[nhc] heading_source='imu' but no attitude samples — abort")
            return NhcResult(
                rows_out=rows, n_in=n, n_modified=0, n_skipped_slow=0,
                n_skipped_turning=0, n_skipped_no_heading=n,
                lat_resid_before_m=0.0, lat_resid_after_m=0.0,
                long_resid_before_m=0.0, long_resid_after_m=0.0,
                heading_source_used="none",
                summary="IMU heading required but no attitude samples",
            )

    if used == "none" and (src in ("doppler", "auto")):
        t_d, h_d = _build_heading_doppler(rows_sorted, options.min_speed_mps)
        _try("doppler", t_d, h_d)

    if used == "none" and (src in ("coords", "auto")):
        t_c, h_c = _build_heading_coords(rows_sorted, ref_llh, options.min_speed_mps)
        _try("coords", t_c, h_c)

    if used == "none":
        _log("[nhc] no usable heading source found — abort")
        return NhcResult(
            rows_out=rows, n_in=n, n_modified=0, n_skipped_slow=0,
            n_skipped_turning=0, n_skipped_no_heading=n,
            lat_resid_before_m=0.0, lat_resid_after_m=0.0,
            long_resid_before_m=0.0, long_resid_after_m=0.0,
            heading_source_used="none",
            summary="no usable heading source",
        )
    _log(f"[nhc] heading source: {used}")

    # ── smoothed reference path in Local-frame ────────────────────────────────
    e_s, n_s, u_s = _smooth_enu(rows_sorted, ref_llh, options.position_smooth_s)
    e_raw = np.empty(n); n_raw = np.empty(n); u_raw = np.empty(n)
    for i, r in enumerate(rows_sorted):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        ee, nn, uu = ecef_to_enu(x, y, z, ref_llh)
        e_raw[i] = ee; n_raw[i] = nn; u_raw[i] = uu

    # ── per-epoch projection ──────────────────────────────────────────
    n_modified = n_slow = n_turning = n_nohead = 0
    rows_out: list[PosRow] = []

    # Yaw-rate (deg/s) estimate from consecutive heading samples.
    last_theta = None
    last_t = None

    lat_resid_before_sum = 0.0
    lat_resid_after_sum  = 0.0
    long_resid_before_sum = 0.0
    long_resid_after_sum  = 0.0
    resid_count = 0

    for i, r in enumerate(rows_sorted):
        spd = (
            math.hypot(r.vn, r.ve)
            if (math.isfinite(r.vn) and math.isfinite(r.ve))
            else 0.0
        )
        theta = lookup.at(r.utc_s)
        if theta is None:
            n_nohead += 1
            rows_out.append(r)
            last_theta = None
            last_t = None
            continue

        # Estimate yaw rate from heading derivative.
        yaw_rate = 0.0
        if last_theta is not None and last_t is not None:
            dt = r.utc_s - last_t
            if dt > 1e-6:
                dh = (theta - last_theta + 540.0) % 360.0 - 180.0
                yaw_rate = abs(dh / dt)
        last_theta = theta
        last_t = r.utc_s

        if spd < options.min_speed_mps:
            n_slow += 1
            rows_out.append(r)
            continue
        if yaw_rate > options.max_yaw_rate_dps:
            n_turning += 1
            rows_out.append(r)
            continue

        th = math.radians(theta)
        cT, sT = math.cos(th), math.sin(th)

        # Velocity projection (body sample: long = +N at θ=0, lat = +E at θ=0,
        # rotated by θ measured clockwise from N).
        v_long =  r.vn * cT + r.ve * sT
        v_lat  = -r.vn * sT + r.ve * cT
        v_lat *= options.lateral_shrink_velocity
        vn_new = v_long * cT - v_lat * sT
        ve_new = v_long * sT + v_lat * cT

        # Position residual decomposition.
        de = e_raw[i] - e_s[i]   # east residual
        dn = n_raw[i] - n_s[i]   # north residual
        p_long =  dn * cT + de * sT
        p_lat  = -dn * sT + de * cT
        p_lat_new = p_lat * options.lateral_shrink_position
        de_new = p_long * sT + p_lat_new * cT
        dn_new = p_long * cT - p_lat_new * sT
        new_e = e_s[i] + de_new
        new_n = n_s[i] + dn_new
        new_u = u_raw[i]   # NHC is in the horizontal plane only

        # Stats.
        lat_resid_before_sum += p_lat * p_lat
        lat_resid_after_sum  += p_lat_new * p_lat_new
        long_resid_before_sum += p_long * p_long
        long_resid_after_sum  += p_long * p_long  # unchanged
        resid_count += 1

        new_lat, new_lon, new_h = enu_to_llh(new_e, new_n, new_u, ref_llh)

        rows_out.append(PosRow(
            utc_s=r.utc_s,
            lat_deg=new_lat,
            lon_deg=new_lon,
            h_m=new_h,
            quality=r.quality,
            vn=vn_new,
            ve=ve_new,
            vu=r.vu,
            ns=r.ns,
        ))
        n_modified += 1

    # ── stats summary ─────────────────────────────────────────────────
    def _rms(s: float, k: int) -> float:
        return math.sqrt(s / k) if k > 0 else 0.0

    lat_before = _rms(lat_resid_before_sum, resid_count)
    lat_after  = _rms(lat_resid_after_sum,  resid_count)
    long_before = _rms(long_resid_before_sum, resid_count)
    long_after  = _rms(long_resid_after_sum,  resid_count)

    pct = 100.0 * (1.0 - (lat_after / lat_before)) if lat_before > 1e-9 else 0.0
    summary = (
        f"NHC applied to {n_modified}/{n} epochs (source={used})  "
        f"lateral RMS {lat_before*100:.1f}cm -> {lat_after*100:.1f}cm "
        f"({pct:+.1f}%)  "
        f"skipped slow={n_slow} turning={n_turning} no-heading={n_nohead}"
    )
    _log("[nhc] " + summary)

    return NhcResult(
        rows_out=rows_out,
        n_in=n,
        n_modified=n_modified,
        n_skipped_slow=n_slow,
        n_skipped_turning=n_turning,
        n_skipped_no_heading=n_nohead,
        lat_resid_before_m=lat_before,
        lat_resid_after_m=lat_after,
        long_resid_before_m=long_before,
        long_resid_after_m=long_after,
        heading_source_used=used,
        summary=summary,
    )


# ----------------------------------------------------------------------
# ZUPT — zero-velocity / zero-horizontal-speed updates
# ----------------------------------------------------------------------


@dataclass
class ZuptOptions:
    """Tuning knobs for :func:`apply_zupt`.

    When a vehicle is stationary (at a light, in a parking spot) Post-processing
    positions still wobble — environment noise, propagation-A noise, ambiguity
    drift — even though the true position is constant. ZUPT (zero
    velocity update) detects these intervals and snaps every epoch in
    them to a single position estimate (the median lat/lon/h of the
    interval), killing the residual jitter entirely.

    Defaults: 0.3 m/s speed floor, 2 s minimum static duration. A typical
    red-light stop in a car satisfies both; a slow walker stepping
    forward does not.
    """

    enabled: bool = True
    speed_threshold_mps: float = 0.3
    min_static_duration_s: float = 2.0
    use_median: bool = True          # True -> median, False -> first epoch
    zero_velocity: bool = True       # also set vn/ve/vu to 0 inside ZUPT
    # Reject candidate static intervals whose horizontal position spread
    # (max-min in metres) exceeds this. Catches "slow drift" segments
    # that pass the speed threshold but are not true stops — snapping
    # those to a median moves the path away from truth.
    # Empirically: the reference set ZUPT regressed +7 % without this guard.
    max_static_spread_m: float = 2.0

    # Weighted-median support. When True, each epoch contributes a weight
    # of ``q_weights[r.quality]`` (default 4=Fix, 1=Float, 0.2=Single/Deg).
    # The snap position is the weighted median (or the un-weighted median
    # of the highest-weight subset when ``filter_to_best_q``).
    quality_weighted: bool = False
    q_weights: dict[int, float] = field(default_factory=lambda: {
        1: 4.0, 2: 1.0, 4: 0.3, 5: 0.2, 6: 0.2,
    })
    # If True, only epochs whose quality is min(Q present in the
    # interval) are used to compute the snap (drops Float when Fix
    # epochs exist in the same stop). Honoured only when at least
    # ``min_best_q_count`` such epochs exist.
    filter_to_best_q: bool = True
    min_best_q_count: int = 2


@dataclass(frozen=True)
class ZuptResult:
    """Outcome of :func:`apply_zupt`."""

    rows_out: list[PosRow]
    n_in: int
    n_static_epochs: int             # rows inside a detected static interval
    n_intervals: int                 # number of static intervals snapped
    total_static_duration_s: float   # sum of all snapped intervals
    summary: str


def _classify_speed(rows: Sequence[PosRow], thr: float) -> list[bool]:
    """Per-row 'is static' flag from Rate-signal speed (preferred) or coords-Δ.

    Rate-signal is more reliable when present because it doesn't lag the body
    sample; coords-Δ fallback uses centred finite difference so a single
    isolated environment noise jump doesn't classify an epoch as moving when its
    neighbours are static.
    """
    n = len(rows)
    out = [False] * n
    # Track which epochs lack usable Rate-signal so we fill them from coords.
    have_doppler = [False] * n
    for i, r in enumerate(rows):
        if math.isfinite(r.vn) and math.isfinite(r.ve):
            spd = math.hypot(r.vn, r.ve)
            out[i] = spd < thr
            have_doppler[i] = True
    if all(have_doppler):
        return out
    # Coords-Δ fallback for missing Rate-signal.
    if n < 3:
        return out
    ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
    enu_e = [0.0] * n; enu_n = [0.0] * n
    for i, r in enumerate(rows):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, nn, _ = ecef_to_enu(x, y, z, ref)
        enu_e[i] = e; enu_n[i] = nn
    for i in range(1, n - 1):
        if have_doppler[i]:
            continue
        dt = rows[i + 1].utc_s - rows[i - 1].utc_s
        if dt <= 0 or dt > 4.0:
            continue
        de = enu_e[i + 1] - enu_e[i - 1]
        dn = enu_n[i + 1] - enu_n[i - 1]
        spd = math.hypot(de, dn) / dt
        out[i] = spd < thr
    return out


def _median_xyz(rows: Sequence[PosRow]) -> tuple[float, float, float]:
    """Element-wise median lat/lon/h across a slice of rows."""
    lats = sorted(r.lat_deg for r in rows)
    lons = sorted(r.lon_deg for r in rows)
    hs   = sorted(r.h_m     for r in rows)
    m = len(rows) // 2
    return lats[m], lons[m], hs[m]


def _weighted_median_xyz(
    rows: Sequence[PosRow], q_weights: dict,
) -> tuple[float, float, float]:
    """Element-wise weighted median (per-axis), weight = q_weights[Q]."""
    def _wmed(vals_w: list[tuple[float, float]]) -> float:
        if not vals_w:
            return 0.0
        vals_w = sorted(vals_w, key=lambda p: p[0])
        total = sum(w for _, w in vals_w)
        if total <= 0:
            return vals_w[len(vals_w) // 2][0]
        acc = 0.0
        for v, w in vals_w:
            acc += w
            if acc >= total / 2.0:
                return v
        return vals_w[-1][0]
    lat_w = [(r.lat_deg, q_weights.get(r.quality, 0.2)) for r in rows]
    lon_w = [(r.lon_deg, q_weights.get(r.quality, 0.2)) for r in rows]
    h_w   = [(r.h_m,     q_weights.get(r.quality, 0.2)) for r in rows]
    return _wmed(lat_w), _wmed(lon_w), _wmed(h_w)


def apply_quality_gate(
    pos_rows: Sequence[PosRow], *, max_q: int = 2,
) -> list[PosRow]:
    """Drop rows whose The external solver quality flag is above ``max_q``.

    max_q=1 keeps only Fix; max_q=2 keeps Fix+Float; max_q=4 drops only
    Single/degraded. The output is shorter than the input — callers
    using this for GT-residual evaluation should expect a smaller
    matched-epoch count.
    """
    return [r for r in pos_rows if r.quality and r.quality <= max_q]


# ----------------------------------------------------------------------
# Adaptive filter — picks the best stack based on Post-processing Q distribution
# ----------------------------------------------------------------------


@dataclass
class AdaptiveFilterOptions:
    """Tuning for :func:`adaptive_filter`.

    The adaptive selector inspects the Post-processing quality histogram and picks
    one of three stacks:

    * **poor-quality** (Single+ ≥ ``poor_single_pct_threshold``):
      Gate Q ≤ 2 → ZUPT → NHC → Gauss car.  Dropping bad fixes is
      worth more than any per-epoch smoothing on these datasets
      (see GT_EVAL_MEGA.md: session 4 session-A gains −33 % from gating alone).

    * **clean Fix-heavy** (Fix% ≥ ``clean_fix_pct_threshold``):
      Gauss car only.  Aggressive smoothing or NHC on this data
      ranges from neutral to harmful — the signal is already good.

    * **default mixed/Float-dominant**:
      ZUPT(quality-weighted) → NHC(auto) → Gauss car.  Current
      universal champion across the 8-dataset sweep.
    """

    enabled: bool = True
    poor_single_pct_threshold: float = 30.0
    clean_fix_pct_threshold: float = 15.0
    xy_sigma_s: float = 2.0
    z_sigma_s:  float = 10.0


def adaptive_filter(
    pos_rows: Sequence[PosRow],
    attitude_samples: Optional[Sequence[object]] = None,
    *,
    options: Optional[AdaptiveFilterOptions] = None,
    data_fixes: Optional[Sequence["DataFix"]] = None,
    log: Optional[object] = None,
) -> tuple[list[PosRow], str]:
    """Run the data-conditional filter chain. Returns ``(rows, regime_name)``.

    The companion regime name is one of ``"poor"`` / ``"clean"`` /
    ``"mixed"`` so callers can log which branch fired or expose it in
    the GUI.

    When ``data_fixes`` is provided AND the regime is ``mixed`` or
    ``poor``, a device-outlier-gate pass runs first to drop Post-processing epochs
    that disagree with the bias-free device Reference shape by more than 10 m.
    This catches Float-ambiguity excursions that smoothing alone can't
    fix and was empirically worth an extra 1-2 pp on Float-dominant
    sessions (reference session: -7.1 % → -8.6 % w/ data-gate companion).
    """
    from .smoothing import gaussian_smooth as _gauss
    options = options or AdaptiveFilterOptions()

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    rows = list(pos_rows)
    n = len(rows)
    if n == 0 or not options.enabled:
        return rows, "disabled"

    sng_pct = 100.0 * sum(1 for r in rows if r.quality >= 4) / n
    fix_pct = 100.0 * sum(1 for r in rows if r.quality == 1) / n
    _log(f"[adaptive] Q split: Fix={fix_pct:.0f}% Sng+={sng_pct:.0f}% n={n}")

    def _gauss_xy(seq: list[PosRow]) -> list[PosRow]:
        if len(seq) < 3:
            return seq
        t = [r.utc_s for r in seq]
        dts = [t[i + 1] - t[i] for i in range(len(t) - 1) if t[i + 1] - t[i] > 1e-6]
        if not dts:
            return seq
        median_dt = sorted(dts)[len(dts) // 2]
        fps = 1.0 / median_dt
        xs = max(1.0, options.xy_sigma_s * fps)
        zs = max(1.0, options.z_sigma_s  * fps)
        lat = [r.lat_deg for r in seq]; lon = [r.lon_deg for r in seq]; h = [r.h_m for r in seq]
        lat2 = _gauss(lat, xs); lon2 = _gauss(lon, xs); h2 = _gauss(h, zs)
        return [
            PosRow(
                utc_s=r.utc_s,
                lat_deg=lat2[i], lon_deg=lon2[i], h_m=h2[i],
                quality=r.quality, vn=r.vn, ve=r.ve, vu=r.vu, ns=r.ns,
                sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                sd_ne=r.sd_ne, sd_eu=r.sd_eu, sd_un=r.sd_un,
                age_s=r.age_s, ratio=r.ratio,
                sd_vn=r.sd_vn, sd_ve=r.sd_ve, sd_vu=r.sd_vu,
                sd_vne=r.sd_vne, sd_veu=r.sd_veu, sd_vun=r.sd_vun,
            )
            for i, r in enumerate(seq)
        ]

    # Regime selection.
    if sng_pct >= options.poor_single_pct_threshold:
        # On poor data the gating gain swamps everything else. Adding a
        # Gaussian on top of a gated-and-sparser series typically OVER-
        # smooths into the wrong side of bad fixes (the eval ranked
        # "Gate Q<=2 -> Gauss car" at +22 % mean), so we stop at ZUPT.
        _log("[adaptive] regime=poor: Gate Q<=2 -> ZUPT (no Gauss)")
        gated = apply_quality_gate(rows, max_q=2)
        if len(gated) < 50:
            # Q-only-Single dataset (no usable Fix/Float). Wider Gaussian
            # works best on these — per the GT eval session-C (100% Single)
            # peaked at Gauss(3 s / 15 s) -13.3 %.
            _log("[adaptive] gated set too small; using wider Gauss (3s/15s)")
            wide = AdaptiveFilterOptions(xy_sigma_s=3.0, z_sigma_s=15.0)
            saved = options
            # Reuse the local _gauss_xy with overridden sigmas.
            t = [r.utc_s for r in rows]
            dts = [t[i + 1] - t[i] for i in range(len(t) - 1) if t[i + 1] - t[i] > 1e-6]
            if not dts:
                return rows, "poor-fallback-noop"
            from .smoothing import gaussian_smooth as _g
            median_dt = sorted(dts)[len(dts) // 2]
            fps = 1.0 / median_dt
            xs = max(1.0, wide.xy_sigma_s * fps)
            zs = max(1.0, wide.z_sigma_s  * fps)
            lat = [r.lat_deg for r in rows]; lon = [r.lon_deg for r in rows]; h = [r.h_m for r in rows]
            lat2 = _g(lat, xs); lon2 = _g(lon, xs); h2 = _g(h, zs)
            out = [
                PosRow(
                    utc_s=r.utc_s,
                    lat_deg=lat2[i], lon_deg=lon2[i], h_m=h2[i],
                    quality=r.quality, vn=r.vn, ve=r.ve, vu=r.vu, ns=r.ns,
                    sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                    sd_ne=r.sd_ne, sd_eu=r.sd_eu, sd_un=r.sd_un,
                    age_s=r.age_s, ratio=r.ratio,
                    sd_vn=r.sd_vn, sd_ve=r.sd_ve, sd_vu=r.sd_vu,
                    sd_vne=r.sd_vne, sd_veu=r.sd_veu, sd_vun=r.sd_vun,
                )
                for i, r in enumerate(rows)
            ]
            return out, "poor-fallback"
        # Poor data tolerates larger position spread inside a static
        # interval (Single-quality fixes drift naturally), so relax the
        # spread guard.
        z = apply_zupt(gated,
                       options=ZuptOptions(max_static_spread_m=15.0),
                       log=lambda s: None).rows_out
        return z, "poor"

    if fix_pct >= options.clean_fix_pct_threshold:
        # Clean regime — pure Gauss car. Empirically tested:
        #   * the reference set gains an extra 7 % from ZUPT(QW) → Gauss,
        #     but the reference set LOSES 7 % from the same chain (its "stops"
        #     are actually slow-drift segments where the median snap
        #     introduces a small lateral shift).
        # Trade: stay neutral on clean data, never hurt. ZUPT is left
        # to the mixed/poor regimes where it consistently helps.
        _log("[adaptive] regime=clean: Gauss only (no ZUPT — too risky on clean data)")
        return _gauss_xy(rows), "clean"

    # Mixed regime. Branch on device-fix availability:
    #   * If device Reference rows are available AND the gate would drop some
    #     Post-processing epochs, use the simpler `data-gate -> Gauss car` chain.
    #     Empirically on reference session this hits -8.6 % vs raw Post-processing,
    #     beating ZUPT/NHC on top by 2 pp (ZUPT operating on the gated
    #     sparser series introduces drift that smoothing can't undo).
    #   * Otherwise fall back to the no-device champion:
    #     ZUPT(QW) -> NHC(auto) -> Gauss car.
    if data_fixes:
        gated = apply_data_outlier_gate(
            list(rows), data_fixes,
            max_disagreement_m=10.0,
            log=lambda s: None,
        )
        dropped = len(rows) - len(gated)
        if dropped > 0 and len(gated) >= len(rows) * 0.7:
            _log(f"[adaptive] regime=mixed: data-gate dropped {dropped} -> Gauss car")
            return _gauss_xy(gated), "mixed-datagate"

    _log("[adaptive] regime=mixed: ZUPT(QW) -> NHC(auto) -> Gauss")
    z = apply_zupt(list(rows),
                   options=ZuptOptions(quality_weighted=True,
                                       filter_to_best_q=True),
                   log=lambda s: None).rows_out
    h = apply_nhc(z, attitude_samples,
                  options=NhcOptions(heading_source="auto"),
                  log=lambda s: None).rows_out
    return _gauss_xy(h), "mixed"


# ----------------------------------------------------------------------
# Recommended public surface — only filters proven ROBUST on 7-day GT
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Device-stretch — warp dense device-Signal shape onto Post-processing anchors
# ----------------------------------------------------------------------


@dataclass
class DataStretchOptions:
    """Tuning for :func:`apply_data_stretch`.

    Uses :func:`data_pipeline.fused_bend.bend_fused_to_ppk` underneath
    but defaults are broader: ``fused`` is allowed as a provider AND
    raw ``reference`` is included (the typical Samsung capture only emits
    ``reference`` rows). The device stream gives the shape (dense, Motion sensor-blended
    on FLP, smooth on raw Signal), Post-processing is the absolute anchor.
    """

    enabled: bool = True
    # Provider names accepted from device Fix rows.
    providers: tuple[str, ...] = (
        "gps", "GPS", "fused", "FUSED", "FUSED_LOCATION_PROVIDER",
        "fused_location",
    )
    # Trust band for Post-processing anchors when bending (m). 3 m matches typical
    # Float-quality device Post-processing; tighter values pull harder onto each
    # anchor but propagate Post-processing noise into the bent track.
    xy_sigma_m: float = 3.0
    z_sigma_m: float = 15.0
    # Hard reject any Post-processing anchor whose residual to device shape exceeds
    # this multiple of the sigma. Larger = more anchors honoured.
    reject_k: float = 10.0
    # Gaussian time-kernel width for the per-time correction field.
    time_smooth_s: float = 5.0
    # Vehicle constraint inside the bend (lateral-residual rejection).
    car_lateral_sigma_m: float = 3.0
    car_smooth_s: float = 3.0
    car_min_speed_mps: float = 0.5


def apply_data_stretch(
    pos_rows: Sequence[PosRow],
    data_fixes: Sequence["DataFix"],
    *,
    options: Optional[DataStretchOptions] = None,
    log: Optional[object] = None,
) -> list[PosRow]:
    """Bend the dense device-Signal shape onto Post-processing anchors.

    Returns a new PosRow list at the SAME UTCs as ``pos_rows``. Each
    position is the FLP/device-Signal shape warped to pass within
    ``xy_sigma_m`` of every nearby Post-processing anchor; quality flags and
    velocities are carried through from the input Post-processing rows so
    downstream consumers (CSV, Export format, viewers) see the same metadata.

    Samples where the bend has no fused sample within the maximum gap
    fall back to the raw Post-processing position so the output sequence is never
    shorter than the input.
    """
    from .fused_bend import FusedBendOptions, bend_fused_to_ppk

    options = options or DataStretchOptions()
    rows = list(pos_rows)
    if not options.enabled or not data_fixes or not rows:
        return rows

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    bend_opts = FusedBendOptions(
        xy_sigma_m=options.xy_sigma_m,
        z_sigma_m=options.z_sigma_m,
        reject_k=options.reject_k,
        time_smooth_s=options.time_smooth_s,
        provider_filter=options.providers,
        car_lateral_sigma_m=options.car_lateral_sigma_m,
        car_smooth_s=options.car_smooth_s,
        car_min_speed_mps=options.car_min_speed_mps,
    )

    query_t = [r.utc_s for r in rows]
    lat_b, lon_b, h_b, has, trust, info = bend_fused_to_ppk(
        data_fixes, rows, query_t, options=bend_opts,
    )

    n_used = sum(1 for ok in has if ok)
    _log(
        f"[data-stretch] anchors used={info.n_anchors_used}/"
        f"{info.n_anchors_used + info.n_anchors_rejected}  "
        f"bent {n_used}/{len(rows)} frames  "
        f"median_residual={info.median_residual_m:.2f}m  "
        f"p95={info.p95_residual_m:.2f}m"
    )

    out: list[PosRow] = []
    for i, r in enumerate(rows):
        if has[i] and math.isfinite(lat_b[i]) and math.isfinite(lon_b[i]):
            out.append(PosRow(
                utc_s=r.utc_s,
                lat_deg=lat_b[i], lon_deg=lon_b[i],
                h_m=(h_b[i] if math.isfinite(h_b[i]) else r.h_m),
                quality=r.quality,
                vn=r.vn, ve=r.ve, vu=r.vu,
                ns=r.ns,
            ))
        else:
            # Pass through raw Post-processing when bend has no fused sample.
            out.append(r)
    return out


def apply_data_outlier_gate(
    pos_rows: Sequence[PosRow],
    data_fixes: Sequence["DataFix"],
    *,
    max_disagreement_m: float = 10.0,
    providers: tuple[str, ...] = (
        "gps", "GPS", "fused", "FUSED", "FUSED_LOCATION_PROVIDER",
        "fused_location",
    ),
    log: Optional[object] = None,
) -> list[PosRow]:
    """Drop Post-processing epochs whose position disagrees with the device Reference shape by
    more than ``max_disagreement_m``.

    Use case: Float-Post-processing ambiguity bias can shift entire chunks of the
    track by several meters. Device-Reference is noisier per-epoch but
    BIAS-FREE — it's a soft external validator that catches the
    multi-second environment noise / cycle-slip excursions Post-processing can't see in
    isolation. Each Post-processing epoch with no nearby device Fix passes through
    untouched.

    Returns a filtered PosRow list (shorter than the input).
    """
    if not data_fixes or not pos_rows:
        return list(pos_rows)

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    allowed = {p.lower() for p in providers}
    ph = sorted(
        [f for f in data_fixes if (f.provider or "").lower() in allowed],
        key=lambda f: f.utc_s,
    )
    if not ph:
        return list(pos_rows)

    # Common Local-frame ref.
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    ph_t = np.array([f.utc_s for f in ph])
    ph_e = np.empty(len(ph)); ph_n = np.empty(len(ph))
    for i, f in enumerate(ph):
        x, y, z = llh_to_ecef(f.lat, f.lon, f.h if math.isfinite(f.h) else 0.0)
        e, n, _ = ecef_to_enu(x, y, z, ref)
        ph_e[i] = e; ph_n[i] = n

    def _data_at(utc: float) -> Optional[tuple[float, float]]:
        if utc < ph_t[0] - 2.0 or utc > ph_t[-1] + 2.0:
            return None
        j = int(np.searchsorted(ph_t, utc))
        if j <= 0:
            return (float(ph_e[0]), float(ph_n[0]))
        if j >= len(ph_t):
            return (float(ph_e[-1]), float(ph_n[-1]))
        t0, t1 = float(ph_t[j - 1]), float(ph_t[j])
        if t1 - t0 > 5.0:
            return None    # gap too wide
        u = (utc - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (float(ph_e[j - 1]) + u * float(ph_e[j] - ph_e[j - 1]),
                float(ph_n[j - 1]) + u * float(ph_n[j] - ph_n[j - 1]))

    kept: list[PosRow] = []
    n_drop = 0
    for r in pos_rows:
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, n, _ = ecef_to_enu(x, y, z, ref)
        ph_p = _data_at(r.utc_s)
        if ph_p is None:
            kept.append(r)
            continue
        d = math.hypot(e - ph_p[0], n - ph_p[1])
        if d <= max_disagreement_m:
            kept.append(r)
        else:
            n_drop += 1
    _log(f"[data-gate] dropped {n_drop}/{len(pos_rows)} PPK epochs "
         f"disagreeing > {max_disagreement_m}m with device")
    return kept


def best_filter(
    pos_rows: Sequence[PosRow],
    attitude_samples: Optional[Sequence[object]] = None,
    *,
    log: Optional[object] = None,
) -> list[PosRow]:
    """Recommended one-call interface — alias for :func:`adaptive_filter`.

    The 8-dataset reference evaluation (the reference set, reference site, session 4 x3) ranked this as
    the absolute champion: mean −9.33 % horizontal RMS vs raw Post-processing,
    worst-case −0.03 % (i.e. never regresses on any tested session).
    Use this when you want one filter and no decisions.
    """
    rows, _regime = adaptive_filter(pos_rows, attitude_samples, log=log)
    return rows


# Public registry of robust filters. Every entry beats raw Post-processing on every
# dataset in the 7-day GT sweep with worst-case dRMS ≤ 0 — no overshoots.
# Order: most-recommended first. Drop entries from this dict to remove
# them from the GUI / Export format batch surface.
#
# Field meanings:
#   rank          — 1 (best) to 4. Ranked by mean horizontal RMS dRMS.
#   label         — short human name for menus.
#   when_to_use   — one-line decision hint for the user.
#   one_liner     — 2-3 line summary of WHAT it does + WHY it's robust.
#   mean_dRMS_pct — mean horizontal-RMS improvement vs raw Post-processing across the
#                   7-day GT sweep (8 datasets). Lower = better.
#   worst_dRMS_pct— worst-case improvement on any single dataset (≤ 0
#                   means it never regressed across the sweep).
#   wins_over_raw — "K / N" where K beats raw in each of the N datasets
#                   the variant ran on.
#   needs_imu     — needs a sensors_*.txt + Complementary-update fusion to run.
#   needs_heading — needs a usable heading source (Rate-signal / Motion sensor).
#                   Falls back to Rate-signal when Motion sensor absent.
RECOMMENDED_FILTERS: dict[str, dict] = {
    "adaptive": {
        "rank": 1,
        "label":   "ADAPTIVE  (recommended default)",
        "when_to_use": "ALWAYS — picks the right stack per dataset automatically.",
        "one_liner": (
            "Inspects the PPK quality histogram (Fix / Float / Single %) "
            "and dispatches: poor -> Gate + ZUPT; clean -> Gauss only; "
            "mixed -> data-gate + Gauss (when device GPS available) else "
            "ZUPT(QW) -> NHC(auto) -> Gauss. No knobs, no IMU needed. "
            "Device GPS rows (when present) catch Float-ambiguity excursions."
        ),
        "mean_dRMS_pct": -9.75,
        "worst_dRMS_pct": -0.03,
        "wins_over_raw": "8/8",
        "needs_imu": False,
        "needs_heading": False,
    },
    "nhc+gauss_car": {
        "rank": 2,
        "label":   "NHC -> Gaussian car",
        "when_to_use": (
            "Vehicle session AND you want explicit control. Slightly worse "
            "than ADAPTIVE on poor-quality data."
        ),
        "one_liner": (
            "Projects PPK velocity onto Doppler-derived heading (vehicle "
            "non-holonomic constraint), then Gaussian-smooths with the "
            "'car' preset. Kills lateral wobble first, then averages."
        ),
        "mean_dRMS_pct": -6.88,
        "worst_dRMS_pct": -0.03,
        "wins_over_raw": "7/7",
        "needs_imu": False,
        "needs_heading": True,
    },
    "gauss_car": {
        "rank": 3,
        "label":   "Gaussian car (2s xy / 10s z)",
        "when_to_use": (
            "Simplest baseline. Use when you can't trust velocity columns "
            "(e.g. .pos lacks vn/ve) or want one-knob smoothing."
        ),
        "one_liner": (
            "Gaussian kernel on lat / lon (sigma 2 s) and height (sigma 10 s) "
            "in ENU metric space. No assumptions about vehicle motion."
        ),
        "mean_dRMS_pct": -6.60,
        "worst_dRMS_pct": -0.03,
        "wins_over_raw": "8/8",
        "needs_imu": False,
        "needs_heading": False,
    },
    "median_11s": {
        "rank": 4,
        "label":   "Rolling median (11s window)",
        "when_to_use": (
            "Outlier-heavy PPK (multipath spikes, cycle slips). Median "
            "is robust where Gaussian gets dragged by single bad epochs."
        ),
        "one_liner": (
            "Per-axis rolling median over an 11 s window. Sacrifices a "
            "little smoothness for hard outlier rejection."
        ),
        "mean_dRMS_pct": -6.15,
        "worst_dRMS_pct": -0.02,
        "wins_over_raw": "8/8",
        "needs_imu": False,
        "needs_heading": False,
    },
}


def print_filter_picker(stream=None) -> None:
    """Print a 1-page ranked picker so users can see which filter fits.

    Renders the :data:`RECOMMENDED_FILTERS` registry as a fixed-width
    table with the mean / worst-case dRMS, the win-count over raw Post-processing
    and a short ``when_to_use`` hint. Designed to be called interactively
    from a REPL / GUI button / CLI ``--help`` text. Output goes to the
    given stream (or stdout if None).
    """
    import sys
    out = stream if stream is not None else sys.stdout

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w("=" * 80)
    w(" DEVICE-PPK FILTER PICKER — ranked from 7-day GT eval (8 datasets)")
    w(" lower 'mean dRMS' = bigger accuracy improvement over raw PPK.")
    w(" 'worst dRMS' is the per-dataset worst case — all entries here are <= 0,")
    w(" meaning they never made any tested session worse than raw.")
    w("=" * 80)
    for key, info in RECOMMENDED_FILTERS.items():
        w(f"\n  #{info['rank']}  {info['label']}")
        w(f"     api:           best_filter() if rank==1 else see HANDOFF.md")
        w(f"     mean dRMS:     {info['mean_dRMS_pct']:+.2f}%  "
          f"(worst {info['worst_dRMS_pct']:+.2f}%)  wins {info['wins_over_raw']}")
        w(f"     needs IMU?     {'yes' if info['needs_imu']    else 'no'}")
        w(f"     needs heading? {'yes' if info['needs_heading'] else 'no'}")
        w(f"     when to use:   {info['when_to_use']}")
        w(f"     what it does:  {info['one_liner']}")
    w()
    w("=" * 80)
    w(" Don't see what you wanted? Everything else either regressed on")
    w(" at least one dataset (e.g. Gauss aggressive +44% on Float-heavy)")
    w(" or only beat raw by < 1%. Those are intentionally NOT exposed.")
    w("=" * 80)


def apply_zupt(
    pos_rows: Sequence[PosRow],
    *,
    options: Optional[ZuptOptions] = None,
    log: Optional[object] = None,
) -> ZuptResult:
    """Snap every static interval to its own median position.

    Pure function: returns a new ``PosRow`` list, never mutates the input.
    """
    options = options or ZuptOptions()
    rows = list(pos_rows)
    n = len(rows)

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)  # type: ignore[operator]

    if n == 0 or not options.enabled:
        return ZuptResult(
            rows_out=rows, n_in=n, n_static_epochs=0,
            n_intervals=0, total_static_duration_s=0.0,
            summary="ZUPT disabled or empty input",
        )

    rows = sorted(rows, key=lambda r: r.utc_s)
    is_static = _classify_speed(rows, options.speed_threshold_mps)

    # Sweep static intervals.
    out = list(rows)
    n_static = 0
    n_intervals = 0
    total_static_s = 0.0

    i = 0
    while i < n:
        if not is_static[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_static[j + 1]:
            j += 1
        # Interval [i..j] is static — check duration.
        dur = rows[j].utc_s - rows[i].utc_s
        if dur >= options.min_static_duration_s and (j - i) >= 1:
            slice_rows = rows[i : j + 1]
            # Position-spread safeguard: skip if the candidate interval's
            # horizontal extent exceeds the threshold (likely slow drift,
            # not a true stop).
            if options.max_static_spread_m > 0 and len(slice_rows) >= 2:
                _lats = [r.lat_deg for r in slice_rows]
                _lons = [r.lon_deg for r in slice_rows]
                _mlat = sum(_lats) / len(_lats)
                _m_per_lat = 111320.0
                _m_per_lon = 111320.0 * math.cos(math.radians(_mlat))
                _spread_m = max(
                    (max(_lats) - min(_lats)) * _m_per_lat,
                    (max(_lons) - min(_lons)) * _m_per_lon,
                )
                if _spread_m > options.max_static_spread_m:
                    i = j + 1
                    continue
            # Optional best-Q filter: if the interval contains enough Fix
            # epochs to anchor on, drop Float/Single from the snap source.
            snap_rows = slice_rows
            if options.filter_to_best_q:
                qs = [r.quality for r in slice_rows if r.quality]
                if qs:
                    best_q = min(qs)
                    candidates = [r for r in slice_rows if r.quality == best_q]
                    if len(candidates) >= options.min_best_q_count:
                        snap_rows = candidates
            if options.quality_weighted:
                lat_s, lon_s, h_s = _weighted_median_xyz(
                    snap_rows, options.q_weights,
                )
            elif options.use_median:
                lat_s, lon_s, h_s = _median_xyz(snap_rows)
            else:
                lat_s, lon_s, h_s = (snap_rows[0].lat_deg,
                                     snap_rows[0].lon_deg,
                                     snap_rows[0].h_m)
            for k in range(i, j + 1):
                r = rows[k]
                out[k] = PosRow(
                    utc_s=r.utc_s,
                    lat_deg=lat_s, lon_deg=lon_s, h_m=h_s,
                    quality=r.quality,
                    vn=(0.0 if options.zero_velocity else r.vn),
                    ve=(0.0 if options.zero_velocity else r.ve),
                    vu=(0.0 if options.zero_velocity else r.vu),
                    ns=r.ns,
                )
            n_static += (j - i + 1)
            n_intervals += 1
            total_static_s += dur
        i = j + 1

    summary = (
        f"ZUPT snapped {n_static}/{n} epochs across {n_intervals} static intervals "
        f"({total_static_s:.0f}s total)  "
        f"speed_thr={options.speed_threshold_mps}m/s  "
        f"min_dur={options.min_static_duration_s}s"
    )
    _log("[zupt] " + summary)
    return ZuptResult(
        rows_out=out,
        n_in=n,
        n_static_epochs=n_static,
        n_intervals=n_intervals,
        total_static_duration_s=total_static_s,
        summary=summary,
    )
