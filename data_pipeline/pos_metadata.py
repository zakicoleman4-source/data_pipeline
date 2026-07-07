"""Per-epoch metadata export + calibration for The external solver ``.pos`` files.

Filters (Recursive-filter, FGO, ADAPTIVE) need realistic per-epoch trust weights.
The external solver's self-reported sigmas (``sd_n/sd_e/sd_u``) are systematically too
tight by 5-30x on device Post-processing — empirical actual_error / solver_sigma >> 1.

This module:
  - ``to_metadata_csv`` dumps full per-epoch metadata for downstream consumers
  - ``calibrate_sigma_inflation`` estimates a per-session sigma scale factor
    using local-variance-of-position vs The external solver sigma (no GT needed)
  - ``effective_sigma`` returns realistic per-epoch sigmas (The external solver × inflation
    factor)

Columns dumped (one row per .pos epoch):
  utc_s, lat_deg, lon_deg, h_m, quality, ns, ratio, age_s,
  sd_n, sd_e, sd_u, sd_ne, sd_eu, sd_un,
  vn, ve, vu, sd_vn, sd_ve, sd_vu, sd_vne, sd_veu, sd_vun,
  speed_mps, sd_h_m, sd_v_h_mps,
  effective_sd_h_m, ratio_norm, quality_score
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .parsers import PosRow


def calibrate_sigma_inflation(rows: list[PosRow], window_s: float = 10.0) -> float:
    """Estimate sigma inflation factor: actual local std / The external solver sd_h.

    Strategy: compute local std of position over a sliding ``window_s``
    window; compare to the median solver ``sd_h``. The ratio represents how
    overconfident The external solver sigmas are for downstream filter R = sigma**2.

    Returns 1.0 when sigmas are absent or non-informative. Floors at 1.0.
    """
    if not rows:
        return 1.0

    # Estimate sample rate from row timestamps
    ts = np.array([r.utc_s for r in rows])
    if len(ts) < 5:
        return 1.0
    dt_med = float(np.median(np.diff(ts)))
    if not (0.0 < dt_med < 5.0):
        return 1.0
    win = max(3, int(round(window_s / dt_med)))

    # Use lat/lon converted to local metric via small-angle approx
    lat0 = rows[0].lat_deg
    cos_lat = math.cos(math.radians(lat0))
    n_arr = np.array([(r.lat_deg - lat0) * 111139.0 for r in rows])
    e_arr = np.array([(r.lon_deg - rows[0].lon_deg) * 111139.0 * cos_lat for r in rows])

    # Local std via centred window (detrend with median)
    def local_std(arr):
        out = np.full(len(arr), np.nan)
        for i in range(len(arr)):
            lo = max(0, i - win // 2); hi = min(len(arr), i + win // 2)
            if hi - lo < 3:
                continue
            seg = arr[lo:hi]
            # Detrend with linear fit (CV model)
            x = np.arange(len(seg))
            try:
                p = np.polyfit(x, seg, 1)
                res = seg - (p[0] * x + p[1])
                out[i] = float(np.std(res))
            except (np.linalg.LinAlgError, ValueError, TypeError):
                # Singular fit (all-equal segment) or NaN in seg — skip
                continue
        return out

    n_std = local_std(n_arr)
    e_std = local_std(e_arr)
    h_std = np.sqrt(n_std ** 2 + e_std ** 2) / math.sqrt(2.0)

    sd_h = np.sqrt(
        np.array([r.sd_n for r in rows]) ** 2 + np.array([r.sd_e for r in rows]) ** 2
    ) / math.sqrt(2.0)

    valid = np.isfinite(h_std) & np.isfinite(sd_h) & (sd_h > 1e-6)
    if not np.any(valid):
        return 1.0
    ratios = h_std[valid] / sd_h[valid]
    # Robust: median; clip to [1, 100]
    factor = float(np.median(ratios))
    return float(max(1.0, min(factor, 100.0)))


def effective_sd_h(rows: list[PosRow], inflation: Optional[float] = None) -> np.ndarray:
    """Per-epoch horizontal sigma after inflation correction.

    Returns ``sd_h * inflation_factor``. If ``inflation`` is None, derived
    via ``calibrate_sigma_inflation``. Use as a realistic R-matrix input.
    """
    sd_h = np.sqrt(
        np.array([r.sd_n for r in rows]) ** 2 + np.array([r.sd_e for r in rows]) ** 2
    ) / math.sqrt(2.0)
    if inflation is None:
        inflation = calibrate_sigma_inflation(rows)
    return sd_h * inflation


def quality_score(rows: list[PosRow], inflation: Optional[float] = None) -> np.ndarray:
    """Per-epoch composite quality score in [0, 1].

    Combines:
      - Q (The external solver quality flag): Q=1 -> 1.0, Q=2 -> 0.5, Q=4 -> 0.2, Q=5 -> 0.05
      - ns (source count, normalised by 20)
      - ratio (AR test, normalised by 3.0)
      - effective_sd_h (inverse-normalised by 5 m)

    Use to weight epochs in filters / FGO.
    """
    n = len(rows)
    if n == 0:
        return np.array([])
    Q = np.array([r.quality for r in rows])
    ns = np.array([r.ns for r in rows])
    ratio = np.array([r.ratio for r in rows], float)
    eff_sd = effective_sd_h(rows, inflation)

    q_map = {1: 1.0, 2: 0.5, 4: 0.2, 5: 0.05}
    q_score = np.array([q_map.get(int(q), 0.0) for q in Q])
    ns_score = np.clip(ns / 20.0, 0.0, 1.0)
    ratio_score = np.clip(np.where(np.isfinite(ratio), ratio, 0.0) / 3.0, 0.0, 1.0)
    # NaN sd treated as lowest trust (sd_score=0). Otherwise clip 1 - sd/5.
    sd_safe = np.where(np.isfinite(eff_sd), eff_sd, np.inf)
    sd_score = np.clip(1.0 - sd_safe / 5.0, 0.0, 1.0)

    # Composite — weighted average of (q, sd, ns, ratio)
    return (q_score * 0.4 + sd_score * 0.4 + ns_score * 0.1 + ratio_score * 0.1)


def to_metadata_csv(rows: list[PosRow], out_path: Path, inflation: Optional[float] = None) -> None:
    """Dump full per-epoch metadata to CSV for downstream filter consumers.

    Atomic write: streams to ``<out_path>.tmp`` then os.replace.
    """
    import os
    if not rows:
        return
    inflation = inflation if inflation is not None else calibrate_sigma_inflation(rows)
    eff_sd = effective_sd_h(rows, inflation)
    qscore = quality_score(rows, inflation)
    speeds = np.sqrt(
        np.array([r.vn for r in rows], float) ** 2
        + np.array([r.ve for r in rows], float) ** 2
    )
    sd_v_h = np.sqrt(
        np.array([r.sd_vn for r in rows], float) ** 2
        + np.array([r.sd_ve for r in rows], float) ** 2
    ) / math.sqrt(2.0)
    sd_h = np.sqrt(
        np.array([r.sd_n for r in rows], float) ** 2
        + np.array([r.sd_e for r in rows], float) ** 2
    ) / math.sqrt(2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(out_path) + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "utc_s", "lat_deg", "lon_deg", "h_m", "quality", "ns",
            "ratio", "age_s",
            "sd_n", "sd_e", "sd_u", "sd_ne", "sd_eu", "sd_un",
            "vn", "ve", "vu", "sd_vn", "sd_ve", "sd_vu",
            "sd_vne", "sd_veu", "sd_vun",
            "speed_mps", "sd_h_m", "sd_v_h_mps",
            "effective_sd_h_m", "inflation", "quality_score",
        ])
        for i, r in enumerate(rows):
            w.writerow([
                f"{r.utc_s:.6f}", f"{r.lat_deg:.9f}", f"{r.lon_deg:.9f}", f"{r.h_m:.4f}",
                r.quality, r.ns,
                f"{r.ratio:.3f}" if math.isfinite(r.ratio) else "",
                f"{r.age_s:.3f}" if math.isfinite(r.age_s) else "",
                f"{r.sd_n:.4f}" if math.isfinite(r.sd_n) else "",
                f"{r.sd_e:.4f}" if math.isfinite(r.sd_e) else "",
                f"{r.sd_u:.4f}" if math.isfinite(r.sd_u) else "",
                f"{r.sd_ne:.4f}" if math.isfinite(r.sd_ne) else "",
                f"{r.sd_eu:.4f}" if math.isfinite(r.sd_eu) else "",
                f"{r.sd_un:.4f}" if math.isfinite(r.sd_un) else "",
                f"{r.vn:.5f}" if math.isfinite(r.vn) else "",
                f"{r.ve:.5f}" if math.isfinite(r.ve) else "",
                f"{r.vu:.5f}" if math.isfinite(r.vu) else "",
                f"{r.sd_vn:.5f}" if math.isfinite(r.sd_vn) else "",
                f"{r.sd_ve:.5f}" if math.isfinite(r.sd_ve) else "",
                f"{r.sd_vu:.5f}" if math.isfinite(r.sd_vu) else "",
                f"{r.sd_vne:.5f}" if math.isfinite(r.sd_vne) else "",
                f"{r.sd_veu:.5f}" if math.isfinite(r.sd_veu) else "",
                f"{r.sd_vun:.5f}" if math.isfinite(r.sd_vun) else "",
                f"{float(speeds[i]):.4f}",
                f"{float(sd_h[i]):.4f}" if math.isfinite(sd_h[i]) else "",
                f"{float(sd_v_h[i]):.4f}" if math.isfinite(sd_v_h[i]) else "",
                f"{float(eff_sd[i]):.4f}",
                f"{inflation:.3f}",
                f"{float(qscore[i]):.4f}",
            ])
    os.replace(tmp_path, out_path)
