"""Score one path against a reference (same drive), matched by Reference time.

Horizontal error is reported after removing the median east/north offset, so a
constant datum shift between the two solves does not masquerade as noise. Used
to measure accuracy against cross-device consensus (no ground truth needed).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict

import numpy as np

_MLAT = 111320.0


def _read(path: Path, time_col: str):
    t, lat, lon = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row[time_col]))
                lat.append(float(row["lat_deg"]))
                lon.append(float(row["lon_deg"]))
            except (KeyError, ValueError):
                continue
    return np.array(t), np.array(lat), np.array(lon)


def score_trajectories(ref_csv, test_csv, *, time_col: str = "gpstime",
                       max_dt_s: float = 0.05) -> Dict[str, float]:
    tr, latr, lonr = _read(Path(ref_csv), time_col)
    tt, latt, lont = _read(Path(test_csv), time_col)
    if tr.size == 0 or tt.size == 0:
        return {"n": 0, "median_offset_m": float("nan"), "two_sigma_m": float("nan"),
                "max_m": float("nan"), "le1m_pct": float("nan"), "rmse_m": float("nan")}
    order = np.argsort(tr)
    tr, latr, lonr = tr[order], latr[order], lonr[order]
    idx = np.searchsorted(tr, tt)
    de = []; dn = []
    lat0 = float(latr[0])
    mlon = _MLAT * math.cos(math.radians(lat0))
    for k, ti in enumerate(tt):
        cands = [j for j in (idx[k] - 1, idx[k]) if 0 <= j < tr.size]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(tr[j] - ti))
        if abs(tr[j] - ti) > max_dt_s:
            continue
        dn.append((latt[k] - latr[j]) * _MLAT)
        de.append((lont[k] - lonr[j]) * mlon)
    de = np.array(de); dn = np.array(dn)
    if de.size == 0:
        return {"n": 0, "median_offset_m": float("nan"), "two_sigma_m": float("nan"),
                "max_m": float("nan"), "le1m_pct": float("nan"), "rmse_m": float("nan")}
    off_e = float(np.median(de)); off_n = float(np.median(dn))
    median_offset_m = math.hypot(off_e, off_n)
    err = np.hypot(de - off_e, dn - off_n)
    return {
        "n": int(err.size),
        "median_offset_m": round(median_offset_m, 4),
        "two_sigma_m": round(float(np.percentile(err, 95.45)), 4),
        "max_m": round(float(np.max(err)), 4),
        "le1m_pct": round(float(np.mean(err <= 1.0) * 100.0), 2),
        "rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 4),
    }
