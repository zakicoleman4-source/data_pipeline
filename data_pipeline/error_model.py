"""Empirical measurement-error model: per-(quality, ns-bucket) horizontal 1-sigma
fit from cross-device residuals (robust MAD). Honest per-epoch uncertainty."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


def _ns_bucket(ns: int) -> str:
    if ns is None: return "na"
    if ns < 6: return "lo"
    if ns < 10: return "mid"
    return "hi"


def _key(row) -> Tuple:
    return (int(getattr(row, "quality", 0)), _ns_bucket(int(getattr(row, "ns", 0) or 0)))


def _mad_sigma(vals: List[float]) -> float:
    """Robust 1-sigma estimate of the *magnitude* of a bin's horizontal
    errors: values are already non-negative distances (never signed
    residuals), so dispersion-only estimators (plain MAD-about-median) can
    degenerate to 0 when a bin is tight or near-constant — e.g. a
    quantised/synthetic fixture, or a bin whose fix quality is genuinely
    very consistent. That would collide with the "unseen bin" sentinel in
    ErrorModel.sigma_h and silently discard a real, tightly-estimated bin.
    We combine the classic MAD-about-median (robust to outliers) with the
    RMS of the values themselves (captures the typical error magnitude even
    when spread is ~0), taking the larger of the two so a tight-but-large
    bin is never underestimated.
    """
    if not vals: return float("nan")
    n = len(vals)
    s = sorted(vals); med = s[n // 2]
    dev = sorted(abs(v - med) for v in vals)
    mad_sigma = 1.4826 * dev[n // 2]
    rms = math.sqrt(sum(v * v for v in vals) / n)
    sigma = max(mad_sigma, rms)
    return sigma if sigma > 0.0 else 1e-6


@dataclass
class ErrorModel:
    bins: Dict[Tuple, float]
    global_sigma: float

    def sigma_h(self, row) -> float:
        v = self.bins.get(_key(row))
        if v is None or not math.isfinite(v) or v <= 0:
            return self.global_sigma
        return v


def fit_error_model(samples: List[Tuple[object, float]]) -> ErrorModel:
    """samples = [(pos_row, horizontal_error_m), ...] from cross-device residuals."""
    by_bin: Dict[Tuple, List[float]] = {}
    allv: List[float] = []
    for row, err in samples:
        if not math.isfinite(err): continue
        by_bin.setdefault(_key(row), []).append(err)
        allv.append(err)
    g = _mad_sigma(allv) or 1.0
    bins = {k: _mad_sigma(v) for k, v in by_bin.items() if len(v) >= 5}
    return ErrorModel(bins=bins, global_sigma=(g if g > 0 else 1.0))
