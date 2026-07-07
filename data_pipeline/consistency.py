"""Filter consistency: Normalized Estimation Error Squared (NEES).
A filter whose reported covariance is honest has mean NEES ~ state dim, inside
the chi-square 95% interval. Larger => overconfident."""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2


def nees(errors: np.ndarray, covs: np.ndarray, alpha: float = 0.05) -> dict:
    errors = np.asarray(errors, float)
    covs = np.asarray(covs, float)
    n, d = errors.shape
    vals = np.empty(n)
    for i in range(n):
        Ci = covs[i] + 1e-12 * np.eye(d)
        vals[i] = float(errors[i] @ np.linalg.solve(Ci, errors[i]))
    mean_nees = float(np.mean(vals))
    lo = chi2.ppf(alpha / 2, n * d) / n
    hi = chi2.ppf(1 - alpha / 2, n * d) / n
    return {"mean_nees": mean_nees, "dim": d, "lo": lo, "hi": hi,
            "consistent": bool(lo <= mean_nees <= hi)}
