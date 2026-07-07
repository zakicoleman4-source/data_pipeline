"""Per-epoch position sigma derived from Post-processing source count (``ns``).

Empirical fit against pooled the reference set+reference site Post-processing vs reference (n=10005 epochs).
Horizontal logistic fit: high sigma at low ns, saturating high-ns floor.

  sigma_h(ns) = lo + (hi - lo) / (1 + exp((ns - k) / w))

Defaults from curve_fit on pooled per-ns P50 / 0.6745:
  lo = 0.10 m   hi = 6.06 m   k = 10.23   w = 4.18

Vertical scaled by ``v_over_h_ratio`` (default 2.5) — data Signal vertical
is consistently ~2-3x noisier than horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NsSigmaParams:
    lo: float = 0.10
    hi: float = 6.06
    k: float = 10.23
    w: float = 4.18
    v_over_h_ratio: float = 2.5


def sigma_h_from_ns(ns: np.ndarray | float, params: NsSigmaParams | None = None) -> np.ndarray | float:
    p = params or NsSigmaParams()
    ns_arr = np.asarray(ns, dtype=np.float64)
    sig = p.lo + (p.hi - p.lo) / (1.0 + np.exp((ns_arr - p.k) / p.w))
    if np.isscalar(ns):
        return float(sig)
    return sig


def sigma_v_from_ns(ns: np.ndarray | float, params: NsSigmaParams | None = None) -> np.ndarray | float:
    p = params or NsSigmaParams()
    return sigma_h_from_ns(ns, p) * p.v_over_h_ratio


@dataclass(frozen=True)
class AdaptiveBwParams:
    """Per-epoch smoothing bandwidth as a function of ns.

    Bandwidth in seconds is interpolated linearly between
    ``sigma_clean_s`` at ``ns_clean`` (or above) and ``sigma_noisy_s`` at
    ``ns_noisy`` (or below). Beyond either edge it clamps.

    Defaults grid-searched across reference session, a secondary test session,
    session 4 session-A / session-B / code-only session, session 5 reference session / device-G:
      sigma_clean_s=1.0   (ns>=10 -> moderate window)
      sigma_noisy_s=2.0   (ns<=3  -> wider window, denoise environment noise)
      ns_clean=10, ns_noisy=3
    Result: mean hRMSE -2.06 %, worst +0.52 %, wins 7/8 vs uniform xy=2s.
    Earlier aggressive defaults (sc=0.3, sn=3.0, nc=18) overfit reference site
    reference session and regressed by +4 % on session 5 device-G.
    """
    sigma_clean_s: float = 1.0
    sigma_noisy_s: float = 2.0
    ns_clean: float = 10.0
    ns_noisy: float = 3.0
    z_scale: float = 3.0  # vertical bandwidth multiplier


def sigma_samples_from_ns(
    ns: np.ndarray,
    rate_hz: float,
    params: AdaptiveBwParams | None = None,
    axis: str = "h",
) -> np.ndarray:
    """Per-epoch smoothing sigma in samples for use with
    ``gaussian_smooth_adaptive_bw``.

    ``axis`` in {"h", "v"} — vertical bandwidth scaled by ``z_scale``.
    Epochs with ns <= 0 or non-finite ns are treated as "unknown" and
    mapped to the clean-side sigma (no adaptive inflation). Callers
    that want to skip ns-adaptive entirely on files with no ns column
    should consult ``ns_is_informative`` first.
    """
    if axis not in {"h", "v"}:
        raise ValueError(f"sigma_samples_from_ns: axis must be 'h' or 'v', got {axis!r}.")
    p = params or AdaptiveBwParams()
    ns_arr = np.asarray(ns, dtype=np.float64)
    unknown = ~np.isfinite(ns_arr) | (ns_arr <= 0)
    span = max(1e-9, p.ns_clean - p.ns_noisy)
    frac = np.clip((p.ns_clean - ns_arr) / span, 0.0, 1.0)
    sigma_s = p.sigma_clean_s + (p.sigma_noisy_s - p.sigma_clean_s) * frac
    sigma_s = np.where(unknown, p.sigma_clean_s, sigma_s)
    if axis == "v":
        sigma_s = sigma_s * p.z_scale
    return sigma_s * max(rate_hz, 1e-9)


def ns_is_informative(ns: np.ndarray, min_nonzero_frac: float = 0.10) -> bool:
    """Return True if the ``ns`` series carries enough non-zero counts to
    drive adaptive smoothing. The external solver .pos variants without the ns column
    return zeros for every epoch; in that case adaptive smoothing has no
    signal and uniform Gaussian should be used.
    """
    ns_arr = np.asarray(ns, dtype=np.float64)
    if ns_arr.size == 0:
        return False
    nonzero_frac = float(np.mean((ns_arr > 0) & np.isfinite(ns_arr)))
    return nonzero_frac >= min_nonzero_frac


def weights_from_ns(ns: np.ndarray, params: NsSigmaParams | None = None, axis: str = "h") -> np.ndarray:
    """Inverse-variance weights for weighted smoothing / WLS.

    ``axis`` in {"h", "v"}. Returns ``1 / sigma**2`` per epoch.
    Non-finite ns -> zero weight (epoch contributes nothing).
    """
    if axis not in {"h", "v"}:
        raise ValueError(f"weights_from_ns: axis must be 'h' or 'v', got {axis!r}.")
    p = params or NsSigmaParams()
    ns_arr = np.asarray(ns, dtype=np.float64)
    mask = np.isfinite(ns_arr) & (ns_arr > 0)
    sig = np.where(mask, sigma_h_from_ns(ns_arr, p), np.inf)
    if axis == "v":
        sig = sig * p.v_over_h_ratio
    w = np.zeros_like(ns_arr)
    w[mask] = 1.0 / (sig[mask] ** 2)
    return w
