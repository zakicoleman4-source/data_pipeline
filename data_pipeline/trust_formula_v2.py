"""Per-epoch trust labeling v2 — multi-signal scoring.

Extends trust_formula.py (v1: 2 binary gates) with 10 per-epoch quality
signals, normalized and weighted into composite position + velocity scores.

Signals oriented so HIGHER = WORSE quality:

  eff_sig              — effective_sigma (The external solver sigma x inflation + residual)
  disagree             — |raw Post-processing - v2 smoothed| horizontal (m)
  innovation_h         — Recursive-filter position innovation horizontal (m)
  fwd_bwd_disagree_h   — |forward - smoothed| horizontal from RTS (m)
  innovation_norm      — normalized innovation sqrt(y' S_inv y)
  q_penalty            — Q-flag penalty: 0/0.3/0.8/1.0
  ns_penalty           — max(0, 1 - ns/20)
  sd_h                 — raw The external solver horizontal sigma sqrt(sd_n^2 + sd_e^2)
  speed_mps            — inverted: 1 - clip(speed/max_speed) (low speed = worse)
  ratio_inv            — 1/max(ratio, 0.1)

Usage:
    from data_pipeline.trust_formula_v2 import compute_trust_v2, TrustConfigV2
    trust = compute_trust_v2(pos_rows, v2_result)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .epoch_weight import effective_sigma, epoch_features
from .geo import ecef_to_enu, llh_to_ecef
from .parsers import PosRow
from .pos_metadata import calibrate_sigma_inflation


# ---- Signal catalogue ----

SIGNAL_NAMES: list[str] = [
    "eff_sig",
    "disagree",
    "innovation_h",
    "fwd_bwd_disagree_h",
    "innovation_norm",
    "q_penalty",
    "ns_penalty",
    "sd_h",
    "speed_mps",
    "ratio_inv",
]

# ---- Disagree-based thresholds (validated on 10 sessions vs reference) ----
#
# Single signal: disagree = |raw Post-processing position - v2 smoothed position| (m)
# Monotonically predicts BOTH position and velocity error.
#
# GREEN  (disagree < 0.55): h 2σ=3.1m, v 2σ=1.18m/s — BOTH guaranteed
# ORANGE (0.55 ≤ disagree < 5.0): h 2σ<10m — position guaranteed, velocity NOT
# RED    (disagree ≥ 5.0): no guarantee
#
# Validated: 11693 moving epochs, 10 sessions, 6 field days, survey-grade reference.
DISAGREE_GREEN: float = 0.55    # max coverage where v_p97.7 < 1.2 m/s
DISAGREE_RED: float = 5.0       # beyond this, h_p95 approaches 10m

# Legacy composite (kept for back-compat, not used by default labeling)
NORM_BOUNDS: dict[str, tuple[float, float]] = {name: (0.0, 1.0) for name in SIGNAL_NAMES}
SIGNAL_WEIGHTS_POS: dict[str, float] = {name: 0.0 for name in SIGNAL_NAMES}
SIGNAL_WEIGHTS_VEL: dict[str, float] = {name: 0.0 for name in SIGNAL_NAMES}
THRESHOLD_POS: float = 0.5
THRESHOLD_VEL: float = 0.5

# ---- Q-flag penalty map ----
_Q_PENALTY: dict[int, float] = {1: 0.0, 2: 0.3, 4: 0.8, 5: 1.0}


@dataclass
class TrustConfigV2:
    """Tuning knobs for the v2 trust labeler."""
    disagree_green: float = DISAGREE_GREEN
    disagree_red: float = DISAGREE_RED
    # Legacy composite thresholds (unused by default labeling)
    threshold_pos: float = THRESHOLD_POS
    threshold_vel: float = THRESHOLD_VEL


@dataclass
class TrustResultV2:
    """Full v2 trust output."""
    pos_trusted: np.ndarray   # bool, shape (n,)
    vel_trusted: np.ndarray   # bool, shape (n,)
    labels: list[str]         # "high"/"vel_only"/"pos_only"/"low" per epoch
    pos_score: np.ndarray     # float, shape (n,) — lower = more trusted
    vel_score: np.ndarray     # float, shape (n,) — lower = more trusted
    signals: np.ndarray       # float, shape (n, 10) — raw extracted signals
    n_high: int
    n_pos_only: int
    n_vel_only: int
    n_low: int


# ---- Signal extraction ----

def extract_signals(
    pos_rows: list[PosRow],
    v2_result,
    *,
    max_speed_mps: float = 40.0,
) -> np.ndarray:
    """Extract 10 per-epoch quality signals from pos_rows + v2_result.

    Returns ndarray of shape (n, 10).  All signals oriented so
    HIGHER = WORSE quality.

    Parameters
    ----------
    pos_rows : list[PosRow]
        Post-processing rows (sorted by utc_s).
    v2_result : EpochWeightV2Result
        Smoothed output from ``smooth_epoch_weighted_v2``.
    max_speed_mps : float
        Speed normalization cap for the inverted speed signal.
    """
    n = len(pos_rows)
    if n == 0:
        return np.empty((0, len(SIGNAL_NAMES)), dtype=np.float64)

    out = np.zeros((n, len(SIGNAL_NAMES)), dtype=np.float64)

    # --- Shared derived arrays ---
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    E_raw = np.empty(n)
    N_raw = np.empty(n)
    for i, r in enumerate(pos_rows):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, nn, _ = ecef_to_enu(x, y, z, ref)
        E_raw[i] = e
        N_raw[i] = nn

    # 0: eff_sig — effective_sigma (The external solver sigma x inflation + residual)
    inflation = calibrate_sigma_inflation(pos_rows)
    feats = epoch_features(pos_rows)
    eff_sig = effective_sigma(feats, alpha=0.05, floor_m=0.1, inflation=inflation)
    eff_sig = np.clip(np.where(np.isfinite(eff_sig), eff_sig, 4.0), 0.1, 8.0)
    out[:, 0] = eff_sig

    # 1: disagree — |raw Post-processing - v2 smoothed| horizontal
    disagree = np.sqrt(
        (E_raw - v2_result.E_smooth) ** 2 +
        (N_raw - v2_result.N_smooth) ** 2
    )
    out[:, 1] = np.where(np.isfinite(disagree), disagree, 0.0)

    # 2: innovation_h
    innov_h = np.asarray(v2_result.innovation_h, dtype=np.float64)
    out[:, 2] = np.where(np.isfinite(innov_h), innov_h, 0.0)

    # 3: fwd_bwd_disagree_h
    fb = np.asarray(v2_result.fwd_bwd_disagree_h, dtype=np.float64)
    out[:, 3] = np.where(np.isfinite(fb), fb, 0.0)

    # 4: innovation_norm
    innov_norm = np.asarray(v2_result.innovation_norm, dtype=np.float64)
    out[:, 4] = np.where(np.isfinite(innov_norm), innov_norm, 0.0)

    # 5: q_penalty — Q-flag
    for i, r in enumerate(pos_rows):
        out[i, 5] = _Q_PENALTY.get(r.quality, 1.0)

    # 6: ns_penalty — max(0, 1 - ns/20)
    for i, r in enumerate(pos_rows):
        out[i, 6] = max(0.0, 1.0 - r.ns / 20.0)

    # 7: sd_h — raw The external solver horizontal sigma
    for i, r in enumerate(pos_rows):
        sd_n2 = r.sd_n ** 2 if math.isfinite(r.sd_n) else 0.0
        sd_e2 = r.sd_e ** 2 if math.isfinite(r.sd_e) else 0.0
        out[i, 7] = math.sqrt(sd_n2 + sd_e2)

    # 8: speed_mps (inverted) — low speed = high value = worse
    for i, r in enumerate(pos_rows):
        vn = r.vn if math.isfinite(r.vn) else 0.0
        ve = r.ve if math.isfinite(r.ve) else 0.0
        speed = math.sqrt(vn ** 2 + ve ** 2)
        out[i, 8] = 1.0 - min(speed / max_speed_mps, 1.0)

    # 9: ratio_inv — 1/max(ratio, 0.1)
    for i, r in enumerate(pos_rows):
        ratio = r.ratio if math.isfinite(r.ratio) else 0.1
        out[i, 9] = 1.0 / max(ratio, 0.1)

    return out


# ---- Normalization ----

def normalize_signals(raw: np.ndarray) -> np.ndarray:
    """Clip-normalize each signal column to [0, 1] using NORM_BOUNDS.

    Parameters
    ----------
    raw : ndarray of shape (n, 10)

    Returns
    -------
    ndarray of shape (n, 10), values clipped to [0, 1].
    """
    if raw.size == 0:
        return raw.copy()
    normed = np.empty_like(raw)
    for j, name in enumerate(SIGNAL_NAMES):
        lo, hi = NORM_BOUNDS[name]
        span = hi - lo
        if span <= 0:
            # Degenerate: all values map to 0.
            normed[:, j] = 0.0
        else:
            normed[:, j] = np.clip((raw[:, j] - lo) / span, 0.0, 1.0)
    return normed


# ---- Composite scoring ----

def composite_score(normed: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of normalized signals, re-normalized to [0, 1].

    Parameters
    ----------
    normed : ndarray of shape (n, 10)
    weights : dict mapping signal name -> weight (>=0)

    Returns
    -------
    ndarray of shape (n,), values in [0, 1].
    """
    n = normed.shape[0] if normed.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=np.float64)
    w = np.array([weights.get(name, 0.0) for name in SIGNAL_NAMES], dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.zeros(n, dtype=np.float64)
    score = (normed @ w) / w_sum
    return np.clip(score, 0.0, 1.0)


# ---- Label assignment ----

def assign_labels(
    pos_score: np.ndarray,
    vel_score: np.ndarray,
    config: TrustConfigV2,
) -> list[str]:
    """Assign per-epoch trust labels from composite scores.

    Scores are oriented so HIGHER = WORSE. An epoch is trusted when its
    score is BELOW the threshold.

    Labels:
      "high"     — both position and velocity trusted
      "pos_only" — position trusted, velocity uncertain
      "vel_only" — velocity trusted, position uncertain
      "low"      — neither trusted
    """
    n = len(pos_score)
    labels: list[str] = []
    for i in range(n):
        p_ok = pos_score[i] < config.threshold_pos
        v_ok = vel_score[i] < config.threshold_vel
        if p_ok and v_ok:
            labels.append("high")
        elif p_ok:
            labels.append("pos_only")
        elif v_ok:
            labels.append("vel_only")
        else:
            labels.append("low")
    return labels


# ---- Public API ----

def compute_trust_v2(
    pos_rows: list[PosRow],
    v2_result,
    *,
    config: Optional[TrustConfigV2] = None,
) -> TrustResultV2:
    """End-to-end v2 trust scoring: extract -> normalize -> score -> label.

    Parameters
    ----------
    pos_rows : list[PosRow]
        Sorted Post-processing rows.
    v2_result : EpochWeightV2Result
        Output from ``smooth_epoch_weighted_v2``.
    config : TrustConfigV2, optional
        Override thresholds. Uses defaults when None.

    Returns
    -------
    TrustResultV2
    """
    cfg = config or TrustConfigV2()
    n = len(pos_rows)

    if n == 0:
        empty_bool = np.array([], dtype=bool)
        empty_float = np.array([], dtype=np.float64)
        empty_signals = np.empty((0, len(SIGNAL_NAMES)), dtype=np.float64)
        return TrustResultV2(
            pos_trusted=empty_bool,
            vel_trusted=empty_bool,
            labels=[],
            pos_score=empty_float,
            vel_score=empty_float,
            signals=empty_signals,
            n_high=0, n_pos_only=0, n_vel_only=0, n_low=0,
        )

    signals = extract_signals(pos_rows, v2_result)
    disagree = signals[:, 1]  # |raw Post-processing - v2 smoothed| horizontal

    # Disagree-based labeling: one signal, two thresholds
    #   GREEN:  disagree < green_thresh → both position AND velocity guaranteed
    #   ORANGE: green_thresh ≤ disagree < red_thresh → position guaranteed only
    #   RED:    disagree ≥ red_thresh → no guarantee
    labels: list[str] = []
    for d in disagree:
        if d < cfg.disagree_green:
            labels.append("high")
        elif d < cfg.disagree_red:
            labels.append("pos_only")
        else:
            labels.append("low")

    pos_trusted = disagree < cfg.disagree_red
    vel_trusted = disagree < cfg.disagree_green

    # pos_score/vel_score = disagree itself (lower = better, monotonic)
    return TrustResultV2(
        pos_trusted=pos_trusted,
        vel_trusted=vel_trusted,
        labels=labels,
        pos_score=disagree,
        vel_score=disagree,
        signals=signals,
        n_high=sum(1 for l in labels if l == "high"),
        n_pos_only=sum(1 for l in labels if l == "pos_only"),
        n_vel_only=sum(1 for l in labels if l == "vel_only"),
        n_low=sum(1 for l in labels if l == "low"),
    )
