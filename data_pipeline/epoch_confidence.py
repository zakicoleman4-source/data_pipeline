"""Per-epoch confidence gate for guaranteed accuracy ceiling.

Empirically validated across 18 sessions (6 days, 9 devices) against
survey-grade reference. Each strategy provides a boolean keep/reject
mask and a predicted accuracy bound.

Usage:
    from data_pipeline.epoch_confidence import EpochGateConfig, compute_epoch_gate

    gate = compute_epoch_gate(pos_rows, v2_result, config=EpochGateConfig(strategy="sd_h"))
    # gate.keep_mask: bool array, True = epoch passes confidence check
    # gate.predicted_max_m: expected worst-case error of kept epochs
    # gate.n_kept, gate.n_rejected, gate.rejection_reason per epoch
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .accuracy_predictor import smart_session_std, SessionStdProfile
from .parsers import PosRow


@dataclass
class EpochGateConfig:
    """Choosable epoch confidence gate strategy.

    Strategies (validated max error on 18-session benchmark):
      "off"       — no gating, keep all epochs
      "sd_h"      — session gate + raw The external solver sd_h threshold (10% retention, max 4.75m)
      "combo"     — session gate + pred_std + innov_norm (11% retention, max 5.58m)
      "eff_sig"   — effective_sigma only, no session gate (8% retention, max 5.47m)
      "custom"    — user-defined thresholds
    """
    strategy: str = "off"

    session_smart_std_max: float = 4.0
    sd_h_max: float = 0.243
    eff_sig_max: float = 0.727
    pred_std_max: float = 2.79
    innov_norm_max: float = 4.678

    max_error_target_m: float = 6.0


@dataclass
class EpochGateResult:
    keep_mask: np.ndarray
    predicted_max_m: float
    n_total: int
    n_kept: int
    n_rejected: int
    strategy: str
    session_passed: bool
    session_smart_std: float
    rejection_reasons: list[str]


# Strategies and their empirically validated max errors (18-session benchmark)
_STRATEGY_INFO = {
    "off":     {"desc": "No gating",                         "validated_max": None},
    "sd_h":    {"desc": "S<=4 + sd_h<0.243",                 "validated_max": 4.75},
    "combo":   {"desc": "S<=4 + pred_std<2.79 + innov<4.68", "validated_max": 5.58},
    "eff_sig": {"desc": "eff_sig<0.727 (no session gate)",    "validated_max": 5.47},
}


def compute_epoch_gate(
    pos_rows: list[PosRow],
    v2_result=None,
    *,
    config: Optional[EpochGateConfig] = None,
) -> EpochGateResult:
    """Compute per-epoch keep/reject mask based on chosen strategy.

    ``v2_result`` is an ``EpochWeightV2Result`` — needed for strategies
    that use smoother diagnostics (combo). Pass None for strategies that
    only use raw .pos features (sd_h, eff_sig).
    """
    cfg = config or EpochGateConfig()
    n = len(pos_rows)
    if n == 0:
        return EpochGateResult(
            keep_mask=np.array([], dtype=bool), predicted_max_m=0.0,
            n_total=0, n_kept=0, n_rejected=0,
            strategy=cfg.strategy, session_passed=True,
            session_smart_std=0.0, rejection_reasons=[],
        )

    if cfg.strategy == "off":
        return EpochGateResult(
            keep_mask=np.ones(n, dtype=bool), predicted_max_m=float("inf"),
            n_total=n, n_kept=n, n_rejected=0,
            strategy="off", session_passed=True,
            session_smart_std=0.0, rejection_reasons=[],
        )

    profile = smart_session_std(pos_rows)
    reasons = [""] * n

    # Session gate (used by sd_h and combo strategies)
    needs_session_gate = cfg.strategy in ("sd_h", "combo", "custom")
    session_passed = True
    if needs_session_gate and profile.smart_std_m > cfg.session_smart_std_max:
        session_passed = False
        mask = np.zeros(n, dtype=bool)
        info = _STRATEGY_INFO.get(cfg.strategy, {})
        return EpochGateResult(
            keep_mask=mask,
            predicted_max_m=profile.smart_std_m * 2,
            n_total=n, n_kept=0, n_rejected=n,
            strategy=cfg.strategy, session_passed=False,
            session_smart_std=profile.smart_std_m,
            rejection_reasons=[f"session_gate(smart_std={profile.smart_std_m:.2f}>{cfg.session_smart_std_max})"] * n,
        )

    mask = np.ones(n, dtype=bool)

    if cfg.strategy == "sd_h" or (cfg.strategy == "custom" and cfg.sd_h_max < 999):
        sd_h = np.array([
            math.hypot(r.sd_n, r.sd_e) if (math.isfinite(r.sd_n) and math.isfinite(r.sd_e)) else 999.0
            for r in pos_rows
        ])
        bad = sd_h >= cfg.sd_h_max
        mask &= ~bad
        for i in np.where(bad)[0]:
            reasons[i] = f"sd_h={sd_h[i]:.3f}>={cfg.sd_h_max}"

    elif cfg.strategy == "combo":
        from .accuracy_predictor import predicted_epoch_std
        pred_std = predicted_epoch_std(pos_rows, profile)
        bad_pred = pred_std >= cfg.pred_std_max
        mask &= ~bad_pred
        for i in np.where(bad_pred)[0]:
            reasons[i] = f"pred_std={pred_std[i]:.3f}>={cfg.pred_std_max}"

        if v2_result is not None and hasattr(v2_result, 'innovation_norm'):
            bad_innov = v2_result.innovation_norm >= cfg.innov_norm_max
            still_in = mask.copy()
            mask &= ~bad_innov
            newly_rejected = still_in & bad_innov
            for i in np.where(newly_rejected)[0]:
                reasons[i] = f"innov_norm={v2_result.innovation_norm[i]:.3f}>={cfg.innov_norm_max}"

    elif cfg.strategy == "eff_sig":
        from .epoch_weight import effective_sigma, epoch_features
        from .pos_metadata import calibrate_sigma_inflation
        inflation = calibrate_sigma_inflation(pos_rows)
        feats = epoch_features(pos_rows)
        eff = effective_sigma(feats, alpha=0.05, floor_m=0.1, inflation=inflation)
        eff = np.clip(np.where(np.isfinite(eff), eff, 4.0), 0.1, 8.0)
        bad = eff >= cfg.eff_sig_max
        mask &= ~bad
        for i in np.where(bad)[0]:
            reasons[i] = f"eff_sig={eff[i]:.3f}>={cfg.eff_sig_max}"

    n_kept = int(mask.sum())
    info = _STRATEGY_INFO.get(cfg.strategy, {})
    validated_max = info.get("validated_max", cfg.max_error_target_m)

    return EpochGateResult(
        keep_mask=mask,
        predicted_max_m=validated_max if validated_max else cfg.max_error_target_m,
        n_total=n, n_kept=n_kept, n_rejected=n - n_kept,
        strategy=cfg.strategy, session_passed=session_passed,
        session_smart_std=profile.smart_std_m,
        rejection_reasons=reasons,
    )


def available_strategies() -> dict[str, str]:
    """Return {name: description} for all built-in strategies."""
    return {k: v["desc"] for k, v in _STRATEGY_INFO.items()}
