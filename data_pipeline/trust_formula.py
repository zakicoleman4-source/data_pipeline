"""Per-epoch trust labeling for client deliverables.

Two independent trust gates validated across 18 sessions (6 days, 9 devices):

  POSITION:  eff_sig < 0.85  -> h_max <= 9.53m (hard ceiling, 14.3% of epochs)
  VELOCITY:  |raw - v2| < 4.0 -> v_p95 <= 1.18 m/s (93.6% of epochs)

Each epoch gets a trust label:
  "high"    — both position AND velocity trusted
  "pos_only"— position trusted, velocity uncertain
  "vel_only"— velocity trusted, position uncertain
  "low"     — neither trusted

Usage:
    from data_pipeline.trust_formula import compute_trust, TrustConfig, TrustResult
    trust = compute_trust(pos_rows, v2_result)
    # trust.labels: list[str] — "high"/"pos_only"/"vel_only"/"low" per epoch
    # trust.pos_trusted: bool array
    # trust.vel_trusted: bool array
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .epoch_weight import effective_sigma, epoch_features
from .geo import ecef_to_enu, llh_to_ecef
from .parsers import PosRow
from .pos_metadata import calibrate_sigma_inflation


@dataclass
class TrustConfig:
    eff_sig_max: float = 0.85
    disagree_max: float = 4.0
    pos_validated_max_m: float = 9.53
    vel_validated_p95_mps: float = 1.18


@dataclass
class TrustResult:
    pos_trusted: np.ndarray
    vel_trusted: np.ndarray
    labels: list[str]
    eff_sig: np.ndarray
    disagree: np.ndarray
    n_high: int
    n_pos_only: int
    n_vel_only: int
    n_low: int


def compute_trust(
    pos_rows: list[PosRow],
    v2_result,
    *,
    config: Optional[TrustConfig] = None,
) -> TrustResult:
    cfg = config or TrustConfig()
    n = len(pos_rows)
    if n == 0:
        empty = np.array([], dtype=bool)
        return TrustResult(empty, empty, [], np.array([]), np.array([]),
                           0, 0, 0, 0)

    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    E_raw = np.array([ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)[0] for r in pos_rows])
    N_raw = np.array([ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)[1] for r in pos_rows])

    inflation = calibrate_sigma_inflation(pos_rows)
    feats = epoch_features(pos_rows)
    eff_sig = effective_sigma(feats, alpha=0.05, floor_m=0.1, inflation=inflation)
    eff_sig = np.clip(np.where(np.isfinite(eff_sig), eff_sig, 4.0), 0.1, 8.0)

    disagree = np.sqrt((E_raw - v2_result.E_smooth)**2 +
                        (N_raw - v2_result.N_smooth)**2)

    pos_trusted = eff_sig < cfg.eff_sig_max
    vel_trusted = disagree < cfg.disagree_max

    labels = []
    for i in range(n):
        if pos_trusted[i] and vel_trusted[i]:
            labels.append("high")
        elif pos_trusted[i]:
            labels.append("pos_only")
        elif vel_trusted[i]:
            labels.append("vel_only")
        else:
            labels.append("low")

    return TrustResult(
        pos_trusted=pos_trusted,
        vel_trusted=vel_trusted,
        labels=labels,
        eff_sig=eff_sig,
        disagree=disagree,
        n_high=sum(1 for l in labels if l == "high"),
        n_pos_only=sum(1 for l in labels if l == "pos_only"),
        n_vel_only=sum(1 for l in labels if l == "vel_only"),
        n_low=sum(1 for l in labels if l == "low"),
    )
