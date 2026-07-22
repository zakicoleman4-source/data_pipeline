"""Predicted-error (smart accuracy std) calibrated against 14 real Post-processing sessions.

PROBLEM
-------
The external solver sigmas (``sd_n / sd_e / sd_u``) are filter covariance only — they
describe noise, not bias. Local-variance calibration
(:func:`data_pipeline.pos_metadata.calibrate_sigma_inflation`) catches some
mid-frequency environment noise, but slow ambiguity drift remains invisible. On
``a secondary test session`` we measured actual horizontal P95 = 10–16 m while
``raw_sd × inflation`` predicted < 1 m — a 7× under-estimate.

SMART FORMULA (tuned 2026-05-19 on 14 sessions across the reference set/reference site/session 2-6 vs
survey-grade reference)
----------------------------------------------------------------
predicted_std = sqrt(
        (raw_sd_h × inflation)^2          # noise + environment noise part
      + ambig_bias^2                       # bias from Q distribution
      + spike_bias^2                       # session has outlier spikes?
      + hidden_bias^2                      # tiny The external solver cov but big local var?
    ) × q_scale

where:
    raw_sd_h          = median(sqrt(sd_n^2 + sd_e^2))  per epoch
    inflation         = pos_metadata.calibrate_sigma_inflation(rows)
    ambig_bias        = 0.3·q1 + 2.0·q2 + 5.0·q4 + 10.0·q5   (Q-mix weighted)
    spike_bias        = 1.5  if raw_sd_p95 > 5 × raw_sd_med else 0
    hidden_bias       = 4.0  if inflation > 6 AND raw_sd_h < 0.2 else 0
    q_scale           = 1.3  if Q=2 (float) dominates the session
    Floor             = quality-aware (PP6): fix 0.2 / float 1.0 / differential 2.0 /
                        single 3.0 m (was a constant 0.5 m, which under-reported
                        float & single sigma)

VALIDATION (14 sessions, reference):
  err_med < 2 × smart_std :    14/14
  err_p95 < 2 × smart_std :     7/14   (calibrated envelope)
  err_p95 < 3 × smart_std :    12/14   (safe envelope)

2 sessions exceed 3σ at p95 (session 4 session-C, session 4 code-only session) — both have rare
20+ m environment noise spikes no session-level formula can predict. These sessions
get ``trust_class = "spike_risk"`` and the client report should explicitly
warn that occasional outlier epochs may exceed predicted bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .parsers import PosRow
from .pos_metadata import calibrate_sigma_inflation


# ---------------------------------------------------------------------------
# PP6 — quality-aware sigma floor (replaces the old constant 0.5 m pin)
# ---------------------------------------------------------------------------
# The old code clamped every predicted 1-sigma to a *constant* 0.5 m floor
# regardless of solution quality. On a Q=1 (true fix) epoch 0.5 m is already
# pessimistic; on a Q=2 (float) or Q=5 (single) epoch it is wildly optimistic —
# it made the reported 2-sigma claim sub-metre accuracy the float solution does
# not have. The floor below is keyed on the per-epoch The external solver quality flag so the
# reported sigma reflects the *real* best-case precision of that solution type.
# These are conservative lower bounds, not constants: the computed model value is
# used whenever it is larger (the floor only ever raises an unrealistically tiny
# number, never lowers an honest one).
_QUALITY_FLOOR_M = {
    1: 0.20,   # fix      — sub-decimetre is plausible; do not over-floor
    2: 1.00,   # float    — honest float floor; 0.5 m was optimistic
    4: 2.00,   # Differential     — code-differential, metre-level at best
    5: 3.00,   # single   — autonomous, several metres at best
}
# Fallback when no per-epoch quality is available (session-level call): use the
# float floor, the common day14 case, rather than the old optimistic 0.5.
_QUALITY_FLOOR_DEFAULT_M = 1.0


def quality_floor_m(quality: Optional[int] = None) -> float:
    """Honest, quality-aware 1-sigma floor (m). PP6.

    ``quality`` is the external solver Q flag (1 fix / 2 float / 4 differential / 5 single).
    When unknown, returns the float-grade default (1.0 m) — never the old
    constant 0.5 m, which under-reported float/single sigma.
    """
    if quality is None:
        return _QUALITY_FLOOR_DEFAULT_M
    return _QUALITY_FLOOR_M.get(int(quality), _QUALITY_FLOOR_DEFAULT_M)


@dataclass
class SessionStdProfile:
    """Per-session predicted-accuracy summary."""

    smart_std_m: float                 # session-level horizontal 1-sigma (m)
    trust_class: str                   # 'trustworthy' / 'tight' / 'spike_risk'
    inflation: float
    raw_sd_med_m: float
    raw_sd_spread: float               # raw_sd_p95 / raw_sd_med
    ns_med: float
    q1_frac: float                     # fraction with Q=1 (fix)
    q2_frac: float
    q4_frac: float
    q5_frac: float
    ratio_med: float                   # AR validation ratio median
    components: dict                   # debug: noise/ambig/spike/hidden/scale


def _session_features(pos_rows: list[PosRow]) -> dict:
    """Extract session-level features needed by smart_std."""
    if not pos_rows:
        raise ValueError("_session_features: empty pos_rows")
    sd_h = np.array([
        math.hypot(r.sd_n, r.sd_e)
        if (math.isfinite(r.sd_n) and math.isfinite(r.sd_e))
        else float("nan")
        for r in pos_rows
    ])
    ns = np.array([float(r.ns) for r in pos_rows])
    ql = np.array([r.quality for r in pos_rows], int)
    ratio = np.array([r.ratio for r in pos_rows], float)
    finite_sd = np.isfinite(sd_h)
    raw_sd_med = float(np.nanmedian(sd_h)) if finite_sd.any() else float("nan")
    raw_sd_p95 = float(np.nanpercentile(sd_h, 95)) if finite_sd.any() else raw_sd_med
    raw_sd_spread = (raw_sd_p95 / max(raw_sd_med, 1e-3)) if math.isfinite(raw_sd_med) else 1.0
    finite_ratio = np.isfinite(ratio)
    ratio_med = float(np.median(ratio[finite_ratio])) if finite_ratio.any() else 0.0
    return {
        "infl": calibrate_sigma_inflation(pos_rows),
        "raw_sd": raw_sd_med if math.isfinite(raw_sd_med) else 1.0,
        "raw_sd_spread": raw_sd_spread,
        "ns_med": float(np.median(ns)),
        "q1": float(np.mean(ql == 1)),
        "q2": float(np.mean(ql == 2)),
        "q4": float(np.mean(ql == 4)),
        "q5": float(np.mean(ql == 5)),
        "ratio_med": ratio_med,
    }


def smart_session_std(pos_rows: list[PosRow]) -> SessionStdProfile:
    """Return predicted session-level horizontal 1-sigma + trust class."""
    if not pos_rows:
        raise ValueError("smart_session_std: empty pos_rows")
    f = _session_features(pos_rows)

    noise = f["raw_sd"] * f["infl"]
    ambig = (
        0.3 * f["q1"] + 2.0 * f["q2"] + 5.0 * f["q4"] + 10.0 * f["q5"]
    )
    spike = 1.5 if f["raw_sd_spread"] > 5.0 else 0.0
    hidden = 4.0 if (f["infl"] > 6.0 and f["raw_sd"] < 0.2) else 0.0
    base = math.sqrt(noise ** 2 + ambig ** 2 + spike ** 2 + hidden ** 2)
    q_scale = 1.3 if f["q2"] > 0.5 else 1.0
    # PP6: quality-aware floor instead of the old constant 0.5 m. Pick the floor
    # from the dominant per-session Q flag so a float/single session is not
    # pinned to an optimistic sub-metre sigma it never achieves.
    q_fracs = {1: f["q1"], 2: f["q2"], 4: f["q4"], 5: f["q5"]}
    dominant_q = max(q_fracs, key=q_fracs.get) if max(q_fracs.values()) > 0 else None
    floor = quality_floor_m(dominant_q)
    smart = max(floor, base * q_scale)

    # Trust class: based on session features, no GT used.
    # 'trustworthy'  — q1 dominant OR ratio_med>3 (true fix) OR raw_sd_spread tame
    # 'tight'        — typical Q=2 float session, std calibrated to ~2x envelope
    # 'spike_risk'   — high spread + tiny inflation -> hidden environment noise spikes
    if f["q1"] > 0.5 or f["ratio_med"] > 3.0:
        cls = "trustworthy"
    elif f["raw_sd_spread"] > 5.0 and f["raw_sd"] < 0.5:
        # Environment noise-spike pattern. session 4 session-C / session 4 code-only session signature.
        cls = "spike_risk"
    else:
        cls = "tight"

    return SessionStdProfile(
        smart_std_m=smart,
        trust_class=cls,
        inflation=f["infl"],
        raw_sd_med_m=f["raw_sd"],
        raw_sd_spread=f["raw_sd_spread"],
        ns_med=f["ns_med"],
        q1_frac=f["q1"], q2_frac=f["q2"],
        q4_frac=f["q4"], q5_frac=f["q5"],
        ratio_med=f["ratio_med"],
        components={
            "noise": noise, "ambig": ambig,
            "spike": spike, "hidden": hidden,
            "q_scale": q_scale, "base": base,
        },
    )


def predicted_epoch_std(
    pos_rows: list[PosRow],
    session_profile: Optional[SessionStdProfile] = None,
) -> np.ndarray:
    """Per-epoch predicted horizontal 1-sigma (m).

    Per-epoch refinement on top of the session-level baseline:
      sigma_epoch_i = sqrt(
          (sd_h_i * inflation)^2          # this epoch's noise
        + q_bias_i^2                       # this epoch's Q-flag contribution
        + spike_bias_session^2             # constant per session
      ) * q_scale

    Non-finite sd_h falls back to the session baseline.
    """
    if not pos_rows:
        return np.array([])
    profile = session_profile or smart_session_std(pos_rows)
    spike = profile.components["spike"]
    q_scale = profile.components["q_scale"]
    infl = profile.inflation
    q_bias_lookup = {1: 0.3, 2: 2.0, 4: 5.0, 5: 10.0}
    out = np.empty(len(pos_rows), dtype=np.float64)
    for i, r in enumerate(pos_rows):
        if math.isfinite(r.sd_n) and math.isfinite(r.sd_e):
            noise_i = math.hypot(r.sd_n, r.sd_e) * infl
        else:
            noise_i = profile.raw_sd_med_m * infl
        qb = q_bias_lookup.get(int(r.quality), 10.0)
        base_i = math.sqrt(noise_i ** 2 + qb ** 2 + spike ** 2)
        # PP6: per-epoch quality-aware floor (not a constant 0.5 m). A float/
        # single epoch keeps an honest floor; a true-fix epoch is allowed to
        # report tighter.
        out[i] = max(quality_floor_m(int(r.quality)), base_i * q_scale)
    return out
