"""Per-epoch Signal weighting from The external solver filter covariance + residuals.

Joint-binning evidence on reference session (n=2071):
  low sd + low p_resid    -> median h_err 2.10 m
  high sd + high p_resid  -> median h_err 3.72 m   (1.77x worse)

Q-flag (Q=1 fix) does NOT discriminate (43% Fix in best bin vs 35% in worst).
sd_n / sd_e from .pos and coarse measurement residual RMS from .pos.stat ARE
strongly monotone predictors of actual horizontal error.

Four recipes (paste into your filter):

  Recipe 1: inverse-variance smoothing weights
    sigma_h_i = sqrt(sd_n_i**2 + sd_e_i**2)
    w_i       = 1.0 / sigma_h_i**2

  Recipe 2: Recursive-filter R matrix per epoch
    R_k = diag(sd_n_k**2, sd_e_k**2, sd_u_k**2)
    (shipped by session-recursive-filter-sigma — K -> A stack, -13.97 % mean dRMS)

  Recipe 3: IRLS Huber with combined sigma
    sigma_eff = sqrt(sd_n_i**2 + alpha * p_resid_rms_i**2)   # alpha=0.25
    loss      = sum(huber(r_i / sigma_eff, delta=1.0))

  Recipe 4: quality gate
    keep = (sd_n_i < SD_THRESH) & (p_resid_rms_i < RESID_THRESH)
    (tuned reference session: SD_THRESH=0.5, RESID_THRESH=6.0 -> drop ~25 %)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import numpy as np

from .parsers import PosRow


@dataclass(frozen=True)
class EpochFeatures:
    """Per-epoch Signal metadata for weighting."""

    utc_s: float
    sd_h_m: float          # sqrt(sd_n^2 + sd_e^2)
    sd_u_m: float
    p_resid_rms_m: float   # RMS of coarse measurement residuals across sources this epoch
    n_sats_used: int       # count of valid sources (valid_flag=1, nonzero res)
    quality: int           # The external solver Q flag


def aggregate_p_resid_per_epoch(stat_path: Path, valid_only: bool = True) -> dict[float, dict]:
    """Read .pos.stat $Source lines, aggregate per-epoch coarse measurement-residual RMS.

    Returns dict keyed by utc_s (rounded 3 dp) with:
        p_resid_rms_m  — RMS of res_p across valid sources this epoch
        n_sats_used    — count of contributing sources
        snr_mean       — mean SNR (dB-Hz) across valid sources

    Returns an empty dict if ``stat_path`` is missing / unreadable / has
    no $Source lines — callers fall back to NaN p_resid which the downstream
    sigma model treats as "no extra info".
    """
    from .stat_to_csv import parse_stat
    try:
        stat_rows = parse_stat(stat_path)
    except (FileNotFoundError, OSError, RuntimeError):
        return {}
    by_epoch: dict[float, list] = defaultdict(list)
    for r in stat_rows:
        if valid_only and r.valid_flag != 1:
            continue
        if r.res_p_m == 0.0:
            # The external solver placeholder zero for unused frequency
            continue
        by_epoch[round(r.utc_s, 3)].append(r)
    out: dict[float, dict] = {}
    for utc_s, sats in by_epoch.items():
        rs = np.array([s.res_p_m for s in sats], dtype=float)
        snrs = np.array([s.snr_db_hz for s in sats], dtype=float)
        # NaN-safe RMS — NaN residuals from partial The external solver stat parses must
        # not propagate to downstream Recipe 3 (where they cause divide-by-NaN
        # in effective_sigma and crash the Recursive-filter R-matrix construction).
        rs_finite = rs[np.isfinite(rs)]
        snrs_finite = snrs[np.isfinite(snrs)]
        out[utc_s] = {
            "p_resid_rms_m": float(np.sqrt(np.mean(rs_finite ** 2))) if rs_finite.size else float("nan"),
            "n_sats_used": int(len(sats)),
            "snr_mean": float(np.mean(snrs_finite)) if snrs_finite.size else 0.0,
        }
    return out


def epoch_features(
    pos_rows: list[PosRow],
    stat_path: Optional[Path] = None,
) -> list[EpochFeatures]:
    """Build per-epoch feature vectors for weighting.

    When ``stat_path`` is None, ``p_resid_rms_m`` is set to NaN.
    """
    p_resid_map = aggregate_p_resid_per_epoch(stat_path) if stat_path else {}
    out: list[EpochFeatures] = []
    for r in pos_rows:
        sd_h = math.sqrt(
            (r.sd_n ** 2 if math.isfinite(r.sd_n) else 0.0) +
            (r.sd_e ** 2 if math.isfinite(r.sd_e) else 0.0)
        )
        info = p_resid_map.get(round(r.utc_s, 3), {})
        out.append(EpochFeatures(
            utc_s=r.utc_s,
            sd_h_m=sd_h,
            sd_u_m=r.sd_u if math.isfinite(r.sd_u) else float("nan"),
            p_resid_rms_m=info.get("p_resid_rms_m", float("nan")),
            n_sats_used=info.get("n_sats_used", r.ns),
            quality=r.quality,
        ))
    return out


# ----- Recipe 1: inverse-variance weights for smoother -----

def inverse_variance_weights(feats: list[EpochFeatures], sigma_floor_m: float = 0.05) -> np.ndarray:
    """w_i = 1 / (sigma_h_i^2 + floor^2). Floor prevents div-by-zero when sigma is tiny."""
    if not feats:
        return np.array([])
    sd_h = np.array([f.sd_h_m for f in feats], dtype=np.float64)
    # When all sd_h are NaN or 0, fall back to uniform weights via 1m sigma.
    valid_mask = np.isfinite(sd_h) & (sd_h > 0)
    if not valid_mask.any():
        med = 1.0
    else:
        med = float(np.median(sd_h[valid_mask]))
    sd_h = np.where(valid_mask, sd_h, med)
    sigma2 = sd_h ** 2 + sigma_floor_m ** 2
    return 1.0 / sigma2


# ----- Recipe 3: IRLS Huber combined sigma -----

def effective_sigma(
    feats: list[EpochFeatures],
    alpha: float = 0.25,
    floor_m: float = 0.1,
    inflation: float = 1.0,
) -> np.ndarray:
    """sigma_eff_i = inflation * sqrt(sd_h_i^2 + alpha * p_resid_rms_i^2 + floor^2).

    ``alpha`` trades ambiguity-uncertainty vs source-disagreement; 0.25 empirically.
    ``inflation`` is the per-session The external solver-σ calibration factor (see
    pos_metadata.calibrate_sigma_inflation).
    Falls back to sd_h alone when p_resid is NaN.
    """
    sd_h = np.array([f.sd_h_m for f in feats], dtype=np.float64)
    p_r  = np.array([f.p_resid_rms_m for f in feats], dtype=np.float64)
    p_r_safe = np.where(np.isfinite(p_r), p_r, 0.0)
    sigma = np.sqrt(sd_h ** 2 + alpha * p_r_safe ** 2 + floor_m ** 2)
    return inflation * sigma


# ----- Recipe 4: quality gate -----

def quality_gate(
    feats: list[EpochFeatures],
    sd_thresh_m: float = 0.5,
    resid_thresh_m: float = 6.0,
) -> np.ndarray:
    """Boolean ``keep`` mask. True when sd AND p_resid pass; NaN p_resid treated as fail."""
    sd_h = np.array([f.sd_h_m for f in feats])
    p_r  = np.array([f.p_resid_rms_m for f in feats])
    return (sd_h < sd_thresh_m) & np.isfinite(p_r) & (p_r < resid_thresh_m)


# ----- Recipe 1+3 combined: weighted CV+RTS smoother -----

def smooth_epoch_weighted(
    pos_rows: list[PosRow],
    *,
    stat_path: Optional[Path] = None,
    K_dop_gate: float = 4.0,
    alpha_resid: float = 0.05,
    sigma_floor_m: float = 0.1,
    sigma_clamp_hi_m: float = 8.0,
    v_scale: float = 1.0,
    v_floor_mps: float = 0.02,
    v_clamp_hi_mps: float = 2.0,
    sigma_a: float = 0.15,
    inflation: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tuned epoch-weighted CV+RTS smoother.

    Combines:
      - Recipe 1+3 effective sigma per epoch from sd_n/sd_e + p_resid_rms
      - Per-epoch Rate-signal velocity sigma from The external solver sd_vn/sd_ve
      - K-MAD Rate-signal gate for Post-processing outliers

    Returns (E_smooth, N_smooth, U_smooth) in Local-frame about first Post-processing row.

    reference session hRMSE 2.330 m vs cv_rts_pv constant-sigma 2.498 (-6.7 %);
    wins 5/5 datasets, mean -2.8 %.
    """
    from .geo import ecef_to_enu, llh_to_ecef
    from .cv_rts import doppler_gate, lin_interp_through

    if not pos_rows:
        return np.array([]), np.array([]), np.array([])
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    def enu(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref)

    E = np.array([enu(r)[0] for r in pos_rows])
    N = np.array([enu(r)[1] for r in pos_rows])
    U = np.array([enu(r)[2] for r in pos_rows])
    VE = np.array([r.ve for r in pos_rows], float)
    VN = np.array([r.vn for r in pos_rows], float)
    VU = np.array([r.vu for r in pos_rows], float)
    SD_VN = np.array([r.sd_vn for r in pos_rows], float)
    SD_VE = np.array([r.sd_ve for r in pos_rows], float)
    SD_VU = np.array([r.sd_vu for r in pos_rows], float)
    ts = np.array([r.utc_s for r in pos_rows])
    n = len(pos_rows)

    feats = epoch_features(pos_rows, stat_path)
    if inflation is None:
        from .pos_metadata import calibrate_sigma_inflation
        inflation = calibrate_sigma_inflation(pos_rows)

    dt_med = float(np.median(np.diff(ts))) if n > 1 else 1.0
    bd = doppler_gate(E, N, VE, VN, ts, K=K_dop_gate)
    use_v = ~bd
    Eg = lin_interp_through(E, bd)
    Ng = lin_interp_through(N, bd)
    Ug = lin_interp_through(U, bd)

    sigma_eff = effective_sigma(feats, alpha=alpha_resid, floor_m=sigma_floor_m, inflation=inflation)
    sigma_eff = np.clip(sigma_eff, sigma_floor_m, sigma_clamp_hi_m)
    sigma_eff = np.where(np.isfinite(sigma_eff), sigma_eff, 4.0)

    sv_arr_h = np.sqrt(SD_VN ** 2 + SD_VE ** 2) / math.sqrt(2)
    sv_arr = np.where(np.isfinite(sv_arr_h) & (sv_arr_h > 0), sv_arr_h * v_scale, 0.3)
    sv_arr = np.clip(sv_arr, v_floor_mps, v_clamp_hi_mps)

    sv_u_arr = np.where(np.isfinite(SD_VU) & (SD_VU > 0), SD_VU * v_scale, 0.3)
    sv_u_arr = np.clip(sv_u_arr, v_floor_mps, v_clamp_hi_mps)

    sigma_u = sigma_eff * 2.5  # vertical typically 2.5x horizontal
    Es = cv_rts_pv_weighted(Eg, VE, use_v, dt_med, sigma_eff, sv_arr, sigma_a)
    Ns = cv_rts_pv_weighted(Ng, VN, use_v, dt_med, sigma_eff, sv_arr, sigma_a)
    Us = cv_rts_pv_weighted(Ug, VU, use_v, dt_med, sigma_u, sv_u_arr, sigma_a * 2)
    return Es, Ns, Us


def cv_rts_pv_weighted(
    z: np.ndarray, v: np.ndarray, use_v: np.ndarray, dt: float,
    sigma_p_arr: np.ndarray, sigma_v_arr: np.ndarray, sigma_a: float,
) -> np.ndarray:
    """Scalar forward-backward Recursive-filter with per-epoch measurement sigma.

    Replaces data_pipeline.cv_rts.cv_rts_pv constant-sigma version. Pass
    ``effective_sigma`` per axis (e.g. sigma_p_arr from Recipe 3).
    """
    n = len(z)
    if n == 0:
        return np.array([])
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"cv_rts_pv_weighted: dt must be > 0 (got {dt})")
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Hp = np.array([[1.0, 0.0]])
    Hv = np.array([[0.0, 1.0]])
    # Initial-state seeding: if z[0] or v[0] is non-finite, find the first
    # finite sample. If everything is NaN, fall back to zero with a wide P.
    SIG_FLOOR = 1e-6  # m  — minimum sigma; prevents S=0 div-by-zero
    z0 = float(z[0]) if math.isfinite(float(z[0])) else 0.0
    v0 = float(v[0]) if (use_v[0] and math.isfinite(float(v[0]))) else 0.0
    sp0 = max(SIG_FLOOR, float(sigma_p_arr[0])) if math.isfinite(float(sigma_p_arr[0])) else 10.0
    sv0 = max(SIG_FLOOR, float(sigma_v_arr[0])) if (use_v[0] and math.isfinite(float(sigma_v_arr[0]))) else 1.0
    x = np.array([z0, v0])
    P = np.diag([sp0 ** 2, sv0 ** 2])
    x_fwd = np.zeros((n, 2)); P_fwd = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2)); P_pred = np.zeros((n, 2, 2))
    for k in range(n):
        x_p = F @ x; P_p = F @ P @ F.T + Q
        x_pred[k] = x_p; P_pred[k] = P_p
        # Position update — skip when z[k] is NaN; else clamp sigma.
        if math.isfinite(float(z[k])):
            sp_k = max(SIG_FLOOR, float(sigma_p_arr[k])) if math.isfinite(float(sigma_p_arr[k])) else 10.0
            S = float((Hp @ P_p @ Hp.T)[0, 0]) + sp_k ** 2
            K = (P_p @ Hp.T / S).flatten()
            innov = float(z[k] - (Hp @ x_p)[0])
            x = x_p + K * innov
            P = P_p - np.outer(K, Hp @ P_p)
        else:
            x = x_p; P = P_p
        # Velocity update — skip when v[k] NaN or use_v[k] false.
        if use_v[k] and math.isfinite(float(v[k])):
            sv_k = max(SIG_FLOOR, float(sigma_v_arr[k])) if math.isfinite(float(sigma_v_arr[k])) else 1.0
            S = float((Hv @ P @ Hv.T)[0, 0]) + sv_k ** 2
            K = (P @ Hv.T / S).flatten()
            innov = float(v[k] - (Hv @ x)[0])
            x = x + K * innov
            P = P - np.outer(K, Hv @ P)
        x_fwd[k] = x; P_fwd[k] = P
    x_sm = x_fwd.copy()
    for k in range(n - 2, -1, -1):
        try:
            C = P_fwd[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_sm[k] = x_fwd[k] + C @ (x_sm[k + 1] - x_pred[k + 1])
    return x_sm[:, 0]
