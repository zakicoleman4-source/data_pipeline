"""Uniform front-end for every smoother shipped in data_pipeline.

Why this module
===============
There are ~10 different smoothers across the package (Gaussian, CV+RTS
per-axis, CV+RTS with Rate-signal, gate-then-CV, epoch-weighted, 9-state
EKF+RTS, FGO, …). Each has a different signature: some take
``list[PosRow]``, some take raw ``E/N/U`` arrays, some need Motion sensor rows,
some need a stat file. End-users (GUI / scripts / clients) need a
single switchboard.

This module gives that:

    list_smoothers() -> list[str]               # names available
    describe(name)    -> SmootherInfo           # docs, requires-Motion sensor, ...
    run_smoother(name, pos_rows, ...) -> SmoothResult
    run_all_smoothers(pos_rows, ..., gt_rows=) -> list[SmoothResult]

Each :class:`SmoothResult` carries the smoothed ``list[PosRow]``,
runtime, success / error code, and (if ``gt_rows`` supplied) the
horizontal RMSE vs reference.

When a smoother raises a :class:`~data_pipeline.errors.PipelineError`
the runner captures the code + hint into the result so the GUI can
present it without the worker dying. Crashing smoothers don't crash
the comparison.
"""
from __future__ import annotations

import logging
import math
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .errors import PipelineError
from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .parsers import ImuRow, PosRow

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result + info structs
# ---------------------------------------------------------------------------

@dataclass
class SmoothResult:
    """Outcome of one smoother run."""
    name: str
    fused: list[PosRow] = field(default_factory=list)
    runtime_s: float = 0.0
    n_input: int = 0
    n_output: int = 0
    ok: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_hint: Optional[str] = None
    hrmse_m: Optional[float] = None
    h_p95_m: Optional[float] = None


@dataclass(frozen=True)
class SmootherInfo:
    """Metadata for a smoother shown in the GUI."""
    name: str
    description: str
    requires_imu: bool
    requires_stat: bool = False
    optional_dep: Optional[str] = None     # e.g. "the factor library"


# ---------------------------------------------------------------------------
# Registry — single source of truth for smoother names + metadata
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, SmootherInfo] = {
    "raw_ppk": SmootherInfo(
        name="raw_ppk",
        description="Raw PPK (no smoothing) — baseline.",
        requires_imu=False,
    ),
    "gaussian_car": SmootherInfo(
        name="gaussian_car",
        description="Gaussian smoother in ENU, car profile (xy=2s, z=10s).",
        requires_imu=False,
    ),
    "gaussian_aggressive": SmootherInfo(
        name="gaussian_aggressive",
        description="Wider Gaussian (xy=5s, z=20s) for clean float-quality PPK.",
        requires_imu=False,
    ),
    "cv_rts": SmootherInfo(
        name="cv_rts",
        description="CV+RTS per-axis (PPK position only, no Doppler).",
        requires_imu=False,
    ),
    "cv_rts_pv": SmootherInfo(
        name="cv_rts_pv",
        description="CV+RTS with PPK position AND Doppler velocity jointly. "
                    "No-video champion on the reference session.",
        requires_imu=False,
    ),
    "gate_then_cv": SmootherInfo(
        name="gate_then_cv",
        description="Doppler MAD-gate outliers, then CV+RTS per-axis.",
        requires_imu=False,
    ),
    "epoch_weight": SmootherInfo(
        name="epoch_weight",
        description="Recipe 1+3 epoch-weighted CV+RTS (uses sd_n/sd_e if "
                    "PosRow exposes them; .pos.stat raises p_resid_rms when "
                    "supplied).",
        requires_imu=False,
    ),
    "ekf_smoothed": SmootherInfo(
        name="ekf_smoothed",
        description="9-state EKF + RTS smoother with NHC, ZUPT, bias init.",
        requires_imu=True,
    ),
    "fgo": SmootherInfo(
        name="fgo",
        description="Factor-graph optimisation (PPK + IMU via GTSAM).",
        requires_imu=True, optional_dep="gtsam",
    ),
    "epoch_weight_v2": SmootherInfo(
        name="epoch_weight_v2",
        description="6D Kalman + ZUPT + per-step Q from IMU. v2 of epoch_weight.",
        requires_imu=False,
    ),
    "epoch_weight_v2_imu_bridge": SmootherInfo(
        name="epoch_weight_v2_imu_bridge",
        description="v2 + IMU bridge: replaces CV prediction with IMU-mechanized "
                    "prediction at weak GNSS epochs, downweights PPK position.",
        requires_imu=True,
    ),
    "v2_imu_adaptive": SmootherInfo(
        name="v2_imu_adaptive",
        description="v2 + adaptive gradient IMU: 3-tier (strong/medium/weak) coupling "
                    "with bias calibration. Targets 6m@2σ for weak GNSS with device IMU.",
        requires_imu=True,
    ),
    "v2_urban_canyon": SmootherInfo(
        name="v2_urban_canyon",
        description="v2 + urban canyon mode: quality-triggered IMU bridge when "
                    "ns<8 + Q>=2 + sigma>2m. Forces full IMU dead-reckoning "
                    "through alleyways/canyons instead of trusting multipath PPK.",
        requires_imu=True,
    ),
    "kalman_simple_cv": SmootherInfo(
        name="kalman_simple_cv",
        description="Simple constant-velocity Kalman over (lat, lon, h).",
        requires_imu=False,
    ),
    "gnss_imu_dr": SmootherInfo(
        name="gnss_imu_dr",
        description="RTKLIB-style 9-state EKF + RTS dead reckoning. "
                    "Auto IMU mode when sensors available, GNSS-only otherwise.",
        requires_imu=False,
    ),
}


def list_smoothers() -> list[str]:
    """All available smoother names, in suggested display order."""
    return list(_REGISTRY.keys())


def describe(name: str) -> SmootherInfo:
    if name not in _REGISTRY:
        raise KeyError(f"unknown smoother {name!r}; available: {list_smoothers()}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_to_enu_arrays(pos_rows: Sequence[PosRow]):
    """Return (E, N, U, ts, ref_llh) for downstream array-style smoothers."""
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    E = np.empty(len(pos_rows)); N = np.empty(len(pos_rows))
    U = np.empty(len(pos_rows)); ts = np.empty(len(pos_rows))
    for i, r in enumerate(pos_rows):
        e, n, u = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)
        E[i] = e; N[i] = n; U[i] = u; ts[i] = r.utc_s
    return E, N, U, ts, ref


def _enu_arrays_to_pos_rows(E, N, U, ts, ref_llh,
                            template: Sequence[PosRow]) -> list[PosRow]:
    """Stitch (E, N, U) back into PosRow list, copying quality from template."""
    out: list[PosRow] = []
    for i, t in enumerate(ts):
        lat, lon, h = enu_to_llh(float(E[i]), float(N[i]), float(U[i]), ref_llh)
        src = template[i] if i < len(template) else template[-1]
        out.append(PosRow(
            utc_s=float(t), lat_deg=lat, lon_deg=lon, h_m=h,
            quality=src.quality, vn=src.vn, ve=src.ve, vu=src.vu,
            ns=src.ns,
            sd_n=src.sd_n, sd_e=src.sd_e, sd_u=src.sd_u,
            ratio=src.ratio, age_s=src.age_s,
        ))
    return out


def _eval_horiz_rmse(fused: Sequence[PosRow],
                     gt: Sequence[PosRow]) -> tuple[Optional[float], Optional[float]]:
    """Median-offset-removed horizontal RMSE + P95 vs GT. None on no overlap."""
    if not fused or not gt:
        return None, None
    gt_sorted = sorted(gt, key=lambda r: r.utc_s)
    gt_t = [r.utc_s for r in gt_sorted]
    ref = (gt_sorted[0].lat_deg, gt_sorted[0].lon_deg, gt_sorted[0].h_m)
    pairs = []
    for f in fused:
        i = bisect_left(gt_t, f.utc_s)
        if i >= len(gt_sorted):
            continue
        if i == 0:
            if abs(f.utc_s - gt_t[0]) > 2.0:
                continue
            glat, glon, ghm = gt_sorted[0].lat_deg, gt_sorted[0].lon_deg, gt_sorted[0].h_m
        else:
            a, b = gt_sorted[i - 1], gt_sorted[i]
            dt = b.utc_s - a.utc_s
            if dt <= 0 or dt > 2.0:
                continue
            al = (f.utc_s - a.utc_s) / dt
            glat = a.lat_deg + al * (b.lat_deg - a.lat_deg)
            glon = a.lon_deg + al * (b.lon_deg - a.lon_deg)
            ghm = a.h_m + al * (b.h_m - a.h_m)
        ge, gn, _ = ecef_to_enu(*llh_to_ecef(glat, glon, ghm), ref)
        fe, fn, _ = ecef_to_enu(*llh_to_ecef(f.lat_deg, f.lon_deg, f.h_m), ref)
        pairs.append((fe - ge, fn - gn))
    if not pairs:
        return None, None
    arr = np.array(pairs)
    arr[:, 0] -= np.median(arr[:, 0]); arr[:, 1] -= np.median(arr[:, 1])
    h = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    return float(np.sqrt(np.mean(h ** 2))), float(np.percentile(h, 95))


# ---------------------------------------------------------------------------
# Per-smoother adapters
# ---------------------------------------------------------------------------

def _adapt_raw_ppk(pos_rows, **_kw) -> list[PosRow]:
    return list(pos_rows)


def _adapt_gaussian(pos_rows, xy_s, z_s) -> list[PosRow]:
    from .smoothing import estimate_rate_hz, gaussian_smooth
    if not pos_rows:
        return []
    if xy_s <= 0 and z_s <= 0:
        return list(pos_rows)
    E, N, U, ts, ref = _pos_to_enu_arrays(pos_rows)
    rate = estimate_rate_hz(ts.tolist())
    sig_xy = max(1.0, xy_s * rate) if xy_s > 0 else 0.0
    sig_z = max(1.0, z_s * rate) if z_s > 0 else 0.0
    Es = np.asarray(gaussian_smooth(E.tolist(), sig_xy)) if sig_xy > 0 else E
    Ns = np.asarray(gaussian_smooth(N.tolist(), sig_xy)) if sig_xy > 0 else N
    Us = np.asarray(gaussian_smooth(U.tolist(), sig_z)) if sig_z > 0 else U
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_cv_rts(pos_rows, **_kw) -> list[PosRow]:
    from .cv_rts import cv_rts
    if len(pos_rows) < 2:
        return list(pos_rows)
    E, N, U, ts, ref = _pos_to_enu_arrays(pos_rows)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    Es = cv_rts(E, dt, sigma_z=2.0, sigma_a=0.5)
    Ns = cv_rts(N, dt, sigma_z=2.0, sigma_a=0.5)
    Us = cv_rts(U, dt, sigma_z=4.0, sigma_a=0.5)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_cv_rts_pv(pos_rows, **_kw) -> list[PosRow]:
    from .cv_rts import cv_rts_pv, doppler_gate, lin_interp_through
    if len(pos_rows) < 2:
        return list(pos_rows)
    E, N, U, ts, ref = _pos_to_enu_arrays(pos_rows)
    ve = np.array([r.ve for r in pos_rows], float)
    vn = np.array([r.vn for r in pos_rows], float)
    vu = np.array([r.vu for r in pos_rows], float)
    bad = doppler_gate(E, N, ve, vn, ts, K=5.0)
    use = ~bad
    Eg = lin_interp_through(E, bad); Ng = lin_interp_through(N, bad)
    Ug = lin_interp_through(U, bad)
    dt = float(np.median(np.diff(ts)))
    Es = cv_rts_pv(Eg, ve, use, dt, sigma_p=4.0, sigma_v=0.3, sigma_a=0.2)
    Ns = cv_rts_pv(Ng, vn, use, dt, sigma_p=4.0, sigma_v=0.3, sigma_a=0.2)
    use_v = use & np.isfinite(vu)
    Us = cv_rts_pv(Ug, vu, use_v, dt, sigma_p=4.0, sigma_v=0.5, sigma_a=0.5)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_gate_then_cv(pos_rows, **_kw) -> list[PosRow]:
    from .cv_rts import cv_rts, gate_then_cv
    if len(pos_rows) < 2:
        return list(pos_rows)
    E, N, U, ts, ref = _pos_to_enu_arrays(pos_rows)
    ve = np.array([r.ve for r in pos_rows], float)
    vn = np.array([r.vn for r in pos_rows], float)
    Es, Ns = gate_then_cv(E, N, ve, vn, ts, K=5.0, sigma_z=1.0, sigma_a=0.5)
    dt = float(np.median(np.diff(ts)))
    Us = cv_rts(U, dt, sigma_z=2.0, sigma_a=0.5)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_epoch_weight(pos_rows, **kw) -> list[PosRow]:
    from .epoch_weight import smooth_epoch_weighted
    if len(pos_rows) < 2:
        return list(pos_rows)
    _E, _N, U, ts, ref = _pos_to_enu_arrays(pos_rows)
    Es, Ns, Us = smooth_epoch_weighted(
        list(pos_rows), stat_path=kw.get("stat_path"),
    )
    if len(Es) == 0:
        return list(pos_rows)
    # The smoother returns Local-frame about the FIRST row of pos_rows, same as
    # _pos_to_enu_arrays. Convert back.
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_ekf_smoothed(pos_rows, *, imu_rows, **_kw) -> list[PosRow]:
    from .ekf_smoothed import RtsOptions, run_ekf_rts
    if not imu_rows:
        raise PipelineError(
            "E-PP-400", "ekf_smoothed requires IMU rows (got none)",
            hint="Drop a sensors_*.txt next to the .pos OR pick a different "
                 "smoother that doesn't need IMU (cv_rts_pv, gate_then_cv, "
                 "gaussian_car).",
        )
    res = run_ekf_rts(imu_rows, pos_rows, options=RtsOptions(),
                     log=lambda m: None)
    return res.fused


def _adapt_fgo(pos_rows, *, imu_rows, **_kw) -> list[PosRow]:
    from .fgo import FgoOptions, run_fgo
    if not imu_rows:
        raise PipelineError(
            "E-PP-400", "fgo requires IMU rows (got none)",
            hint="Drop a sensors_*.txt next to the .pos OR pick a different "
                 "smoother that doesn't need IMU.",
        )
    res = run_fgo(list(pos_rows), list(imu_rows), options=FgoOptions(),
                  log=lambda m: None)
    out: list[PosRow] = []
    for i in range(len(res.utc_s)):
        if not (math.isfinite(res.lat_deg[i]) and math.isfinite(res.lon_deg[i])):
            continue
        src = pos_rows[i] if i < len(pos_rows) else pos_rows[-1]
        out.append(PosRow(
            utc_s=float(res.utc_s[i]),
            lat_deg=float(res.lat_deg[i]),
            lon_deg=float(res.lon_deg[i]),
            h_m=float(res.h_m[i]),
            quality=src.quality, vn=src.vn, ve=src.ve, vu=src.vu, ns=src.ns,
            sd_n=src.sd_n, sd_e=src.sd_e, sd_u=src.sd_u,
            ratio=src.ratio, age_s=src.age_s,
        ))
    return out


def _apply_calibration(opts, kw):
    """If kw carries an ``ImuCalibration``, map it onto the v2 options.

    Returns the (possibly-replaced) options. Unchanged behaviour when no
    calibration is supplied.
    """
    cal = kw.get("calibration")
    if cal is None:
        return opts
    from .epoch_weight_v2 import options_from_calibration
    return options_from_calibration(cal, opts)


def _adapt_epoch_weight_v2(pos_rows, *, imu_rows=None, **kw) -> list[PosRow]:
    from .epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
    if len(pos_rows) < 2:
        return list(pos_rows)
    imu = list(imu_rows) if imu_rows else None
    opts = EpochWeightV2Options(stat_path=kw.get("stat_path"))
    opts = _apply_calibration(opts, kw)
    v2 = smooth_epoch_weighted_v2(list(pos_rows), imu_rows=imu, options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(pos_rows)
    Es, Ns, Us = v2.E_smooth, v2.N_smooth, v2.U_smooth
    if len(Es) == 0:
        return list(pos_rows)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_epoch_weight_v2_imu_bridge(pos_rows, *, imu_rows=None, **kw) -> list[PosRow]:
    from .epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
    if len(pos_rows) < 2:
        return list(pos_rows)
    if not imu_rows:
        raise PipelineError(
            "E-PP-400", "epoch_weight_v2_imu_bridge requires IMU rows (got none)",
            hint="Drop a sensors_*.txt next to the .pos OR pick a smoother "
                 "that doesn't need IMU.",
        )
    opts = EpochWeightV2Options(
        stat_path=kw.get("stat_path"),
        sigma_a_base=0.10,
        imu_bridge_enabled=True,
        imu_bridge_thresh=6.0,
        imu_bridge_medium_thresh=2.5,
        imu_bridge_q_mult=2.0,
        imu_bridge_dw_mult=5.0,
        # Innovation gate NOT enabled by default: A/B across the 3 cross-device
        # pairs showed it is session-dependent — thresh=2.5 lowered pair1 MAX
        # (6.73->4.71 m) but REGRESSED pair2 (6.66->7.99) and pair3 (6.35->6.72).
        # Net-negative as a global default, so it stays opt-in (see
        # _adapt_v2_urban_canyon, which enables it deliberately). scripts/accuracy_ab.py
        # reproduces the per-pair numbers.
    )
    opts = _apply_calibration(opts, kw)
    v2 = smooth_epoch_weighted_v2(list(pos_rows), imu_rows=list(imu_rows), options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(pos_rows)
    Es, Ns, Us = v2.E_smooth, v2.N_smooth, v2.U_smooth
    if len(Es) == 0:
        return list(pos_rows)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_v2_urban_canyon(pos_rows, *, imu_rows=None, **kw) -> list[PosRow]:
    """v2 high-noise environment — quality-triggered R inflation through alleyways."""
    from .epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
    if len(pos_rows) < 2:
        return list(pos_rows)
    opts = EpochWeightV2Options(
        stat_path=kw.get("stat_path"),
        canyon_detect_enabled=True,
        canyon_ns_thresh=8,
        canyon_q_thresh=2,
        canyon_sigma_thresh=2.0,
        canyon_min_indicators=2,
        canyon_r_mult=15.0,
        innov_gate_enabled=True,
        innov_gate_thresh=5.0,
        innov_gate_r_mult=10.0,
    )
    opts = _apply_calibration(opts, kw)
    v2 = smooth_epoch_weighted_v2(list(pos_rows), imu_rows=imu_rows, options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(pos_rows)
    Es, Ns, Us = v2.E_smooth, v2.N_smooth, v2.U_smooth
    if len(Es) == 0:
        return list(pos_rows)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_v2_imu_adaptive(pos_rows, *, imu_rows=None, **kw) -> list[PosRow]:
    """v2 Motion sensor adaptive gradient — 3-tier Signal/Motion sensor coupling with bias calibration."""
    from .imu_adaptive import ImuAdaptiveOptions, smooth_imu_adaptive
    if len(pos_rows) < 2:
        return list(pos_rows)
    if not imu_rows:
        raise PipelineError(
            "E-PP-400", "v2_imu_adaptive requires IMU rows (got none)",
            hint="Drop a sensors_*.txt next to the .pos OR pick a smoother "
                 "that doesn't need IMU.",
        )
    opts = ImuAdaptiveOptions(stat_path=kw.get("stat_path"))
    result = smooth_imu_adaptive(list(pos_rows), list(imu_rows), options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(pos_rows)
    Es, Ns, Us = result.E_smooth, result.N_smooth, result.U_smooth
    if len(Es) == 0:
        return list(pos_rows)
    return _enu_arrays_to_pos_rows(Es, Ns, Us, ts, ref, pos_rows)


def _adapt_kalman_simple_cv(pos_rows, **_kw) -> list[PosRow]:
    from .kalman_simple import run_cv_kf
    if len(pos_rows) < 2:
        return list(pos_rows)
    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    out_times = [r.utc_s for r in pos_rows]
    fused = run_cv_kf(list(pos_rows), ref_llh, out_times=out_times)
    src_by_t = {r.utc_s: r for r in pos_rows}
    for i, r in enumerate(fused):
        src = src_by_t.get(r.utc_s, pos_rows[min(i, len(pos_rows) - 1)])
        fused[i] = PosRow(
            utc_s=r.utc_s, lat_deg=r.lat_deg, lon_deg=r.lon_deg, h_m=r.h_m,
            quality=src.quality, ns=src.ns,
            vn=r.vn, ve=r.ve, vu=r.vu,
            sd_n=src.sd_n, sd_e=src.sd_e, sd_u=src.sd_u,
            ratio=src.ratio, age_s=src.age_s,
        )
    return fused


def _adapt_gnss_imu_dr(pos_rows, *, imu_rows=None, **_kw) -> list[PosRow]:
    from .dead_reckoning import run_dr
    if len(pos_rows) < 2:
        return list(pos_rows)
    dr = run_dr(pos_rows, imu_rows=imu_rows, log=lambda m: None)
    return dr.fused


_ADAPTERS: dict[str, Callable[..., list[PosRow]]] = {
    "raw_ppk": _adapt_raw_ppk,
    "gaussian_car": lambda rows, **kw: _adapt_gaussian(rows, xy_s=2.0, z_s=10.0),
    "gaussian_aggressive": lambda rows, **kw: _adapt_gaussian(rows, xy_s=5.0, z_s=20.0),
    "cv_rts": _adapt_cv_rts,
    "cv_rts_pv": _adapt_cv_rts_pv,
    "gate_then_cv": _adapt_gate_then_cv,
    "epoch_weight": _adapt_epoch_weight,
    "ekf_smoothed": _adapt_ekf_smoothed,
    "fgo": _adapt_fgo,
    "epoch_weight_v2": _adapt_epoch_weight_v2,
    "epoch_weight_v2_imu_bridge": _adapt_epoch_weight_v2_imu_bridge,
    "v2_imu_adaptive": _adapt_v2_imu_adaptive,
    "v2_urban_canyon": _adapt_v2_urban_canyon,
    "kalman_simple_cv": _adapt_kalman_simple_cv,
    "gnss_imu_dr": _adapt_gnss_imu_dr,
}


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_smoother(
    name: str,
    pos_rows: Sequence[PosRow],
    *,
    imu_rows: Optional[Sequence[ImuRow]] = None,
    gt_rows: Optional[Sequence[PosRow]] = None,
    pre_filter: bool = False,
    pre_filter_cfg=None,
    pre_filter_disagreement: Optional[Sequence[float]] = None,
    log: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> SmoothResult:
    """Run one smoother by name and return a structured result.

    Captures any :class:`PipelineError` into the result instead of
    raising — the GUI can then continue to the next smoother in a
    comparison run. Unknown smoother name raises ``KeyError`` (a config
    bug, not an operator failure).

    Set ``pre_filter=True`` to clean ``pos_rows`` with the GT-free
    :mod:`data_pipeline.robust_filter` (physical-plausibility gates) BEFORE the
    smoother runs — this rejects/repairs the impossible-Post-processing epochs (160 m
    altitude, 172 km/h spikes) that otherwise dominate MAX error. Default
    ``False`` so existing behaviour is unchanged. ``pre_filter_cfg`` overrides
    the shipped car preset; ``pre_filter_disagreement`` (index-aligned to
    ``pos_rows``) enables the optional multimask-disagreement gate.
    """
    info = describe(name)
    adapter = _ADAPTERS[name]
    res = SmoothResult(name=name, n_input=len(pos_rows))
    if pre_filter:
        from .robust_filter import robust_filter, car_preset
        fr = robust_filter(pos_rows, pre_filter_cfg or car_preset(),
                           disagreement=pre_filter_disagreement, log=log)
        pos_rows = fr.fused if hasattr(fr, "fused") else fr.rows
    t0 = time.perf_counter()
    try:
        fused = adapter(pos_rows, imu_rows=imu_rows, **kwargs)
        res.fused = list(fused)
        res.n_output = len(res.fused)
    except PipelineError as e:
        res.ok = False
        res.error_code = e.code
        res.error_message = e.message
        res.error_hint = e.hint
        if log is not None:
            log(f"[smoothers] {name} failed: {e.format()}")
    except ImportError as e:
        res.ok = False
        res.error_code = "E-PP-003" if info.optional_dep == "gtsam" else "E-PP-002"
        res.error_message = str(e)
        res.error_hint = (
            f"Install the missing optional dep: `pip install {info.optional_dep}`"
            if info.optional_dep
            else "Install required package via `pip install -r requirements.txt`"
        )
        if log is not None:
            log(f"[smoothers] {name} skipped: {e}")
    except Exception as e:                  # noqa: BLE001
        res.ok = False
        res.error_code = "E-PP-900"
        res.error_message = f"{type(e).__name__}: {e}"
        res.error_hint = "Unexpected error — please report with last_error.json."
        if log is not None:
            log(f"[smoothers] {name} crashed: {type(e).__name__}: {e}")
    res.runtime_s = time.perf_counter() - t0
    if res.ok and gt_rows:
        res.hrmse_m, res.h_p95_m = _eval_horiz_rmse(res.fused, gt_rows)
    return res


def run_all_smoothers(
    pos_rows: Sequence[PosRow],
    *,
    imu_rows: Optional[Sequence[ImuRow]] = None,
    gt_rows: Optional[Sequence[PosRow]] = None,
    only: Optional[Sequence[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> list[SmoothResult]:
    """Run every registered smoother (or the subset in ``only``).

    Returns one :class:`SmoothResult` per name. Sorts the result list
    by ``hrmse_m`` ascending (smoothers that errored or had no GT land
    at the end, in their original registry order).
    """
    names = list(only) if only else list_smoothers()
    out = [run_smoother(n, pos_rows, imu_rows=imu_rows,
                        gt_rows=gt_rows, log=log) for n in names]

    def _sort_key(r: SmoothResult):
        if r.hrmse_m is not None:
            return (0, r.hrmse_m)
        return (1, names.index(r.name))

    out.sort(key=_sort_key)
    return out


# ---------------------------------------------------------------------------
# Plugin bridge — surface registered FusionPlugins as smoothers
# ---------------------------------------------------------------------------

def _register_fusion_plugins() -> None:
    """Expose every registered FusionPlugin as a smoother so it runs through
    the same ``run_smoother`` error-capture path and shows up in the GUI/CLI
    next to the built-ins. Never shadows a built-in name."""
    try:
        from .plugins_api import list_fusion_plugins, get_fusion_plugin
    except Exception:
        return
    for pname in list_fusion_plugins():
        if pname in _REGISTRY:
            continue  # never shadow a built-in
        _REGISTRY[pname] = SmootherInfo(
            name=pname,
            description=f"External fusion plugin: {pname}",
            requires_imu=False,
        )

        def _make(p):
            def _adapter(pos_rows, *, imu_rows=None, **kw):
                plugin = get_fusion_plugin(p)
                cal = kw.get("calibration")
                cal_dict = cal.to_dict() if hasattr(cal, "to_dict") else cal
                return plugin.run(
                    list(pos_rows),
                    list(imu_rows) if imu_rows else [],
                    cal_dict,
                    dict(kw.get("options", {})),
                )
            return _adapter

        _ADAPTERS[pname] = _make(pname)


def load_and_register_plugins() -> None:
    """Load drop-in + entry-point plugins, then surface fusion plugins as
    smoothers. Safe to call repeatedly; tolerant of a missing plugin layer."""
    try:
        from .plugin_loader import load_all_plugins
        load_all_plugins()
    except Exception:
        pass
    _register_fusion_plugins()


# Auto-load at import so the GUI/CLI see plugins with no extra wiring.
load_and_register_plugins()
