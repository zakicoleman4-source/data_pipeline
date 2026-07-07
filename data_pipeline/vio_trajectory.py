"""Motion model-shape + Post-processing-anchor path reconstruction.

Insight: the EKF + RTS pipeline minimises a weighted sum of process-model
+ measurement residuals, and with the position R tuned tight, the state
**snaps to each Post-processing epoch** — i.e. the smoother absorbs Post-processing noise into
the output path.

This module takes the opposite approach:

1. **Integrate Motion model velocities** (already Post-processing-Rate-signal scaled and rotated
   to Local-frame via the auto-calibrated R_body_from_cam) to obtain a *relative*
   path p_vio(t).
2. **Fit a smooth offset(t)** function — represented as a low-frequency
   B-spline — so that ``p_vio(t) + offset(t)`` matches the absolute Post-processing
   observations in a regularised least-squares sense.

The smoothness regulariser stops ``offset(t)`` from absorbing Post-processing
high-frequency noise; the Post-processing observations stop the Motion model shape from
drifting freely. The result preserves Motion model's high-fidelity local
geometry while keeping global coordinate tagging + scale tied to Post-processing.

Formulation
-----------
Let::

    t_v[i]  = Motion model sample times
    p_v[i]  = ∫_{t0}^{t_v[i]} v_vio dt'    (cumulative Motion model position in Local-frame)
    t_p[j]  = Post-processing epoch times
    z_p[j]  = Post-processing Local-frame position at t_p[j]

We parameterise ``offset(t) = sum_k a_k · B_k(t)`` with cubic B-spline
basis functions ``B_k`` on a uniform knot grid with spacing
``smoothness_s`` seconds. The control points ``a_k ∈ ℝ³`` are solved
for jointly.

Cost::

    J(a) = sum_j || p_v(t_p[j]) + offset(t_p[j]) − z_p[j] ||²    (data)
         + λ · sum_k || a_{k+2} − 2 a_{k+1} + a_k ||²              (smoothness)

This is linear in ``a`` (the basis is linear in a), so the solve is a
single ``scipy.linalg.lstsq`` on a sparse system.

API
---
``fit_vio_anchored_trajectory(vio_vels, pos_rows, smoothness_s=30.0,
lambda_smooth=1e-2)`` returns a callable ``path(t) → (E, N, U)``
and a list of :class:`PosRow` at a uniform output rate.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .parsers import PosRow


def _cubic_bspline_basis(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Evaluate uniform cubic B-spline basis at ``t``.

    ``knots`` is a 1-D array of evenly-spaced knot positions. Returns
    an ``(len(t), len(knots))`` matrix B where each row sums to 1 (over
    the four non-zero supports) — partition-of-unity B-spline.

    Standard de Boor recurrence is overkill for *uniform* knots; we use
    the closed-form cubic B-spline kernel
    ``b(u) = 1/6 · (max(0, u+2)³ − 4·max(0, u+1)³ + 6·max(0, u)³
                    − 4·max(0, u-1)³)`` where ``u = (t − knot[k]) / h``.
    """
    h = float(knots[1] - knots[0])
    # u_{ij} = (t_i − knot_j) / h
    u = (t[:, None] - knots[None, :]) / h
    # Cubic B-spline kernel centred at 0 with support [-2, 2].
    def _pow3(x):
        return np.where(x > 0, x ** 3, 0.0)
    B = (_pow3(u + 2) - 4 * _pow3(u + 1) + 6 * _pow3(u)
         - 4 * _pow3(u - 1)) / 6.0
    return B


@dataclass
class VioFitResult:
    """Output of :func:`fit_vio_anchored_trajectory`."""
    fused: list[PosRow] = field(default_factory=list)
    rms_ppk_residual_m: float = float("nan")
    # Diagnostic: per-Post-processing-epoch fit residual magnitude.
    ppk_residuals_m: list[float] = field(default_factory=list)
    n_control_points: int = 0
    lambda_smooth: float = 0.0


def _interp_cum_vio(
    vio_vels: Sequence[tuple[float, np.ndarray]],
    query_t: np.ndarray,
    max_gap_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulatively integrate Motion model velocity and sample at ``query_t``.

    Returns ``(cum_at_query, cum_full)`` where ``cum_at_query`` has shape
    ``(len(query_t), 3)`` and ``cum_full`` has shape ``(len(vio_vels), 3)``.
    Large gaps (> ``max_gap_s``) are treated as zero step (no spurious
    integration over Motion model outages).
    """
    vio_t = np.asarray([t for t, _ in vio_vels], dtype=np.float64)
    vio_v = np.asarray([v for _, v in vio_vels], dtype=np.float64)
    dt = np.diff(vio_t, prepend=vio_t[0])
    dt[dt > max_gap_s] = 0.0
    cum_full = np.cumsum(vio_v * dt[:, None], axis=0)

    def _at(t: float) -> np.ndarray:
        j = int(np.searchsorted(vio_t, t))
        if j == 0:
            return cum_full[0].copy()
        if j >= len(vio_t):
            return cum_full[-1].copy()
        t0, t1 = float(vio_t[j - 1]), float(vio_t[j])
        if t1 <= t0:
            return cum_full[j - 1].copy()
        a = (t - t0) / (t1 - t0)
        return cum_full[j - 1] + a * (cum_full[j] - cum_full[j - 1])

    cum_at = np.array([_at(float(t)) for t in query_t])
    return cum_at, cum_full


def _enu_to_llh_factory(ref_llh: tuple[float, float, float]) -> Callable:
    """Local-tangent-plane Local-frame → LLH inversion."""
    from .geo import _A, _E2
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    ref_ecef = np.array(llh_to_ecef(*ref_llh))

    def _f(e: float, n: float, u: float) -> tuple[float, float, float]:
        dx = -so * e - sl * co * n + cl * co * u
        dy = co * e - sl * so * n + cl * so * u
        dz = cl * n + sl * u
        x = ref_ecef[0] + dx
        y = ref_ecef[1] + dy
        z = ref_ecef[2] + dz
        p = math.sqrt(x * x + y * y)
        lon = math.atan2(y, x)
        lat = math.atan2(z, p * (1.0 - _E2))
        for _ in range(5):
            sl_ = math.sin(lat)
            n_ = _A / math.sqrt(1.0 - _E2 * sl_ ** 2)
            lat = math.atan2(z + _E2 * n_ * sl_, p)
        sl_ = math.sin(lat)
        n_ = _A / math.sqrt(1.0 - _E2 * sl_ ** 2)
        h = (p / math.cos(lat) - n_) if abs(math.cos(lat)) > 1e-9 \
            else (abs(z) / sl_ - n_ * (1.0 - _E2))
        return math.degrees(lat), math.degrees(lon), h

    return _f


def fit_vio_anchored_trajectory(
    vio_vels: Sequence[tuple[float, np.ndarray]],
    pos_rows: Sequence[PosRow],
    smoothness_s: float = 30.0,
    lambda_smooth: float = 1e-2,
    output_rate_hz: float = 5.0,
    robust: bool = True,
    robust_huber_k: float = 3.0,
    robust_max_iter: int = 5,
    use_pos_sigma_weights: bool = False,
    sigma_floor_h_m: float = 3.0,
    sigma_floor_v_m: float = 15.0,
    log: Optional[Callable[[str], None]] = None,
) -> VioFitResult:
    """Fit ``offset(t)`` so ``p_vio(t) + offset(t)`` matches Post-processing in LS.

    Parameters
    ----------
    vio_vels
        Sequence of ``(utc_s, v_enu_3vec)`` from
        :func:`motion model.vio_to_enu_velocities`. Velocity assumed already
        rotated to Local-frame via the auto-calibrated R_body_from_cam.
    pos_rows
        Post-processing epochs to anchor the fit against. The very first row sets
        the Local-frame origin.
    smoothness_s
        Knot spacing of the offset spline (seconds). Larger = smoother
        offset, more shape preservation. Smaller = offset can absorb more
        local Post-processing detail. 30 s is a good default for 1-Hz Post-processing.
    lambda_smooth
        Weight on the second-difference penalty on offset control
        points. Larger = stiffer offset (more Motion model shape, more Post-processing
        residual). Smaller = looser offset (more Post-processing fit, less shape).
    output_rate_hz
        Sample rate of the emitted path.
    """
    def _log(m: str) -> None:
        if log is not None:
            log(m)

    if not vio_vels:
        raise ValueError(
            "fit_vio_anchored_trajectory: empty VIO list. Run "
            "data_pipeline.vio.run_vio first and verify it produced "
            "non-empty samples (check VIO log for tracking failures)."
        )
    if not pos_rows:
        raise ValueError(
            "fit_vio_anchored_trajectory: empty PPK list. Pass at least "
            "2 PosRow entries from parse_rtkpos(.pos)."
        )

    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    # Post-processing Local-frame observations.
    ppk_t = np.asarray([r.utc_s for r in pos_rows], dtype=np.float64)
    ppk_enu = np.array([
        list(ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh))
        for r in pos_rows
    ], dtype=np.float64)

    # Per-epoch weights. By default we IGNORE the external solver sd_ columns —
    # they are routinely over-confident at environment noise spikes (e.g. .pos
    # reports σ=0.3 m at an epoch whose true error is 24 m), which makes
    # the LS solver overfit those bad epochs. Uniform weights treat
    # every Post-processing observation equally; Post-processing then acts purely as the global
    # *coordinate output anchor* and Motion model supplies all the shape. Set
    # ``use_pos_sigma_weights=True`` only when The external solver sigmas are
    # genuinely well-calibrated for your dataset (rare).
    if use_pos_sigma_weights:
        sig_h = np.array([r.sd_e if math.isfinite(r.sd_e) else sigma_floor_h_m
                          for r in pos_rows])
        sig_v = np.array([r.sd_u if math.isfinite(r.sd_u) else sigma_floor_v_m
                          for r in pos_rows])
        sig_h = np.maximum(sig_h, 0.1)
        sig_v = np.maximum(sig_v, 0.1)
    else:
        sig_h = np.full(len(pos_rows), sigma_floor_h_m)
        sig_v = np.full(len(pos_rows), sigma_floor_v_m)
    w_h = 1.0 / sig_h
    w_v = 1.0 / sig_v

    # Motion model cumulative path sampled at Post-processing epochs.
    cum_at_ppk, _ = _interp_cum_vio(vio_vels, ppk_t)

    # Each Post-processing epoch contributes ``z_p − cum_at_ppk = sum_k a_k · B_k(t_p)``.
    # Build B-spline basis matrix B (N_ppk × N_ctrl).
    t_start = float(min(ppk_t[0], vio_vels[0][0]))
    t_end   = float(max(ppk_t[-1], vio_vels[-1][0]))
    # Pad knots by 2 on each side for cubic support.
    n_knots = max(8, int(math.ceil((t_end - t_start) / smoothness_s)) + 4)
    knots = np.linspace(t_start - 2 * smoothness_s,
                        t_end + 2 * smoothness_s, n_knots)
    B = _cubic_bspline_basis(ppk_t, knots)  # (N_ppk, N_knots)

    n_ctrl = B.shape[1]
    n_ppk = B.shape[0]

    # Per-axis solves are independent. Build them simultaneously to save
    # one factorisation: stack into a (3N_ppk, 3N_ctrl) block-diagonal
    # system for clarity. With ~2000 Post-processing and ~80 control points per axis
    # the dense solve is sub-second.
    # E axis residual: z_E - cum_E = B @ a_E. Weighted by w_h.
    # Same for N (weighted w_h) and U (weighted w_v).
    rows = []
    rhs  = []
    # Position-fit rows
    for k, axis in enumerate(["e", "n", "u"]):
        w = w_h if axis != "u" else w_v
        Aw = B * w[:, None]
        rhs_w = (ppk_enu[:, k] - cum_at_ppk[:, k]) * w
        # Stack with leading zeros for other axes (block-diagonal).
        full_row = np.zeros((n_ppk, 3 * n_ctrl))
        full_row[:, k * n_ctrl:(k + 1) * n_ctrl] = Aw
        rows.append(full_row)
        rhs.append(rhs_w)
    # Smoothness rows: λ · (a_{k+2} − 2 a_{k+1} + a_k) = 0
    # Build per-axis 2nd-difference penalty matrix.
    D = np.zeros((n_ctrl - 2, n_ctrl))
    for i in range(n_ctrl - 2):
        D[i, i] = 1; D[i, i + 1] = -2; D[i, i + 2] = 1
    lam = math.sqrt(lambda_smooth)
    for k in range(3):
        full_row = np.zeros((n_ctrl - 2, 3 * n_ctrl))
        full_row[:, k * n_ctrl:(k + 1) * n_ctrl] = lam * D
        rows.append(full_row)
        rhs.append(np.zeros(n_ctrl - 2))

    A = np.vstack(rows)
    b = np.concatenate(rhs)
    _log(f"[vio-fit] solving LS: rows={A.shape[0]} cols={A.shape[1]} "
         f"n_ctrl_per_axis={n_ctrl} smoothness={smoothness_s}s lam={lambda_smooth} "
         f"robust={robust}")

    # ─ Iteratively Reweighted Least Squares (IRLS) with Huber loss ─
    # Catches Post-processing environment noise spikes whose reported sigma is over-optimistic
    # (.pos files routinely under-estimate true error by 5-50× at outlier
    # epochs). The smoothness rows are NOT reweighted — only Post-processing rows.
    base_weights = np.ones(b.size)
    huber_w = np.ones(b.size)
    n_per_axis = n_ppk
    sol = None
    for it in range(robust_max_iter if robust else 1):
        W = base_weights * huber_w
        Aw = A * W[:, None]
        bw = b * W
        sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        # Compute per-Post-processing-epoch residual magnitude after THIS solve.
        a_e_t = sol[0 * n_ctrl:1 * n_ctrl]
        a_n_t = sol[1 * n_ctrl:2 * n_ctrl]
        a_u_t = sol[2 * n_ctrl:3 * n_ctrl]
        fit_off = np.stack([B @ a_e_t, B @ a_n_t, B @ a_u_t], axis=1)
        fit_pp = cum_at_ppk + fit_off
        r_mag = np.linalg.norm(fit_pp - ppk_enu, axis=1)
        if not robust:
            break
        # MAD-based scale for the Huber threshold. Median absolute deviation
        # is robust to up to 50% outliers in the residual distribution.
        med = float(np.median(r_mag))
        mad = float(np.median(np.abs(r_mag - med)))
        sigma_r = max(0.5, 1.4826 * mad)
        cutoff = robust_huber_k * sigma_r
        # New per-Post-processing-epoch Huber weight: 1 inside cutoff, decays as
        # cutoff/|r| outside. Stack to match block-diagonal row layout
        # (each Post-processing epoch contributes 3 rows: E, N, U).
        new_w_pp = np.where(r_mag <= cutoff, 1.0, cutoff / np.maximum(r_mag, 1e-9))
        new_huber = np.ones(b.size)
        # Post-processing rows occupy [0:3*n_per_axis] in stacked order (E first,
        # then N, then U — each block has n_per_axis rows). Apply same
        # per-epoch weight to all three axes for that epoch.
        for ax in range(3):
            s = ax * n_per_axis
            e = (ax + 1) * n_per_axis
            new_huber[s:e] = new_w_pp
        # Convergence check.
        change = float(np.max(np.abs(new_huber - huber_w)))
        huber_w = new_huber
        n_down = int((new_w_pp < 1.0).sum())
        _log(f"[vio-fit] IRLS iter {it+1}: sigma_r={sigma_r:.2f}m cutoff={cutoff:.2f}m "
             f"downweighted={n_down} max_change={change:.4f}")
        if change < 1e-3:
            break

    a_e = sol[0 * n_ctrl:1 * n_ctrl]
    a_n = sol[1 * n_ctrl:2 * n_ctrl]
    a_u = sol[2 * n_ctrl:3 * n_ctrl]

    # Evaluate at Post-processing epochs for residual report.
    fit_offsets_ppk = np.stack([B @ a_e, B @ a_n, B @ a_u], axis=1)
    fit_traj_ppk = cum_at_ppk + fit_offsets_ppk
    res_ppk = fit_traj_ppk - ppk_enu
    rms = float(np.sqrt(np.mean(np.linalg.norm(res_ppk, axis=1) ** 2)))
    _log(f"[vio-fit] PPK fit RMS residual: {rms:.3f} m  (smaller "
         f"= closer to PPK; larger = more VIO-shape preserved)")

    # Sample output path at requested rate.
    dt_out = 1.0 / max(0.1, output_rate_hz)
    out_t = np.arange(t_start, t_end + dt_out, dt_out)
    cum_at_out, _ = _interp_cum_vio(vio_vels, out_t)
    B_out = _cubic_bspline_basis(out_t, knots)
    off_out = np.stack([B_out @ a_e, B_out @ a_n, B_out @ a_u], axis=1)
    traj_enu = cum_at_out + off_out

    enu_to_llh = _enu_to_llh_factory(ref_llh)
    fused: list[PosRow] = []
    for i, t in enumerate(out_t):
        lat, lon, h = enu_to_llh(*traj_enu[i])
        fused.append(PosRow(
            utc_s=float(t),
            lat_deg=lat, lon_deg=lon, h_m=h,
            quality=1, vn=float("nan"), ve=float("nan"), vu=float("nan"),
            ns=0,
        ))

    return VioFitResult(
        fused=fused,
        rms_ppk_residual_m=rms,
        ppk_residuals_m=np.linalg.norm(res_ppk, axis=1).tolist(),
        n_control_points=n_ctrl,
        lambda_smooth=lambda_smooth,
    )
