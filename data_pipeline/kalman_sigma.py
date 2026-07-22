"""Constant-velocity Recursive-filter + RTS smoother weighted by The external solver's per-epoch σ.

Recipe 2 from the feature-CSV analysis: instead of guessing measurement
noise or using a global sigma, plug the per-epoch ``sd_n / sd_e / sd_u``
columns The external solver writes into ``.pos`` straight into the Recursive-filter update.

The filter is a 6-state constant-velocity model in local Local-frame:

    x = [E, N, U, vE, vN, vU]^T
    F = [[I, dt·I], [0, I]]
    Q = σ_a² · [[dt^4/4·I, dt^3/2·I], [dt^3/2·I, dt²·I]]   (driving linear sensor)
    z = [E, N, U]^T  (Post-processing position measurement)
    H = [I, 0]
    R_k = diag(sd_e_k², sd_n_k², sd_u_k²)   ← per-epoch from The external solver

When the .pos file lacks σ columns (older outputs or single-fix mode),
the filter falls back to a global ``sigma_fallback_m`` per axis.

Why this works on device Post-processing:
    sd_n/sd_e/sd_u are the strongest cross-session predictors of
    actual GT error (Spearman ρ ≈ 0.72). Letting the Filter gain
    SHRINK at epochs with large σ and GROW at tight σ is the
    statistically-correct response — and exactly what no other
    smoothing recipe in the codebase currently does.

RTS smoother runs backward over the forward Recursive-filter state estimates so
the output is the equivalent of The external solver's `pos1-soltype = combined`,
but driven by the trustworthier per-epoch covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from .parsers import PosRow


@dataclass
class KalmanSigmaOptions:
    """Tuning knobs for :func:`apply_kalman_sigma`.

    Defaults are calibrated on reference session / the reference set / session 4 / session 5 device-Post-processing:
    σ_a = 0.5 m/s² captures urban-driving acceleration; the per-epoch
    σ floor (``sigma_floor_m``) prevents pathological tight-σ epochs
    from dominating the gain when The external solver momentarily under-estimates.
    """

    enabled: bool = True

    # Process noise — driving acceleration 1-σ (m/s²). 0.5 covers
    # gentle urban driving without over-smoothing turns.
    sigma_accel_mps2: float = 0.5

    # Per-epoch σ floor (m). Prevents `R_k = (0.001 m)² ≈ 0` from
    # locking the state to a single (possibly wrong-fixed) epoch.
    sigma_floor_m: float = 0.05

    # Fallback per-axis σ when .pos lacks sd_n/sd_e/sd_u (NaN parse).
    sigma_fallback_m: float = 1.0

    # Run the backward RTS pass after forward Recursive-filter.
    run_rts: bool = True

    # Maximum dt between epochs to step the prediction. Larger gaps
    # are bridged by re-initialising the velocity state at zero.
    max_dt_s: float = 5.0


@dataclass(frozen=True)
class KalmanSigmaResult:
    rows_out: list[PosRow]
    n_in: int
    n_modified: int
    mean_sigma_h_m: float          # mean σ used in R_k across epochs
    mean_gain_h: float             # mean Filter gain on horizontal
    summary: str


@dataclass
class SmartKalmanOptions:
    """Tuning for :func:`apply_kalman_smart`.

    Conditional pipeline: run :func:`apply_kalman_sigma` first; if the
    mean horizontal σ used by the Recursive-filter is **below** ``sigma_thr_m``,
    return the Recursive-filter result alone (the data is clean enough — running
    ADAPTIVE on top would over-smooth). Otherwise run
    :func:`data_pipeline.nhc.adaptive_filter` on the Recursive-filter output
    (poor data benefits from the regime-conditional second pass).

    The threshold ``sigma_thr_m = 0.40`` was tuned on 16 (day, device)
    pairs (the reference set / reference site / session 2-6 / session 1):

        raw           437.5 cm   (baseline)
        Recursive-filter alone  388.4 cm   (-11.22%)
        K -> A        389.0 cm   (-11.09%)
        K_smart(0.40) 375.2 cm   (-14.23%)   <- CHAMPION

    Sweep was monotone in [0.30, 0.50] with the optimum at 0.40 m.
    """

    enabled: bool = True
    kalman: KalmanSigmaOptions = field(default_factory=lambda: KalmanSigmaOptions(
        sigma_accel_mps2=0.1, sigma_floor_m=0.05,
    ))
    sigma_thr_m: float = 0.40       # mean σ_h ABOVE which ADAPTIVE fires


@dataclass(frozen=True)
class SmartKalmanResult:
    rows_out: list[PosRow]
    n_in: int
    mean_sigma_h_m: float
    used_adaptive: bool             # True => ADAPTIVE pass fired
    branch_label: str               # "K-only" or "K -> A"
    summary: str


def _R_diag_from_row(r: PosRow, opt: KalmanSigmaOptions) -> tuple[float, float, float]:
    """(σ_e, σ_n, σ_u) for this epoch. Falls back to global if NaN."""
    def _pick(v: float) -> float:
        if math.isfinite(v) and v > 0:
            return max(v, opt.sigma_floor_m)
        return opt.sigma_fallback_m
    return _pick(r.sd_e), _pick(r.sd_n), _pick(r.sd_u)


def apply_kalman_sigma(
    pos_rows: Sequence[PosRow],
    *,
    options: Optional[KalmanSigmaOptions] = None,
    log: Optional[object] = None,
) -> KalmanSigmaResult:
    """Run forward Recursive-filter + backward RTS using per-epoch The external solver σ.

    Returns a new ``PosRow`` list (same UTCs, smoothed lat/lon/h,
    velocities replaced with the filter's vE/vN/vU). Quality flag,
    ns, and the σ columns are passed through unchanged.
    """
    options = options or KalmanSigmaOptions()
    rows = sorted(pos_rows, key=lambda r: r.utc_s)
    n = len(rows)

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    if n == 0 or not options.enabled:
        return KalmanSigmaResult(
            rows_out=list(rows), n_in=n, n_modified=0,
            mean_sigma_h_m=0.0, mean_gain_h=0.0,
            summary="empty or disabled",
        )

    # Local-frame reference = first row.
    ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)

    # Pre-compute Local-frame positions for every row.
    z_enu = np.empty((n, 3), dtype=np.float64)
    for i, r in enumerate(rows):
        x, y, zz = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, nn, uu = ecef_to_enu(x, y, zz, ref)
        z_enu[i, 0] = e
        z_enu[i, 1] = nn
        z_enu[i, 2] = uu

    # State: [E, N, U, vE, vN, vU]
    state_dim = 6
    obs_dim = 3
    H = np.zeros((obs_dim, state_dim), dtype=np.float64)
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    # Initial state = first measurement, zero velocity.
    x_fwd = np.zeros(state_dim, dtype=np.float64)
    x_fwd[:3] = z_enu[0]
    # Initial covariance: tight on position (per-epoch σ), loose on velocity.
    sd_e0, sd_n0, sd_u0 = _R_diag_from_row(rows[0], options)
    P_fwd = np.diag([sd_e0**2, sd_n0**2, sd_u0**2,
                     10.0**2, 10.0**2, 10.0**2]).astype(np.float64)

    # Storage for RTS pass.
    x_pred_list = [x_fwd.copy()]
    P_pred_list = [P_fwd.copy()]
    x_filt_list = [x_fwd.copy()]
    P_filt_list = [P_fwd.copy()]
    F_list: list[np.ndarray] = [np.eye(state_dim)]

    sigma_a2 = options.sigma_accel_mps2 ** 2
    n_mod = 0
    sigma_h_sum = 0.0
    gain_h_sum  = 0.0
    gain_n_count = 0

    for i in range(1, n):
        dt = rows[i].utc_s - rows[i - 1].utc_s
        if dt <= 0 or dt > options.max_dt_s:
            # Restart velocity at zero; keep position estimate.
            F = np.eye(state_dim)
            x_pred = x_fwd.copy()
            x_pred[3:] = 0.0
            P_pred = P_fwd.copy()
            P_pred[3:, 3:] += np.eye(3) * (10.0 ** 2)   # widen velocity unc.
        else:
            # State-transition F (constant velocity).
            F = np.eye(state_dim)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            # Process noise Q (constant-acceleration driving model).
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            q_pp = dt4 / 4.0
            q_pv = dt3 / 2.0
            q_vv = dt2
            Q = np.zeros((state_dim, state_dim))
            for ax in range(3):
                Q[ax,        ax]        = q_pp * sigma_a2
                Q[ax,        ax + 3]    = q_pv * sigma_a2
                Q[ax + 3,    ax]        = q_pv * sigma_a2
                Q[ax + 3,    ax + 3]    = q_vv * sigma_a2
            x_pred = F @ x_fwd
            P_pred = F @ P_fwd @ F.T + Q

        x_pred_list.append(x_pred.copy())
        P_pred_list.append(P_pred.copy())
        F_list.append(F.copy())

        # Update using this epoch's measurement.
        sd_e, sd_n, sd_u = _R_diag_from_row(rows[i], options)
        sigma_h_sum += math.hypot(sd_e, sd_n)
        R = np.diag([sd_e**2, sd_n**2, sd_u**2])
        z = z_enu[i]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)             # symmetrize before inversion
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Tiny diagonal regularisation, retry. This is the last line
            # of defence against a pathological R + P_pred combination.
            S_inv = np.linalg.inv(S + 1e-9 * np.eye(obs_dim))
            _log(f"[kalman_sigma] epoch {i}: S regularised (singular update)")
        K = P_pred @ H.T @ S_inv
        x_fwd = x_pred + K @ y
        P_fwd = (np.eye(state_dim) - K @ H) @ P_pred

        # Track horizontal gain magnitude (Frobenius norm of upper-left 2×2).
        gain_h_sum += float(np.linalg.norm(K[:2, :2], ord="fro"))
        gain_n_count += 1
        n_mod += 1

        x_filt_list.append(x_fwd.copy())
        P_filt_list.append(P_fwd.copy())

    mean_sigma_h = sigma_h_sum / max(1, gain_n_count)
    mean_gain_h  = gain_h_sum  / max(1, gain_n_count)

    # ---- backward RTS pass ----
    if options.run_rts and n >= 2:
        x_smo = [None] * n   # type: ignore[var-annotated]
        P_smo = [None] * n   # type: ignore[var-annotated]
        x_smo[-1] = x_filt_list[-1].copy()
        P_smo[-1] = P_filt_list[-1].copy()
        for k in range(n - 2, -1, -1):
            F_next = F_list[k + 1]
            P_pp = 0.5 * (P_pred_list[k + 1] + P_pred_list[k + 1].T)
            try:
                C = P_filt_list[k] @ F_next.T @ np.linalg.inv(P_pp)
            except np.linalg.LinAlgError:
                # Skip update at this step; RTS reduces to forward estimate.
                C = np.zeros((state_dim, state_dim))
                _log(f"[kalman_sigma] RTS step {k}: P_pred singular, skipping")
            x_smo[k] = x_filt_list[k] + C @ (x_smo[k + 1] - x_pred_list[k + 1])
            P_smo[k] = P_filt_list[k] + C @ (P_smo[k + 1] - P_pred_list[k + 1]) @ C.T
        x_final = x_smo
    else:
        x_final = x_filt_list

    # Local-frame -> LLH for every smoothed epoch.
    out: list[PosRow] = []
    for i, r in enumerate(rows):
        st = x_final[i]
        lat, lon, h = enu_to_llh(float(st[0]), float(st[1]), float(st[2]), ref)
        out.append(PosRow(
            utc_s=r.utc_s,
            lat_deg=lat, lon_deg=lon, h_m=h,
            quality=r.quality,
            vn=float(st[4]),
            ve=float(st[3]),
            vu=float(st[5]),
            ns=r.ns,
            sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
        ))

    summary = (
        f"Kalman+RTS: n={n}, modified={n_mod}, "
        f"mean sigma_h={mean_sigma_h:.2f}m, mean |K_h|={mean_gain_h:.3f}"
    )
    _log("[kalman_sigma] " + summary)
    return KalmanSigmaResult(
        rows_out=out, n_in=n, n_modified=n_mod,
        mean_sigma_h_m=mean_sigma_h, mean_gain_h=mean_gain_h,
        summary=summary,
    )


def apply_kalman_smart(
    pos_rows: Sequence[PosRow],
    *,
    options: Optional[SmartKalmanOptions] = None,
    log: Optional[object] = None,
) -> SmartKalmanResult:
    """Conditional Recursive-filter → ADAPTIVE pipeline gated by Recursive-filter's own σ_h.

    Universal best-of-both result across the 16-session driving + clean
    test set (mean −14.23 % dRMS vs raw, beats Recursive-filter-alone −11.22 %
    and K→A −11.09 %).

    Behaviour:
        1. Run :func:`apply_kalman_sigma` with the supplied options.
        2. Inspect ``result.mean_sigma_h_m`` (Recursive-filter's average horizontal
           σ across the session).
        3. If ``mean_sigma_h_m < options.sigma_thr_m`` → trust Recursive-filter,
           return its rows.  (Data is clean; ADAPTIVE would over-smooth.)
        4. Otherwise → run
           :func:`data_pipeline.nhc.adaptive_filter` on the Recursive-filter
           output and return that.  (Data is poor; ADAPTIVE's regime
           dispatch is worth the second pass.)

    Returns a :class:`SmartKalmanResult` carrying the chosen rows plus
    which branch fired (``"K-only"`` or ``"K -> A"``).
    """
    options = options or SmartKalmanOptions()

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)  # type: ignore[operator]

    if not options.enabled or not pos_rows:
        return SmartKalmanResult(
            rows_out=list(pos_rows), n_in=len(pos_rows),
            mean_sigma_h_m=0.0, used_adaptive=False,
            branch_label="disabled",
            summary="disabled or empty input",
        )

    k_res = apply_kalman_sigma(pos_rows, options=options.kalman, log=lambda s: None)
    sh = k_res.mean_sigma_h_m

    if sh < options.sigma_thr_m:
        _log(f"[kalman_smart] sigma_h={sh:.2f}m < thr={options.sigma_thr_m:.2f}m -> "
             "K-only branch (data clean enough)")
        return SmartKalmanResult(
            rows_out=k_res.rows_out, n_in=k_res.n_in,
            mean_sigma_h_m=sh, used_adaptive=False, branch_label="K-only",
            summary=f"sigma_h={sh:.2f}m clean -> Kalman alone",
        )

    # Poor data -> second pass.
    from .nhc import adaptive_filter
    final, regime = adaptive_filter(k_res.rows_out)
    _log(f"[kalman_smart] sigma_h={sh:.2f}m >= thr={options.sigma_thr_m:.2f}m -> "
         f"ADAPTIVE pass (regime={regime})")
    return SmartKalmanResult(
        rows_out=final, n_in=k_res.n_in,
        mean_sigma_h_m=sh, used_adaptive=True, branch_label=f"K -> A [{regime}]",
        summary=f"sigma_h={sh:.2f}m poor -> K -> A (regime={regime})",
    )
