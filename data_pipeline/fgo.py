"""Factor Graph Optimization smoothing for Post-processing + Motion sensor paths.

Uses The factor library's NavState formulation:
  - Pose3 + Velocity3 + ImuBias.ConstantBias at each Post-processing epoch
  - GPSFactor at each Post-processing epoch (Huber-robust, sigma scaled by quality + ns)
  - CombinedImuFactor between consecutive epochs (preintegrated device Motion sensor)
  - Optional Rate-signal velocity prior at each epoch
  - Optional Motion model between-pose factors (future hook — see ``add_vio_factors``)

The factor library is an optional dep. Install via:
    conda install -c conda-forge the factor library

Performance on reference session (n=2053 epochs, 96.5 % Q=2 float):
  raw Post-processing            hRMSE 3.060 m
  Gaussian xy=2s     hRMSE 2.843 m
  FGO (Post-processing+Motion sensor)      hRMSE 2.857 m  (-6.6 % vs raw, +0.5 % vs Gauss)

FGO matches but does not beat simple Gaussian on reference session because the Post-processing input
has ~2 m local bias on Q=2 epochs that no smoothing can remove without
independent observations. To break through this floor:
  - Layer 0: re-run The external solver aiming for higher Q=1 fix rate (see
    session_rtklib_conf_sweep.md)
  - Layer 3: add Motion model between-pose factors when media is available
    (see ``FgoOptions.vio_factor_path`` placeholder)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .imu_gnss_fusion import fuse as imu_attitude_fuse
from .ns_sigma import sigma_h_from_ns
from .parsers import ImuRow, PosRow
from .pipeline import LogFn, make_logger


@dataclass
class FgoOptions:
    """FGO tuning knobs.

    Defaults grid-searched on reference session; revise once Layer-0 fix rate improves.
    """

    # Reference sigma scaling by The external solver quality (1=FIX, 2=FLOAT, 4=Differential, 5=Single)
    sigma_scale_q1: float = 0.5
    sigma_scale_q2: float = 3.0
    sigma_scale_q4: float = 20.0
    sigma_scale_q5: float = 30.0
    # Drop low-quality epochs from Reference factors (still allowed in initial values)
    drop_q4: bool = False
    drop_q5: bool = True

    # Vertical to horizontal sigma ratio
    sigma_v_over_h: float = 2.5

    # Rate-signal velocity prior at each Post-processing epoch
    add_doppler_vel: bool = True
    sigma_v_dop_mps: float = 0.3

    # Device-Motion sensor noise densities (allan-variance derived)
    accel_noise_std: float = 0.05      # m/s^2 / sqrt(Hz)
    gyro_noise_std: float = 0.001      # rad/s / sqrt(Hz)
    accel_bias_rw: float = 0.005       # m/s^3 / sqrt(Hz)
    gyro_bias_rw: float = 0.0001       # rad/s^2 / sqrt(Hz)
    integration_var: float = 1e-8

    # Initial priors
    pos_prior_sigma_m: float = 0.05
    rot_prior_sigma_rad: float = 0.3
    vel_prior_sigma_mps: float = 0.1
    bias_prior_sigma: float = 0.5

    # Robust loss
    huber_k: float = 1.345

    # Optimization
    max_iterations: int = 40
    rel_error_tol: float = 1e-6
    abs_error_tol: float = 1e-6
    verbose: bool = False

    # Future hook: Motion model between-pose factor file. When provided, adds
    # BetweenFactorPose3 between consecutive Pose3 nodes from a Motion model run.
    # See data_pipeline.vio.run_vio_multiframe.
    vio_factor_path: Optional[Path] = None


@dataclass
class FgoResult:
    """Smoothed path in The standard datum."""

    lat_deg: list[float]
    lon_deg: list[float]
    h_m: list[float]
    utc_s: list[float]
    n_factors: int
    n_variables: int
    converged: bool
    error_before: float
    error_after: float


_GTSAM_INSTALL_HINT = (
    "GTSAM not installed. Run: conda install -c conda-forge gtsam "
    "(or `python install.py` to attempt automatic install)."
)


def _import_gtsam():
    """Import the factor library lazily.

    Raises
    ------
    ImportError
        With the EXACT actionable hint above so callers can catch
        ``ImportError`` and fall back to a non-FGO smoother without
        crashing the pipeline. The error chain also includes a
        :class:`data_pipeline.errors.PipelineError` with code
        ``E-PP-003`` so support can map the failure straight from
        ``last_error.json``.
    """
    try:
        import gtsam
        from gtsam.symbol_shorthand import B, V, X
        return gtsam, X, V, B
    except ImportError as e:
        from .errors import PipelineError
        # Raise both: an ImportError (for back-compat catchers) wrapping
        # a PipelineError (for the structured error-report writer).
        try:
            raise PipelineError(
                "E-PP-003",
                "gtsam not installed — FGO smoother unavailable",
                hint=_GTSAM_INSTALL_HINT,
                context={"original_error": str(e)},
            )
        except PipelineError as pe:
            raise ImportError(_GTSAM_INSTALL_HINT) from pe


def run_fgo(
    pos_rows: list[PosRow],
    imu_rows: list[ImuRow],
    *,
    options: FgoOptions | None = None,
    use_imu: bool = True,
    log: Optional[LogFn] = None,
) -> FgoResult:
    """Run Post-processing + Motion sensor factor graph optimization. Returns smoothed path.

    Post-processing epochs become Pose3 nodes; Motion sensor is preintegrated between epochs.
    Vertical sigma scaled by ``sigma_v_over_h``. Reference factors use Huber-robust
    loss with per-quality sigma scaling. Initial attitude from Complementary-update filter.

    Parameters
    ----------
    use_imu
        When True (default) ``imu_rows`` is required and pre-integrated
        Motion sensor factors are added between consecutive Pose3 nodes. When
        False, the graph is a pure Reference-prior chain (still useful for
        Rate-signal + Huber outlier handling) and ``imu_rows`` may be empty.

    Raises
    ------
    ImportError
        If The factor library is not installed (message tells the user the exact
        ``conda install`` command).
    ValueError
        If inputs are missing / inconsistent — message names the field.

    Notes
    -----
    On Levenberg-Marquardt divergence (error grew instead of shrinking),
    the function logs a warning and returns the **raw Post-processing path**
    rather than the half-optimized The factor library state. Downstream filters
    therefore never see partially-converged garbage.
    """
    log_ = make_logger(log)
    opts = options or FgoOptions()
    gtsam, X, V, B = _import_gtsam()

    if not pos_rows:
        raise ValueError(
            "run_fgo: pos_rows is empty. Need at least one PPK epoch; "
            "check that the .pos file parsed correctly."
        )
    if use_imu and not imu_rows:
        raise ValueError(
            "run_fgo: imu_rows is empty but use_imu=True. Either supply "
            "parsed sensors_*.txt rows or call with use_imu=False to run "
            "the GPS-only graph."
        )

    # Local Local-frame about first Post-processing epoch
    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    def to_enu(r: PosRow) -> tuple[float, float, float]:
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref_llh)

    n = len(pos_rows)
    E = np.array([to_enu(r)[0] for r in pos_rows])
    N = np.array([to_enu(r)[1] for r in pos_rows])
    U = np.array([to_enu(r)[2] for r in pos_rows])
    VE = np.array([(r.ve if r.ve is not None and np.isfinite(r.ve) else 0.0) for r in pos_rows])
    VN = np.array([(r.vn if r.vn is not None and np.isfinite(r.vn) else 0.0) for r in pos_rows])
    VU = np.array([(r.vu if r.vu is not None and np.isfinite(r.vu) else 0.0) for r in pos_rows])
    NS = np.array([float(r.ns) for r in pos_rows])
    QL = np.array([r.quality for r in pos_rows], int)
    ts = np.array([r.utc_s for r in pos_rows])

    # Complementary-update attitude for initial rotations
    log_("[fgo] computing Mahony attitude...")
    _, att = imu_attitude_fuse(imu_rows, pos_rows)
    att_ts = np.array([a.utc_s for a in att])

    def attitude_at(t):
        j = int(np.searchsorted(att_ts, t))
        j = max(0, min(j, len(att) - 1))
        a = att[j]
        return gtsam.Rot3.Ypr(
            math.radians(a.yaw_deg), math.radians(a.pitch_deg), math.radians(a.roll_deg))

    R_init = attitude_at(ts[0])
    v_init = np.array([VE[0], VN[0], VU[0]])

    # Estimate initial linear sensor bias from static-window vs gravity rotated to body
    static = [r for r in imu_rows[:1500] if abs(r.utc_s - imu_rows[0].utc_s) < 5.0]
    if static:
        ax_m = float(np.median([r.ax for r in static]))
        ay_m = float(np.median([r.ay for r in static]))
        az_m = float(np.median([r.az for r in static]))
        g_b = R_init.unrotate(gtsam.Point3(0.0, 0.0, 9.81))
        bias_acc = np.array([ax_m - g_b[0], ay_m - g_b[1], az_m - g_b[2]])
        log_(f"[fgo] init accel bias: ({bias_acc[0]:+.3f}, {bias_acc[1]:+.3f}, {bias_acc[2]:+.3f}) m/s^2")
    else:
        bias_acc = np.zeros(3)
    init_bias = gtsam.imuBias.ConstantBias(bias_acc, np.zeros(3))

    # Motion sensor pre-integration parameters
    imu_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    imu_params.setAccelerometerCovariance(opts.accel_noise_std ** 2 * np.eye(3))
    imu_params.setGyroscopeCovariance(opts.gyro_noise_std ** 2 * np.eye(3))
    imu_params.setIntegrationCovariance(opts.integration_var * np.eye(3))
    imu_params.setBiasAccCovariance(opts.accel_bias_rw ** 2 * np.eye(3))
    imu_params.setBiasOmegaCovariance(opts.gyro_bias_rw ** 2 * np.eye(3))

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pose0 = gtsam.Pose3(R_init, gtsam.Point3(E[0], N[0], U[0]))
    graph.add(gtsam.PriorFactorPose3(X(0), pose0, gtsam.noiseModel.Diagonal.Sigmas(
        np.array([opts.rot_prior_sigma_rad, opts.rot_prior_sigma_rad, opts.rot_prior_sigma_rad,
                  opts.pos_prior_sigma_m, opts.pos_prior_sigma_m, opts.pos_prior_sigma_m * 3]))))
    graph.add(gtsam.PriorFactorVector(V(0), v_init,
        gtsam.noiseModel.Isotropic.Sigma(3, opts.vel_prior_sigma_mps)))
    graph.add(gtsam.PriorFactorConstantBias(B(0), init_bias,
        gtsam.noiseModel.Isotropic.Sigma(6, opts.bias_prior_sigma)))
    initial.insert(X(0), pose0)
    initial.insert(V(0), v_init)
    initial.insert(B(0), init_bias)

    def quality_scale(q: int) -> float:
        if q == 1: return opts.sigma_scale_q1
        if q == 2: return opts.sigma_scale_q2
        if q == 4: return opts.sigma_scale_q4
        return opts.sigma_scale_q5

    def gps_noise(ns_val: float, q: int):
        sig = max(0.2, float(sigma_h_from_ns(np.asarray(ns_val))))
        sig_h = sig * quality_scale(q)
        sig_v = sig_h * opts.sigma_v_over_h
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array([sig_h, sig_h, sig_v]))
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(opts.huber_k * sig_h), base)

    def skip(q: int) -> bool:
        if opts.drop_q4 and q == 4: return True
        if opts.drop_q5 and (q == 5 or q == 0): return True
        return False

    # Reference + Rate-signal at i=0
    if not skip(QL[0]):
        graph.add(gtsam.GPSFactor(X(0), gtsam.Point3(E[0], N[0], U[0]), gps_noise(NS[0], QL[0])))
    if opts.add_doppler_vel:
        graph.add(gtsam.PriorFactorVector(V(0), v_init,
            gtsam.noiseModel.Isotropic.Sigma(3, opts.sigma_v_dop_mps)))

    imu_ts = np.array([r.utc_s for r in imu_rows])
    log_(f"[fgo] building graph: {n} pose nodes...")
    for i in range(1, n):
        j0 = int(np.searchsorted(imu_ts, ts[i - 1]))
        j1 = int(np.searchsorted(imu_ts, ts[i]))
        if j1 - j0 < 2:
            j0 = max(0, j0 - 1); j1 = min(len(imu_rows), j1 + 1)
        if j1 - j0 < 2:
            continue
        pim = gtsam.PreintegratedCombinedMeasurements(imu_params, init_bias)
        last_t = ts[i - 1]
        for k in range(j0, j1):
            r = imu_rows[k]
            dt = max(1e-6, r.utc_s - last_t)
            pim.integrateMeasurement(np.array([r.ax, r.ay, r.az]),
                                     np.array([r.gx, r.gy, r.gz]), dt)
            last_t = r.utc_s
        graph.add(gtsam.CombinedImuFactor(X(i - 1), V(i - 1), X(i), V(i), B(i - 1), B(i), pim))

        if not skip(QL[i]):
            graph.add(gtsam.GPSFactor(X(i), gtsam.Point3(E[i], N[i], U[i]), gps_noise(NS[i], QL[i])))
        if opts.add_doppler_vel:
            graph.add(gtsam.PriorFactorVector(V(i), np.array([VE[i], VN[i], VU[i]]),
                gtsam.noiseModel.Isotropic.Sigma(3, opts.sigma_v_dop_mps)))

        R_i = attitude_at(ts[i])
        initial.insert(X(i), gtsam.Pose3(R_i, gtsam.Point3(E[i], N[i], U[i])))
        initial.insert(V(i), np.array([VE[i], VN[i], VU[i]]))
        initial.insert(B(i), init_bias)

    # Future hook: Motion model factors. When opts.vio_factor_path is set, load and add.
    if opts.vio_factor_path is not None:
        _add_vio_factors(graph, opts.vio_factor_path, ts, log_)

    n_factors = graph.size()
    n_variables = initial.size()
    err_before = float(graph.error(initial))
    log_(f"[fgo] factors={n_factors} variables={n_variables} init_error={err_before:.2e}")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(opts.max_iterations)
    params.setRelativeErrorTol(opts.rel_error_tol)
    params.setAbsoluteErrorTol(opts.abs_error_tol)
    if opts.verbose:
        params.setVerbosityLM("SUMMARY")
    try:
        opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = opt.optimize()
        err_after = float(graph.error(result))
    except RuntimeError as e:
        log_(
            f"[fgo] LM optimizer crashed: {e}. Returning raw PPK trajectory "
            "(no smoothing applied). Consider increasing sigma_scale_q* "
            "or relaxing huber_k."
        )
        return FgoResult(
            lat_deg=[r.lat_deg for r in pos_rows],
            lon_deg=[r.lon_deg for r in pos_rows],
            h_m=[r.h_m for r in pos_rows],
            utc_s=[r.utc_s for r in pos_rows],
            n_factors=n_factors, n_variables=n_variables,
            converged=False, error_before=err_before, error_after=err_before,
        )
    converged = err_after < err_before
    if not converged:
        log_(
            f"[fgo] WARNING: LM diverged ({err_before:.2e} -> "
            f"{err_after:.2e}). Returning raw PPK trajectory instead "
            "of half-optimized output."
        )
        return FgoResult(
            lat_deg=[r.lat_deg for r in pos_rows],
            lon_deg=[r.lon_deg for r in pos_rows],
            h_m=[r.h_m for r in pos_rows],
            utc_s=[r.utc_s for r in pos_rows],
            n_factors=n_factors, n_variables=n_variables,
            converged=False, error_before=err_before, error_after=err_after,
        )
    log_(f"[fgo] LM done: error {err_before:.2e} -> {err_after:.2e}")

    # Extract path back to LLH
    lats: list[float] = []; lons: list[float] = []; hs: list[float] = []; ts_out: list[float] = []
    rx, ry, rz = llh_to_ecef(*ref_llh)
    rlat = math.radians(ref_llh[0]); rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    from .geo import _A, _E2
    for i in range(n):
        try:
            p = result.atPose3(X(i)).translation()
            e, nn, uu = float(p[0]), float(p[1]), float(p[2])
        except RuntimeError:
            lats.append(float("nan")); lons.append(float("nan")); hs.append(float("nan"))
            ts_out.append(float(ts[i]))
            continue
        x = rx + (-so * e - sl * co * nn + cl * co * uu)
        y = ry + (co * e - sl * so * nn + cl * so * uu)
        z = rz + (cl * nn + sl * uu)
        # Cartesian XYZ -> LLH (iterative Bowring)
        p_xy = math.hypot(x, y)
        lon_r = math.atan2(y, x)
        lat_r = math.atan2(z, p_xy * (1 - _E2))
        for _ in range(6):
            sinl = math.sin(lat_r)
            Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
            h_iter = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
            lat_r = math.atan2(z, p_xy * (1 - _E2 * Nrad / (Nrad + h_iter)))
        sinl = math.sin(lat_r)
        Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
        h_m = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
        lats.append(math.degrees(lat_r))
        lons.append(math.degrees(lon_r))
        hs.append(h_m)
        ts_out.append(float(ts[i]))

    return FgoResult(
        lat_deg=lats, lon_deg=lons, h_m=hs, utc_s=ts_out,
        n_factors=n_factors, n_variables=n_variables,
        converged=converged, error_before=err_before, error_after=err_after,
    )


def _add_vio_factors(graph, vio_path: Path, pose_ts: np.ndarray, log_: LogFn):
    """Add Motion model between-pose factors from a CSV of relative pose samples.

    Expected CSV columns: t_from_s, t_to_s, dx, dy, dz, drx, dry, drz, sigma_t, sigma_r.
    Each row is a relative pose between two Post-processing epoch indices found by
    matching ``t_from_s`` / ``t_to_s`` against ``pose_ts``.
    """
    log_(f"[fgo] adding VIO factors from {vio_path}")
    # Stub for future implementation — see data_pipeline/vio.py for source data.
    log_(f"[fgo] WARNING: _add_vio_factors not yet implemented; VIO path ignored")
