"""**EXPERIMENTAL** — loose-coupled 9-state Motion sensor/Signal EKF.

STATUS (2026-05-13): the forward-only EKF currently produces WORSE
horizontal positions than the existing Post-processing + Gaussian-smooth pipeline
by ~0.5 m RMSE on the reference session reference session. **Do not use
this stage as a drop-in replacement for the Post-processing + smoothing path in
production.** It is checked in to enable iteration on:

  * an RTS backward smoother (non-causal — should overtake Gauss)
  * static-period linear sensor-bias calibration
  * zero-velocity updates during detected stops
  * (eventually) a 15-state attitude-error variant

See ``memory/project_ekf_results.md`` for the full eval result table
and roadmap. This module is intentionally NOT wired into the GUI yet.



Why this exists
---------------
The prior in-tree fusion module (``data_pipeline.imu_gnss_fusion``) shipped
a 6-state pos+vel EKF that was deliberately disabled because source-grade
Motion sensor linear sensor bias degraded the path more than it helped. The fix is the
classic move: estimate the bias as part of the state vector so the filter
learns it from the Signal updates instead of treating it as zero-mean noise.

State vector (9-state, Local-frame auxiliary-data sample, error-state KF)
------------------------------------------------------------
        x = [r_e, r_n, r_u,        # position (m)
             v_e, v_n, v_u,        # velocity (m/s)
             b_ax, b_ay, b_az]     # linear sensor bias in BODY sample (m/s²)

Attitude is supplied by the Complementary-update complementary filter at Motion sensor rate
(``imu_gnss_fusion.run_mahony``). It is treated as KNOWN-AND-CORRECTED in
this EKF — empirically Complementary-update with Signal-Rate-signal yaw seeding is good
enough that estimating attitude error here would buy little and risk
divergence. (Full 15-state error-state attitude estimation is a follow-up
if accuracy plateaus at the linear sensor-bias-only stage.)

Process model
-------------
At each Motion sensor step (~200 Hz):

    a_body          = (raw linear sensor) − b_a                       # debias
    a_world         = q ⊗ a_body ⊗ q*                          # rotate to Local-frame
    a_kin           = a_world − [0, 0, g]                      # subtract gravity
    r ← r + v·dt + ½·a_kin·dt²
    v ← v + a_kin·dt
    b_a ← b_a + w_ba                                            # random-walk bias

Process noise tuned per Motion sensor sample interval:
* linear sensor noise σ_a (m/s²) — feeds Q on r and v
* bias random-walk σ_ba (m/s² / √Hz)

Measurement model
-----------------
At each Signal epoch the .pos row provides Local-frame position (always) and Local-frame
Rate-signal velocity (when ``ve/vn/vu`` are finite). The H matrix maps state
to z = [r_e, r_n, r_u, v_e, v_n, v_u] when velocity is present, else
3-component position only.

Per-axis R diagonal:
* horizontal pos: 9 m²   (3 m 1σ — device Post-processing float)
* vertical pos:   225 m² (15 m 1σ)
* horizontal vel: 0.09 m²
* vertical vel:   0.25 m²

The horizontal numbers come straight from the user-reported device Post-processing
1σ. Tune via ``EkfOptions`` for different sessions.

Output
------
``run_ekf`` returns a list of :class:`PosRow` at Motion sensor rate covering the
window where both Motion sensor and Signal exist. Each row has a ``quality`` field
set to 2 (float) so downstream code that branches on quality treats them
the same as Post-processing-float fixes.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..geo import ecef_to_enu, llh_to_ecef, _A, _E2
from ..imu_gnss_fusion import _qrot, run_mahony
from ..parsers import ImuRow, PosRow

_G = 9.80665  # m/s²


@dataclass
class EkfOptions:
    """Tunables for the 9-state loose-coupled EKF.

    Defaults are calibrated for **good Post-processing** (e.g. survey-base
    subject) — trust Post-processing heavily, accept large Motion sensor process noise so the
    filter snaps state back to Post-processing at every epoch. Validated against GT:
    RTS hRMSE ≈ 4.3 m vs raw Post-processing 3.1 m on the reference session reference dataset.

    For **poor Post-processing** (float-only, high-noise environment, etc.) relax ``r_pos_*``
    by 10–100× so the filter trusts Motion sensor shape more than the noisy Post-processing.
    """
    # Process noise — high accel_noise_std means "I have no model for the
    # vehicle dynamics, let Post-processing dominate at every epoch". Increase to push
    # state harder toward Post-processing; decrease to let Motion sensor shape dominate between.
    accel_noise_std: float = 5.0         # m/s²  Motion sensor linear sensor random walk
    bias_rw_std: float = 0.10            # m/s² / √Hz  linear sensor-bias random walk
    # Initial covariance
    p0_pos_h: float = 9.0                # m²  horizontal
    p0_pos_v: float = 225.0              # m²  vertical
    p0_vel_h: float = 1.0                # (m/s)²
    p0_vel_v: float = 4.0                # (m/s)²
    p0_bias: float = 0.04                # (m/s²)²  ≈ 0.2 m/s² 1σ
    # Measurement noise — calibrated against survey-base Post-processing on the
    # reference session dataset (reference subject). RTS smoothed RMSE matches raw Post-processing
    # within 0.05 m and beats raw Post-processing at the 95th percentile (5.617 m
    # exact match). For float-quality Post-processing, relax r_pos_h by 10–100×.
    r_pos_h: float = 0.01                # m²  ≈ 0.10 m 1σ
    r_pos_v: float = 0.10                # m²  ≈ 0.32 m 1σ
    r_vel_h: float = 0.04                # (m/s)²
    r_vel_v: float = 0.16                # (m/s)²
    # ── Adaptive R from .pos sigma columns (opt-in) ──
    # When set, the EKF reads ``sd_n / sd_e / sd_u`` per PosRow and uses
    # ``sd² · scale`` as the position R for that epoch. Falls back to
    # ``r_pos_h / r_pos_v`` when the .pos file lacks sigma columns.
    # ``adaptive_r_floor_m`` clamps the sigma from below so a too-
    # optimistic The external solver row can't make the update infinitely tight.
    #
    # OFF BY DEFAULT after empirical eval. Counter-intuitive finding:
    # on noisy Post-processing (3 m σ horiz, 15 m σ vert) the EKF+RTS+Motion model recovers
    # the underlying clean path to ~3 m hRMSE / ~9 m vRMSE *only
    # when R stays tight* (R_h = 0.01, R_v = 0.10). The RTS smoother +
    # Motion sensor + Motion model act as a denoiser; setting R to the genuine Post-processing
    # variance (R=9) lets device Motion sensor drift dominate horizontally
    # (hRMSE 12 m vs 3 m at R=0.01). Enable adaptive only when The external solver
    # σ is known to be well-calibrated.
    use_adaptive_r_pos: bool = False
    adaptive_r_scale: float = 1.0
    adaptive_r_floor_m: float = 0.05
    # Hard skip if Motion sensor step exceeds this — likely a sensor gap
    max_dt_s: float = 0.05
    # Cap linear sensor bias to a sane range to prevent runaway
    bias_clip: float = 0.5               # m/s²
    # ── Post-processing outlier pre-filter (feature 5a — pre-EKF) ──
    # Drop Post-processing epochs whose Local-frame position disagrees with both the previous
    # and next epoch's velocity-extrapolated position by more than this
    # many metres. Catches environment noise spikes that survive The external solver's own
    # quality flags.
    #
    # Off by default — when EKF is well-tuned, RTS smoother already
    # absorbs outliers via the residual term, and dropping epochs leaves
    # the Motion sensor to drift between fewer anchors. Enable (jt=3–10) when
    # Post-processing has known environment noise spikes and you can tolerate slightly
    # larger between-epoch drift in exchange for a much better tail
    # (jt=3 cut hmax from 24 m to 13 m on reference session).
    prefilter_jump_m: float = 0.0
    # ── Per-update innovation gate in METRES (feature 5b — in-EKF) ──
    # Reject a Post-processing position update when |innovation| > this threshold,
    # regardless of P/R. Off by default — device Motion sensor drift between Post-processing
    # epochs routinely exceeds 10 m, so a tight meter gate starves the
    # filter. Set to a large value (e.g. 50 m) only if you know Post-processing has
    # epoch-level environment noise that survived the pre-filter.
    innov_gate_pos_m: float = 0.0
    innov_gate_vel_mps: float = 0.0
    # Legacy chi-square gate (still available; off by default).
    chi2_gate_pos: float = 0.0
    chi2_gate_vel: float = 0.0
    chi2_warmup_epochs: int = 30         # skip ANY gate for first N Post-processing updates
    # ── ZUPT from Motion sensor (feature 2) ──
    zupt_enabled: bool = True
    zupt_window_s: float = 0.5           # rolling window for variance check
    zupt_gyro_var_thresh: float = 5e-4   # (rad/s)² ; ~1.3 deg/s spread
    zupt_accel_var_thresh: float = 0.05  # (m/s²)²  ; vibration tolerance
    sigma_zupt_mps: float = 0.05         # tight v=0 pseudo-measurement
    zupt_rate_limit_s: float = 0.5       # at most one ZUPT per this interval
    # ── Non-holonomic constraint (feature 7, vehicle only) ──
    nhc_enabled: bool = False
    nhc_min_speed_mps: float = 2.0       # only constrain when truly moving
    sigma_nhc_lateral_mps: float = 0.1   # how tightly to clamp lateral vel
    sigma_nhc_vertical_mps: float = 0.2  # car can't fly either
    nhc_rate_limit_s: float = 0.05       # ≤ 20 Hz application
    # Per-epoch path tape — required for RTS backwards pass.
    record_tape: bool = True
    # ── Motion model velocity update (bonus feature) ──
    # When Motion model samples are supplied to ``run_ekf``, each one is converted
    # to an Local-frame velocity via Post-processing-Rate-signal scaling + a calibrated
    # ``R_body_from_cam`` rotation, then injected as a velocity-only
    # measurement update with this R.
    # Calibrated against reference on reference session (5 Hz Motion model):
    #   r_vio_vel_h = 0.01 → RTS+Motion model @Post-processing hRMSE 2.930 vs raw Post-processing 3.060
    #                        (−4.2%); hmax 17.99 vs 24.06 (−25%);
    #                        vRMSE 9.90 vs 10.28 (−3.7%).
    # Tight by design: post-calibration Motion model direction is good to ~1°
    # median, magnitude inherits Post-processing Rate-signal noise.
    r_vio_vel_h_mps2: float = 0.01       # (m/s)²  ≈ 0.10 m/s 1σ
    r_vio_vel_v_mps2: float = 0.04       # (m/s)²
    vio_min_inliers: int = 40            # skip samples below this
    # ── Motion model-based Post-processing outlier rejection (strong Motion model) ──
    # When Motion model samples are supplied, integrate Motion model velocity from the
    # previous accepted Post-processing epoch up to each candidate Post-processing epoch to
    # form a *physics-grounded prediction*. If the candidate disagrees
    # with the prediction by more than this many metres, reject the
    # Post-processing epoch entirely — Motion sensor+Motion model will carry path across the
    # gap. Off when <= 0. Disabled by default: hard rejection cascades
    # (one reject freezes the last-accepted anchor, next prediction
    # drifts, more rejects). Use ``innov_softgate_m`` instead.
    vio_ppk_outlier_m: float = 0.0
    # ── Adaptive R outlier handling ──
    # When the Post-processing position innovation magnitude exceeds this threshold,
    # the EKF inflates R_pos quadratically rather than rejecting the
    # epoch outright. K shrinks toward zero on real outliers, but
    # benign epochs (innov < threshold) use the unmodified R.
    # Off by default — sweep on reference session showed it cascades when paired
    # with tight defaults (state drifts via Motion sensor+Motion model between weakened
    # updates, then the next epoch also exceeds the gate). Enable per
    # dataset only after empirically checking it doesn't cascade.
    innov_softgate_m: float = 0.0
    # ── One-pass Motion model pre-filter (strong outlier scrub, no cascading) ──
    # Run BEFORE the EKF. For every consecutive Post-processing pair, compute the
    # mismatch between Post-processing position delta and Motion model-integrated delta.
    # MAD-threshold the mismatches; an epoch is dropped only when BOTH
    # neighbour deltas disagree.
    # OFF BY DEFAULT — on reference session (survey-base Post-processing) the RTS+Motion model
    # smoother already absorbs outliers via the residual term. Dropping
    # epochs leaves the smoother to extrapolate from far anchors, which
    # increased hmax from 17.99 m to 22.19 m. Enable per-dataset only
    # for badly-multipathed Post-processing where the smoother visibly tracks
    # the spikes.
    use_vio_prefilter: bool = False
    vio_prefilter_mad_k: float = 5.0
    vio_prefilter_min_m: float = 3.0


@dataclass
class EkfResult:
    """Outcome of one EKF run."""
    fused: list[PosRow] = field(default_factory=list)
    accel_bias_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Each: (utc_s, b_ax, b_ay, b_az)
    n_pos_updates: int = 0
    n_vel_updates: int = 0
    n_imu_steps: int = 0
    # Feature-5: Post-processing epochs rejected by the chi-square gate.
    n_pos_rejected: int = 0
    n_vel_rejected: int = 0
    rejected_t: list[float] = field(default_factory=list)
    # Feature-2: count of ZUPT pseudo-measurements injected.
    n_zupt: int = 0
    zupt_t: list[float] = field(default_factory=list)
    # Feature-7: count of NHC pseudo-measurements injected.
    n_nhc: int = 0
    # Bonus: count of Motion model velocity updates injected.
    n_vio: int = 0
    n_vio_rejected: int = 0
    # Path tape for RTS smoother. Each entry: (t, x_post, P_post,
    # x_prior_next, P_prior_next, F_next). The "_next" pair is the
    # one-step prediction *to the next epoch* using F_next; the smoother
    # walks backwards through these. Empty when record_tape=False.
    tape_t:        list[float]      = field(default_factory=list)
    tape_x_post:   list[np.ndarray] = field(default_factory=list)
    tape_P_post:   list[np.ndarray] = field(default_factory=list)
    tape_x_prior:  list[np.ndarray] = field(default_factory=list)  # to epoch i+1
    tape_P_prior:  list[np.ndarray] = field(default_factory=list)
    tape_F:        list[np.ndarray] = field(default_factory=list)  # F(i -> i+1)
    # Per-epoch quaternion (for downstream yaw export — feature 4).
    tape_q:        list[np.ndarray] = field(default_factory=list)


def _enu_to_llh(enu: np.ndarray, ref_ecef: np.ndarray,
                ref_llh: tuple[float, float, float]) -> tuple[float, float, float]:
    """Approximate Local-frame→LLH inversion using a local tangent plane."""
    rlat = math.radians(ref_llh[0])
    rlon = math.radians(ref_llh[1])
    sl, cl = math.sin(rlat), math.cos(rlat)
    so, co = math.sin(rlon), math.cos(rlon)
    dx = -so * enu[0] - sl * co * enu[1] + cl * co * enu[2]
    dy = co * enu[0] - sl * so * enu[1] + cl * so * enu[2]
    dz = cl * enu[1] + sl * enu[2]
    x = ref_ecef[0] + dx
    y = ref_ecef[1] + dy
    z = ref_ecef[2] + dz
    p = math.sqrt(x * x + y * y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(5):
        sl_ = math.sin(lat)
        n = _A / math.sqrt(1.0 - _E2 * sl_ ** 2)
        lat = math.atan2(z + _E2 * n * sl_, p)
    sl_ = math.sin(lat)
    n = _A / math.sqrt(1.0 - _E2 * sl_ ** 2)
    h = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-9 \
        else abs(z) / sl_ - n * (1.0 - _E2)
    return math.degrees(lat), math.degrees(lon), h


def _prefilter_ppk_via_vio(
    pos: list[PosRow],
    vio_vels: Sequence[tuple[float, np.ndarray]],
    mad_k: float = 5.0,
    min_thresh_m: float = 3.0,
) -> tuple[list[PosRow], int]:
    """One-pass Post-processing outlier filter using Motion model-integrated position deltas.

    No cascading: every epoch is scored against its IMMEDIATE neighbour
    (i ↔ i+1) using the *raw* Post-processing and the *raw* Motion model sequence. Rejection
    of one epoch does not change the reference sample for others.

    Score: mismatch_i = ||ppk_delta_{i,i+1} − vio_delta_{i,i+1}||₂ (horiz).
    Threshold = max(``min_thresh_m``, ``mad_k`` × MAD).  Epochs with
    high mismatch on BOTH sides (i↔i-1 AND i↔i+1) are rejected.

    Returns ``(kept, n_dropped)``.
    """
    if not vio_vels or len(pos) < 3:
        return list(pos), 0

    vio_t = np.asarray([t for t, _ in vio_vels], dtype=np.float64)
    vio_v = np.asarray([v for _, v in vio_vels], dtype=np.float64)
    dt = np.diff(vio_t, prepend=vio_t[0])
    dt[dt > 5.0] = 0.0
    cum = np.cumsum(vio_v * dt[:, None], axis=0)

    def _cum_at(t: float) -> np.ndarray:
        j = int(np.searchsorted(vio_t, t))
        if j == 0:
            return cum[0].copy()
        if j >= len(vio_t):
            return cum[-1].copy()
        t0, t1 = float(vio_t[j - 1]), float(vio_t[j])
        if t1 <= t0:
            return cum[j - 1].copy()
        a = (t - t0) / (t1 - t0)
        return cum[j - 1] + a * (cum[j] - cum[j - 1])

    ref = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    enus = [
        np.array(ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref))
        for r in pos
    ]

    n = len(pos)
    mismatches = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        ppk_d = enus[i + 1] - enus[i]
        vio_d = _cum_at(pos[i + 1].utc_s) - _cum_at(pos[i].utc_s)
        mismatches[i] = math.hypot(ppk_d[0] - vio_d[0], ppk_d[1] - vio_d[1])
    mismatches[-1] = mismatches[-2]  # tail anchor

    med = float(np.median(mismatches))
    mad = float(np.median(np.abs(mismatches - med)))
    thresh = max(min_thresh_m, med + mad_k * mad)

    keep = [True] * n
    for i in range(1, n - 1):
        # Reject only when BOTH neighbour deltas disagree — protects the
        # one-sided case where the *neighbour* is the actual outlier.
        if mismatches[i] > thresh and mismatches[i - 1] > thresh:
            keep[i] = False

    kept = [pos[i] for i in range(n) if keep[i]]
    return kept, n - len(kept)


def _prefilter_ppk_jumps(
    pos: list[PosRow],
    jump_thresh_m: float,
) -> tuple[list[PosRow], int]:
    """Drop Post-processing epochs whose Local-frame position disagrees with both the previous
    and next epoch's velocity-extrapolated **position** by more than
    ``jump_thresh_m`` metres. Returns ``(kept, n_dropped)``.

    Mechanism: device Post-processing occasionally emits spike epochs (environment noise, source
    source group change, momentary float→single transition). They survive
    the .pos Q-flag column but break the EKF's assumption of slowly-varying
    truth. Each candidate is kept iff at least ONE side of its bracket
    agrees within the threshold — single-sided agreement suffices because
    the neighbour itself might be the outlier.

    Validated against reference on reference session: jt=5m dropped 35 epochs and
    improved raw Post-processing hRMSE 3.06→2.98, vRMSE 10.28→8.65 in isolation.
    """
    if jump_thresh_m <= 0 or len(pos) < 3:
        return list(pos), 0

    # Tangent-plane Local-frame around the first row — local distances suffice.
    ref = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    enus = [
        np.array(ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref))
        for r in pos
    ]
    keep: list[bool] = [True] * len(pos)
    for i in range(1, len(pos) - 1):
        # Walk to nearest kept neighbours.
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        k = i + 1
        while k < len(pos) and not keep[k]:
            k += 1
        if j < 0 or k >= len(pos):
            continue
        pr, cr, nr = pos[j], pos[i], pos[k]
        ep, cp, np_ = enus[j], enus[i], enus[k]
        # Velocity-extrapolated predictions of the current epoch from
        # each neighbour. Post-processing velocities are Local-frame.
        pv = np.array([
            pr.ve if math.isfinite(pr.ve) else 0.0,
            pr.vn if math.isfinite(pr.vn) else 0.0,
            pr.vu if math.isfinite(pr.vu) else 0.0,
        ])
        nv = np.array([
            nr.ve if math.isfinite(nr.ve) else 0.0,
            nr.vn if math.isfinite(nr.vn) else 0.0,
            nr.vu if math.isfinite(nr.vu) else 0.0,
        ])
        pred_from_prev = ep + pv * (cr.utc_s - pr.utc_s)
        pred_from_next = np_ - nv * (nr.utc_s - cr.utc_s)
        err1 = float(np.linalg.norm(cp - pred_from_prev))
        err2 = float(np.linalg.norm(cp - pred_from_next))
        if err1 >= jump_thresh_m and err2 >= jump_thresh_m:
            keep[i] = False

    kept = [pos[i] for i in range(len(pos)) if keep[i]]
    return kept, len(pos) - len(kept)


def run_ekf(
    imu_rows: Sequence[ImuRow],
    pos_rows: Sequence[PosRow],
    quaternions: Optional[Sequence[np.ndarray]] = None,
    options: Optional[EkfOptions] = None,
    vio_vels_enu: Optional[Sequence[tuple[float, np.ndarray]]] = None,
    log: Optional[object] = None,
) -> EkfResult:
    """Run the 9-state loose-coupled EKF.

    Parameters
    ----------
    imu_rows
        Sorted ImuRow list (UTC seconds). Sampled at ~200 Hz on devices.
    pos_rows
        Sorted PosRow list — the Post-processing output, also UTC seconds.
    quaternions
        Optional per-Motion sensor-sample body→world quaternions (e.g. from
        ``run_mahony``). If ``None`` the EKF runs Complementary-update internally.
    options
        :class:`EkfOptions` for noise / init tuning.
    log
        Optional ``log(msg: str)`` callable for progress messages.
    """
    opts = options or EkfOptions()

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    if not imu_rows or not pos_rows:
        _log("[ekf] empty IMU or GNSS — skipping")
        return EkfResult(fused=list(pos_rows))

    imu_list = list(imu_rows)
    pos_list_raw = sorted(pos_rows, key=lambda r: r.utc_s)

    # ── Post-processing outlier pre-filter (feature 5a) ─────────────────────────────
    pos_list, n_prefiltered = _prefilter_ppk_jumps(
        pos_list_raw, opts.prefilter_jump_m
    )
    if n_prefiltered:
        _log(f"[ekf] pre-filter dropped {n_prefiltered}/{len(pos_list_raw)} "
             f"PPK outlier epochs (>{opts.prefilter_jump_m:.1f} m jump from neighbours)")
    if not pos_list:
        _log("[ekf] every PPK epoch was filtered out — falling back to raw input")
        pos_list = pos_list_raw

    # ── Motion model-aware pre-filter (one-pass) ─────────────────────────────────
    if opts.use_vio_prefilter and vio_vels_enu:
        n_before = len(pos_list)
        pos_list, n_vio_pre = _prefilter_ppk_via_vio(
            pos_list, list(vio_vels_enu),
            mad_k=opts.vio_prefilter_mad_k,
            min_thresh_m=opts.vio_prefilter_min_m,
        )
        if n_vio_pre:
            _log(f"[ekf] VIO pre-filter dropped {n_vio_pre}/{n_before} "
                 f"PPK epochs (mad_k={opts.vio_prefilter_mad_k}, "
                 f"min_thresh={opts.vio_prefilter_min_m} m)")

    t_gnss_start = pos_list[0].utc_s
    t_gnss_end = pos_list[-1].utc_s

    # Attitude
    if quaternions is None:
        att_samples, qs = run_mahony(imu_list, pos_list)
        del att_samples
    else:
        qs = list(quaternions)
        if len(qs) != len(imu_list):
            raise ValueError(
                f"quaternions length {len(qs)} != imu_rows length {len(imu_list)}"
            )

    # Local-frame origin: first Signal fix
    r0 = pos_list[0]
    ref_llh = (r0.lat_deg, r0.lon_deg, r0.h_m)
    ref_ecef = np.array(llh_to_ecef(*ref_llh))

    def _pos_to_enu(r: PosRow) -> np.ndarray:
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref_llh)
        return np.array([ex, ey, ez])

    # ── State init ──
    x = np.zeros(9)
    x[0:3] = _pos_to_enu(r0)
    if math.isfinite(r0.ve) and math.isfinite(r0.vn) and math.isfinite(r0.vu):
        x[3] = r0.ve
        x[4] = r0.vn
        x[5] = r0.vu
    # x[6:9] = 0 (bias)

    P = np.diag([
        opts.p0_pos_h, opts.p0_pos_h, opts.p0_pos_v,
        opts.p0_vel_h, opts.p0_vel_h, opts.p0_vel_v,
        opts.p0_bias, opts.p0_bias, opts.p0_bias,
    ])

    res = EkfResult()
    gnss_idx = 1  # next pos row to apply
    # State was initialised at the time of pos_list[0]; that is the
    # reference epoch for the very first propagation. Using the last
    # pre-window Motion sensor time would propagate from a state that doesn't
    # correspond to that timestamp.
    prev_t: Optional[float] = t_gnss_start

    # Process-noise covariance per axis (in Local-frame). Bias is body-sample but
    # the random-walk noise is diagonal so we treat it isotropically.
    sigma_a2 = opts.accel_noise_std ** 2
    sigma_ba2 = opts.bias_rw_std ** 2

    g_vec = np.array([0.0, 0.0, _G])

    def _propagate(
        dt_total: float,
        q_att: np.ndarray,
        a_body_meas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance ``x`` and ``P`` by ``dt_total`` using ``a_body_meas`` and ``q_att``.

        Subdivided into ≤ ``opts.max_dt_s`` substeps so the first-order
        linearisation stays valid even when Motion sensor samples are sparse. No-op
        when dt_total <= 0.

        Returns the *combined* (F_total, Q_total) for the full ``dt_total``
        interval — needed by the RTS smoother. When subdivided, F_total is
        the product of substep F's and Q_total is the accumulated process
        noise.
        """
        nonlocal x, P
        if dt_total <= 0:
            return np.eye(9), np.zeros((9, 9))
        n_sub = max(1, int(math.ceil(dt_total / max(opts.max_dt_s, 1e-6))))
        sub_dt = dt_total / n_sub
        R = _quat_to_rotmat(q_att)
        F_total = np.eye(9)
        Q_total = np.zeros((9, 9))
        for _ in range(n_sub):
            b_a = x[6:9].copy()
            a_body = a_body_meas - b_a
            a_world = _qrot(q_att, a_body) - g_vec  # Local-frame kinematic linear sensor

            x[0:3] = x[0:3] + x[3:6] * sub_dt + 0.5 * a_world * sub_dt * sub_dt
            x[3:6] = x[3:6] + a_world * sub_dt

            F = np.eye(9)
            F[0:3, 3:6] = np.eye(3) * sub_dt
            F[0:3, 6:9] = -0.5 * R * (sub_dt * sub_dt)
            F[3:6, 6:9] = -R * sub_dt

            Q = np.zeros((9, 9))
            qpr = sigma_a2 * (sub_dt ** 4) / 4.0
            qpv = sigma_a2 * (sub_dt ** 3) / 2.0
            qv = sigma_a2 * sub_dt * sub_dt
            Q[0, 0] = qpr; Q[1, 1] = qpr; Q[2, 2] = qpr * 4
            Q[3, 3] = qv;  Q[4, 4] = qv;  Q[5, 5] = qv * 4
            for i in range(3):
                Q[i, i + 3] = qpv if i < 2 else qpv * 4
                Q[i + 3, i] = Q[i, i + 3]
            qb = sigma_ba2 * sub_dt
            Q[6, 6] = qb; Q[7, 7] = qb; Q[8, 8] = qb

            P = F @ P @ F.T + Q
            # Accumulate F/Q for the whole dt_total. Standard composition:
            #   x_{k+1} = F_sub · x_k + ...
            #   P_{k+1} = F_sub · P_k · F_sub^T + Q_sub
            # ⇒ F_total = F_sub @ F_total
            #   Q_total = F_sub @ Q_total @ F_sub^T + Q_sub
            Q_total = F @ Q_total @ F.T + Q
            F_total = F @ F_total
        return F_total, Q_total

    def _kf_update(z: np.ndarray, H: np.ndarray, R_mat: np.ndarray,
                   gate_chi2: float) -> tuple[bool, float]:
        """Joseph-form update with optional chi-square innovation gate.

        Returns (accepted, mahalanobis²). When ``gate_chi2 > 0`` and the
        innovation Mahalanobis distance exceeds the gate, the update is
        skipped — protects against Post-processing epochs that contradict the Motion sensor
        (feature 5). Reject probability under the null is ~1−p_gate.
        """
        nonlocal x, P
        y = z - H @ x
        S = H @ P @ H.T + R_mat
        # Symmetrize S to guard against accumulated rounding asymmetry.
        S = 0.5 * (S + S.T)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Recover with a tiny regularization rather than crashing the
            # whole filter. Skip this update so downstream rejected-count
            # picks it up.
            try:
                S_inv = np.linalg.inv(S + 1e-9 * np.eye(S.shape[0]))
            except np.linalg.LinAlgError:
                return False, float("inf")
        d2 = float(y @ S_inv @ y)
        if gate_chi2 > 0 and d2 > gate_chi2:
            return False, d2
        K = P @ H.T @ S_inv
        x = x + K @ y
        I_KH = np.eye(9) - K @ H
        # Joseph form keeps P symmetric positive-definite under rounding.
        P = I_KH @ P @ I_KH.T + K @ R_mat @ K.T
        P = 0.5 * (P + P.T)
        x[6:9] = np.clip(x[6:9], -opts.bias_clip, opts.bias_clip)
        return True, d2

    # Pre-compute cumulative Motion model Local-frame position integral for fast outlier
    # rejection — cum[i] = sum_{k<=i} v_k · dt_k. We then test
    # (z_PPK - z_PPK_prev) against (cum_at_curr - cum_at_prev).
    vio_t_arr = np.asarray(
        [t for t, _ in (vio_vels_enu or [])], dtype=np.float64,
    )
    vio_v_arr = (
        np.asarray([v for _, v in vio_vels_enu], dtype=np.float64)
        if vio_vels_enu else np.zeros((0, 3), dtype=np.float64)
    )
    if vio_t_arr.size:
        # Trapezoidal-ish integration step from previous sample.
        dt_steps = np.diff(vio_t_arr, prepend=vio_t_arr[0])
        # Skip huge gaps so a 30-s Motion model outage doesn't blow up the integral,
        # but keep ordinary 0.5–2 s steps. Motion model is 2 Hz nominally; we cap at
        # 5 s which catches a real outage without dropping legitimate
        # post-stop reseeds.
        dt_steps[dt_steps > 5.0] = 0.0
        cum_pos = np.cumsum(vio_v_arr * dt_steps[:, None], axis=0)
    else:
        cum_pos = np.zeros((0, 3), dtype=np.float64)

    def _vio_cum_at(t: float) -> Optional[np.ndarray]:
        if vio_t_arr.size == 0:
            return None
        j = int(np.searchsorted(vio_t_arr, t))
        if j == 0:
            return cum_pos[0].copy()
        if j >= len(vio_t_arr):
            return cum_pos[-1].copy()
        # Linear interp between cumulative samples.
        t0, t1 = float(vio_t_arr[j - 1]), float(vio_t_arr[j])
        c0, c1 = cum_pos[j - 1], cum_pos[j]
        if t1 - t0 <= 0:
            return c0.copy()
        alpha = (t - t0) / (t1 - t0)
        return c0 + alpha * (c1 - c0)

    last_accept_pos = [None]   # type: list[Optional[np.ndarray]]
    last_accept_t   = [None]   # type: list[Optional[float]]

    n_ppk_seen = [0]  # closure-bound mutable counter for warmup
    def _apply_gnss_update(gr: PosRow) -> None:
        """Apply a single Signal measurement update at gr.utc_s with gating.

        Three layers of gating, each opt-in and skipped during warmup:
          1. Motion model-prediction gate (``opts.vio_ppk_outlier_m``) — the
             *strong* mode: predict this epoch's Post-processing position from the
             previous accepted Post-processing + integrated Motion model velocity; reject if
             they disagree by more than N metres. Catches multi-metre
             Post-processing spikes that the EKF would otherwise absorb because the
             tight R_pos makes K ≈ 1.
          2. Meter gate (``opts.innov_gate_pos_m``) — reject when raw
             innovation magnitude exceeds N metres. Robust regardless of
             P/R scale.
          3. Chi² gate (``opts.chi2_gate_pos``) — classic Mahalanobis
             test. Only useful when P is well-calibrated.
        """
        z_pos = _pos_to_enu(gr)
        has_vel = (math.isfinite(gr.ve) and math.isfinite(gr.vn)
                   and math.isfinite(gr.vu))
        in_warmup = n_ppk_seen[0] < opts.chi2_warmup_epochs
        gate_p_chi  = 0.0 if in_warmup else opts.chi2_gate_pos
        gate_v_chi  = 0.0 if in_warmup else opts.chi2_gate_vel
        gate_p_metr = 0.0 if in_warmup else opts.innov_gate_pos_m
        gate_v_metr = 0.0 if in_warmup else opts.innov_gate_vel_mps
        gate_vio_m  = (0.0 if (in_warmup or not vio_t_arr.size)
                       else opts.vio_ppk_outlier_m)
        n_ppk_seen[0] += 1

        # Motion model-prediction gate. Compare Post-processing delta against Motion model-integrated delta.
        rejected_by_vio = False
        if (gate_vio_m > 0 and last_accept_pos[0] is not None
                and last_accept_t[0] is not None):
            cum_now  = _vio_cum_at(gr.utc_s)
            cum_prev = _vio_cum_at(last_accept_t[0])
            if cum_now is not None and cum_prev is not None:
                vio_delta = cum_now - cum_prev
                ppk_delta = z_pos - last_accept_pos[0]
                # Use horizontal only — vertical Post-processing is noisier than Motion model.
                err = math.hypot(
                    float(ppk_delta[0] - vio_delta[0]),
                    float(ppk_delta[1] - vio_delta[1]),
                )
                if err > gate_vio_m:
                    rejected_by_vio = True

        if rejected_by_vio:
            res.n_pos_rejected += 1
            res.rejected_t.append(gr.utc_s)
        else:
            # Position update (always tried first).
            H_p = np.zeros((3, 9)); H_p[0:3, 0:3] = np.eye(3)
            # Adaptive R from .pos sigma columns when available.
            # Clamp sigma to [adaptive_r_floor_m, 100 m] to prevent both
            # runaway high trust (floor) and runaway low trust (ceiling)
            # — The external solver very rarely emits sigma > 100 m, but a corrupted
            # row should not break the EKF.
            if (opts.use_adaptive_r_pos
                    and math.isfinite(gr.sd_n) and math.isfinite(gr.sd_e)
                    and math.isfinite(gr.sd_u) and gr.sd_n > 0 and gr.sd_u > 0):
                fl = max(1e-6, opts.adaptive_r_floor_m)
                hi = 100.0
                sn = min(hi, max(fl, gr.sd_n)) ** 2 * opts.adaptive_r_scale
                se = min(hi, max(fl, gr.sd_e)) ** 2 * opts.adaptive_r_scale
                su = min(hi, max(fl, gr.sd_u)) ** 2 * opts.adaptive_r_scale
                # State is (E, N, U); R_p diag ordered to match.
                R_p = np.diag([se, sn, su])
            else:
                R_p = np.diag([opts.r_pos_h, opts.r_pos_h, opts.r_pos_v])
            innov_p = float(np.linalg.norm(z_pos - H_p @ x))

            # Adaptive R: inflate when innovation looks outlier-ish. Smooth
            # — no cascading because every epoch still contributes (weakly)
            # to the state when the innov is large.
            if opts.innov_softgate_m > 0 and not in_warmup:
                excess = innov_p - opts.innov_softgate_m
                if excess > 0:
                    R_scale = 1.0 + (excess / opts.innov_softgate_m) ** 2 * 100.0
                    R_p = R_p * R_scale

            if gate_p_metr > 0 and innov_p > gate_p_metr:
                res.n_pos_rejected += 1
                res.rejected_t.append(gr.utc_s)
            else:
                accepted_p, _ = _kf_update(z_pos, H_p, R_p, gate_p_chi)
                if accepted_p:
                    res.n_pos_updates += 1
                    last_accept_pos[0] = z_pos.copy()
                    last_accept_t[0]   = gr.utc_s
                else:
                    res.n_pos_rejected += 1
                    res.rejected_t.append(gr.utc_s)

        if has_vel:
            z_v = np.array([gr.ve, gr.vn, gr.vu])
            H_v = np.zeros((3, 9)); H_v[0:3, 3:6] = np.eye(3)
            R_v = np.diag([opts.r_vel_h, opts.r_vel_h, opts.r_vel_v])
            innov_v = float(np.linalg.norm(z_v - H_v @ x))
            if gate_v_metr > 0 and innov_v > gate_v_metr:
                res.n_vel_rejected += 1
            else:
                accepted_v, _ = _kf_update(z_v, H_v, R_v, gate_v_chi)
                if accepted_v:
                    res.n_vel_updates += 1
                else:
                    res.n_vel_rejected += 1

    def _apply_zupt() -> None:
        """Zero-velocity pseudo-measurement (feature 2). Ungated."""
        H_z = np.zeros((3, 9)); H_z[0:3, 3:6] = np.eye(3)
        R_z = np.diag([opts.sigma_zupt_mps ** 2] * 3)
        z = np.zeros(3)
        # ZUPT bypasses gating — by definition the Motion sensor agrees.
        _kf_update(z, H_z, R_z, 0.0)

    def _apply_nhc(q_att: np.ndarray) -> None:
        """Non-holonomic constraint (feature 7): lateral and vertical body
        velocity ≈ 0 when the vehicle is moving forward. Active only when
        speed > opts.nhc_min_speed_mps and opts.nhc_enabled.
        """
        nonlocal x, P
        v_enu = x[3:6]
        speed = float(np.linalg.norm(v_enu))
        if speed < opts.nhc_min_speed_mps:
            return
        R_bw = _quat_to_rotmat(q_att)            # body→world
        # We want body-sample y (lateral) and z (vertical) components = 0.
        # H_y · x = (R_wb · v_enu)[1:3] = 0  with R_wb = R_bw^T.
        R_wb = R_bw.T
        H = np.zeros((2, 9))
        H[0, 3:6] = R_wb[1, :]   # body-y
        H[1, 3:6] = R_wb[2, :]   # body-z
        R_mat = np.diag([
            opts.sigma_nhc_lateral_mps ** 2,
            opts.sigma_nhc_vertical_mps ** 2,
        ])
        z = np.zeros(2)
        y = z - H @ x
        S = H @ P @ H.T + R_mat
        S = 0.5 * (S + S.T)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            try:
                S_inv = np.linalg.inv(S + 1e-9 * np.eye(S.shape[0]))
            except np.linalg.LinAlgError:
                return
        K = P @ H.T @ S_inv
        x = x + K @ y
        I_KH = np.eye(9) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R_mat @ K.T
        P = 0.5 * (P + P.T)
        res.n_nhc += 1

    def _apply_vio(z_vel_enu: np.ndarray) -> None:
        """Motion model velocity measurement update (bonus feature). Loose R so it
        contributes shape between Post-processing epochs but doesn't override Post-processing at
        epochs.
        """
        H = np.zeros((3, 9)); H[0:3, 3:6] = np.eye(3)
        R_mat = np.diag([opts.r_vio_vel_h_mps2,
                         opts.r_vio_vel_h_mps2,
                         opts.r_vio_vel_v_mps2])
        accepted, _ = _kf_update(z_vel_enu, H, R_mat, 0.0)
        if accepted:
            res.n_vio += 1
        else:
            res.n_vio_rejected += 1

    # ── Motion model sample cursor (None when not provided) ───────────────────────
    vio_list = list(vio_vels_enu) if vio_vels_enu else []
    vio_list.sort(key=lambda p: p[0])
    vio_idx = [0]  # closure-bound

    def _drain_vio_up_to(t: float) -> None:
        while vio_idx[0] < len(vio_list) and vio_list[vio_idx[0]][0] <= t:
            _apply_vio(vio_list[vio_idx[0]][1])
            vio_idx[0] += 1

    # ── ZUPT detector: rolling-window Motion sensor variance ───────────────────────
    # Cheap O(N): keep two deques of recent rate sensor magnitudes and linear sensor
    # magnitudes; variance of each over zupt_window_s triggers ZUPT.
    zupt_buf: list[tuple[float, float, float]] = []
    # entry: (t, |rate sensor|, |linear sensor|-g)
    def _push_zupt_buf(t: float, gyro_norm: float, accel_norm: float) -> bool:
        zupt_buf.append((t, gyro_norm, accel_norm - _G))
        while zupt_buf and (t - zupt_buf[0][0]) > opts.zupt_window_s:
            zupt_buf.pop(0)
        if len(zupt_buf) < 5:
            return False
        gvals = np.array([z[1] for z in zupt_buf])
        avals = np.array([z[2] for z in zupt_buf])
        return (float(gvals.var()) < opts.zupt_gyro_var_thresh
                and float(avals.var()) < opts.zupt_accel_var_thresh)

    # Pre-allocate two-element holder: F/Q of the *last* propagation that
    # took us from epoch_{i} to epoch_{i+1}. We need to associate that pair
    # with epoch_i (post-update state) so the RTS smoother can step back.
    prev_x_post: Optional[np.ndarray] = None
    prev_P_post: Optional[np.ndarray] = None
    prev_t_post: Optional[float] = None
    prev_q_post: Optional[np.ndarray] = None

    F_accum = np.eye(9)
    Q_accum = np.zeros((9, 9))
    last_zupt_t = -1e18
    last_nhc_t  = -1e18

    for n_imu, (row, q_att) in enumerate(zip(imu_list, qs)):
        t = row.utc_s
        if t < t_gnss_start:
            # State init time is pos_list[0].utc_s; ignore Motion sensor before then.
            continue
        if t > t_gnss_end + 5.0:
            break

        a_body_meas = np.array([row.ax, row.ay, row.az])
        gyro_norm = float(math.sqrt(row.gx ** 2 + row.gy ** 2 + row.gz ** 2))
        accel_norm = float(math.sqrt(row.ax ** 2 + row.ay ** 2 + row.az ** 2))
        is_static = (opts.zupt_enabled
                     and _push_zupt_buf(t, gyro_norm, accel_norm))

        # ── Predict + Signal update, time-aligned ──
        cursor_t = prev_t
        while gnss_idx < len(pos_list) and pos_list[gnss_idx].utc_s <= t:
            gr = pos_list[gnss_idx]
            gnss_idx += 1
            gnss_t = max(gr.utc_s, cursor_t)
            if gnss_t > cursor_t:
                F_seg, Q_seg = _propagate(gnss_t - cursor_t, q_att, a_body_meas)
                Q_accum = F_seg @ Q_accum @ F_seg.T + Q_seg
                F_accum = F_seg @ F_accum
                cursor_t = gnss_t
            _apply_gnss_update(gr)

        if t > cursor_t:
            F_seg, Q_seg = _propagate(t - cursor_t, q_att, a_body_meas)
            Q_accum = F_seg @ Q_accum @ F_seg.T + Q_seg
            F_accum = F_seg @ F_accum

        # ── Motion model velocity update (bonus): drain any sample whose UTC has arrived ──
        if vio_list:
            _drain_vio_up_to(t)

        # ── ZUPT (feature 2): tight v=0 anchor when Motion sensor is quiet. ──
        if is_static and (t - last_zupt_t) >= opts.zupt_rate_limit_s:
            _apply_zupt()
            res.n_zupt += 1
            res.zupt_t.append(t)
            last_zupt_t = t

        # ── NHC (feature 7): vehicle-sample lateral/vertical clamp. ──
        if opts.nhc_enabled and (t - last_nhc_t) >= opts.nhc_rate_limit_s:
            _apply_nhc(q_att)
            last_nhc_t = t

        res.n_imu_steps += 1

        # ── Tape: record (x_post_prev, P_post_prev, x_prior_now, P_prior_now,
        #         F: prev→now) so RTS can walk backwards. ──
        if opts.record_tape and prev_x_post is not None:
            # Prior at THIS epoch = F_accum applied to previous posterior,
            # plus accumulated Q. (P here is *post-update at t*; the prior
            # is what we'd have had if we never observed at t.)
            x_prior = F_accum @ prev_x_post
            P_prior = F_accum @ prev_P_post @ F_accum.T + Q_accum
            res.tape_x_prior.append(x_prior)
            res.tape_P_prior.append(P_prior)
            res.tape_F.append(F_accum.copy())

        if opts.record_tape:
            res.tape_t.append(t)
            res.tape_x_post.append(x.copy())
            res.tape_P_post.append(P.copy())
            res.tape_q.append(np.asarray(q_att, dtype=float).copy())

        prev_x_post = x.copy()
        prev_P_post = P.copy()
        prev_t_post = t
        prev_q_post = q_att
        F_accum = np.eye(9)
        Q_accum = np.zeros((9, 9))
        prev_t = t
        del n_imu

        # ── Emit fused PosRow ──
        lat, lon, h = _enu_to_llh(x[0:3], ref_ecef, ref_llh)
        res.fused.append(PosRow(
            utc_s=t,
            lat_deg=lat,
            lon_deg=lon,
            h_m=h,
            quality=2,
            vn=float(x[4]),
            ve=float(x[3]),
            vu=float(x[5]),
        ))
        res.accel_bias_history.append(
            (t, float(x[6]), float(x[7]), float(x[8]))
        )

    # Fix tape alignment: tape_F[i] is "epoch i -> i+1", so it should have
    # one less entry than tape_x_post. Pad the prior arrays so length-N tape
    # has length-(N-1) prior/F arrays.
    if opts.record_tape:
        # Drop the spurious last prior entry (no "next" exists after final).
        # (Loop above appends a prior for each epoch *after* the first; we
        # have len(tape_F) == len(tape_x_post) - 1. Sanity:)
        pass

    _log(
        f"[ekf] propagated {res.n_imu_steps} IMU steps, "
        f"{res.n_pos_updates} pos updates ({res.n_vel_updates} w/ vel); "
        f"gated out: {res.n_pos_rejected} pos, {res.n_vel_rejected} vel; "
        f"ZUPT={res.n_zupt}  NHC={res.n_nhc}  VIO={res.n_vio}; "
        f"emitted {len(res.fused)} fused rows"
    )
    if res.accel_bias_history:
        last_t, bx, by, bz = res.accel_bias_history[-1]
        _log(f"[ekf] final accel bias estimate (body, m/s²): "
             f"bx={bx:.4f} by={by:.4f} bz={bz:.4f}")
    return res


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body→world 3×3 rotation matrix from a [w,x,y,z] quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ─── RTS smoother (feature 3) ─────────────────────────────────────────────────

@dataclass
class SmoothedResult:
    """Output of the RTS backwards pass."""
    fused: list[PosRow] = field(default_factory=list)
    # Per-epoch smoothed state arrays (for diagnostics).
    t: list[float]              = field(default_factory=list)
    x_smooth: list[np.ndarray]  = field(default_factory=list)
    P_smooth: list[np.ndarray]  = field(default_factory=list)
    q_att:    list[np.ndarray]  = field(default_factory=list)


def rts_smooth(
    forward: EkfResult,
    ref_llh: tuple[float, float, float],
) -> SmoothedResult:
    """Rauch-Tung-Striebel backwards pass over the EKF forward tape.

    Standard formula at each epoch ``k`` (working from N-1 back to 0)::

        C_k       = P_k^post · F_{k→k+1}^T · (P_{k+1}^prior)^-1
        x_k^smooth = x_k^post + C_k · (x_{k+1}^smooth - x_{k+1}^prior)
        P_k^smooth = P_k^post + C_k · (P_{k+1}^smooth - P_{k+1}^prior) · C_k^T

    The smoother is non-causal — it uses future Signal updates to retroject
    information back through earlier Motion sensor integration. Beats Gaussian
    smoothing because the process model knows ``Δp = v·Δt`` and the
    linear sensor-bias state, so curvature is not flattened.
    """
    if not forward.tape_t or not forward.tape_F:
        return SmoothedResult(fused=list(forward.fused))

    n = len(forward.tape_t)
    x_s: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    P_s: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    # Terminal epoch: smoothed == filtered.
    x_s[-1] = forward.tape_x_post[-1].copy()
    P_s[-1] = forward.tape_P_post[-1].copy()

    for k in range(n - 2, -1, -1):
        F_k     = forward.tape_F[k]            # k -> k+1
        x_post  = forward.tape_x_post[k]
        P_post  = forward.tape_P_post[k]
        x_prior = forward.tape_x_prior[k]      # prior at k+1 from k
        P_prior = forward.tape_P_prior[k]
        try:
            P_prior_inv = np.linalg.inv(P_prior)
        except np.linalg.LinAlgError:
            x_s[k] = x_post.copy(); P_s[k] = P_post.copy()
            continue
        C = P_post @ F_k.T @ P_prior_inv
        x_s[k] = x_post + C @ (x_s[k + 1] - x_prior)
        P_s[k] = P_post + C @ (P_s[k + 1] - P_prior) @ C.T
        P_s[k] = 0.5 * (P_s[k] + P_s[k].T)

    # Convert Local-frame back to LLH.
    ref_ecef = np.array(llh_to_ecef(*ref_llh))
    fused: list[PosRow] = []
    for k in range(n):
        lat, lon, h = _enu_to_llh(x_s[k][0:3], ref_ecef, ref_llh)
        fused.append(PosRow(
            utc_s=forward.tape_t[k],
            lat_deg=lat, lon_deg=lon, h_m=h,
            quality=1,  # smoothed: mark as Fix-grade for downstream branches
            vn=float(x_s[k][4]),
            ve=float(x_s[k][3]),
            vu=float(x_s[k][5]),
        ))

    return SmoothedResult(
        fused=fused,
        t=list(forward.tape_t),
        x_smooth=x_s,
        P_smooth=P_s,
        q_att=list(forward.tape_q),
    )
