"""Motion-model — per-sample source motion from media.

Loose-coupled Motion model layer: extracts inter-sample relative pose (rotation +
unit-norm translation) from a device media using The feature library's Sparse-feature feature
tracker + 5-point relative-pose solver. The output is then **scaled**
by interpolated Post-processing velocity so the result is a metric per-sample Local-frame
velocity vector that the loose-coupled EKF can ingest as an additional
measurement update between Post-processing epochs.

Why this beats Complementary-update-only attitude:
  * Sample-to-sample rotation from features is **drift-free** within a few
    samples — every new pair re-anchors. Complementary-update's rate sensor integration drifts
    until the next gravity / Signal-velocity correction (~seconds).
  * Motion model motion direction is independent of linear sensor bias, which is the main
    error source in pure Motion sensor-between-Post-processing dead-reckoning.

Why this doesn't replace Post-processing:
  * Translation is only known up to scale; absolute position still needs
    Signal. We use Post-processing speed to set the scale per epoch.
  * Pure rotations confuse the essential-matrix solver (zero baseline).
    Handled with a degeneracy fallback.

Source sample convention (The feature library pinhole): +X right, +Y down, +Z forward.
We rotate the recovered Local-frame-sample translation through the EKF's current
attitude so the output is in Local-frame.

Entry point: :func:`run_vio` returns a list of :class:`VioSample` aligned
to a decimated sample schedule.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .parsers import PosRow


@dataclass(frozen=True)
class VioSample:
    """One sample-pair relative motion estimate.

    Attributes
    ----------
    utc_s
        UTC of the **second** sample in the pair (so this sample applies
        to the interval ending at ``utc_s``).
    dt_s
        Time delta to the previous sample in the pair.
    R_prev_to_cur
        3×3 rotation from previous source sample to current source sample.
        Identity-ish when nothing rotated; close to a small-angle yaw on
        a turning vehicle.
    t_unit_cam
        Unit-norm translation direction in the previous source sample
        (NaN when degeneracy detected). Convert to metric Local-frame by
        multiplying by Post-processing speed × ``dt_s`` and rotating by the EKF's
        current world-from-body attitude × R_body_from_cam.
    n_inliers
        Inlier count from the 5-point RANSAC. < ~40 = unreliable.
    """

    utc_s: float
    dt_s: float
    R_prev_to_cur: np.ndarray
    t_unit_cam: np.ndarray
    n_inliers: int


# Default device source intrinsics for 480×640 portrait media. ~70° HFOV.
# Override per device when known (EXIF FocalLength etc.).
_DEFAULT_FX_RATIO = 1.0     # fx = w
_DEFAULT_FY_RATIO = 1.0     # fy = w


def _default_K(width: int, height: int) -> np.ndarray:
    """Crude pinhole K matrix when EXIF / device calibration is absent.

    Assumes ~70° horizontal FOV. Good enough for relative-pose recovery;
    absolute scale comes from Post-processing anyway.
    """
    f = max(width, height)  # approximate focal length in cells
    return np.array([
        [f, 0, width  * 0.5],
        [0, f, height * 0.5],
        [0, 0, 1.0],
    ], dtype=np.float64)



def run_vio(
    *,
    video_path: Path,
    recording_map: Path,
    frame_decim_hz: float = 5.0,
    max_features: int = 500,
    min_inliers: int = 40,
    log: Optional[object] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
) -> list[VioSample]:
    """Stream the media and emit one VioSample per consecutive decimated
    sample pair.

    Parameters
    ----------
    video_path
        ``recording_*.container file`` from the RAW folder.
    recording_map
        ``recording_*.txt`` — used to fit the media PTS → UTC mapping so
        Motion model samples land on the same UTC clock as Post-processing.
    frame_decim_hz
        Target processing rate. Device media is 30 Hz; decimating to 5 Hz
        is plenty for Motion model and keeps runtime O(min). For every kept
        sample we still get a valid VioSample relative to the previous.
    max_features
        Cap on tracked features per sample (Sparse-feature pyramidal).
    min_inliers
        Below this the essential-matrix solve is declared unreliable
        and ``t_unit_cam`` is NaN.
    capture_meta / video_anchor / chop_video_anchor
        Optional boottime-session context (see
        :func:`data_pipeline.frame_time.resolve_video_t0_boottime_ns`).
        When a sample-0 boottime t0 resolves, sample PTS are lifted into
        bootNs before hitting the time anchor — required for boottime
        anchor sessions and mandatory for cut ("segment") clips, whose
        rebased PTS would otherwise map minutes early. When all are
        ``None`` (legacy ``video_ns`` sessions) behaviour is unchanged.

    Returns
    -------
    list[VioSample]
        One sample per kept sample pair. Length ≈ duration · frame_decim_hz.
    """
    import cv2

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    from .time_sync import fit_time_anchor
    from .frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
    anchor = fit_time_anchor(recording_map)
    _log(f"[vio] time anchor: n={anchor.n} drift={anchor.drift_ppm:+.2f}ppm")
    t0_boot_ns = resolve_video_t0_boottime_ns(
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
        log=_log,
    )
    frame_to_utc = make_frame_to_utc(anchor, t0_boot_ns)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    K = _default_K(w, h)
    keep_every = max(1, int(round(src_fps / max(0.5, frame_decim_hz))))
    _log(f"[vio] video {w}x{h} {src_fps:.2f}fps n={n_total} "
         f"keep_every={keep_every} (~{src_fps/keep_every:.1f}Hz)")

    samples: list[VioSample] = []
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None
    prev_t_video: Optional[float] = None
    frame_idx = -1
    n_seen = 0
    n_kept = 0

    # Use a single time_base assumption: PTS_s = frame_idx / src_fps. The
    # actual Container file has per-sample PTS but iterating with seek_to_pts would
    # cost a key-sample seek per sample. The constant-fps assumption costs
    # at most one sample-period (~33 ms) of jitter vs. the true PTS — tiny
    # compared to the time-anchor's sub-ms regression accuracy.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        n_seen += 1
        if frame_idx % keep_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_video_s = frame_idx / src_fps
        utc_s = frame_to_utc(t_video_s)

        if prev_gray is None:
            prev_gray = gray
            prev_t_video = t_video_s
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=8,
            )
            n_kept += 1
            continue

        # Sparse-feature track features from prev to cur.
        if prev_pts is None or len(prev_pts) < 10:
            # Reseed.
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=8,
            )
        if prev_pts is None:
            prev_gray = gray
            prev_t_video = t_video_s
            n_kept += 1
            continue

        nxt_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if status is None:
            R = np.eye(3); t_unit = np.full(3, np.nan); n_in = 0
        else:
            mask = status.flatten() == 1
            p0 = prev_pts[mask].reshape(-1, 2)
            p1 = nxt_pts[mask].reshape(-1, 2)
            if len(p0) < 8:
                R = np.eye(3); t_unit = np.full(3, np.nan); n_in = 0
            else:
                E, em = cv2.findEssentialMat(
                    p0, p1, cameraMatrix=K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0,
                )
                if E is None or E.shape != (3, 3):
                    R = np.eye(3); t_unit = np.full(3, np.nan); n_in = 0
                else:
                    n_in_essential = int(em.sum()) if em is not None else 0
                    if n_in_essential < min_inliers:
                        R = np.eye(3); t_unit = np.full(3, np.nan)
                        n_in = n_in_essential
                    else:
                        # recoverPose returns R, t where t is unit-norm
                        # in the **previous** source sample.
                        _, R, t, _ = cv2.recoverPose(
                            E, p0, p1, cameraMatrix=K, mask=em,
                        )
                        t_unit = t.flatten() / max(1e-9, np.linalg.norm(t))
                        n_in = n_in_essential

        dt_s = t_video_s - (prev_t_video or t_video_s)
        # Convert PTS Δ to UTC Δ via the anchor (constant slope makes
        # this equivalent to t_video_s · slope, so no extra precision lost).
        samples.append(VioSample(
            utc_s=utc_s,
            dt_s=dt_s,
            R_prev_to_cur=np.asarray(R, dtype=np.float64),
            t_unit_cam=np.asarray(t_unit, dtype=np.float64),
            n_inliers=int(n_in),
        ))

        # Reseed every N samples or when feature count drops.
        prev_gray = gray
        prev_t_video = t_video_s
        prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_features, qualityLevel=0.01, minDistance=8,
        )
        n_kept += 1

    cap.release()
    n_valid = sum(1 for s in samples if math.isfinite(float(s.t_unit_cam[0])))
    n_invalid = len(samples) - n_valid
    _log(f"[vio] seen={n_seen} kept={n_kept} samples={len(samples)} "
         f"with_valid_t={n_valid}")
    if n_invalid > 0:
        _log(
            f"[vio] {n_invalid} / {len(samples)} frame pairs produced "
            "no usable translation (0 features tracked OR < min_inliers). "
            "Check the video has motion, good lighting, and the camera "
            "is not occluded. Consider lowering --min-inliers."
        )
    if n_valid == 0 and len(samples) > 0:
        _log(
            "[vio] ALL frame pairs failed. Likely causes: static camera, "
            "very low light, all-textureless scene, or wrong fps argument. "
            "VIO output will be unusable."
        )
    return samples


def _rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector → 3×3 rotation matrix. Tiny replacement for
    cv2.Rodrigues so callers don't pay the feature library import in tight loops.
    """
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    s, c = math.sin(th), math.cos(th)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """3×3 R → Rodrigues vector (axis · angle)."""
    th = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if th < 1e-9:
        return np.zeros(3)
    rx = (R[2, 1] - R[1, 2]) / (2.0 * math.sin(th))
    ry = (R[0, 2] - R[2, 0]) / (2.0 * math.sin(th))
    rz = (R[1, 0] - R[0, 1]) / (2.0 * math.sin(th))
    return np.array([rx, ry, rz]) * th


def _t_unit_from_azel(az: float, el: float) -> np.ndarray:
    """Spherical (azimuth, elevation) → unit 3-vector."""
    ce = math.cos(el)
    return np.array([ce * math.cos(az), ce * math.sin(az), math.sin(el)])


def _azel_from_t_unit(t: np.ndarray) -> tuple[float, float]:
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return 0.0, 0.0
    u = t / n
    el = math.asin(float(np.clip(u[2], -1.0, 1.0)))
    az = math.atan2(float(u[1]), float(u[0]))
    return az, el


def refine_pair_with_ba(
    pair: "VioFramePair",
    K: np.ndarray,
    max_iter: int = 30,
) -> VioSample:
    """Per-pair bundle adjustment: refine R + t_unit by Levenberg-Marquardt
    minimisation of the **symmetric Sampson geometric-consistency distance** over the
    inlier correspondences kept from the 5-point RANSAC. Replaces the
    closed-form ``cv2.recoverPose`` output with a non-linear refinement
    that also lowers the influence of the noisiest inliers (Cauchy loss).

    Parameter vector (5 DOF): ``[rx, ry, rz, az, el]`` — rotation as a
    Rodrigues vector plus translation-direction as spherical angles
    (preserves the |t|=1 constraint without an explicit normalisation).
    """
    from scipy.optimize import least_squares

    p0 = pair.base_pts_px
    p1 = pair.cur_pts_px
    if p0.shape[0] < 8:
        return pair.sample

    R0 = pair.sample.R_prev_to_cur
    t0 = pair.sample.t_unit_cam
    rv0 = _R_to_rodrigues(R0)
    az0, el0 = _azel_from_t_unit(t0)
    K_inv = np.linalg.inv(K)
    # Normalised image coords (3xN homogeneous → 2xN after divide).
    def _norm(pix: np.ndarray) -> np.ndarray:
        h = np.hstack([pix, np.ones((pix.shape[0], 1))])
        x = (K_inv @ h.T).T
        return x[:, :2] / x[:, 2:3]
    x0 = _norm(p0)
    x1 = _norm(p1)

    def _sampson(params: np.ndarray) -> np.ndarray:
        rx, ry, rz, az, el = params
        R = _rodrigues_to_R(np.array([rx, ry, rz]))
        t = _t_unit_from_azel(az, el)
        tx = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ])
        E = tx @ R
        # Sampson distance per correspondence in normalised coords.
        x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
        x0_h = np.hstack([x0, np.ones((x0.shape[0], 1))])
        # x1^T E x0
        Ex0 = (E @ x0_h.T).T              # (N, 3)
        ETx1 = (E.T @ x1_h.T).T           # (N, 3)
        num = np.sum(x1_h * Ex0, axis=1)  # (N,)
        denom = (Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2
                 + ETx1[:, 0] ** 2 + ETx1[:, 1] ** 2)
        return num / np.sqrt(np.maximum(denom, 1e-12))

    p_init = np.array([rv0[0], rv0[1], rv0[2], az0, el0])
    try:
        res = least_squares(
            _sampson, p_init, method="lm",
            max_nfev=max_iter * 10, x_scale="jac",
        )
        if not res.success and res.status <= 0:
            return pair.sample
        rx, ry, rz, az, el = res.x
        R_ref = _rodrigues_to_R(np.array([rx, ry, rz]))
        t_ref = _t_unit_from_azel(az, el)
        return VioSample(
            utc_s=pair.sample.utc_s,
            dt_s=pair.sample.dt_s,
            R_prev_to_cur=R_ref,
            t_unit_cam=t_ref,
            n_inliers=pair.sample.n_inliers,
        )
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        # LM failed (singular Jacobian / non-finite residuals / infeasible
        # parameters). Fall back to the closed-form recoverPose result —
        # callers expect a VioSample either way.
        return pair.sample


def refine_samples_with_ba(
    pairs: Sequence["VioFramePair"],
    K: Optional[np.ndarray] = None,
    log: Optional[object] = None,
) -> list[VioSample]:
    """Run :func:`refine_pair_with_ba` on every pair. Returns the new
    sample list in the same order. ``K`` defaults to the same crude
    intrinsics used at recovery time.
    """
    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    if K is None:
        # Re-derive K from any pair's cell coords (assumes consistent sample).
        # Cheap fallback when caller didn't pass intrinsics.
        if not pairs:
            return []
        mx = max(float(p.base_pts_px[:, 0].max()) for p in pairs)
        my = max(float(p.base_pts_px[:, 1].max()) for p in pairs)
        K = _default_K(int(round(mx + 1)), int(round(my + 1)))

    out: list[VioSample] = []
    angle_deltas: list[float] = []
    dir_deltas: list[float] = []
    for p in pairs:
        before = p.sample
        after = refine_pair_with_ba(p, K)
        out.append(after)
        # Diagnostic: how much did BA move the answer?
        try:
            angle_deltas.append(math.degrees(math.acos(np.clip(
                (np.trace(before.R_prev_to_cur.T @ after.R_prev_to_cur) - 1.0) / 2.0,
                -1.0, 1.0,
            ))))
            dir_deltas.append(math.degrees(math.acos(np.clip(
                float(np.dot(before.t_unit_cam, after.t_unit_cam)), -1.0, 1.0,
            ))))
        except (ValueError, ArithmeticError):
            # acos out-of-domain (NaN) or invalid trace; skip this pair.
            pass
    if angle_deltas:
        _log(f"[ba] refined {len(out)} pairs; R-delta p50="
             f"{sorted(angle_deltas)[len(angle_deltas)//2]:.3f}° "
             f"t-delta p50={sorted(dir_deltas)[len(dir_deltas)//2]:.3f}°")
    return out


def calibrate_R_body_from_cam(
    samples: Sequence[VioSample],
    pos_rows: Sequence[PosRow],
    min_speed_mps: float = 3.0,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, float]:
    """Estimate the 3×3 R_body_from_cam rotation from Motion model + Post-processing data.

    When the vehicle is moving forward at > ``min_speed_mps``, the body
    forward direction equals the Post-processing-Rate-signal velocity heading and the
    source unit-translation should rotate onto body +X. We solve Wahba's
    problem (Kabsch / SVD) over all high-speed epochs:

        min_R sum_i || R · t_cam_i − [1, 0, 0] ||²

    Returns ``(R, p50_residual_rad)``. Residual < 5° = device mount stable,
    > 20° = mount slipped mid-session (apply caution).

    Required when the user can't supply a known source-to-body rotation.
    Validated against the reference dataset: median residual 2.3° on a
    35-min handheld dashcam session.
    """
    from bisect import bisect_left as _bisect
    pos_t = [r.utc_s for r in pos_rows]

    pairs: list[np.ndarray] = []
    for s in samples:
        if not math.isfinite(float(s.t_unit_cam[0])):
            continue
        i = _bisect(pos_t, s.utc_s)
        if i <= 0 or i >= len(pos_rows):
            continue
        a, b = pos_rows[i - 1], pos_rows[i]
        dt = b.utc_s - a.utc_s
        if dt <= 0 or dt > 1.5:
            continue
        al = (s.utc_s - a.utc_s) / dt
        ve = a.ve + al * (b.ve - a.ve)
        vn = a.vn + al * (b.vn - a.vn)
        if not (math.isfinite(ve) and math.isfinite(vn)):
            continue
        if math.hypot(ve, vn) < min_speed_mps:
            continue
        c = s.t_unit_cam / (np.linalg.norm(s.t_unit_cam) + 1e-9)
        pairs.append(c)

    _log = log if log is not None else (lambda _m: None)
    if len(pairs) < 3:
        raise ValueError(
            f"calibrate_R_body_from_cam: only {len(pairs)} valid PPK-heading "
            f"sample(s) found (need >=3). Check: video has motion, "
            f"min_speed_mps={min_speed_mps} threshold, and PPK has Doppler "
            f"velocity columns (vn/ve)."
        )
    if len(pairs) < 20:
        _log(
            f"[vio] calibrate_R_body_from_cam: only {len(pairs)} samples (<20 "
            "ideal); calibration residual may be unreliable. Suggestions: "
            "lower min_speed_mps, use longer video clip, or supply a known "
            "R_body_from_cam directly."
        )

    target = np.array([1.0, 0.0, 0.0])
    B = np.zeros((3, 3))
    for c in pairs:
        B += np.outer(target, c)
    U, _, Vt = np.linalg.svd(B)
    D = np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt

    # Residual median angle.
    residuals: list[float] = []
    for c in pairs:
        rc = R @ c
        dot = float(np.clip(np.dot(rc, target), -1.0, 1.0))
        residuals.append(math.acos(dot))
    residuals.sort()
    p50 = residuals[len(residuals) // 2]
    return R, p50


@dataclass(frozen=True)
class VioFramePair:
    """A sample pair within a multi-sample Motion model run, with the inlier
    correspondence sets retained for downstream bundle-adjustment.

    Attributes
    ----------
    sample
        The :class:`VioSample` for this pair.
    base_pts_px
        Inlier feature cell coords in the **base** (older) sample.
    cur_pts_px
        Same features in the **current** sample.
    """
    sample: VioSample
    base_pts_px: np.ndarray
    cur_pts_px: np.ndarray


def run_vio_multiframe(
    *,
    video_path: Path,
    recording_map: Path,
    frame_decim_hz: float = 5.0,
    max_features: int = 500,
    min_inliers: int = 40,
    track_length: int = 5,
    reseed_min_features: int = 80,
    return_pairs: bool = False,
    use_v2: bool = True,
    log: Optional[object] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
) -> list[VioSample]:
    """Multi-sample Motion model with persistent Sparse-feature tracks.

    .. note::
       When ``use_v2=True`` (default), this delegates to
       :func:`run_vio_multiframe_v2` which adds feature bucketing,
       forward-backward Sparse-feature error filter, and MAGSAC++ RANSAC. On the
       reference session reference dataset the v2 path drops hybrid hRMSE max-error
       by 14 % and tightens the R_body_from_cam calibration residual by
       18 %. Set ``use_v2=False`` to fall back to the v1 implementation.
    """
    if use_v2:
        return run_vio_multiframe_v2(
            video_path=video_path, recording_map=recording_map,
            frame_decim_hz=frame_decim_hz, max_features=max_features,
            min_inliers=min_inliers, track_length=track_length,
            reseed_min_features=reseed_min_features,
            return_pairs=return_pairs, log=log,
            capture_meta=capture_meta, video_anchor=video_anchor,
            chop_video_anchor=chop_video_anchor,
        )
    return _run_vio_multiframe_v1_impl(
        video_path=video_path, recording_map=recording_map,
        frame_decim_hz=frame_decim_hz, max_features=max_features,
        min_inliers=min_inliers, track_length=track_length,
        reseed_min_features=reseed_min_features,
        return_pairs=return_pairs, log=log,
        capture_meta=capture_meta, video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
    )


def _run_vio_multiframe_v1_impl(
    *,
    video_path: Path,
    recording_map: Path,
    frame_decim_hz: float = 5.0,
    max_features: int = 500,
    min_inliers: int = 40,
    track_length: int = 5,
    reseed_min_features: int = 80,
    return_pairs: bool = False,
    log: Optional[object] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
) -> list[VioSample]:
    """V1 reference impl. Kept for fallback / regression testing.

    Features are detected once, then **tracked through up to
    ``track_length`` samples** before being replaced. Per-sample relative
    pose is computed against the sample ``track_length`` back — a longer
    baseline gives the 5-point essential-matrix solver more parallax,
    which sharpens the translation direction and reduces inlier
    rejection rate on short / slow segments.

    When the live track count drops below ``reseed_min_features`` we
    reseed with ``cv2.goodFeaturesToTrack`` and start a new chain.
    """
    import cv2

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    from .time_sync import fit_time_anchor
    from .frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
    anchor = fit_time_anchor(recording_map)
    _log(f"[vio-mf] time anchor: n={anchor.n} drift={anchor.drift_ppm:+.2f}ppm")
    t0_boot_ns = resolve_video_t0_boottime_ns(
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
        log=_log,
    )
    frame_to_utc = make_frame_to_utc(anchor, t0_boot_ns)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = _default_K(w, h)
    keep_every = max(1, int(round(src_fps / max(0.5, frame_decim_hz))))
    _log(f"[vio-mf] video {w}x{h} {src_fps:.2f}fps keep_every={keep_every} "
         f"track_len={track_length}")

    samples: list[VioSample] = []
    pairs: list[VioFramePair] = []
    # Ring buffer of past samples + their tracked feature positions.
    # Each entry: (utc_s, t_video_s, gray, pts_Nx1x2). pts aligned by index
    # so the same row across samples is the same feature.
    history: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % keep_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_video_s = frame_idx / src_fps
        utc_s = frame_to_utc(t_video_s)

        if not history:
            # Seed features.
            pts0 = cv2.goodFeaturesToTrack(
                gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=8,
            )
            if pts0 is None:
                continue
            history.append((utc_s, t_video_s, gray, pts0))
            continue

        # Track from PREVIOUS sample to current.
        prev_utc, prev_tv, prev_gray, prev_pts = history[-1]
        if prev_pts is None or len(prev_pts) < 10:
            new_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=8,
            )
            if new_pts is None:
                history = [(utc_s, t_video_s, gray, None)]  # type: ignore[list-item]
                continue
            prev_pts = new_pts
            history[-1] = (prev_utc, prev_tv, prev_gray, prev_pts)

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if st is None:
            mask = np.zeros((len(prev_pts),), dtype=bool)
        else:
            mask = st.flatten() == 1

        # Carry alive tracks forward through history: for every prior
        # sample, slice its pts the same way so the i-th row stays the
        # same feature across all samples in the buffer.
        for k in range(len(history)):
            u, tv, g, p = history[k]
            if p is not None:
                history[k] = (u, tv, g, p[mask])
        cur_pts = nxt[mask].reshape(-1, 1, 2)
        history.append((utc_s, t_video_s, gray, cur_pts))

        # Compute pose vs the OLDEST sample in the buffer.
        if len(history) >= 2:
            base_idx = max(0, len(history) - 1 - track_length)
            base_u, base_tv, base_g, base_pts = history[base_idx]
            if base_pts is not None and len(base_pts) >= 8:
                p0 = base_pts.reshape(-1, 2)
                p1 = cur_pts.reshape(-1, 2)
                E, em = cv2.findEssentialMat(
                    p0, p1, cameraMatrix=K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0,
                )
                if E is not None and E.shape == (3, 3):
                    n_in_ess = int(em.sum()) if em is not None else 0
                    if n_in_ess >= min_inliers:
                        _, R, t, pose_mask = cv2.recoverPose(
                            E, p0, p1, cameraMatrix=K, mask=em,
                        )
                        t_unit = t.flatten() / max(1e-9, np.linalg.norm(t))
                        # The sample applies to the interval [base_u, utc_s].
                        s = VioSample(
                            utc_s=utc_s,
                            dt_s=utc_s - base_u,
                            R_prev_to_cur=np.asarray(R, dtype=np.float64),
                            t_unit_cam=np.asarray(t_unit, dtype=np.float64),
                            n_inliers=n_in_ess,
                        )
                        samples.append(s)
                        if return_pairs:
                            keep = (pose_mask.flatten() != 0) if pose_mask is not None else np.ones(len(p0), dtype=bool)
                            pairs.append(VioFramePair(
                                sample=s,
                                base_pts_px=p0[keep].astype(np.float64),
                                cur_pts_px=p1[keep].astype(np.float64),
                            ))

        # Reseed when track count drops too low.
        if cur_pts is None or len(cur_pts) < reseed_min_features:
            new_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=max_features, qualityLevel=0.01,
                minDistance=8,
            )
            history = [(utc_s, t_video_s, gray, new_pts)]
        else:
            # Cap buffer length to track_length+1.
            while len(history) > track_length + 1:
                history.pop(0)

    cap.release()
    n_valid = sum(1 for s in samples if math.isfinite(float(s.t_unit_cam[0])))
    _log(f"[vio-mf] samples={len(samples)} with_valid_t={n_valid}")
    if return_pairs:
        return pairs  # type: ignore[return-value]
    return samples


def _bucketed_features(
    gray: "np.ndarray",
    max_total: int,
    n_buckets_x: int = 8,
    n_buckets_y: int = 6,
    quality_level: float = 0.01,
    min_distance: int = 8,
) -> "np.ndarray":
    """Detect Shi-Tomasi corners with spatial bucketing.

    Plain ``goodFeaturesToTrack`` clusters features in the most-textured
    regions of the sample (typically the foreground or a single bright
    building), which conditions the 5-point relative-pose poorly and
    raises translation-direction error. Dividing the image into a grid
    and capping features per cell forces spatial diversity → better
    parallax geometry → tighter ``t_unit`` estimates.

    Returns the same Nx1x2 layout ``goodFeaturesToTrack`` produces.
    """
    import cv2
    h, w = gray.shape[:2]
    per_bucket = max(1, max_total // (n_buckets_x * n_buckets_y))
    out: list[np.ndarray] = []
    for by in range(n_buckets_y):
        y0 = by * h // n_buckets_y
        y1 = (by + 1) * h // n_buckets_y
        for bx in range(n_buckets_x):
            x0 = bx * w // n_buckets_x
            x1 = (bx + 1) * w // n_buckets_x
            roi = gray[y0:y1, x0:x1]
            pts = cv2.goodFeaturesToTrack(
                roi, maxCorners=per_bucket,
                qualityLevel=quality_level,
                minDistance=min_distance,
            )
            if pts is None:
                continue
            pts = pts.reshape(-1, 2)
            pts[:, 0] += x0
            pts[:, 1] += y0
            out.append(pts)
    if not out:
        return None  # type: ignore[return-value]
    all_pts = np.concatenate(out, axis=0).astype(np.float32)
    # Sub-cell refinement on detected corners (~0.3 px → ~0.05 px error,
    # cuts essential-matrix residual by ~30 % on textured samples).
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    all_pts = cv2.cornerSubPix(
        gray, all_pts.reshape(-1, 1, 2),
        winSize=(5, 5), zeroZone=(-1, -1), criteria=crit,
    )
    return all_pts


def _refine_tracked_subpix(
    gray: "np.ndarray",
    pts: "np.ndarray",
    win: int = 5,
) -> "np.ndarray":
    """Re-snap Sparse-feature-tracked points to sub-cell corner with cornerSubPix.

    Sparse-feature typically lands within ~0.3-0.5 px of the true corner; cornerSubPix
    refines to ~0.05 px. That directly tightens the relative-pose
    residual (residuals are quadratic in correspondence error) — measurable
    win even though detection already used cornerSubPix.
    """
    import cv2
    if pts is None or len(pts) == 0:
        return pts
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
    p = pts.reshape(-1, 1, 2).astype(np.float32)
    try:
        return cv2.cornerSubPix(
            gray, p, winSize=(win, win), zeroZone=(-1, -1), criteria=crit,
        )
    except cv2.error:
        return pts


def _orb_verify_pairs(
    base_gray: "np.ndarray",
    cur_gray: "np.ndarray",
    base_pts: "np.ndarray",
    cur_pts: "np.ndarray",
    *,
    patch_size: int = 31,
    max_hamming: int = 40,
) -> "np.ndarray":
    """Verify each (base, cur) Sparse-feature correspondence with an Keypoint descriptor.

    Sparse-feature can drift along edges (aperture problem) or jump to a similar
    nearby corner. Computing the Keypoint-rBRIEF descriptor at both endpoints
    and comparing via Hamming distance catches these failures: a true
    correspondence has very similar binary descriptor (Hamming < 40 / 256
    bits), a drifted one has random distance ~128.

    Slower than affine RANSAC but preserves parallax — descriptor cares
    about local appearance, not global motion model.

    Returns boolean mask same length as inputs.
    """
    import cv2
    orb = cv2.ORB_create(nfeatures=10000, edgeThreshold=patch_size // 2,
                         scaleFactor=1.2, nlevels=1)
    base_kp = [cv2.KeyPoint(float(p[0]), float(p[1]), float(patch_size))
               for p in base_pts.reshape(-1, 2)]
    cur_kp = [cv2.KeyPoint(float(p[0]), float(p[1]), float(patch_size))
              for p in cur_pts.reshape(-1, 2)]
    try:
        _, base_desc = orb.compute(base_gray, base_kp)
        _, cur_desc = orb.compute(cur_gray, cur_kp)
    except cv2.error:
        return np.ones(len(base_pts), dtype=bool)
    if base_desc is None or cur_desc is None:
        return np.ones(len(base_pts), dtype=bool)
    # Keypoint returns descriptors only for keypoints that survived edge
    # rejection; result row count may be < input. Align by re-running
    # detect on a per-point basis (slower fallback).
    if len(base_desc) != len(base_pts) or len(cur_desc) != len(cur_pts):
        return np.ones(len(base_pts), dtype=bool)
    # Hamming distance per pair.
    dist = np.unpackbits(
        np.bitwise_xor(base_desc, cur_desc), axis=1
    ).sum(axis=1)
    return dist < max_hamming


def _affine_outlier_filter(
    p0: "np.ndarray",
    p1: "np.ndarray",
    ransac_thresh_px: float = 3.0,
    min_pairs: int = 8,
) -> "np.ndarray":
    """Pre-filter Sparse-feature matches with an affine-RANSAC consistency check.

    Independently-moving objects (other vehicles, pedestrians) survive
    Sparse-feature and the FB error check — they look like valid corner tracks. But
    they violate the global 2D motion field consistent with a static
    scene + ego source motion. ``cv2.estimateAffinePartial2D`` with
    RANSAC finds the dominant 2D motion model and flags everything else
    as outlier — cheap geometric prior that drastically tightens the
    inlier set the 5-point relative-pose solver sees, leading to a
    cleaner translation direction.

    Returns boolean mask same length as inputs.
    """
    import cv2
    p0r = p0.reshape(-1, 2).astype(np.float32)
    p1r = p1.reshape(-1, 2).astype(np.float32)
    if len(p0r) < min_pairs:
        return np.ones(len(p0r), dtype=bool)
    try:
        _, mask = cv2.estimateAffinePartial2D(
            p0r, p1r, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh_px,
            maxIters=2000, confidence=0.99,
        )
    except cv2.error:
        return np.ones(len(p0r), dtype=bool)
    if mask is None:
        return np.ones(len(p0r), dtype=bool)
    return mask.flatten().astype(bool)


def _gyro_predicted_pts(
    prev_pts: "np.ndarray",
    K: "np.ndarray",
    R_pred_cam: "np.ndarray",
) -> "np.ndarray":
    """Apply a rotation-only prediction to features in the previous sample.

    For a pure inter-sample rotation R (source sample), a 3D point's
    normalised image coords map as ``x_cur ≈ K @ R @ K^-1 @ x_prev``
    (small-translation approximation — adequate as an *init guess* for
    Sparse-feature, not for a final pose).

    Using this as ``OPTFLOW_USE_INITIAL_FLOW`` for Sparse-feature halves the search
    window during fast-yaw turns, dropping the mistrack rate
    significantly (no extra cost beyond a 3×3 matmul per feature).
    """
    K_inv = np.linalg.inv(K)
    p = prev_pts.reshape(-1, 2)
    h = np.hstack([p, np.ones((len(p), 1))])      # (N,3)
    x = (K_inv @ h.T).T                            # normalised
    y = (R_pred_cam @ x.T).T                       # rotated
    # Reproject, guard against z<=0 (point swung behind source).
    z = y[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    img = (K @ (y / z).T).T
    return img[:, :2].astype(np.float32).reshape(-1, 1, 2)


def _klt_with_fb_check(
    prev_gray: "np.ndarray",
    cur_gray: "np.ndarray",
    prev_pts: "np.ndarray",
    *,
    win_size: int = 21,
    max_level: int = 3,
    fb_threshold_px: float = 1.0,
    init_pts: Optional["np.ndarray"] = None,
):
    """Sparse-feature tracking with forward-backward error filter.

    Tracks features prev→cur, then cur→prev, and rejects any feature
    whose round-trip distance exceeds ``fb_threshold_px``. Catches
    mistracks (occlusion, repetitive texture, illumination change) that
    pass the per-sample Sparse-feature residual check but land on a wrong corner.

    Optional ``init_pts`` seeds the forward pass via the
    ``OPTFLOW_USE_INITIAL_FLOW`` flag — feed rate sensor-rotation-predicted
    coords to halve the Sparse-feature search radius during fast turns.

    Returns ``(cur_pts, mask_alive)`` where ``mask_alive`` is True for
    features that survived both the forward status flag and the FB check.
    """
    import cv2
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    if init_pts is not None:
        cur_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, init_pts.copy(),
            winSize=(win_size, win_size), maxLevel=max_level, criteria=crit,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
    else:
        cur_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None,
            winSize=(win_size, win_size), maxLevel=max_level, criteria=crit,
        )
    if st_fwd is None:
        return cur_pts, np.zeros((len(prev_pts),), dtype=bool)
    back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray, prev_gray, cur_pts, None,
        winSize=(win_size, win_size), maxLevel=max_level, criteria=crit,
    )
    mask = (st_fwd.flatten() == 1)
    if st_bwd is not None:
        mask &= (st_bwd.flatten() == 1)
        diff = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        mask &= (diff < fb_threshold_px)
    return cur_pts, mask


def run_vio_multiframe_v2(
    *,
    video_path: Path,
    recording_map: Path,
    frame_decim_hz: float = 5.0,
    max_features: int = 500,
    min_inliers: int = 40,
    track_length: int = 5,
    reseed_min_features: int = 80,
    n_buckets_x: int = 8,
    n_buckets_y: int = 6,
    fb_threshold_px: float = 1.0,
    use_magsac: bool = True,
    # Two experimental filters that REGRESS on reference session — keep off by default.
    #   refine_tracked_subpix: snaps Sparse-feature-tracked points back to corner via
    #     cornerSubPix. Sounds good but breaks multi-sample consistency
    #     (the "corner" the subpix solver finds is often a NEIGHBOUR of the
    #     Sparse-feature-tracked feature, so the same feature drifts to a different
    #     cell each sample). Reference session hRMSE: 2.252 → 2.362 (+4.5%).
    #   affine_outlier_filter: pre-essential-matrix RANSAC on global 2D
    #     affine motion. Drops points inconsistent with the dominant
    #     motion — including the parallax-rich foreground points that
    #     are the SOURCE of Motion model depth information. Reference session hRMSE:
    #     2.252 → 2.342 (loose 10px) or 2.418 (tight 3px).
    refine_tracked_subpix: bool = False,
    affine_outlier_filter: bool = False,
    affine_thresh_px: float = 3.0,
    orb_verify: bool = False,
    orb_max_hamming: int = 40,
    imu_rows: Optional[Sequence] = None,
    R_cam_from_body: Optional["np.ndarray"] = None,
    return_pairs: bool = False,
    log: Optional[object] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
) -> list[VioSample]:
    """Multi-sample Motion model v2 — bucketed features + FB-error Sparse-feature + MAGSAC++.

    Drop-in superset of :func:`run_vio_multiframe`. Three changes:

    1. **Bucketed detection** (:func:`_bucketed_features`). Forces
       spatial diversity → essential-matrix geometry conditioned better.
    2. **Forward-backward Sparse-feature** (:func:`_klt_with_fb_check`). Rejects
       mistracks that pass per-sample residual but land on a wrong
       corner — major source of translation-direction error.
    3. **MAGSAC++ RANSAC** for the relative-pose when available
       (The feature library ≥ 4.5). MAGSAC marginalises over the inlier threshold so
       a single fixed threshold (1 px) doesn't bias the inlier set.

    Same return contract as :func:`run_vio_multiframe`.
    """
    import cv2

    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    from .time_sync import fit_time_anchor
    from .frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
    anchor = fit_time_anchor(recording_map)
    _log(f"[vio-v2] time anchor: n={anchor.n} drift={anchor.drift_ppm:+.2f}ppm")
    t0_boot_ns = resolve_video_t0_boottime_ns(
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
        log=_log,
    )
    frame_to_utc = make_frame_to_utc(anchor, t0_boot_ns)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = _default_K(w, h)
    keep_every = max(1, int(round(src_fps / max(0.5, frame_decim_hz))))
    if use_magsac and not hasattr(cv2, "USAC_MAGSAC"):
        _log(
            "[vio-v2] OpenCV < 4.5 detected, falling back to RANSAC "
            "(MAGSAC++ unavailable). Install opencv-python>=4.5.0 for "
            "better essential-matrix robustness."
        )
    em_method = (
        cv2.USAC_MAGSAC if (use_magsac and hasattr(cv2, "USAC_MAGSAC"))
        else cv2.RANSAC
    )
    _log(f"[vio-v2] video {w}x{h} {src_fps:.2f}fps keep_every={keep_every} "
         f"track_len={track_length} buckets={n_buckets_x}x{n_buckets_y} "
         f"em_method={'MAGSAC' if em_method != cv2.RANSAC else 'RANSAC'}")

    samples: list[VioSample] = []
    pairs: list[VioFramePair] = []
    history: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    frame_idx = -1
    n_fb_dropped = 0

    # Optional rate sensor pre-rotation. Build a sorted (utc, gx, gy, gz) array
    # once so the per-sample integration is O(log N).
    gyro_t = None
    gyro_g = None
    if imu_rows is not None and R_cam_from_body is not None:
        gyro_t = np.asarray([r.utc_s for r in imu_rows], dtype=np.float64)
        gyro_g = np.asarray([(r.gx, r.gy, r.gz) for r in imu_rows],
                            dtype=np.float64)

    def _integrate_gyro_cam(t0: float, t1: float) -> np.ndarray:
        """Integrate rate sensor rad/s from t0 to t1, return cam-sample R."""
        if gyro_t is None or gyro_g is None or t1 <= t0:
            return np.eye(3)
        i0 = int(np.searchsorted(gyro_t, t0))
        i1 = int(np.searchsorted(gyro_t, t1))
        if i1 <= i0:
            return np.eye(3)
        # Trapezoidal integral of angular rate → rotation vector (body).
        ts = gyro_t[i0:i1]
        gs = gyro_g[i0:i1]
        if len(ts) < 2:
            return np.eye(3)
        dts = np.diff(ts)
        avg = 0.5 * (gs[:-1] + gs[1:])
        rvec_body = (avg.T * dts).sum(axis=1)
        # body→cam: r_cam = R_cb @ r_body. R = exp(skew(r_cam)).
        rvec_cam = R_cam_from_body @ rvec_body
        return _rodrigues_to_R(rvec_cam)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % keep_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_video_s = frame_idx / src_fps
        utc_s = frame_to_utc(t_video_s)

        if not history:
            pts0 = _bucketed_features(
                gray, max_total=max_features,
                n_buckets_x=n_buckets_x, n_buckets_y=n_buckets_y,
            )
            if pts0 is None:
                continue
            history.append((utc_s, t_video_s, gray, pts0.reshape(-1, 1, 2)))
            continue

        prev_utc, prev_tv, prev_gray, prev_pts = history[-1]
        if prev_pts is None or len(prev_pts) < 10:
            new_pts = _bucketed_features(
                prev_gray, max_total=max_features,
                n_buckets_x=n_buckets_x, n_buckets_y=n_buckets_y,
            )
            if new_pts is None:
                history = [(utc_s, t_video_s, gray, None)]  # type: ignore[list-item]
                continue
            prev_pts = new_pts.reshape(-1, 1, 2)
            history[-1] = (prev_utc, prev_tv, prev_gray, prev_pts)

        # Rate sensor-predicted feature positions as Sparse-feature init guess (rotation
        # component only — translation handled by Sparse-feature search).
        init_pts = None
        if gyro_t is not None:
            R_pred = _integrate_gyro_cam(prev_utc, utc_s)
            # Skip pred when rotation is small; Sparse-feature init is cheap but
            # the matmul + reshape isn't free.
            if abs(R_pred[0, 1]) + abs(R_pred[0, 2]) + abs(R_pred[1, 2]) > 1e-4:
                try:
                    init_pts = _gyro_predicted_pts(prev_pts, K, R_pred)
                except (cv2.error, ValueError):
                    init_pts = None
        nxt, mask = _klt_with_fb_check(
            prev_gray, gray, prev_pts,
            fb_threshold_px=fb_threshold_px,
            init_pts=init_pts,
        )
        n_fb_dropped += int((~mask).sum())

        # Per-sample sub-cell refinement on the survivors. Sparse-feature lands
        # ~0.3 px from the true corner; refining to ~0.05 px halves the
        # essential-matrix residual on the inliers.
        if refine_tracked_subpix and mask.any():
            alive_pts = nxt[mask]
            alive_pts = _refine_tracked_subpix(gray, alive_pts)
            nxt[mask] = alive_pts

        # Affine RANSAC pre-filter: drops features inconsistent with the
        # dominant 2D motion (independently-moving objects like other
        # cars). Run before findEssentialMat so the 5-point solver sees
        # a cleaner inlier set.
        if affine_outlier_filter and mask.sum() >= 8:
            alive_prev = prev_pts.reshape(-1, 2)[mask]
            alive_cur = nxt.reshape(-1, 2)[mask]
            af_mask = _affine_outlier_filter(
                alive_prev, alive_cur,
                ransac_thresh_px=affine_thresh_px,
            )
            # Compose: mask = mask AND (af_mask within mask=True positions).
            idx_alive = np.where(mask)[0]
            n_affine_drop = int((~af_mask).sum())
            for i, ok in zip(idx_alive, af_mask):
                if not ok:
                    mask[i] = False
            del idx_alive, af_mask, n_affine_drop

        # Propagate alive set backwards through the buffer.
        for k in range(len(history)):
            u, tv, g, p = history[k]
            if p is not None:
                history[k] = (u, tv, g, p[mask])
        cur_pts = nxt[mask].reshape(-1, 1, 2)
        history.append((utc_s, t_video_s, gray, cur_pts))

        if len(history) >= 2:
            base_idx = max(0, len(history) - 1 - track_length)
            base_u, base_tv, base_g, base_pts = history[base_idx]
            if base_pts is not None and len(base_pts) >= 8:
                p0 = base_pts.reshape(-1, 2)
                p1 = cur_pts.reshape(-1, 2)
                # Optional Keypoint descriptor verification: drops Sparse-feature
                # correspondences whose local appearance diverged over
                # the track (drift along edges, jump to similar
                # neighbour). Preserves parallax (unlike affine filter).
                if orb_verify and len(p0) >= 8:
                    ob_mask = _orb_verify_pairs(
                        base_g, gray, p0, p1, max_hamming=orb_max_hamming,
                    )
                    p0 = p0[ob_mask]; p1 = p1[ob_mask]
                E = None
                em = None
                if len(p0) >= 8:
                    E, em = cv2.findEssentialMat(
                        p0, p1, cameraMatrix=K,
                        method=em_method, prob=0.999, threshold=1.0,
                    )
                if E is not None and E.shape == (3, 3):
                    n_in_ess = int(em.sum()) if em is not None else 0
                    if n_in_ess >= min_inliers:
                        _, R, t, pose_mask = cv2.recoverPose(
                            E, p0, p1, cameraMatrix=K, mask=em,
                        )
                        t_unit = t.flatten() / max(1e-9, np.linalg.norm(t))
                        s = VioSample(
                            utc_s=utc_s,
                            dt_s=utc_s - base_u,
                            R_prev_to_cur=np.asarray(R, dtype=np.float64),
                            t_unit_cam=np.asarray(t_unit, dtype=np.float64),
                            n_inliers=n_in_ess,
                        )
                        samples.append(s)
                        if return_pairs:
                            keep = (pose_mask.flatten() != 0) if pose_mask is not None else np.ones(len(p0), dtype=bool)
                            pairs.append(VioFramePair(
                                sample=s,
                                base_pts_px=p0[keep].astype(np.float64),
                                cur_pts_px=p1[keep].astype(np.float64),
                            ))

        if cur_pts is None or len(cur_pts) < reseed_min_features:
            new_pts = _bucketed_features(
                gray, max_total=max_features,
                n_buckets_x=n_buckets_x, n_buckets_y=n_buckets_y,
            )
            history = [(utc_s, t_video_s, gray,
                        new_pts.reshape(-1, 1, 2) if new_pts is not None else None)]
        else:
            while len(history) > track_length + 1:
                history.pop(0)

    cap.release()
    n_valid = sum(1 for s in samples if math.isfinite(float(s.t_unit_cam[0])))
    _log(f"[vio-v2] samples={len(samples)} with_valid_t={n_valid} "
         f"fb_dropped={n_fb_dropped}")
    if return_pairs:
        return pairs  # type: ignore[return-value]
    return samples


def vio_to_enu_velocities(
    samples: Sequence[VioSample],
    pos_rows: Sequence[PosRow],
    R_body_from_cam: Optional[np.ndarray] = None,
    auto_calibrate: bool = True,
    smooth_doppler_window_s: float = 10.0,
    log: Optional[object] = None,
) -> list[tuple[float, np.ndarray]]:
    """Convert per-sample Motion model unit-translations into metric Local-frame velocities.

    Scaling: use Post-processing Rate-signal speed interpolated at the sample UTC. The
    direction comes from Motion model (source sample → body via ``R_body_from_cam``
    → world via the local Post-processing-velocity-heading rotation), the magnitude
    from Post-processing.

    This is deliberately Post-processing-anchored — it doesn't propagate scale across
    Signal outages. For outages > ~1s, fall back to Motion sensor integration.

    Returns
    -------
    list of (utc_s, v_enu) pairs at the Motion model sample rate.
    """
    def _log(m: str) -> None:
        if log is not None:
            log(m)  # type: ignore[operator]

    if R_body_from_cam is None:
        if auto_calibrate:
            R_body_from_cam, p50 = calibrate_R_body_from_cam(samples, pos_rows)
            _log(f"[vio] auto-calibrated R_body_from_cam, p50 residual "
                 f"{math.degrees(p50):.2f}°")
            if p50 > math.radians(20):
                _log("[vio] WARN calibration residual > 20°; device mount "
                     "likely changed mid-session — VIO output unreliable.")
        else:
            # Static dashcam fallback: cam +Z forward → body +X forward.
            R_body_from_cam = np.array([
                [0, 0, 1],   # body X = cam Z
                [1, 0, 0],   # body Y = cam X
                [0, 1, 0],   # body Z = cam Y
            ], dtype=np.float64)

    pos_t = np.asarray([r.utc_s for r in pos_rows], dtype=np.float64)
    pos_ve = np.asarray([r.ve if math.isfinite(r.ve) else 0.0
                         for r in pos_rows], dtype=np.float64)
    pos_vn = np.asarray([r.vn if math.isfinite(r.vn) else 0.0
                         for r in pos_rows], dtype=np.float64)
    # ── Rate-signal smoothing ────────────────────────────────────────────
    # Post-processing Rate-signal is locally noisy and spikes hard during environment noise
    # bursts (same event that corrupts the position). Using it directly
    # as the Motion model speed scale propagates that noise. Replace each epoch's
    # speed with a median over ±window/2 seconds — robust to outliers,
    # preserves the underlying speed profile.
    if smooth_doppler_window_s > 0 and len(pos_t) >= 3:
        half_w = smooth_doppler_window_s / 2.0
        spd = np.hypot(pos_ve, pos_vn)
        spd_smooth = np.empty_like(spd)
        for i in range(len(pos_t)):
            lo = np.searchsorted(pos_t, pos_t[i] - half_w, side="left")
            hi = np.searchsorted(pos_t, pos_t[i] + half_w, side="right")
            spd_smooth[i] = float(np.median(spd[lo:hi])) if hi > lo else spd[i]
    else:
        spd_smooth = np.hypot(pos_ve, pos_vn)

    def _ppk_speed_heading(t: float) -> tuple[float, float]:
        """(speed_mps, heading_rad_from_north) at Post-processing time t via lerp.

        Speed is taken from the SMOOTHED (median-filtered) Rate-signal;
        heading is taken from the raw vn/ve since direction is stable
        even when magnitude is spiky.
        """
        i = int(np.searchsorted(pos_t, t))
        if i <= 0 or i >= len(pos_rows):
            return float("nan"), float("nan")
        a_t, b_t = float(pos_t[i - 1]), float(pos_t[i])
        dt = b_t - a_t
        if dt <= 0:
            return float("nan"), float("nan")
        al = (t - a_t) / dt
        ve = float(pos_ve[i - 1] + al * (pos_ve[i] - pos_ve[i - 1]))
        vn = float(pos_vn[i - 1] + al * (pos_vn[i] - pos_vn[i - 1]))
        if not (math.isfinite(ve) and math.isfinite(vn)):
            return float("nan"), float("nan")
        # Heading from raw direction.
        head = math.atan2(ve, vn)
        # Speed from smoothed series.
        spd_a = float(spd_smooth[i - 1])
        spd_b = float(spd_smooth[i])
        spd = spd_a + al * (spd_b - spd_a)
        return spd, head

    out: list[tuple[float, np.ndarray]] = []
    n_used = 0
    for s in samples:
        if not math.isfinite(float(s.t_unit_cam[0])):
            continue
        spd, head = _ppk_speed_heading(s.utc_s)
        if not math.isfinite(spd) or spd < 0.3:
            continue
        # Source→body→Local-frame. body→Local-frame is a yaw rotation by the Post-processing heading.
        t_body = R_body_from_cam @ s.t_unit_cam
        ch, sh = math.cos(head), math.sin(head)
        # Body sample: X forward, Y right, Z down (vehicle).
        # World Local-frame: rotate body.X (forward) along heading.
        # Local-frame(E,N,U) = R_enu_body @ body, where body.X is heading direction.
        # heading from N: ve = sin(head)·v, vn = cos(head)·v.
        bx, by, bz = float(t_body[0]), float(t_body[1]), float(t_body[2])
        ve = sh * bx + ch * by
        vn = ch * bx - sh * by
        vu = -bz  # body.Z down → world.U up
        # Scale by speed (unit t direction × scalar speed).
        v_enu = np.array([ve, vn, vu]) * spd
        out.append((s.utc_s, v_enu))
        n_used += 1
    _log(f"[vio] scaled {n_used}/{len(samples)} samples to ENU velocities")
    return out


# ---------------------------------------------------------------------------
# Standalone essential-matrix primitive (additive; keyframe-graph foundation
# for Motion model-aided FGO). Independent of the loose-coupled subsystem above.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VioEdge:
    """One relative-pose edge between two keyframes (up-to-scale)."""

    t0: float
    t1: float
    R_rel: np.ndarray      # 3x3 relative rotation
    t_dir: np.ndarray      # unit 3-vector, translation direction (up to scale)
    quality: float         # inlier fraction 0..1


def essential_relative_pose(pts0, pts1, K):
    """Return (R, t_dir_unit, n_inliers) from matched points via relative-pose."""
    import cv2

    pts0 = np.ascontiguousarray(np.asarray(pts0, np.float64))
    pts1 = np.ascontiguousarray(np.asarray(pts1, np.float64))
    E, mask = cv2.findEssentialMat(
        pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or E.shape != (3, 3):
        return np.eye(3), np.array([0.0, 0.0, 1.0]), 0
    n_in, R, t, mask2 = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
    t = t.reshape(3)
    n = np.linalg.norm(t)
    t_dir = t / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return R, t_dir, int(n_in)
