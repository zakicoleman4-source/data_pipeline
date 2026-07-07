"""Rate-signal / CV-driven adaptive sample selection.

The default ``samples`` stage extracts at a fixed FPS — fine for a vehicle
driving at roughly constant speed, but it produces an avalanche of samples
during static stops and may under-sample tight turns where the source
swings faster than the path.

This stage computes a *keep list* of source sample indices using two rules:

* **Straight motion** — keep the next sample each time the cumulative
  along-track distance from the last kept sample reaches ``spacing_m``
  metres. Distance comes from the Post-processing Rate-signal speed integrated between
  sample timestamps; static stops naturally generate very few samples.
* **Turns** — when the Post-processing heading rate exceeds
  ``yaw_rate_threshold_dps`` deg/s, the distance rule is overridden by a
  vision-based overlap check: Keypoint keypoints are extracted on a thumbnail
  of the candidate sample, matched against the last kept thumbnail, and a
  planar model is RANSAC-fit. Overlap is the fraction of the candidate's
  cell area that warps into the previous sample; a new sample is kept as
  soon as overlap drops below ``turn_overlap`` (default 0.80, i.e. keep
  when at most 80% of the new sample is shared with the last kept one).

Hard guards ensure the selector always makes progress:

* ``min_interval_s`` — never keep two samples closer than this in time
  (caps the burst rate during very fast turns).
* ``max_interval_s`` — always keep a sample after this long without one,
  even during long stops (anchor point for downstream alignment).

The keep-list is then fed into :func:`data_pipeline.stages.samples.run`
via the ``select_indices`` parameter; high-quality lossless extraction
runs in a single the external converter pass against the source media.

The feature library (``cv2``) is required for the turn-time overlap check. It is a
soft dependency — if it cannot be imported, the stage still works using
distance-only selection and logs a warning when it skips a turn.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe
from ..frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
from ..parsers import parse_rtkpos, PosRow
from ..pipeline import LogFn, make_logger
from ..time_sync import fit_time_anchor

try:
    import cv2  # type: ignore[import-not-found]
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


_SHOWINFO_RE = re.compile(r"showinfo.*?n:\s*(\d+).*?pts_time:\s*([\d.]+)")


@dataclass
class AdaptiveOptions:
    """Tunables for the adaptive selector."""

    # Straight-motion target spacing in metres along-track.
    spacing_m: float = 2.0
    # During turns, keep the new sample as soon as overlap with the previous
    # kept sample drops to this fraction (e.g. 0.80 = 80% shared).
    turn_overlap: float = 0.80
    # Post-processing heading-rate threshold (deg/s) above which the turn branch kicks in.
    yaw_rate_threshold_dps: float = 5.0
    # Thumbnail height for the CV overlap test. Width is computed to preserve
    # aspect ratio. 240 px ≈ small enough for Keypoint to be fast, big enough for
    # robust keypoint matches on driving imagery.
    cv_thumb_height: int = 240
    # Maximum time-window (seconds) used for the heading-rate finite-difference.
    heading_rate_window_s: float = 1.0
    # Speed threshold (m/s) below which a sample is considered "static"; the
    # heading derived from coords is unreliable there, so we always fall back
    # to the time-based rules.
    static_speed_mps: float = 0.4
    # Hard interval guards.
    min_interval_s: float = 0.10
    max_interval_s: float = 30.0
    # Keypoint tuning.
    orb_n_features: int = 600
    # Minimum number of RANSAC inliers needed to trust a planar model.
    min_inliers: int = 12
    # When The feature library is missing or the planar model fails, fall back to the
    # distance rule with ``turn_fallback_spacing_m`` instead of skipping.
    turn_fallback_spacing_m: float = 0.5


@dataclass
class AdaptiveResult:
    """Outcome of the selector."""
    keep_indices: List[int] = field(default_factory=list)
    keep_pts_s: List[float] = field(default_factory=list)
    n_total: int = 0
    n_kept: int = 0
    n_turn_decisions: int = 0
    n_straight_decisions: int = 0
    n_cv_overlaps_computed: int = 0
    cv_available: bool = _HAS_CV2


# ---------------------------------------------------------------------------
# Source enumeration: list every (frame_index, pts_seconds) without encoding
# ---------------------------------------------------------------------------

def enumerate_source_frames(video: Path) -> List[Tuple[int, float]]:
    """Return ``[(n, pts_s)]`` for every sample in ``media``.

    Uses the external converter's ``showinfo`` filter with a null muxer so no images are
    written to disk; only the stderr lines are parsed. This is fast even on
    long sessions (decode is hardware-accelerated when available).
    """
    cmd = [
        resolve_ffmpeg(),
        "-hide_banner", "-loglevel", "info",
        "-i", str(video),
        "-vf", "showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    # the external converter streams showinfo to stderr.
    out: List[Tuple[int, float]] = []
    for line in proc.stderr.splitlines():
        m = _SHOWINFO_RE.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Post-processing lookups
# ---------------------------------------------------------------------------

def _speed_at(pos_rows: Sequence[PosRow], utc_s: float) -> float:
    """Rate-signal horizontal speed at ``utc_s`` (linear interp; 0 outside window)."""
    if not pos_rows:
        return 0.0
    if utc_s <= pos_rows[0].utc_s or utc_s >= pos_rows[-1].utc_s:
        return 0.0
    import bisect
    times = [r.utc_s for r in pos_rows]
    i = bisect.bisect_left(times, utc_s)
    a, b = pos_rows[i - 1], pos_rows[i]
    if not (math.isfinite(a.ve) and math.isfinite(a.vn)
            and math.isfinite(b.ve) and math.isfinite(b.vn)):
        return 0.0
    sa = math.hypot(a.ve, a.vn)
    sb = math.hypot(b.ve, b.vn)
    t = (utc_s - a.utc_s) / max(b.utc_s - a.utc_s, 1e-9)
    return sa + (sb - sa) * t


def _heading_deg_at(pos_rows: Sequence[PosRow], utc_s: float,
                    min_speed: float) -> Optional[float]:
    """Rate-signal-derived heading in degrees at ``utc_s``; ``None`` when static.

    Linearly interpolates the (ve, vn) components between the bracketing
    Post-processing rows before taking ``atan2``. The previous nearest-past-row
    implementation made ``_heading_rate_dps`` return 0 whenever both
    ``utc_s ± window/2`` samples fell into the same 1 Hz Post-processing interval
    (any heading-rate window < 1 s); the interpolated form has continuous
    derivatives so the yaw-rate finite-difference is meaningful for any
    positive window.
    """
    if not pos_rows:
        return None
    import bisect
    times = [r.utc_s for r in pos_rows]
    if utc_s <= times[0] or utc_s >= times[-1]:
        return None
    i = bisect.bisect_left(times, utc_s)
    a, b = pos_rows[i - 1], pos_rows[i]
    if not (math.isfinite(a.ve) and math.isfinite(a.vn)
            and math.isfinite(b.ve) and math.isfinite(b.vn)):
        return None
    span = max(b.utc_s - a.utc_s, 1e-9)
    alpha = (utc_s - a.utc_s) / span
    ve = a.ve + alpha * (b.ve - a.ve)
    vn = a.vn + alpha * (b.vn - a.vn)
    if math.hypot(ve, vn) < min_speed:
        return None
    return math.degrees(math.atan2(ve, vn)) % 360.0


def _angle_wrap_deg(d: float) -> float:
    """Wrap a degree difference to (-180, 180]."""
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def _heading_rate_dps(
    pos_rows: Sequence[PosRow], utc_s: float,
    window_s: float, static_speed_mps: float,
) -> float:
    """Finite-difference yaw rate in degrees per second around ``utc_s``."""
    h0 = _heading_deg_at(pos_rows, utc_s - window_s / 2, static_speed_mps)
    h1 = _heading_deg_at(pos_rows, utc_s + window_s / 2, static_speed_mps)
    if h0 is None or h1 is None:
        return 0.0
    return abs(_angle_wrap_deg(h1 - h0)) / max(window_s, 1e-6)


# ---------------------------------------------------------------------------
# CV overlap (Keypoint + planar model corner-warp)
# ---------------------------------------------------------------------------

def _orb_overlap_fraction(
    img_new: "np.ndarray", img_prev: "np.ndarray",
    *, n_features: int = 600, min_inliers: int = 12,
) -> Optional[float]:
    """Return the fraction of ``img_new``'s area that warps into ``img_prev``.

    Returns ``None`` when matching is too weak to trust (the caller should
    fall back to the distance rule).
    """
    if cv2 is None:
        return None
    g_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY) if img_new.ndim == 3 else img_new
    g_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev
    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp_n, des_n = orb.detectAndCompute(g_new, None)
    kp_p, des_p = orb.detectAndCompute(g_prev, None)
    if des_n is None or des_p is None or len(kp_n) < 8 or len(kp_p) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_n, des_p)
    if len(matches) < min_inliers:
        return None
    pts_new = np.float32([kp_n[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_prev = np.float32([kp_p[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_new, pts_prev, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < min_inliers:
        return None
    # Warp the new sample's corners through H, intersect with previous sample.
    h_n, w_n = g_new.shape[:2]
    h_p, w_p = g_prev.shape[:2]
    corners = np.float32([[0, 0], [w_n, 0], [w_n, h_n], [0, h_n]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    # Clip warped polygon against previous sample's image rectangle by clamping
    # vertex coordinates — adequate for the gentle, near-affine warps produced
    # by a single ~0.5 s step on a vehicle.
    clipped = np.stack([
        np.clip(warped[:, 0], 0, w_p),
        np.clip(warped[:, 1], 0, h_p),
    ], axis=1)
    # Shoelace area of the (possibly self-intersecting) quad.
    area = 0.5 * abs(
        clipped[0, 0] * (clipped[1, 1] - clipped[3, 1])
        + clipped[1, 0] * (clipped[2, 1] - clipped[0, 1])
        + clipped[2, 0] * (clipped[3, 1] - clipped[1, 1])
        + clipped[3, 0] * (clipped[0, 1] - clipped[2, 1])
    )
    frame_area = float(w_n * h_n)
    if frame_area <= 0:
        return None
    return min(1.0, max(0.0, area / frame_area))


# ---------------------------------------------------------------------------
# Thumbnail streamer
# ---------------------------------------------------------------------------

class _ThumbStream:
    """Stream the media as small samples in source-sample order.

    Uses ``cv2.VideoCapture`` so the per-sample index here matches what
    ``the external converter -vf select=eq(n,N)`` and the showinfo enumeration see. A
    direct raw-rgb24 pipe was tried first and silently misaligned the
    sample counter on the reference session device media (different scene per N) —
    cv2.VideoCapture stays in lockstep with the standard decode path.
    """

    def __init__(self, video: Path, thumb_h: int) -> None:
        if cv2 is None:
            raise RuntimeError(
                "_ThumbStream needs OpenCV (`pip install opencv-python`)."
            )
        self._cap = cv2.VideoCapture(str(video))
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open {video}")
        # Probe the auto-rotated sample size by peeking once. We CACHE the
        # first sample instead of seeking back to 0: cv2 seek on inter-sample
        # codecs rounds to the nearest preceding keyframe, which only happens
        # to be sample 0 for well-formed device H.264 — counting on that as an
        # invariant has bit other tools. Yielding the cached first sample on
        # the first read_next() keeps the sample-index → PTS mapping aligned
        # for every backend.
        ok, first = self._cap.read()
        if not ok:
            raise RuntimeError(f"Empty video stream: {video}")
        H, W = first.shape[:2]
        scale_h = max(64, int(thumb_h))
        scale_w = max(64, int(round(W * scale_h / H / 2.0)) * 2)
        self.w = scale_w
        self.h = scale_h
        self._n = -1
        self._first_frame: Optional["np.ndarray"] = first

    def read_next(self) -> Optional[Tuple[int, "np.ndarray"]]:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
        else:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
        self._n += 1
        # cv2 returns BGR; downscale with INTER_AREA for clean thumbnails.
        thumb = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return self._n, thumb

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def compute_keep_list(
    *,
    video: Path,
    pos_file: Path,
    recording_map: Path,
    options: Optional[AdaptiveOptions] = None,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[LogFn] = None,
) -> AdaptiveResult:
    """Return source sample indices to keep under the adaptive policy.

    ``media`` is decoded once as small RGB thumbnails; each sample's PTS
    is converted to UTC via the session map's time anchor, looked up in
    the Post-processing path, and tested against the spacing / turn rules.

    ``capture_meta`` / ``video_anchor`` / ``chop_video_anchor`` resolve the
    media sample-0 ``CLOCK_BOOTTIME`` t0 for boottime-format sessions (see
    ``data_pipeline.frame_time``). For those sessions the time anchor's
    x-domain is absolute bootNs, so mapping a raw PTS directly through
    ``video_pts_to_utc_s`` lands ~t0 seconds early — every sample then falls
    outside the .pos window, speed/heading lookups return 0 and the selector
    runs blind (keeping only via the ``max_interval_s`` floor). Pass the
    session's ``capture_meta.json`` (and, for a cut clip, the segment's own
    ``*.video_anchor.txt``, which WINS over capture_meta). When none resolve
    a t0 the legacy direct-PTS mapping is used, unchanged.
    """
    log_ = make_logger(log)
    opts = options or AdaptiveOptions()
    video = Path(video); pos_file = Path(pos_file); recording_map = Path(recording_map)
    if not _HAS_CV2:
        log_("[adaptive] WARNING: OpenCV unavailable; turn-overlap branch "
             "will fall back to a shorter spacing rule.")
    log_(f"[adaptive] spacing={opts.spacing_m:.2f} m, "
         f"turn_overlap={opts.turn_overlap:.2f}, "
         f"yaw_rate_thr={opts.yaw_rate_threshold_dps:.1f} deg/s")
    anchor = fit_time_anchor(recording_map)
    pos_rows = parse_rtkpos(pos_file)
    pos_rows = sorted(pos_rows, key=lambda r: r.utc_s)
    log_(f"[adaptive] time anchor RMSE = {anchor.rmse_s*1000:.3f} ms; "
         f"PPK rows = {len(pos_rows)}")

    # Sample->UTC mapping: boottime-aware (lifts PTS into bootNs when the
    # session/segment t0 is known), byte-for-byte legacy PTS mapping otherwise.
    video_t0_boottime_ns = resolve_video_t0_boottime_ns(
        capture_meta=Path(capture_meta) if capture_meta is not None else None,
        video_anchor=Path(video_anchor) if video_anchor is not None else None,
        chop_video_anchor=(
            Path(chop_video_anchor) if chop_video_anchor is not None else None
        ),
        log=log_,
    )
    frame_to_utc = make_frame_to_utc(anchor, video_t0_boottime_ns)
    if video_t0_boottime_ns is not None:
        log_(f"[adaptive] boottime session: PTS lifted by video t0 = "
             f"{video_t0_boottime_ns:.0f} ns before the anchor lookup")

    # CRITICAL — TRUE PER-Sample PTS.
    # Earlier revisions reconstructed pts = n / avg_fps, which is wrong for
    # variable-FPS device media (reference session jumps from pts=0.123 at n=4 to
    # pts=2.229 at n=5). That mislabelled the time anchor lookup and the
    # spacing math for every sample whose true PTS differs from n/avg_fps.
    # Now we enumerate every (n, pts) once via the external converter showinfo (no encoding
    # = fast) and look up the exact PTS for each thumbnail we walk through.
    pts_list = enumerate_source_frames(video)
    if not pts_list:
        raise RuntimeError(
            f"showinfo enumeration produced no frames for {video}. "
            "Adaptive mode cannot guess PTS without it."
        )
    frame_pts_by_n: dict[int, float] = {n: t for n, t in pts_list}
    log_(f"[adaptive] enumerated {len(frame_pts_by_n)} source frames via "
         f"showinfo; pts range {pts_list[0][1]:.3f}..{pts_list[-1][1]:.3f} s")

    res = AdaptiveResult()
    stream = _ThumbStream(video, opts.cv_thumb_height)

    last_kept_n: Optional[int] = None
    last_kept_pts: Optional[float] = None
    last_kept_thumb: Optional["np.ndarray"] = None
    cum_dist_m = 0.0
    prev_pts: Optional[float] = None
    prev_speed: float = 0.0

    try:
        while True:
            item = stream.read_next()
            if item is None:
                break
            n, thumb = item
            # TRUE PTS from showinfo. Aligns with what `select=eq(n,N)` sees
            # and with what the streaming-extract path will write into the
            # CSV / filename — guaranteeing the kept sample's UTC lookup uses
            # the same instant the photo was actually captured.
            pts = frame_pts_by_n.get(n)
            if pts is None:
                # showinfo skipped this index (rare — invalid sample).
                # Fall through to next thumbnail; do not enter selector logic
                # without a trustworthy PTS.
                res.n_total = n + 1
                continue
            res.n_total = n + 1

            utc = frame_to_utc(pts)
            speed = _speed_at(pos_rows, utc)
            yaw_rate = _heading_rate_dps(
                pos_rows, utc, opts.heading_rate_window_s, opts.static_speed_mps,
            )

            # First sample is always kept.
            if last_kept_n is None:
                _keep(res, n, pts)
                last_kept_n, last_kept_pts = n, pts
                last_kept_thumb = thumb.copy()
                prev_pts = pts
                prev_speed = speed
                continue

            dt = pts - (prev_pts if prev_pts is not None else pts)
            # Trapezoidal speed integration → distance increment.
            cum_dist_m += 0.5 * (prev_speed + speed) * max(dt, 0.0)
            prev_pts = pts
            prev_speed = speed

            # NOTE: ``last_kept_pts`` can legitimately be 0.0 for the first
            # sample; use ``is None`` rather than ``or`` (which is falsy on 0.0).
            time_since_kept = pts - (
                last_kept_pts if last_kept_pts is not None else pts
            )

            # Hard cap: never bunch samples too tightly.
            if time_since_kept < opts.min_interval_s:
                continue
            # Hard floor: long stop ⇒ anchor sample.
            if time_since_kept >= opts.max_interval_s:
                _keep(res, n, pts)
                cum_dist_m = 0.0
                last_kept_n, last_kept_pts = n, pts
                last_kept_thumb = thumb.copy()
                continue

            in_turn = yaw_rate >= opts.yaw_rate_threshold_dps \
                and speed >= opts.static_speed_mps

            if in_turn:
                res.n_turn_decisions += 1
                ov: Optional[float] = None
                if last_kept_thumb is not None and _HAS_CV2:
                    ov = _orb_overlap_fraction(
                        thumb, last_kept_thumb,
                        n_features=opts.orb_n_features,
                        min_inliers=opts.min_inliers,
                    )
                    res.n_cv_overlaps_computed += 1
                if ov is None:
                    # Fallback when CV is missing or the planar model is shaky.
                    if cum_dist_m >= opts.turn_fallback_spacing_m:
                        _keep(res, n, pts)
                        cum_dist_m = 0.0
                        last_kept_n, last_kept_pts = n, pts
                        last_kept_thumb = thumb.copy()
                elif ov <= opts.turn_overlap:
                    _keep(res, n, pts)
                    cum_dist_m = 0.0
                    last_kept_n, last_kept_pts = n, pts
                    last_kept_thumb = thumb.copy()
            else:
                res.n_straight_decisions += 1
                if cum_dist_m >= opts.spacing_m:
                    _keep(res, n, pts)
                    cum_dist_m = 0.0
                    last_kept_n, last_kept_pts = n, pts
                    last_kept_thumb = thumb.copy()
    finally:
        stream.close()

    log_(f"[adaptive] decoded {res.n_total} source frames, "
         f"kept {res.n_kept} ({100.0 * res.n_kept / max(res.n_total,1):.1f}%); "
         f"turns={res.n_turn_decisions}, straight={res.n_straight_decisions}, "
         f"CV overlaps computed={res.n_cv_overlaps_computed}")
    return res


def _keep(res: AdaptiveResult, n: int, pts: float) -> None:
    res.keep_indices.append(n)
    res.keep_pts_s.append(pts)
    res.n_kept += 1
