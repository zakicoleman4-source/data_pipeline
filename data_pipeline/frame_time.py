"""Canonical media-sample -> UTC time mapping (boottime + segment aware).

Several stages independently mapped a media sample's source PTS to UTC. The
naive `anchor.video_pts_to_utc_s(pts)` is correct ONLY for the legacy
`video_ns` anchor dialect. For boottime-format sessions the time anchor's
x-domain is absolute ``CLOCK_BOOTTIME`` ns, so a raw PTS (which starts near 0)
extrapolates ~t0 seconds early -- and for a cut ("segment") clip whose PTS are
rebased to 0 it is doubly wrong. Every sample->time site must therefore lift the
PTS into bootNs with the clip's real sample-0 boottime before hitting the anchor.

This module is the single source of truth for that resolution, mirroring the
(previously duplicated) logic in ``stages.georef._load_frames`` and
``stages.viewers``. The segment rule matches ``docs/findings/segment-time-contract.md``:
the segment's own ``video_anchor.txt`` ``min(bootNs)`` is the authoritative sample-0
boottime and WINS over the parent ``capture_meta`` t0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


def first_boottime_ns_from_video_anchor(path: Optional[Path]) -> Optional[float]:
    """Earliest ``bootNs`` from a per-sample ``video_anchor.txt``.

    Format (header + rows): ``frameNumber,sensorTimestampNs(raw),bootNs,timestampSource``.
    Returns the minimum bootNs across data rows, or ``None`` if unreadable/empty.
    This is the physically-real sample-0 boottime (the authoritative segment t0).
    """
    if path is None:
        return None
    try:
        boots: list[float] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    boots.append(float(int(parts[2])))
                except ValueError:
                    continue
        if not boots:
            return None
        return min(boots)
    except OSError:
        return None


def resolve_video_t0_boottime_ns(
    *,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """Resolve the media sample-0 ``CLOCK_BOOTTIME`` ns for a session.

    Precedence (segment-correct):
      1. ``chop_video_anchor`` min bootNs -- for a cut clip this WINS over
         everything (the parent ``capture_meta`` t0 is the original full
         session's sample 0 and would map segment samples minutes early).
      2. ``capture_meta.video_t0_boottime_ns`` (non-segment boottime sessions).
      3. ``video_anchor`` min bootNs (fallback when capture_meta lacks a t0).

    Returns ``None`` for legacy `video_ns` sessions (no boottime t0 available),
    in which case callers should use ``anchor.video_pts_to_utc_s`` directly.
    """
    def _log(m: str) -> None:
        if log is not None:
            log(m)

    if chop_video_anchor is not None:
        t0 = first_boottime_ns_from_video_anchor(Path(chop_video_anchor))
        if t0 is not None:
            _log("[frame_time] chop clip: t0 from chop video_anchor min bootNs, "
                 "capture_meta t0 overridden")
            return t0
        _log(f"[frame_time] WARN chop video_anchor {Path(chop_video_anchor).name} "
             "unreadable/empty; falling back to standard t0 resolution")

    if capture_meta is not None and Path(capture_meta).is_file():
        try:
            from .capture_meta import parse_capture_meta
            cm = parse_capture_meta(Path(capture_meta))
            if cm.video_t0_boottime_ns is not None:
                return float(cm.video_t0_boottime_ns)
        except FileNotFoundError:
            pass
        except Exception as e:  # tolerant: a bad manifest must not kill mapping
            _log(f"[frame_time] capture_meta parse failed ({e}); trying video_anchor")

    if video_anchor is not None:
        t0 = first_boottime_ns_from_video_anchor(Path(video_anchor))
        if t0 is not None:
            _log(f"[frame_time] timeline offset from {Path(video_anchor).name}")
            return t0

    return None


def make_frame_to_utc(
    anchor,
    video_t0_boottime_ns: Optional[float],
    *,
    manual_shift_s: float = 0.0,
) -> Callable[[float], float]:
    """Return ``pts_seconds -> utc_s`` for a fitted :class:`TimeAnchor`.

    When ``video_t0_boottime_ns`` is known (boottime/segment session) the PTS is
    lifted into bootNs first: ``boottime_to_utc_s(t0 + pts*1e9)``. Otherwise the
    legacy direct mapping ``video_pts_to_utc_s(pts)`` is used. ``manual_shift_s``
    is added to the result (operator override), matching coordinate output.
    """
    if video_t0_boottime_ns is not None:
        t0 = float(video_t0_boottime_ns)
        return lambda pts: anchor.boottime_to_utc_s(t0 + pts * 1e9) + manual_shift_s
    return lambda pts: anchor.video_pts_to_utc_s(pts) + manual_shift_s
