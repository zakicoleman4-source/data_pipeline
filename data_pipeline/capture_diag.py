"""Capture / media diagnostics for a single session.

This module gathers, in one read-only pass over a session folder (or an
explicit set of file paths), the numbers a reviewer needs to judge how clean a
capture is:

1. SYNCHRONISATION (stream -> Signal and media -> Signal)
   Every clock reconciles through CLOCK_BOOTTIME:

       stream sample --(audio_anchor)--> boot --(Signal boot->UTC anchor)--> UTC
       media pts   --(media t0 + pts)--> boot --(Signal boot->UTC anchor)--> UTC
       media sample --(video_anchor)----> boot --(Signal boot->UTC anchor)--> UTC

   We report, for each of stream and media, the clock OFFSET (ms) of that
   timeline's t0 relative to Signal UTC and the DRIFT (ppm) of that timeline
   versus Signal.

   * Stream offset/drift: reuse :mod:`data_pipeline.audio_sync`
     (``analyze_audio`` / ``SyncStats``) — the stream device sample-clock drift
     and the stream-start UTC come straight from there.
   * Media offset/drift: the media sample clock (``video_anchor.txt`` bootNs vs
     sample index) gives the media timeline's own rate; compared with the Signal
     boot->UTC slope this yields a media<->Signal drift in ppm. The media
     offset is the residual between the Signal-mapped media t0 and the stream t0
     (i.e. how far apart the two media timelines start in UTC) when stream is
     present, else 0 (media t0 *is* the Signal reference instant by construction).

2. Cut
   How much leading/trailing media has NO valid Signal/Post-processing coverage. Coverage is
   the UTC interval spanned by the Signal anchor (or a supplied ``.pos`` file,
   honouring a maximum interpolation gap). ``head_trim_s`` is media before
   coverage starts, ``tail_trim_s`` is media after coverage ends,
   ``total_trim_s = video_duration - covered_duration`` and ``pct_kept`` is the
   kept fraction.

3. Media STATS (via the probe tool)
   resolution, fps (avg_frame_rate), duration, file size and DATA RATE
   (MB per minute = file_size_MB / duration_min).

4. FOCAL LENGTH
   Probed from the probe tool stream/format tags and capture_meta.json. Device container file
   usually do NOT carry focal length; when absent we report ``None`` with a
   note rather than fabricating a value. A 35mm-equivalent or cell focal is
   surfaced with its source when present anywhere.

Everything here is READ-ONLY on the inputs and tolerant of missing pieces:
each field is optional and ``notes`` explains every gap.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ffmpeg_paths import resolve_ffprobe
from .time_sync import TimeAnchor


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class CaptureDiag:
    """Capture / media diagnostics for one session.

    Every field is optional; ``notes`` records why anything is missing. Numeric
    fields are ``None`` (not NaN) when unavailable so ``to_dict`` is JSON-clean.
    """

    # --- Synchronisation -----------------------------------------------------
    audio_gnss_offset_ms: Optional[float] = None
    audio_gnss_drift_ppm: Optional[float] = None
    video_gnss_offset_ms: Optional[float] = None
    video_gnss_drift_ppm: Optional[float] = None

    # --- Cut ----------------------------------------------------------------
    head_trim_s: Optional[float] = None
    tail_trim_s: Optional[float] = None
    total_trim_s: Optional[float] = None
    pct_kept: Optional[float] = None

    # --- Media stats ---------------------------------------------------------
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    file_size_bytes: Optional[int] = None
    mb_per_min: Optional[float] = None

    # --- Focal length --------------------------------------------------------
    focal_length: Optional[float] = None
    focal_source: Optional[str] = None  # where it came from, or "unavailable"

    notes: List[str] = field(default_factory=list)

    # --- convenience properties ---------------------------------------------
    @property
    def resolution(self) -> Optional[str]:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        def _r(v: Optional[float], nd: int) -> Optional[float]:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return None
            return round(float(v), nd)

        return {
            "audio_gnss_offset_ms": _r(self.audio_gnss_offset_ms, 2),
            "audio_gnss_drift_ppm": _r(self.audio_gnss_drift_ppm, 2),
            "video_gnss_offset_ms": _r(self.video_gnss_offset_ms, 2),
            "video_gnss_drift_ppm": _r(self.video_gnss_drift_ppm, 2),
            "head_trim_s": _r(self.head_trim_s, 3),
            "tail_trim_s": _r(self.tail_trim_s, 3),
            "total_trim_s": _r(self.total_trim_s, 3),
            "pct_kept": _r(self.pct_kept, 2),
            "width": int(self.width) if self.width else None,
            "height": int(self.height) if self.height else None,
            "resolution": self.resolution,
            "fps": _r(self.fps, 3),
            "duration_s": _r(self.duration_s, 3),
            "file_size_bytes": int(self.file_size_bytes) if self.file_size_bytes else None,
            "mb_per_min": _r(self.mb_per_min, 2),
            "focal_length": self.focal_length,
            "focal_source": self.focal_source or "unavailable",
            "notes": list(self.notes),
        }


# =============================================================================
# the probe tool media stats
# =============================================================================


@dataclass(frozen=True)
class VideoProbe:
    """Raw fields parsed from the probe tool for a media file."""

    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_s: Optional[float]
    file_size_bytes: Optional[int]
    format_tags: Dict[str, str]
    stream_tags: Dict[str, str]
    raw: Dict[str, Any]


def _parse_fraction(text: Optional[str]) -> Optional[float]:
    """Parse a probe-tool rational like ``104868000/3504319`` into a float."""
    if not text:
        return None
    text = text.strip()
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(mp4: Path, *, ffprobe: Optional[str] = None) -> VideoProbe:
    """Run the probe tool on ``container file`` and extract resolution/fps/duration/size/tags.

    Raises ``FileNotFoundError`` if the probe tool cannot be resolved or the file is
    missing; raises ``RuntimeError`` if the probe tool fails or emits no media stream.
    """
    mp4 = Path(mp4)
    if not mp4.is_file():
        raise FileNotFoundError(f"video file not found: {mp4}")
    exe = ffprobe or resolve_ffprobe()
    cmd = [
        exe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}"
        )
    return parse_ffprobe_json(proc.stdout, file_size_bytes=mp4.stat().st_size)


def parse_ffprobe_json(
    text: str, *, file_size_bytes: Optional[int] = None
) -> VideoProbe:
    """Parse the probe tool ``-print_format json`` output into a :class:`VideoProbe`.

    Split out from :func:`probe_video` so tests can feed canned the probe tool JSON
    without an actual the probe tool binary.
    """
    data = json.loads(text)
    streams = data.get("streams") or []
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    fmt = data.get("format") or {}

    width = height = None
    fps = None
    stream_tags: Dict[str, str] = {}
    if vstreams:
        v = vstreams[0]
        width = _to_int(v.get("width"))
        height = _to_int(v.get("height"))
        fps = _parse_fraction(v.get("avg_frame_rate")) or _parse_fraction(
            v.get("r_frame_rate")
        )
        stream_tags = {str(k): str(val) for k, val in (v.get("tags") or {}).items()}

    duration_s = _to_float(fmt.get("duration"))
    if duration_s is None and vstreams:
        duration_s = _to_float(vstreams[0].get("duration"))

    size = file_size_bytes
    if size is None:
        size = _to_int(fmt.get("size"))

    format_tags = {str(k): str(val) for k, val in (fmt.get("tags") or {}).items()}

    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        file_size_bytes=size,
        format_tags=format_tags,
        stream_tags=stream_tags,
        raw=data,
    )


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# =============================================================================
# Focal length
# =============================================================================

# Tag keys that have ever been seen to carry a focal length (cell, mm, or
# 35mm-equivalent). Device container file almost never write any of these.
_FOCAL_TAG_KEYS = (
    "focal_length",
    "focallength",
    "focal_length_35mm",
    "focal_length_in_35mm_format",
    "focallengthin35mmfilm",
    "com.apple.quicktime.camera.focal_length_35mm_equivalent",
    "lens",
    "lensmodel",
)

# capture_meta.json nested keys to search for a focal length.
_FOCAL_META_KEYS = (
    "focal_length",
    "focal_length_px",
    "focal_length_mm",
    "focal_length_35mm",
    "focal_length_35mm_equiv",
    "fx",
)


def extract_focal_length(
    probe: Optional[VideoProbe],
    capture_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[str], List[str]]:
    """Best-effort focal length lookup.

    Returns ``(focal_length, source, notes)``. ``focal_length`` is ``None`` and
    ``source`` is ``None`` when nothing usable is found (the caller marks it
    "unavailable"). NEVER fabricates a value.
    """
    notes: List[str] = []

    # 1) the probe tool stream/format tags.
    if probe is not None:
        for scope, tags in (("stream", probe.stream_tags), ("format", probe.format_tags)):
            lowered = {k.lower(): v for k, v in tags.items()}
            for key in _FOCAL_TAG_KEYS:
                if key in lowered:
                    val = _to_float(lowered[key])
                    if val is not None and val > 0:
                        return val, f"ffprobe {scope} tag '{key}'", notes
                    # Non-numeric (e.g. a lens *name*) — surface as text source.
                    raw = lowered[key].strip()
                    if raw:
                        notes.append(
                            f"focal-related tag '{key}' present but non-numeric: "
                            f"{raw!r} (no numeric focal length)."
                        )

    # 2) capture_meta.json (search a couple of plausible nestings).
    if capture_meta:
        candidates: List[Dict[str, Any]] = [capture_meta]
        for sub in ("video", "camera", "lens", "intrinsics"):
            v = capture_meta.get(sub)
            if isinstance(v, dict):
                candidates.append(v)
        for block in candidates:
            for key in _FOCAL_META_KEYS:
                if key in block:
                    val = _to_float(block[key])
                    if val is not None and val > 0:
                        return val, f"capture_meta.json '{key}'", notes

    notes.append(
        "focal length unavailable: device mp4 exposes no focal_length tag and "
        "capture_meta.json carries no intrinsics."
    )
    return None, None, notes


# =============================================================================
# Media sample clock (video_anchor.txt) -> rate vs Signal
# =============================================================================


def parse_video_anchor(path: Path) -> List[Tuple[float, float]]:
    """Read ``(frameNumber, bootNs)`` pairs from a ``*.video_anchor.txt``.

    File layout (current app)::

        # frameNumber,sensorTimestampNs(raw),bootNs,timestampSource
        1383,2129920259000,2129920259000,REALTIME

    Tolerates a header line and rows where the optional source column is
    absent. ``bootNs`` is taken from column 2 (the boot-time column); when only
    two columns exist the second column is used.
    """
    pairs: List[Tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                frame = float(parts[0])
                boot_ns = float(parts[2]) if len(parts) >= 3 else float(parts[1])
            except (ValueError, IndexError):
                continue
            pairs.append((frame, boot_ns))
    return pairs


def video_frame_period_ns(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """OLS slope ``ns per sample`` of bootNs vs frameNumber (the media rate).

    Returns ``None`` when fewer than two distinct samples are available.
    """
    if len(pairs) < 2:
        return None
    frames = [p[0] for p in pairs]
    boots = [p[1] for p in pairs]
    n = len(pairs)
    fmean = sum(frames) / n
    bmean = sum(boots) / n
    sxx = sum((f - fmean) ** 2 for f in frames)
    if sxx <= 0:
        return None
    sxy = sum((f - fmean) * (b - bmean) for f, b in zip(frames, boots))
    return sxy / sxx


# =============================================================================
# Cut coverage
# =============================================================================


def _pos_coverage_utc(
    pos_rows: List[Any], *, max_gap_s: float
) -> Optional[Tuple[float, float]]:
    """Largest contiguous UTC coverage window in a ``.pos`` (rows with utc_s).

    Splits the (sorted) epochs wherever two consecutive epochs are more than
    ``max_gap_s`` apart and returns the [start, end] of the LONGEST run. Returns
    ``None`` for empty input.
    """
    times = sorted(float(r.utc_s) for r in pos_rows if math.isfinite(getattr(r, "utc_s", float("nan"))))
    if not times:
        return None
    if len(times) == 1:
        return (times[0], times[0])
    best_start = run_start = times[0]
    best_end = prev = times[0]
    for t in times[1:]:
        if t - prev > max_gap_s:
            if (prev - run_start) > (best_end - best_start):
                best_start, best_end = run_start, prev
            run_start = t
        prev = t
    if (prev - run_start) > (best_end - best_start):
        best_start, best_end = run_start, prev
    return (best_start, best_end)


def compute_trim(
    *,
    video_t0_utc_s: float,
    video_end_utc_s: float,
    coverage_start_utc_s: float,
    coverage_end_utc_s: float,
) -> Tuple[float, float, float, float]:
    """Return ``(head_trim_s, tail_trim_s, total_trim_s, pct_kept)``.

    Cut is the leading/trailing media that lies OUTSIDE the Signal/Post-processing coverage
    window. All inputs are absolute UTC seconds.
    """
    video_dur = max(0.0, video_end_utc_s - video_t0_utc_s)
    # Intersection of [media] and [coverage].
    kept_start = max(video_t0_utc_s, coverage_start_utc_s)
    kept_end = min(video_end_utc_s, coverage_end_utc_s)
    covered = max(0.0, kept_end - kept_start)
    head = max(0.0, kept_start - video_t0_utc_s)
    tail = max(0.0, video_end_utc_s - kept_end)
    total = max(0.0, video_dur - covered)
    pct_kept = (covered / video_dur * 100.0) if video_dur > 0 else 0.0
    return head, tail, total, pct_kept


# =============================================================================
# Local boot->UTC fallback from measurements Fix rows
# =============================================================================
#
# The canonical bridge lives in time_sync (session.txt, else a measurements
# *Raw*-row fallback). When BOTH are unavailable — e.g. the DAY14 "dodge" Cell
# captures, whose session.txt is empty AND whose Raw rows carry a 0
# ChipsetElapsedRealtimeNanos — the measurements *Fix* rows still pair
# UnixTimeMillis with elapsedRealtimeNanos (boottime). We fit that bridge here,
# READ-ONLY, as a last resort so the viewer can still place the media timeline.


def boot_utc_pairs_from_fix_rows(path: Path) -> List[Tuple[float, float]]:
    """Extract ``(boottime_ns, UTC_s)`` from ``# Fix`` rows of measurements.

    Header (canonical):
        Fix,Provider,Lat,Lon,Alt,Speed,Acc,Bearing,UnixTimeMillis,...,
        elapsedRealtimeNanos,...                                  (col 8 + col 11)

    Provider filtering: only the ``reference`` provider's ``UnixTimeMillis`` is
    Signal-derived; ``network``/``fused`` rows can carry the (possibly
    unsynchronised) SYSTEM clock, seconds off from true UTC. When any reference
    rows are present, non-reference rows are excluded from the bridge; when none
    are, all providers are used (better than nothing, robust fit downstream).

    ACCURACY NOTE (validated on SM-S901B day14/s21 sessions that also carry a
    recording_*.txt ground truth): ``Location.getElapsedRealtimeNanos()`` is
    stamped ~107-140 ms AFTER the fix epoch that ``UnixTimeMillis`` refers to
    (fix delivery latency), so an anchor fit from these pairs maps
    boottime -> UTC ~0.1-0.15 s EARLY. Use only as a last resort and warn.
    """
    pairs_gps: List[Tuple[float, float]] = []
    pairs_all: List[Tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.startswith("Fix,"):
                continue
            parts = raw.rstrip("\n").split(",")
            if len(parts) < 12:
                continue
            try:
                unix_ms = float(parts[8])
                boot_ns = float(parts[11])
            except (ValueError, IndexError):
                continue
            if not (boot_ns > 0) or not (unix_ms > 0):
                continue
            pair = (boot_ns, unix_ms / 1e3)
            pairs_all.append(pair)
            if parts[1].strip().lower() == "gps":
                pairs_gps.append(pair)
    return pairs_gps if len(pairs_gps) >= 2 else pairs_all


def _fit_anchor_local(pairs: List[Tuple[float, float]]) -> Optional[TimeAnchor]:
    """Robust boot->UTC fit (delegates to time_sync's MAD-rejecting OLS).

    Previously a plain (non-robust) OLS: one-sided scheduling outliers in the
    pairs pulled the intercept several ms off the quiet-majority line. Now
    identical to the fit every other bridge uses.
    """
    if len(pairs) < 2:
        return None
    from .time_sync import fit_time_anchor_from_pairs
    try:
        return fit_time_anchor_from_pairs(iter(pairs), robust=True)
    except ValueError:
        return None


def resolve_boot_anchor(
    *,
    recording_txt: Optional[Path],
    measurements_txt: Optional[Path],
    notes: List[str],
) -> Tuple[Optional[TimeAnchor], Optional[str]]:
    """Resolve a Signal boot->UTC :class:`TimeAnchor`, session any fallback.

    Order: session.txt (or measurements Raw fallback, via time_sync if the
    helper exists) -> local measurements Fix-row fit. Returns ``(anchor, source)``;
    ``(None, None)`` when no bridge is recoverable. Never edits time_sync.
    """
    # 1) Canonical helper from time_sync (the other agent may add/extend it).
    try:
        from . import time_sync as _ts
        helper = getattr(_ts, "fit_time_anchor_with_fallback", None)
        if helper is not None and recording_txt is not None:
            try:
                anchor, source = helper(
                    Path(recording_txt),
                    Path(measurements_txt) if measurements_txt else None,
                )
                return anchor, source
            except Exception as exc:  # documented empty-anchor case, etc.
                notes.append(f"time_sync bridge unavailable ({type(exc).__name__}).")
    except Exception:
        pass

    # 2) Local Fix-row fallback (read-only).
    if measurements_txt is not None and Path(measurements_txt).is_file():
        pairs = boot_utc_pairs_from_fix_rows(Path(measurements_txt))
        anchor = _fit_anchor_local(pairs)
        if anchor is not None:
            notes.append(
                f"GNSS boot->UTC derived locally from {len(pairs)} measurements "
                "Fix rows (recording.txt empty and Raw ChipsetElapsedRealtimeNanos=0). "
                "WARNING: Fix-row bridge carries the fix delivery latency -- "
                "absolute UTC is typically ~0.1-0.15 s EARLY; audio<->video "
                "relative sync is unaffected."
            )
            return anchor, "measurements-fix-fallback"

    notes.append("no GNSS boot->UTC bridge recoverable; sync/trim unavailable.")
    return None, None


# =============================================================================
# Top-level entry point
# =============================================================================


def compute_capture_diag(
    session_dir: Optional[Path] = None,
    *,
    mp4: Optional[Path] = None,
    recording_txt: Optional[Path] = None,
    measurements_txt: Optional[Path] = None,
    audio_wav: Optional[Path] = None,
    audio_anchor_txt: Optional[Path] = None,
    video_anchor_txt: Optional[Path] = None,
    capture_meta_json: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    pos_file: Optional[Path] = None,
    frame_times: Optional[Path] = None,
    max_gap_s: float = 5.0,
    ffprobe: Optional[str] = None,
) -> CaptureDiag:
    """Compute :class:`CaptureDiag` for a session.

    Either pass ``session_dir`` (files are resolved via
    :meth:`RawInputs.from_folder`) or pass the individual file paths. Explicit
    paths override anything resolved from ``session_dir``. Every step is
    optional: a missing input downgrades the relevant fields to ``None`` and
    appends a note, never raises.

    ``chop_video_anchor`` — when diagnosing a cut ("segment") clip, pass the
    segment's own ``*.video_anchor.txt``. Its ``min(bootNs)`` is the clip's real
    sample-0 boottime and WINS over the parent ``capture_meta``
    ``video_t0_boottime_ns`` (which is the ORIGINAL full session's sample 0
    and would misplace every media<->Signal/stream offset and cut by however
    far into the session the segment starts). See
    ``docs/findings/segment-time-contract.md``.
    """
    diag = CaptureDiag()
    notes = diag.notes

    # ---- Resolve files ------------------------------------------------------
    if session_dir is not None:
        try:
            from .pipeline import RawInputs
            ri = RawInputs.from_folder(Path(session_dir))
            mp4 = mp4 or ri.recording_mp4
            recording_txt = recording_txt or ri.recording_txt
            measurements_txt = measurements_txt or ri.measurements_txt
            audio_wav = audio_wav or ri.audio_wav
            audio_anchor_txt = audio_anchor_txt or ri.audio_anchor_txt
            video_anchor_txt = video_anchor_txt or ri.video_anchor_txt
            capture_meta_json = capture_meta_json or ri.capture_meta_json
        except Exception as exc:
            notes.append(f"session resolution failed ({type(exc).__name__}: {exc}).")

    # ---- Capture meta -------------------------------------------------------
    capture_meta_raw: Optional[Dict[str, Any]] = None
    video_t0_boottime_ns: Optional[float] = None
    if capture_meta_json is not None and Path(capture_meta_json).is_file():
        try:
            from .capture_meta import parse_capture_meta
            cm = parse_capture_meta(Path(capture_meta_json))
            capture_meta_raw = cm.raw
            video_t0_boottime_ns = (
                float(cm.video_t0_boottime_ns)
                if cm.video_t0_boottime_ns is not None
                else None
            )
        except Exception as exc:
            notes.append(f"capture_meta parse failed ({type(exc).__name__}).")

    # ---- Media stats (the probe tool) ---------------------------------------------
    probe: Optional[VideoProbe] = None
    if mp4 is not None and Path(mp4).is_file():
        try:
            probe = probe_video(Path(mp4), ffprobe=ffprobe)
            diag.width = probe.width
            diag.height = probe.height
            diag.fps = probe.fps
            diag.duration_s = probe.duration_s
            diag.file_size_bytes = probe.file_size_bytes
            if probe.duration_s and probe.duration_s > 0 and probe.file_size_bytes:
                mb = probe.file_size_bytes / (1024.0 * 1024.0)
                minutes = probe.duration_s / 60.0
                diag.mb_per_min = mb / minutes if minutes > 0 else None
        except FileNotFoundError as exc:
            notes.append(f"ffprobe unavailable; video stats skipped ({exc}).")
        except Exception as exc:
            notes.append(f"ffprobe failed ({type(exc).__name__}: {exc}).")
    else:
        notes.append("no mp4 found; video stats unavailable.")

    # ---- Focal length -------------------------------------------------------
    focal, focal_src, focal_notes = extract_focal_length(probe, capture_meta_raw)
    diag.focal_length = focal
    diag.focal_source = focal_src if focal_src is not None else "unavailable"
    notes.extend(focal_notes)

    # ---- Signal boot->UTC bridge ---------------------------------------------
    boot_anchor, _bridge_src = resolve_boot_anchor(
        recording_txt=recording_txt,
        measurements_txt=measurements_txt,
        notes=notes,
    )

    # ---- Media timeline t0 in boot/UTC -------------------------------------
    # Canonical resolution (frame_time helper): a segment's own video_anchor
    # min(bootNs) WINS over the parent capture_meta t0; else capture_meta;
    # else the session video_anchor min(bootNs). Using min() rather than the
    # first row makes the t0 robust to unsorted anchor rows.
    from .frame_time import resolve_video_t0_boottime_ns
    # For span/cut purposes the segment's anchor rows (when given) ARE the media.
    video_pairs: List[Tuple[float, float]] = []
    if chop_video_anchor is not None and Path(chop_video_anchor).is_file():
        video_pairs = parse_video_anchor(Path(chop_video_anchor))
    elif video_anchor_txt is not None and Path(video_anchor_txt).is_file():
        video_pairs = parse_video_anchor(Path(video_anchor_txt))
    resolved_t0 = resolve_video_t0_boottime_ns(
        capture_meta=capture_meta_json,
        video_anchor=video_anchor_txt,
        chop_video_anchor=chop_video_anchor,
        log=notes.append,
    )
    if resolved_t0 is not None:
        video_t0_boottime_ns = resolved_t0
    elif video_t0_boottime_ns is None and video_pairs:
        # Defensive fallback (the resolver already covers these sources).
        video_t0_boottime_ns = min(p[1] for p in video_pairs)

    # ---- Stream sync (reuse audio_sync) -------------------------------------
    audio_start_utc: Optional[float] = None
    if (
        boot_anchor is not None
        and audio_wav is not None and Path(audio_wav).is_file()
        and audio_anchor_txt is not None and Path(audio_anchor_txt).is_file()
    ):
        try:
            from .audio_sync import analyze_audio
            res = analyze_audio(
                wav=Path(audio_wav),
                audio_anchor=Path(audio_anchor_txt),
                boot_anchor=boot_anchor,
                video_t0_boottime_ns=video_t0_boottime_ns,
            )
            stats = res.stats
            diag.audio_gnss_drift_ppm = (
                stats.audio_drift_ppm if math.isfinite(stats.audio_drift_ppm) else None
            )
            audio_start_utc = stats.audio_start_utc_s
            # stream -> Signal offset: where the stream timeline t0 sits relative to
            # the Signal coverage start (how far into Signal the stream begins).
            # Reported as the stream-start UTC minus media-start UTC when both
            # exist, else 0 (stream sample 0 IS the Signal-referenced instant).
            notes.extend(f"[audio] {n}" for n in stats.notes)
        except Exception as exc:
            notes.append(f"audio sync failed ({type(exc).__name__}: {exc}).")
    else:
        if audio_wav is None or audio_anchor_txt is None:
            notes.append("no audio wav/anchor; audio->GNSS sync unavailable.")

    # ---- Media sync: offset + drift vs Signal --------------------------------
    video_t0_utc: Optional[float] = None
    if boot_anchor is not None and video_t0_boottime_ns is not None:
        video_t0_utc = boot_anchor.boottime_to_utc_s(float(video_t0_boottime_ns))
        # Media<->Signal drift: the media sample clock rate vs the Signal boot rate.
        # The Signal anchor maps boot(ns)->UTC(s) with slope ~1e-9 (1+drift). The
        # media sample clock advances at video_period ns/sample in the SAME boot
        # timebase, so its rate relative to Signal UTC is the Signal bridge drift
        # PLUS any deviation of the media clock from boot. video_anchor bootNs
        # IS boot, so the only media<->Signal rate error is the Signal bridge drift.
        diag.video_gnss_drift_ppm = (
            boot_anchor.drift_ppm if math.isfinite(boot_anchor.drift_ppm) else None
        )
        # Offset: stream-vs-media start skew in UTC when stream present, else 0.
        if audio_start_utc is not None:
            diag.video_gnss_offset_ms = 0.0  # media t0 is our Signal reference
            diag.audio_gnss_offset_ms = (audio_start_utc - video_t0_utc) * 1e3
        else:
            diag.video_gnss_offset_ms = 0.0
    else:
        notes.append("video t0 or GNSS bridge missing; video->GNSS sync unavailable.")

    # If stream drift is known but offset wasn't set above (no media t0), still
    # report stream offset as 0 relative to Signal (stream sample 0 maps to UTC).
    if diag.audio_gnss_drift_ppm is not None and diag.audio_gnss_offset_ms is None:
        diag.audio_gnss_offset_ms = 0.0

    # ---- Cut ---------------------------------------------------------------
    # Media span in UTC.
    video_end_utc: Optional[float] = None
    if video_t0_utc is not None and diag.duration_s:
        video_end_utc = video_t0_utc + diag.duration_s
    elif video_t0_utc is not None and video_pairs and boot_anchor is not None:
        # max() mirrors the min()-based t0: robust to unsorted anchor rows.
        video_end_utc = boot_anchor.boottime_to_utc_s(max(p[1] for p in video_pairs))

    # Coverage window (prefer .pos; else Signal-anchor span via Fix rows).
    coverage: Optional[Tuple[float, float]] = None
    if pos_file is not None and Path(pos_file).is_file():
        try:
            from .parsers import parse_rtkpos
            pos_rows = parse_rtkpos(Path(pos_file))
            coverage = _pos_coverage_utc(pos_rows, max_gap_s=max_gap_s)
            if coverage is None:
                notes.append("pos file had no usable epochs; trim falls back to anchor span.")
        except Exception as exc:
            notes.append(f"pos parse failed ({type(exc).__name__}); trim falls back.")
    if coverage is None and boot_anchor is not None and measurements_txt is not None:
        # Signal-anchor coverage = UTC span of the Fix rows used for the bridge.
        try:
            fix_pairs = boot_utc_pairs_from_fix_rows(Path(measurements_txt))
            if fix_pairs:
                utcs = [u for _, u in fix_pairs]
                coverage = (min(utcs), max(utcs))
        except Exception:
            pass

    if (
        video_t0_utc is not None
        and video_end_utc is not None
        and coverage is not None
    ):
        head, tail, total, pct = compute_trim(
            video_t0_utc_s=video_t0_utc,
            video_end_utc_s=video_end_utc,
            coverage_start_utc_s=coverage[0],
            coverage_end_utc_s=coverage[1],
        )
        diag.head_trim_s = head
        diag.tail_trim_s = tail
        diag.total_trim_s = total
        diag.pct_kept = pct
    else:
        notes.append("trim unavailable (need video span + GNSS/PPK coverage).")

    return diag


__all__ = [
    "CaptureDiag",
    "VideoProbe",
    "compute_capture_diag",
    "probe_video",
    "parse_ffprobe_json",
    "extract_focal_length",
    "parse_video_anchor",
    "video_frame_period_ns",
    "compute_trim",
    "boot_utc_pairs_from_fix_rows",
    "resolve_boot_anchor",
]
