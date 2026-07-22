"""Tiny orchestration helpers shared by the GUI and CLI entrypoints.

This module is deliberately thin: each *stage* lives in its own module under
``data_pipeline.stages`` and exposes a pure-Python ``run(...)`` function.
The GUI calls those ``run`` functions on background threads and forwards their
log lines to the on-screen log widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class RawInputs:
    """Resolved file handles for a single capture session.

    Supported session layouts are detected automatically; callers use the
    resolved handles and need not know which layout was matched.
    """

    measurements_txt: Path
    recording_txt: Path
    recording_mp4: Optional[Path]
    sensors_txt: Path
    capture_meta_json: Optional[Path] = None
    audio_anchor_txt: Optional[Path] = None
    # Per-sample boot->pts anchor written next to the container file by the platform
    # capture pipeline (``recording_<ts>.video_anchor.txt``). Optional: when
    # present it lets sample timing use the per-sample anchor instead of (or in
    # addition to) the single ``video_t0_boottime_ns`` from capture_meta.json.
    video_anchor_txt: Optional[Path] = None
    # Detected capture format: "old" (legacy logger-style: recording_*.txt
    # carries video_ns<->UTC pairs, no capture_meta.json, sample timing via
    # media PTS) or "new" (current app: capture_meta.json present, recording_*.txt
    # carries ABSOLUTE boottime_ns<->UTC, anchor_format=2, optional per-sample
    # video_anchor.txt). Auto-detected by :meth:`from_folder`.
    capture_format: str = "old"
    # The ``anchor_format`` value read from capture_meta.json (e.g. 2 for the
    # absolute-boottime layout). None for the old format / when unspecified.
    anchor_format: Optional[int] = None
    # Session stream session (``audio_<ts>.wav``), used by the sync player for
    # stream playback + feature map. Optional (old-format sessions have none).
    audio_wav: Optional[Path] = None
    # --- Cut-clip ("segment") support -----------------------------------
    # A segment session is a ``chop_*`` subdir holding a cut container file whose PTS
    # are rebased to zero, plus its own per-sample video_anchor.txt and a
    # ``*.chop_meta.json`` manifest. Stream/Signal/sensors stay in the PARENT
    # session dir. When a segment is detected: ``recording_mp4`` and
    # ``video_anchor_txt`` point at the segment's files, everything else at the
    # parent's, and ``chop_video_anchor`` must be forwarded to coordinate output so the
    # sample-0 boottime comes from the segment anchor (NOT capture_meta's
    # original-session ``video_t0_boottime_ns``).
    chop_meta_json: Optional[Path] = None
    is_chop: bool = False
    chop_video_anchor: Optional[Path] = None

    @property
    def is_boottime(self) -> bool:
        return self.capture_meta_json is not None

    @property
    def is_new_format(self) -> bool:
        """True when this session is the current (boottime/capture_meta) app format."""
        return self.capture_format == "new"

    @property
    def has_per_frame_anchor(self) -> bool:
        """True when a per-sample ``recording_*.video_anchor.txt`` was found."""
        return self.video_anchor_txt is not None

    @classmethod
    def from_folder(cls, folder: Path) -> "RawInputs":
        """Resolve the session files inside ``folder``.

        Tolerant by design: the current The platform app writes ~8 files per
        session (session/measurements/sensors .txt, session .container file,
        audio_*.wav, audio_anchor_*.txt, capture_meta.json, and a per-sample
        recording_*.video_anchor.txt). Only the three core .txt logs are
        required; everything else is optional and the presence of any extra
        files (audio_*.wav, video_anchor.txt, ...) must never cause a failure.
        """
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {folder}")

        # --- Cut-clip ("segment") detection ---------------------------------
        # ``folder`` may be a segment dir itself, or a parent session holding
        # exactly one ``chop_*`` subdir. In that case the container file + per-sample
        # media anchor come from the segment while stream/Signal/sensors resolve
        # from the parent session (via chop_meta ``source_*`` names, falling
        # back to the usual globs on the parent).
        chop = _detect_chop(folder)
        if chop is not None:
            return cls._from_chop(folder, *chop)

        # The per-sample anchor is named ``recording_<ts>.video_anchor.txt`` —
        # it collides with the ``recording_*.txt`` glob for the timing file.
        # Detect it first and exclude it from the session-log match so a
        # session that ships both does not trip the "multiple matches" guard.
        video_anchor = _pick_optional(folder, "recording_*.video_anchor.txt")

        meas = _pick_one(folder, "measurements_*.txt")
        rec = _pick_one(
            folder, "recording_*.txt",
            exclude=("*.video_anchor.txt",),
        )
        sens = _pick_one(folder, "sensors_*.txt")
        capture_meta = _pick_optional(folder, "capture_meta.json")
        audio_anchor = _pick_optional(folder, "audio_anchor_*.txt")
        audio_wav = _pick_optional(folder, "audio_*.wav")

        mp4 = _pick_optional(folder, "recording_*.mp4")
        if mp4 is None and capture_meta is not None:
            try:
                from .capture_meta import parse_capture_meta
                cm = parse_capture_meta(capture_meta)
                if cm.video_name:
                    cand = folder / cm.video_name
                    if cand.is_file():
                        mp4 = cand
            except Exception:
                pass
        if mp4 is None:
            mp4 = _pick_optional(folder, "*.mp4")

        # --- Old vs new capture-format auto-detection ----------------------
        # The presence of capture_meta.json (and especially anchor_format >= 2)
        # is the primary signal for the current app format. When it is absent
        # we treat the session as the legacy (media-PTS) layout. As a defensive
        # fallback we also sniff the per-sample video_anchor.txt, which only the
        # new pipeline writes.
        capture_format = "old"
        anchor_format: Optional[int] = None
        if capture_meta is not None:
            try:
                from .capture_meta import parse_capture_meta
                cm = parse_capture_meta(capture_meta)
                anchor_format = cm.anchor_format
                # capture_meta present => current app format. anchor_format may
                # be None on early builds; the boottime offset still routes it.
                capture_format = "new"
            except Exception:
                # A malformed manifest must not crash detection; the boottime
                # path in coordinate output tolerates a bad manifest and falls back.
                capture_format = "new"
        elif video_anchor is not None:
            # No manifest but a per-sample anchor exists -> still the new layout.
            capture_format = "new"

        return cls(
            meas, rec, mp4, sens, capture_meta, audio_anchor, video_anchor,
            capture_format=capture_format, anchor_format=anchor_format,
            audio_wav=audio_wav,
        )

    @classmethod
    def _from_chop(
        cls,
        folder: Path,
        chop_dir: Path,
        chop_meta_path: Path,
        chop_mp4: Path,
        chop_anchor: Path,
    ) -> "RawInputs":
        """Resolve a cut-clip session.

        The segment's own container file/video_anchor are taken explicitly (never globbed
        from the parent, so the parent's full session never trips the
        multiple-match guard); everything else comes from the PARENT session
        dir, preferring the filenames logged in ``chop_meta.json``
        (``source_gnss``, ``source_audio_wav``, ...) and falling back to the
        standard glob patterns.
        """
        import json

        parent = folder.parent if chop_dir == folder else folder
        try:
            meta = json.loads(chop_meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            # A malformed segment manifest must not kill resolution — the
            # parent-side globs below still find the session files.
            meta = {}

        def _src(key: str) -> Optional[Path]:
            name = meta.get(key)
            if isinstance(name, str) and name:
                cand = parent / name
                if cand.is_file():
                    return cand
            return None

        meas = _pick_one(parent, "measurements_*.txt")
        rec = _src("source_gnss") or _pick_one(
            parent, "recording_*.txt", exclude=("*.video_anchor.txt",),
        )
        sens = _pick_one(parent, "sensors_*.txt")
        capture_meta = (
            _src("source_capture_meta")
            or _pick_optional(parent, "capture_meta.json")
        )
        audio_anchor = (
            _src("source_audio_anchor")
            or _pick_optional(parent, "audio_anchor_*.txt")
        )
        audio_wav = _src("source_audio_wav") or _pick_optional(
            parent, "audio_*.wav",
        )

        anchor_format: Optional[int] = None
        if capture_meta is not None:
            try:
                from .capture_meta import parse_capture_meta
                anchor_format = parse_capture_meta(capture_meta).anchor_format
            except Exception:
                pass

        return cls(
            meas, rec, chop_mp4, sens, capture_meta, audio_anchor,
            chop_anchor,
            capture_format="new", anchor_format=anchor_format,
            audio_wav=audio_wav,
            chop_meta_json=chop_meta_path, is_chop=True,
            chop_video_anchor=chop_anchor,
        )


def _detect_chop(
    folder: Path,
) -> Optional[tuple[Path, Path, Path, Path]]:
    """Detect a cut-clip ("segment") layout under ``folder``.

    Returns ``(chop_dir, chop_meta_json, chop_mp4, chop_video_anchor)`` when
    ``folder`` itself, or exactly one ``chop_*`` subdirectory of it, contains
    all three of: ``*.chop_meta.json``, a ``chop_*.container file`` and a
    ``chop_*.video_anchor.txt``. Returns ``None`` otherwise (including the
    ambiguous multiple-segment-subdir case, which falls back to the normal
    session resolution).

    ``folder`` ITSELF is only treated as a segment dir when it does NOT also
    hold the full-session files (``measurements_*.txt``): if a user drops the
    segment files directly into a session dir, resolving that dir AS the segment
    dir would set the parent to ``folder.parent`` and silently glob
    Signal/stream from the GRANDPARENT — the wrong session's data. A real segment
    dir written by the source app never contains ``measurements_*.txt``.
    """

    def _resolve(chop_dir: Path) -> Optional[tuple[Path, Path, Path, Path]]:
        metas = sorted(chop_dir.glob("*.chop_meta.json"))
        if not metas:
            return None
        meta = metas[0]
        base = meta.name[: -len(".chop_meta.json")]
        mp4: Optional[Path] = chop_dir / f"{base}.mp4"
        if not mp4.is_file():
            mp4s = sorted(chop_dir.glob("chop_*.mp4"))
            mp4 = mp4s[0] if mp4s else None
        anchor: Optional[Path] = chop_dir / f"{base}.video_anchor.txt"
        if not anchor.is_file():
            anchors = sorted(chop_dir.glob("chop_*.video_anchor.txt"))
            anchor = anchors[0] if anchors else None
        if mp4 is None or anchor is None:
            return None
        return (chop_dir, meta, mp4, anchor)

    # Only consider ``folder`` itself a segment dir when the full-session files
    # are absent from it (see docstring: grandparent mis-resolution guard).
    if not any(folder.glob("measurements_*.txt")):
        hit = _resolve(folder)
        if hit is not None:
            return hit
    try:
        subs = sorted(
            d for d in folder.iterdir()
            if d.is_dir() and d.name.startswith("chop_")
        )
    except OSError:
        return None
    hits = [h for h in (_resolve(d) for d in subs) if h is not None]
    if len(hits) == 1:
        return hits[0]
    return None


def _pick_one(
    folder: Path, pattern: str, *, exclude: Iterable[str] = (),
) -> Path:
    """Resolve exactly one file matching ``pattern`` in ``folder``.

    ``exclude`` is a list of glob patterns whose matches are removed from the
    candidate set (used so ``recording_*.txt`` does not also catch the
    per-sample ``recording_*.video_anchor.txt`` sidecar).

    On a missing required file the error lists every file that IS present so
    the user can see at a glance what the folder actually contains.
    """
    matches = sorted(folder.glob(pattern))
    if exclude:
        skip = set()
        for ex in exclude:
            skip.update(folder.glob(ex))
        matches = [m for m in matches if m not in skip]
    if not matches:
        present = sorted(p.name for p in folder.iterdir() if p.is_file())
        present_str = ", ".join(present) if present else "(no files)"
        raise FileNotFoundError(
            f"No file matching {pattern!r} in {folder}. "
            f"Files present: {present_str}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(
            f"Multiple files match {pattern!r} in {folder}: {names}. "
            "Please keep only one session per RAW folder."
        )
    return matches[0]


def _pick_optional(folder: Path, pattern: str) -> Optional[Path]:
    """Like :func:`_pick_one` but returns ``None`` when nothing matches.

    On multiple matches returns the first by sorted name (lenient — these are
    optional auxiliary files, not the core session inputs).
    """
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


# A logger callback: ``log("some line")``. Defaults to ``print``.
LogFn = Callable[[str], None]


def _safe_print(msg: str) -> None:
    """Print to stdout without crashing on encoding-incompatible characters.

    On non-UTF-8 Windows consoles (e.g. Hebrew cp1255), unicode glyphs in
    log strings (sigma, arrows, math symbols) raise UnicodeEncodeError
    mid-pipeline. Fall back to ASCII replacement so the run completes;
    the user still sees the diagnostic, just with '?' placeholders for
    untranslatable glyphs.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        import sys
        enc = (sys.stdout.encoding or "ascii")
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def make_logger(callback: Optional[LogFn]) -> LogFn:
    """Return ``callback`` if provided, else an encoding-safe ``print``."""
    if callback is None:
        return _safe_print
    return callback


def log_lines(log: LogFn, lines: Iterable[str]) -> None:
    for line in lines:
        log(line)
