"""Sample coordinate export on the stream-zero timeline.

Places every extracted media sample on a timeline whose ZERO is the Stream
session start (stream sample 0), names each sample file by its offset in
seconds from that origin ("0 seconds onward"), and emits a CSV mapping each
sample name to its Signal coordinates for an external tool ("sample coordinate
export").

Timeline math (all in CLOCK_BOOTTIME nanoseconds, then converted):

    audio_start_boot_ns = AudioAnchor.frame_to_boot_ns(0.0)
    frame_boot_ns       = video_t0_boot_ns + t_video_s * 1e9
    t_audio_s           = (frame_boot_ns - audio_start_boot_ns) / 1e9
    utc_s               = TimeAnchor.boottime_to_utc_s(frame_boot_ns)

``t_video_s`` is the sample's true source PTS recovered by
:mod:`data_pipeline.stages.samples` (the external converter showinfo — sub-millisecond).
``video_t0_boot_ns`` resolution mirrors :mod:`data_pipeline.stages.georef`:

* cut ("segment") session : min bootNs of the segment's own video_anchor.txt
  (NEVER capture_meta's original-session ``video_t0_boottime_ns``);
* full session             : min bootNs of ``recording_*.video_anchor.txt``,
  falling back to capture_meta ``video_t0_boottime_ns``.

Samples whose ``t_audio_s`` is negative (captured before the stream started)
are DROPPED so the export starts at the stream 0.0 origin. Kept samples are
COPIED (the extraction output is never mutated) into an export directory
under names like ``0012.346.png`` (zero-padded seconds, millisecond
precision — lexicographically sortable in time order).

The emitted ``frames_for_external.csv`` uses the Coordinate output-style header
``Image, Latitude, Longitude, Altitude`` (readable by
``frame_compare.load_external_frame_coords`` /
``load_gnss_frame_coords_from_georef``) plus a trailing ``t_audio_s``
traceability column. Coordinates come from linear interpolation of an
The external solver ``.pos`` at each sample's UTC; when no ``.pos`` is given (or a sample
has no bracketing Signal epoch) the coordinate cells are left blank and the
sample is still exported — the coordinate join is optional.

CLI:

    python -m data_pipeline.audio_frame_export --session <dir> --out <dir>
        [--pos solution.pos] [--fps 6] [--format png] [--rotation 0]
        [--max-gap-s 2.0]
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .pipeline import LogFn, RawInputs, make_logger

__all__ = [
    "FrameAudioRow",
    "SessionAnchors",
    "AudioFrameExportResult",
    "frame_name_for_time",
    "resolve_session_anchors",
    "compute_frame_audio_times",
    "write_external_csv",
    "export_frames_from_audio",
    "main",
]

#: Default fallback when neither the WAV header nor capture_meta declares the
#: stream sample rate (only used for the degenerate single-anchor-pair fit).
DEFAULT_AUDIO_RATE_HZ = 48000.0

#: Column header of the external-tool CSV. The first four columns match the
#: Georef.csv convention (``Image, Latitude, Longitude, Altitude``) that the
#: repo's ``frame_compare`` loaders already parse; ``t_audio_s`` is appended
#: for traceability (extra columns are ignored by those loaders).
EXTERNAL_CSV_HEADER = ("Image", "Latitude", "Longitude", "Altitude", "t_audio_s")

#: Filename of the emitted mapping CSV.
EXTERNAL_CSV_NAME = "frames_for_external.csv"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def frame_name_for_time(t_s: float, *, decimals: int = 3, seconds_width: int = 4) -> str:
    """Filename stem for a sample at ``t_s`` seconds after stream start.

    Zero-padded integer seconds (default width 4 => sessions up to 9999 s
    stay lexicographically sortable) with ``decimals`` fractional digits
    (default 3 = millisecond precision), e.g.::

        0.0     -> "0000.000"
        12.3456 -> "0012.346"
        100.0   -> "0100.000"

    Values are half-up rounded at the requested precision. Negative times
    raise ``ValueError`` — pre-stream samples must be dropped before naming
    (a minus sign would break both sorting and the "0 seconds onward" rule).
    """
    if not math.isfinite(t_s):
        raise ValueError(f"t_s must be finite (got {t_s!r})")
    # Round first so values like -0.0004 (numerically negative, 0.000 at
    # millisecond precision) are accepted rather than rejected.
    scale = 10 ** decimals
    units = math.floor(t_s * scale + 0.5)  # half-up, sign-stable
    if units < 0:
        raise ValueError(
            f"t_s is negative ({t_s!r}): pre-audio frames must be dropped "
            "before naming"
        )
    secs, frac = divmod(units, scale)
    return f"{secs:0{seconds_width}d}.{frac:0{decimals}d}"


# ---------------------------------------------------------------------------
# Session anchor resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionAnchors:
    """Resolved timeline anchors for one capture session.

    ``audio_start_boot_ns`` is the CLOCK_BOOTTIME of stream sample 0 (the
    export's time origin). ``video_t0_boot_ns`` is the CLOCK_BOOTTIME of
    media PTS 0 for the container file that samples are extracted from (the segment's own
    t0 for a cut session). ``boot_anchor`` maps boottime to UTC and may
    be ``None`` when no boot->UTC bridge could be fit (the stream-relative
    timeline still works; only the Signal join is unavailable).
    """

    audio_start_boot_ns: float
    video_t0_boot_ns: float
    boot_anchor: Optional[object] = None  # time_sync.TimeAnchor
    boot_anchor_source: str = ""
    audio_nominal_rate_hz: float = DEFAULT_AUDIO_RATE_HZ
    audio_anchor_n: int = 0
    audio_anchor_rmse_ns: float = 0.0
    is_chop: bool = False


def _wav_sample_rate_hz(wav_path: Optional[Path]) -> Optional[float]:
    """Sample rate from the WAV header (header-only read), or ``None``."""
    if wav_path is None:
        return None
    try:
        import wave

        with wave.open(str(wav_path), "rb") as w:
            rate = float(w.getframerate())
        return rate if rate > 0 else None
    except Exception:
        return None


def _capture_meta_audio_rate_hz(capture_meta_json: Optional[Path]) -> Optional[float]:
    """``audio.sample_rate`` from capture_meta.json, or ``None``."""
    if capture_meta_json is None:
        return None
    try:
        from .capture_meta import parse_capture_meta

        raw = parse_capture_meta(capture_meta_json).raw
        rate = float((raw.get("audio") or {}).get("sample_rate"))
        return rate if rate > 0 else None
    except Exception:
        return None


def _resolve_video_t0_boot_ns(inputs: RawInputs, log: LogFn) -> float:
    """CLOCK_BOOTTIME (ns) of media PTS 0 for the session's container file.

    Segment session: min bootNs of the segment's own video_anchor.txt — REQUIRED
    (falling back to the full-session t0 would map every sample minutes
    early). Full session: min bootNs of the per-sample video_anchor.txt,
    else capture_meta ``video_t0_boottime_ns``.
    """
    from .stages.georef import _first_boottime_ns_from_video_anchor

    if inputs.is_chop:
        if inputs.chop_video_anchor is None:
            raise ValueError(
                "Chop session without a chop video_anchor.txt; cannot "
                "recover the chop frame-0 boottime."
            )
        t0 = _first_boottime_ns_from_video_anchor(inputs.chop_video_anchor)
        if t0 is None:
            raise ValueError(
                f"Chop video anchor {inputs.chop_video_anchor} is unreadable/"
                "empty; refusing the full-session t0 fallback (it would map "
                "every chop frame minutes early)."
            )
        log(f"[audio-export] video t0 from chop anchor: {t0:.0f} ns")
        return float(t0)

    if inputs.video_anchor_txt is not None:
        t0 = _first_boottime_ns_from_video_anchor(inputs.video_anchor_txt)
        if t0 is not None:
            log(f"[audio-export] video t0 from {Path(inputs.video_anchor_txt).name}: "
                f"{t0:.0f} ns")
            return float(t0)

    if inputs.capture_meta_json is not None:
        try:
            from .capture_meta import parse_capture_meta

            cm = parse_capture_meta(inputs.capture_meta_json)
            if cm.video_t0_boottime_ns is not None:
                log(f"[audio-export] video t0 from capture_meta: "
                    f"{cm.video_t0_boottime_ns} ns")
                return float(cm.video_t0_boottime_ns)
        except Exception as e:
            log(f"[audio-export] capture_meta parse failed ({e})")

    raise ValueError(
        "Could not resolve the video frame-0 boottime: no usable per-frame "
        "video_anchor.txt and no capture_meta video_t0_boottime_ns. The "
        "audio-relative export needs a boottime-format session."
    )


def resolve_session_anchors(
    session_dir: Path,
    *,
    inputs: Optional[RawInputs] = None,
    need_utc: bool = True,
    log: Optional[LogFn] = None,
) -> SessionAnchors:
    """Resolve the stream-0 origin, media t0 and boot->UTC anchor for a session.

    ``inputs`` may be passed to skip re-resolving the folder. When
    ``need_utc`` is False a failed boot->UTC fit is tolerated (``boot_anchor``
    comes back ``None``); when True it raises.
    """
    log_ = make_logger(log)
    if inputs is None:
        inputs = RawInputs.from_folder(Path(session_dir))

    # --- Stream origin -----------------------------------------------------
    if inputs.audio_anchor_txt is None:
        raise ValueError(
            f"No audio_anchor_*.txt found in {session_dir}: the audio-zero "
            "timeline needs the audio anchor to locate audio sample 0."
        )
    from .audio_sync import fit_audio_anchor, parse_audio_anchor

    pairs = parse_audio_anchor(inputs.audio_anchor_txt)
    rate = (
        _wav_sample_rate_hz(inputs.audio_wav)
        or _capture_meta_audio_rate_hz(inputs.capture_meta_json)
        or DEFAULT_AUDIO_RATE_HZ
    )
    audio_anchor = fit_audio_anchor(pairs, nominal_rate_hz=rate, robust=True)
    audio_start_boot_ns = float(audio_anchor.frame_to_boot_ns(0.0))
    log_(
        f"[audio-export] audio anchor fit: n={audio_anchor.n} "
        f"(rejected {audio_anchor.n_rejected}) rmse={audio_anchor.rmse_ns / 1e6:.3f} ms "
        f"drift={audio_anchor.rate_drift_ppm:+.1f} ppm; "
        f"audio_start_boot_ns={audio_start_boot_ns:.0f}"
    )

    # --- Media t0 ----------------------------------------------------------
    video_t0_boot_ns = _resolve_video_t0_boot_ns(inputs, log_)

    # --- Boot -> UTC -------------------------------------------------------
    boot_anchor = None
    source = ""
    try:
        from .time_sync import fit_time_anchor_with_fallback

        boot_anchor, source = fit_time_anchor_with_fallback(
            inputs.recording_txt, inputs.measurements_txt
        )
        log_(f"[audio-export] boot->UTC anchor from {source}")
        if source == "measurements-fix-fallback":
            log_(
                "[audio-export] WARN: Fix-row bridge includes GNSS fix "
                "delivery latency (~0.10-0.15 s EARLY absolute UTC); the "
                "audio-relative timeline is unaffected."
            )
    except Exception as e:
        if need_utc:
            raise
        log_(f"[audio-export] boot->UTC anchor unavailable ({e}); "
             "UTC/GNSS join disabled")

    return SessionAnchors(
        audio_start_boot_ns=audio_start_boot_ns,
        video_t0_boot_ns=video_t0_boot_ns,
        boot_anchor=boot_anchor,
        boot_anchor_source=source,
        audio_nominal_rate_hz=rate,
        audio_anchor_n=audio_anchor.n,
        audio_anchor_rmse_ns=audio_anchor.rmse_ns,
        is_chop=inputs.is_chop,
    )


# ---------------------------------------------------------------------------
# Pure timeline math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameAudioRow:
    """One extracted sample placed on the stream-zero timeline."""

    image: str          # source image filename (from extraction)
    t_video_s: float    # true source PTS, seconds
    t_audio_s: float    # seconds from stream sample 0 (negative = pre-stream)
    utc_s: Optional[float] = None  # absolute UTC, None when no boot->UTC anchor

    @property
    def pre_audio(self) -> bool:
        """True when the sample was captured before the stream started."""
        return self.t_audio_s < 0.0


def compute_frame_audio_times(
    session_dir: Optional[Path],
    frame_times: Sequence[Tuple[str, float]],
    *,
    audio_start_boot_ns: Optional[float] = None,
    video_t0_boot_ns: Optional[float] = None,
    boot_anchor: Optional[object] = None,
    log: Optional[LogFn] = None,
) -> List[FrameAudioRow]:
    """Place each ``(image, t_video_s)`` on the stream-zero timeline.

    Pure math when the anchors are supplied explicitly (``session_dir`` may
    then be ``None`` — nothing is read from disk):

        frame_boot_ns = video_t0_boot_ns + t_video_s * 1e9
        t_audio_s     = (frame_boot_ns - audio_start_boot_ns) / 1e9
        utc_s         = boot_anchor.boottime_to_utc_s(frame_boot_ns)  # if given

    When ``audio_start_boot_ns`` / ``video_t0_boot_ns`` are omitted they are
    resolved from ``session_dir`` via :func:`resolve_session_anchors` (which
    also supplies ``boot_anchor`` unless one was passed explicitly).

    Returns one :class:`FrameAudioRow` per input row, in input order —
    including pre-stream (negative ``t_audio_s``) samples, identifiable via
    :attr:`FrameAudioRow.pre_audio`; dropping them is the caller's choice.
    """
    if audio_start_boot_ns is None or video_t0_boot_ns is None:
        if session_dir is None:
            raise ValueError(
                "session_dir is required when audio_start_boot_ns / "
                "video_t0_boot_ns are not supplied"
            )
        anchors = resolve_session_anchors(Path(session_dir), need_utc=False, log=log)
        if audio_start_boot_ns is None:
            audio_start_boot_ns = anchors.audio_start_boot_ns
        if video_t0_boot_ns is None:
            video_t0_boot_ns = anchors.video_t0_boot_ns
        if boot_anchor is None:
            boot_anchor = anchors.boot_anchor

    rows: List[FrameAudioRow] = []
    for image, t_video_s in frame_times:
        t_video_s = float(t_video_s)
        frame_boot_ns = float(video_t0_boot_ns) + t_video_s * 1e9
        t_audio_s = (frame_boot_ns - float(audio_start_boot_ns)) / 1e9
        utc_s: Optional[float] = None
        if boot_anchor is not None:
            utc_s = float(boot_anchor.boottime_to_utc_s(frame_boot_ns))
        rows.append(
            FrameAudioRow(
                image=str(image), t_video_s=t_video_s,
                t_audio_s=t_audio_s, utc_s=utc_s,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------


def write_external_csv(
    csv_path: Path,
    entries: Sequence[
        Tuple[str, Optional[Tuple[float, float, float]], float]
    ],
) -> Path:
    """Write the external-tool mapping CSV.

    ``entries`` rows are ``(image_name, coords_or_None, t_audio_s)`` where
    ``coords`` is ``(lat_deg, lon_deg, h_m)``. Header is
    ``Image, Latitude, Longitude, Altitude, t_audio_s`` (Coordinate output-style first
    four columns); samples without coordinates keep blank coordinate cells so
    the time mapping survives even without a Signal join.
    """
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EXTERNAL_CSV_HEADER)
        for image, coords, t_audio_s in entries:
            if coords is None:
                lat_s = lon_s = alt_s = ""
            else:
                lat, lon, h = coords
                lat_s = f"{lat:.9f}"
                lon_s = f"{lon:.9f}"
                alt_s = f"{h:.4f}"
            w.writerow([image, lat_s, lon_s, alt_s, f"{t_audio_s:.6f}"])
    return csv_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFrameExportResult:
    """Summary of one stream-zero sample export."""

    export_dir: Path            # renamed sample copies live here
    csv_path: Path              # frames_for_external.csv
    source_frames_dir: Path     # untouched extraction output
    anchors: SessionAnchors
    n_extracted: int            # samples extracted from the container file
    n_exported: int             # samples kept (t_audio_s >= 0) and copied
    n_dropped_pre_audio: int    # samples before stream 0.0, dropped
    n_with_coords: int          # exported samples that got Signal coordinates
    rows: Tuple[FrameAudioRow, ...]  # kept rows, time-ascending


def _read_frame_times_csv(path: Path) -> List[Tuple[str, float]]:
    """Read ``extracted_frame_times.csv`` (``Image, t_video_s``)."""
    out: List[Tuple[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Image") or "").strip()
            t_raw = (row.get("t_video_s") or "").strip()
            if not name or not t_raw:
                continue
            try:
                out.append((name, float(t_raw)))
            except ValueError:
                continue
    return out


def export_frames_from_audio(
    session_dir: Path,
    out_dir: Path,
    *,
    pos_path: Optional[Path] = None,
    fps: float = 6.0,
    fmt: str = "png",
    rotation: int = 0,
    max_gap_s: float = 2.0,
    name_decimals: int = 3,
    log: Optional[LogFn] = None,
) -> AudioFrameExportResult:
    """Extract samples, rebase them onto the stream-zero timeline and export.

    Steps:

    1. Resolve the session (:class:`RawInputs`), fit the stream anchor
       (stream sample 0 boottime = the origin) and the boot->UTC anchor.
    2. Extract samples via ``stages.samples.run`` (true source PTS) into
       ``out_dir`` (``out_dir/samples`` + ``extracted_frame_times.csv``).
    3. Compute each sample's ``t_audio_s``; DROP samples before stream 0.0.
    4. COPY kept samples into ``out_dir/frames_from_audio/`` named
       ``<t_audio>.<ext>`` per :func:`frame_name_for_time`.
    5. Write ``out_dir/frames_for_external.csv`` (Coordinate output-style columns +
       ``t_audio_s``). With ``pos_path`` the coordinates are interpolated
       from the external solver ``.pos`` at each sample's UTC; without it (or where
       interpolation fails) coordinate cells stay blank.
    """
    log_ = make_logger(log)
    session_dir = Path(session_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = RawInputs.from_folder(session_dir)
    if inputs.recording_mp4 is None:
        raise ValueError(f"No video (recording_*.mp4) found in {session_dir}")

    # Anchors first: fail fast before the (slow) extraction. UTC is only a
    # hard requirement when a .pos join was requested.
    anchors = resolve_session_anchors(
        session_dir, inputs=inputs, need_utc=pos_path is not None, log=log_,
    )

    # --- Extraction (true source PTS via the external converter showinfo) ------------------
    from .stages import frames as frames_stage

    extraction = frames_stage.run(
        video=Path(inputs.recording_mp4),
        out_dir=out_dir,
        fps=fps,
        fmt=fmt,  # type: ignore[arg-type]
        rotation=rotation,  # type: ignore[arg-type]
        log=log_,
    )
    frame_times = _read_frame_times_csv(extraction.frame_times_csv)
    if len(frame_times) != extraction.frame_count:
        log_(
            f"[audio-export] WARN: frame_times rows ({len(frame_times)}) != "
            f"extracted count ({extraction.frame_count})"
        )

    # --- Stream-zero timeline ------------------------------------------------
    all_rows = compute_frame_audio_times(
        session_dir,
        frame_times,
        audio_start_boot_ns=anchors.audio_start_boot_ns,
        video_t0_boot_ns=anchors.video_t0_boot_ns,
        boot_anchor=anchors.boot_anchor,
        log=log_,
    )
    kept = [r for r in all_rows if not r.pre_audio]
    n_dropped = len(all_rows) - len(kept)
    if n_dropped:
        log_(
            f"[audio-export] dropped {n_dropped} pre-audio frame(s) "
            f"(t_audio_s < 0; earliest {min(r.t_audio_s for r in all_rows):+.3f} s)"
        )
    else:
        log_("[audio-export] no pre-audio frames (video starts after audio 0.0)")
    kept.sort(key=lambda r: r.t_audio_s)

    # --- Copy/rename into the export dir ------------------------------------
    export_dir = out_dir / "frames_from_audio"
    export_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    seen: dict = {}
    for r in kept:
        ext = Path(r.image).suffix
        name = frame_name_for_time(r.t_audio_s, decimals=name_decimals) + ext
        if name in seen:
            raise ValueError(
                f"Frame name collision at {name!r}: frames {seen[name]!r} and "
                f"{r.image!r} are closer than {10 ** -name_decimals:g} s on the "
                "audio timeline — raise name_decimals."
            )
        seen[name] = r.image
        names.append(name)

    for r, name in zip(kept, names):
        src = Path(extraction.frames_dir) / r.image
        shutil.copy2(src, export_dir / name)
    log_(f"[audio-export] copied {len(kept)} frame(s) -> {export_dir}")

    # --- Signal join (optional) ------------------------------------------------
    coords_by_idx: List[Optional[Tuple[float, float, float]]] = [None] * len(kept)
    n_with_coords = 0
    if pos_path is not None:
        from .parsers import interp_pos, parse_rtkpos

        pos_rows = parse_rtkpos(Path(pos_path))
        if not pos_rows:
            log_(f"[audio-export] WARN: {pos_path} has no usable epochs; "
                 "coordinates left blank")
        else:
            times = [p.utc_s for p in pos_rows]
            n_no_utc = 0
            for i, r in enumerate(kept):
                if r.utc_s is None:
                    n_no_utc += 1
                    continue
                hit = interp_pos(pos_rows, r.utc_s, max_gap_s, times=times)
                if hit is not None:
                    coords_by_idx[i] = hit
                    n_with_coords += 1
            n_miss = len(kept) - n_with_coords - n_no_utc
            log_(
                f"[audio-export] GNSS join: {n_with_coords}/{len(kept)} frames "
                f"got coordinates ({n_miss} outside/beyond max_gap_s="
                f"{max_gap_s:g}, {n_no_utc} without UTC)"
            )
    else:
        log_("[audio-export] no .pos supplied; emitting the time mapping "
             "with blank coordinates")

    csv_path = write_external_csv(
        out_dir / EXTERNAL_CSV_NAME,
        [
            (name, coords_by_idx[i], kept[i].t_audio_s)
            for i, name in enumerate(names)
        ],
    )
    log_(f"[audio-export] wrote {csv_path}")

    return AudioFrameExportResult(
        export_dir=export_dir,
        csv_path=csv_path,
        source_frames_dir=Path(extraction.frames_dir),
        anchors=anchors,
        n_extracted=len(all_rows),
        n_exported=len(kept),
        n_dropped_pre_audio=n_dropped,
        n_with_coords=n_with_coords,
        rows=tuple(kept),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m data_pipeline.audio_frame_export",
        description=(
            "Export video frames on the audio-zero timeline (frame names = "
            "seconds from audio sample 0) plus a frame->coordinate CSV for "
            "an external tool."
        ),
    )
    ap.add_argument("--session", required=True, type=Path,
                    help="Capture session folder (audio/video/GNSS files).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output folder (frames_from_audio/ + CSV).")
    ap.add_argument("--pos", type=Path, default=None,
                    help="RTKLIB .pos for the GNSS coordinate join (optional).")
    ap.add_argument("--fps", type=float, default=6.0,
                    help="Approximate extraction rate (default 6).")
    ap.add_argument("--format", dest="fmt", default="png",
                    choices=["png", "tiff", "jpeg1"],
                    help="Image format (default png).")
    ap.add_argument("--rotation", type=int, default=0,
                    choices=[0, 90, 180, 270],
                    help="Rotate frames by this many degrees (default 0).")
    ap.add_argument("--max-gap-s", type=float, default=2.0,
                    help="Max GNSS bracketing gap for interpolation "
                         "(default 2.0 s).")
    ap.add_argument("--name-decimals", type=int, default=3,
                    help="Fractional digits in frame names (default 3 = ms).")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = export_frames_from_audio(
        args.session,
        args.out,
        pos_path=args.pos,
        fps=args.fps,
        fmt=args.fmt,
        rotation=args.rotation,
        max_gap_s=args.max_gap_s,
        name_decimals=args.name_decimals,
        log=print,
    )
    print(
        f"[audio-export] done: {res.n_exported} frame(s) exported "
        f"({res.n_dropped_pre_audio} pre-audio dropped), "
        f"{res.n_with_coords} with coordinates.\n"
        f"  frames : {res.export_dir}\n"
        f"  csv    : {res.csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
