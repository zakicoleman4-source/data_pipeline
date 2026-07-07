"""Combine a session's stream with a chosen media, synced on CLOCK_BOOTTIME.

Point it at a session folder. It lists every media it can combine:

* the FULL session   (``recording_*.container file`` in the session root), if present
* every Cut slice  (``chop_*/chop_*.container file`` with a ``*.chop_meta.json``
  sidecar — or the session dir itself when it IS a segment dir)

You pick one; it pulls the matching window out of the session's full WAV and
writes a muxed container file (``-c:v copy -c:a aac -shortest``).

WHY ONE CODE PATH for full and cut — both reduce to "the chosen clip's
first-sample boottime":

* full -> the per-sample ``recording_*.video_anchor.txt`` min bootNs, falling
  back to capture_meta ``media.video_t0_boottime_ns``;
* cut -> the segment's own ``chop_*.video_anchor.txt`` min bootNs (the
  physically-real sample-0 boot; see docs/findings/segment-time-contract.md),
  falling back to chop_meta ``start_boottime_ns`` only if the anchor is
  unreadable.

The stream is then seeked to that same boottime in the full WAV (``-ss``
before ``-i`` when the seek is >= 0, ``-itsoffset`` when the stream starts
after the clip) and ``-shortest`` cuts it to the clip. The stream sample
clock's crystal drift vs the nominal WAV rate is corrected with ``atempo``
when it exceeds 0.5 ppm.

This module reuses the pipeline's verified primitives instead of ad-hoc
re-implementations:

* :func:`data_pipeline.audio_sync.fit_audio_anchor` — robust (MAD-rejecting)
  fit of the stream-sample -> boottime map. The anchor writer's outliers are
  one-sided (late scheduling), so a plain OLS fit is biased several ms; the
  robust fit is not.
* :meth:`data_pipeline.pipeline.RawInputs.from_folder` — session layout
  resolution (WAV / stream anchor / capture_meta / segment detection).
* ``data_pipeline.stages.georef._first_boottime_ns_from_video_anchor`` —
  min-bootNs reader for per-sample media anchors.
* :mod:`data_pipeline.ffmpeg_paths` — env var / vendored / PATH resolution
  of the external converter + the probe tool.

Usage:
    python -m data_pipeline.combine_av SESSION_DIR              # interactive picker
    python -m data_pipeline.combine_av SESSION_DIR --video 2    # menu item N
    python -m data_pipeline.combine_av SESSION_DIR --video full
    python -m data_pipeline.combine_av SESSION_DIR --out out.mp4 [--no-rate] [--dry-run]

The planning layer (:func:`plan_mux`) is pure — it builds the external converter command
and all the numbers without running anything, so it is importable and
testable on a machine without the external converter. Only :func:`run_mux` (and the probe tool
clip-duration probe in the CLI) touch external executables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .audio_sync import AudioAnchor, fit_audio_anchor, parse_audio_anchor
from .pipeline import RawInputs
from .stages.georef import _first_boottime_ns_from_video_anchor

__all__ = [
    "ClipInfo",
    "MuxPlan",
    "audio_start_boottime_ns",
    "clip_first_frame_boot_ns",
    "discover_videos",
    "plan_mux",
    "run_mux",
    "main",
]


# -----------------------------------------------------------------------------
# the external converter / the probe tool resolution (tolerant: planning must work without them)
# -----------------------------------------------------------------------------


def _resolve_exe(name: str) -> str:
    """Resolve the external converter/the probe tool via :mod:`data_pipeline.ffmpeg_paths`, else PATH name.

    Never raises: :func:`plan_mux` must stay usable on a machine without
    the external converter (the command simply carries the bare name and fails at *run* time).
    """
    try:
        from .ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe

        return resolve_ffmpeg() if name == "ffmpeg" else resolve_ffprobe()
    except Exception:
        return name


def _ffprobe_duration_s(path: Path) -> Optional[float]:
    """Container duration of ``path`` in seconds via the probe tool, or None."""
    try:
        out = subprocess.run(
            [
                _resolve_exe("ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nk=1:nw=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Session resolution
# -----------------------------------------------------------------------------


def _resolve_raw(session: Union[str, Path, RawInputs]) -> RawInputs:
    """``RawInputs`` for a session dir (pass-through when already resolved).

    Primary path is :meth:`RawInputs.from_folder`. When the folder lacks the
    three core Signal logs (from_folder raises), fall back to direct globs so
    the muxer still works on a stream+media-only folder.
    """
    if isinstance(session, RawInputs):
        return session
    folder = Path(session)
    try:
        return RawInputs.from_folder(folder)
    except Exception:
        return _fallback_raw(folder)


def _fallback_raw(folder: Path) -> RawInputs:
    """Minimal resolution for folders ``RawInputs.from_folder`` rejects."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    def _opt(pattern: str) -> Optional[Path]:
        hits = sorted(folder.glob(pattern))
        return hits[0] if hits else None

    audio_anchor = _opt("audio_anchor_*.txt")
    wavs = [p for p in sorted(folder.glob("audio_*.wav"))]
    audio_wav = wavs[0] if wavs else None
    missing = Path(folder / "_missing_")
    return RawInputs(
        measurements_txt=missing,
        recording_txt=missing,
        recording_mp4=_opt("recording_*.mp4"),
        sensors_txt=missing,
        capture_meta_json=_opt("capture_meta.json"),
        audio_anchor_txt=audio_anchor,
        video_anchor_txt=_opt("recording_*.video_anchor.txt"),
        capture_format="new",
        audio_wav=audio_wav,
    )


def _nominal_sample_rate_hz(
    capture_meta_json: Optional[Path], wav_path: Optional[Path]
) -> float:
    """Declared stream sample rate: capture_meta ``stream.sample_rate``, else the
    WAV header, else 48000."""
    if capture_meta_json is not None:
        try:
            data = json.loads(Path(capture_meta_json).read_text(encoding="utf-8"))
            rate = (data.get("audio") or {}).get("sample_rate")
            if rate:
                return float(rate)
        except Exception:
            pass
    if wav_path is not None:
        try:
            with wave.open(str(wav_path), "rb") as w:
                if w.getframerate() > 0:
                    return float(w.getframerate())
        except Exception:
            pass
    return 48000.0


def _fit_session_audio(
    session: Union[str, Path, RawInputs],
) -> Tuple[AudioAnchor, RawInputs]:
    """Robust stream-sample -> boottime fit for the session's stream anchor."""
    raw = _resolve_raw(session)
    if raw.audio_anchor_txt is None:
        raise FileNotFoundError(
            "no audio_anchor_*.txt in session; cannot map audio to boottime"
        )
    pairs = parse_audio_anchor(Path(raw.audio_anchor_txt))
    if not pairs:
        raise ValueError(
            f"audio anchor {raw.audio_anchor_txt} has no usable rows; cannot fit"
        )
    nominal = _nominal_sample_rate_hz(raw.capture_meta_json, raw.audio_wav)
    anchor = fit_audio_anchor(pairs, nominal_rate_hz=nominal, robust=True)
    return anchor, raw


def audio_start_boottime_ns(session: Union[str, Path, RawInputs]) -> float:
    """CLOCK_BOOTTIME (ns) of the session WAV's first sample (sample 0).

    Uses the robust (MAD outlier-rejecting) stream-anchor fit, so late anchor
    writes do not bias the start estimate.
    """
    anchor, _ = _fit_session_audio(session)
    return anchor.frame_to_boot_ns(0.0)


# -----------------------------------------------------------------------------
# Clip discovery + first-sample boottime
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipInfo:
    """One combinable media: the FULL session or a cut segment slice."""

    kind: str  # "full" | "cut"
    label: str
    mp4: Path
    video_anchor: Optional[Path] = None
    chop_meta: Optional[Path] = None
    # chop_meta start/end (cut only) — the fallback t0 and the known duration.
    start_boottime_ns: Optional[int] = None
    duration_s: Optional[float] = None


def _trim_clip_from_meta(meta_path: Path) -> Optional[ClipInfo]:
    """Build a Cut :class:`ClipInfo` from a ``*.chop_meta.json`` sidecar."""
    try:
        cm = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(cm, dict):
            return None
    except Exception:
        return None
    chop_dir = meta_path.parent
    base = meta_path.name[: -len(".chop_meta.json")]
    mp4: Optional[Path] = chop_dir / f"{base}.mp4"
    if not mp4.is_file():
        mp4s = sorted(chop_dir.glob("chop_*.mp4"))
        mp4 = mp4s[0] if mp4s else None
    if mp4 is None:
        return None
    anchor: Optional[Path] = chop_dir / f"{base}.video_anchor.txt"
    if not anchor.is_file():
        anchors = sorted(chop_dir.glob("chop_*.video_anchor.txt"))
        anchor = anchors[0] if anchors else None
    start = cm.get("start_boottime_ns")
    end = cm.get("end_boottime_ns")
    dur = (
        (float(end) - float(start)) / 1e9
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
        else None
    )
    label = f"TRIM  {mp4.name}" + (f"  ({dur:.2f} s)" if dur is not None else "")
    return ClipInfo(
        kind="trim",
        label=label,
        mp4=mp4,
        video_anchor=anchor,
        chop_meta=meta_path,
        start_boottime_ns=int(start) if isinstance(start, (int, float)) else None,
        duration_s=dur,
    )


def discover_videos(session_dir: Union[str, Path]) -> List[ClipInfo]:
    """All combinable media files in ``session_dir``: FULL session(s) first, then
    every ``chop_*/`` cut slice (and the dir itself when it is a segment dir)."""
    session = Path(session_dir)
    items: List[ClipInfo] = []

    # FULL session(s) in the session root.
    for mp4 in sorted(session.glob("recording_*.mp4")):
        anchor: Optional[Path] = session / f"{mp4.stem}.video_anchor.txt"
        if not anchor.is_file():
            anchors = sorted(session.glob("recording_*.video_anchor.txt"))
            anchor = anchors[0] if anchors else None
        items.append(
            ClipInfo(kind="full", label=f"FULL  {mp4.name}", mp4=mp4,
                     video_anchor=anchor)
        )

    # Cut slices: segment meta in the dir itself (session IS a segment dir) or
    # in chop_* subdirs.
    metas = sorted(session.glob("*.chop_meta.json")) + sorted(
        session.glob("chop_*/*.chop_meta.json")
    )
    for meta_path in metas:
        clip = _trim_clip_from_meta(meta_path)
        if clip is not None:
            items.append(clip)
    return items


def clip_first_frame_boot_ns(
    clip: ClipInfo, capture_meta_json: Optional[Path] = None
) -> float:
    """CLOCK_BOOTTIME (ns) of the chosen clip's first sample.

    FULL -> per-sample media anchor min bootNs, else capture_meta
    ``media.video_t0_boottime_ns``.
    Cut -> the segment's own media anchor min bootNs (the physically-real
    sample-0 boot), else chop_meta ``start_boottime_ns``.
    """
    if clip.kind == "trim":
        import warnings
        anchor_t0: Optional[float] = None
        if clip.video_anchor is not None:
            anchor_t0 = _first_boottime_ns_from_video_anchor(Path(clip.video_anchor))
        if anchor_t0 is not None:
            # The anchor min bootNs IS the real first captured sample's boottime and
            # is authoritative. chop_meta start_boottime_ns is the *requested* cut-in
            # point and can sit up to ~a sample-interval before the real first sample
            # (observed ~100 ms on real data) -- surface a large divergence so the
            # capture-side anomaly is visible, but keep using the anchor value.
            if clip.start_boottime_ns is not None:
                div_ms = abs(anchor_t0 - float(clip.start_boottime_ns)) / 1e6
                if div_ms > 20.0:
                    warnings.warn(
                        f"combine_av: chop anchor min bootNs differs from chop_meta "
                        f"start_boottime_ns by {div_ms:.0f} ms for {clip.mp4}; using "
                        "the anchor (the real first-frame boottime).",
                        stacklevel=2,
                    )
            return float(anchor_t0)
        if clip.start_boottime_ns is not None:
            # Last resort: the anchor is unreadable. start_boottime_ns is the
            # REQUESTED cut point and may be ~100 ms before the real first sample,
            # so stream can be up to ~0.1 s misaligned. Warn loudly.
            warnings.warn(
                f"combine_av: chop video anchor unreadable for {clip.mp4}; falling "
                "back to chop_meta start_boottime_ns (the requested trim point, which "
                "may be ~100 ms before the real first frame -> audio up to ~0.1 s off). "
                "Provide the chop video_anchor.txt for exact sync.",
                stacklevel=2,
            )
            return float(clip.start_boottime_ns)
        raise ValueError(
            f"trim clip {clip.mp4} has no readable video anchor and no "
            "start_boottime_ns in its chop_meta; cannot sync"
        )

    # FULL session.
    if clip.video_anchor is not None:
        t0 = _first_boottime_ns_from_video_anchor(Path(clip.video_anchor))
        if t0 is not None:
            return float(t0)
    if capture_meta_json is not None:
        try:
            from .capture_meta import parse_capture_meta

            cm = parse_capture_meta(Path(capture_meta_json))
            if cm.video_t0_boottime_ns is not None:
                return float(cm.video_t0_boottime_ns)
        except Exception:
            pass
    raise ValueError(
        f"full recording {clip.mp4} has no readable video anchor and "
        "capture_meta carries no video_t0_boottime_ns; cannot sync"
    )


# -----------------------------------------------------------------------------
# The mux plan (pure) + execution
# -----------------------------------------------------------------------------


#: |drift| below this (ppm) is not worth an atempo resample.
DRIFT_PPM_THRESHOLD = 0.5


@dataclass(frozen=True)
class MuxPlan:
    """Everything needed to mux: numbers + the exact the external converter command.

    Built by :func:`plan_mux` without running anything; executed by
    :func:`run_mux`.
    """

    clip_path: Path
    clip_kind: str  # "full" | "cut"
    clip_label: str
    wav_path: Path
    clip_boot_ns: float          # boottime of the clip's first sample
    audio_start_boot_ns: float   # boottime of WAV sample 0 (robust anchor fit)
    audio_seek_s: float          # seconds into the WAV for the clip's sample 0
    nominal_fs: float            # declared WAV sample rate (Hz)
    true_fs: float               # anchor-fit effective sample rate (Hz)
    ppm: float                   # stream crystal drift vs nominal (ppm)
    atempo: Optional[float]      # tempo factor, or None when no correction
    out_path: Path
    ffmpeg_cmd: List[str]
    warnings: List[str] = field(default_factory=list)
    # Extras for reporting (not needed to run the command).
    wav_duration_s: Optional[float] = None
    clip_duration_s: Optional[float] = None
    anchor_n: int = 0
    anchor_rejected: int = 0


def _select_clip(
    items: List[ClipInfo], which: Union[None, int, str, ClipInfo]
) -> ClipInfo:
    """Resolve ``which`` (menu number, 'full', or a ClipInfo) to a clip."""
    if not items:
        raise ValueError(
            "No combinable videos found (no recording_*.mp4 and no chop_*/ slices)."
        )
    if isinstance(which, ClipInfo):
        return which
    if which is None:
        if len(items) == 1:
            return items[0]
        raise ValueError(
            f"session has {len(items)} combinable videos; pass which="
            "'full' or a 1-based menu number"
        )
    if isinstance(which, str) and which.strip().lower() == "full":
        for it in items:
            if it.kind == "full":
                return it
        raise ValueError("'full' requested but no full recording in this session.")
    try:
        idx = int(which) - 1
    except (TypeError, ValueError):
        raise ValueError(f"{which!r} is not a valid menu number (1..{len(items)}) or 'full'.")
    if 0 <= idx < len(items):
        return items[idx]
    raise ValueError(f"{which!r} is not a valid menu number (1..{len(items)}).")


def plan_mux(
    session_dir: Union[str, Path],
    which: Union[None, int, str, ClipInfo] = None,
    *,
    out: Union[None, str, Path] = None,
    no_rate: bool = False,
    clip_duration_s: Optional[float] = None,
    ffmpeg: Optional[str] = None,
) -> MuxPlan:
    """Build the full mux plan for ``session_dir``. Pure: runs nothing.

    ``which`` selects the clip: ``'full'``, a 1-based menu number matching
    :func:`discover_videos` order, a :class:`ClipInfo`, or None when the
    session has exactly one combinable media.

    ``clip_duration_s`` (optional) enables the silent-tail check for FULL
    clips without the probe tool; Cut clips use the chop_meta duration automatically.
    """
    session = Path(session_dir)
    anchor, raw = _fit_session_audio(session)
    if raw.audio_wav is None:
        raise FileNotFoundError("no audio_*.wav in session; nothing to mux")
    wav_path = Path(raw.audio_wav)

    clip = _select_clip(discover_videos(session), which)
    clip_boot = clip_first_frame_boot_ns(clip, raw.capture_meta_json)

    audio_start_boot = anchor.frame_to_boot_ns(0.0)
    audio_seek_s = (clip_boot - audio_start_boot) / 1e9

    nominal_fs = anchor.nominal_rate_hz
    true_fs = anchor.effective_rate_hz
    ppm = anchor.rate_drift_ppm

    wav_dur: Optional[float] = None
    try:
        with wave.open(str(wav_path), "rb") as w:
            if true_fs > 0:
                wav_dur = w.getnframes() / true_fs
    except Exception:
        pass

    clip_dur = clip_duration_s if clip_duration_s is not None else clip.duration_s

    atempo: Optional[float] = None
    if not no_rate and abs(ppm) > DRIFT_PPM_THRESHOLD and nominal_fs > 0:
        atempo = true_fs / nominal_fs
    af = ["-af", f"atempo={atempo:.9f}"] if atempo is not None else []

    out_path = Path(out) if out else session / f"combined_{clip.mp4.stem}.mp4"

    # audio_seek_s >= 0 : clip sample 0 is inside the stream -> seek into the
    #                     WAV (-ss before -i).
    # audio_seek_s <  0 : stream begins after the clip -> delay stream by
    #                     |seek| (-itsoffset).
    exe = ffmpeg or _resolve_exe("ffmpeg")
    cmd: List[str] = [exe, "-y", "-i", str(clip.mp4)]
    if audio_seek_s >= 0:
        cmd += ["-ss", f"{audio_seek_s:.6f}", "-i", str(wav_path)]
    else:
        cmd += ["-itsoffset", f"{-audio_seek_s:.6f}", "-i", str(wav_path)]
    cmd += [
        "-map", "0:v", "-map", "1:a", *af,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_path),
    ]

    warnings: List[str] = []
    if audio_seek_s < -0.5:
        warnings.append(
            "audio starts well after the clip -> the clip head will be silent."
        )
    if clip_dur is not None and wav_dur is not None and audio_seek_s + clip_dur > wav_dur + 0.5:
        warnings.append(
            "clip runs past the end of the audio -> its tail will be silent."
        )

    return MuxPlan(
        clip_path=clip.mp4,
        clip_kind=clip.kind,
        clip_label=clip.label,
        wav_path=wav_path,
        clip_boot_ns=clip_boot,
        audio_start_boot_ns=audio_start_boot,
        audio_seek_s=audio_seek_s,
        nominal_fs=nominal_fs,
        true_fs=true_fs,
        ppm=ppm,
        atempo=atempo,
        out_path=out_path,
        ffmpeg_cmd=cmd,
        warnings=warnings,
        wav_duration_s=wav_dur,
        clip_duration_s=clip_dur,
        anchor_n=anchor.n,
        anchor_rejected=anchor.n_rejected,
    )


def run_mux(plan: MuxPlan) -> Path:
    """Execute the plan's the external converter command. Returns the output path."""
    subprocess.run(plan.ffmpeg_cmd, check=True)
    return plan.out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _choose_interactive(items: List[ClipInfo]) -> ClipInfo:
    print("\nVideos in this recording:")
    for i, it in enumerate(items, 1):
        print(f"  {i}. {it.label}")
    while True:
        ans = input(f"Pick a video to combine [1-{len(items)}]: ").strip()
        try:
            idx = int(ans) - 1
            if 0 <= idx < len(items):
                return items[idx]
        except ValueError:
            pass
        print("  invalid choice.")


def _print_plan(plan: MuxPlan) -> None:
    print(f"\nchosen        : {plan.clip_label}")
    print(f"clip frame0   : boot {plan.clip_boot_ns:.0f} ns  ({plan.clip_kind})")
    print(
        f"audio start   : boot {plan.audio_start_boot_ns:.0f} ns  "
        f"(robust fit over {plan.anchor_n} anchors, {plan.anchor_rejected} rejected)"
    )
    wav_part = f"(WAV {plan.wav_duration_s:.1f} s" if plan.wav_duration_s is not None else "("
    tail = f", clip {plan.clip_duration_s:.1f} s)" if plan.clip_duration_s is not None else ")"
    print(f"audio seek    : {plan.audio_seek_s:+.3f} s into the WAV  {wav_part}{tail}")
    corrected = "  -> drift corrected" if plan.atempo is not None else "  (rate correction off)"
    print(f"true audio fs : {plan.true_fs:.3f} Hz  ({plan.ppm:+.2f} ppm){corrected}")
    for w in plan.warnings:
        print(f"WARN: {w}", file=sys.stderr)
    print("ffmpeg:\n  " + " ".join(plan.ffmpeg_cmd))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m data_pipeline.combine_av",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("session")
    ap.add_argument("--video", default=None,
                    help="menu number, or 'full' (skips the prompt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-rate", action="store_true",
                    help="skip the audio-crystal drift correction (start-offset only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + ffmpeg command, do not run")
    a = ap.parse_args(argv)

    session = Path(a.session)
    items = discover_videos(session)
    if not items:
        print("No combinable videos found (no recording_*.mp4 and no chop_*/ slices).",
              file=sys.stderr)
        return 2

    try:
        clip = _select_clip(items, a.video) if a.video is not None else _choose_interactive(items)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    # Clip duration: chop_meta gives it for cuts; the probe tool (tolerant) for fulls.
    clip_dur = clip.duration_s
    if clip_dur is None:
        clip_dur = _ffprobe_duration_s(clip.mp4)

    try:
        plan = plan_mux(
            session,
            clip,
            out=a.out,
            no_rate=a.no_rate,
            clip_duration_s=clip_dur,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2

    _print_plan(plan)
    if a.dry_run:
        return 0
    run_mux(plan)
    print(f"-> {plan.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
