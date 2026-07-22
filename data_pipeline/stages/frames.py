"""Stage 2: lossless sample extraction from the logged Container file.

The goal is to recover the **exact source PTS** (presentation timestamp) of
every saved sample so the rest of the pipeline can geocode each sample to
sub-millisecond UTC. We therefore avoid the external converter's ``fps=N`` filter (which
*resamples* and emits synthetic 1/N-second PTSes) and instead use
``select='not(mod(n,K))'`` which **keeps** samples at integer multiples of
the source sample index, preserving each sample's original PTS unchanged.

Lossless options:

* ``png``  : true lossless, ~5-10x JPEG size, best for coordinate tagging.
* ``tiff`` : true lossless, no compression by default - faster encoding,
             biggest files.
* ``jpeg1``: the external converter ``-q:v 1`` near-lossless JPEG, the smallest visually
             lossless option (use only for quick iteration).
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

from ..ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe
from ..pipeline import LogFn, make_logger


ImageFormat  = Literal["png", "tiff", "jpeg1"]
RotationDeg  = Literal[0, 90, 180, 270]

_ROTATION_VF: dict[int, list[str]] = {
    0:   [],
    90:  ["transpose=1"],        # 90° clockwise
    180: ["hflip", "vflip"],     # 180°
    270: ["transpose=2"],        # 90° counter-clockwise
}

_SHOWINFO_RE = re.compile(
    r"Parsed_showinfo_\d+.*\bn:\s*(\d+)\b.*\bpts_time:\s*(-?[0-9.]+)"
)
_TMPSEQ_RE = re.compile(r"^__tmp_pts__(\d{6})", re.I)


def _tmp_seq_key(path: Path) -> int:
    m = _TMPSEQ_RE.match(path.name)
    return int(m.group(1)) if m else -1


@dataclass(frozen=True)
class FrameExtractionResult:
    """Summary of a single extraction run."""

    frames_dir: Path
    frame_times_csv: Path
    frame_count: int
    requested_fps: float
    effective_fps: float
    source_fps: float
    decimation_factor: int
    fmt: ImageFormat
    pts_name_decimals: int
    rotation: int = 0


def _seq_pad_width(n_frames: int) -> int:
    """Zero-pad width for sequential sample indices.

    At least 6 digits (``frame_000000``); widens automatically for very long
    captures so the last index is fully padded (no dots, no variable width).
    """
    if n_frames <= 1:
        return 6
    return max(6, len(str(n_frames - 1)))


def _stem_from_seq(seq: int, pad: int, prefix: str = "frame_") -> str:
    """Stem ``<prefix><seq>`` with a zero-padded *sequential* index.

    Filenames are DOT-FREE on purpose: the only dot in the final filename is
    the real image extension. the external tool's reference importer matches sources by
    label (filename stem); a decimal PTS in the name (``frame_0.200111.png``)
    made the stem ambiguous across the external tool versions (it could collapse on the
    first dot to ``frame_0``), so the join silently matched 0 sources. A bare
    ``frame_000001`` stem is unambiguous.

    The sample's source PTS (``t_video_s``) is preserved with full precision in
    ``extracted_frame_times.csv``; ALL timing downstream reads it from there,
    never from the filename.
    """
    return f"{prefix}{seq:0{pad}d}"


def _ffmpeg_args_for_format(fmt: ImageFormat, out_pattern: str) -> tuple[list[str], str]:
    # ``-pix_fmt`` is set explicitly so 10-bit HDR sources (HEVC ``yuv420p10le``,
    # AV1) are tone-mapped to 8-bit for encoders that don't speak deep-bit-depth.
    # PNG/TIFF support deep bit depths natively so leave them alone.
    if fmt == "png":
        return (
            ["-c:v", "png", "-compression_level", "1", out_pattern + ".png"],
            ".png",
        )
    if fmt == "tiff":
        return (["-c:v", "tiff", out_pattern + ".tiff"], ".tiff")
    if fmt == "jpeg1":
        return (
            ["-pix_fmt", "yuvj420p", "-q:v", "1", out_pattern + ".jpg"],
            ".jpg",
        )
    raise ValueError(f"Unknown format: {fmt}")


def _ffprobe_source_fps(video: Path) -> float:
    """Read the source media's average sample rate via the probe tool.

    We use ``avg_frame_rate`` (computed from total samples and duration); for
    practically-CFR device media files this matches ``r_frame_rate``. Result is a
    plain float in Hz.
    """
    out = subprocess.run(
        [
            resolve_ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=nw=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    avg_m = re.search(r"^avg_frame_rate=(\S+)\s*$", out, re.M)
    r_m = re.search(r"^r_frame_rate=(\S+)\s*$", out, re.M)
    raw = (avg_m or r_m).group(1) if (avg_m or r_m) else "0/0"
    try:
        f = Fraction(raw)
    except Exception:
        return 0.0
    return float(f) if f != 0 else 0.0


def _decimation_factor(source_fps: float, target_fps: float) -> int:
    """Pick K so effective ~ source/K is closest to target.

    Uses half-up rounding (``int(x + 0.5)``) instead of Python's banker's
    rounding -- e.g. ``round(60/24)`` is ``2`` in Python which gave K=2 for
    source=60 target=24 (effective=30 Hz, twice the request). Half-up gives
    K=3 (effective=20 Hz), which is the intuitive "closest" answer.
    """
    if source_fps <= 0 or target_fps <= 0:
        return 1
    ratio = source_fps / target_fps
    return max(1, int(ratio + 0.5))


def _ffprobe_frame_count(video: Path, K: int) -> Optional[int]:
    """Estimate how many samples will be extracted (nb_frames / K, or duration-based)."""
    try:
        out = subprocess.run(
            [
                resolve_ffprobe(),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
                "-of", "default=nw=1",
                str(video),
            ],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).stdout
        nb_m = re.search(r"^nb_frames=(\d+)\s*$", out, re.M)
        if nb_m and int(nb_m.group(1)) > 0:
            return max(1, int(nb_m.group(1)) // K)
        dur_m = re.search(r"^duration=([\d.]+)\s*$", out, re.M)
        fps_m = re.search(r"^avg_frame_rate=(\S+)\s*$", out, re.M)
        if dur_m and fps_m:
            try:
                src_fps = float(Fraction(fps_m.group(1)))
                return max(1, int(float(dur_m.group(1)) * src_fps / K))
            except Exception:
                pass
    except Exception:
        pass
    return None



def _run_streaming_select(
    *, video: Path, out_dir: Path, frames_dir: Path,
    tmp_prefix: str, fmt: "ImageFormat",
    pts_name_decimals: int, rotation: int, name_prefix: str,
    select: list[int],
    pts_for_select: Optional[list[float]],
    src_fps: float,
    progress_cb: Optional[Callable[[int, Optional[int]], None]],
    log_: "Callable[[str], None]",
    fps_requested: float, K: int, eff_fps: float,
) -> "FrameExtractionResult":
    """Stream raw RGB from the external converter and write only the requested sample indices.

    Used when ``select_indices`` is long enough to overflow the external converter's
    select-expression evaluator (~hundreds of OR terms) or Windows'
    32 KB ``CreateProcess`` cap. The decoder runs once; Python advances
    the pipe to each next target index and writes the sample via The feature library.

    PTS is reconstructed as ``n / src_fps``. For variable-FPS sources this
    will drift from the true presentation time — the adaptive selector
    already operates on this CFR approximation, so the two stay coherent.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Streaming frame extraction (used for large adaptive runs) "
            "requires OpenCV + NumPy. Install with `pip install opencv-python`."
        ) from e

    if src_fps <= 0:
        src_fps = _ffprobe_source_fps(video) or 30.0
    log_(f"[frames] streaming select via cv2.VideoCapture: "
         f"src_fps={src_fps:.4f} keep={len(select)}")

    keep_set = set(select)
    if pts_for_select is None:
        # Caller (``run``) refuses to enter here without true PTS, so this
        # path is unreachable from the public API; we keep the guard so a
        # future direct caller can't silently corrupt the t_video_s
        # output by falling back to the n/src_fps approximation.
        raise ValueError(
            "pts_for_select is required for the streaming path — supply "
            "the showinfo-derived PTS array per source-index."
        )
    if len(pts_for_select) != len(select):
        raise ValueError(
            "pts_for_select length must match select length"
        )
    # Each (target_index, true_pts) pair, popped from the end as O(1).
    pair_stack = list(zip(select, pts_for_select))
    pair_stack.reverse()

    # Pre-resolve format → cv2 imwrite parameters and extension.
    if fmt == "png":
        ext = ".png"
        imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif fmt == "tiff":
        ext = ".tiff"
        imwrite_params = []
    elif fmt == "jpeg1":
        ext = ".jpg"
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    else:
        raise ValueError(f"Unsupported fmt for streaming path: {fmt!r}")

    # Clean stale tmp files from any previous aborted run.
    for stale in frames_dir.glob(f"{tmp_prefix}*"):
        try:
            stale.unlink()
        except OSError:
            pass

    # cv2.VideoCapture handles auto-rotation from display-matrix side data
    # and colorspace conversion (BT.709 matrix vs BT.601) consistently — both
    # paths produced byte-identical output to ``the external converter -i media -vf select PNG``
    # in cross-check tests, whereas a raw rgb24 pipe diverged.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open {video}")

    rot_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(int(rotation))

    # Per-write bookkeeping: keep an explicit ``seq -> (n_src, pts)`` map so
    # the rename phase never relies on the parallel-sort coincidence
    # between tmp filenames (write order) and ``sorted(times.keys())``
    # (source-index order). The two orderings are identical TODAY because
    # pair_stack is sorted ascending, but the explicit map removes the
    # latent risk if any caller ever supplies unsorted indices.
    seq_to_npts: list[tuple[int, float]] = []
    n = -1
    written = 0

    try:
        while pair_stack:
            target, target_pts = pair_stack[-1]
            # Skip forward to ``target``. Sequential reads avoid the seek
            # accuracy hit (CV_CAP_PROP_POS_FRAMES rounds to nearest IDR
            # on many backends).
            frame = None
            while n < target:
                ok, frame = cap.read()
                if not ok:
                    from ..errors import PipelineError
                    raise PipelineError(
                        "E-PP-302",
                        f"Video ended at frame {n}; needed up to frame {target}",
                        hint="Frame index outside the video's duration. Either "
                             "shorten the requested range or supply a longer "
                             "recording_*.mp4.",
                        context={"got_frame": n, "wanted_frame": target},
                    )
                n += 1
            if frame is None:
                from ..errors import PipelineError
                raise PipelineError(
                    "E-PP-900",
                    f"Internal invariant: frame None after successful cap.read() at index {n}",
                    hint="This is a bug — please report with the JSON error file.",
                    context={"index": n, "target": target},
                )
            # cv2.VideoCapture returns BGR by default.
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)
            tmp_path = frames_dir / f"{tmp_prefix}{written:06d}{ext}"
            ok = cv2.imwrite(str(tmp_path), frame, imwrite_params)
            if not ok:
                from ..errors import PipelineError
                raise PipelineError(
                    "E-PP-303",
                    f"cv2.imwrite failed writing {tmp_path}",
                    hint="Check disk space + write permissions for the output "
                         "directory; Windows path-length limit (260 chars) is "
                         "another common cause.",
                    context={"path": str(tmp_path)},
                )
            # TRUE PTS — supplied by the caller (showinfo-derived). Filename
            # + CSV entries downstream depend on this being right.
            seq_to_npts.append((n, target_pts))
            written += 1
            pair_stack.pop()
            if progress_cb:
                progress_cb(written, len(select))
    finally:
        cap.release()

    if not seq_to_npts:
        from ..errors import PipelineError
        raise PipelineError(
            "E-PP-301",
            "Streaming select produced no frames",
            hint="Video opened but the selector matched zero frames. Check "
                 "the --select expression and the video's frame count.",
            context={"video": str(video)},
        )

    # Rename tmp files to canonical dot-free sequential names and write CSV.
    stale_glob = f"{name_prefix}*{ext}" if name_prefix else f"*{ext}"
    for stale in frames_dir.glob(stale_glob):
        if stale.name.startswith(tmp_prefix):
            continue
        try:
            stale.unlink()
        except OSError:
            pass

    # Assign DOT-FREE sequential names in capture-time (PTS-ascending) order so
    # ``frame_000000`` is the earliest sample. ``seq_to_npts`` is in write order;
    # we sort by PTS to fix the index→time mapping independent of write order.
    pad = _seq_pad_width(len(seq_to_npts))
    order_by_pts = sorted(range(len(seq_to_npts)), key=lambda j: seq_to_npts[j][1])
    # name_for_write_idx[j] = final filename for the tmp file written at seq j.
    name_for_write_idx: list[str] = [""] * len(seq_to_npts)
    csv_rows: list[tuple[str, float]] = []  # (name, pts) in PTS-ascending order
    for new_seq, write_idx in enumerate(order_by_pts):
        new_name = f"{_stem_from_seq(new_seq, pad, name_prefix)}{ext}"
        name_for_write_idx[write_idx] = new_name
        csv_rows.append((new_name, seq_to_npts[write_idx][1]))

    for write_idx, new_name in enumerate(name_for_write_idx):
        tmp_path = frames_dir / f"{tmp_prefix}{write_idx:06d}{ext}"
        if not tmp_path.is_file():
            raise RuntimeError(
                f"Streaming tmp file missing for write seq {write_idx}: {tmp_path}"
            )
        tmp_path.rename(frames_dir / new_name)

    frame_times_csv = out_dir / "extracted_frame_times.csv"
    with frame_times_csv.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["Image", "t_video_s"])
        # CSV rows already in ascending-PTS order (== sequential sample index).
        for name, t in csv_rows:
            img_decimals = max(pts_name_decimals, 12)
            w_csv.writerow([name, f"{t:.{img_decimals}f}"])
    log_(f"[frames] streaming wrote {len(seq_to_npts)} frames -> {frame_times_csv}")

    return FrameExtractionResult(
        frames_dir=frames_dir,
        frame_times_csv=frame_times_csv,
        frame_count=len(seq_to_npts),
        requested_fps=fps_requested,
        effective_fps=eff_fps,
        source_fps=src_fps,
        decimation_factor=K,
        fmt=fmt,
        pts_name_decimals=pts_name_decimals,
        rotation=int(rotation),
    )


def run(
    *,
    video: Path,
    out_dir: Path,
    fps: float,
    fmt: ImageFormat = "png",
    pts_name_decimals: int = 6,
    rotation: RotationDeg = 0,
    name_prefix: str = "frame_",
    select_indices: Optional[Sequence[int]] = None,
    pts_for_indices: Optional[Sequence[float]] = None,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
    log: Optional[LogFn] = None,
) -> FrameExtractionResult:
    """Extract samples from ``media`` at approximately ``fps`` into ``out_dir/samples/``.

    Side effects:

    * Lossless images named ``frame_<seq>.<ext>`` where ``<seq>`` is a
      zero-padded SEQUENTIAL index in capture-time (PTS-ascending) order, e.g.
      ``frame_000000.png``, ``frame_000001.png``. Filenames are DOT-FREE (the
      only dot is the real extension) so the external tool's reference importer matches
      each source by its label (the stem) without collapsing on a decimal point.
    * ``out_dir/extracted_frame_times.csv`` with columns ``Image, t_video_s``.
      ``t_video_s`` is the **source** PTS (seconds, full precision); ``Image``
      matches the file on disk for Coordinate output import. ALL downstream timing reads
      ``t_video_s`` from this CSV, never from the filename.

    ``pts_name_decimals`` no longer affects filenames (kept for API/CLI
    compatibility); it still controls the minimum decimal precision of the
    ``t_video_s`` column.
    """
    log_ = make_logger(log)
    if fps <= 0:
        raise ValueError(f"fps must be > 0 (got {fps!r})")

    out_dir = out_dir.resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Purge any tmp files from a previously-aborted run BEFORE invoking the external converter.
    # Otherwise the post-extraction glob picks them up alongside this run's
    # output and the count-vs-showinfo cross-check trips with a misleading
    # "sample count mismatch" error.
    tmp_prefix = "__tmp_pts__"
    for stale in frames_dir.glob(f"{tmp_prefix}*"):
        try:
            stale.unlink()
        except OSError:
            pass

    src_fps = _ffprobe_source_fps(video)
    if select_indices is not None:
        # Explicit-list mode: source FPS is informational only; effective FPS
        # is the average of selected samples over the full media duration if
        # that information is even meaningful (often not for adaptive-spacing
        # runs). K is reported as 0 to signal "not a uniform decimation".
        K = 0
        eff_fps = float(len(select_indices)) if select_indices else 0.0
        log_(
            f"[frames] source_fps={src_fps:.4f} mode=explicit-indices "
            f"(n_keep={len(select_indices)})"
        )
    else:
        K = _decimation_factor(src_fps, fps)
        eff_fps = src_fps / K if src_fps > 0 else fps
        log_(
            f"[frames] source_fps={src_fps:.4f} requested_fps={fps:g} "
            f"-> K={K} effective_fps={eff_fps:.4f}"
        )
    total_est: Optional[int] = (
        len(select_indices) if select_indices is not None
        else (_ffprobe_frame_count(video, K) if progress_cb else None)
    )

    # The external converter can only enumerate outputs with %%d; write to sequential temp
    # names then rename by PTS so on-disk filenames are human-readable timestamps.
    out_pattern = str(frames_dir / f"{tmp_prefix}%06d")
    output_args, _ext = _ffmpeg_args_for_format(fmt, out_pattern)

    # When the caller supplied an explicit index list, route through the
    # streaming Python writer instead of the external converter `select` filter. The
    # filter's expression evaluator overflows at a few hundred OR terms and
    # Windows' CreateProcess caps the command line at 32 KB, so the
    # filter-string approach fails for adaptive runs producing thousands of
    # indices. Streaming pipes raw RGB once and we encode keepers ourselves.
    if select_indices is not None:
        # Build aligned (index, pts) pairs so we never lose the mapping
        # while sorting. ``pts_for_indices`` is REQUIRED for trustworthy
        # timing — without it we fall back to n/src_fps, which is wrong on
        # variable-FPS device media (timing is the project's load-bearing
        # invariant; the caller MUST supply true PTS for adaptive runs).
        raw_idx = [int(i) for i in select_indices]
        if pts_for_indices is None:
            # Timing is the load-bearing invariant for this pipeline. The
            # ``n / src_fps`` approximation is wrong on every variable-FPS
            # device media we've seen (e.g. reference session jumps 2.1 s between n=4
            # and n=5), so we refuse to silently produce wrong timestamps.
            raise ValueError(
                "pts_for_indices is required when select_indices is given. "
                "Refusing the n/src_fps approximation — call enumerate_source_frames() "
                "and pass the true PTS array."
            )
        if len(pts_for_indices) != len(raw_idx):
            raise ValueError(
                "pts_for_indices length must match select_indices length"
            )
        pairs = list(zip(raw_idx, [float(p) for p in pts_for_indices]))
        # De-duplicate by index, keep sort by index.
        seen: dict[int, float] = {}
        for i, t in pairs:
            seen.setdefault(i, t)
        sorted_pairs = sorted(seen.items(), key=lambda x: x[0])
        sel = [i for i, _ in sorted_pairs]
        pts_for_sel = [t for _, t in sorted_pairs]
        if not sel:
            raise ValueError("select_indices is empty — nothing to extract.")
        # the external converter's `select` filter parser overflows past a few dozen OR
        # terms; the safe threshold is empirically ~50. For explicit-index
        # workloads the streaming path is the right answer almost always.
        if len(sel) > 32:
            return _run_streaming_select(
                video=video, out_dir=out_dir, frames_dir=frames_dir,
                tmp_prefix=tmp_prefix, fmt=fmt,
                pts_name_decimals=pts_name_decimals,
                rotation=int(rotation), name_prefix=name_prefix,
                select=sel, pts_for_select=pts_for_sel, src_fps=src_fps,
                progress_cb=progress_cb, log_=log_,
                fps_requested=fps, K=K, eff_fps=eff_fps,
            )
        terms = "+".join(f"eq(n\\,{i})" for i in sel)
        vf_parts = [f"select='{terms}'"]
    else:
        vf_parts = [f"select='not(mod(n\\,{K}))'"]
    vf_parts += _ROTATION_VF.get(int(rotation), [])
    vf_parts.append("showinfo")
    vf = ",".join(vf_parts)
    if rotation:
        log_(f"[frames] rotation={rotation}°")

    # Windows' CreateProcess caps the command line at ~32 KB. The adaptive
    # selector can produce thousands of indices ⇒ ``-vf "<expr>"`` overflows.
    # Switch to ``-filter_script:v <path>`` when the filter string is large.
    use_filter_script = len(vf) > 8000
    filter_script_path: Optional[Path] = None
    # ``-map 0:v:0`` locks to the first media stream (some captures bundle a
    # depth/pano stream alongside the main media). ``-an -sn -dn`` drops
    # stream/subs/data so the encoder never has to negotiate them.
    # ``-vsync vfr`` is the legacy spelling that the external converter 2.x through 8.x all
    # accept; the renamed ``-fps_mode passthrough`` is left out to avoid
    # breaking older vendored builds.
    common_input = [
        resolve_ffmpeg(),
        "-hide_banner", "-loglevel", "info", "-y",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
    ]
    if use_filter_script:
        filter_script_path = frames_dir / "__filter_script.txt"
        filter_script_path.write_text(vf, encoding="utf-8")
        cmd = [
            *common_input,
            "-filter_script:v", str(filter_script_path),
            "-vsync", "vfr",
            *output_args,
        ]
        log_(f"[frames] vf string is {len(vf):,} chars → "
             f"using -filter_script:v ({filter_script_path.name})")
    else:
        cmd = [
            *common_input,
            "-vf", vf,
            "-vsync", "vfr",
            *output_args,
        ]
    log_(f"[frames] cmd={' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    stderr_lines: list[str] = []
    times: dict[int, float] = {}
    n_seen = 0
    for line in proc.stderr:  # type: ignore[union-attr]
        stderr_lines.append(line)
        m = _SHOWINFO_RE.search(line)
        if m:
            times[int(m.group(1))] = float(m.group(2))
            n_seen += 1
            if progress_cb:
                progress_cb(n_seen, total_est)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}).\n"
            f"stderr (tail):\n{''.join(stderr_lines)[-2000:]}"
        )

    if not times:
        raise RuntimeError(
            "ffmpeg produced frames but no showinfo timestamps were captured. "
            "Run with -loglevel info or check the ffmpeg version."
        )

    ext = _ffmpeg_args_for_format(fmt, "")[1]
    sorted_ns = sorted(times.keys())
    tmp_paths = sorted(frames_dir.glob(f"{tmp_prefix}*{ext}"), key=_tmp_seq_key)
    if len(tmp_paths) != len(sorted_ns):
        raise RuntimeError(
            f"Frame count mismatch: ffmpeg wrote {len(tmp_paths)} images but "
            f"showinfo has {len(sorted_ns)} timestamps (tmp prefix {tmp_prefix!r})."
        )
    # Sort-by-source-index must equal write-order (PTS-ascending) for the
    # parallel-sort rename to be correct. the external converter emits samples in display
    # (PTS) order from its decoder reorder buffer, so for CFR device media
    # this holds. Detect the rare case where a re-ordered codec breaks the
    # assumption and fail loudly rather than mislabel filenames.
    sorted_pts = [times[n_src] for n_src in sorted_ns]
    for _i in range(1, len(sorted_pts)):
        if sorted_pts[_i] < sorted_pts[_i - 1]:
            raise RuntimeError(
                "Source-index order is not PTS-ascending — write order may "
                "not match (B-frame reorder or VFR codec). Rerun via the "
                "streaming select path with explicit indices."
            )

    # Clear stale outputs from earlier runs before renaming here. When the
    # caller passed an empty ``name_prefix`` we cannot safely wildcard the
    # whole folder, so restrict to the exact image extension.
    stale_glob = f"{name_prefix}*{ext}" if name_prefix else f"*{ext}"
    for stale in frames_dir.glob(stale_glob):
        if stale.name.startswith(tmp_prefix):
            continue  # tmp files for this run — handled separately above
        try:
            stale.unlink()
        except OSError:
            pass

    # DOT-FREE sequential naming. ``sorted_ns`` is source-index order, already
    # verified PTS-ascending just above, so position i == capture-time index.
    pad = _seq_pad_width(len(sorted_ns))
    final_names: list[str] = []

    for i, n_src in enumerate(sorted_ns):
        new_name = f"{_stem_from_seq(i, pad, name_prefix)}{ext}"
        new_path = frames_dir / new_name
        tmp_paths[i].rename(new_path)
        final_names.append(new_name)

    frame_times_csv = out_dir / "extracted_frame_times.csv"
    with frame_times_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image", "t_video_s"])
        for name, n_src in zip(final_names, sorted_ns):
            t = times[n_src]
            # Match filename precision plus full float for tools that need more.
            img_decimals = max(pts_name_decimals, 12)
            w.writerow([name, f"{t:.{img_decimals}f}"])

    log_(
        f"[frames] wrote {len(times)} frames as {name_prefix}<NNN>{ext} "
        f"(dot-free sequential, pad={pad}); times -> {frame_times_csv}"
    )
    # Best-effort cleanup of the temporary filter script (created only for
    # large explicit-index runs). Not critical if it sticks around.
    if filter_script_path is not None and filter_script_path.is_file():
        try:
            filter_script_path.unlink()
        except OSError:
            pass
    return FrameExtractionResult(
        frames_dir=frames_dir,
        frame_times_csv=frame_times_csv,
        frame_count=len(times),
        requested_fps=fps,
        effective_fps=eff_fps,
        source_fps=src_fps,
        decimation_factor=K,
        fmt=fmt,
        pts_name_decimals=pts_name_decimals,
        rotation=int(rotation),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output folder.")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument(
        "--format",
        choices=["png", "tiff", "jpeg1"],
        default="png",
        dest="fmt",
    )
    ap.add_argument(
        "--pts-name-decimals",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Decimal places for seconds-from-start in filenames (frame_<t>.ext). "
            "Default 6 (~1 µs in time axis at second resolution)."
        ),
    )
    args = ap.parse_args()
    run(
        video=args.video,
        out_dir=args.out,
        fps=args.fps,
        fmt=args.fmt,
        pts_name_decimals=args.pts_name_decimals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
