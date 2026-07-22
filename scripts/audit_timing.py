"""Timing-health audit for a the source app session.

Point this at a folder containing ``recording_*.container file`` + ``recording_*.txt``
(plus optional ``.pos``) and it prints a one-screen report covering every
clock the pipeline depends on:

* Sample-count agreement: showinfo (canonical) vs cv2.VideoCapture (used by
  the streaming extractor) vs raw rgb24 pipe (deprecated path, sanity).
* PTS monotonicity and largest jump (variable-FPS detector).
* Source-FPS reported vs derived; B-sample presence; codec metadata.
* Display-Matrix side-data rotation.
* Color matrix coefficients (warns when primaries / transfer / matrix
  disagree — e.g. the reference session's smpte170m/smpte170m/bt709 trap).
* recording_*.txt time anchor: RMSE, drift_ppm, n_rejected,
  fit-uncertainty, cubic-vs-linear improvement.
* Optional .pos coverage window vs media span.

Exit code is non-zero when any HARD threshold is breached so this script
can run in CI on every new session before the user trusts the pipeline.

Usage::

    python scripts/audit_timing.py <session_dir> [--pos <path>] [--strict]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys

# Ensure non-ASCII glyphs (e.g. ° in display rotation) don't crash on the
# default Windows cp1255 stdout encoding when this script is piped.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Add repo root to path so the script runs without ``pip install -e .``
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pipeline.ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe  # noqa: E402
from data_pipeline.stages.adaptive_frames import enumerate_source_frames  # noqa: E402
from data_pipeline.time_sync import fit_time_anchor  # noqa: E402
from data_pipeline.frame_time import (  # noqa: E402
    make_frame_to_utc,
    resolve_video_t0_boottime_ns,
)
from data_pipeline.parsers import parse_rtkpos  # noqa: E402


# Thresholds — exceed any of these in --strict mode and the script exits 1.
RMSE_HARD_MS = 100.0          # > this -> the time anchor is unreliable
DRIFT_PPM_HARD = 80.0         # > this -> device clock drift is excessive
MAX_PTS_GAP_HARD_S = 5.0      # > this -> likely a paused / corrupt session
FRAME_COUNT_MISMATCH_HARD = 0  # any mismatch is a regression


@dataclass
class AuditFinding:
    label: str
    value: str
    severity: str  # "ok" | "warn" | "fail"
    detail: str = ""


@dataclass
class AuditReport:
    findings: List[AuditFinding] = field(default_factory=list)

    def add(self, label: str, value: str, severity: str = "ok",
            detail: str = "") -> None:
        self.findings.append(AuditFinding(label, value, severity, detail))

    @property
    def has_failures(self) -> bool:
        return any(f.severity == "fail" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warn" for f in self.findings)


def _ffprobe_json(video: Path) -> dict:
    out = subprocess.run(
        [resolve_ffprobe(), "-v", "error",
         "-select_streams", "v:0",
         "-show_streams", "-show_format",
         "-print_format", "json",
         str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _count_cv2_frames(video: Path) -> int:
    try:
        import cv2
    except ImportError:
        return -1
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return -1
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def _count_raw_rgb24_frames(video: Path, w: int, h: int) -> int:
    """Count samples coming out of a raw rgb24 pipe (the deprecated path).

    On the reference session this came up SHORT vs showinfo (62280 vs 62468) — the bug
    that originally mislabelled every PTS. Keep this measurement so we
    catch the same divergence on any future device.
    """
    if w <= 0 or h <= 0:
        return -1
    frame_bytes = w * h * 3
    proc = subprocess.Popen(
        [resolve_ffmpeg(),
         "-hide_banner", "-loglevel", "error",
         "-i", str(video),
         "-an", "-sn",
         "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, bufsize=frame_bytes * 4,
    )
    n = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
            if len(buf) < frame_bytes:
                break
            n += 1
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass
    return n


def audit_video(video: Path, report: AuditReport) -> tuple[List[tuple[int, float]], dict]:
    meta = _ffprobe_json(video)
    streams = meta.get("streams", [])
    stream = streams[0] if streams else {}
    fmt = meta.get("format", {})
    w = int(stream.get("width", 0))
    h = int(stream.get("height", 0))
    codec = stream.get("codec_name", "?")
    pix_fmt = stream.get("pix_fmt", "?")
    profile = stream.get("profile", "?")
    has_bframes = int(stream.get("has_b_frames", 0))
    r_fps = stream.get("r_frame_rate", "0/0")
    avg_fps = stream.get("avg_frame_rate", "0/0")
    dur_s = float(fmt.get("duration", 0) or 0.0)
    bit_rate = int(fmt.get("bit_rate", 0) or 0)
    color_pri = stream.get("color_primaries", "?")
    color_trc = stream.get("color_transfer", "?")
    color_spc = stream.get("color_space", "?")

    report.add("video.codec", f"{codec} ({profile})")
    report.add("video.size", f"{w}x{h}")
    report.add("video.pix_fmt", pix_fmt)
    report.add("video.fps", f"r={r_fps}  avg={avg_fps}")
    report.add("video.duration", f"{dur_s:.2f} s  ({dur_s/60:.1f} min)")
    report.add("video.bitrate", f"{bit_rate/1000:.0f} kb/s" if bit_rate else "?")

    # B-samples change decode order ≠ presentation order. cv2.VideoCapture
    # and showinfo both honour presentation order; raw rgb24 doesn't always.
    sev = "warn" if has_bframes else "ok"
    report.add(
        "video.b_frames", f"has_b_frames={has_bframes}", sev,
        detail=("B-frames present: raw rgb24 pipe MAY misalign. "
                "Streaming extractor uses cv2.VideoCapture which handles "
                "this. Verify if you see drift." if has_bframes else ""),
    )

    # Colorspace consistency — the reference session's bt601 primaries + bt709 matrix was
    # exactly the trap.
    color_consistent = (color_pri == color_trc == color_spc) or (
        color_pri == "?" and color_trc == "?" and color_spc == "?"
    )
    csev = "ok" if color_consistent else "warn"
    report.add(
        "video.colorspace",
        f"primaries={color_pri}  transfer={color_trc}  matrix={color_spc}",
        csev,
        detail=("Inconsistent colorspace tags. cv2.VideoCapture decodes "
                "via FFmpeg's matrix coefficient and is correct; raw rgb24 "
                "pipe would shift colours uniformly." if not color_consistent else ""),
    )

    # Display Matrix rotation from side_data_list.
    rot_deg: Optional[int] = None
    for sd in stream.get("side_data_list", []) or []:
        if sd.get("side_data_type") == "Display Matrix":
            rv = sd.get("rotation")
            if rv is not None:
                try:
                    rot_deg = int(float(rv))
                except (TypeError, ValueError):
                    pass
    report.add(
        "video.display_rotation",
        f"{rot_deg}°" if rot_deg is not None else "none",
    )

    # Showinfo enumeration — canonical PTS map.
    pts_list = enumerate_source_frames(video)
    n_show = len(pts_list)
    report.add("frames.showinfo_count", str(n_show))

    n_cv2 = _count_cv2_frames(video)
    cv2_sev = "ok" if n_cv2 == n_show else "fail"
    report.add(
        "frames.cv2_count", str(n_cv2), cv2_sev,
        detail=("cv2.VideoCapture diverges from showinfo. Streaming "
                "extraction will mislabel PTS." if cv2_sev == "fail" else ""),
    )

    n_raw = _count_raw_rgb24_frames(video, w, h)
    raw_sev = "ok" if n_raw == n_show else "warn"
    report.add(
        "frames.raw_rgb24_count", str(n_raw), raw_sev,
        detail=("Raw rgb24 pipe diverges from showinfo (this is the "
                "deprecated path; only a problem if any code path still "
                "reads from it)." if raw_sev == "warn" else ""),
    )

    # PTS monotonicity + max gap.
    if len(pts_list) >= 2:
        gaps = [b[1] - a[1] for a, b in zip(pts_list, pts_list[1:])]
        max_gap = max(gaps)
        min_gap = min(gaps)
        n_non_monotonic = sum(1 for g in gaps if g <= 0)
        gap_sev = "fail" if max_gap > MAX_PTS_GAP_HARD_S else "ok"
        report.add(
            "pts.max_gap", f"{max_gap:.3f} s",
            gap_sev,
            detail=(f"Largest PTS gap exceeds {MAX_PTS_GAP_HARD_S} s — "
                    "possible paused recording or VFR pathology."
                    if gap_sev == "fail" else ""),
        )
        report.add("pts.min_gap", f"{min_gap:.6f} s")
        mono_sev = "fail" if n_non_monotonic else "ok"
        report.add(
            "pts.monotonic",
            f"non-monotonic = {n_non_monotonic}", mono_sev,
            detail=("PTS not strictly increasing — decoder reorder leaked "
                    "into the showinfo stream." if mono_sev == "fail" else ""),
        )

        derived_fps = (n_show - 1) / (pts_list[-1][1] - pts_list[0][1]) \
            if pts_list[-1][1] > pts_list[0][1] else 0.0
        report.add(
            "pts.derived_fps",
            f"{derived_fps:.4f}  (showinfo end/start)",
        )

    return pts_list, stream


def audit_recording_txt(rec_txt: Path, video_pts: List[tuple[int, float]],
                        report: AuditReport,
                        video_t0_boottime_ns: Optional[float] = None) -> None:
    if not rec_txt.is_file():
        report.add("anchor.file", str(rec_txt), "fail",
                   "recording_*.txt missing — cannot fit a time anchor.")
        return
    anchor = fit_time_anchor(rec_txt)
    rmse_ms = anchor.rmse_s * 1000.0
    drift_ppm = abs(anchor.drift_ppm)
    rmse_sev = "fail" if rmse_ms > RMSE_HARD_MS else (
        "warn" if rmse_ms > 30.0 else "ok"
    )
    report.add(
        "anchor.rmse_ms", f"{rmse_ms:.3f} ms", rmse_sev,
        detail=(f"OLS time-anchor RMSE > {RMSE_HARD_MS} ms — frame-to-UTC "
                "mapping is unreliable." if rmse_sev == "fail" else ""),
    )
    drift_sev = "fail" if drift_ppm > DRIFT_PPM_HARD else (
        "warn" if drift_ppm > 20.0 else "ok"
    )
    report.add(
        "anchor.drift_ppm", f"{drift_ppm:.2f}", drift_sev,
        detail=("Excessive clock drift — device may be thermally throttling "
                "or the recording_*.txt is corrupted." if drift_sev == "fail" else ""),
    )
    report.add("anchor.n_anchors", str(anchor.n))
    report.add("anchor.n_rejected", str(anchor.n_rejected))
    report.add(
        "anchor.fit_uncertainty_ms",
        f"{anchor.fit_uncertainty_s*1000:.3f}",
    )
    report.add(
        "anchor.cubic_improvement_ms",
        f"{anchor.cubic_rmse_improvement_s*1000:.3f}",
    )
    report.add(
        "anchor.max_abs_ms",
        f"{anchor.max_abs_s*1000:.3f}",
    )

    # Sanity: media PTS range must fall inside anchor coverage.
    if video_pts:
        _f2u = make_frame_to_utc(anchor, video_t0_boottime_ns)
        first_utc = _f2u(video_pts[0][1])
        last_utc = _f2u(video_pts[-1][1])
        report.add(
            "anchor.video_utc_range",
            f"{first_utc:.3f} -> {last_utc:.3f} ({last_utc - first_utc:.1f} s)",
        )


def audit_pos(pos_path: Optional[Path], video_pts: List[tuple[int, float]],
              rec_txt: Path, report: AuditReport,
              video_t0_boottime_ns: Optional[float] = None) -> None:
    if pos_path is None:
        report.add("ppk.file", "(not supplied)", "warn",
                   "No .pos file — skipping PPK coverage audit.")
        return
    if not pos_path.is_file():
        report.add("ppk.file", str(pos_path), "fail",
                   ".pos file path does not exist.")
        return
    rows = parse_rtkpos(pos_path)
    if not rows:
        report.add("ppk.rows", "0", "fail", "Empty .pos file.")
        return
    report.add("ppk.rows", str(len(rows)))
    rows.sort(key=lambda r: r.utc_s)
    report.add(
        "ppk.utc_range",
        f"{rows[0].utc_s:.3f} -> {rows[-1].utc_s:.3f} "
        f"({rows[-1].utc_s - rows[0].utc_s:.1f} s)",
    )
    # Window coverage vs media span.
    if not video_pts:
        return
    if not rec_txt.is_file():
        return
    anchor = fit_time_anchor(rec_txt)
    _f2u = make_frame_to_utc(anchor, video_t0_boottime_ns)
    v_first = _f2u(video_pts[0][1])
    v_last = _f2u(video_pts[-1][1])
    inside_start = rows[0].utc_s <= v_first
    inside_end = rows[-1].utc_s >= v_last
    sev = "ok" if (inside_start and inside_end) else "warn"
    msg = []
    if not inside_start:
        msg.append(f"video starts {rows[0].utc_s - v_first:.1f} s before PPK")
    if not inside_end:
        msg.append(f"video ends {v_last - rows[-1].utc_s:.1f} s after PPK")
    report.add(
        "ppk.video_coverage",
        "fully inside" if sev == "ok" else "; ".join(msg),
        sev,
        detail=("Frames outside the PPK window get no position — "
                "max_interp_gap_s should reject them downstream."
                if sev == "warn" else ""),
    )


def _find_session_files(session_dir: Path) -> tuple[Path, Path, Optional[Path]]:
    mp4_candidates = list(session_dir.glob("recording_*.mp4"))
    txt_candidates = list(session_dir.glob("recording_*.txt"))
    pos_candidates = (
        list(session_dir.glob("*.pos"))
        + list(session_dir.glob("measurements_*_*_base.pos"))
    )
    if not mp4_candidates:
        raise FileNotFoundError(
            f"No recording_*.mp4 in {session_dir}. Pass --video explicitly."
        )
    if not txt_candidates:
        raise FileNotFoundError(
            f"No recording_*.txt in {session_dir}. Pass --recording-txt explicitly."
        )
    pos = pos_candidates[0] if pos_candidates else None
    return mp4_candidates[0], txt_candidates[0], pos


def _print_report(report: AuditReport) -> None:
    width = max(len(f.label) for f in report.findings) + 2
    glyphs = {"ok": "  ", "warn": "! ", "fail": "X "}
    for f in report.findings:
        prefix = glyphs.get(f.severity, "  ")
        line = f"{prefix}{f.label.ljust(width)} {f.value}"
        print(line)
        if f.detail:
            for sub in f.detail.splitlines():
                print(f"     {sub}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path,
                    help="Folder with recording_*.mp4 + recording_*.txt.")
    ap.add_argument("--video", type=Path, default=None,
                    help="Override video path.")
    ap.add_argument("--recording-txt", type=Path, default=None,
                    help="Override recording_*.txt path.")
    ap.add_argument("--pos", type=Path, default=None,
                    help="Override .pos path.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit with code 1 on any 'fail' finding.")
    args = ap.parse_args()

    if args.video is None or args.recording_txt is None:
        mp4, rec, pos = _find_session_files(args.session_dir)
        video = args.video or mp4
        rec_txt = args.recording_txt or rec
        pos_path = args.pos or pos
    else:
        video = args.video
        rec_txt = args.recording_txt
        pos_path = args.pos

    print(f"=== timing audit: {video.name} ===")
    print(f"session dir : {args.session_dir}")
    print(f"video       : {video}")
    print(f"recording   : {rec_txt}")
    print(f"pos         : {pos_path or '(none)'}")
    print()

    # Segment-aware media sample-0 boottime, so the media<->anchor/Post-processing range checks
    # are correct for cut clips (and all boottime sessions), not ~t0 early.
    video_t0_boot: Optional[float] = None
    try:
        from data_pipeline.pipeline import RawInputs
        _raw = RawInputs.from_folder(args.session_dir)
        video_t0_boot = resolve_video_t0_boottime_ns(
            capture_meta=_raw.capture_meta_json,
            video_anchor=_raw.video_anchor_txt,
            chop_video_anchor=(_raw.chop_video_anchor
                               if getattr(_raw, "is_chop", False) else None),
        )
    except Exception:
        video_t0_boot = None

    report = AuditReport()
    pts_list, _stream = audit_video(video, report)
    audit_recording_txt(rec_txt, pts_list, report, video_t0_boottime_ns=video_t0_boot)
    audit_pos(pos_path, pts_list, rec_txt, report, video_t0_boottime_ns=video_t0_boot)
    print()
    _print_report(report)
    print()
    n_fail = sum(1 for f in report.findings if f.severity == "fail")
    n_warn = sum(1 for f in report.findings if f.severity == "warn")
    n_ok = sum(1 for f in report.findings if f.severity == "ok")
    print(f"summary: {n_ok} ok, {n_warn} warn, {n_fail} fail")

    if args.strict and (report.has_failures or n_warn > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
