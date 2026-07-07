"""Synthesise pathological-but-legal Container file variants from a clean base clip.

Some device-format traps don't show up in random sample downloads — we have
to construct them. The variants below all derive from the smallest H.264
clip already on disk so this is fast and offline.

Variants produced (all under ``test_fixtures/media files/synthetic/``):

* ``rotate_90.container file``    — same content, Display Matrix rotation = 90
* ``rotate_180.container file``   — Display Matrix rotation = 180
* ``rotate_270.container file``   — Display Matrix rotation = 270
* ``bt601_pure.container file``   — all colorspace tags pinned to bt470bg / smpte170m
* ``bt709_pure.container file``   — all colorspace tags pinned to bt709
* ``vfr_paused.container file``   — concat copy that simulates a pause (large PTS gap)
* ``hevc_main10.container file``  — re-encoded to HEVC 10-bit (yuv420p10le)

The script writes a ``synthetic_fixtures.json`` manifest alongside.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pipeline.ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe  # noqa: E402


_FIXTURES_DIR = Path(
    os.environ.get("DTF_TEST_FIXTURES_DIR",
                   str(_REPO / "test_fixtures" / "videos"))
)
BASE = _FIXTURES_DIR / "samplelib_5s_360p_h264.mp4"
OUT_DIR = _FIXTURES_DIR / "synthetic"


def _run(cmd: list[str], log) -> bool:
    log(f"    cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"    FAILED ({r.returncode}): {r.stderr.strip()[-400:]}")
        return False
    return True


def _ffprobe_summary(video: Path) -> dict:
    out = subprocess.run(
        [resolve_ffprobe(), "-v", "error",
         "-select_streams", "v:0",
         "-show_streams", "-show_format",
         "-print_format", "json",
         str(video)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return {"error": out.stderr.strip()}
    j = json.loads(out.stdout)
    s = (j.get("streams") or [{}])[0]
    sd = s.get("side_data_list") or []
    rot = None
    for d in sd:
        if d.get("side_data_type") == "Display Matrix":
            try:
                rot = int(float(d.get("rotation", 0)))
                break
            except (TypeError, ValueError):
                pass
    return {
        "codec": s.get("codec_name"),
        "width": s.get("width"),
        "height": s.get("height"),
        "pix_fmt": s.get("pix_fmt"),
        "color_primaries": s.get("color_primaries"),
        "color_transfer": s.get("color_transfer"),
        "color_space": s.get("color_space"),
        "rotation_deg": rot,
        "has_b_frames": s.get("has_b_frames"),
        "size_mb": video.stat().st_size / 1024 / 1024,
    }


def make_rotate(base: Path, out: Path, deg: int, log) -> bool:
    """KNOWN LIMITATION — does not actually attach a Display Matrix.

    Multiple injection paths were attempted on the external converter 8.1.x:

    * Legacy ``-metadata:s:v:0 rotate=N`` — silently dropped in 8.x.
    * ``-display_rotation:v:0 <deg>`` on input or output — accepted but
      not written into the container file tkhd matrix on stream copy or re-encode.
    * ``h264_metadata`` BSF with ``display_orientation`` / ``rotate`` —
      inserts a Display Orientation SEI into the H.264 stream itself,
      but the probe tool / cv2 read the container file Display Matrix at the container
      track level, so the SEI is not surfaced.

    Reliable rotation-tagged fixture is the real device session
    ``recording_20260505_152247_615.container file`` (reference session, ``rotation=-90``).
    These synthetic outputs are kept as plain-copy controls so the
    fixture set still has clea stream-copy reference points.
    """
    log(f"  rotate {deg}° -> {out.name}   (KNOWN: tag not injectable via ffmpeg CLI)")
    return _run([
        resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(base),
        "-c", "copy",
        str(out),
    ], log)


def make_colorspace(base: Path, out: Path, label: str, log) -> bool:
    log(f"  colorspace {label} -> {out.name}")
    if label == "bt601":
        pri = "bt470bg"; trc = "smpte170m"; spc = "smpte170m"
    elif label == "bt709":
        pri = "bt709"; trc = "bt709"; spc = "bt709"
    else:
        return False
    # Re-encode (light: crf 23) so we can pin colorspace flags.
    return _run([
        resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(base),
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-color_primaries", pri, "-color_trc", trc, "-colorspace", spc,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out),
    ], log)


def make_vfr_paused(base: Path, out: Path, log) -> bool:
    """Concatenate base+base with a synthetic gap to make a multi-second
    PTS jump. Produces a single container file whose PTS sequence has a real
    discontinuity in the middle — the kind of pathology reference session has at
    n=4→n=5 but exaggerated to multi-second."""
    log(f"  vfr_paused -> {out.name}")
    # Build a concat file with two copies of base + a 5-second offset on the
    # second segment. We use the concat filter to splice with explicit
    # setpts, producing a real gap in PTS rather than a re-timestamped
    # contiguous run.
    cmd = [
        resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(base),
        "-i", str(base),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[v0];"
        "[1:v]setpts=PTS-STARTPTS+10/TB[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    return _run(cmd, log)


def make_hevc_main10(base: Path, out: Path, log) -> bool:
    log(f"  hevc main10 -> {out.name}")
    return _run([
        resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(base),
        "-c:v", "libx265", "-crf", "26", "-preset", "veryfast",
        "-pix_fmt", "yuv420p10le",
        "-tag:v", "hvc1",
        "-c:a", "copy",
        str(out),
    ], log)


def main() -> int:
    if not BASE.is_file():
        print(f"base clip missing: {BASE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(m: str) -> None:
        print(m, flush=True)

    log(f"=== synthesise_trap_videos ===")
    log(f"base: {BASE.name} ({BASE.stat().st_size:,} bytes)")
    log(f"out : {OUT_DIR}")
    print()

    jobs = [
        ("rotate_90.mp4",    lambda out: make_rotate(BASE, out, 90, log),   "rotation-90"),
        ("rotate_180.mp4",   lambda out: make_rotate(BASE, out, 180, log),  "rotation-180"),
        ("rotate_270.mp4",   lambda out: make_rotate(BASE, out, 270, log),  "rotation-270"),
        ("bt601_pure.mp4",   lambda out: make_colorspace(BASE, out, "bt601", log), "colorspace-bt601"),
        ("bt709_pure.mp4",   lambda out: make_colorspace(BASE, out, "bt709", log), "colorspace-bt709"),
        ("vfr_paused.mp4",   lambda out: make_vfr_paused(BASE, out, log),   "vfr-multi-second-gap"),
        ("hevc_main10.mp4",  lambda out: make_hevc_main10(BASE, out, log),  "h265-main10-10bit"),
    ]

    manifest: list[dict] = []
    for name, fn, trap in jobs:
        out = OUT_DIR / name
        log(f"[{trap}]")
        ok = fn(out) if not out.is_file() else True
        if out.is_file() and out.stat().st_size > 0:
            log(f"  exists: {out.name} ({out.stat().st_size:,} bytes)")
            entry = {"name": name, "trap_tag": trap, "ok": True,
                     **_ffprobe_summary(out)}
        else:
            entry = {"name": name, "trap_tag": trap, "ok": False}
        manifest.append(entry)
        print()

    out_json = OUT_DIR / "synthetic_fixtures.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    log(f"manifest -> {out_json}")
    log(f"successful: {sum(1 for e in manifest if e.get('ok'))}/{len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
