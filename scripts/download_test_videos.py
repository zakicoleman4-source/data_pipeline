"""Download a curated set of test media files for the timing audit suite.

Each entry below is hand-picked to exercise a specific trap the pipeline
must survive on real-world device footage:

* H.264 8-bit at 360p, 720p, 1080p, 4K — resolution scaling sanity
* H.265 / HEVC at multiple resolutions — modern device default
* VP9 in Container file container — The platform Cell raw sessions
* 30 fps progressive, 29.97 fps interlaced — codec-fps interactions
* Device-shot drive footage — variable scene content for Keypoint

Files land under ``test_fixtures/media files/`` relative to the repo root
(override with ``--dest`` or ``DTF_TEST_FIXTURES_DIR`` env var). Existing
files are skipped (resumable). After downloading, each file is
ff-probed and a ``fixtures.json`` manifest is written alongside.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pipeline.ffmpeg_paths import resolve_ffprobe  # noqa: E402


DEST_DEFAULT = Path(
    os.environ.get("DTF_TEST_FIXTURES_DIR",
                   str(_REPO / "test_fixtures" / "videos"))
)


# (url, local_name, trap_tag)
CATALOG: list[tuple[str, str, str]] = [
    # samplelib.com — clean H.264 traffic clips, resolution sweep
    ("https://samplelib.com/lib/preview/mp4/sample-5s.mp4",
     "samplelib_5s_1080p_h264.mp4", "h264-1080p-short"),
    ("https://samplelib.com/lib/preview/mp4/sample-5s-720p.mp4",
     "samplelib_5s_720p_h264.mp4", "h264-720p-short"),
    ("https://samplelib.com/lib/preview/mp4/sample-5s-360p.mp4",
     "samplelib_5s_360p_h264.mp4", "h264-360p-short"),
    ("https://samplelib.com/lib/preview/mp4/sample-30s.mp4",
     "samplelib_30s_1080p_h264.mp4", "h264-1080p-medium"),
    ("https://samplelib.com/lib/preview/mp4/sample-30s-720p.mp4",
     "samplelib_30s_720p_h264.mp4", "h264-720p-medium"),
    # NOTE: samplelib's "sample-10s-2160p.container file" is mis-labelled on their
    # server — the file actually delivered is 640x360 at 30 fps, ~277 KB.
    # Kept under its honest name; real 4K coverage comes from the
    # jellyfish 1080p high-bitrate clip + user-supplied device samples.
    ("https://samplelib.com/lib/preview/mp4/sample-10s-2160p.mp4",
     "samplelib_10s_mislabelled_360p_h264.mp4", "samplelib-mislabelled"),
    ("https://samplelib.com/lib/preview/mp4/sample-10s-h265.mp4",
     "samplelib_10s_720p_h265.mp4", "h265-720p"),
    ("https://samplelib.com/lib/preview/mp4/sample-10s-vp9.mp4",
     "samplelib_10s_720p_vp9.mp4", "vp9-mp4-container"),

    # test-media files.co.uk — Jellyfish high-bitrate H.265 (stress test)
    ("https://test-videos.co.uk/vids/jellyfish/mp4/h265/1080/"
     "Jellyfish_1080_10s_30MB.mp4",
     "jellyfish_1080_10s_h265_high.mp4", "h265-1080p-highbr"),
    ("https://test-videos.co.uk/vids/jellyfish/mp4/h265/720/"
     "Jellyfish_720_10s_30MB.mp4",
     "jellyfish_720_10s_h265_high.mp4", "h265-720p-highbr"),

    # test-media files.co.uk — Big Buck Bunny (animated, no B-samples in their build)
    ("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h265/1080/"
     "Big_Buck_Bunny_1080_10s_20MB.mp4",
     "bbb_1080_10s_h265.mp4", "h265-1080p-animation"),

    # test-media files.co.uk — H.264 1080p 29.97 fps (matches device cadence)
    ("https://test-videos.co.uk/vids/jellyfish/mp4/h264/1080/"
     "Jellyfish_1080_10s_30MB.mp4",
     "jellyfish_1080_10s_h264_high.mp4", "h264-1080p-highbr"),
    ("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/"
     "Big_Buck_Bunny_1080_10s_30MB.mp4",
     "bbb_1080_10s_h264.mp4", "h264-1080p-animation"),
]


def download(url: str, dest: Path, log) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"  exists: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    log(f"  fetch:  {url} -> {dest.name}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (data_pipeline audit)"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        dest.write_bytes(data)
        log(f"          {len(data):,} bytes")
        return True
    except urllib.error.HTTPError as e:
        log(f"  HTTPError {e.code} on {url}")
    except urllib.error.URLError as e:
        log(f"  URLError {e.reason} on {url}")
    except Exception as e:  # noqa: BLE001
        log(f"  ERROR {type(e).__name__}: {e}")
    if dest.is_file() and dest.stat().st_size == 0:
        dest.unlink()
    return False


def ffprobe_summary(video: Path) -> dict:
    """Return a compact stream summary that highlights the traps we audit."""
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
    stream = (j.get("streams") or [{}])[0]
    fmt = j.get("format") or {}
    sd = stream.get("side_data_list") or []
    rot: Optional[int] = None
    for d in sd:
        if d.get("side_data_type") == "Display Matrix":
            try:
                rot = int(float(d.get("rotation", 0)))
                break
            except (TypeError, ValueError):
                pass
    return {
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pix_fmt": stream.get("pix_fmt"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "has_b_frames": stream.get("has_b_frames"),
        "color_primaries": stream.get("color_primaries"),
        "color_transfer": stream.get("color_transfer"),
        "color_space": stream.get("color_space"),
        "rotation_deg": rot,
        "duration_s": float(fmt.get("duration") or 0.0),
        "bit_rate_kbps": int(fmt.get("bit_rate") or 0) // 1000,
        "size_mb": video.stat().st_size / 1024 / 1024,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT,
                    help=f"Destination folder (default: {DEST_DEFAULT})")
    args = ap.parse_args()
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    def log(m: str) -> None:
        print(m, flush=True)

    log(f"=== download_test_videos ===")
    log(f"dest: {dest}")
    log(f"catalog: {len(CATALOG)} entries")
    print()

    manifest: list[dict] = []
    for url, name, trap in CATALOG:
        log(f"[{trap}]")
        local = dest / name
        ok = download(url, local, log)
        entry = {"name": name, "url": url, "trap_tag": trap,
                 "downloaded": ok}
        if ok:
            entry.update(ffprobe_summary(local))
        manifest.append(entry)
        print()

    out_json = dest / "fixtures.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    log(f"manifest -> {out_json}")
    log(f"successful: {sum(1 for e in manifest if e.get('downloaded'))}/{len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
