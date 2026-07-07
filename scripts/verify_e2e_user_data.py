"""
End-to-end verification on a real RAW session + Post-processing .pos.

Set environment variables before running:
    DTF_E2E_RAW_DIR  — path to a the source app RAW folder
    DTF_E2E_POS_FILE — path to the Post-processing .pos file

Uses a short media clip for speed; exercises Interchange-format, samples, Coordinate output CSV, viewers.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("DTF_E2E_RAW_DIR", ""))
POS = Path(os.environ.get("DTF_E2E_POS_FILE", ""))
OUT = REPO / "_verify_run"
CLIP_DUR_S = 45.0
FPS = 4.0


def main() -> int:
    sys.path.insert(0, str(REPO))
    from data_pipeline.ffmpeg_paths import resolve_ffmpeg
    from data_pipeline.pipeline import RawInputs
    from data_pipeline.stages import frames, georef, rinex, viewers

    if not RAW.is_dir():
        print("FAIL: RAW folder missing. Set DTF_E2E_RAW_DIR env var.", RAW)
        return 1
    if not POS.is_file():
        print("FAIL: .pos missing. Set DTF_E2E_POS_FILE env var.", POS)
        return 1

    raw = RawInputs.from_folder(RAW)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Short clip (same PTS domain as full file start — time sync valid)
    clip = OUT / "clip.mp4"
    if not clip.is_file() or clip.stat().st_size < 1_000_000:
        print("[clip] ffmpeg -t", CLIP_DUR_S, "s …")
        subprocess.run(
            [
                resolve_ffmpeg(),
                "-y",
                "-loglevel",
                "error",
                "-ss",
                "0",
                "-i",
                str(raw.recording_mp4),
                "-t",
                str(CLIP_DUR_S),
                "-c",
                "copy",
                str(clip),
            ],
            check=True,
        )
    print("[clip] OK", clip, "size", clip.stat().st_size)

    # 2) Interchange-format
    obs = OUT / "session.obs"
    rinex.run(
        measurements_txt=raw.measurements_txt,
        output_obs=obs,
        android_rinex_src=None,
        log=print,
    )
    if not obs.is_file() or obs.stat().st_size < 100:
        print("FAIL: RINEX output empty or missing", obs)
        return 1
    print("[rinex] OK", obs.stat().st_size, "bytes")

    # 3) Samples
    fr = frames.run(video=clip, out_dir=OUT, fps=FPS, fmt="png", log=print)
    if fr.frame_count < 10:
        print("FAIL: too few frames", fr.frame_count)
        return 1
    print("[frames] OK", fr.frame_count, "frames ->", fr.frame_times_csv)

    # 4) Coordinate output CSV (lat/lon only default, data log = measurements for Fix)
    csv_path = OUT / "georef.csv"
    csv_res = georef.run(
        frame_times_csv=fr.frame_times_csv,
        recording_map=raw.recording_txt,
        pos_file=POS,
        data_log=raw.measurements_txt,
        out_csv=csv_path,
        fps=FPS,
        options=georef.CsvOptions(
            smoothing="car",
            include_altitude=False,
            add_ypr=False,
        ),
        log=print,
    )
    if csv_res.n_with_position < 5:
        print("FAIL: almost no positions", csv_res)
        return 1
    print("[georef] OK positioned", csv_res.n_with_position, "/", csv_res.n_frames)

    # 5) Viewers
    viewers.build_trajectory_viewer(
        data_log=raw.measurements_txt,
        georef_csv=csv_path,
        out_html=OUT / "trajectory_viewer.html",
        recording_map=raw.recording_txt,
        log=print,
    )
    viewers.build_comparison_viewer(
        data_log=raw.measurements_txt,
        pos_file=POS,
        frame_times_csv=fr.frame_times_csv,
        recording_map=raw.recording_txt,
        out_html=OUT / "comparison_viewer.html",
        fps=FPS,
        log=print,
    )
    viewers.build_sync_player(
        video=clip,
        pos_file=POS,
        frame_times_csv=fr.frame_times_csv,
        recording_map=raw.recording_txt,
        out_html=OUT / "sync_player.html",
        log=print,
    )
    for name in ("trajectory_viewer.html", "comparison_viewer.html", "sync_player.html"):
        p = OUT / name
        if not p.is_file():
            print("FAIL: missing", p)
            return 1
        if not (OUT / "plotly.min.js").is_file():
            print("FAIL: plotly not vendored next to HTML")
            return 1

    sp = (OUT / "sync_player.html").read_text(encoding="utf-8")
    if "clip.mp4" not in sp and '"clip.mp4"' not in sp:
        print("WARN: sync_player may not reference clip.mp4 relatively")

    print()
    print("ALL STAGES PASSED.")
    print("Output folder:", OUT)
    print("Open in browser: trajectory_viewer.html, comparison_viewer.html, sync_player.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
