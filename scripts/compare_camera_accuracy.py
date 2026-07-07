"""Compare camera-model vs GPS per-frame positions against ground truth.

After the pipeline extracts and coordinate-tags frames, the user builds a 3D
reconstruction of those frames in an external tool ("the camera model") and
wants to know which per-frame positions are more accurate against a
survey-grade ground-truth ``.pos`` track: the camera model's, or the GPS
post-processed ones.

Typical run (Georef.csv as the GPS side, GPS .pos to recover frame times):

    python scripts/compare_camera_accuracy.py \
        --colmap recon/sparse/0/images.bin \
        --georef-csv out/Georef.csv \
        --gt-pos survey/truth.pos \
        --pos out/rover.pos \
        --out out/camera_compare

``--colmap`` accepts ``images.txt``, ``images.bin``, or a reconstruction
directory containing either (``sparse/``, ``sparse/0/`` are searched).

Frame UTC times (needed to interpolate the ground truth) are resolved from,
in order: an explicit ``--frame-times`` CSV carrying a UTC column, a time
column inside the Georef.csv, the session time anchor (``--session`` with an
``extracted_frame_times.csv``), or by projecting each frame's GPS position
onto the ``--pos`` track.

Georeferencing of the reconstruction is auto-detected; a raw local-frame
reconstruction is aligned to the GPS track with a 7-parameter similarity fit
(scale + rotation + translation, outlier-rejecting) and the fitted scale +
RMS residual are reported.

Writes ``<out>/camera_vs_gps_vs_gt.csv`` and a self-contained
``camera_vs_gps_vs_gt.html`` report, and prints the verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--colmap", required=True, type=Path,
        help="Reconstruction images.txt / images.bin, or a directory "
             "containing either (sparse/, sparse/0/ are searched)",
    )
    p.add_argument(
        "--gt-pos", required=True, type=Path,
        help="Survey-grade ground-truth .pos track",
    )
    p.add_argument(
        "--georef-csv", type=Path, default=None,
        help="Pipeline Georef.csv with the GPS per-frame positions "
             "(the primary GPS input)",
    )
    p.add_argument(
        "--pos", type=Path, default=None,
        help="GPS post-processed .pos. Used to recover frame UTC times "
             "(track projection) and, when --georef-csv is absent, to "
             "interpolate the GPS per-frame positions directly",
    )
    p.add_argument(
        "--frame-times", type=Path, default=None,
        help="Optional per-frame times CSV. With a UTC column it is used "
             "directly; an Image,t_video_s file needs --session for the "
             "time anchor",
    )
    p.add_argument(
        "--session", type=Path, default=None,
        help="Raw session directory (recording_*.txt / measurements_*.txt "
             "/ capture_meta.json) for the boot->UTC time anchor",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for the CSV + HTML report",
    )
    p.add_argument(
        "--speed-floor", type=float, default=0.5,
        help="Min ground-truth speed (m/s) for azimuth-error frames "
             "(default 0.5)",
    )
    p.add_argument(
        "--max-gap-s", type=float, default=2.0,
        help="Max ground-truth .pos gap to interpolate across (default 2.0)",
    )
    p.add_argument(
        "--max-step-s", type=float, default=5.0,
        help="Max frame-to-frame dt for motion (speed/azimuth) metrics "
             "(default 5.0)",
    )
    p.add_argument(
        "--mode", choices=("auto", "llh", "local", "raw"), default="auto",
        help="Override the reconstruction-frame auto-detection "
             "(default: auto)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    from data_pipeline.photo_compare import build_report

    if args.georef_csv is None and args.pos is None:
        raise SystemExit(
            "need --georef-csv (GPS per-frame positions) or --pos "
            "(GPS .pos to interpolate at frame times)"
        )

    result = build_report(
        args.colmap,
        args.gt_pos,
        args.out,
        georef_csv=args.georef_csv,
        pos=args.pos,
        frame_times=args.frame_times,
        session_dir=args.session,
        speed_floor_mps=args.speed_floor,
        max_gap_s=args.max_gap_s,
        max_step_s=args.max_step_s,
        force_mode=None if args.mode == "auto" else args.mode,
        log=print,
    )

    print()
    print(f"matched frames : {result.summary['n_matched']} "
          f"(camera={result.summary['n_camera']}, "
          f"gps={result.summary['n_gps']}, gt={result.summary['n_gt']})")
    print(f"frame mode     : {result.mode}")
    if result.fit is not None:
        print(f"alignment      : scale={result.fit.scale:.6f} "
              f"rms={result.fit.rms_m:.3f} m "
              f"({result.fit.n_used}/{result.fit.n_total} used)")
    print(f"wrote          : {result.csv_path}")
    if result.html_path is not None:
        print(f"                 {result.html_path}")
    print()
    print(result.verdict)
    return 0 if result.records else 2


if __name__ == "__main__":
    raise SystemExit(main())
