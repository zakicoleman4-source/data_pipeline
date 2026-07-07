"""Re-upload refined per-sample coordinates and compare them to Signal.

After the pipeline extracts and coordinate-tags samples, the user refines them
in an external tool and gets refined per-sample coordinates back. This CLI
joins that CSV against the Signal-at-sample-time positions and quantifies the
delta (systematic offset vs random scatter).

Signal side, either:

  A) an existing Georef.csv (join by Image label):

     python scripts/compare_frame_coords.py \
         --external-csv refined_coords.csv --georef-csv out/Georef.csv \
         --out out/compare

  B) recompute from raw: per-sample UTC via the time anchor, then interpolate
     the post-processed .pos at each sample time:

     python scripts/compare_frame_coords.py \
         --external-csv refined_coords.csv \
         --frame-times-csv out/extracted_frame_times.csv \
         --recording-map raw/recording_1234.txt \
         --pos out/rover.pos \
         [--measurements raw/measurements_1234.txt] \
         [--capture-meta raw/capture_meta.json] \
         [--video-anchor raw/video_anchor.txt] \
         [--chop-video-anchor raw/chop_x/chop_x.video_anchor.txt] \
         --out out/compare

     For a cut ("segment") clip, --chop-video-anchor is REQUIRED: the segment's
     own anchor min bootNs overrides the parent capture_meta media t0
     (segment PTS are rebased to 0).

External CSV columns are auto-detected (case-insensitive): the image column
is image|label|name|sample|file; coordinates may be datum-based (latitude,
longitude[, altitude]), projected/Grid (easting/x, northing/y + zone or epsg
column, or --epsg/--utm-zone), or cartesian (x_ecef, y_ecef, z_ecef).

Writes <out>/frame_delta.csv and prints a summary.
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
        "--external-csv", required=True, type=Path,
        help="Refined external per-frame coordinates CSV (columns auto-detected)",
    )
    p.add_argument(
        "--epsg", type=int, default=None,
        help="EPSG code of projected external coordinates (when the CSV has "
             "easting/northing but no zone/epsg column)",
    )
    p.add_argument(
        "--utm-zone", type=str, default=None,
        help="UTM zone like '32U' / '10T' for projected external coordinates "
             "(alternative to --epsg)",
    )

    g = p.add_argument_group("GNSS side A: existing Georef.csv")
    g.add_argument("--georef-csv", type=Path, default=None,
                   help="Georef.csv produced by the pipeline (join by Image)")

    r = p.add_argument_group("GNSS side B: recompute from raw")
    r.add_argument("--frame-times-csv", type=Path, default=None,
                   help="extracted_frame_times.csv (Image, t_video_s)")
    r.add_argument("--recording-map", type=Path, default=None,
                   help="recording_*.txt boot->UTC bridge")
    r.add_argument("--measurements", type=Path, default=None,
                   help="measurements_*.txt (time-anchor fallback when "
                        "recording map is empty/missing)")
    r.add_argument("--pos", type=Path, default=None,
                   help="RTKLIB .pos solution to interpolate at frame times")
    r.add_argument("--capture-meta", type=Path, default=None,
                   help="capture_meta.json (optional timeline offset)")
    r.add_argument("--video-anchor", type=Path, default=None,
                   help="video_anchor.txt (optional per-frame boot anchor)")
    r.add_argument("--chop-video-anchor", type=Path, default=None,
                   help="trimmed ('chop') clip's own *.video_anchor.txt — "
                        "REQUIRED when the frames came from a chop: its min "
                        "bootNs overrides the parent capture_meta video t0 "
                        "(chop PTS are rebased to 0)")
    r.add_argument("--video-offset-s", type=float, default=0.0,
                   help="Manual video-GNSS offset in seconds (default 0)")
    r.add_argument("--max-gap-s", type=float, default=2.0,
                   help="Max .pos gap to interpolate across (default 2.0 s)")

    p.add_argument("--out", type=Path, default=Path("."),
                   help="Output directory for frame_delta.csv (default: cwd)")
    return p.parse_args(argv)


def _gnss_from_raw(args: argparse.Namespace) -> dict:
    """Recompute per-sample Signal positions from raw timing files + .pos."""
    from data_pipeline.frame_compare import (
        normalize_image_key,
        warn_duplicate_stems,
    )
    from data_pipeline.parsers import interp_pos, parse_rtkpos
    from data_pipeline.stages.georef import _load_frames

    missing = [
        flag for flag, val in (
            ("--frame-times-csv", args.frame_times_csv),
            ("--pos", args.pos),
        ) if val is None
    ]
    if missing:
        raise SystemExit(
            "recompute-from-raw needs " + " and ".join(missing) +
            " (or use --georef-csv instead)"
        )
    if args.recording_map is None and args.measurements is None:
        raise SystemExit(
            "recompute-from-raw needs --recording-map or --measurements "
            "for the boot->UTC time anchor"
        )

    pos_rows = parse_rtkpos(args.pos)
    if not pos_rows:
        raise SystemExit(f"{args.pos}: no epochs parsed")

    def log(msg: str) -> None:
        print(msg)

    # A missing session map is fine: _load_frames falls back to the
    # measurements-derived anchor when the session file is empty/absent.
    recording_map = args.recording_map or (
        args.frame_times_csv.parent / "__no_recording_map__.txt"
    )
    frames, _anchor = _load_frames(
        args.frame_times_csv,
        recording_map,
        log,
        pos_rows=pos_rows,
        video_offset_s=args.video_offset_s,
        capture_meta=args.capture_meta,
        video_anchor=args.video_anchor,
        chop_video_anchor=args.chop_video_anchor,
        measurements_txt=args.measurements,
    )
    times = [r.utc_s for r in pos_rows]
    gnss: dict = {}
    collided: list = []
    n_no_pos = 0
    for f in frames:
        llh = interp_pos(pos_rows, f.utc_s, args.max_gap_s, times=times)
        if llh is None:
            n_no_pos += 1
            continue
        key = normalize_image_key(f.image)
        if key in gnss:
            collided.append(key)
        gnss[key] = (llh[0], llh[1], llh[2])
    n_collided = warn_duplicate_stems(str(args.frame_times_csv), collided)
    print(f"[gnss] frames={len(frames)} with-position={len(gnss)} "
          f"outside-coverage={n_no_pos}"
          + (f" stem-collisions={n_collided} (last row wins!)"
             if n_collided else ""))
    if not gnss:
        raise SystemExit("no frame received a GNSS position "
                         "(check time anchor inputs and --max-gap-s)")
    return gnss


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    from data_pipeline.frame_compare import (
        compute_deltas,
        format_summary,
        load_external_frame_coords,
        load_gnss_frame_coords_from_georef,
        write_delta_csv,
    )

    external = load_external_frame_coords(
        args.external_csv, epsg=args.epsg, utm_zone=args.utm_zone,
    )
    print(f"[external] {len(external)} refined frame coordinates "
          f"from {args.external_csv}")

    if args.georef_csv is not None:
        gnss = load_gnss_frame_coords_from_georef(args.georef_csv)
        print(f"[gnss] {len(gnss)} frame positions from {args.georef_csv}")
    else:
        gnss = _gnss_from_raw(args)

    result = compute_deltas(external, gnss)
    out_csv = write_delta_csv(result.records, Path(args.out) / "frame_delta.csv")

    print()
    print(format_summary(result.summary))
    print()
    print(f"wrote {out_csv} ({len(result.records)} rows)")
    return 0 if result.records else 2


if __name__ == "__main__":
    raise SystemExit(main())
