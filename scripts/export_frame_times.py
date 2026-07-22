"""Export the per-frame every-time-notation table for a session.

One row per extracted video frame with every clock side by side:

    Image, video_pts_s, boot_ns, utc_s, utc_iso, gpst_s, t_audio_s

Typical runs:

    python scripts/export_frame_times.py --session <raw session dir>
    python scripts/export_frame_times.py --session <dir> \
        --frame-times out/extracted_frame_times.csv --out out/frame_time_table.csv

``--session`` resolves the time anchors (boot->UTC, video t0, audio origin).
``--frame-times`` defaults to ``<session>/extracted_frame_times.csv`` or the
first match found under the session directory. A self-contained sortable
HTML twin is written next to the CSV unless ``--no-html``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _find_frame_times_csv(session: Path) -> Optional[Path]:
    """Locate extracted_frame_times.csv for a session (direct, then search)."""
    direct = session / "extracted_frame_times.csv"
    if direct.is_file():
        return direct
    hits = sorted(session.glob("*/extracted_frame_times.csv"))
    if not hits:
        hits = sorted(session.rglob("extracted_frame_times.csv"))
    return hits[0] if hits else None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--session", type=Path,
        help="Raw session directory (used to resolve the time anchors, and "
             "to find extracted_frame_times.csv when --frame-times is omitted)",
    )
    p.add_argument(
        "--frame-times", type=Path,
        help="extracted_frame_times.csv (Image,t_video_s); defaults to "
             "<session>/extracted_frame_times.csv or a search under --session",
    )
    p.add_argument(
        "--out", type=Path,
        help="Output CSV path (default: frame_time_table.csv next to the "
             "frame-times CSV)",
    )
    p.add_argument(
        "--no-html", action="store_true",
        help="Skip the self-contained sortable HTML twin",
    )
    args = p.parse_args(argv)
    if args.session is None and args.frame_times is None:
        p.error("need --session and/or --frame-times")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    frame_times = args.frame_times
    if frame_times is None:
        frame_times = _find_frame_times_csv(args.session)
        if frame_times is None:
            print(f"ERROR: no extracted_frame_times.csv found under "
                  f"{args.session} (pass --frame-times)")
            return 2
        print(f"[frame-times] using {frame_times}")
    if not Path(frame_times).is_file():
        print(f"ERROR: frame-times CSV not found: {frame_times}")
        return 2

    if args.session is None:
        print("ERROR: --session is required to resolve the time anchors "
              "(boot->UTC, video t0, audio origin)")
        return 2

    out_csv = args.out or Path(frame_times).parent / "frame_time_table.csv"

    from data_pipeline.frame_time_table import build_frame_time_table

    try:
        out = build_frame_time_table(
            Path(frame_times),
            session_dir=Path(args.session),
            out_csv=Path(out_csv),
            write_html=not args.no_html,
            log=print,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    print(f"\nWrote: {out}")
    if not args.no_html:
        print(f"Wrote: {out.with_suffix('.html')}")

    # 3-row preview
    with out.open("r", encoding="utf-8") as f:
        lines = [next(f, "").rstrip("\n") for _ in range(4)]
    print("\nPreview:")
    for ln in lines:
        if ln:
            print("  " + ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
