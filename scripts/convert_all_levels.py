#!/usr/bin/env python3
"""Run gnsslogger_to_rnx at all three strictness presets on one .txt.

Writes three sibling ``.obs`` files (``_strict.obs``, ``_relaxed.obs``,
``_permissive.obs``) next to the input and prints a side-by-side
comparison so the operator can pick the preset their device needs.

Usage:
    python convert_all_levels.py "C:/path/to/measurements_xxx.txt"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ANDROID_RINEX_SRC = SCRIPT_DIR.parent / "vendor" / "android_rinex" / "src"


def count_obs(obs_path: Path) -> tuple[int, int]:
    """Return ``(epoch_count, sat_obs_count)`` from a Interchange-format-3 .obs file."""
    epochs = 0
    sats = 0
    in_body = False
    with obs_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not in_body:
                if "END OF HEADER" in line:
                    in_body = True
                continue
            if line.startswith(">"):
                epochs += 1
            elif line[:1] in "GREICJSI":
                sats += 1
    return epochs, sats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_txt", type=Path,
                    help="Path to a measurements_*.txt from the capture app")
    args = ap.parse_args()

    if not args.input_txt.is_file():
        print(f"Input not found: {args.input_txt}", file=sys.stderr)
        return 2

    converter = ANDROID_RINEX_SRC / "gnsslogger_to_rnx.py"
    if not converter.is_file():
        print(f"Converter not found at {converter}", file=sys.stderr)
        return 2

    in_path = args.input_txt.resolve()
    stem = in_path.with_suffix("")  # ``measurements_xxx`` (no extension)

    results = []
    for level in ("strict", "relaxed", "permissive"):
        out_path = Path(f"{stem}_{level}.obs")
        print(f"\n--- {level} -> {out_path.name} ---")
        rc = subprocess.run(
            [sys.executable, str(converter),
             str(in_path), "-o", str(out_path),
             "--keep-level", level],
            cwd=str(ANDROID_RINEX_SRC),
        ).returncode
        if rc != 0 or not out_path.is_file():
            print(f"  FAILED (exit {rc})")
            results.append((level, None, None, None))
            continue
        size = out_path.stat().st_size
        ep, sat = count_obs(out_path)
        results.append((level, size, ep, sat))

    print("\nSummary")
    print("-------")
    print(f"{'level':<12} {'size':>12} {'epochs':>8} {'sat-obs':>10}")
    for level, size, ep, sat in results:
        if size is None:
            print(f"{level:<12} {'FAILED':>12}")
        else:
            print(f"{level:<12} {f'{size/1e6:.2f} MB':>12} {ep:>8} {sat:>10}")
    print(f"\nInputs at:  {in_path}")
    print(f"Outputs in: {in_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
