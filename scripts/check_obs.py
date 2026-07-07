"""CLI: check a Interchange-format 3 obs file for usable fine measurements before Post-processing.

Usage::

    python scripts/check_obs.py <subject.obs>

Exit code 0 when fine measurements is present, 2 when absent (Live-correction/Post-processing
impossible - recapture with "Force full Signal measurements"), 1 on error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_pipeline.obs_check import check_carrier_phase  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check a RINEX 3 obs file for usable carrier phase (L*)."
    )
    ap.add_argument("obs", type=Path, help="Path to the rover .obs file")
    args = ap.parse_args(argv)

    if not args.obs.is_file():
        print(f"error: {args.obs} not found", file=sys.stderr)
        return 1

    try:
        report = check_carrier_phase(args.obs)
    except Exception as e:
        print(f"error: could not parse {args.obs}: {e}", file=sys.stderr)
        return 1

    print(f"file          : {args.obs}")
    print(f"has_phase     : {report.has_phase}")
    print(f"phase nonzero : {report.n_phase_nonzero} / {report.n_sat_obs} sat-phase slots")
    print("per system    :")
    for sys_letter, st in sorted(report.per_system.items()):
        types = ",".join(st["phase_types"]) or "(none)"
        print(
            f"  {sys_letter}: {st['n_phase_nonzero']:>8} / {st['n_sat_obs']:<8}"
            f" nonzero  phase types: {types}"
        )
    print()
    print(report.message)
    return 0 if report.has_phase else 2


if __name__ == "__main__":
    raise SystemExit(main())
