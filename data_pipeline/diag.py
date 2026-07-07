"""Diagnostic tools for auditing the time-sync pipeline.

Run with::

    python -m data_pipeline.diag timing --recording-map RAW\\recording_*.txt

It prints a full audit (anchor count, drift, RMSE, MAD, cubic-vs-linear
improvement, rejected-outlier count) and optionally writes
``time_anchor_residuals.csv`` so you can plot residuals vs media time and
visually verify the fit is clean.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .time_sync import fit_time_anchor, per_anchor_residuals


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 100:
        return s[-1]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    devs = sorted(abs(v - med) for v in values)
    return devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])


def cmd_timing(args: argparse.Namespace) -> int:
    rec: Path = args.recording_map
    print(f"# time-sync audit  --  {rec}")
    print()
    raw = fit_time_anchor(rec, robust=False)
    rob = fit_time_anchor(rec, robust=True)

    print(f"raw fit (no outlier rejection):")
    print(f"  n                       = {raw.n}")
    print(f"  drift                   = {raw.drift_ppm:+.3f} ppm")
    print(f"  per-anchor RMSE         = {raw.rmse_s * 1e3:8.3f} ms")
    print(f"  per-anchor max-abs      = {raw.max_abs_s * 1e3:8.3f} ms")
    print(f"  fit uncertainty         = {raw.fit_uncertainty_s * 1e3:8.3f} ms (rmse / sqrt(n))")
    print(f"  cubic-vs-linear gain    = {raw.cubic_rmse_improvement_s * 1e3:8.3f} ms (smaller = more linear)")
    print()
    print(f"robust fit (5*MAD outlier rejection):")
    print(f"  n_kept / n_rejected     = {rob.n} / {rob.n_rejected}")
    print(f"  drift                   = {rob.drift_ppm:+.3f} ppm")
    print(f"  per-anchor RMSE         = {rob.rmse_s * 1e3:8.3f} ms")
    print(f"  per-anchor max-abs      = {rob.max_abs_s * 1e3:8.3f} ms")
    print(f"  fit uncertainty         = {rob.fit_uncertainty_s * 1e3:8.3f} ms")
    print(f"  cubic-vs-linear gain    = {rob.cubic_rmse_improvement_s * 1e3:8.3f} ms")
    print()

    residuals_all = per_anchor_residuals(rec, rob)
    rs = [r for _t, r in residuals_all]
    abs_rs = [abs(r) for r in rs]
    pct = [_percentile(abs_rs, q) for q in (50, 90, 95, 99)]
    print("residual distribution (|y - yhat|, robust fit):")
    print(f"  median (p50)            = {pct[0] * 1e3:8.3f} ms")
    print(f"  p90                     = {pct[1] * 1e3:8.3f} ms")
    print(f"  p95                     = {pct[2] * 1e3:8.3f} ms")
    print(f"  p99                     = {pct[3] * 1e3:8.3f} ms")
    print(f"  MAD                     = {_mad(rs) * 1e3:8.3f} ms")
    print()
    drift_over_span = rob.slope * 1e9 - 1.0
    span_s = max(t for t, _r in residuals_all) - min(
        t for t, _r in residuals_all
    )
    print("interpretation:")
    print(
        f"  - The slope absorbs camera/system drift exactly. Over the {span_s:.1f}s")
    print(
        f"    recording span, the naive single-anchor approach would have"
    )
    print(
        f"    accumulated {abs(drift_over_span) * span_s * 1e3:.2f} ms of error;"
    )
    print(f"    the regression removes that.")
    print(
        f"  - rmse/sqrt(n) = {rob.fit_uncertainty_s * 1e3:.3f} ms is the 1-sigma"
    )
    print(
        f"    uncertainty in the fit-predicted UTC at any video timestamp."
    )
    if rob.cubic_rmse_improvement_s * 1e3 > 2.0:
        print(
            "  - WARNING: cubic fit lowers RMSE by >2 ms. The camera/system"
        )
        print(
            "    relationship is not perfectly linear; consider a polynomial fit."
        )
    else:
        print("  - Linear model is sufficient (cubic fit doesn't help).")

    if args.dump_csv:
        out = args.dump_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # anchor_x_s = the time-anchor x value (device tick, = bootNs/1e9 for
            # boottime sessions), NOT a media PTS. Do not derive sample times from it.
            w.writerow(["anchor_x_s", "residual_s", "residual_ms"])
            for t, r in residuals_all:
                w.writerow([f"{t:.6f}", f"{r:.6f}", f"{r * 1e3:.3f}"])
        print(f"\nWrote per-anchor residuals to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pt = sp.add_parser("timing", help="audit the recording_*.txt time anchors")
    pt.add_argument("--recording-map", required=True, type=Path)
    pt.add_argument(
        "--dump-csv",
        type=Path,
        default=None,
        help="write per-anchor residuals to this CSV for external plotting.",
    )
    pt.set_defaults(func=cmd_timing)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
