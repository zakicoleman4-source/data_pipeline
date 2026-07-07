"""The external solver AR-config sweep for day14 dodge190336: find a conf that raises the
Q=1 (fix) rate WITHOUT worsening horizontal error vs The reference unit ground truth.

Attacks the ~2 m float-solution bias documented in
docs/findings/fixrate.md / ACCURACY_PLAN_2026-06-29.md: the baseline conf
resolves very few epochs to a fixed (Q=1) ambiguity, so most of the
path rides the noisier float (Q=2) solution.

This script does NOT touch any data_pipeline production module. It only
drives the vendored rnx2rtkp.exe with generated variant .conf files and
scores the resulting .pos files against The reference unit ground truth using
data_pipeline.traj_score.score_trajectories (read-only import).

Usage:
    python scripts/ppk_conf_sweep.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_pipeline.parsers import parse_rtkpos, PosRow  # noqa: E402
from data_pipeline.traj_score import score_trajectories  # noqa: E402

# ---------------------------------------------------------------------------
# Verified inputs (day14 dodge 20260628_190336_677 vs The reference unit GT)
# ---------------------------------------------------------------------------
RNX2RTKP = Path(r"C:\Aj\gps\data_pipeline\vendor\rtklib\rnx2rtkp.exe")
BASE_CONF = Path(
    r"C:\Aj\gps\day14\solved_2026-06-28\dodge\20260628_190336_677\rover.patched.conf"
)
ROVER_OBS = Path(
    r"C:\Aj\gps\day14\solved_2026-06-28\dodge\20260628_190336_677\rover.obs"
)
BASE_OBS = Path(r"C:\Aj\gps\day14\solved_2026-06-28\rinex\base\log0628a.26o")
BASE_NAV = [
    Path(r"C:\Aj\gps\day14\solved_2026-06-28\rinex\base\log0628a.26N"),
    Path(r"C:\Aj\gps\day14\solved_2026-06-28\rinex\base\log0628a.26G"),
    Path(r"C:\Aj\gps\day14\solved_2026-06-28\rinex\base\log0628a.26L"),
    Path(r"C:\Aj\gps\day14\solved_2026-06-28\rinex\base\log0628a.26C"),
]
GT_POS = Path(r"C:/Aj/gps/day14/solved_2026-06-28/gt/gt_log0628a.pos")

# Scratch working dir (never under the repo; never touches data_pipeline).
SCRATCH_ROOT = Path(tempfile.gettempdir()) / "ppk_conf_sweep"


# ---------------------------------------------------------------------------
# Conf handling
# ---------------------------------------------------------------------------
def load_conf_lines(conf_path: Path) -> List[str]:
    return conf_path.read_text(encoding="utf-8").splitlines()


def apply_overrides(lines: List[str], overrides: Dict[str, str]) -> List[str]:
    """Return a new conf line list with ``overrides`` (key -> raw value text,
    no inline comment) applied. Keys not present in the base conf are
    appended. Preserves original ordering/comments for untouched lines."""
    remaining = dict(overrides)
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            comment = ""
            if "#" in line:
                comment = "  # " + line.split("#", 1)[1].strip()
            out.append(f"{key:<19}={remaining.pop(key)}{comment}")
        else:
            out.append(line)
    for key, val in remaining.items():
        out.append(f"{key:<19}={val}")
    return out


def write_conf(lines: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# the solver binary driver
# ---------------------------------------------------------------------------
def run_rnx2rtkp(conf_path: Path, out_pos: Path) -> subprocess.CompletedProcess:
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(RNX2RTKP),
        "-k", str(conf_path),
        "-o", str(out_pos),
        str(ROVER_OBS),
        str(BASE_OBS),
        *[str(p) for p in BASE_NAV],
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=900)


def write_score_csv(rows: List[PosRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gpstime", "lat_deg", "lon_deg"])
        for r in rows:
            w.writerow([r.utc_s, r.lat_deg, r.lon_deg])


def fix_pct(rows: List[PosRow]) -> float:
    if not rows:
        return float("nan")
    c = Counter(r.quality for r in rows)
    return round(100.0 * c.get(1, 0) / len(rows), 2)


# ---------------------------------------------------------------------------
# Variant definitions: vary one axis at a time from the baseline conf, plus
# a few promising combos. Baseline already has pos1-elmask=10,
# pos1-snrmask_L1=18(all bands), pos2-armode=fix-and-hold,
# pos2-gloarmode=fix-and-hold, pos2-arthres=3.
# ---------------------------------------------------------------------------
def snrmask_l1(val: int) -> str:
    return ",".join([str(val)] * 9)


VARIANTS: "list[tuple[str, Dict[str, str]]]" = [
    ("baseline", {}),
    # --- armode axis ---
    ("armode_continuous", {"pos2-armode": "continuous"}),
    # --- arthres axis (baseline is 3.0; try looser 2.0) ---
    ("arthres_2.0", {"pos2-arthres": "2"}),
    # --- gloarmode axis (baseline fix-and-hold) ---
    ("gloarmode_off", {"pos2-gloarmode": "off"}),
    ("gloarmode_on", {"pos2-gloarmode": "on"}),
    # --- elmask axis (baseline 10) ---
    ("elmask_15", {"pos1-elmask": "15"}),
    # --- snrmask axis (baseline effectively 18 for all bands; task asks
    #     30/35 comparison points) ---
    ("snrmask_30", {"pos1-snrmask_r": "on", "pos1-snrmask_b": "on",
                     "pos1-snrmask_L1": snrmask_l1(30)}),
    ("snrmask_35", {"pos1-snrmask_r": "on", "pos1-snrmask_b": "on",
                     "pos1-snrmask_L1": snrmask_l1(35)}),
    # --- combos ---
    ("combo_arthres2_elmask15", {"pos2-arthres": "2", "pos1-elmask": "15"}),
    ("combo_arthres2_gloaron", {"pos2-arthres": "2", "pos2-gloarmode": "on"}),
    ("combo_elmask15_snrmask30", {"pos1-elmask": "15", "pos1-snrmask_r": "on",
                                   "pos1-snrmask_b": "on",
                                   "pos1-snrmask_L1": snrmask_l1(30)}),
]


def main(argv: Optional[List[str]] = None) -> int:
    base_lines = load_conf_lines(BASE_CONF)
    gt_rows = parse_rtkpos(GT_POS)
    gt_csv = SCRATCH_ROOT / "gt_ref.csv"
    write_score_csv(gt_rows, gt_csv)

    results = []
    header = f"{'conf':28} {'fix%':>7} {'2sigma_m':>9} {'MAX_m':>8} {'<=1m%':>7} {'n':>6}"
    print(header)
    print("-" * len(header))

    for name, overrides in VARIANTS:
        variant_dir = SCRATCH_ROOT / name
        conf_path = variant_dir / f"{name}.conf"
        out_pos = variant_dir / f"{name}.pos"

        lines = apply_overrides(base_lines, overrides) if overrides else base_lines
        write_conf(lines, conf_path)

        try:
            proc = run_rnx2rtkp(conf_path, out_pos)
        except subprocess.TimeoutExpired as e:
            print(f"{name:28} TIMEOUT: {e}")
            results.append((name, overrides, None))
            continue

        if not out_pos.exists() or out_pos.stat().st_size == 0:
            print(f"{name:28} BLOCKED: no output. stderr={proc.stderr[:500]!r}")
            results.append((name, overrides, None))
            continue

        rows = parse_rtkpos(out_pos)
        if not rows:
            print(f"{name:28} BLOCKED: 0 parsed rows. stderr={proc.stderr[:500]!r}")
            results.append((name, overrides, None))
            continue

        fp = fix_pct(rows)
        test_csv = variant_dir / f"{name}_test.csv"
        write_score_csv(rows, test_csv)
        s = score_trajectories(gt_csv, test_csv)

        print(f"{name:28} {fp:>7.2f} {s['two_sigma_m']:>9.4f} "
              f"{s['max_m']:>8.4f} {s['le1m_pct']:>7.2f} {s['n']:>6}")
        results.append((name, overrides, {"fix_pct": fp, **s}))

    # --- pick winner: raises fix% vs baseline AND does not worsen MAX ---
    baseline_result = next((r for n, o, r in results if n == "baseline"), None)
    if baseline_result is None:
        print("\nBLOCKED: baseline run failed, cannot select a winner.")
        return 1

    base_fix = baseline_result["fix_pct"]
    base_max = baseline_result["max_m"]
    print(f"\nBaseline: fix%={base_fix} MAX={base_max}")

    winners = [
        (n, o, r) for n, o, r in results
        if r is not None and n != "baseline"
        and r["fix_pct"] > base_fix and r["max_m"] <= base_max
    ]
    if winners:
        winners.sort(key=lambda t: (-t[2]["fix_pct"], t[2]["max_m"]))
        wn, wo, wr = winners[0]
        print(f"WINNER: {wn} overrides={wo} fix%={wr['fix_pct']} "
              f"MAX={wr['max_m']} 2sigma={wr['two_sigma_m']}")
    else:
        print("No winner: no variant raised fix% without worsening MAX "
              "vs Javad (observation-capped).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
