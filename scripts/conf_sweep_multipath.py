"""Post-processing config sweep: elevation-mask x SNR-mask grid, scored for accuracy
(vs optional ground truth) AND environment noise resistance (GT-free), with a
weighted ranking that emits the single best config + its .pos.

For ONE subject .obs file, generate a grid of The external solver .conf variants from
``data_pipeline/configs/javad.conf`` (elmask x SNR threshold), run
``the solver binary`` per variant, then score each:

* accuracy (needs --gt-pos): ``data_pipeline.traj_score.score_trajectories``
  -> two_sigma_m (lower=better), max_m (lower=better)
* environment noise resistance (GT-free, from the ``.pos.stat`` residual file):
  median / 95th-pct of |coarse measurement residual| (lower=better), mean SNR
  (higher=better, reported only), fix rate %Q==1 (higher=better)

Combined score (LOWER = BETTER). Each metric is min-max normalized to
[0,1] and oriented so 0 = best:

    score = w_accuracy  * (norm(two_sigma_m) + norm(max_m)) / 2
          + w_multipath * (norm(p_resid_p95) + norm(100 - fix_pct)) / 2

Without --gt-pos the accuracy term is dropped and only the environment noise term
is used.

Outputs (under --out):
    <variant>/           per-variant scratch (conf, pos, pos.stat, csv)
    sweep_results.csv    full ranked table
    best.conf, best.pos  the winning config + solution

Usage:
    python scripts/conf_sweep_multipath.py --rover-obs R.obs --base-obs B.obs \
        --nav N1 [N2 ...] --out OUTDIR [--gt-pos GT.pos] \
        [--elmask-grid 5,10,15,20] [--snr-grid off,30,33,35,38] \
        [--rnx2rtkp EXE] [--w-accuracy 0.5] [--w-multipath 0.5] \
        [--max-dt-s 0.05]

This script never edits data_pipeline production modules (read-only imports).
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_BASE_CONF = REPO / "data_pipeline" / "configs" / "javad.conf"
FALLBACK_RNX2RTKP = REPO / "vendor" / "rtklib" / "rnx2rtkp.exe"
RNX2RTKP_TIMEOUT_S = 900

# One entry per 5-degree elevation bin (0-5 .. 40-45+), per The external solver snrmask fmt.
SNRMASK_BINS = 9


# ---------------------------------------------------------------------------
# PURE helpers (unit-testable without The external solver / data_pipeline)
# ---------------------------------------------------------------------------
def snrmask_str(threshold_db_hz: Union[int, float]) -> str:
    """Broadcast a scalar dB-Hz threshold into The external solver's 9-value snrmask string
    (one threshold per 5-degree elevation bin)."""
    v = threshold_db_hz
    txt = str(int(v)) if float(v).is_integer() else str(v)
    return ",".join([txt] * SNRMASK_BINS)


@dataclass(frozen=True)
class Variant:
    """One grid point: quality mask (deg) x SNR threshold ('off' or dB-Hz)."""
    name: str
    elmask_deg: float
    snr: str                                # "off" or numeric text e.g. "33"
    overrides: Dict[str, str] = field(default_factory=dict)


def build_overrides(elmask_deg: float, snr: str) -> Dict[str, str]:
    """Conf-key overrides for one grid point. Always forces
    ``out-outstat=residual`` so a .pos.stat is produced for environment noise scoring."""
    el_txt = str(int(elmask_deg)) if float(elmask_deg).is_integer() else str(elmask_deg)
    ov: Dict[str, str] = {
        # quality mask: all three keys move together (per javad_el*.conf).
        "pos1-elmask": el_txt,
        "pos2-arelmask": el_txt,
        "pos2-elmaskhold": el_txt,
        # required for environment noise scoring (.pos.stat with $Source residual rows)
        "out-outstat": "residual",
    }
    if str(snr).lower() == "off":
        ov["pos1-snrmask_r"] = "off"
        ov["pos1-snrmask_b"] = "off"
    else:
        mask = snrmask_str(float(snr))
        ov["pos1-snrmask_r"] = "on"
        ov["pos1-snrmask_b"] = "on"
        ov["pos1-snrmask_L1"] = mask
        ov["pos1-snrmask_L2"] = mask
        ov["pos1-snrmask_L5"] = mask
        ov["pos1-snrmask_L6"] = mask
    return ov


def expand_grid(elmask_grid: Sequence[float],
                snr_grid: Sequence[str]) -> List[Variant]:
    """Cartesian product of elevation masks x SNR thresholds -> Variant list."""
    out: List[Variant] = []
    for el in elmask_grid:
        for snr in snr_grid:
            snr_txt = str(snr).lower() if str(snr).lower() == "off" else str(snr)
            el_txt = str(int(el)) if float(el).is_integer() else str(el)
            name = f"el{el_txt}_snr{snr_txt}"
            out.append(Variant(name=name, elmask_deg=float(el), snr=snr_txt,
                               overrides=build_overrides(float(el), snr_txt)))
    return out


def minmax_normalize(values: Sequence[float],
                     higher_is_better: bool = False) -> List[float]:
    """Min-max normalize to [0,1], oriented so **0 = best, 1 = worst**.

    * ``higher_is_better=False`` (default): the minimum maps to 0.
    * ``higher_is_better=True``: the maximum maps to 0 (value is flipped).
    * NaN inputs -> 1.0 (worst) so failed variants never win.
    * Degenerate all-equal (or single finite value) -> 0.5 for every finite
      entry (metric carries no information; no div-by-zero).
    """
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return [1.0] * len(values)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out: List[float] = []
    for v in values:
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            out.append(1.0)
        elif span == 0:
            out.append(0.5)
        else:
            n = (v - lo) / span
            out.append(1.0 - n if higher_is_better else n)
    return out


def combine_scores(two_sigma_n: Sequence[float], max_n: Sequence[float],
                   p_resid_p95_n: Sequence[float], fix_pct_n: Sequence[float],
                   w_accuracy: float, w_multipath: float,
                   have_accuracy: bool) -> List[float]:
    """Weighted combined score per variant (LOWER = BETTER).

    All inputs must already be normalized+oriented (0=best) via
    :func:`minmax_normalize`. When ``have_accuracy`` is False the accuracy
    term is dropped and the environment noise term gets full weight.
    """
    n = len(p_resid_p95_n)
    if have_accuracy:
        w_sum = w_accuracy + w_multipath
        wa = w_accuracy / w_sum if w_sum > 0 else 0.5
        wm = w_multipath / w_sum if w_sum > 0 else 0.5
    else:
        wa, wm = 0.0, 1.0
    out: List[float] = []
    for i in range(n):
        mp = (p_resid_p95_n[i] + fix_pct_n[i]) / 2.0
        acc = (two_sigma_n[i] + max_n[i]) / 2.0 if have_accuracy else 0.0
        out.append(wa * acc + wm * mp)
    return out


def rank_variants(metrics: Sequence[Dict[str, float]],
                  w_accuracy: float, w_multipath: float,
                  have_accuracy: bool) -> List[Dict[str, float]]:
    """Normalize + combine + rank. ``metrics`` is one dict per variant with
    keys: two_sigma_m, max_m (lower=better; may be NaN), p_resid_p95
    (lower=better), fix_pct (higher=better). Extra keys pass through.

    Returns NEW dicts (input untouched) with ``combined_score`` and ``rank``
    (1 = best) added, sorted best-first (ascending combined_score).
    """
    two_sigma_n = minmax_normalize([m.get("two_sigma_m", float("nan")) for m in metrics])
    max_n = minmax_normalize([m.get("max_m", float("nan")) for m in metrics])
    p95_n = minmax_normalize([m.get("p_resid_p95", float("nan")) for m in metrics])
    fix_n = minmax_normalize([m.get("fix_pct", float("nan")) for m in metrics],
                             higher_is_better=True)
    combined = combine_scores(two_sigma_n, max_n, p95_n, fix_n,
                              w_accuracy, w_multipath, have_accuracy)
    enriched = [dict(m, combined_score=round(c, 6)) for m, c in zip(metrics, combined)]
    enriched.sort(key=lambda m: m["combined_score"])
    for rank, m in enumerate(enriched, start=1):
        m["rank"] = rank
    return enriched


def run_failure_status(returncode: int, pos_exists: bool, pos_size: int,
                       stderr: str) -> Optional[str]:
    """Classify one solver run. ``None`` means the run produced usable
    output; otherwise a status string -> the variant is FAILED (metrics stay
    NaN so it ranks last and can never win).

    A non-zero return code fails the variant even when a .pos file exists:
    combined with :func:`clear_variant_outputs` this guarantees a stale .pos
    from a previous sweep into the same --out is never parsed or scored.
    """
    if returncode != 0:
        return f"rnx2rtkp_failed rc={returncode} stderr={stderr[:200]!r}"
    if not pos_exists or pos_size == 0:
        return f"no_output rc={returncode} stderr={stderr[:200]!r}"
    return None


def accuracy_all_nan(metrics: Sequence[Dict[str, float]]) -> bool:
    """True when accuracy scoring produced no usable result for ANY variant
    (every ``two_sigma_m`` is missing/NaN). In that case min-max
    normalization maps every accuracy channel to the same value and the
    ranking degrades to environment noise-only even though GT was supplied."""
    if not metrics:
        return False

    def _bad(v: object) -> bool:
        return not (isinstance(v, (int, float)) and math.isfinite(v))

    return all(_bad(m.get("two_sigma_m", float("nan"))) for m in metrics)


def percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an ASCENDING-sorted sequence."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


# ---------------------------------------------------------------------------
# Conf handling (same approach as scripts/ppk_conf_sweep.py apply_overrides)
# ---------------------------------------------------------------------------
def apply_overrides(lines: List[str], overrides: Dict[str, str]) -> List[str]:
    """Return a new conf line list with overrides applied; keys missing from
    the base conf are appended. Preserves untouched lines verbatim."""
    remaining = dict(overrides)
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
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


# ---------------------------------------------------------------------------
# The external solver driver + scoring (impure; needs the solver binary / data_pipeline)
# ---------------------------------------------------------------------------
def resolve_exe(override: Optional[Path]) -> Path:
    """Resolve the solver binary: explicit override -> data_pipeline resolver ->
    vendored fallback exe."""
    if override is not None:
        return Path(override).resolve()
    try:
        from data_pipeline.stages.ppk import resolve_rnx2rtkp
        return Path(resolve_rnx2rtkp()).resolve()
    except Exception:
        pass
    if FALLBACK_RNX2RTKP.is_file():
        return FALLBACK_RNX2RTKP.resolve()
    raise FileNotFoundError(
        "rnx2rtkp.exe not found: pass --rnx2rtkp, or install so "
        "data_pipeline.stages.ppk.resolve_rnx2rtkp finds it, or vendor it "
        f"at {FALLBACK_RNX2RTKP}"
    )


def clear_variant_outputs(pos_path: Path) -> Path:
    """Delete any stale per-variant ``.pos`` / ``.pos.stat`` left by a
    previous run into the same --out, so a failed solve can never be scored
    against the previous run's files. Returns the ``.pos.stat`` path."""
    stat_path = Path(str(pos_path) + ".stat")
    pos_path.unlink(missing_ok=True)
    stat_path.unlink(missing_ok=True)
    return stat_path


def run_rnx2rtkp(exe: Path, conf: Path, out_pos: Path, rover_obs: Path,
                 base_obs: Path, navs: Sequence[Path]) -> subprocess.CompletedProcess:
    """Run one Post-processing solve. All paths passed as ABSOLUTE backslash paths
    (The external solver EX 2.5.0 rejects forward-slash absolute paths on Windows)."""
    out_pos.parent.mkdir(parents=True, exist_ok=True)

    def w(p: Path) -> str:
        return str(Path(p).resolve())  # native separators (backslash on win32)

    args = [w(exe), "-k", w(conf), "-o", w(out_pos), w(rover_obs), w(base_obs),
            *[w(n) for n in navs]]
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=RNX2RTKP_TIMEOUT_S)


def write_score_csv(rows, out_csv: Path) -> None:
    """PosRow list -> the reference time,lat_deg,lon_deg CSV shape that
    traj_score.score_trajectories reads (reference time column carries utc_s)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["gpstime", "lat_deg", "lon_deg"])
        for r in rows:
            wtr.writerow([r.utc_s, r.lat_deg, r.lon_deg])


def fix_pct(rows) -> float:
    """Percent of epochs with Q==1 (ambiguity-fixed)."""
    if not rows:
        return float("nan")
    n_fix = sum(1 for r in rows if r.quality == 1)
    return round(100.0 * n_fix / len(rows), 2)


def multipath_metrics(stat_path: Path) -> Dict[str, float]:
    """GT-free environment noise resistance from the .pos.stat residual file.

    Returns p_resid_med / p_resid_p95 = median / 95th-pct of |coarse measurement
    residual| across all valid source/epoch rows (lower = more resistant),
    and mean_snr = mean of per-epoch mean SNR (dB-Hz, higher = better).
    """
    from data_pipeline.epoch_weight import aggregate_p_resid_per_epoch
    from data_pipeline.stat_to_csv import parse_stat

    nanres = {"p_resid_med": float("nan"), "p_resid_p95": float("nan"),
              "mean_snr": float("nan")}
    try:
        rows = parse_stat(stat_path)
    except (FileNotFoundError, OSError, RuntimeError):
        return nanres
    # valid_flag==1 (used in solution); res_p==0.0 is The external solver's placeholder
    # for untracked frequencies -> no information.
    abs_res = sorted(abs(r.res_p_m) for r in rows
                     if r.valid_flag == 1 and r.res_p_m != 0.0
                     and math.isfinite(r.res_p_m))
    if not abs_res:
        return nanres
    per_epoch = aggregate_p_resid_per_epoch(stat_path)
    snr_means = [e["snr_mean"] for e in per_epoch.values()
                 if math.isfinite(e.get("snr_mean", float("nan")))]
    return {
        "p_resid_med": round(percentile(abs_res, 50.0), 4),
        "p_resid_p95": round(percentile(abs_res, 95.0), 4),
        "mean_snr": round(sum(snr_means) / len(snr_means), 2) if snr_means
                    else float("nan"),
    }


def run_variant(variant: Variant, base_lines: List[str], out_dir: Path,
                exe: Path, rover_obs: Path, base_obs: Path,
                navs: Sequence[Path],
                gt_csv: Optional[Path],
                max_dt_s: float = 0.05) -> Dict[str, float]:
    """Solve + score one grid point. Returns the metrics row (NaN metrics on
    failure so the variant ranks last, never wins)."""
    from data_pipeline.parsers import parse_rtkpos
    from data_pipeline.traj_score import score_trajectories

    vdir = out_dir / variant.name
    conf_path = vdir / f"{variant.name}.conf"
    pos_path = vdir / f"{variant.name}.pos"
    vdir.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(
        "\n".join(apply_overrides(base_lines, variant.overrides)) + "\n",
        encoding="utf-8")

    row: Dict[str, float] = {
        "variant": variant.name, "elmask": variant.elmask_deg,
        "snr": variant.snr,
        "fix_pct": float("nan"), "two_sigma_m": float("nan"),
        "max_m": float("nan"), "median_off_m": float("nan"),
        "p_resid_med": float("nan"),
        "p_resid_p95": float("nan"), "mean_snr": float("nan"),
        "n_epochs": 0, "status": "ok",
        "conf_path": str(conf_path), "pos_path": str(pos_path),
    }
    # Delete stale outputs from any previous run into the same --out BEFORE
    # solving, so a failed solver run can never be scored on the old files.
    stat_path = clear_variant_outputs(pos_path)
    try:
        proc = run_rnx2rtkp(exe, conf_path, pos_path, rover_obs, base_obs, navs)
    except subprocess.TimeoutExpired:
        row["status"] = f"timeout>{RNX2RTKP_TIMEOUT_S}s"
        return row
    fail = run_failure_status(
        proc.returncode, pos_path.exists(),
        pos_path.stat().st_size if pos_path.exists() else 0,
        proc.stderr or "")
    if fail is not None:
        row["status"] = fail
        return row
    pos_rows = parse_rtkpos(pos_path)
    if not pos_rows:
        row["status"] = "0_rows_parsed"
        return row

    row["n_epochs"] = len(pos_rows)
    row["fix_pct"] = fix_pct(pos_rows)
    row.update(multipath_metrics(stat_path))

    if gt_csv is not None:
        test_csv = vdir / f"{variant.name}_traj.csv"
        write_score_csv(pos_rows, test_csv)
        s = score_trajectories(gt_csv, test_csv, max_dt_s=max_dt_s)
        row["two_sigma_m"] = s["two_sigma_m"]
        row["max_m"] = s["max_m"]
        # Accuracy metrics are bias-removed (median offset subtracted before
        # 2sigma/max). Surface the removed constant offset so a variant with
        # tight scatter but a big datum/lever-arm bias is visible.
        row["median_off_m"] = s.get("median_offset_m", float("nan"))
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep elevation-mask x SNR-mask PPK configs for one "
                    "rover .obs; rank by accuracy (vs GT) + multipath "
                    "resistance (GT-free); emit best.conf/best.pos.")
    ap.add_argument("--rover-obs", type=Path, required=True)
    ap.add_argument("--base-obs", type=Path, required=True)
    ap.add_argument("--nav", type=Path, nargs="+", required=True,
                    help="One or more nav/ephemeris files")
    ap.add_argument("--gt-pos", type=Path, default=None,
                    help="Optional ground-truth .pos; enables accuracy term")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir (per-variant subdirs + results)")
    ap.add_argument("--elmask-grid", type=str, default="5,10,15,20",
                    help="CSV of elevation masks in deg (default 5,10,15,20)")
    ap.add_argument("--snr-grid", type=str, default="off,30,33,35,38",
                    help="CSV of SNR thresholds in dB-Hz; 'off' disables the "
                         "mask (default off,30,33,35,38)")
    ap.add_argument("--base-conf", type=Path, default=DEFAULT_BASE_CONF,
                    help=f"Base conf to patch (default {DEFAULT_BASE_CONF})")
    ap.add_argument("--rnx2rtkp", type=Path, default=None,
                    help="Explicit rnx2rtkp.exe path (else auto-resolve)")
    ap.add_argument("--w-accuracy", type=float, default=0.5)
    ap.add_argument("--w-multipath", type=float, default=0.5)
    ap.add_argument("--max-dt-s", type=float, default=0.05,
                    help="Max |GT - rover| epoch time difference in seconds "
                         "for accuracy matching (default 0.05)")
    return ap.parse_args(argv)


RESULT_COLS = ["rank", "variant", "elmask", "snr", "fix_pct", "two_sigma_m",
               "max_m", "median_off_m", "p_resid_med", "p_resid_p95",
               "mean_snr", "combined_score", "n_epochs", "status"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    from data_pipeline.parsers import parse_rtkpos

    elmask_grid = [float(x) for x in args.elmask_grid.split(",") if x.strip()]
    snr_grid = [x.strip() for x in args.snr_grid.split(",") if x.strip()]
    variants = expand_grid(elmask_grid, snr_grid)

    exe = resolve_exe(args.rnx2rtkp)
    base_lines = args.base_conf.read_text(encoding="utf-8").splitlines()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    have_gt = args.gt_pos is not None
    gt_csv: Optional[Path] = None
    if have_gt:
        gt_rows = parse_rtkpos(args.gt_pos)
        if not gt_rows:
            print(f"WARNING: no rows parsed from GT {args.gt_pos}; "
                  "accuracy term disabled.")
            have_gt = False
        else:
            gt_csv = out_dir / "gt_ref.csv"
            write_score_csv(gt_rows, gt_csv)

    print(f"rnx2rtkp : {exe}")
    print(f"base conf: {args.base_conf}")
    print(f"grid     : {len(variants)} variants "
          f"(elmask={elmask_grid} x snr={snr_grid})")
    print(f"accuracy : {'ON (GT=' + str(args.gt_pos) + ')' if have_gt else 'OFF (multipath+fix only)'}\n")

    metrics: List[Dict[str, float]] = []
    for i, v in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {v.name} ...", end=" ", flush=True)
        row = run_variant(v, base_lines, out_dir, exe, args.rover_obs,
                          args.base_obs, args.nav, gt_csv,
                          max_dt_s=args.max_dt_s)
        print(row["status"] if row["status"] != "ok" else
              f"fix%={_fmt(row['fix_pct'])} p95={_fmt(row['p_resid_p95'])}"
              + (f" 2sig={_fmt(row['two_sigma_m'])}" if have_gt else ""))
        metrics.append(row)

    if have_gt and accuracy_all_nan(metrics):
        print("\nWARNING: accuracy scoring produced NO time-matched epochs "
              "for any variant (every two_sigma_m is NaN). Check GT/rover "
              "time alignment (GPST vs UTC?) or raise --max-dt-s "
              f"(currently {args.max_dt_s} s). The ranking below is "
              "effectively multipath-only even though accuracy is ON.")

    ranked = rank_variants(metrics, args.w_accuracy, args.w_multipath, have_gt)

    # --- ranked table (combined_score: LOWER = BETTER) ---
    hdr = (f"{'rk':>3} {'variant':16} {'el':>4} {'snr':>4} {'fix%':>7} "
           f"{'2sigma_m':>9} {'max_m':>9} {'med_off':>8} {'presid_med':>10} "
           f"{'presid_p95':>10} {'meanSNR':>8} {'combined':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in ranked:
        print(f"{m['rank']:>3} {m['variant']:16} {_fmt(m['elmask']):>4} "
              f"{m['snr']:>4} {_fmt(m['fix_pct']):>7} "
              f"{_fmt(m['two_sigma_m']):>9} {_fmt(m['max_m']):>9} "
              f"{_fmt(m.get('median_off_m', float('nan'))):>8} "
              f"{_fmt(m['p_resid_med']):>10} {_fmt(m['p_resid_p95']):>10} "
              f"{_fmt(m['mean_snr']):>8} {m['combined_score']:>9.4f}")

    results_csv = out_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=RESULT_COLS, extrasaction="ignore")
        wtr.writeheader()
        for m in ranked:
            wtr.writerow(m)
    print(f"\nresults -> {results_csv}")

    winner = next((m for m in ranked if m["status"] == "ok"), None)
    if winner is None:
        print("BLOCKED: every variant failed; no winner.")
        return 1
    best_conf = out_dir / "best.conf"
    best_pos = out_dir / "best.pos"
    shutil.copyfile(winner["conf_path"], best_conf)
    shutil.copyfile(winner["pos_path"], best_pos)
    print(f"\nWINNER: {winner['variant']} "
          f"(elmask={_fmt(winner['elmask'])} deg, snr={winner['snr']}) "
          f"combined={winner['combined_score']:.4f} "
          f"fix%={_fmt(winner['fix_pct'])} "
          f"p_resid_p95={_fmt(winner['p_resid_p95'])}"
          + (f" 2sigma={_fmt(winner['two_sigma_m'])} m" if have_gt else ""))
    if have_gt:
        w_off = winner.get("median_off_m", float("nan"))
        if isinstance(w_off, (int, float)) and math.isfinite(w_off) \
                and abs(w_off) > 1.0:
            print(f"INFO: winner has a LARGE constant offset from GT "
                  f"(median_off={w_off:.2f} m). Accuracy metrics are "
                  "bias-removed (scatter about the median offset), so this "
                  "bias did not hurt its ranking -- verify datum / "
                  "lever-arm / time-sync before trusting absolute "
                  "coordinates.")
    print(f"  -> {best_conf}")
    print(f"  -> {best_pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
