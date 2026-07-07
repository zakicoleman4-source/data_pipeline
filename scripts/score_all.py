"""Score a smoother option set across all 3 cross-device pairs. Single scoring
entry so every accuracy experiment reports the same 3-pair table."""
import sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_pipeline.parsers import parse_rtkpos
from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2, EpochWeightV2Options
from data_pipeline.smoothers import _pos_to_enu_arrays, _enu_arrays_to_pos_rows
from data_pipeline.stages.user_export import export_trajectory
from data_pipeline.traj_score import score_trajectories

_AB = Path(r"C:/Aj/gps/day14/ppk_ab_2026-06-28")
PAIRS = [(n, _AB / n / "app.pos", _AB / n / "logger.pos")
         for n in ["pair1_190336", "pair2_202751", "pair3_205044"]]


def solve_and_score(app_pos, ref_pos, opts, tmp):
    app = parse_rtkpos(app_pos)
    v2 = smooth_epoch_weighted_v2(app, imu_rows=None, options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(app)
    rows = _enu_arrays_to_pos_rows(v2.E_smooth, v2.N_smooth, v2.U_smooth, ts, ref, app)
    out = tmp / "cand.csv"; ref_csv = tmp / "ref.csv"
    export_trajectory(rows, out, source_tag="ab", robust_filter_enabled=True)
    export_trajectory(parse_rtkpos(ref_pos), ref_csv, source_tag="ref", robust_filter_enabled=False)
    return score_trajectories(ref_csv, out)


def mean_metric(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else float("nan")


def score_options(opts, label="opts"):
    rows = []
    print(f"== {label} ==")
    print(f"{'pair':16}{'2sigma':>8}{'MAX':>8}{'<=1m%':>7}")
    for name, app, ref in PAIRS:
        with tempfile.TemporaryDirectory() as td:
            s = solve_and_score(app, ref, opts, Path(td))
        rows.append(s)
        print(f"{name:16}{s['two_sigma_m']:>8}{s['max_m']:>8}{s['le1m_pct']:>7}")
    print(f"{'MEAN':16}{mean_metric(rows,'two_sigma_m'):>8}{mean_metric(rows,'max_m'):>8}")
    return rows


if __name__ == "__main__":
    score_options(EpochWeightV2Options(zupt_enabled=True, nhc_enabled=True), "baseline")
