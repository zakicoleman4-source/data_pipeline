"""Solve dodge app.pos with the innovation gate off/on; score vs the logger
(cross-device consensus) path. Prints a before/after table. No GT."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_pipeline.parsers import parse_rtkpos
from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2, EpochWeightV2Options
from data_pipeline.smoothers import _pos_to_enu_arrays, _enu_arrays_to_pos_rows
from data_pipeline.stages.user_export import export_trajectory
from data_pipeline.traj_score import score_trajectories

PAIR = Path(r"C:/Aj/gps/day14/ppk_ab_2026-06-28/pair1_190336")
APP_POS = PAIR / "app.pos"
LOGGER_POS = PAIR / "logger.pos"


def solve(rows, opts, out_csv):
    v2 = smooth_epoch_weighted_v2(rows, imu_rows=None, options=opts)
    _E, _N, _U, ts, ref = _pos_to_enu_arrays(rows)
    out_rows = _enu_arrays_to_pos_rows(v2.E_smooth, v2.N_smooth, v2.U_smooth, ts, ref, rows)
    export_trajectory(out_rows, out_csv, source_tag="ab", robust_filter_enabled=True)
    return out_csv


def main():
    tmp = REPO / "_ab_out"; tmp.mkdir(exist_ok=True)
    app = parse_rtkpos(APP_POS)
    logger = parse_rtkpos(LOGGER_POS)
    ref_csv = tmp / "logger_ref.csv"
    export_trajectory(logger, ref_csv, source_tag="ref", robust_filter_enabled=False)
    variants = {
        "baseline": EpochWeightV2Options(zupt_enabled=True, nhc_enabled=True),
        "innov25":  EpochWeightV2Options(zupt_enabled=True, nhc_enabled=True,
                     innov_gate_enabled=True, innov_gate_thresh=2.5, innov_gate_r_mult=10.0),
        "innov30":  EpochWeightV2Options(zupt_enabled=True, nhc_enabled=True,
                     innov_gate_enabled=True, innov_gate_thresh=3.0, innov_gate_r_mult=10.0),
    }
    print(f"{'variant':10} {'2sigma_m':>9} {'max_m':>7} {'<=1m%':>7} {'n':>6}")
    for name, opts in variants.items():
        s = score_trajectories(ref_csv, solve(app, opts, tmp / f"{name}.csv"))
        print(f"{name:10} {s['two_sigma_m']:>9} {s['max_m']:>7} {s['le1m_pct']:>7} {s['n']:>6}")


if __name__ == "__main__":
    raise SystemExit(main())
