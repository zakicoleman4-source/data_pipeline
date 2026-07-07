"""For a real session: of the epochs whose cross-device horizontal error > 1 m,
how many does the reported 2-sigma bar (err_horiz_2sigma_m) actually flag?
A high catch-rate means suppression removes the right epochs. No GT used."""
import csv, sys, math
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_pipeline.parsers import parse_rtkpos
from data_pipeline.stages.user_export import export_trajectory

APP_CSV = Path(r"C:/Aj/gps/day14/solved_2026-06-28/dodge/20260628_190336_677/trajectory_user.csv")
LOGGER_POS = Path(r"C:/Aj/gps/day14/ppk_ab_2026-06-28/pair1_190336/logger.pos")
BAR_M = 6.0   # user_export HORIZ_BAR_2SIGMA_M


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[round(float(r["gpstime"]), 3)] = r
            except (KeyError, ValueError):
                continue
    return out


def main():
    tmp = REPO / "_sigma_out"; tmp.mkdir(exist_ok=True)
    ref_csv = tmp / "logger_ref.csv"
    export_trajectory(parse_rtkpos(LOGGER_POS), ref_csv, source_tag="ref",
                      robust_filter_enabled=False)
    app = load(APP_CSV); ref = load(ref_csv)
    mlat = 111320.0
    tp = fp = fn = 0
    matched = 0
    for t, ar in app.items():
        rr = ref.get(t)
        if rr is None:
            continue
        matched += 1
        lat0 = float(ar["lat_deg"]); mlon = mlat * math.cos(math.radians(lat0))
        dn = (float(ar["lat_deg"]) - float(rr["lat_deg"])) * mlat
        de = (float(ar["lon_deg"]) - float(rr["lon_deg"])) * mlon
        err = math.hypot(de, dn)
        try:
            flagged = float(ar.get("err_horiz_2sigma_m") or "nan") > BAR_M
        except ValueError:
            flagged = False
        bad = err > 1.0
        tp += bad and flagged
        fn += bad and not flagged
        fp += (not bad) and flagged
    caught = tp / (tp + fn) * 100 if (tp + fn) else float("nan")
    print(f"matched epochs: {matched}")
    print(f"bad>1m epochs caught by 2-sigma bar: {caught:.1f}%  (tp={tp} fn={fn} fp={fp})")
    print("catch-rate >=80% => reported 2-sigma already flags MAX-spike epochs; "
          "<80% => sigma under-reports on spikes (raise accuracy_predictor inflation).")


if __name__ == "__main__":
    raise SystemExit(main())
