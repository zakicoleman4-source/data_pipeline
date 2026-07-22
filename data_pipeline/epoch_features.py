"""Per-epoch feature CSV builder.

Combines The external solver `.pos` (positions + sigmas + Q + ns + Rate-signal vel) with the
companion `.stat` file (per-source az/el for DOP computation) into ONE flat
CSV that downstream filters / FGO / Recursive-filter / Export format viewers can consume
without re-parsing both files.

Per-epoch columns:
    utc_s
    gps_week, gps_tow_s
    lat_deg, lon_deg, h_m
    q                       1=Fix, 2=Float, 4=Differential, 5=Single
    ns_used                 sources used in solution (The external solver col 7)
    n_sat_visible           total visible at this epoch (from .stat)
    n_sat_used              valid-flagged in .stat ($Source vsat=1)
    sd_n_m, sd_e_m, sd_u_m  per-epoch 1-sigma uncertainties (.pos cols 7-9)
    vn_mps, ve_mps, vu_mps  Rate-signal velocity (.pos cols 15-17)
    speed_mps               sqrt(vn² + ve²)
    GDOP, PDOP, HDOP, VDOP, TDOP   from $Source geometry (cos/sin az·el)
    n_GPS, n_GLO, n_GAL, n_BDS, n_QZS  per-source group valid count
    mean_el_deg, min_el_deg, max_el_deg
    mean_snr_db_hz
    fix_flag_pct            % of visible sources with $Source fix flag set
    pseudorange_resid_rms_m
    cphase_resid_rms_m

When a GT .pos is supplied via `eval_against_gt`, an extra column
`h_err_vs_gt_m` (horizontal error vs interpolated GT) is appended.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .parsers import parse_rtkpos
from .geo import llh_to_ecef, ecef_to_enu
from .stat_to_csv import parse_stat, StatRow


_PRN_SYSTEM = {"G": "GPS", "R": "GLO", "E": "GAL", "C": "BDS", "J": "QZS", "I": "IRN", "S": "SBS"}


@dataclass
class EpochFeatures:
    utc_s:      float
    gps_week:   int
    gps_tow_s:  float
    lat_deg:    float
    lon_deg:    float
    h_m:        float
    q:          int
    ns_used:    int
    n_sat_visible: int
    n_sat_used:    int
    sd_n_m:     float
    sd_e_m:     float
    sd_u_m:     float
    vn_mps:     float
    ve_mps:     float
    vu_mps:     float
    speed_mps:  float
    GDOP:       float
    PDOP:       float
    HDOP:       float
    VDOP:       float
    TDOP:       float
    n_GPS:      int
    n_GLO:      int
    n_GAL:      int
    n_BDS:      int
    n_QZS:      int
    mean_el_deg:    float
    min_el_deg:     float
    max_el_deg:     float
    mean_snr_db_hz: float
    fix_flag_pct:   float
    p_resid_rms_m:  float
    c_resid_rms_m:  float
    h_err_vs_gt_m:  float = float("nan")


def _compute_dops(rows: list[StatRow]) -> tuple[float, float, float, float, float]:
    """Build geometry matrix H from valid sources, return (GDOP, PDOP, HDOP, VDOP, TDOP).

    H rows: [-cos(el)·sin(az), -cos(el)·cos(az), -sin(el), 1]   (Local-frame + clock)
    Q = (H^T H)^-1 ; DOPs from diag(Q).
    """
    used = [r for r in rows if r.valid_flag == 1 and math.isfinite(r.el_deg)
            and math.isfinite(r.az_deg)]
    if len(used) < 4:
        return (float("nan"),) * 5
    H = np.empty((len(used), 4), dtype=np.float64)
    for i, r in enumerate(used):
        el = math.radians(r.el_deg)
        az = math.radians(r.az_deg)
        ce, se = math.cos(el), math.sin(el)
        sa, ca = math.sin(az), math.cos(az)
        H[i, 0] = -ce * sa
        H[i, 1] = -ce * ca
        H[i, 2] = -se
        H[i, 3] =  1.0
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return (float("nan"),) * 5
    qxx, qyy, qzz, qtt = Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3]
    GDOP = math.sqrt(max(0.0, qxx + qyy + qzz + qtt))
    PDOP = math.sqrt(max(0.0, qxx + qyy + qzz))
    HDOP = math.sqrt(max(0.0, qxx + qyy))
    VDOP = math.sqrt(max(0.0, qzz))
    TDOP = math.sqrt(max(0.0, qtt))
    return GDOP, PDOP, HDOP, VDOP, TDOP


def build_epoch_features(
    pos_path: Path,
    stat_path: Path,
    gt_pos: Optional[Path] = None,
) -> list[EpochFeatures]:
    """Build per-epoch feature list from a (.pos, .stat) pair.

    If ``gt_pos`` is provided, every epoch gets ``h_err_vs_gt_m`` populated
    via linear interpolation of the GT path to the epoch UTC.

    Raises :class:`FileNotFoundError` with actionable hint when ``pos_path``
    or ``stat_path`` is missing.
    """
    pos_path = Path(pos_path)
    stat_path = Path(stat_path)
    if not pos_path.is_file():
        raise FileNotFoundError(
            f".pos file not found: {pos_path}. "
            "Run the PPK stage first (data_pipeline.stages.ppk.run)."
        )
    if not stat_path.is_file():
        raise FileNotFoundError(
            f".stat file not found: {stat_path}. "
            "Re-run rnx2rtkp with -x 1 (or higher) to emit a .stat companion."
        )
    pos_rows = parse_rtkpos(pos_path)
    stat_rows = parse_stat(stat_path)

    # Group stat rows by UTC second so the inner loop is O(1) per pos row
    # instead of O(N stat-buckets). Skip rows with non-finite utc_s — the
    # `int(round(NaN))` cast would raise ValueError mid-build.
    by_utc: dict[int, list[StatRow]] = defaultdict(list)
    for s in stat_rows:
        if not math.isfinite(s.utc_s):
            continue
        by_utc[int(round(s.utc_s))].append(s)

    # GT eval setup.
    gt_t: Optional[np.ndarray] = None
    gt_e: Optional[np.ndarray] = None
    gt_n: Optional[np.ndarray] = None
    ref:  Optional[tuple[float, float, float]] = None
    if gt_pos is not None and gt_pos.is_file():
        gt = parse_rtkpos(gt_pos)
        if gt:
            ref = (gt[0].lat_deg, gt[0].lon_deg, gt[0].h_m)
            gt_t = np.array([r.utc_s for r in gt])
            gt_e = np.empty(len(gt))
            gt_n = np.empty(len(gt))
            for i, r in enumerate(gt):
                x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
                e, n, _ = ecef_to_enu(x, y, z, ref)
                gt_e[i] = e
                gt_n[i] = n

    out: list[EpochFeatures] = []
    for pr in pos_rows:
        if not math.isfinite(pr.utc_s):
            continue                  # corrupt epoch — skip rather than crash
        # O(1) lookup: stat rows binned by round(utc). Check the same and
        # +/-1 second buckets for boundary cases.
        bucket = int(round(pr.utc_s))
        sats: list[StatRow] = (
            by_utc.get(bucket, []) + by_utc.get(bucket - 1, []) + by_utc.get(bucket + 1, [])
        )
        # Within the +/-1 s window, keep only sources within 0.5 s of pos epoch.
        sats = [s for s in sats if abs(s.utc_s - pr.utc_s) < 0.5]

        used_sats = [s for s in sats if s.valid_flag == 1]

        # Per-Source views (one row per PRN). The external solver-EX emits one $Source line
        # *per frequency* (L1/L2/L5/...) per source, so a dual-frequency source
        # appears 2-4 times with identical az/el. Source COUNTS and the DOP
        # geometry matrix must be per-source: counting every freq row
        # over-counts sources, and feeding duplicate az/el rows into H makes
        # H.T@H rank-deficient and yields nonsense DOPs (observed HDOP=0.0 on
        # real dual-freq data). Residual / SNR aggregates below intentionally
        # stay per-observation (all freq rows) — only counts/DOP are deduped.
        def _dedup_by_prn(rows: list[StatRow]) -> list[StatRow]:
            best: dict[str, StatRow] = {}
            for s in rows:
                cur = best.get(s.prn)
                # Prefer the lower freq index for a determinate representative.
                if cur is None or s.freq < cur.freq:
                    best[s.prn] = s
            return list(best.values())

        vis_by_prn = _dedup_by_prn(sats)
        used_by_prn = _dedup_by_prn(used_sats)
        n_vis = len(vis_by_prn)
        n_used = len(used_by_prn)

        # Source group tallies (per-source, deduped by PRN).
        per_sys = {"GPS": 0, "GLO": 0, "GAL": 0, "BDS": 0, "QZS": 0}
        for s in used_by_prn:
            sys_tag = _PRN_SYSTEM.get(s.prn[:1], "")
            if sys_tag in per_sys:
                per_sys[sys_tag] += 1

        # Aggregates. Elevation and the fix-flag count are per-source
        # (deduped) so mean_el / fix_flag_pct are not skewed by the number of
        # frequencies tracked. SNR and residuals stay per-observation.
        els = [s.el_deg for s in used_by_prn if math.isfinite(s.el_deg)]
        snrs = [s.snr_db_hz for s in used_sats
                if math.isfinite(s.snr_db_hz) and s.snr_db_hz > 0]
        fix_flagged = sum(1 for s in used_by_prn if s.fix_flag == 1)
        p_resids = [s.res_p_m for s in used_sats
                    if math.isfinite(s.res_p_m) and s.res_p_m != 0.0]
        c_resids = [s.res_c_m for s in used_sats
                    if math.isfinite(s.res_c_m) and s.res_c_m != 0.0]

        GDOP, PDOP, HDOP, VDOP, TDOP = _compute_dops(used_by_prn)

        # GT residual.
        h_err = float("nan")
        if gt_t is not None and ref is not None:
            if gt_t[0] <= pr.utc_s <= gt_t[-1]:
                j = int(np.searchsorted(gt_t, pr.utc_s))
                if j == 0:
                    ge, gn = float(gt_e[0]), float(gt_n[0])
                elif j >= len(gt_t):
                    ge, gn = float(gt_e[-1]), float(gt_n[-1])
                else:
                    t0, t1 = float(gt_t[j-1]), float(gt_t[j])
                    u = (pr.utc_s - t0) / (t1 - t0) if t1 > t0 else 0.0
                    ge = float(gt_e[j-1]) + u * float(gt_e[j] - gt_e[j-1])
                    gn = float(gt_n[j-1]) + u * float(gt_n[j] - gt_n[j-1])
                x, y, z = llh_to_ecef(pr.lat_deg, pr.lon_deg, pr.h_m)
                e, n, _ = ecef_to_enu(x, y, z, ref)
                h_err = math.hypot(e - ge, n - gn)

        # Reference week/tow from any matched source row (all sources in the bucket
        # share the same epoch).
        if sats:
            week, tow = sats[0].gps_week, sats[0].gps_tow_s
        else:
            week, tow = 0, 0.0

        speed = (math.hypot(pr.vn, pr.ve)
                 if math.isfinite(pr.vn) and math.isfinite(pr.ve) else float("nan"))

        out.append(EpochFeatures(
            utc_s=pr.utc_s, gps_week=week, gps_tow_s=tow,
            lat_deg=pr.lat_deg, lon_deg=pr.lon_deg, h_m=pr.h_m,
            q=pr.quality, ns_used=pr.ns,
            n_sat_visible=n_vis, n_sat_used=n_used,
            sd_n_m=pr.sd_n, sd_e_m=pr.sd_e, sd_u_m=pr.sd_u,
            vn_mps=pr.vn, ve_mps=pr.ve, vu_mps=pr.vu, speed_mps=speed,
            GDOP=GDOP, PDOP=PDOP, HDOP=HDOP, VDOP=VDOP, TDOP=TDOP,
            n_GPS=per_sys["GPS"], n_GLO=per_sys["GLO"],
            n_GAL=per_sys["GAL"], n_BDS=per_sys["BDS"], n_QZS=per_sys["QZS"],
            mean_el_deg=float(np.mean(els)) if els else float("nan"),
            min_el_deg=float(np.min(els)) if els else float("nan"),
            max_el_deg=float(np.max(els)) if els else float("nan"),
            mean_snr_db_hz=float(np.mean(snrs)) if snrs else float("nan"),
            fix_flag_pct=100.0 * fix_flagged / n_used if n_used else float("nan"),
            p_resid_rms_m=float(np.sqrt(np.mean(np.square(p_resids)))) if p_resids else float("nan"),
            c_resid_rms_m=float(np.sqrt(np.mean(np.square(c_resids)))) if c_resids else float("nan"),
            h_err_vs_gt_m=h_err,
        ))
    return out


def write_features_csv(features: list[EpochFeatures], out_path: Path) -> int:
    if not features:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(EpochFeatures.__dataclass_fields__.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for ef in features:
            row = []
            for k in fields:
                v = getattr(ef, k)
                if isinstance(v, float):
                    row.append(f"{v:.6f}" if math.isfinite(v) else "")
                else:
                    row.append(v)
            w.writerow(row)
    return len(features)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pos",  type=Path, help="Input .pos file")
    ap.add_argument("stat", type=Path, help="Companion .stat file")
    ap.add_argument("csv",  type=Path, help="Output features CSV")
    ap.add_argument("--gt", type=Path, default=None,
                    help="Optional GT .pos for h_err_vs_gt_m column")
    args = ap.parse_args()
    feats = build_epoch_features(args.pos, args.stat, args.gt)
    n = write_features_csv(feats, args.csv)
    print(f"wrote {n} epochs -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
