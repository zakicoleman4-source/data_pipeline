"""Parse The external solver-EX .stat (`out-outstat=residual`) file into a flat raw-observation CSV.

Format of input ``.stat`` $Source lines (The external solver-EX 2.5):

    $Source,week,tow,prn,freq,az,el,res_p,res_c,vsat,snr*4,fix,slip,lock,outc,...

Where:
    * week, tow:     Reference week + time of week (s) -> UTC seconds
    * prn:           e.g. G08 R21 E26 C12
    * freq:          1=L1, 2=L2, 3=L5, 4=L6  (The external solver internal)
    * az, el:        source azimuth / elevation (deg)
    * res_p:         coarse measurement residual (m)        ← FGO input
    * res_c:         fine measurements residual (m)      ← FGO input
                      (The external solver-EX writes metres, not cycles; convert to cyc
                       in caller with wavelength if needed)
    * vsat:          valid-source flag (1/0)
    * snr*4:         SNR multiplied by 4 (dB-Hz integer encoding)
    * fix:           ambiguity-fix flag at this epoch

Output CSV columns:
    utc_s, prn, freq, pseudorange_residual_m, cphase_residual_m, snr_db_hz,
    az_deg, el_deg, fix_flag, valid_flag, gps_week, gps_tow_s

UTC conversion uses the Reference-UTC epoch offset table from time_sync. Pre-2017
sessions get 17 s; modern (post-2017) sessions get 18 s.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from .time_sync import get_leap_seconds_for_epoch

# Reference epoch: 1980-01-06 00:00:00 UTC.
_GPS_EPOCH_UNIX = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc).timestamp()


@dataclass(frozen=True)
class StatRow:
    """One $Source entry decoded into typed fields."""

    utc_s:              float
    gps_week:           int
    gps_tow_s:          float
    prn:                str
    freq:               int
    az_deg:             float
    el_deg:             float
    res_p_m:            float
    res_c_m:            float
    snr_db_hz:          float
    fix_flag:           int     # 1 if ambiguity fixed at this epoch
    valid_flag:         int     # 1 if used in solution


def _gps_to_utc(week: int, tow_s: float) -> float:
    """Reference week + TOW -> UTC POSIX seconds."""
    gpst_unix = _GPS_EPOCH_UNIX + week * 7 * 86400 + tow_s
    ls = get_leap_seconds_for_epoch(gpst_unix)
    return gpst_unix - ls


def parse_stat(path: Path) -> list[StatRow]:
    """Read every $Source line. Skips $POS, $VELACC, $CLK, etc.

    Raises :class:`FileNotFoundError` when the path is missing. Returns an
    empty list if the file exists but contains no $Source lines (e.g. The external solver
    was run without ``-x`` and only emitted $POS).

    Malformed $Source lines are skipped silently up to a count; if every $Source
    line fails to parse, raises :class:`RuntimeError` (file is corrupt).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RTKLIB .stat file not found: {path}. "
            "Re-run rnx2rtkp with -x 1 (or higher) to produce a .stat file."
        )
    rows: list[StatRow] = []
    n_sat_lines = 0
    n_skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.startswith("$SAT,"):
                continue
            n_sat_lines += 1
            parts = raw.strip().split(",")
            if len(parts) < 12:
                n_skipped += 1
                continue
            try:
                week = int(parts[1])
                tow  = float(parts[2])
                prn  = parts[3]
                freq = int(parts[4])
                az   = float(parts[5])
                el   = float(parts[6])
                resp = float(parts[7])
                resc = float(parts[8])
                vsat = int(parts[9])
                snr4 = float(parts[10])   # The external solver writes SNR*4
                fix  = int(parts[11])
            except (ValueError, IndexError):
                n_skipped += 1
                continue
            utc = _gps_to_utc(week, tow)
            rows.append(StatRow(
                utc_s=utc, gps_week=week, gps_tow_s=tow,
                prn=prn, freq=freq,
                az_deg=az, el_deg=el,
                res_p_m=resp, res_c_m=resc,
                snr_db_hz=snr4 / 4.0,
                fix_flag=fix, valid_flag=vsat,
            ))
    if n_sat_lines > 0 and not rows:
        raise RuntimeError(
            f"{path}: found {n_sat_lines} $SAT lines but every one failed "
            "to parse. File appears corrupt. Re-run rnx2rtkp."
        )
    if n_skipped:
        import warnings
        warnings.warn(
            f"{path.name}: skipped {n_skipped} of {n_sat_lines} malformed "
            "$SAT lines.",
            RuntimeWarning,
            stacklevel=2,
        )
    return rows


def stat_to_csv(
    stat_path: Path,
    csv_path: Path,
    *,
    min_el_deg: float = 0.0,
    valid_only: bool = False,
    nonzero_resp_only: bool = False,
) -> int:
    """Convert a .stat file to a flat CSV. Returns number of rows written.

    Filters:
        * min_el_deg:        drop sources below this elevation
        * valid_only:        keep only rows where vsat=1 (used in solution)
        * nonzero_resp_only: drop rows whose coarse measurement residual is exactly
                             zero (The external solver emits placeholder zeros for
                             observations on frequencies not tracked at this
                             epoch — those carry no information)
    """
    stat_path = Path(stat_path)
    csv_path = Path(csv_path)
    rows = parse_stat(stat_path)
    if valid_only:
        rows = [r for r in rows if r.valid_flag == 1]
    if nonzero_resp_only:
        rows = [r for r in rows if r.res_p_m != 0.0]
    rows = [r for r in rows if r.el_deg >= min_el_deg]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "utc_s", "prn", "freq",
            "pseudorange_residual_m", "cphase_residual_m",
            "snr_db_hz", "az_deg", "el_deg",
            "fix_flag", "valid_flag", "gps_week", "gps_tow_s",
        ])
        for r in rows:
            w.writerow([
                f"{r.utc_s:.3f}", r.prn, r.freq,
                f"{r.res_p_m:.4f}", f"{r.res_c_m:.4f}",
                f"{r.snr_db_hz:.2f}", f"{r.az_deg:.2f}", f"{r.el_deg:.2f}",
                r.fix_flag, r.valid_flag,
                r.gps_week, f"{r.gps_tow_s:.3f}",
            ])
    return len(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stat", type=Path, help="Input .stat file")
    ap.add_argument("csv",  type=Path, help="Output .csv file")
    ap.add_argument("--min-el",   type=float, default=0.0)
    ap.add_argument("--valid-only", action="store_true")
    ap.add_argument("--nonzero",  action="store_true",
                    help="Drop rows whose pseudorange residual is 0 (untracked freq)")
    args = ap.parse_args()
    n = stat_to_csv(
        args.stat, args.csv,
        min_el_deg=args.min_el,
        valid_only=args.valid_only,
        nonzero_resp_only=args.nonzero,
    )
    print(f"wrote {n} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
