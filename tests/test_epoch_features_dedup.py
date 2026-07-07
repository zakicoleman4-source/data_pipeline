"""Regression test for per-source de-duplication in build_epoch_features.

The external solver-EX emits one ``$Source`` line per *frequency* (L1/L2/L5/...) for every
source, so a dual-frequency source appears 2-4 times with identical az/el.
Before the fix, ``n_sat_used`` / ``n_sat_visible`` counted every frequency
row, the source group tallies over-counted, and the DOP geometry matrix got
duplicate rows (rank-deficient H -> nonsense DOP, observed HDOP=0.0).

The fix de-duplicates by PRN for counts and DOP while keeping the residual /
SNR aggregates per-observation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest


def _write_pos(path: Path, week: int, tow: float) -> None:
    # Minimal valid The external solver .pos: one epoch. Date/time from week/tow.
    import datetime as dt

    gps_epoch = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
    t = gps_epoch + dt.timedelta(weeks=week, seconds=tow)
    date_s = t.strftime("%Y/%m/%d")
    time_s = t.strftime("%H:%M:%S.%f")[:-3]
    path.write_text(
        f"% header\n"
        f"{date_s} {time_s}  35.000000000 139.000000000   10.0000 "
        f"5  6  0.01 0.01 0.02 0.0 0.0 0.0  0.0  1.0\n",
        encoding="utf-8",
    )


def _write_stat_multifreq(path: Path, week: int, tow: float) -> None:
    """4 unique Reference sources, each tracked on 3 frequencies, all vsat=1.

    Naive per-row counting would report 12 used sources and feed 12 duplicate-
    geometry rows into the DOP matrix.
    """
    sats = [
        ("G01", 30.0, 313.9),
        ("G02", 60.0, 331.1),
        ("G08", 51.0, 227.8),
        ("G27", 35.0, 187.1),
    ]
    lines = []
    for prn, el, az in sats:
        for freq in (1, 2, 5):
            # week,tow,prn,freq,az,el,res_p,res_c,vsat,snr,fix,...
            lines.append(
                f"$SAT,{week},{tow:.3f},{prn},{freq},{az:.1f},{el:.1f},"
                f"1.2345,0.0010,1,28,0,0,0,0"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_multifreq_sats_counted_once(tmp_path: Path) -> None:
    from data_pipeline.epoch_features import build_epoch_features

    week, tow = 2401, 335849.0
    pos = tmp_path / "x.pos"
    stat = tmp_path / "x.pos.stat"
    _write_pos(pos, week, tow)
    _write_stat_multifreq(stat, week, tow)

    feats = build_epoch_features(pos, stat)
    assert len(feats) == 1
    ef = feats[0]

    # 4 unique sources, NOT 12 frequency rows.
    assert ef.n_sat_used == 4, f"expected 4 unique used sats, got {ef.n_sat_used}"
    assert ef.n_sat_visible == 4
    assert ef.n_GPS == 4

    # DOP must be finite and physically sane (>0) for 4 sources, never 0.0
    # (the rank-deficient-H symptom of the old duplicate-row bug).
    assert math.isfinite(ef.HDOP) and ef.HDOP > 0.0
    assert math.isfinite(ef.PDOP) and ef.PDOP > 0.0

    # fix_flag_pct must never exceed 100% (denominator is now per-PRN too).
    assert 0.0 <= ef.fix_flag_pct <= 100.0
