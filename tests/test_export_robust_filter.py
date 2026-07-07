"""Regression tests for the robust_filter default baked into export_trajectory (P4).

Covers the four guarantees of the shipped "best" export profile:

  1. A clean, physically-plausible drive is a strict no-op (no rows dropped,
     no gap flags) — the filter never regresses a clean route.
  2. A physically-impossible divergence spike (the day14 s21/101315 signature:
     a horizontal teleport) is rejected — the bad epoch never reaches the CSV.
  3. A repaired/dropped span carries the new ``gap`` flag in the CSV (PP5: a
     downstream consumer can never silently bridge an un-flagged hole).
  4. PP6: the reported sigma is NOT pinned to the old constant 0.5/1.0 floor —
     a float/single session reports an honest, quality-aware sigma.

GT-free throughout; the filter uses only car-plausible physical bounds.
"""
from __future__ import annotations

import csv
import math

import pytest

from data_pipeline.accuracy_predictor import (
    quality_floor_m,
    predicted_epoch_std,
    smart_session_std,
)
from data_pipeline.parsers import PosRow
from data_pipeline.stages.user_export import export_trajectory, winning_export_filter


LAT0, LON0, H0 = 32.06, 34.80, 47.0
MLAT = 111_320.0
MLON = 111_320.0 * math.cos(math.radians(LAT0))


def _row(t, east_m, north_m, h=H0, q=2, ns=12, sd=0.4):
    return PosRow(
        utc_s=float(t),
        lat_deg=LAT0 + north_m / MLAT,
        lon_deg=LON0 + east_m / MLON,
        h_m=float(h),
        quality=q, ns=ns,
        sd_n=sd, sd_e=sd, sd_u=2.0 * sd,
        vn=0.0, ve=13.0, vu=0.0,
        sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
    )


def _clean_traj(n=120, speed=13.0):
    """Straight eastward drive at ``speed`` m/s, 1 Hz, flat altitude."""
    return [_row(i, east_m=speed * i, north_m=0.0) for i in range(n)]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. clean route -> no-op
# ---------------------------------------------------------------------------
def test_clean_route_no_op(tmp_path):
    rows = _clean_traj()
    out = tmp_path / "clean.csv"
    res = export_trajectory(rows, out, suppress_inaccurate=False)
    # default robust_filter ON, but a clean drive trips no gate:
    assert res.n_filter_repaired == 0
    assert res.n_filter_dropped == 0
    assert res.n_rows == len(rows)
    recs = _read_csv(out)
    assert len(recs) == len(rows)
    assert all(r["gap"] == "0" for r in recs)


def test_clean_route_filter_on_vs_off_identical(tmp_path):
    rows = _clean_traj()
    a = tmp_path / "on.csv"
    b = tmp_path / "off.csv"
    export_trajectory(rows, a, suppress_inaccurate=False, robust_filter_enabled=True)
    export_trajectory(rows, b, suppress_inaccurate=False, robust_filter_enabled=False)
    # Coordinates must be byte-identical on a clean route (no-op guarantee).
    ra = [(r["lat_deg"], r["lon_deg"], r["h_m"]) for r in _read_csv(a)]
    rb = [(r["lat_deg"], r["lon_deg"], r["h_m"]) for r in _read_csv(b)]
    assert ra == rb


# ---------------------------------------------------------------------------
# 2. impossible spike rejected
# ---------------------------------------------------------------------------
def test_spike_rejected(tmp_path):
    rows = _clean_traj(n=120)
    # a single horizontal teleport (~1 km jump in 1 s -> ~1000 m/s) at i=60,
    # the day14 s21/101315 divergence signature.
    rows[60] = _row(60, east_m=13.0 * 60 + 1000.0, north_m=400.0)
    out = tmp_path / "spike.csv"
    res = export_trajectory(rows, out, suppress_inaccurate=False)
    assert (res.n_filter_repaired + res.n_filter_dropped) >= 1
    recs = _read_csv(out)
    # the teleported longitude must not appear in the exported CSV
    spike_lon = rows[60].lon_deg
    assert all(abs(float(r["lon_deg"]) - spike_lon) > 1e-4 for r in recs)


def test_spike_max_error_crushed(tmp_path):
    """Filter-ON MAX deviation from the clean path is far below filter-OFF."""
    clean = _clean_traj(n=120)
    spiked = list(clean)
    spiked[60] = _row(60, east_m=13.0 * 60 + 1000.0, north_m=400.0)

    on = tmp_path / "on.csv"
    off = tmp_path / "off.csv"
    export_trajectory(spiked, on, suppress_inaccurate=False, robust_filter_enabled=True)
    export_trajectory(spiked, off, suppress_inaccurate=False, robust_filter_enabled=False)

    def max_dev(path):
        recs = _read_csv(path)
        worst = 0.0
        for r in recs:
            e = (float(r["lon_deg"]) - LON0) * MLON
            n = (float(r["lat_deg"]) - LAT0) * MLAT
            # expected east position for this epoch given a 13 m/s clean drive
            t = float(r["gpstime"])
            # use north deviation as the spike injected 400 m north
            worst = max(worst, abs(n))
        return worst

    # clean route has ~0 north; the spike injected 400 m north.
    assert max_dev(off) > 100.0      # filter off keeps the 400 m spike
    assert max_dev(on) < 50.0        # filter on removes/repairs it


# ---------------------------------------------------------------------------
# 3. gap flag on repaired/dropped span (PP5)
# ---------------------------------------------------------------------------
def test_gap_flag_emitted(tmp_path):
    rows = _clean_traj(n=120)
    # short impossible run (3 epochs) bracketed by good neighbours -> repaired,
    # boundary epochs flagged gap=1.
    for k in (60, 61, 62):
        rows[k] = _row(k, east_m=13.0 * k, north_m=300.0)  # 300 m north jump
    out = tmp_path / "gap.csv"
    res = export_trajectory(rows, out, suppress_inaccurate=False)
    assert (res.n_filter_repaired + res.n_filter_dropped) >= 1
    recs = _read_csv(out)
    assert any(r["gap"] == "1" for r in recs), "no gap flag emitted for repaired span"


def test_gap_column_present_when_filter_off(tmp_path):
    rows = _clean_traj(n=10)
    out = tmp_path / "nofilt.csv"
    export_trajectory(rows, out, suppress_inaccurate=False, robust_filter_enabled=False)
    recs = _read_csv(out)
    assert "gap" in recs[0]
    assert all(r["gap"] == "0" for r in recs)


# ---------------------------------------------------------------------------
# 4. PP6 — sigma not pinned to the old constant floor
# ---------------------------------------------------------------------------
def test_quality_floor_is_quality_aware():
    # fix is allowed tighter than the old 0.5; float/single are floored higher
    # (honest, not optimistic).
    assert quality_floor_m(1) < 0.5      # true fix can report sub-0.5
    assert quality_floor_m(2) >= 1.0     # float floor honest (was 0.5)
    assert quality_floor_m(5) >= quality_floor_m(2)  # single >= float
    # unknown / session-level fallback is NOT the old optimistic 0.5
    assert quality_floor_m(None) >= 1.0


def test_float_session_sigma_not_floored_to_half():
    """A float (Q=2) session must not report the old pinned 0.5 / 1.0 sigma."""
    rows = _clean_traj(n=60)  # all Q=2 float
    prof = smart_session_std(rows)
    # session sigma must reflect the honest float floor, not 0.5
    assert prof.smart_std_m >= 1.0
    eps = predicted_epoch_std(rows, prof)
    # no epoch may be pinned at the old 0.5 floor
    assert all(e >= 1.0 - 1e-9 for e in eps)


def test_export_2sigma_not_pinned_to_one(tmp_path):
    """err_horiz_2sigma_m must not be the old constant 1.0000 on a float run."""
    rows = _clean_traj(n=60)  # Q=2 float, tiny sd -> old code floored to 0.5/1.0
    out = tmp_path / "sig.csv"
    export_trajectory(rows, out, suppress_inaccurate=False)
    recs = _read_csv(out)
    two_sig = [float(r["err_horiz_2sigma_m"]) for r in recs if r["err_horiz_2sigma_m"]]
    assert two_sig
    # honest float 2-sigma >= 2 * 1.0 floor; the old pin produced exactly 1.0000.
    assert all(v >= 2.0 - 1e-6 for v in two_sig)
    assert not all(abs(v - 1.0) < 1e-6 for v in two_sig)


def test_winning_export_filter_matches_preset():
    cfg = winning_export_filter()
    assert cfg.enabled is True
    assert cfg.max_vert_speed_mps == 8.0
    assert cfg.max_horiz_speed_mps == 45.0
    assert cfg.alt_below_median_m == 30.0
    assert cfg.alt_above_median_m == 40.0
    assert cfg.jump_mad_k == 6.0
    assert cfg.jump_floor_m == 8.0
    assert cfg.max_repair_epochs == 10
