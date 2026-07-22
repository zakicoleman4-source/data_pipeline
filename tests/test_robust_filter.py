"""Tests for the GT-free physical-plausibility robust filter.

A synthetic car path (constant-speed straight drive at ~13 m/s, 1 Hz) is
built, then corrupted with the exact day14 failure signatures:

  * a 160 m altitude spike (PP3 s21/101315),
  * a 172 km/h horizontal teleport (PP3),

and the filter is asserted to reject/repair the bad epochs, drop the MAX
horizontal error, leave clean epochs untouched, and emit a gap flag.
"""
from __future__ import annotations

import math

import numpy as np

from data_pipeline.geo import ecef_to_enu, llh_to_ecef
from data_pipeline.parsers import PosRow
from data_pipeline.robust_filter import (
    DROP,
    KEEP,
    REPAIR,
    RobustFilterConfig,
    car_preset,
    clean_before_smoothing,
    detect,
    robust_filter,
)


LAT0, LON0, H0 = 32.06, 34.80, 47.0
MLAT = 111_320.0
MLON = 111_320.0 * math.cos(math.radians(LAT0))


def _row(t, east_m, north_m, h=H0, q=1, ns=14):
    return PosRow(
        utc_s=float(t),
        lat_deg=LAT0 + north_m / MLAT,
        lon_deg=LON0 + east_m / MLON,
        h_m=float(h),
        quality=q, ns=ns,
        sd_n=0.3, sd_e=0.3, sd_u=0.5,
    )


def _clean_traj(n=120, speed=13.0):
    """Straight eastward drive at ``speed`` m/s, 1 Hz, flat altitude."""
    return [_row(i, east_m=speed * i, north_m=0.0) for i in range(n)]


def _horiz_offset_m(a: PosRow, b: PosRow) -> float:
    ref = (b.lat_deg, b.lon_deg, b.h_m)
    ea, na, _ = ecef_to_enu(*llh_to_ecef(a.lat_deg, a.lon_deg, a.h_m), ref)
    return math.hypot(ea, na)


# ---------------------------------------------------------------------------
# Clean path passes through untouched (no-harm)
# ---------------------------------------------------------------------------
def test_clean_trajectory_untouched():
    rows = _clean_traj()
    res = robust_filter(rows, car_preset())
    assert res.n_dropped == 0
    assert res.n_repaired == 0
    assert res.n_kept == len(rows)
    assert all(v.outcome == KEEP for v in res.verdicts)
    # positions identical
    for a, b in zip(rows, res.rows):
        assert a.lat_deg == b.lat_deg
        assert a.lon_deg == b.lon_deg
        assert a.h_m == b.h_m


def test_disabled_is_identity():
    rows = _clean_traj()
    rows[50] = _row(50, east_m=5000.0, north_m=0.0, h=160.0)   # garbage
    cfg = RobustFilterConfig(enabled=False)
    res = robust_filter(rows, cfg)
    assert res.rows == rows
    assert res.n_dropped == 0 and res.n_repaired == 0


# ---------------------------------------------------------------------------
# 160 m altitude spike (PP3) -> rejected/repaired
# ---------------------------------------------------------------------------
def test_altitude_spike_rejected_and_repaired():
    rows = _clean_traj()
    # inject the s21/101315 signature: one epoch leaps to 160 m altitude
    bad_i = 60
    rows[bad_i] = _row(bad_i, east_m=13.0 * bad_i, north_m=0.0, h=160.2, q=4)

    reasons = detect(rows, car_preset())
    assert "alt_high" in reasons[bad_i]
    # vertical-speed gate also catches the jump to/from 160 m
    assert "vert_speed" in reasons[bad_i]

    res = robust_filter(rows, car_preset())
    # single bad epoch bracketed by good -> repaired, not dropped
    assert res.verdicts[bad_i].outcome == REPAIR
    # repaired altitude is interpolated back to ~47 m, not 160 m
    assert abs(res.rows[bad_i].h_m - H0) < 1.0
    # gap flag emitted
    assert res.verdicts[bad_i].gap


# ---------------------------------------------------------------------------
# 172 km/h horizontal teleport (PP3) -> max error drops
# ---------------------------------------------------------------------------
def test_horizontal_teleport_max_error_drops():
    rows = _clean_traj()
    bad_i = 70
    # 172 km/h = 47.7 m/s; teleport ~180 m sideways in one 1 s epoch
    rows[bad_i] = _row(bad_i, east_m=13.0 * bad_i, north_m=180.0)

    # MAX deviation from a straight line BEFORE filtering
    truth_north = 0.0
    def _north(r):
        ref = (LAT0, LON0, H0)
        _, n, _ = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)
        return n
    max_before = max(abs(_north(r) - truth_north) for r in rows)
    assert max_before > 150.0   # the spike

    reasons = detect(rows, car_preset())
    assert reasons[bad_i] & {"horiz_speed", "pos_jump"}

    res = robust_filter(rows, car_preset())
    max_after = max(abs(_north(r) - truth_north) for r in res.rows)
    assert max_after < 5.0      # spike crushed by repair
    assert max_after < max_before
    assert res.verdicts[bad_i].outcome in (REPAIR, DROP)


# ---------------------------------------------------------------------------
# Long bad run -> hard-dropped (flagged gap), short run -> repaired
# ---------------------------------------------------------------------------
def test_long_bad_run_is_dropped_with_gap_flag():
    rows = _clean_traj(n=140)
    # a long contiguous garbage run (20 epochs > max_repair_epochs=10)
    for i in range(40, 60):
        rows[i] = _row(i, east_m=13.0 * i, north_m=300.0, h=160.0, q=4)
    res = robust_filter(rows, car_preset())
    dropped = [v for v in res.verdicts if v.outcome == DROP]
    # >=20: the 20 injected epochs plus the recovery epoch that trips the
    # speed/jump gate as the position snaps back are all part of one long run.
    assert len(dropped) >= 20
    assert res.n_dropped == len(dropped)
    assert res.n_repaired == 0          # too long to repair -> hard drop
    assert len(res.rows) == len(rows) - len(dropped)
    # gap flagged on the dropped epochs
    assert all(v.gap for v in dropped)


def test_short_run_repaired_not_dropped():
    rows = _clean_traj()
    for i in range(50, 53):     # 3 epochs <= max_repair_epochs
        rows[i] = _row(i, east_m=13.0 * i, north_m=200.0, h=150.0, q=4)
    res = robust_filter(rows, car_preset())
    # the 3 injected epochs (+ the recovery epoch that trips the speed gate as
    # the position snaps back) form one short run -> all repaired, none dropped.
    assert res.n_repaired >= 3
    assert res.n_dropped == 0
    # repaired north ~ interpolated back toward 0
    for i in range(50, 53):
        ref = (LAT0, LON0, H0)
        _, n, _ = ecef_to_enu(
            *llh_to_ecef(res.rows[i].lat_deg, res.rows[i].lon_deg, res.rows[i].h_m), ref)
        assert abs(n) < 5.0


# ---------------------------------------------------------------------------
# Optional multimask-disagreement gate
# ---------------------------------------------------------------------------
def test_disagreement_gate():
    rows = _clean_traj()
    dis = np.zeros(len(rows))
    dis[80] = 12.0              # high inter-mask spread -> environment noise
    reasons = detect(rows, car_preset(), disagreement=dis)
    assert "disagreement" in reasons[80]
    res = robust_filter(rows, car_preset(), disagreement=dis)
    assert res.verdicts[80].outcome in (REPAIR, DROP)


# ---------------------------------------------------------------------------
# clean_before_smoothing wrapper
# ---------------------------------------------------------------------------
def test_clean_before_smoothing_wrapper():
    rows = _clean_traj()
    rows[60] = _row(60, east_m=13.0 * 60, north_m=0.0, h=160.2, q=4)
    cleaned, res = clean_before_smoothing(rows, car_preset())
    assert len(cleaned) == len(rows)         # repaired in place
    assert res.n_repaired >= 1
    # pass-through when disabled
    cleaned2, res2 = clean_before_smoothing(rows, RobustFilterConfig(enabled=False))
    assert cleaned2 == rows
