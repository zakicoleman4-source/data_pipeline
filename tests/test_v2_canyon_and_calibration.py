"""JOB C regression tests:

1. v2_urban_canyon must NOT crash with E-PP-900 (the EpochWeightV2Options
   dataclass now accepts the canyon_*/innov_gate_* kwargs the adapter passes).
2. epoch_weight_v2_imu_bridge must accept its imu_bridge_* kwargs.
3. An Motion sensor calibration maps onto the fusion process noise (sigma_a_base).
"""

import math

import numpy as np
import pytest

from data_pipeline.parsers import PosRow, ImuRow
from data_pipeline.smoothers import run_smoother
from data_pipeline.epoch_weight_v2 import (
    EpochWeightV2Options,
    options_from_calibration,
    smooth_epoch_weighted_v2,
)
from data_pipeline.imu_calibration import compute_calibration


def _drive(n=120, degraded=True):
    rows = []
    for s in range(n):
        rows.append(PosRow(
            utc_s=float(s),
            lat_deg=32.0 + s * 1e-5, lon_deg=34.0 + s * 1e-5, h_m=40.0,
            quality=5 if (degraded and s % 10 == 0) else 1,
            ns=4 if (degraded and s % 7 == 0) else 12,
            sd_n=4.0 if (degraded and s % 9 == 0) else 0.1, sd_e=0.1, sd_u=0.2,
            vn=8.0, ve=0.5, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
        ))
    return rows


def test_v2_urban_canyon_does_not_crash():
    """Previously crashed E-PP-900 because the dataclass rejected
    canyon_detect_enabled / innov_gate_* kwargs."""
    res = run_smoother("v2_urban_canyon", _drive(), imu_rows=None)
    assert res.ok, f"v2_urban_canyon failed: {res.error_code} {res.error_message}"
    assert res.error_code is None
    assert res.n_output == res.n_input


def test_v2_urban_canyon_options_accept_kwargs():
    """The dataclass must accept the canyon + innov-gate kwargs directly."""
    opts = EpochWeightV2Options(
        canyon_detect_enabled=True, canyon_ns_thresh=8, canyon_q_thresh=2,
        canyon_sigma_thresh=2.0, canyon_min_indicators=2, canyon_r_mult=15.0,
        innov_gate_enabled=True, innov_gate_thresh=5.0, innov_gate_r_mult=10.0,
    )
    assert opts.canyon_detect_enabled is True
    assert opts.innov_gate_enabled is True


def test_imu_bridge_options_accept_kwargs():
    opts = EpochWeightV2Options(
        imu_bridge_enabled=True, imu_bridge_thresh=6.0,
        imu_bridge_medium_thresh=2.5, imu_bridge_q_mult=2.0,
        imu_bridge_dw_mult=5.0,
    )
    assert opts.imu_bridge_enabled is True


def test_canyon_inflation_changes_result():
    """With canyon detection on, degraded epochs get R-inflated -> a different
    (smoother through bad fixes) path than the default."""
    rows = _drive(degraded=True)
    base = smooth_epoch_weighted_v2(rows, options=EpochWeightV2Options())
    canyon = smooth_epoch_weighted_v2(rows, options=EpochWeightV2Options(
        canyon_detect_enabled=True, canyon_min_indicators=1, canyon_r_mult=15.0,
    ))
    assert not np.allclose(base.E_smooth, canyon.E_smooth)


def test_calibration_maps_to_sigma_a_base():
    """A calibration with a known linear sensor VRW overrides sigma_a_base."""
    fs = 200.0
    n = 20_000
    dt = 1.0 / fs
    rng = np.random.default_rng(2)
    vrw = 0.05  # m/s^2/sqrt(Hz)
    rows = [
        ImuRow(utc_s=i * dt,
               ax=rng.normal(0, vrw * math.sqrt(fs)),
               ay=rng.normal(0, vrw * math.sqrt(fs)),
               az=9.81 + rng.normal(0, vrw * math.sqrt(fs)),
               gx=rng.normal(0, 0.001), gy=rng.normal(0, 0.001), gz=rng.normal(0, 0.001))
        for i in range(n)
    ]
    cal = compute_calibration("Calib Device", imu_rows=rows)

    base = EpochWeightV2Options()
    mapped = options_from_calibration(cal, base)

    # sigma_a_base should now reflect vrw * sqrt(fs), clamped to [min,max].
    expected = min(max(vrw * math.sqrt(fs), base.sigma_a_min), base.sigma_a_max)
    assert mapped.sigma_a_base == pytest.approx(expected, rel=0.25)
    assert mapped.sigma_a_base != base.sigma_a_base
    assert mapped.calib_accel_vrw is not None
    assert mapped.calib_gyro_arw is not None


def test_calibration_none_is_noop():
    base = EpochWeightV2Options()
    assert options_from_calibration(None, base) is base


def test_calibration_flows_through_adapter():
    """run_smoother with calibration= produces a different result than without
    (the calibration changes the process noise floor)."""
    fs = 200.0
    n = 15_000
    dt = 1.0 / fs
    rng = np.random.default_rng(9)
    big_vrw = 2.0  # large -> sigma_a_base jumps to the clamp
    imu = [
        ImuRow(utc_s=i * dt,
               ax=rng.normal(0, big_vrw), ay=rng.normal(0, big_vrw),
               az=9.81 + rng.normal(0, big_vrw),
               gx=rng.normal(0, 0.01), gy=rng.normal(0, 0.01), gz=rng.normal(0, 0.01))
        for i in range(n)
    ]
    cal = compute_calibration("Adapter Device", imu_rows=imu)
    rows = _drive(degraded=False)

    without = run_smoother("epoch_weight_v2", rows, imu_rows=None)
    withcal = run_smoother("epoch_weight_v2", rows, imu_rows=None, calibration=cal)
    assert without.ok and withcal.ok
    # Compare in metres (lat differences are tiny in degrees but real).
    lat_a = np.array([r.lat_deg for r in without.fused])
    lat_b = np.array([r.lat_deg for r in withcal.fused])
    diff_m = np.abs(lat_a - lat_b) * 111_320.0  # deg latitude -> metres
    assert diff_m.max() > 1e-4, f"calibration had no effect (max {diff_m.max():.2e} m)"
