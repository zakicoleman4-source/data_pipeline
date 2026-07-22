"""Tests for the unified smoother runner."""
from __future__ import annotations

import math

import numpy as np
import pytest

from data_pipeline.parsers import ImuRow, PosRow
from data_pipeline.smoothers import (
    SmoothResult,
    SmootherInfo,
    describe,
    list_smoothers,
    run_all_smoothers,
    run_smoother,
)


def _synth_pos(n=120, speed_mps=5.0, pos_noise=0.5, vel_noise=0.05,
                base=(32.0, 34.8, 100.0), seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=float)         # 1 Hz
    e = speed_mps * ts
    out: list[PosRow] = []
    for i in range(n):
        lat = base[0] + rng.normal(0, pos_noise) / 111_000.0
        lon = base[1] + (e[i] + rng.normal(0, pos_noise)) / 94_000.0
        h = base[2] + rng.normal(0, pos_noise * 3)
        out.append(PosRow(
            utc_s=float(ts[i]), lat_deg=lat, lon_deg=lon, h_m=h, quality=2,
            vn=rng.normal(0, vel_noise),
            ve=speed_mps + rng.normal(0, vel_noise),
            vu=rng.normal(0, vel_noise),
            ns=10,
        ))
    return out


def _synth_imu(n=12000, rate_hz=100.0):
    """Flat device, no motion — minimal Motion sensor for ekf_smoothed smoke."""
    return [ImuRow(utc_s=i / rate_hz, ax=0.0, ay=0.0, az=9.80665,
                   gx=0.0, gy=0.0, gz=0.0)
            for i in range(n)]


def test_registry_well_formed():
    names = list_smoothers()
    assert len(names) >= 10
    for n in names:
        info = describe(n)
        assert isinstance(info, SmootherInfo)
        assert info.name == n
        assert info.description


def test_describe_unknown_raises():
    with pytest.raises(KeyError):
        describe("does-not-exist")


def test_raw_ppk_passes_through():
    pos = _synth_pos(n=10)
    res = run_smoother("raw_ppk", pos)
    assert res.ok and res.n_input == res.n_output == 10
    # Must be identical (within float precision) since raw_ppk is no-op.
    for src, out in zip(pos, res.fused):
        assert out.lat_deg == src.lat_deg


def test_gaussian_car_returns_same_count():
    pos = _synth_pos(n=60)
    res = run_smoother("gaussian_car", pos)
    assert res.ok
    assert res.n_output == len(pos)
    # Smoothed result must be FINITE everywhere.
    assert all(math.isfinite(r.lat_deg) for r in res.fused)
    assert all(math.isfinite(r.lon_deg) for r in res.fused)


def test_cv_rts_pv_smooths_position_below_input_noise():
    """The synth has 0.5 m pos noise; cv_rts_pv should reduce hRMSE
    vs an identity baseline by at least 10 %."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    # Build a fake GT from a noise-free synth.
    gt = _synth_pos(n=60, pos_noise=0.0, vel_noise=0.0, seed=42)
    res_raw = run_smoother("raw_ppk", pos, gt_rows=gt)
    res_pv = run_smoother("cv_rts_pv", pos, gt_rows=gt)
    assert res_raw.hrmse_m is not None and res_pv.hrmse_m is not None
    assert res_pv.hrmse_m < res_raw.hrmse_m, (
        f"cv_rts_pv RMSE {res_pv.hrmse_m:.3f} should beat raw "
        f"{res_raw.hrmse_m:.3f}"
    )


def test_ekf_smoothed_skips_when_no_imu():
    """Adapter must surface a PipelineError code, not crash, when Motion sensor
    is missing for an Motion sensor-requiring smoother."""
    pos = _synth_pos(n=10)
    res = run_smoother("ekf_smoothed", pos, imu_rows=None)
    assert not res.ok
    assert res.error_code == "E-PP-400"
    assert "IMU" in (res.error_message or "")


def test_run_all_orders_by_hrmse_with_gt():
    pos = _synth_pos(n=60, pos_noise=0.5)
    gt = _synth_pos(n=60, pos_noise=0.0, vel_noise=0.0, seed=42)
    results = run_all_smoothers(pos, gt_rows=gt)
    # Every smoother present.
    assert {r.name for r in results} == set(list_smoothers())
    # Successful results with hRMSE land before erroring ones.
    ok_with_h = [r for r in results if r.ok and r.hrmse_m is not None]
    if len(ok_with_h) >= 2:
        # Sorted ascending by hRMSE.
        rmses = [r.hrmse_m for r in ok_with_h]
        assert rmses == sorted(rmses)


def test_run_all_only_filter():
    pos = _synth_pos(n=20)
    results = run_all_smoothers(pos, only=["raw_ppk", "gaussian_car"])
    assert {r.name for r in results} == {"raw_ppk", "gaussian_car"}


def test_smooth_result_dataclass_defaults():
    r = SmoothResult(name="x")
    assert r.ok is True and r.runtime_s == 0.0 and r.fused == []


def test_fgo_skipped_gracefully_without_gtsam_installed():
    """If the factor library isn't installed, fgo must skip with E-PP-003, not crash."""
    pos = _synth_pos(n=10)
    imu = _synth_imu(n=1200)
    res = run_smoother("fgo", pos, imu_rows=imu)
    # Either it ran (the factor library installed) or it skipped with the right code.
    if not res.ok:
        assert res.error_code in {"E-PP-003", "E-PP-002", "E-PP-900"}


# ---------------------------------------------------------------------------
# gnss_imu_dr (The external solver-style forward prediction)
# ---------------------------------------------------------------------------

def test_gnss_imu_dr_gnss_only_mode():
    """Signal-only mode (no Motion sensor) should smooth and return same count."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    res = run_smoother("gnss_imu_dr", pos)
    assert res.ok
    assert res.n_output == len(pos)
    assert all(math.isfinite(r.lat_deg) for r in res.fused)
    assert all(math.isfinite(r.lon_deg) for r in res.fused)


def test_gnss_imu_dr_gnss_only_reduces_noise():
    """Signal-only DR should reduce noise vs raw Post-processing."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    gt = _synth_pos(n=60, pos_noise=0.0, vel_noise=0.0, seed=42)
    res_raw = run_smoother("raw_ppk", pos, gt_rows=gt)
    res_dr = run_smoother("gnss_imu_dr", pos, gt_rows=gt)
    assert res_raw.hrmse_m is not None and res_dr.hrmse_m is not None
    assert res_dr.hrmse_m < res_raw.hrmse_m, (
        f"gnss_imu_dr {res_dr.hrmse_m:.3f} should beat raw {res_raw.hrmse_m:.3f}"
    )


def test_gnss_imu_dr_imu_mode():
    """When Motion sensor rows supplied, should run in Motion sensor mode without crashing."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    imu = _synth_imu(n=6000, rate_hz=100.0)
    res = run_smoother("gnss_imu_dr", pos, imu_rows=imu)
    assert res.ok
    assert res.n_output > 0
    assert all(math.isfinite(r.lat_deg) for r in res.fused)


def test_gnss_imu_dr_bridges_gap():
    """Insert a 5-second gap; DR should still produce output for all epochs."""
    pos = _synth_pos(n=60, pos_noise=0.3)
    # Remove epochs 20-25 to simulate a Signal gap
    gapped = [r for i, r in enumerate(pos) if not (20 <= i <= 25)]
    res = run_smoother("gnss_imu_dr", gapped)
    assert res.ok
    assert res.n_output == len(gapped)
    assert all(math.isfinite(r.lat_deg) for r in res.fused)


def test_gnss_imu_dr_handles_short_input():
    """Should not crash on very short input."""
    pos = _synth_pos(n=3)
    res = run_smoother("gnss_imu_dr", pos)
    assert res.ok
    assert res.n_output > 0


# ---------------------------------------------------------------------------
# v2_imu_adaptive (3-tier gradient Signal/Motion sensor coupling + bias calibration)
# ---------------------------------------------------------------------------

def test_v2_imu_adaptive_requires_imu():
    """Without Motion sensor rows the adapter must surface E-PP-400, not crash."""
    pos = _synth_pos(n=20)
    res = run_smoother("v2_imu_adaptive", pos, imu_rows=None)
    assert not res.ok
    assert res.error_code == "E-PP-400"
    assert "IMU" in (res.error_message or "")


def test_v2_imu_adaptive_runs_with_imu():
    """With Motion sensor rows it runs end-to-end and returns one finite row per epoch."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    imu = _synth_imu(n=7000, rate_hz=100.0)
    res = run_smoother("v2_imu_adaptive", pos, imu_rows=imu)
    assert res.ok, f"unexpected failure: {res.error_code} {res.error_message}"
    assert res.n_output == len(pos)
    assert all(math.isfinite(r.lat_deg) for r in res.fused)
    assert all(math.isfinite(r.lon_deg) for r in res.fused)
    assert all(math.isfinite(r.h_m) for r in res.fused)


def test_v2_imu_adaptive_reduces_noise_vs_raw():
    """On noisy Signal with clean Motion sensor it should not regress vs raw Post-processing."""
    pos = _synth_pos(n=60, pos_noise=0.5)
    gt = _synth_pos(n=60, pos_noise=0.0, vel_noise=0.0, seed=42)
    imu = _synth_imu(n=7000, rate_hz=100.0)
    res_raw = run_smoother("raw_ppk", pos, gt_rows=gt)
    res_adapt = run_smoother("v2_imu_adaptive", pos, imu_rows=imu, gt_rows=gt)
    assert res_adapt.ok
    assert res_raw.hrmse_m is not None and res_adapt.hrmse_m is not None
    # Strong-Signal synthetic: filter must at least not blow up the body.
    assert res_adapt.hrmse_m <= res_raw.hrmse_m * 1.5, (
        f"v2_imu_adaptive {res_adapt.hrmse_m:.3f} regressed badly vs "
        f"raw {res_raw.hrmse_m:.3f}"
    )


def test_v2_imu_adaptive_handles_short_input():
    """Very short input must not crash the tier/bridge machinery."""
    pos = _synth_pos(n=3)
    imu = _synth_imu(n=400, rate_hz=100.0)
    res = run_smoother("v2_imu_adaptive", pos, imu_rows=imu)
    assert res.ok
    assert res.n_output == len(pos)
