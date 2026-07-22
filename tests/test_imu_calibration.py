"""Tests for data_pipeline.imu_calibration — compute/save/load by label."""

import math

import numpy as np
import pytest

from data_pipeline.imu_calibration import (
    AxisParams,
    ImuCalibration,
    compute_calibration,
    export_calibration,
    find_calibration_by_label,
    load_calibration,
    save_calibration,
)
from data_pipeline.parsers import ImuRow, PosRow


def _static_imu_rows(n=40_000, fs=200.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    return [
        ImuRow(
            utc_s=i * dt,
            ax=rng.normal(0, 0.02), ay=rng.normal(0, 0.02), az=9.81 + rng.normal(0, 0.02),
            gx=rng.normal(0, 0.001), gy=rng.normal(0, 0.001), gz=rng.normal(0, 0.001),
        )
        for i in range(n)
    ]


def _write_sensors_file(path, rows):
    # File columns: GPS_seconds, gx, gy, gz, ax, ay, az
    # parse_imu converts col0 Reference seconds -> UTC, so write Reference seconds.
    GPS_EPOCH_UNIX_S = 315964800
    LEAP = 18
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            gps_s = r.utc_s - GPS_EPOCH_UNIX_S + LEAP
            f.write(f"{gps_s:.3f},{r.gx},{r.gy},{r.gz},{r.ax},{r.ay},{r.az}\n")


def test_compute_from_dedicated_static_rows():
    rows = _static_imu_rows()
    cal = compute_calibration("Test Device A", imu_rows=rows)
    assert cal.source == "dedicated_static"
    assert cal.device_label == "Test Device A"
    assert cal.sample_rate_hz == pytest.approx(200.0, rel=0.02)
    assert set(cal.axes) == {"gx", "gy", "gz", "ax", "ay", "az"}
    assert cal.mean_accel_vrw() > 0
    assert cal.mean_gyro_arw() > 0


def test_compute_from_static_file(tmp_path):
    rows = _static_imu_rows(n=20_000)
    p = tmp_path / "sensors_static.txt"
    _write_sensors_file(p, rows)
    cal = compute_calibration("Desk Device", static_imu_path=p)
    assert cal.source == "dedicated_static"
    assert cal.n_samples > 10_000


def test_save_load_round_trip(tmp_path):
    rows = _static_imu_rows(n=15_000)
    cal = compute_calibration("RoundTrip Device", imu_rows=rows)
    p = tmp_path / "calib.json"
    save_calibration(cal, p)
    assert p.exists()

    loaded = load_calibration(p)
    assert loaded.device_label == cal.device_label
    assert loaded.source == cal.source
    assert loaded.sample_rate_hz == pytest.approx(cal.sample_rate_hz)
    for axn in cal.axes:
        assert isinstance(loaded.axes[axn], AxisParams)
        assert loaded.axes[axn].random_walk == pytest.approx(
            cal.axes[axn].random_walk, rel=1e-9
        )


def test_find_by_label_match_and_miss(tmp_path):
    rows = _static_imu_rows(n=12_000)
    cal = compute_calibration("Eli's S23 Ultra", imu_rows=rows)
    export_calibration(cal, tmp_path)

    # Case-insensitive match on the stored label, not the filename.
    found = find_calibration_by_label(tmp_path, "eli's s23 ultra")
    assert found is not None
    assert found.device_label == "Eli's S23 Ultra"

    # No match for a different label.
    assert find_calibration_by_label(tmp_path, "Some Other Device") is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "does_not_exist.json")


def test_find_in_missing_dir_returns_none(tmp_path):
    assert find_calibration_by_label(tmp_path / "nope", "x") is None


def test_blank_label_rejected():
    with pytest.raises(ValueError):
        compute_calibration("   ", imu_rows=_static_imu_rows(n=5000))


def test_mine_zupt_from_drive():
    """When only a drive is available, mine stationary ZUPT segments."""
    fs = 200.0
    dt = 1.0 / fs
    rng = np.random.default_rng(5)
    # 600 s drive; Motion sensor rows the whole time.
    n = int(600 * fs)
    imu = [
        ImuRow(
            utc_s=i * dt,
            ax=rng.normal(0, 0.02), ay=rng.normal(0, 0.02), az=9.81 + rng.normal(0, 0.02),
            gx=rng.normal(0, 0.001), gy=rng.normal(0, 0.001), gz=rng.normal(0, 0.001),
        )
        for i in range(n)
    ]
    # PosRow stream at 1 Hz: two long stops (static) + moving in between.
    pos = []
    for s in range(600):
        moving = not (50 <= s < 130 or 300 <= s < 420)  # two stationary windows
        v = 8.0 if moving else 0.0
        pos.append(PosRow(
            utc_s=float(s), lat_deg=32.0, lon_deg=34.0, h_m=40.0,
            quality=1, ns=12, sd_n=0.1, sd_e=0.1, sd_u=0.2,
            vn=v, ve=0.0, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
        ))
    cal = compute_calibration(
        "Drive Device", drive_pos_rows=pos, imu_rows=imu,
    )
    assert cal.source == "mined_zupt"
    assert cal.n_static_segments >= 2
    assert any("ZUPT" in w for w in cal.warnings)
    assert cal.mean_accel_vrw() > 0


def test_no_source_raises():
    with pytest.raises(ValueError):
        compute_calibration("X")
