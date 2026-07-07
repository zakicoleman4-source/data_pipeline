"""GUI smoke tests for the Motion sensor Calibration tab (JOB D).

Instantiates the real App (no mainloop), confirms the tab + its callbacks
exist, and exercises the compute/format/export helpers with a synthetic
sensors file so the wiring is verified without a real drive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import tkinter

_TK_AVAILABLE = True
try:
    _root = tkinter.Tk()
    _root.withdraw()
    _root.destroy()
except Exception:
    _TK_AVAILABLE = False
pytestmark = pytest.mark.skipif(not _TK_AVAILABLE, reason="No Tk display")


GPS_EPOCH_UNIX_S = 315964800
LEAP = 18


def _write_static_sensors(path: Path, n=20_000, fs=200.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            utc = 1.6e9 + i * dt
            gps_s = utc - GPS_EPOCH_UNIX_S + LEAP
            f.write(f"{gps_s:.3f},"
                    f"{rng.normal(0,0.001)},{rng.normal(0,0.001)},{rng.normal(0,0.001)},"
                    f"{rng.normal(0,0.02)},{rng.normal(0,0.02)},{9.81+rng.normal(0,0.02)}\n")


@pytest.fixture
def app(tmp_path: Path):
    import data_pipeline.gui as gui_mod
    orig_recent = gui_mod._RECENT_FILE
    gui_mod._RECENT_FILE = tmp_path / "recent.json"
    from data_pipeline.gui import App
    a = App()
    a.root.update()
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass
        gui_mod._RECENT_FILE = orig_recent


def test_calib_tab_present(app):
    nb = None
    for attr in ("_nb",):
        nb = getattr(app, attr, None)
        if nb is not None:
            break
    assert nb is not None
    tabs = [nb.tab(i, "text") for i in range(nb.index("end"))]
    flat = " | ".join(tabs).lower()
    assert "imu" in flat and "calib" in flat, f"IMU Calib tab missing: {tabs}"


@pytest.mark.parametrize("name", [
    "_build_imu_calib_tab", "_run_allan_calibration", "_show_allan_plot",
    "_export_calibration", "_load_calibration_file",
    "_find_calibration_by_label", "_run_before_after_fusion",
    "_format_calibration", "_before_after_report", "_traj_2sigma",
])
def test_calib_methods_present(app, name):
    assert hasattr(app, name) and callable(getattr(app, name))


def test_calib_vars_present(app):
    for v in ("var_calib_sensors", "var_calib_label", "var_calib_drive_pos",
              "var_calib_loaded", "var_calib_use_in_fusion", "var_calib_ba_pos"):
        assert hasattr(app, v), f"missing {v}"


def test_format_calibration_renders(app, tmp_path):
    from data_pipeline.imu_calibration import compute_calibration
    p = tmp_path / "sensors_static.txt"
    _write_static_sensors(p, n=12_000)
    cal = compute_calibration("GUI Test Device", static_imu_path=p)
    txt = app._format_calibration(cal)
    assert "GUI Test Device" in txt
    assert "mean accel VRW" in txt
    assert "ax" in txt and "gz" in txt


def test_before_after_report_renders(app):
    """The report helper produces a BEFORE/AFTER table without a real run."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.smoothers import run_smoother
    from data_pipeline.imu_calibration import compute_calibration
    from data_pipeline.parsers import ImuRow

    rows = [PosRow(utc_s=float(s), lat_deg=32.0 + s * 1e-5, lon_deg=34.0 + s * 1e-5,
                   h_m=40.0, quality=1, ns=12, sd_n=0.2, sd_e=0.2, sd_u=0.4,
                   vn=8.0, ve=0.3, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1)
            for s in range(80)]
    dt = 1.0 / 200.0
    rng = np.random.default_rng(0)
    imu = [ImuRow(utc_s=i * dt, ax=rng.normal(0, 1.0), ay=rng.normal(0, 1.0),
                  az=9.81 + rng.normal(0, 1.0), gx=rng.normal(0, 0.01),
                  gy=rng.normal(0, 0.01), gz=rng.normal(0, 0.01))
           for i in range(10_000)]
    cal = compute_calibration("BA Device", imu_rows=imu)

    before = run_smoother("epoch_weight_v2", rows, imu_rows=None)
    after = run_smoother("epoch_weight_v2", rows, imu_rows=None, calibration=cal)
    report = app._before_after_report(before, after, cal)
    assert "BEFORE" in report and "AFTER" in report
    assert "2σ horiz" in report
    assert "BA Device" in report
