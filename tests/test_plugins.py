"""Tests for the pluggable fusion/calibration plugin interface.

Covers:
* registering valid fusion + calibration plugins (pass + retrievable);
* registering a malformed plugin raises PluginError (fail loudly);
* the drop-in loader picks up the shipped example plugins;
* get_/list_ accessors behave;
* the entry-points discovery path tolerates none installed.
"""
from __future__ import annotations

import datetime as dt

import pytest

from data_pipeline import plugins_api
from data_pipeline.parsers import ImuRow, PosRow
from data_pipeline.plugins_api import (
    PluginError,
    clear_registry,
    get_calibration_plugin,
    get_fusion_plugin,
    list_calibration_plugins,
    list_fusion_plugins,
    register_calibration,
    register_fusion,
    validate_calibration_plugin,
    validate_fusion_plugin,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a clean registry, restored afterward."""
    saved_fusion = dict(plugins_api._FUSION_PLUGINS)
    saved_calib = dict(plugins_api._CALIBRATION_PLUGINS)
    clear_registry()
    try:
        yield
    finally:
        clear_registry()
        plugins_api._FUSION_PLUGINS.update(saved_fusion)
        plugins_api._CALIBRATION_PLUGINS.update(saved_calib)


# ---------------------------------------------------------------------------
# Helpers — valid plugins
# ---------------------------------------------------------------------------

def _good_calib_dict(label="device", n=8):
    axis = {"random_walk": 0.01, "bias_instability": 0.001,
            "rate_random_walk": 0.0001}
    return {
        "device_label": label,
        "date": dt.date.today().isoformat(),
        "sample_rate_hz": 200.0,
        "duration_s": n / 200.0,
        "source": "test",
        "axes": {ax: dict(axis) for ax in
                 ("gx", "gy", "gz", "ax", "ay", "az")},
    }


# ---------------------------------------------------------------------------
# Valid registration
# ---------------------------------------------------------------------------

def test_register_valid_fusion_plugin():
    @register_fusion("good_fusion")
    class GoodFusion:
        name = "good_fusion"

        def run(self, pos_rows, imu_rows, calibration, options):
            return list(pos_rows)

    assert "good_fusion" in list_fusion_plugins()
    plugin = get_fusion_plugin("good_fusion")
    rows = [PosRow(utc_s=1.0, lat_deg=37.0, lon_deg=-122.0, h_m=1.0,
                   quality=1)]
    out = plugin.run(rows, [], None, {})
    assert out == rows
    assert all(isinstance(r, PosRow) for r in out)


def test_register_valid_calibration_plugin():
    @register_calibration("good_calib")
    class GoodCalib:
        name = "good_calib"

        def compute(self, sensors_rows, options):
            return _good_calib_dict((options or {}).get("device_label", "x"))

    assert "good_calib" in list_calibration_plugins()
    plugin = get_calibration_plugin("good_calib")
    out = plugin.compute([], {"device_label": "Eli"})
    assert out["device_label"] == "Eli"
    assert set(plugins_api.REQUIRED_CALIBRATION_KEYS).issubset(out)


def test_fusion_instance_registration_via_object():
    # Registering a class instance (not the class) also works.
    class F:
        name = "inst_fusion"

        def run(self, pos_rows, imu_rows, calibration, options):
            return list(pos_rows)

    register_fusion("inst_fusion")(F())
    assert "inst_fusion" in list_fusion_plugins()


# ---------------------------------------------------------------------------
# Malformed plugins -> PluginError
# ---------------------------------------------------------------------------

def test_malformed_fusion_wrong_signature_raises():
    with pytest.raises(PluginError) as ei:
        @register_fusion("bad_sig")
        class BadSig:
            name = "bad_sig"

            def run(self, pos_rows):  # missing imu_rows/calibration/options
                return list(pos_rows)
    assert "missing required parameter" in str(ei.value)
    assert "bad_sig" not in list_fusion_plugins()


def test_malformed_fusion_wrong_output_raises():
    with pytest.raises(PluginError) as ei:
        @register_fusion("bad_out")
        class BadOut:
            name = "bad_out"

            def run(self, pos_rows, imu_rows, calibration, options):
                return "not a list"
    assert "must return list" in str(ei.value)
    assert "bad_out" not in list_fusion_plugins()


def test_malformed_fusion_non_posrow_elements_raises():
    with pytest.raises(PluginError) as ei:
        @register_fusion("bad_elem")
        class BadElem:
            name = "bad_elem"

            def run(self, pos_rows, imu_rows, calibration, options):
                return [1, 2, 3]
    assert "non-PosRow" in str(ei.value)


def test_malformed_fusion_missing_name_raises():
    class NoName:
        def run(self, pos_rows, imu_rows, calibration, options):
            return list(pos_rows)

    with pytest.raises(PluginError):
        # explicit name omitted AND no .name attribute
        register_fusion()(NoName())


def test_malformed_calibration_missing_keys_raises():
    with pytest.raises(PluginError) as ei:
        @register_calibration("bad_calib")
        class BadCalib:
            name = "bad_calib"

            def compute(self, sensors_rows, options):
                return {"device_label": "x"}  # missing date/axes/etc
    assert "missing required key" in str(ei.value)
    assert "bad_calib" not in list_calibration_plugins()


def test_malformed_calibration_bad_axes_raises():
    with pytest.raises(PluginError) as ei:
        @register_calibration("bad_axes")
        class BadAxes:
            name = "bad_axes"

            def compute(self, sensors_rows, options):
                d = _good_calib_dict()
                d["axes"]["gx"] = {"random_walk": 0.1}  # missing two keys
                return d
    assert "gx" in str(ei.value)


def test_calibration_raising_compute_is_reported():
    with pytest.raises(PluginError) as ei:
        @register_calibration("boom")
        class Boom:
            name = "boom"

            def compute(self, sensors_rows, options):
                raise RuntimeError("kaboom")
    assert "kaboom" in str(ei.value)


# ---------------------------------------------------------------------------
# Standalone validators
# ---------------------------------------------------------------------------

def test_validate_fusion_accepts_kwargs_signature():
    class KwFusion:
        name = "kw"

        def run(self, *args, **kwargs):
            return list(args[0])

    validate_fusion_plugin(KwFusion())  # should not raise


def test_validate_calibration_accepts_dataclass_axis_values():
    from dataclasses import dataclass

    @dataclass
    class Ax:
        random_walk: float = 0.01
        bias_instability: float = 0.001
        rate_random_walk: float = 0.0001

    class DcCalib:
        name = "dc"

        def compute(self, sensors_rows, options):
            d = _good_calib_dict()
            d["axes"] = {ax: Ax() for ax in ("gx", "gy", "gz",
                                             "ax", "ay", "az")}
            return d

    validate_calibration_plugin(DcCalib())  # should not raise


# ---------------------------------------------------------------------------
# Drop-in loader picks up the shipped example
# ---------------------------------------------------------------------------

def _force_reimport_example():
    """Drop the cached example module so its decorators re-run after the
    autouse fixture clears the registry (import is otherwise a no-op)."""
    import sys
    sys.modules.pop("data_pipeline.plugins.example_passthrough", None)


def test_dropin_loader_picks_up_example():
    from data_pipeline.plugin_loader import load_dropin_plugins

    _force_reimport_example()
    report = load_dropin_plugins()
    assert "example_passthrough" in list_fusion_plugins()
    assert "example_fixed" in list_calibration_plugins()
    assert any("example_passthrough" in m for m in report.loaded_modules)

    # The example fusion is a genuine passthrough.
    rows = [PosRow(utc_s=1.0, lat_deg=37.0, lon_deg=-122.0, h_m=1.0,
                   quality=1)]
    fused = get_fusion_plugin("example_passthrough").run(rows, [], None, {})
    assert fused == rows

    # The example calibration conforms to the schema.
    cal = get_calibration_plugin("example_fixed").compute(
        [ImuRow(utc_s=0.0, ax=0, ay=0, az=9.81, gx=0, gy=0, gz=0)],
        {"device_label": "demo"},
    )
    assert cal["device_label"] == "demo"
    assert set(plugins_api.REQUIRED_CALIBRATION_KEYS).issubset(cal)


# ---------------------------------------------------------------------------
# get_/list_ behaviour
# ---------------------------------------------------------------------------

def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_fusion_plugin("nope")
    with pytest.raises(KeyError):
        get_calibration_plugin("nope")


def test_list_empty_on_clean_registry():
    assert list_fusion_plugins() == []
    assert list_calibration_plugins() == []


# ---------------------------------------------------------------------------
# Entry-points path tolerates none installed
# ---------------------------------------------------------------------------

def test_entry_points_tolerates_none(monkeypatch):
    from data_pipeline import plugin_loader

    report = plugin_loader.load_entry_point_plugins()
    # No data_pipeline.* entry points are installed in the test env, so
    # nothing loads and (importantly) nothing crashes.
    assert isinstance(report.loaded_entry_points, list)
    # No errors expected purely from "none installed".
    assert all(src != plugin_loader.FUSION_ENTRY_POINT_GROUP
               for src, _ in report.errors)


def test_load_all_plugins_combines_paths():
    from data_pipeline.plugin_loader import load_all_plugins

    _force_reimport_example()
    report = load_all_plugins()
    assert "example_passthrough" in report.fusion_plugins
    assert "example_fixed" in report.calibration_plugins
    # summary() is renderable.
    assert "fusion" in report.summary()
