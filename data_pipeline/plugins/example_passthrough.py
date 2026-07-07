"""Minimal end-to-end example plugins.

Proves the drop-in + registry path works without depending on any heavy
algorithm. Drop a file like this into ``data_pipeline/plugins/`` and the
loader imports it at startup, running the decorators below.

* :class:`PassthroughFusion` — a trivial :class:`FusionPlugin` that returns the
  input ``pos_rows`` unchanged (a no-op "fusion", useful as a baseline).
* :class:`FixedCalibration` — a trivial :class:`CalibrationPlugin` that returns
  fixed Allan-style params matching the ImuCalibration schema.
"""
from __future__ import annotations

import datetime as _dt

from ..plugins_api import register_calibration, register_fusion


@register_fusion("example_passthrough")
class PassthroughFusion:
    """No-op fusion: returns the Signal positions unchanged."""

    name = "example_passthrough"

    def run(self, pos_rows, imu_rows, calibration, options):
        # A real plugin would fuse pos_rows + imu_rows here. We just copy.
        return list(pos_rows)


@register_calibration("example_fixed")
class FixedCalibration:
    """Returns a fixed calibration dict matching the ImuCalibration schema."""

    name = "example_fixed"

    def compute(self, sensors_rows, options):
        label = (options or {}).get("device_label", "example-device")
        # Fixed, plausible device-MEMS noise figures. A real plugin would
        # estimate these from sensors_rows (e.g. via an Allan curve).
        axis = {
            "random_walk": 0.01,        # ARW (rate sensor) / VRW (linear sensor)
            "bias_instability": 0.001,
            "rate_random_walk": 0.0001,  # RRW
        }
        return {
            "device_label": label,
            "date": _dt.date.today().isoformat(),
            "sample_rate_hz": 200.0,
            "duration_s": float(len(list(sensors_rows))) / 200.0,
            "source": "example_fixed",
            "axes": {ax: dict(axis) for ax in
                     ("gx", "gy", "gz", "ax", "ay", "az")},
            "n_samples": len(list(sensors_rows)),
            "n_static_segments": 0,
            "warnings": ["fixed example calibration — not from real data"],
            "schema_version": 1,
        }
