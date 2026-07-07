"""Tests for the two 2026-07-07 export-control features:

* Coordinate-source chooser: ``resolve_export_rows`` picks the exported
  trajectory (raw PPK or a specific smoother's output) independently of the
  pipeline smoother, and the CLI exposes it as ``--export-source``.
* Final-velocity export + coord/Doppler disagreement gate:
  ``export_trajectory(emit_final_velocity=..., vel_disagree_threshold_mps=...,
  drop_coords_on_vel_disagree=...)`` appends ``final_v*`` (raw PPK Doppler)
  columns and, over the threshold, blanks final_v* AND the coordinate columns
  (row kept, ``coords_dropped=1`` + ``vel_disagree_mps`` visible).

Backward-compat guard: the default ``export_trajectory(rows, path)`` header
is byte-identical to the historical column set.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from data_pipeline.parsers import PosRow
from data_pipeline.smoothers import list_smoothers
from data_pipeline.stages.user_export import (
    export_trajectory,
    resolve_export_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAT0, _LON0, _H0 = 32.06, 34.80, 50.0
_MLON = 111_320.0 * math.cos(math.radians(_LAT0))


def _drive_row(i, *, ve=13.0, vn=0.0, east_noise_m=0.0):
    """1 Hz eastward drive at ``ve`` m/s with clean coordinates."""
    east = 13.0 * i + east_noise_m
    return PosRow(
        utc_s=1000.0 + i,
        lat_deg=_LAT0, lon_deg=_LON0 + east / _MLON, h_m=_H0,
        quality=1, ns=12,
        vn=vn, ve=ve, vu=0.0,
        sd_n=0.2, sd_e=0.2, sd_u=0.4,
        sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
    )


def _read(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    return list(rdr), rdr.fieldnames


def _export_kwargs(**over):
    """Deterministic export: no filter/smoothing/suppression interference."""
    kw = dict(robust_filter_enabled=False, smooth_z=False,
              suppress_inaccurate=False)
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# FEATURE 1 — resolve_export_rows (coordinate-source chooser)
# ---------------------------------------------------------------------------

def test_resolve_raw_returns_rows_unchanged():
    raw = [_drive_row(i, east_noise_m=2.0 * math.sin(i)) for i in range(30)]
    out = resolve_export_rows(raw, source="raw")
    assert out == raw                 # same values, unchanged
    assert out is not raw             # but a fresh list (caller-safe)
    assert all(a is b for a, b in zip(out, raw))


def test_resolve_default_source_is_raw():
    raw = [_drive_row(i) for i in range(10)]
    assert resolve_export_rows(raw) == raw


def test_resolve_real_smoother_returns_different_rows():
    # Noisy path so a gaussian smoother measurably moves the coordinates.
    raw = [_drive_row(i, east_noise_m=2.0 * math.sin(1.7 * i))
           for i in range(60)]
    assert "gaussian_car" in list_smoothers()
    out = resolve_export_rows(raw, source="gaussian_car")
    assert len(out) == len(raw)
    moved = sum(
        1 for a, b in zip(raw, out)
        if abs(a.lon_deg - b.lon_deg) > 1e-9 or abs(a.lat_deg - b.lat_deg) > 1e-9
    )
    assert moved > 0, "smoother output must differ from raw PPK"


def test_resolve_unknown_source_raises_value_error_listing_options():
    raw = [_drive_row(i) for i in range(5)]
    with pytest.raises(ValueError) as ei:
        resolve_export_rows(raw, source="not_a_smoother")
    msg = str(ei.value)
    assert "raw" in msg
    assert "gaussian_car" in msg      # options are listed


# ---------------------------------------------------------------------------
# FEATURE 5 — final velocity columns + disagreement gate
# ---------------------------------------------------------------------------

def test_final_velocity_columns_are_raw_doppler_when_no_threshold(tmp_path):
    rows = [_drive_row(i) for i in range(20)]
    res = export_trajectory(rows, tmp_path / "t.csv",
                            **_export_kwargs(emit_final_velocity=True))
    assert res.final_velocity_emitted is True
    assert res.vel_disagree_threshold_mps is None
    assert res.n_coords_dropped == 0 and res.n_vel_disagree == 0
    data, cols = _read(res.csv_path)
    for c in ("final_vn_mps", "final_ve_mps", "final_vu_mps",
              "final_speed_mps", "vel_disagree_mps", "coords_dropped"):
        assert c in cols
    assert len(data) == 20
    for d in data:
        # final_v* = raw PPK Doppler, always populated (threshold None).
        assert float(d["final_vn_mps"]) == pytest.approx(0.0)
        assert float(d["final_ve_mps"]) == pytest.approx(13.0)
        assert float(d["final_vu_mps"]) == pytest.approx(0.0)
        assert float(d["final_speed_mps"]) == pytest.approx(13.0, abs=1e-3)
        assert d["coords_dropped"] == "0"
        assert d["lat_deg"] != ""     # coords never dropped
        # Coord-derived vel matches Doppler on a clean constant drive.
        assert float(d["vel_disagree_mps"]) < 0.5


def test_disagree_over_threshold_blanks_final_vel_and_coords(tmp_path):
    rows = [_drive_row(i) for i in range(20)]
    # Row 10: coordinates stay on the clean eastward line, but the Doppler
    # claims 25 m/s NORTH -> |coord_vel - doppler_vel| ~= 25 m/s.
    rows[10] = _drive_row(10, vn=25.0)
    res = export_trajectory(
        rows, tmp_path / "t.csv",
        **_export_kwargs(vel_disagree_threshold_mps=5.0),
    )
    assert res.final_velocity_emitted is True   # threshold implies emit
    assert res.n_vel_disagree == 1
    assert res.n_coords_dropped == 1            # drop_coords default True
    data, cols = _read(res.csv_path)
    assert len(data) == 20                      # row KEPT, not deleted
    bad = data[10]
    # final velocity empty:
    assert bad["final_vn_mps"] == ""
    assert bad["final_ve_mps"] == ""
    assert bad["final_vu_mps"] == ""
    assert bad["final_speed_mps"] == ""
    # coordinates empty (geodetic + derived ECEF):
    assert bad["lat_deg"] == "" and bad["lon_deg"] == "" and bad["h_m"] == ""
    assert bad["x_ecef_m"] == "" and bad["y_ecef_m"] == "" and bad["z_ecef_m"] == ""
    # omission is visible, not silent:
    assert bad["coords_dropped"] == "1"
    assert bad["gpstime"] != ""
    assert float(bad["vel_disagree_mps"]) == pytest.approx(25.0, abs=1.0)
    # neighbouring rows (within threshold) keep everything:
    for j in (9, 11):
        good = data[j]
        assert good["coords_dropped"] == "0"
        assert good["lat_deg"] != "" and good["x_ecef_m"] != ""
        assert float(good["final_ve_mps"]) == pytest.approx(13.0)


def test_disagree_gate_can_keep_coords(tmp_path):
    """drop_coords_on_vel_disagree=False: final_v* blanked over threshold but
    coordinates still ship."""
    rows = [_drive_row(i) for i in range(20)]
    rows[10] = _drive_row(10, vn=25.0)
    res = export_trajectory(
        rows, tmp_path / "t.csv",
        **_export_kwargs(vel_disagree_threshold_mps=5.0,
                         drop_coords_on_vel_disagree=False),
    )
    assert res.n_vel_disagree == 1
    assert res.n_coords_dropped == 0
    data, _ = _read(res.csv_path)
    bad = data[10]
    assert bad["final_ve_mps"] == ""            # velocity not certifiable
    assert bad["lat_deg"] != ""                 # coords kept
    assert bad["coords_dropped"] == "0"


def test_within_threshold_rows_keep_coords_and_final_vel(tmp_path):
    rows = [_drive_row(i) for i in range(20)]
    res = export_trajectory(
        rows, tmp_path / "t.csv",
        **_export_kwargs(vel_disagree_threshold_mps=5.0),
    )
    assert res.n_vel_disagree == 0 and res.n_coords_dropped == 0
    data, _ = _read(res.csv_path)
    for d in data:
        assert d["coords_dropped"] == "0"
        assert d["lat_deg"] != ""
        assert float(d["final_ve_mps"]) == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# Backward compatibility guard
# ---------------------------------------------------------------------------

LEGACY_COLS = [
    "gpstime", "lat_deg", "lon_deg", "h_m",
    "x_ecef_m", "y_ecef_m", "z_ecef_m",
    "vn_mps", "ve_mps", "vu_mps",
    "speed_mps", "vel_error_pct_speed",
    "std_xy_m", "std_xy_smart_m",
    "err_horiz_2sigma_m", "err_speed_2sigma_mps", "err_speed_2sigma_kmh",
    "std_vn_mps", "std_ve_mps", "std_vu_mps",
    "trust_class", "source", "trust_label_v2", "gap",
    "pos_within_bar", "vel_trusted",
]


def test_default_export_columns_unchanged(tmp_path):
    """Guard: without the new opt-ins the header is EXACTLY the historical
    column set — no final_* columns sneak in."""
    rows = [_drive_row(i) for i in range(20)]
    res = export_trajectory(rows, tmp_path / "t.csv")
    _, cols = _read(res.csv_path)
    assert cols == LEGACY_COLS
    assert res.final_velocity_emitted is False


# ---------------------------------------------------------------------------
# CLI wiring (scripts/run_pipeline_from_raw.py)
# ---------------------------------------------------------------------------

_REQ = ["--raw", "r", "--base-obs", "b.obs", "--nav", "n.nav", "--out", "o"]


def test_cli_export_source_and_velocity_flags_parse():
    from scripts.run_pipeline_from_raw import parse_args
    args = parse_args(_REQ + ["--export-source", "raw",
                              "--vel-disagree-threshold", "2.5",
                              "--emit-final-velocity"])
    assert args.export_source == "raw"
    assert args.vel_disagree_threshold == 2.5
    assert args.emit_final_velocity is True


def test_cli_defaults_keep_current_behavior():
    from scripts.run_pipeline_from_raw import parse_args
    args = parse_args(_REQ)
    assert args.export_source is None            # unset -> historical wiring
    assert args.vel_disagree_threshold is None
    assert args.emit_final_velocity is False


def test_cli_export_source_offers_raw_plus_all_smoothers():
    from scripts.run_pipeline_from_raw import _export_source_choices, parse_args
    choices = _export_source_choices()
    assert choices[0] == "raw"
    assert set(list_smoothers()).issubset(set(choices))
    # A smoother name is accepted, decoupled from --smoother.
    args = parse_args(_REQ + ["--export-source", "gaussian_car",
                              "--smoother", "epoch_weighted_v2"])
    assert args.export_source == "gaussian_car"
    assert args.smoother == "epoch_weighted_v2"


def test_cli_export_source_rejects_unknown():
    from scripts.run_pipeline_from_raw import parse_args
    with pytest.raises(SystemExit):
        parse_args(_REQ + ["--export-source", "nope"])
