"""Export coordinate-system chooser + Z (height) smoothing (2026-07-05).

* Backward compatibility: the default ``export_trajectory`` call (no new
  args) must emit exactly the historical column set (datum-based + Cartesian XYZ).
* ``coord_systems`` selection: 'grid' round-trips against a direct pyproj
  transform (mm), 'local-frame' anchors at the first valid fix (first row ~ 0,0,0).
* Z smoothing: DEFAULT ON, time-weighted gaussian; reduces height noise
  variance, preserves the mean, and feeds the Cartesian XYZ z consistently.
  ``smooth_z=False`` leaves heights untouched.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import pytest

from data_pipeline.geo import llh_to_ecef
from data_pipeline.parsers import PosRow
from data_pipeline.stages.user_export import export_trajectory


LAT0, LON0, H0 = 32.06, 34.79, 55.0


def _read(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    return list(rdr), rdr.fieldnames


def _row(utc, h=H0, lat=LAT0, lon=LON0, sd_xy=0.3):
    """Position + position sigma present (smoothed-output shape)."""
    return PosRow(
        utc_s=utc, lat_deg=lat, lon_deg=lon, h_m=h, quality=2,
        ns=20, vn=1.0, ve=0.5, vu=0.0,
        sd_n=sd_xy, sd_e=sd_xy, sd_u=sd_xy * 2,
        sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
    )


def _drive(n=60, h_fn=None):
    """1 Hz drive with slightly varying lat/lon; ``h_fn(i)`` sets height."""
    return [
        _row(
            1000.0 + i,
            h=(h_fn(i) if h_fn is not None else H0),
            lat=LAT0 + 1e-5 * i,
            lon=LON0 + 2e-5 * i,
        )
        for i in range(n)
    ]


def _noisy_heights(n=200, amp=1.0, seed=42):
    rng = random.Random(seed)
    return [H0 + rng.uniform(-amp, amp) for _ in range(n)]


# ---------------------------------------------------------------------------
# Backward compatibility: default call == historical column set.
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


def test_default_columns_unchanged(tmp_path):
    """Guard: no new args -> the exact pre-chooser column set and order."""
    rows = _drive(30)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False)
    _, cols = _read(res.csv_path)
    assert cols == LEGACY_COLS
    assert res.n_rows == 30


def test_unknown_coord_system_rejected(tmp_path):
    rows = _drive(10)
    with pytest.raises(ValueError, match="unknown coord system"):
        export_trajectory(rows, tmp_path / "t.csv",
                          robust_filter_enabled=False,
                          coord_systems=["mgrs"])


# ---------------------------------------------------------------------------
# Coordinate-system chooser.
# ---------------------------------------------------------------------------

def test_utm_columns_roundtrip_pyproj(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    rows = _drive(40)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False,
                            coord_systems=["geodetic", "utm"],
                            smooth_z=False)
    data, cols = _read(res.csv_path)
    assert "utm_easting_m" in cols
    assert "utm_northing_m" in cols
    assert "utm_zone" in cols
    # h_m shared between datum-based and grid blocks: emitted exactly once.
    assert cols.count("h_m") == 1

    # Expected zone from the path longitude (LON0=34.79 -> zone 36N).
    zone = int((LON0 + 180.0) // 6.0) + 1
    epsg = 32600 + zone  # northern hemisphere
    assert all(d["utm_zone"] == f"{zone}N" for d in data)
    # EPSG logged in the '#' header comment.
    raw = res.csv_path.read_text(encoding="utf-8")
    assert f"EPSG:{epsg}" in raw.splitlines()[0]

    xform = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    for d, r in zip(data, rows):
        e_ref, n_ref = xform.transform(r.lon_deg, r.lat_deg)
        assert abs(float(d["utm_easting_m"]) - e_ref) < 1e-3   # mm
        assert abs(float(d["utm_northing_m"]) - n_ref) < 1e-3  # mm


def test_utm_zone_from_first_fix_at_antimeridian(tmp_path):
    """A track straddling +/-180 deg must NOT pick its zone from the
    arithmetic mean longitude (which averages to ~0 deg -> zone ~31,
    hundreds of km of error). The zone comes from the FIRST valid fix
    (179.999 E -> zone 60N) and the logged EPSG matches the transform
    actually applied to every row, including the western-hemisphere ones.
    """
    pyproj = pytest.importorskip("pyproj")
    lons = [179.9990 + 5e-5 * i for i in range(20)]        # -> 179.99995
    lons += [-180.0 + 5e-5 * (i + 1) for i in range(20)]   # -179.99995 ->
    rows = [
        _row(1000.0 + i, lat=LAT0 + 1e-5 * i, lon=lons[i])
        for i in range(len(lons))
    ]
    # sanity: the arithmetic mean is near 0 deg (the broken selector's input)
    assert abs(sum(lons) / len(lons)) < 1.0

    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False,
                            coord_systems=["geodetic", "utm"],
                            smooth_z=False)
    data, _ = _read(res.csv_path)

    # First-fix longitude 179.999 E, northern lat -> zone 60N / EPSG:32660.
    assert all(d["utm_zone"] == "60N" for d in data)
    raw = res.csv_path.read_text(encoding="utf-8")
    assert "EPSG:32660" in raw.splitlines()[0]

    # Logged EPSG is the transform actually used, for BOTH sides of the
    # antimeridian (mm agreement with a direct pyproj transform).
    xform = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:32660", always_xy=True)
    for d, r in zip(data, rows):
        e_ref, n_ref = xform.transform(r.lon_deg, r.lat_deg)
        assert abs(float(d["utm_easting_m"]) - e_ref) < 1e-3
        assert abs(float(d["utm_northing_m"]) - n_ref) < 1e-3


def test_enu_first_row_is_origin(tmp_path):
    rows = _drive(40)
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False,
                            coord_systems=["enu"])
    data, cols = _read(res.csv_path)
    assert {"e_m", "n_m", "u_m"} <= set(cols)
    # datum-based/cartesian XYZ not requested -> not emitted.
    assert "lat_deg" not in cols and "x_ecef_m" not in cols
    first = data[0]
    assert abs(float(first["e_m"])) < 1e-3
    assert abs(float(first["n_m"])) < 1e-3
    assert abs(float(first["u_m"])) < 1e-3
    # path moves away from the origin afterwards.
    last = data[-1]
    assert math.hypot(float(last["e_m"]), float(last["n_m"])) > 10.0


# ---------------------------------------------------------------------------
# Z (height) smoothing.
# ---------------------------------------------------------------------------

def test_z_smoothing_reduces_variance_preserves_mean(tmp_path):
    hs = _noisy_heights()
    rows = _drive(len(hs), h_fn=lambda i: hs[i])

    res_on = export_trajectory(rows, tmp_path / "on.csv",
                               robust_filter_enabled=False)  # smooth_z default ON
    res_off = export_trajectory(rows, tmp_path / "off.csv",
                                robust_filter_enabled=False, smooth_z=False)
    on, _ = _read(res_on.csv_path)
    off, _ = _read(res_off.csv_path)
    h_on = [float(d["h_m"]) for d in on]
    h_off = [float(d["h_m"]) for d in off]

    # smooth_z=False leaves heights untouched (up to CSV 4-decimal format).
    for h_csv, h_in in zip(h_off, hs):
        assert abs(h_csv - h_in) < 5e-5

    def _var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    # Noise crushed, mean preserved.
    assert _var(h_on) < 0.3 * _var(h_off)
    mean_on = sum(h_on) / len(h_on)
    mean_off = sum(h_off) / len(h_off)
    assert abs(mean_on - mean_off) < 0.1
    # Smoothing actually changed the series.
    assert any(abs(a - b) > 1e-3 for a, b in zip(h_on, h_off))


def test_ecef_z_reflects_smoothed_height(tmp_path):
    hs = _noisy_heights(n=120)
    rows = _drive(len(hs), h_fn=lambda i: hs[i])
    res = export_trajectory(rows, tmp_path / "t.csv",
                            robust_filter_enabled=False)  # smoothing default ON
    data, _ = _read(res.csv_path)

    n_changed = 0
    for d, r in zip(data, rows):
        h_csv = float(d["h_m"])
        # Cartesian XYZ columns must be derived from the SMOOTHED height...
        x_s, y_s, z_s = llh_to_ecef(r.lat_deg, r.lon_deg, h_csv)
        assert abs(float(d["x_ecef_m"]) - x_s) < 1e-3
        assert abs(float(d["y_ecef_m"]) - y_s) < 1e-3
        assert abs(float(d["z_ecef_m"]) - z_s) < 1e-3
        # ...not from the raw one (where they differ measurably).
        _, _, z_raw = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        if abs(h_csv - r.h_m) > 0.05:
            assert abs(float(d["z_ecef_m"]) - z_raw) > 1e-3
            n_changed += 1
    assert n_changed > 10  # smoothing visibly moved a good share of epochs
