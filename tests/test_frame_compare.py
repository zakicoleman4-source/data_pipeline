"""Tests for data_pipeline.frame_compare.

Covers: column auto-detection for datum-based / projected-Grid / Cartesian XYZ external
inputs, local-Local-frame delta math against a known injected offset, systematic-bias
vs scattered classification, and extension-insensitive image matching.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from data_pipeline.frame_compare import (
    CompareResult,
    _utm_zone_to_epsg,
    compute_deltas,
    load_external_frame_coords,
    load_gnss_frame_coords_from_georef,
    normalize_image_key,
    warn_duplicate_stems,
    write_delta_csv,
)
from data_pipeline.geo import enu_to_llh, llh_to_ecef

# A plausible mid-latitude reference track.
LAT0, LON0, H0 = 47.397_5, 8.545_2, 432.7


def _track(n: int = 12) -> dict[str, tuple[float, float, float]]:
    """Small synthetic sample track: ~2 m spacing heading north-east."""
    out: dict[str, tuple[float, float, float]] = {}
    for i in range(n):
        lat, lon, h = enu_to_llh(1.4 * i, 1.6 * i, 0.05 * i, (LAT0, LON0, H0))
        out[f"frame_{i:06d}"] = (lat, lon, h)
    return out


# -----------------------------
# Column auto-detection
# -----------------------------


def test_load_geodetic_flexible_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "ext.csv"
    csv_path.write_text(
        "# refined per-frame coordinates\n"
        "LABEL,Lat,LON,Alt\n"
        "frame_000001,47.5,8.5,430.25\n"
        "frame_000002,47.500001,8.500001,431.0\n",
        encoding="utf-8",
    )
    got = load_external_frame_coords(csv_path)
    assert set(got) == {"frame_000001", "frame_000002"}
    lat, lon, h = got["frame_000001"]
    assert lat == pytest.approx(47.5)
    assert lon == pytest.approx(8.5)
    assert h == pytest.approx(430.25)


def test_load_geodetic_without_altitude(tmp_path: Path) -> None:
    csv_path = tmp_path / "ext.csv"
    csv_path.write_text(
        "name,latitude,longitude\nimg_01,47.4,8.6\n", encoding="utf-8"
    )
    got = load_external_frame_coords(csv_path)
    assert got["img_01"] == (pytest.approx(47.4), pytest.approx(8.6), None)


def test_load_projected_utm_round_trip(tmp_path: Path) -> None:
    pyproj = pytest.importorskip("pyproj")
    lat, lon, h = LAT0, LON0, H0  # zone 32N
    fwd = pyproj.Transformer.from_crs(4326, 32632, always_xy=True)
    easting, northing = fwd.transform(lon, lat)

    csv_path = tmp_path / "ext_utm.csv"
    csv_path.write_text(
        "Frame,Easting,Northing,Height,Zone\n"
        f"frame_000003.png,{easting:.4f},{northing:.4f},{h:.4f},32U\n",
        encoding="utf-8",
    )
    got = load_external_frame_coords(csv_path)
    glat, glon, gh = got["frame_000003"]
    # ~1e-7 deg is about 1 cm on the ground.
    assert glat == pytest.approx(lat, abs=1e-7)
    assert glon == pytest.approx(lon, abs=1e-7)
    assert gh == pytest.approx(h, abs=0.01)


def test_load_projected_epsg_cli_override(tmp_path: Path) -> None:
    pyproj = pytest.importorskip("pyproj")
    fwd = pyproj.Transformer.from_crs(4326, 32632, always_xy=True)
    easting, northing = fwd.transform(LON0, LAT0)
    csv_path = tmp_path / "ext_xy.csv"
    csv_path.write_text(
        f"image,x,y\nframe_000004,{easting:.4f},{northing:.4f}\n",
        encoding="utf-8",
    )
    # No zone/epsg column: must fail without an override...
    with pytest.raises(ValueError):
        load_external_frame_coords(csv_path)
    # ...and succeed with one.
    got = load_external_frame_coords(csv_path, epsg=32632)
    glat, glon, gh = got["frame_000004"]
    assert glat == pytest.approx(LAT0, abs=1e-7)
    assert glon == pytest.approx(LON0, abs=1e-7)
    assert gh is None


def test_load_ecef_round_trip(tmp_path: Path) -> None:
    lat, lon, h = LAT0, LON0, H0
    x, y, z = llh_to_ecef(lat, lon, h)
    csv_path = tmp_path / "ext_ecef.csv"
    csv_path.write_text(
        "image,X_ECEF,Y_ECEF,Z_ECEF\n"
        f"frame_000005,{x:.4f},{y:.4f},{z:.4f}\n",
        encoding="utf-8",
    )
    got = load_external_frame_coords(csv_path)
    glat, glon, gh = got["frame_000005"]
    assert glat == pytest.approx(lat, abs=1e-8)
    assert glon == pytest.approx(lon, abs=1e-8)
    assert gh == pytest.approx(h, abs=0.005)


def test_load_georef_csv_with_comment_header(tmp_path: Path) -> None:
    georef = tmp_path / "Georef.csv"
    georef.write_text(
        "# reference CSV (data_pipeline). Image=camera label\n"
        "Image,Latitude,Longitude,Altitude,AccuracyX,AccuracyY,AccuracyZ\n"
        "frame_000001,47.500000000,8.500000000,430.2500,0.05,0.05,0.10\n"
        "frame_000002,47.500001000,8.500001000,430.3000,0.05,0.05,0.10\n",
        encoding="utf-8",
    )
    got = load_gnss_frame_coords_from_georef(georef)
    assert set(got) == {"frame_000001", "frame_000002"}
    assert got["frame_000001"][0] == pytest.approx(47.5)
    assert got["frame_000001"][2] == pytest.approx(430.25)


# -----------------------------
# Delta math
# -----------------------------


def test_known_east_offset_reports_systematic_bias() -> None:
    gnss = _track(10)
    # Shift every external point exactly 1.5 m east of the Signal point.
    external = {
        k: enu_to_llh(1.5, 0.0, 0.0, llh) for k, llh in gnss.items()
    }
    result = compute_deltas(external, gnss)
    assert isinstance(result, CompareResult)
    s = result.summary
    assert s["n_matched"] == 10
    assert s["mean_east_m"] == pytest.approx(1.5, abs=1e-3)
    assert s["mean_north_m"] == pytest.approx(0.0, abs=1e-3)
    assert s["mean_horiz_m"] == pytest.approx(1.5, abs=1e-3)
    assert s["bearing_deg"] == pytest.approx(90.0, abs=0.1)
    assert s["classification"] == "systematic_bias"
    # Rigid shift: scatter about the mean is ~zero.
    assert s["std_horiz_m"] < 0.01
    # Per-sample records carry the offset too.
    for r in result.records:
        assert r.d_east_m == pytest.approx(1.5, abs=1e-3)
        assert r.d_horiz_m == pytest.approx(1.5, abs=1e-3)


def test_zero_mean_scatter_reports_scattered() -> None:
    gnss = _track(12)
    # Deterministic zero-mean noise: alternate +/- in east and north so the
    # mean offset vector is exactly zero while per-sample deltas are ~0.5 m.
    external: dict[str, tuple[float, float, float]] = {}
    for i, (k, llh) in enumerate(sorted(gnss.items())):
        de = 0.5 if i % 2 == 0 else -0.5
        dn = 0.4 if i % 4 < 2 else -0.4
        external[k] = enu_to_llh(de, dn, 0.0, llh)
    result = compute_deltas(external, gnss)
    s = result.summary
    assert s["n_matched"] == 12
    assert abs(s["mean_east_m"]) < 0.01
    assert abs(s["mean_north_m"]) < 0.01
    assert s["std_horiz_m"] == pytest.approx(math.hypot(0.5, 0.4), abs=0.01)
    assert s["classification"] == "scattered"


def test_vertical_stats_present_only_with_heights_on_both_sides() -> None:
    gnss = _track(6)
    external = {
        k: enu_to_llh(0.0, 0.0, 0.30, llh) for k, llh in gnss.items()
    }
    s = compute_deltas(external, gnss).summary
    assert s["n_vert"] == 6
    assert s["mean_up_m"] == pytest.approx(0.30, abs=1e-3)

    # Drop the external heights: no vertical stats, horizontal unaffected.
    external_no_h = {k: (v[0], v[1], None) for k, v in external.items()}
    s2 = compute_deltas(external_no_h, gnss).summary
    assert s2["n_vert"] == 0
    assert "mean_up_m" not in s2
    assert s2["mean_horiz_m"] < 0.01


# -----------------------------
# Image key matching
# -----------------------------


def test_image_matching_is_extension_insensitive(tmp_path: Path) -> None:
    assert normalize_image_key("frame_000001.png") == "frame_000001"
    assert normalize_image_key("frame_000001") == "frame_000001"
    assert normalize_image_key("sub/dir/frame_000002.JPG") == "frame_000002"

    # End-to-end: external keeps .png extensions, Coordinate output has bare stems.
    gnss = dict(list(_track(3).items()))
    csv_path = tmp_path / "ext.csv"
    lines = ["image,latitude,longitude,altitude"]
    for k, (lat, lon, h) in gnss.items():
        lines.append(f"{k}.png,{lat:.9f},{lon:.9f},{h:.4f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    external = load_external_frame_coords(csv_path)
    assert set(external) == set(gnss)  # stems matched despite .png
    s = compute_deltas(external, gnss).summary
    assert s["n_matched"] == 3
    assert s["mean_horiz_m"] < 1e-3


# -----------------------------
# Duplicate-stem collisions must be visible (last wins, but warned)
# -----------------------------


def test_external_duplicate_stems_warn_and_last_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # a/frame_1.png and b/frame_1.jpg normalise to the same stem "frame_1".
    csv_path = tmp_path / "ext.csv"
    csv_path.write_text(
        "image,latitude,longitude\n"
        "a/frame_1.png,47.5,8.5\n"
        "b/frame_1.jpg,47.6,8.6\n"
        "frame_2.png,47.7,8.7\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        got = load_external_frame_coords(csv_path)
    assert set(got) == {"frame_1", "frame_2"}
    # last row wins (documented behaviour) ...
    assert got["frame_1"][0] == pytest.approx(47.6)
    # ... but the collision is loudly reported with a count.
    warnings = [r for r in caplog.records if "collided" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "1 row(s)" in msg
    assert "frame_1" in msg


def test_external_no_duplicates_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "ext.csv"
    csv_path.write_text(
        "image,latitude,longitude\nframe_1,47.5,8.5\nframe_2,47.6,8.6\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        load_external_frame_coords(csv_path)
    assert not [r for r in caplog.records if "collided" in r.getMessage()]


def test_georef_duplicate_stems_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    georef = tmp_path / "Georef.csv"
    georef.write_text(
        "Image,Latitude,Longitude,Altitude\n"
        "frame_000001,47.500000000,8.500000000,430.25\n"
        "frame_000001,47.500001000,8.500001000,430.30\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        got = load_gnss_frame_coords_from_georef(georef)
    assert len(got) == 1
    assert got["frame_000001"][2] == pytest.approx(430.30)  # last wins
    warnings = [r for r in caplog.records if "collided" in r.getMessage()]
    assert len(warnings) == 1
    assert "frame_000001" in warnings[0].getMessage()


def test_warn_duplicate_stems_returns_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        assert warn_duplicate_stems("src.csv", []) == 0
        assert warn_duplicate_stems("src.csv", ["f1", "f1", "f2"]) == 3
    warnings = [r for r in caplog.records if "collided" in r.getMessage()]
    assert len(warnings) == 1  # silent when no collisions


# -----------------------------
# Grid zone letter: MGRS band vs hemisphere ambiguity
# -----------------------------


def test_utm_zone_bare_s_resolves_north_but_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        epsg = _utm_zone_to_epsg("32S")
    # Default mapping kept: MGRS band S (>= 'N') is the NORTHERN hemisphere.
    assert epsg == 32632
    warnings = [r for r in caplog.records if "AMBIGUOUS" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    # The warning documents the interpretation and the southern alternative.
    assert "32632" in msg
    assert "32732" in msg


def test_utm_zone_bare_n_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        assert _utm_zone_to_epsg("32N") == 32632
    assert [r for r in caplog.records if "ambiguous" in r.getMessage().lower()]


def test_utm_zone_unambiguous_bands_stay_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="data_pipeline.frame_compare"):
        assert _utm_zone_to_epsg("32U") == 32632   # north band
        assert _utm_zone_to_epsg("19K") == 32719   # south band
    assert not caplog.records


# -----------------------------
# Output CSV
# -----------------------------


def test_write_delta_csv_columns_and_blanks(tmp_path: Path) -> None:
    gnss = _track(4)
    external = {k: enu_to_llh(1.0, 0.0, 0.0, llh) for k, llh in gnss.items()}
    # One sample without an external height -> blank d_up_m / d_vert_m cells.
    first = sorted(external)[0]
    external[first] = (external[first][0], external[first][1], None)

    result = compute_deltas(external, gnss)
    out = write_delta_csv(result.records, tmp_path / "sub" / "frame_delta.csv")
    assert out.is_file()

    text = out.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    assert header == [
        "Image", "ext_lat", "ext_lon", "ext_h",
        "gnss_lat", "gnss_lon", "gnss_h",
        "d_east_m", "d_north_m", "d_up_m", "d_horiz_m", "d_vert_m",
    ]
    assert len(text) == 1 + 4
    first_row = text[1].split(",")
    assert first_row[0] == first
    assert first_row[3] == ""   # ext_h blank
    assert first_row[9] == ""   # d_up_m blank
    assert first_row[11] == ""  # d_vert_m blank
    assert float(first_row[7]) == pytest.approx(1.0, abs=1e-3)  # d_east_m
