"""TIME-basis chooser for the user path export (2026-07-05).

* Backward compatibility: the default ``export_trajectory`` call (no
  ``time_bases``) must emit exactly the historical column set with the
  single ``reference time`` time column (header guard).
* ``time_bases=("reference time","utc","stream","iso")`` emits the four TIME columns
  FIRST, in request order: ``reference time`` (= utc_s + leap), ``utc_s`` (row
  UTC, exact), ``t_audio_s`` (= utc_s - audio_start_utc_s, exact) and
  ``utc_iso`` (ISO-8601 UTC that parses back to the same instant).
* 'stream' without ``audio_start_utc_s`` -> ValueError.
* unknown basis -> ValueError.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from data_pipeline.parsers import PosRow
from data_pipeline.stages.user_export import export_trajectory
from data_pipeline.time_sync import get_leap_seconds_for_epoch


# A modern epoch (2025-06-19T12:26:40Z) so epoch offset are non-zero and the
# ISO string is meaningful; chosen ms-exact in binary so CSV %.6f round-trips.
UTC0 = 1_750_336_000.0
AUDIO_START_UTC_S = UTC0 - 12.5  # stream sample 0 rode the UTC clock 12.5 s earlier

LAT0, LON0, H0 = 32.06, 34.79, 55.0


def _read(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    return list(rdr), rdr.fieldnames


def _row(utc, lat=LAT0, lon=LON0, sd_xy=0.3):
    """Position + position sigma present (smoothed-output shape)."""
    return PosRow(
        utc_s=utc, lat_deg=lat, lon_deg=lon, h_m=H0, quality=2,
        ns=20, vn=1.0, ve=0.5, vu=0.0,
        sd_n=sd_xy, sd_e=sd_xy, sd_u=sd_xy * 2,
        sd_vn=0.05, sd_ve=0.05, sd_vu=0.1,
    )


def _drive(n=30):
    """1 Hz drive with slightly varying lat/lon."""
    return [
        _row(UTC0 + i, lat=LAT0 + 1e-5 * i, lon=LON0 + 2e-5 * i)
        for i in range(n)
    ]


def _export(tmp_path, rows, name="t.csv", **kw):
    return export_trajectory(rows, tmp_path / name,
                             robust_filter_enabled=False, smooth_z=False,
                             **kw)


# ---------------------------------------------------------------------------
# Backward compatibility: default call == historical header, single reference time.
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


def test_default_header_is_legacy_single_gpstime(tmp_path):
    """Guard: no ``time_bases`` arg -> exactly the legacy header (single
    ``reference time`` time column, byte-for-byte header line)."""
    rows = _drive(20)
    res = _export(tmp_path, rows)
    data, cols = _read(res.csv_path)
    assert cols == LEGACY_COLS
    # Header line literally identical to the legacy one.
    first_line = res.csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == ",".join(LEGACY_COLS)
    # No other time column sneaked in.
    assert "utc_s" not in cols and "t_audio_s" not in cols and "utc_iso" not in cols
    assert res.n_rows == 20
    assert tuple(res.time_bases) == ("gpst",)
    assert res.audio_start_utc_s is None


def test_default_matches_explicit_gpst(tmp_path):
    """``time_bases=("reference time",)`` reproduces the default export byte-for-byte."""
    rows = _drive(20)
    res_default = _export(tmp_path, rows, name="d.csv")
    res_gpst = _export(tmp_path, rows, name="g.csv", time_bases=("gpst",))
    assert (res_default.csv_path.read_bytes()
            == res_gpst.csv_path.read_bytes())


# ---------------------------------------------------------------------------
# Full chooser: order + values.
# ---------------------------------------------------------------------------

def test_all_bases_order_and_values(tmp_path):
    rows = _drive(30)
    res = _export(tmp_path, rows,
                  time_bases=("gpst", "utc", "audio", "iso"),
                  audio_start_utc_s=AUDIO_START_UTC_S)
    data, cols = _read(res.csv_path)

    # The four time columns come FIRST, in the requested order, then the
    # (default datum-based+cartesian XYZ) coordinate blocks.
    assert cols[:4] == ["gpstime", "utc_s", "t_audio_s", "utc_iso"]
    assert cols[4:] == LEGACY_COLS[1:]

    assert len(data) == len(rows)
    for d, r in zip(data, rows):
        # utc_s == the row's UTC (exact at these ms-exact test epochs).
        assert float(d["utc_s"]) == r.utc_s
        # t_audio_s == utc - audio_start (exact).
        assert float(d["t_audio_s"]) == r.utc_s - AUDIO_START_UTC_S
        # reference time == utc + leap.
        leap = get_leap_seconds_for_epoch(r.utc_s)
        assert float(d["gpstime"]) == r.utc_s + leap
        # utc_iso parses back to the same UTC instant (ms precision).
        t = dt.datetime.strptime(
            d["utc_iso"], "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=dt.timezone.utc)
        assert abs(t.timestamp() - r.utc_s) < 5e-4

    # Result records what was emitted.
    assert tuple(res.time_bases) == ("gpst", "utc", "audio", "iso")
    assert res.audio_start_utc_s == AUDIO_START_UTC_S


def test_request_order_respected_and_deduped(tmp_path):
    rows = _drive(10)
    res = _export(tmp_path, rows,
                  time_bases=("audio", "gpst", "audio"),
                  audio_start_utc_s=AUDIO_START_UTC_S)
    _, cols = _read(res.csv_path)
    assert cols[:2] == ["t_audio_s", "gpstime"]
    assert cols.count("t_audio_s") == 1
    assert tuple(res.time_bases) == ("audio", "gpst")


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------

def test_audio_without_anchor_raises(tmp_path):
    rows = _drive(10)
    with pytest.raises(ValueError, match="no audio anchor"):
        _export(tmp_path, rows, time_bases=("gpst", "audio"))


def test_unknown_basis_raises(tmp_path):
    rows = _drive(10)
    with pytest.raises(ValueError, match="unknown time basis"):
        _export(tmp_path, rows, time_bases=("gpst", "tai"))


def test_empty_bases_raises(tmp_path):
    rows = _drive(10)
    with pytest.raises(ValueError, match="time_bases is empty"):
        _export(tmp_path, rows, time_bases=())
