"""Interop / back-compat tests for The external solver .pos ingestion.

Guards four safety features:

1. Solformat guard: llh (unchanged), cartesian XYZ (auto-converted), local-frame (rejected),
   unknown (assumed llh + WARN).
2. Header config readout (``parse_pos_header``).
3. Missing time-system token WARN (Reference time still assumed).
4. HARD back-compat: the two real The external solver-EX 2.5.0 files (new DAY14 + old
   DAY12) must parse to exactly the same rows as before these features.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from data_pipeline import geo
from data_pipeline.parsers import (
    detect_pos_solformat,
    parse_pos_header,
    parse_rtkpos,
)

# ---------------------------------------------------------------------------
# Real files (The external solver EX 2.5.0, llh solformat, "%  Reference time" header).
# ---------------------------------------------------------------------------

NEW_POS = Path("C:/Aj/gps/day14/solved_2026-06-28/dodge/20260628_190336_677/rover.pos")
OLD_POS = Path(
    "C:/Aj/gps/DAY12/dodge1/20260505_152247_472/"
    "measurements_20260505_152247_472_javad_base.pos"
)

# Golden values captured from the parser BEFORE the interop features landed.
# (row count, first-row utc_s, lat, lon, h)
_GOLDEN = {
    NEW_POS: (1412, 1782662619.001, 32.061461085, 34.794712512, 52.3442),
    OLD_POS: (2064, 1777983769.001, 32.069768517, 34.838621353, 60.6142),
}

needs_new = pytest.mark.skipif(not NEW_POS.is_file(), reason=f"{NEW_POS} not available")
needs_old = pytest.mark.skipif(not OLD_POS.is_file(), reason=f"{OLD_POS} not available")


@needs_new
def test_backcompat_new_real_pos() -> None:
    n, utc0, lat0, lon0, h0 = _GOLDEN[NEW_POS]
    rows = parse_rtkpos(NEW_POS)
    assert len(rows) == n
    r0 = rows[0]
    assert r0.utc_s == utc0
    assert r0.lat_deg == lat0
    assert r0.lon_deg == lon0
    assert r0.h_m == h0


@needs_old
def test_backcompat_old_real_pos() -> None:
    n, utc0, lat0, lon0, h0 = _GOLDEN[OLD_POS]
    rows = parse_rtkpos(OLD_POS)
    assert len(rows) == n
    r0 = rows[0]
    assert r0.utc_s == utc0
    assert r0.lat_deg == lat0
    assert r0.lon_deg == lon0
    assert r0.h_m == h0


@needs_new
def test_real_pos_no_warns_on_common_case(caplog: pytest.LogCaptureFixture) -> None:
    """The common case (llh + Reference time token) must not trip any WARN path."""
    with caplog.at_level(logging.WARNING, logger="data_pipeline.parsers"):
        parse_rtkpos(NEW_POS)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@needs_new
def test_detect_solformat_llh_new() -> None:
    assert detect_pos_solformat(NEW_POS) == "llh"


@needs_old
def test_detect_solformat_llh_old() -> None:
    assert detect_pos_solformat(OLD_POS) == "llh"


@needs_new
def test_parse_pos_header_new() -> None:
    hdr = parse_pos_header(NEW_POS)
    assert hdr.pos_mode == "Kinematic"
    assert hdr.elev_mask_deg == 10.0
    assert hdr.time_system == "GPST"
    assert hdr.solformat == "llh"
    assert hdr.program is not None and "RTKLIB" in hdr.program
    assert hdr.val_thres == 3.0
    assert hdr.ref_pos is not None and len(hdr.ref_pos) == 3
    assert hdr.obs_start is not None and hdr.obs_start.startswith("2026/06/28")
    # summary_lines is log-ready: one line per field, no exceptions.
    lines = hdr.summary_lines()
    assert any("Kinematic" in ln for ln in lines)


@needs_old
def test_parse_pos_header_old() -> None:
    hdr = parse_pos_header(OLD_POS)
    assert hdr.pos_mode == "Kinematic"
    assert hdr.elev_mask_deg == 5.0
    assert hdr.time_system == "GPST"
    assert hdr.amb_res == "Continuous"


def test_parse_pos_header_minimal(tmp_path: Path) -> None:
    """A header-less file yields all-None fields, never raises."""
    p = tmp_path / "min.pos"
    p.write_text(
        "2026/01/14 21:17:29.000   32.064101442   34.800189222"
        "    94.8836   2   4\n",
        encoding="utf-8",
    )
    hdr = parse_pos_header(p)
    assert hdr.pos_mode is None
    assert hdr.elev_mask_deg is None
    assert hdr.ref_pos is None
    assert hdr.time_system == "GPST"
    assert len(hdr.summary_lines()) == 12


# ---------------------------------------------------------------------------
# Cartesian XYZ solformat: auto-convert to lat/lon/h.
# ---------------------------------------------------------------------------

_ECEF_HEADER = (
    "% program   : RTKPOST-EX 2.5.0\n"
    "% (x/y/z-ecef=WGS84,Q=1:fix,2:float,5:single,ns=# of sats)\n"
    "%  GPST                      x-ecef(m)      y-ecef(m)      z-ecef(m)"
    "   Q  ns   sdx(m)   sdy(m)   sdz(m)\n"
)


def test_ecef_pos_roundtrip(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lat, lon, h = 32.065201145, 34.799470583, 74.1076
    x, y, z = geo.llh_to_ecef(lat, lon, h)
    p = tmp_path / "ecef.pos"
    p.write_text(
        _ECEF_HEADER
        + f"2026/01/14 21:17:29.000 {x:14.4f} {y:14.4f} {z:14.4f}"
          "   1   8   0.0100   0.0100   0.0100\n",
        encoding="utf-8",
    )
    assert detect_pos_solformat(p) == "ecef"
    with caplog.at_level(logging.INFO, logger="data_pipeline.parsers"):
        rows = parse_rtkpos(p)
    assert len(rows) == 1
    r = rows[0]
    assert abs(r.lat_deg - lat) < 1e-6
    assert abs(r.lon_deg - lon) < 1e-6
    assert abs(r.h_m - h) < 1e-3  # 1 mm
    assert r.quality == 1
    assert r.ns == 8
    assert any("ECEF solformat" in rec.getMessage() for rec in caplog.records)


def test_geo_ecef_to_llh_roundtrip_exact() -> None:
    """Pure geo round-trip (no file I/O): <1e-9 deg / <1e-6 m."""
    for lat, lon, h in [(32.0652, 34.7995, 74.1), (-45.3, 170.2, 1200.0), (0.0, 0.0, 0.0)]:
        x, y, z = geo.llh_to_ecef(lat, lon, h)
        lat2, lon2, h2 = geo.ecef_to_llh(x, y, z)
        assert abs(lat2 - lat) < 1e-9
        assert abs(lon2 - lon) < 1e-9
        assert abs(h2 - h) < 1e-6


# ---------------------------------------------------------------------------
# Local-frame / baseline solformat: rejected with a helpful message.
# ---------------------------------------------------------------------------

_ENU_HEADER = (
    "% program   : RTKPOST-EX 2.5.0\n"
    "%  GPST                  e-baseline(m) n-baseline(m) u-baseline(m)"
    "   Q  ns   sde(m)   sdn(m)   sdu(m)\n"
)


def test_enu_pos_raises(tmp_path: Path) -> None:
    p = tmp_path / "enu.pos"
    p.write_text(
        _ENU_HEADER
        + "2026/01/14 21:17:29.000       -12.3456        45.6789"
          "         1.2345   1   8   0.0100   0.0100   0.0100\n",
        encoding="utf-8",
    )
    assert detect_pos_solformat(p) == "enu"
    with pytest.raises(ValueError) as ei:
        parse_rtkpos(p)
    msg = str(ei.value)
    assert "baseline" in msg
    assert "llh" in msg  # tells the client which solformat to export


# ---------------------------------------------------------------------------
# Missing time-system token: still parses (Reference time assumed) but warns.
# ---------------------------------------------------------------------------


def test_no_time_token_warns_but_parses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "notoken.pos"
    # Column header names llh columns but carries no Reference time/UTC token.
    p.write_text(
        "% program   : SomeOtherTool 1.0\n"
        "%   latitude(deg) longitude(deg)  height(m)   Q  ns\n"
        "2026/01/14 21:17:29.000   32.064101442   34.800189222"
        "    94.8836   2   4\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="data_pipeline.parsers"):
        rows = parse_rtkpos(p)
    assert len(rows) == 1  # still parses, Reference time assumed
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("assuming GPST" in m for m in warns)
    assert any("~18 s off" in m for m in warns)


def test_unknown_solformat_warns_and_assumes_llh(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "weird.pos"
    p.write_text(
        "% program   : SomeOtherTool 1.0\n"
        "%  GPST   phi(deg) lam(deg) hgt(m)   Q  ns\n"
        "2026/01/14 21:17:29.000   32.064101442   34.800189222"
        "    94.8836   2   4\n",
        encoding="utf-8",
    )
    assert detect_pos_solformat(p) == "unknown"
    with caplog.at_level(logging.WARNING, logger="data_pipeline.parsers"):
        rows = parse_rtkpos(p)
    assert len(rows) == 1
    assert rows[0].lat_deg == 32.064101442  # llh assumed
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("unrecognized" in m and "phi(deg)" in m for m in warns)


def test_gpst_default_value_unchanged(tmp_path: Path) -> None:
    """The Reference time assumption itself must not move: 21:17:29 Reference time -> 21:17:11 UTC."""
    import datetime as dt

    p = tmp_path / "notoken.pos"
    p.write_text(
        "%   latitude(deg) longitude(deg)  height(m)   Q  ns\n"
        "2026/01/14 21:17:29.000   32.064101442   34.800189222"
        "    94.8836   2   4\n",
        encoding="utf-8",
    )
    rows = parse_rtkpos(p)
    hms = dt.datetime.fromtimestamp(
        rows[0].utc_s, tz=dt.timezone.utc
    ).strftime("%H:%M:%S")
    assert hms == "21:17:11"
