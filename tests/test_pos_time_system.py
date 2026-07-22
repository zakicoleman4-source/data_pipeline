"""Regression tests for .pos time-system handling in parse_rtkpos.

The external solver / The external solver-EX label the time system in the column-header line
(``%  Reference time ...`` or ``%  UTC ...``). The parser must:

* Reference time (default, every observed The external solver-EX 2.5.0 file): subtract epoch offset
  exactly once to reach UTC.
* UTC-labelled: NOT subtract epoch offset (the data is already UTC).
  A double-subtraction would shift every epoch by ~18 s (~180 m of along-track
  error at highway speed).
* No recognisable header token: assume Reference time (historical behaviour) so
  existing files parse identically.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from data_pipeline.parsers import parse_rtkpos, _detect_pos_time_system


_GPST_HEADER = (
    "% program   : RTKPOST-EX 2.5.0\n"
    "% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,5:single,ns=# of sats)\n"
    "%  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns"
    "   sdn(m)   sde(m)   sdu(m)\n"
)
_UTC_HEADER = (
    "% program   : RTKPOST-EX 2.5.0\n"
    "% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,5:single,ns=# of sats)\n"
    "%  UTC                   latitude(deg) longitude(deg)  height(m)   Q  ns"
    "   sdn(m)   sde(m)   sdu(m)\n"
)
_NO_HEADER = "% program   : RTKPOST-EX 2.5.0\n"

# 2026/01/14 21:17:29.000 — well after the 2017 epoch offset insertion (18 s).
_ROW = (
    "2026/01/14 21:17:29.000   32.064101442   34.800189222    94.8836"
    "   2   4   0.0880   0.1341   0.3388\n"
)
_LEAP_2026 = 18


def _utc_hms(utc_s: float) -> str:
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).strftime("%H:%M:%S")


def test_detect_gpst(tmp_path: Path) -> None:
    p = tmp_path / "g.pos"
    p.write_text(_GPST_HEADER + _ROW, encoding="utf-8")
    assert _detect_pos_time_system(p) == "GPST"


def test_detect_utc(tmp_path: Path) -> None:
    p = tmp_path / "u.pos"
    p.write_text(_UTC_HEADER + _ROW, encoding="utf-8")
    assert _detect_pos_time_system(p) == "UTC"


def test_detect_missing_header_defaults_gpst(tmp_path: Path) -> None:
    p = tmp_path / "n.pos"
    p.write_text(_NO_HEADER + _ROW, encoding="utf-8")
    assert _detect_pos_time_system(p) == "GPST"


def test_gpst_subtracts_leap_once(tmp_path: Path) -> None:
    """Reference time 21:17:29 -> UTC 21:17:11 (exactly one 18 s subtraction)."""
    p = tmp_path / "g.pos"
    p.write_text(_GPST_HEADER + _ROW, encoding="utf-8")
    rows = parse_rtkpos(p)
    assert len(rows) == 1
    assert _utc_hms(rows[0].utc_s) == "21:17:11"


def test_utc_not_shifted(tmp_path: Path) -> None:
    """UTC-labelled 21:17:29 stays 21:17:29 (no double-subtraction)."""
    p = tmp_path / "u.pos"
    p.write_text(_UTC_HEADER + _ROW, encoding="utf-8")
    rows = parse_rtkpos(p)
    assert len(rows) == 1
    assert _utc_hms(rows[0].utc_s) == "21:17:29"


def test_gpst_vs_utc_differ_by_leap(tmp_path: Path) -> None:
    g = tmp_path / "g.pos"
    u = tmp_path / "u.pos"
    g.write_text(_GPST_HEADER + _ROW, encoding="utf-8")
    u.write_text(_UTC_HEADER + _ROW, encoding="utf-8")
    gr = parse_rtkpos(g)[0]
    ur = parse_rtkpos(u)[0]
    # UTC file is "later" in absolute UTC by exactly the epoch offset count
    # because the Reference time file got 18 s subtracted and the UTC file did not.
    assert round(ur.utc_s - gr.utc_s) == _LEAP_2026


def test_explicit_leap_ignored_for_utc(tmp_path: Path) -> None:
    """An explicit leap_seconds must NOT be applied to a UTC-labelled file."""
    u = tmp_path / "u.pos"
    u.write_text(_UTC_HEADER + _ROW, encoding="utf-8")
    rows = parse_rtkpos(u, leap_seconds=18)
    assert _utc_hms(rows[0].utc_s) == "21:17:29"
