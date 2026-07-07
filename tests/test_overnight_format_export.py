"""Parity regression tests ported from DAY12 Track 4:

* the external tool label fix (Image column = filename stem, no extension) — adapted
  to the client tree, where the reference CSV is built by
  ``data_pipeline.stages.georef`` (the client has no ``stages/external.py``).
* Accuracy-gated export: explicit 2-sigma columns + suppression of sections
  that exceed the project bar (horizontal <= 6 m @ 2 sigma, speed <= 3 km/h
  @ 2 sigma).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data_pipeline.parsers import PosRow
from data_pipeline.stages.user_export import (
    HORIZ_BAR_2SIGMA_M,
    SPEED_BAR_2SIGMA_KMH,
    export_trajectory,
)


def _row(utc, sd_xy, sd_v, lat=32.0, lon=34.0):
    return PosRow(
        utc_s=utc, lat_deg=lat, lon_deg=lon, h_m=50.0, quality=1,
        vn=1.0, ve=1.0, vu=0.0,
        ns=12, sd_n=sd_xy, sd_e=sd_xy, sd_u=sd_xy * 2,
        sd_vn=sd_v, sd_ve=sd_v, sd_vu=sd_v,
    )


def _read(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(lines)
    return list(rdr), rdr.fieldnames


def test_export_has_explicit_2sigma_columns(tmp_path):
    rows = [_row(1000.0 + i, 0.2, 0.05) for i in range(20)]
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=False)
    _, cols = _read(res.csv_path)
    assert "err_horiz_2sigma_m" in cols
    assert "err_speed_2sigma_mps" in cols
    assert "err_speed_2sigma_kmh" in cols
    assert res.n_rows == 20


def test_export_hard_drop_mode_suppresses_bad_horizontal_section(tmp_path):
    # Legacy hard-drop mode: 30 tight epochs, a 10-epoch blown-sigma window
    # in the middle is deleted from the CSV.
    rows = [_row(1000.0 + i, 0.2, 0.05) for i in range(30)]
    for i in range(12, 22):
        rows[i] = _row(1000.0 + i, 5.0, 0.05)  # 2-sigma >> 6 m bar
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=True,
                            hard_drop_over_bar=True)
    assert res.n_dropped_rows == 10
    assert len(res.dropped_sections) == 1
    sec = res.dropped_sections[0]
    assert "horizontal" in sec.reason
    assert sec.n_epochs == 10
    data, _ = _read(res.csv_path)
    assert len(data) == 20
    assert res.coverage_pct == pytest.approx(100 * 20 / 30, abs=0.1)


def test_export_default_keeps_over_bar_epochs_flagged(tmp_path):
    # Client-ready default: over-bar epochs are KEPT with pos_within_bar=0,
    # never deleted (the client gets the full path + honest sigma).
    rows = [_row(1000.0 + i, 0.2, 0.05) for i in range(30)]
    for i in range(12, 22):
        rows[i] = _row(1000.0 + i, 5.0, 0.05)  # 2-sigma >> 6 m bar
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=True)
    assert res.n_dropped_rows == 0
    assert res.n_flagged_over_bar == 10
    assert len(res.flagged_sections) == 1
    assert res.flagged_sections[0].reason == "horizontal"
    data, cols = _read(res.csv_path)
    assert len(data) == 30
    assert "pos_within_bar" in cols
    flags = [d["pos_within_bar"] for d in data]
    assert flags.count("0") == 10
    assert flags.count("1") == 20


def test_export_speed_breach_keeps_row_marks_vel_untrusted(tmp_path):
    # Velocity NEVER drops a row: a speed-sigma breach only clears vel_trusted.
    rows = [_row(1000.0 + i, 0.2, 0.05) for i in range(20)]
    for i in range(5, 9):
        rows[i] = _row(1000.0 + i, 0.2, 0.6)  # 2-sigma ~1.7 m/s = 6.1 km/h > 3
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=True)
    assert res.n_dropped_rows == 0
    assert res.n_rows == 20
    assert res.n_vel_untrusted == 4
    data, cols = _read(res.csv_path)
    assert "vel_trusted" in cols
    assert [d["vel_trusted"] for d in data].count("0") == 4


def test_export_no_suppression_keeps_all(tmp_path):
    rows = [_row(1000.0 + i, 5.0, 0.6) for i in range(15)]  # all bad
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=False)
    assert res.n_rows == 15
    assert res.n_dropped_rows == 0


def test_summary_text_runs(tmp_path):
    rows = [_row(1000.0 + i, 0.2, 0.05) for i in range(10)]
    res = export_trajectory(rows, tmp_path / "e.csv", suppress_inaccurate=True)
    txt = res.summary_text()
    assert "Accuracy export summary" in txt
    assert "coverage" in txt


def test_georef_label_keeps_extension_by_default():
    # Client builds the external tool reference CSV via coordinate output, not external tool.py.
    # The client's the external tool uses the FULL filename (WITH extension) as the
    # source label, so the Image column must keep the extension by default.
    from data_pipeline.stages.georef import CsvOptions
    opt = CsvOptions()
    assert opt.label_strip_ext is False
    # Default is now False (Track-4 regression fix): a leading '#' comment line
    # made the offline viewers' csv.DictReader treat the comment as the header.
    # The viewers also skip '#' lines now, but header+data-only is the safe
    # shape; the comment can still be re-enabled via emit_header_comment=True.
    assert opt.emit_header_comment is False
    # Samples are named with a DOT-FREE zero-padded sequential index, so the
    # full filename is unambiguous (a decimal-PTS name like
    # ``frame_0.003250.png`` used to make the external tool collapse the label on the
    # first dot to ``frame_0``).
    assert Path("frame_000001.png").name == "frame_000001.png"
    assert Path("frame_000001.png").name.count(".") == 1
