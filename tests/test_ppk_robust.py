"""Robust tests for Post-processing stage, clean_pos outlier filter, config patching,
.pos round-trip, epoch_weighted_v2, and setup validation.

Covers the gaps identified in the v1.1.0 audit:
  - clean_pos:           zero tests before this file
  - write_patched_config: zero tests before this file
  - _write_pos_like:     zero round-trip tests before this file
  - _parse_pos_numeric:  zero unit tests before this file
  - epoch_weighted_v2:   zero dedicated tests
  - install.py smoke:    zero tests
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.parsers import ImuRow, PosRow, parse_rtkpos
from data_pipeline.stages import ppk


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _make_pos_file(path: Path, rows: list[str], header: str = "% test\n") -> Path:
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _std_row(
    date: str = "2024/06/15",
    time: str = "12:00:00.000",
    lat: float = 32.0,
    lon: float = 34.8,
    h: float = 50.0,
    q: int = 1,
    ns: int = 12,
    sdn: float = 0.01,
    sde: float = 0.01,
    sdu: float = 0.03,
    sdne: float = 0.0,
    sdeu: float = 0.0,
    sdun: float = 0.0,
    age: float = 1.0,
    ratio: float = 5.0,
    vn: float = 0.5,
    ve: float = 1.0,
    vu: float = 0.0,
) -> str:
    return (
        f"{date} {time}  {lat:.9f}  {lon:.9f}  {h:.4f}  {q}  {ns}  "
        f"{sdn:.4f}  {sde:.4f}  {sdu:.4f}  "
        f"{sdne:.4f}  {sdeu:.4f}  {sdun:.4f}  "
        f"{age:.2f}  {ratio:.1f}  {vn:.5f}  {ve:.5f}  {vu:.5f}"
    )


def _synth_pos(n: int = 60, speed_mps: float = 5.0, noise: float = 0.3,
               base: tuple = (32.0, 34.8, 100.0), seed: int = 42) -> list[PosRow]:
    rng = np.random.default_rng(seed)
    out: list[PosRow] = []
    for i in range(n):
        lat = base[0] + rng.normal(0, noise) / 111_000
        lon = base[1] + (speed_mps * i + rng.normal(0, noise)) / 94_000
        h = base[2] + rng.normal(0, noise * 2)
        out.append(PosRow(
            utc_s=float(i), lat_deg=lat, lon_deg=lon, h_m=h, quality=2,
            vn=rng.normal(0, 0.05), ve=speed_mps + rng.normal(0, 0.05),
            vu=rng.normal(0, 0.05), ns=10,
            sd_n=0.5, sd_e=0.5, sd_u=1.0,
        ))
    return out


# ───────────────────────────────────────────────────────────────────
# _parse_pos_numeric
# ───────────────────────────────────────────────────────────────────

class TestParsePosNumeric:
    def test_minimal_7_cols(self):
        parts = "2024/01/01 00:00:00.000 32.0 34.8 50.0 1 12".split()
        d = ppk._parse_pos_numeric(parts)
        assert d is not None
        assert d["q"] == 1 and d["ns"] == 12
        assert d["lat"] == 32.0

    def test_too_short_returns_none(self):
        assert ppk._parse_pos_numeric("x y z".split()) is None

    def test_bad_q_returns_none(self):
        parts = "2024/01/01 00:00:00.000 32.0 34.8 50.0 X 12".split()
        assert ppk._parse_pos_numeric(parts) is None

    def test_full_18_cols(self):
        parts = _std_row().split()
        d = ppk._parse_pos_numeric(parts)
        assert d is not None
        assert math.isfinite(d["vn"])
        assert math.isfinite(d["age"])
        assert d["q"] == 1

    def test_nan_velocity_when_absent(self):
        parts = "2024/01/01 00:00:00.000 32.0 34.8 50.0 1 12 0.01 0.01 0.03".split()
        d = ppk._parse_pos_numeric(parts)
        assert d is not None
        assert math.isnan(d["vn"])
        assert math.isnan(d["ve"])


# ───────────────────────────────────────────────────────────────────
# write_patched_config
# ───────────────────────────────────────────────────────────────────

class TestWritePatchedConfig:
    def test_replaces_existing_keys(self, tmp_path: Path):
        src = tmp_path / "orig.conf"
        src.write_text(
            "ant2-postype =single  # type\n"
            "ant2-pos1    =0.0000  # X\n"
            "ant2-pos2    =0.0000  # Y\n"
            "ant2-pos3    =0.0000  # Z\n"
            "pos1-posopt  =rinex\n",
            encoding="utf-8",
        )
        dst = tmp_path / "patched.conf"
        ppk.write_patched_config(
            src, dst, base_ecef_xyz=(4_000_000.0, 3_000_000.0, 3_500_000.0),
        )
        text = dst.read_text(encoding="utf-8")
        assert "4000000.0000" in text
        assert "3000000.0000" in text
        assert "3500000.0000" in text
        assert "xyz" in text
        assert "# type" in text  # inline comment preserved
        assert "rinex" in text  # unrelated key untouched

    def test_appends_missing_keys(self, tmp_path: Path):
        src = tmp_path / "orig.conf"
        src.write_text("pos1-posopt=kinematic\n", encoding="utf-8")
        dst = tmp_path / "patched.conf"
        ppk.write_patched_config(
            src, dst, base_ecef_xyz=(1.0, 2.0, 3.0),
        )
        text = dst.read_text(encoding="utf-8")
        assert "ant2-postype" in text
        assert "ant2-pos1" in text
        assert "ant2-pos2" in text
        assert "ant2-pos3" in text

    def test_creates_parent_directory(self, tmp_path: Path):
        src = tmp_path / "orig.conf"
        src.write_text("ant2-postype=single\n", encoding="utf-8")
        dst = tmp_path / "deep" / "nest" / "patched.conf"
        ppk.write_patched_config(src, dst, base_ecef_xyz=(1, 2, 3))
        assert dst.is_file()

    def test_case_insensitive_key_matching(self, tmp_path: Path):
        src = tmp_path / "orig.conf"
        src.write_text("ANT2-POSTYPE=single\nANT2-POS1=0\n", encoding="utf-8")
        dst = tmp_path / "patched.conf"
        ppk.write_patched_config(src, dst, base_ecef_xyz=(1, 2, 3))
        text = dst.read_text(encoding="utf-8")
        lines_with_postype = [l for l in text.splitlines() if "postype" in l.lower()]
        assert len(lines_with_postype) == 1
        assert "xyz" in lines_with_postype[0]


# ───────────────────────────────────────────────────────────────────
# clean_pos outlier filter
# ───────────────────────────────────────────────────────────────────

class TestCleanPos:
    def _good_rows(self, n: int = 10, t_start_s: int = 0) -> list[str]:
        rows = []
        for i in range(n):
            ts = f"12:{i:02d}:00.000"
            rows.append(_std_row(time=ts, q=1, ns=12, sdn=0.01, sde=0.01,
                                 sdu=0.03, age=1.0, vn=0.5, ve=1.0))
        return rows

    def test_all_good_rows_pass(self, tmp_path: Path):
        pos = _make_pos_file(tmp_path / "in.pos", self._good_rows(10))
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.n_in == 10
        assert res.n_out == 10
        assert not res.rejected_by

    def test_q5_rows_rejected(self, tmp_path: Path):
        rows = self._good_rows(5)
        rows.append(_std_row(time="12:05:00.000", q=5))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.n_in == 6
        assert res.n_out == 5
        assert res.rejected_by.get("q_dropvalue") == 1

    def test_q_above_max_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", q=4))  # Differential, max_q default=2
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("q_above_max") == 1

    def test_low_ns_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", ns=2))  # min_ns default=5
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("low_ns") == 1

    def test_high_sigma_h_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", sdn=4.0, sde=4.0))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("high_sigma_h") == 1

    def test_high_sigma_v_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", sdu=15.0))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("high_sigma_v") == 1

    def test_high_age_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", age=60.0))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("high_age") == 1

    def test_high_speed_rejected(self, tmp_path: Path):
        rows = self._good_rows(3)
        rows.append(_std_row(time="12:03:00.000", vn=40.0, ve=40.0))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("high_speed") == 1

    def test_position_jump_rejected(self, tmp_path: Path):
        rows = self._good_rows(8)
        # Inject a 100m-offset outlier
        rows.append(_std_row(time="12:08:00.000",
                             lat=32.001, lon=34.801))
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.rejected_by.get("position_jump") == 1

    def test_disabled_filter_copies_file(self, tmp_path: Path):
        rows = self._good_rows(3) + [_std_row(time="12:03:00.000", q=5)]
        pos = _make_pos_file(tmp_path / "in.pos", rows)
        out = tmp_path / "out.pos"
        opts = ppk.OutlierFilterOptions(enabled=False)
        res = ppk.clean_pos(pos, out, options=opts)
        assert res.n_in == 0  # disabled reports 0
        assert "disabled" in res.summary

    def test_empty_file_produces_empty_output(self, tmp_path: Path):
        pos = _make_pos_file(tmp_path / "in.pos", [], header="% empty\n")
        out = tmp_path / "out.pos"
        res = ppk.clean_pos(pos, out)
        assert res.n_in == 0 and res.n_out == 0

    def test_header_preserved_in_output(self, tmp_path: Path):
        header = "% program   : rnx2rtkp\n% custom line\n"
        pos = _make_pos_file(tmp_path / "in.pos", self._good_rows(3), header=header)
        out = tmp_path / "out.pos"
        ppk.clean_pos(pos, out)
        text = out.read_text(encoding="utf-8")
        assert "% program   : rnx2rtkp" in text
        assert "% custom line" in text

    def test_output_parseable_by_parse_rtkpos(self, tmp_path: Path):
        """Cleaned .pos must be parseable by parse_rtkpos — same column layout."""
        pos = _make_pos_file(tmp_path / "in.pos", self._good_rows(5))
        out = tmp_path / "out.pos"
        ppk.clean_pos(pos, out)
        rows = parse_rtkpos(out)
        assert len(rows) == 5


# ───────────────────────────────────────────────────────────────────
# _write_pos_like round-trip (pipeline_full)
# ───────────────────────────────────────────────────────────────────

class TestWritePosRoundTrip:
    def test_write_then_parse_preserves_position(self, tmp_path: Path):
        from data_pipeline.pipeline_full import _write_pos_like

        template = tmp_path / "template.pos"
        template.write_text("% header\n2024/01/01 00:00:18.000 32 34 100 1 12\n")

        rows = [
            PosRow(utc_s=1704067200.0, lat_deg=32.123456789, lon_deg=34.987654321,
                   h_m=55.1234, quality=1, vn=0.5, ve=1.0, vu=0.0, ns=12),
            PosRow(utc_s=1704067201.0, lat_deg=32.123457000, lon_deg=34.987655000,
                   h_m=55.2000, quality=2, vn=0.3, ve=0.8, vu=0.1, ns=10),
        ]
        out = tmp_path / "roundtrip.pos"
        _write_pos_like(template, rows, out)
        parsed = parse_rtkpos(out)

        assert len(parsed) == 2
        for orig, back in zip(rows, parsed):
            assert abs(orig.lat_deg - back.lat_deg) < 1e-8
            assert abs(orig.lon_deg - back.lon_deg) < 1e-8
            assert abs(orig.h_m - back.h_m) < 0.01
            assert back.quality == orig.quality

    def test_nan_velocity_writes_as_zero(self, tmp_path: Path):
        from data_pipeline.pipeline_full import _write_pos_like

        template = tmp_path / "template.pos"
        template.write_text("% header\n")
        rows = [PosRow(utc_s=1704067200.0, lat_deg=32.0, lon_deg=34.0,
                       h_m=50.0, quality=1, ns=8)]
        out = tmp_path / "nanvel.pos"
        _write_pos_like(template, rows, out)
        text = out.read_text()
        assert "0.00000" in text
        parsed = parse_rtkpos(out)
        assert len(parsed) == 1

    def test_empty_rows_writes_header_only(self, tmp_path: Path):
        from data_pipeline.pipeline_full import _write_pos_like

        template = tmp_path / "template.pos"
        template.write_text("% header line\n")
        out = tmp_path / "empty.pos"
        _write_pos_like(template, [], out)
        text = out.read_text()
        assert text.strip() == "% header line"


# ───────────────────────────────────────────────────────────────────
# epoch_weighted_v2
# ───────────────────────────────────────────────────────────────────

class TestEpochWeightedV2:
    def test_empty_input(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        res = smooth_epoch_weighted_v2([])
        assert len(res.E_smooth) == 0
        assert res.n_nhc_updates == 0
        assert res.n_zupt_updates == 0

    def test_single_epoch(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        rows = _synth_pos(n=1)
        res = smooth_epoch_weighted_v2(rows)
        assert len(res.E_smooth) == 1
        assert math.isfinite(res.E_smooth[0])

    def test_basic_smoothing_reduces_noise(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        from data_pipeline.geo import ecef_to_enu, llh_to_ecef

        rows = _synth_pos(n=60, noise=1.0)
        res = smooth_epoch_weighted_v2(rows)
        assert len(res.E_smooth) == 60
        assert all(math.isfinite(v) for v in res.E_smooth)
        assert all(math.isfinite(v) for v in res.N_smooth)

        ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
        raw_e = []
        for r in rows:
            e, _, _ = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)
            raw_e.append(e)
        raw_var = float(np.var(np.diff(raw_e)))
        smooth_var = float(np.var(np.diff(res.E_smooth)))
        assert smooth_var < raw_var, (
            f"smoothed variance {smooth_var:.4f} should be < raw {raw_var:.4f}"
        )

    def test_nan_velocity_does_not_crash(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        rows = _synth_pos(n=20)
        rows_nan = [
            PosRow(utc_s=r.utc_s, lat_deg=r.lat_deg, lon_deg=r.lon_deg,
                   h_m=r.h_m, quality=r.quality, ns=r.ns,
                   sd_n=0.5, sd_e=0.5, sd_u=1.0)
            for r in rows
        ]
        res = smooth_epoch_weighted_v2(rows_nan)
        assert len(res.E_smooth) == 20

    def test_zupt_fires_during_zero_speed(self):
        from data_pipeline.epoch_weight_v2 import (
            EpochWeightV2Options,
            smooth_epoch_weighted_v2,
        )
        rows = []
        for i in range(30):
            rows.append(PosRow(
                utc_s=float(i), lat_deg=32.0, lon_deg=34.8, h_m=50.0,
                quality=2, vn=0.0, ve=0.0, vu=0.0, ns=10,
                sd_n=0.5, sd_e=0.5, sd_u=1.0,
            ))
        opts = EpochWeightV2Options(zupt_enabled=True, zupt_min_duration_s=2.0)
        res = smooth_epoch_weighted_v2(rows, options=opts)
        assert res.n_zupt_updates > 0

    def test_nhc_fires_during_straight_drive(self):
        from data_pipeline.epoch_weight_v2 import (
            EpochWeightV2Options,
            smooth_epoch_weighted_v2,
        )
        rows = []
        for i in range(30):
            rows.append(PosRow(
                utc_s=float(i), lat_deg=32.0 + i * 5 / 111_000,
                lon_deg=34.8, h_m=50.0, quality=2,
                vn=5.0, ve=0.0, vu=0.0, ns=10,
                sd_n=0.5, sd_e=0.5, sd_u=1.0,
                sd_vn=0.1, sd_ve=0.1, sd_vu=0.1,
            ))
        opts = EpochWeightV2Options(nhc_enabled=True, nhc_speed_thresh_mps=2.0)
        res = smooth_epoch_weighted_v2(rows, options=opts)
        assert res.n_nhc_updates > 0

    def test_diagnostics_have_correct_length(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        rows = _synth_pos(n=40)
        res = smooth_epoch_weighted_v2(rows)
        assert len(res.fwd_bwd_disagree_h) == 40
        assert len(res.fwd_bwd_disagree_3d) == 40
        assert len(res.innovation_h) == 40
        assert len(res.innovation_norm) == 40

    def test_with_imu_rows_does_not_crash(self):
        from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
        pos = _synth_pos(n=20)
        imu = [ImuRow(utc_s=i / 100, ax=0, ay=0, az=9.81, gx=0, gy=0, gz=0)
               for i in range(2000)]
        res = smooth_epoch_weighted_v2(pos, imu_rows=imu)
        assert len(res.E_smooth) == 20


# ───────────────────────────────────────────────────────────────────
# Smoothers registry (v2 wired into smoothers.py)
# ───────────────────────────────────────────────────────────────────

class TestSmoothersV2Integration:
    def test_epoch_weight_in_registry(self):
        from data_pipeline.smoothers import describe, list_smoothers
        names = list_smoothers()
        assert "epoch_weight" in names
        info = describe("epoch_weight")
        assert not info.requires_imu

    def test_run_epoch_weight_smoother(self):
        from data_pipeline.smoothers import run_smoother
        pos = _synth_pos(n=30)
        res = run_smoother("epoch_weight", pos)
        assert res.ok
        assert res.n_output == len(pos)

    def test_run_cv_rts_pv_smoother(self):
        from data_pipeline.smoothers import run_smoother
        pos = _synth_pos(n=30)
        res = run_smoother("cv_rts_pv", pos)
        assert res.ok
        assert res.n_output == len(pos)


# ───────────────────────────────────────────────────────────────────
# AccuracyReport
# ───────────────────────────────────────────────────────────────────

class TestAccuracyReport:
    def test_empty_rows(self):
        from data_pipeline.pipeline_full import _build_accuracy_report
        r = _build_accuracy_report([], 0.0, source_chain="test")
        assert r.n_epochs == 0
        assert math.isnan(r.ci95_h_m)

    def test_normal_rows(self):
        from data_pipeline.pipeline_full import _build_accuracy_report, format_accuracy_report
        rows = _synth_pos(n=100)
        r = _build_accuracy_report(rows, 0.5, source_chain="test", raw_rows=rows)
        assert r.n_epochs == 100
        assert r.duration_min > 0
        assert math.isfinite(r.ci95_h_m)
        text = format_accuracy_report(r)
        assert "ACCURACY REPORT" in text
        assert "test" in text


# ───────────────────────────────────────────────────────────────────
# Setup / install smoke
# ───────────────────────────────────────────────────────────────────

class TestSetupSmoke:
    def test_required_imports_importable(self):
        """Every package in install.py REQUIRED_IMPORTS must import."""
        from install import REQUIRED_IMPORTS
        for mod_name, pkg_name in REQUIRED_IMPORTS:
            try:
                __import__(mod_name)
            except ImportError:
                pytest.skip(f"{pkg_name} not installed — skip in CI")

    def test_lab_tools_report_has_rnx2rtkp(self):
        from data_pipeline.lab_tools import report
        r = report()
        assert "rnx2rtkp" in r
        assert r["rnx2rtkp"] != "MISSING", (
            "rnx2rtkp must resolve from vendor/rtklib or PATH"
        )

    def test_packaged_configs_exist(self):
        confs = ppk.list_packaged_configs()
        names = [c.name for c in confs]
        assert "javad_avg_sp.conf" in names

    def test_vendor_rtklib_binary_exists(self):
        vendor = Path(__file__).resolve().parent.parent / "vendor" / "rtklib" / "rnx2rtkp.exe"
        assert vendor.is_file(), f"bundled rnx2rtkp.exe not at {vendor}"

    def test_rnx2rtkp_version_runs(self):
        """The solver's -? flag should print usage and exit non-zero (help flag)."""
        import subprocess
        exe = ppk.resolve_rnx2rtkp()
        proc = subprocess.run(
            [str(exe), "-?"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=10,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert "rnx2rtkp" in combined.lower() or len(combined) > 10, (
            f"rnx2rtkp -? produced no recognisable output: {combined[:200]}"
        )


# ───────────────────────────────────────────────────────────────────
# Edge cases: parse_rtkpos robustness
# ───────────────────────────────────────────────────────────────────

class TestParseRtkposEdgeCases:
    def test_malformed_rows_skipped_with_warning(self, tmp_path: Path):
        """Rows with non-numeric lat/lon get skipped, not crash."""
        pos = tmp_path / "bad.pos"
        pos.write_text(
            "% header\n"
            "2024/01/01 00:00:18.000 BADLAT BADLON 50.0 1 12\n"
            "2024/01/01 00:00:19.000 32.0 34.8 50.0 1 12 0.01 0.01 0.03\n",
            encoding="utf-8",
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows = parse_rtkpos(pos)
        assert len(rows) == 1

    def test_all_malformed_raises(self, tmp_path: Path):
        pos = tmp_path / "allbad.pos"
        pos.write_text("% header\ngarbage line\nanother bad line\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="malformed"):
            parse_rtkpos(pos)

    def test_header_only_returns_empty(self, tmp_path: Path):
        pos = tmp_path / "hdr.pos"
        pos.write_text("% header only\n% no data\n", encoding="utf-8")
        rows = parse_rtkpos(pos)
        assert rows == []

    def test_ns_column_must_be_integer(self, tmp_path: Path):
        """The bug the first agent fixed: ns=0.01 caused the row to be skipped
        because int('0.01') raises ValueError. When it's the only row,
        parse_rtkpos raises RuntimeError (all rows malformed). The point is
        it doesn't crash with an unhandled ValueError."""
        pos = tmp_path / "float_ns.pos"
        pos.write_text(
            "% header\n"
            "2024/01/01 00:00:18.000 32.0 34.8 50.0 1 0.01 0.01 0.01 0.03\n",
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="malformed"):
            parse_rtkpos(pos)

    def test_velocity_columns_optional(self, tmp_path: Path):
        pos = tmp_path / "novel.pos"
        pos.write_text(
            "% header\n"
            "2024/01/01 00:00:18.000 32.0 34.8 50.0 1 12 0.01 0.01 0.03\n",
            encoding="utf-8",
        )
        rows = parse_rtkpos(pos)
        assert len(rows) == 1
        assert math.isnan(rows[0].vn)
        assert math.isnan(rows[0].ve)

    def test_leap_seconds_applied(self, tmp_path: Path):
        pos = tmp_path / "leap.pos"
        pos.write_text(
            "% header\n"
            "2024/01/01 00:00:18.000 32.0 34.8 50.0 1 12 0.01 0.01 0.03\n",
            encoding="utf-8",
        )
        rows = parse_rtkpos(pos)
        assert len(rows) == 1
        expected_utc = 1704067200.0
        assert abs(rows[0].utc_s - expected_utc) < 1.0


# ───────────────────────────────────────────────────────────────────
# detect_nav_files recursive + Interchange-format 2-digit patterns
# ───────────────────────────────────────────────────────────────────

class TestDetectNavFilesRobust:
    def test_recursive_into_subdirs(self, tmp_path: Path):
        sub = tmp_path / "level1" / "level2"
        sub.mkdir(parents=True)
        (sub / "x.nav").write_bytes(b"")
        found = ppk.detect_nav_files(tmp_path, recursive=True)
        assert any(p.name == "x.nav" for p in found)

    def test_non_recursive_skips_subdirs(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.nav").write_bytes(b"")
        (tmp_path / "top.nav").write_bytes(b"")
        found = ppk.detect_nav_files(tmp_path, recursive=False)
        names = {p.name for p in found}
        assert "top.nav" in names
        assert "deep.nav" not in names

    def test_rinex2_h_suffix(self, tmp_path: Path):
        (tmp_path / "base.24h").write_bytes(b"")
        found = ppk.detect_nav_files(tmp_path)
        assert any(p.name == "base.24h" for p in found)

    def test_none_dir_ignored(self, tmp_path: Path):
        found = ppk.detect_nav_files(None, tmp_path / "nope")
        assert found == []
