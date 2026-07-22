"""Tests for the Interchange-format 3 fine measurements pre-check (data_pipeline.obs_check).

Real-file anchors:

* day15 measurements.obs — the real duty-cycled capture whose every L*
  value is 0.000 -> has_phase must be False.
* DAY12 dodge1_new.obs — old-format capture with real ADR -> True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.obs_check import ObsPhaseReport, check_carrier_phase

ZERO_PHASE_OBS = Path("C:/Aj/gps/day15/output1/measurements.obs")
HAS_PHASE_OBS = Path("C:/Aj/gps/DAY12/dodge1/20260505_152247_472/dodge1_new.obs")


def _hdr(body: str, label: str) -> str:
    return f"{body:<60}{label}\n"


def _obs_line(sat: str, values: list[tuple[float, str]]) -> str:
    """Build a fixed-width Interchange-format 3 observation record.

    ``values`` is (value, 2-char LLI+SSI) per observable, 16 chars a slot.
    """
    out = sat
    for v, flags in values:
        out += f"{v:14.3f}{flags:>2}"
    return out + "\n"


def _write_synth_obs(path: Path, l1_values: list[float]) -> None:
    """Minimal Interchange-format 3.03 obs: one Reference system (C1C L1C D1C S1C), one epoch
    per entry in ``l1_values`` with two sources each sharing that L1."""
    txt = ""
    txt += _hdr("     3.03           O                   M", "RINEX VERSION / TYPE")
    txt += _hdr("Synth               UNKN                2026-07-05 00:00:00", "PGM / RUN BY / DATE")
    txt += _hdr("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES")
    txt += _hdr("", "END OF HEADER")
    for i, l1 in enumerate(l1_values):
        txt += f"> 2026 07 05 11 48 {50 + i:02d}.0000000  0  2\n"
        txt += _obs_line("G01", [(20194638.441, " 3"), (l1, " 4"),
                                 (-434.246, "  "), (44.7, "  ")])
        txt += _obs_line("G02", [(21386120.095, " 2"), (l1, " 4"),
                                 (-2000.044, "  "), (48.8, "  ")])
    path.write_text(txt, encoding="utf-8")


@pytest.mark.skipif(not ZERO_PHASE_OBS.is_file(), reason=f"{ZERO_PHASE_OBS} not available")
def test_real_zero_phase_file() -> None:
    report = check_carrier_phase(ZERO_PHASE_OBS)
    assert isinstance(report, ObsPhaseReport)
    assert report.has_phase is False
    assert report.n_phase_nonzero == 0
    assert report.n_sat_obs > 1000  # real file, thousands of slots scanned
    assert "no usable carrier phase" in report.message
    assert "Q=4" in report.message
    assert "Force full GNSS measurements" in report.message


@pytest.mark.skipif(not HAS_PHASE_OBS.is_file(), reason=f"{HAS_PHASE_OBS} not available")
def test_real_file_with_phase() -> None:
    report = check_carrier_phase(HAS_PHASE_OBS)
    assert report.has_phase is True
    assert report.n_phase_nonzero > 10000  # ~50k in practice
    # Sanity on real magnitudes: nonzero share far above the 1% threshold.
    assert report.n_phase_nonzero / report.n_sat_obs > 0.1
    # Per-system split covers the source groups declared in the header.
    assert "G" in report.per_system
    assert report.per_system["G"]["n_phase_nonzero"] > 0


def test_synth_all_zero_phase(tmp_path: Path) -> None:
    p = tmp_path / "zero.obs"
    _write_synth_obs(p, l1_values=[0.0, 0.0, 0.0])
    report = check_carrier_phase(p)
    assert report.has_phase is False
    assert report.n_phase_nonzero == 0
    assert report.n_sat_obs == 6  # 3 epochs x 2 sources x 1 phase obs
    assert "no usable carrier phase" in report.message


def test_synth_with_phase(tmp_path: Path) -> None:
    p = tmp_path / "good.obs"
    # Real L1 phase is ~1e7-1e8 cycles.
    _write_synth_obs(p, l1_values=[106123456.789, 106123999.123, 106124512.456])
    report = check_carrier_phase(p)
    assert report.has_phase is True
    assert report.n_phase_nonzero == 6
    assert report.n_sat_obs == 6
    assert report.per_system["G"]["phase_types"] == ["L1C"]


def test_synth_below_one_percent(tmp_path: Path) -> None:
    """A lone nonzero phase among hundreds of zeros is still 'no phase'."""
    p = tmp_path / "sparse.obs"
    values = [0.0] * 200
    values[0] = 106123456.789  # 2 nonzero slots (both sources) of 402 -> <1%
    _write_synth_obs(p, l1_values=values)
    report = check_carrier_phase(p)
    assert report.n_phase_nonzero == 2
    assert report.n_sat_obs == 400
    assert report.has_phase is False


def test_wrapped_continuation_line(tmp_path: Path) -> None:
    """Tolerate writers that wrap a source's observations to a 2nd line."""
    p = tmp_path / "wrap.obs"
    txt = ""
    txt += _hdr("     3.03           O                   M", "RINEX VERSION / TYPE")
    txt += _hdr("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES")
    txt += _hdr("", "END OF HEADER")
    txt += "> 2026 07 05 11 48 50.0000000  0  1\n"
    full = _obs_line("G01", [(20194638.441, " 3"), (106123456.789, " 4"),
                             (-434.246, "  "), (44.7, "  ")])
    # Split after the first observable: rest continues on the next line.
    txt += full[:19] + "\n" + full[19:]
    p.write_text(txt, encoding="utf-8")
    report = check_carrier_phase(p)
    assert report.n_sat_obs == 1
    assert report.n_phase_nonzero == 1
    assert report.has_phase is True
