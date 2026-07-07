"""Tests for the one-file Post-processing Analysis report (data_pipeline.analysis_report)
and the obs SNR/source-count aggregation (data_pipeline.obs_check.summarize_obs).

Real-file anchor: the day15 duty-cycled capture (zero fine measurements) must
produce a report that says Differential-only and shows the no-phase warning.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_pipeline.analysis_report import build_analysis_report
from data_pipeline.obs_check import summarize_obs

DAY15_POS = Path("C:/Aj/gps/day15/output1/measurements2.pos")
DAY15_OBS = Path("C:/Aj/gps/day15/output1/measurements.obs")


# ---------------------------------------------------------------------------
# synth .pos helper
# ---------------------------------------------------------------------------


def _row_time(i: int) -> str:
    """1 Hz epoch time string starting at 11:48:50 (rolls over minutes)."""
    total = 48 * 60 + 50 + i
    return f"11:{total // 60:02d}:{total % 60:02d}.000"


def _write_synth_pos(path: Path, qualities: list[int]) -> None:
    """Minimal The external solver llh .pos: one row per entry in ``qualities``, 1 Hz."""
    hdr = (
        "% program   : RTKLIB ver.EX 2.5.0\n"
        "% pos mode  : Kinematic\n"
        "% freqs     : L1+L2\n"
        "% elev mask : 15.0 deg\n"
        "% navi sys  : GPS GLONASS Galileo\n"
        "% amb res   : Continuous\n"
        "% val thres : 3.0\n"
        "% ref pos   :  32.000000000   34.800000000    50.0000\n"
        "% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,3:sbas,4:dgps,"
        "5:single,6:ppp,ns=# of satellites)\n"
        "%  GPST                  latitude(deg) longitude(deg)  height(m)"
        "   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m)"
        " age(s)  ratio\n"
    )
    rows = ""
    for i, q in enumerate(qualities):
        ns = 14 + (i % 3)
        rows += (
            f"2026/07/05 {_row_time(i)}   "
            f"{32.0 + i * 1e-7:.9f}   {34.8 + i * 1e-7:.9f}    50.0000   "
            f"{q}  {ns:2d}   0.1000   0.1200   0.2500   0.0100   0.0100"
            f"  -0.0100   1.00    2.5\n"
        )
    path.write_text(hdr + rows, encoding="utf-8")


def test_synth_pos_report_panels(tmp_path: Path) -> None:
    """Mixed-Q synth .pos -> report has Q-distribution, ns and predicted
    accuracy panels; missing optional inputs are noted, not fatal."""
    pos = tmp_path / "synth.pos"
    # 12 rows: FLOAT-dominant (7 float, 3 fix, 2 single).
    _write_synth_pos(pos, [2, 2, 1, 2, 2, 5, 2, 1, 2, 5, 2, 1])
    out = tmp_path / "report.html"
    got = build_analysis_report(pos, out)
    assert got == out and out.is_file()
    html = out.read_text(encoding="utf-8")
    # Q-distribution panel + FLOAT-dominant headline (7/12 = 58%).
    assert "Q-distribution" in html
    assert "Solved: FLOAT (58%)" in html
    # Sources panel with the ns charts.
    assert "Satellites used per epoch" in html
    assert "Average satellites used" in html
    # Predicted-accuracy section (session sigma + CDF chart div).
    assert "Predicted accuracy" in html
    assert "pred-cdf" in html
    # Optional inputs missing -> notes, no crash.
    assert "No .obs file provided" in html
    assert "No ground-truth .pos provided" in html
    # Offline single file: plotly must be inlined, not referenced.
    assert 'src="plotly.min.js"' not in html
    assert "Plotly.newPlot" in html


def test_synth_pos_report_all_dgps_headline(tmp_path: Path) -> None:
    pos = tmp_path / "dgps.pos"
    _write_synth_pos(pos, [4] * 10)
    out = tmp_path / "dgps.html"
    build_analysis_report(pos, out)
    html = out.read_text(encoding="utf-8")
    assert "DGPS-only (Q=4)" in html
    assert "no carrier phase" in html


def test_synth_pos_report_with_gt(tmp_path: Path) -> None:
    """A GT .pos offset ~2 m east -> error CDF + error-vs-ns panel appear."""
    pos = tmp_path / "rover.pos"
    _write_synth_pos(pos, [2] * 20)
    gt = tmp_path / "gt.pos"
    # Same times, longitude shifted by ~2 m (at lat 32: 1 deg lon ~ 94.4 km).
    hdr = (
        "%  GPST                  latitude(deg) longitude(deg)  height(m)"
        "   Q  ns   sdn(m)   sde(m)   sdu(m)\n"
    )
    dlon = 2.0 / (111_320.0 * math.cos(math.radians(32.0)))
    rows = ""
    for i in range(20):
        rows += (
            f"2026/07/05 {_row_time(i)}   "
            f"{32.0 + i * 1e-7:.9f}   {34.8 + i * 1e-7 + dlon:.9f}"
            f"    50.0000   1  20   0.0100   0.0100   0.0200\n"
        )
    gt.write_text(hdr + rows, encoding="utf-8")
    out = tmp_path / "gt_report.html"
    build_analysis_report(pos, out, gt_pos=gt)
    html = out.read_text(encoding="utf-8")
    assert "Measured vs ground truth" in html
    assert "median error" in html
    assert "gt-cdf" in html          # error CDF chart
    assert "gt-ns" in html           # error-vs-sources chart
    # 2 m offset -> median error ~2.00 m must be reported.
    assert "2.00 m" in html


# ---------------------------------------------------------------------------
# ObsSummary on a tiny synthesized Interchange-format 3
# ---------------------------------------------------------------------------


def _hdr(body: str, label: str) -> str:
    return f"{body:<60}{label}\n"


def _obs_line(sat: str, values: list[tuple[float, str]]) -> str:
    out = sat
    for v, flags in values:
        out += f"{v:14.3f}{flags:>2}"
    return out + "\n"


def test_summarize_obs_synth(tmp_path: Path) -> None:
    """3 epochs x 2 sources, SNR 44.0 / 48.0 -> exact aggregation numbers."""
    p = tmp_path / "tiny.obs"
    txt = ""
    txt += _hdr("     3.03           O                   M", "RINEX VERSION / TYPE")
    txt += _hdr("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES")
    txt += _hdr("", "END OF HEADER")
    for i in range(3):
        txt += f"> 2026 07 05 11 48 {50 + i:02d}.0000000  0  2\n"
        txt += _obs_line("G01", [(20194638.441, " 3"), (106123456.789, " 4"),
                                 (-434.246, "  "), (44.0, "  ")])
        txt += _obs_line("G02", [(21386120.095, " 2"), (106999888.111, " 4"),
                                 (-2000.044, "  "), (48.0, "  ")])
    p.write_text(txt, encoding="utf-8")

    s = summarize_obs(p)
    assert s.epoch_count == 3
    assert s.avg_sats_per_epoch == pytest.approx(2.0)
    assert s.avg_snr_db == pytest.approx(46.0)
    assert s.snr_per_system == {"G": pytest.approx(46.0)}
    assert s.interval_s == pytest.approx(1.0)
    assert len(s.times_s) == 3
    assert s.sats_per_epoch == [2, 2, 2]
    assert s.snr_per_epoch == [pytest.approx(46.0)] * 3


def test_summarize_obs_no_snr_column(tmp_path: Path) -> None:
    """A file whose header declares no S* observable -> NaN SNR, counts OK."""
    p = tmp_path / "nosnr.obs"
    txt = ""
    txt += _hdr("     3.03           O                   M", "RINEX VERSION / TYPE")
    txt += _hdr("G    2 C1C L1C", "SYS / # / OBS TYPES")
    txt += _hdr("", "END OF HEADER")
    txt += "> 2026 07 05 11 48 50.0000000  0  1\n"
    txt += _obs_line("G01", [(20194638.441, " 3"), (106123456.789, " 4")])
    p.write_text(txt, encoding="utf-8")
    s = summarize_obs(p)
    assert s.epoch_count == 1
    assert s.avg_sats_per_epoch == pytest.approx(1.0)
    assert math.isnan(s.avg_snr_db)
    assert s.snr_per_system == {}


# ---------------------------------------------------------------------------
# sources seen (raw obs) — subject vs base chart in panel 2
# ---------------------------------------------------------------------------


def _write_tiny_obs(path: Path, n_sats: int) -> None:
    """3 epochs x ``n_sats`` Reference sources, Interchange-format 3, epochs at 11:48:50 + i s
    (same clock as the synth .pos rows)."""
    txt = ""
    txt += _hdr("     3.03           O                   M", "RINEX VERSION / TYPE")
    txt += _hdr("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES")
    txt += _hdr("", "END OF HEADER")
    for i in range(3):
        txt += f"> 2026 07 05 11 48 {50 + i:02d}.0000000  0  {n_sats}\n"
        for s in range(n_sats):
            txt += _obs_line(
                f"G{s + 1:02d}",
                [(20194638.441 + s, " 3"), (106123456.789 + s, " 4"),
                 (-434.246, "  "), (44.0 + s, "  ")])
    path.write_text(txt, encoding="utf-8")


def test_report_sats_seen_rover_vs_base(tmp_path: Path) -> None:
    """Subject + base .obs -> panel 2 grows a subject-vs-base 'seen' chart with
    both traces and the seen-vs-used explainer."""
    pos = tmp_path / "rover.pos"
    _write_synth_pos(pos, [1, 2, 2])
    rover = tmp_path / "rover.obs"
    base = tmp_path / "base.obs"
    _write_tiny_obs(rover, 2)
    _write_tiny_obs(base, 3)
    out = tmp_path / "seen.html"
    build_analysis_report(pos, out, rover_obs=rover, base_obs=base)
    html = out.read_text(encoding="utf-8")
    assert "Satellites seen (raw observations)" in html
    assert "sats-seen-time" in html          # the chart div
    assert "Rover (seen)" in html            # both traces present
    assert "Base (seen)" in html
    # explainer distinguishing seen vs ns-used
    assert "ns &le; seen" in html
    # avg seen stat line (subject 2.0, base 3.0 sources/epoch)
    assert "Rover seen: <b>2.0</b> sats/epoch" in html
    assert "Base seen: <b>3.0</b> sats/epoch" in html


def test_report_sats_seen_rover_only(tmp_path: Path) -> None:
    """Subject .obs only -> subject trace renders, base absence is noted."""
    pos = tmp_path / "rover.pos"
    _write_synth_pos(pos, [2, 2, 2])
    rover = tmp_path / "rover.obs"
    _write_tiny_obs(rover, 2)
    out = tmp_path / "seen_rover_only.html"
    build_analysis_report(pos, out, rover_obs=rover)
    html = out.read_text(encoding="utf-8")
    assert "Satellites seen (raw observations)" in html
    assert "Rover (seen)" in html
    assert "Base (seen)" not in html
    assert "Base .obs not provided — base line omitted." in html
    # ns chart still there
    assert "Satellites used per epoch" in html


def test_report_sats_seen_omitted_without_obs(tmp_path: Path) -> None:
    """No .obs at all -> the seen chart is omitted with a note; ns chart
    (pos-only) is untouched."""
    pos = tmp_path / "rover.pos"
    _write_synth_pos(pos, [2, 2, 2])
    out = tmp_path / "noobs.html"
    build_analysis_report(pos, out)
    html = out.read_text(encoding="utf-8")
    assert "sats-seen-time" not in html
    assert "satellites-seen (raw observations) chart omitted" in html
    assert "Satellites used per epoch" in html


# ---------------------------------------------------------------------------
# real-file smoke: day15 duty-cycled capture -> Differential-only / no-phase wording
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (DAY15_POS.is_file() and DAY15_OBS.is_file()),
    reason="day15 real files not available",
)
def test_real_day15_dgps_no_phase(tmp_path: Path) -> None:
    out = tmp_path / "day15.html"
    build_analysis_report(DAY15_POS, out, rover_obs=DAY15_OBS)
    html = out.read_text(encoding="utf-8")
    assert "DGPS" in html
    assert "NO carrier phase" in html
    assert "Force full GNSS measurements" in html
