"""Tests for the multi-elevation-mask Post-processing stage (GT-free selector + fusion).

These tests exercise the pure-Python disagreement + fusion maths with
synthetic per-mask ``.pos`` inputs. The actual solver invocation is NOT
required: the one test that touches it is skipped when the executable is
absent (``shutil.which`` / vendor probe).
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.parsers import GPS_UTC_LEAP_SECONDS_2026 as LS
from data_pipeline.parsers import parse_rtkpos
from data_pipeline.stages import multimask_ppk as M

MASKS = [5, 10, 15, 20, 25, 30]


# ---------------------------------------------------------------------------
# Synthetic .pos writer (The external solver-style subject .pos that parse_rtkpos accepts)
# ---------------------------------------------------------------------------
def _write_pos(path: Path, t0_utc: float, n: int, lat0: float, lon0: float,
               h0: float, *, east_off_m: float = 0.0, north_off_m: float = 0.0,
               q: int = 1, ns: int = 12, ratio: float = 5.0,
               sdh: float = 0.01) -> Path:
    mlat = 111320.0
    mlon = 111320.0 * np.cos(np.radians(lat0))
    sd = sdh / np.sqrt(2.0)
    lines = ["% synthetic per-mask .pos"]
    for i in range(n):
        t = t0_utc + i
        gpst = dt.datetime.fromtimestamp(t + LS, tz=dt.timezone.utc)
        stamp = gpst.strftime("%Y/%m/%d %H:%M:%S.") + f"{gpst.microsecond // 1000:03d}"
        lat = lat0 + north_off_m / mlat
        lon = lon0 + east_off_m / mlon
        lines.append(
            f"{stamp}  {lat:14.9f} {lon:14.9f} {h0:10.4f}  {q}  {ns}  "
            f"{sd:.4f}  {sd:.4f}  0.02  0  0  0  0.0  {ratio:.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _maskset(E_by_mask: dict[int, float], T: int = 30, *, q: int = 1,
             ns: int = 12, ratio: float = 5.0, sdh: float = 0.01,
             ns_by_mask: dict[int, int] | None = None) -> M.MaskSet:
    """Build a MaskSet directly with a constant East offset per mask."""
    masks = sorted(E_by_mask.keys())
    ts = np.arange(T, dtype=float)
    ms = M.MaskSet(masks=masks, ts=ts, ref_llh=(45.0, 7.0, 200.0))
    ms.E = {m: np.full(T, E_by_mask[m]) for m in masks}
    ms.N = {m: np.zeros(T) for m in masks}
    ms.U = {m: np.zeros(T) for m in masks}
    ms.Q = {m: np.full(T, q, dtype=int) for m in masks}
    if ns_by_mask is None:
        ms.ns = {m: np.full(T, ns, dtype=int) for m in masks}
    else:
        ms.ns = {m: np.full(T, ns_by_mask.get(m, ns), dtype=int) for m in masks}
    ms.ratio = {m: np.full(T, ratio) for m in masks}
    ms.sdh = {m: np.full(T, sdh) for m in masks}
    return ms


# ---------------------------------------------------------------------------
# Disagreement / spread
# ---------------------------------------------------------------------------
def test_disagreement_zero_when_masks_agree():
    ms = _maskset({m: 0.0 for m in MASKS})
    sp = M.epoch_spread(ms)
    assert sp.max() < 1e-9


def test_disagreement_equals_max_pairwise_distance():
    # masks span 0..3 m East -> max pairwise horizontal spread is 3 m.
    ms = _maskset({5: 3.0, 10: 1.5, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0})
    sp = M.epoch_spread(ms)
    assert sp.max() == pytest.approx(3.0, abs=1e-6)


def test_flag_fires_above_threshold_and_not_below():
    cfg = M.FuseConfig(flag_threshold_m=0.6)
    hi = _maskset({5: 3.0, 10: 1.5, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0})
    lo = _maskset({m: 0.0 for m in MASKS})
    assert M.fuse(hi, cfg).flag.all()
    assert not M.fuse(lo, cfg).flag.any()


# ---------------------------------------------------------------------------
# Fused selector
# ---------------------------------------------------------------------------
def test_consensus_picks_lowest_mask():
    """Open sky (all masks agree) -> trust the lowest mask (most sources)."""
    ms = _maskset({m: 0.0 for m in MASKS})
    fr = M.fuse(ms, M.FuseConfig())
    assert (fr.chosen_mask == 5).all()
    assert not fr.escalated.any()


def test_disagreement_escalates_to_convergence_mask():
    """Environment noise signature: answer migrates with mask and stabilises at 15+.
    Selector should escalate to the minimal mask where higher masks converge.
    """
    ms = _maskset({5: 3.0, 10: 1.5, 15: 0.05, 20: 0.0, 25: 0.0, 30: 0.0})
    fr = M.fuse(ms, M.FuseConfig(stab_m=0.4, tau_m=0.6))
    assert fr.escalated.all()
    assert (fr.chosen_mask == 15).all()


def test_geometry_guard_falls_back_when_high_masks_starved():
    """If the escalation target is starved of sources, the geometry guard
    must refuse it and fall back to a mask with acceptable geometry.
    """
    ms = _maskset(
        {5: 3.0, 10: 1.5, 15: 0.05, 20: 0.0, 25: 0.0, 30: 0.0},
        q=2,  # float so the ns-based guard is active
        ns_by_mask={5: 12, 10: 12, 15: 2, 20: 2, 25: 2, 30: 2},
    )
    fr = M.fuse(ms, M.FuseConfig(stab_m=0.4, tau_m=0.6, min_ns=5))
    assert (fr.chosen_mask < 15).all()


def test_fuse_uses_fallback_mask_when_chosen_has_nan():
    ms = _maskset({m: 0.0 for m in MASKS})
    # blank the lowest mask at epoch 0 -> fused must still produce a position
    ms.E[5][0] = np.nan
    fr = M.fuse(ms, M.FuseConfig())
    assert np.isfinite(fr.E[0])


# ---------------------------------------------------------------------------
# Loading + alignment from .pos files
# ---------------------------------------------------------------------------
def test_build_maskset_intersects_common_epochs(tmp_path):
    t0 = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    pbm = {}
    for m in MASKS:
        # mask 30 has only 40 epochs; the rest 60 -> common = 40
        n = 40 if m == 30 else 60
        pbm[m] = _write_pos(tmp_path / f"mask{m:02d}.pos", t0, n,
                            45.0, 7.0, 200.0)
    ms = M.build_maskset_from_posfiles(pbm)
    assert ms is not None
    assert ms.masks == MASKS
    assert ms.T == 40


def test_build_maskset_returns_none_with_too_few_masks(tmp_path):
    t0 = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    pbm = {5: _write_pos(tmp_path / "mask05.pos", t0, 60, 45.0, 7.0, 200.0)}
    assert M.build_maskset_from_posfiles(pbm) is None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def test_disagreement_csv_has_expected_columns(tmp_path):
    ms = _maskset({5: 3.0, 10: 1.5, 15: 0.0, 20: 0.0, 25: 0.0, 30: 0.0})
    fr = M.fuse(ms, M.FuseConfig())
    csv = M.write_disagreement_csv(ms, fr, tmp_path / "dis.csv")
    head = csv.read_text(encoding="utf-8").splitlines()
    cols = head[0].split(",")
    assert cols[0] == "gpstime"
    assert "disagreement_m" in cols
    assert "flag" in cols
    assert "chosen_mask" in cols
    for m in MASKS:
        assert f"lat_m{m:02d}" in cols and f"lon_m{m:02d}" in cols
    # one data row per epoch
    assert len(head) - 1 == ms.T


def test_fused_pos_roundtrips_through_parse_rtkpos(tmp_path):
    ms = _maskset({m: 0.0 for m in MASKS}, T=25)
    fr = M.fuse(ms, M.FuseConfig())
    fp = M.write_fused_pos(fr, tmp_path / "fused.pos")
    rows = parse_rtkpos(fp)
    assert len(rows) == ms.T
    # timestamps recover to the same UTC seconds (within 1 ms rounding)
    assert abs(rows[0].utc_s - ms.ts[0]) < 0.01


def test_patch_elmask_overrides_only_that_key(tmp_path):
    src = tmp_path / "base.conf"
    src.write_text(
        "pos1-posmode       =kinematic\n"
        "pos1-elmask        =5          # (deg)\n"
        "pos1-snrmask_r     =on\n"
        "ant2-pos1          =4517590.0\n",
        encoding="utf-8",
    )
    dst = M.patch_elmask(src, tmp_path / "m20.conf", 20)
    txt = dst.read_text(encoding="utf-8")
    assert "pos1-elmask        =20" in txt
    assert "(deg)" in txt                    # inline comment preserved
    assert "pos1-posmode       =kinematic" in txt
    assert "ant2-pos1          =4517590.0" in txt   # base position untouched
    assert "pos1-snrmask_r     =on" in txt


def test_patch_elmask_appends_when_absent(tmp_path):
    src = tmp_path / "base.conf"
    src.write_text("pos1-posmode       =kinematic\n", encoding="utf-8")
    dst = M.patch_elmask(src, tmp_path / "m15.conf", 15)
    assert "pos1-elmask" in dst.read_text(encoding="utf-8")
    assert "=15" in dst.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Real solver invocation — skipped when the exe is absent
# ---------------------------------------------------------------------------
def _have_rnx2rtkp() -> bool:
    return M._resolve_rnx2rtkp(None) is not None


@pytest.mark.skipif(not _have_rnx2rtkp(),
                    reason="rnx2rtkp executable not available")
def test_run_multimask_ppk_raises_on_missing_inputs(tmp_path):
    # exe present but inputs missing -> the solver binary fails every mask, fusion skips.
    res = M.run_multimask_ppk(
        tmp_path / "nope_rover.obs",
        tmp_path / "nope_base.obs",
        [tmp_path / "nope.nav"],
        tmp_path / "nope.conf",   # patch_elmask reads this; create a stub
        masks=[5, 10],
        workdir=tmp_path / "work",
        make_report=False,
    ) if (tmp_path / "nope.conf").write_text("pos1-elmask=5\n") or True else None
    # No masks solve -> no fused output, but the call returns a result object.
    assert res.fused_pos is None
    assert res.maskset is None


def test_run_multimask_ppk_raises_without_exe(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_resolve_rnx2rtkp", lambda override: None)
    (tmp_path / "x.conf").write_text("pos1-elmask=5\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        M.run_multimask_ppk(
            tmp_path / "r.obs", tmp_path / "b.obs", [tmp_path / "n.nav"],
            tmp_path / "x.conf", masks=[5, 10], workdir=tmp_path / "w",
        )
