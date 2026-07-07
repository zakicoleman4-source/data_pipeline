"""Tests for the pure helpers in scripts/conf_sweep_multipath.py.

No The external solver executable and no real data required: only grid expansion, the
snrmask string builder, min-max normalization, the weighted combiner, and
the ranking logic are exercised.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from conf_sweep_multipath import (  # noqa: E402
    RESULT_COLS,
    SNRMASK_BINS,
    Variant,
    accuracy_all_nan,
    apply_overrides,
    build_overrides,
    clear_variant_outputs,
    combine_scores,
    expand_grid,
    minmax_normalize,
    parse_args,
    percentile,
    rank_variants,
    run_failure_status,
    snrmask_str,
)


# ---------------------------------------------------------------------------
# snrmask string builder
# ---------------------------------------------------------------------------
def test_snrmask_str_nine_values():
    s = snrmask_str(33)
    parts = s.split(",")
    assert len(parts) == SNRMASK_BINS == 9
    assert all(p == "33" for p in parts)


def test_snrmask_str_float_integer_collapses():
    assert snrmask_str(35.0) == ",".join(["35"] * 9)


def test_snrmask_str_non_integer_kept():
    assert snrmask_str(32.5).split(",") == ["32.5"] * 9


# ---------------------------------------------------------------------------
# grid expansion
# ---------------------------------------------------------------------------
def test_expand_grid_count_and_contents():
    els = [5, 10, 15, 20]
    snrs = ["off", "30", "33", "35", "38"]
    grid = expand_grid(els, snrs)
    assert len(grid) == 20  # 4 x 5 default grid
    names = [v.name for v in grid]
    assert len(set(names)) == 20  # unique names
    assert "el5_snroff" in names
    assert "el20_snr38" in names
    # every combination present
    combos = {(v.elmask_deg, v.snr) for v in grid}
    assert combos == {(float(e), s) for e in els for s in snrs}


def test_expand_grid_overrides_elmask_keys():
    (v,) = expand_grid([15], ["off"])
    ov = v.overrides
    # all three elevation keys move together
    assert ov["pos1-elmask"] == "15"
    assert ov["pos2-arelmask"] == "15"
    assert ov["pos2-elmaskhold"] == "15"
    # .pos.stat must always be produced
    assert ov["out-outstat"] == "residual"


def test_expand_grid_snr_off_disables_masks():
    (v,) = expand_grid([10], ["off"])
    assert v.overrides["pos1-snrmask_r"] == "off"
    assert v.overrides["pos1-snrmask_b"] == "off"
    assert "pos1-snrmask_L1" not in v.overrides


def test_expand_grid_snr_on_sets_all_bands():
    (v,) = expand_grid([10], ["38"])
    ov = v.overrides
    assert ov["pos1-snrmask_r"] == "on"
    assert ov["pos1-snrmask_b"] == "on"
    expected = ",".join(["38"] * 9)
    for band in ("L1", "L2", "L5", "L6"):
        assert ov[f"pos1-snrmask_{band}"] == expected


def test_build_overrides_matches_grid():
    assert build_overrides(5.0, "30") == expand_grid([5], ["30"])[0].overrides


# ---------------------------------------------------------------------------
# conf patching
# ---------------------------------------------------------------------------
def test_apply_overrides_patches_and_appends():
    base = [
        "# comment kept",
        "pos1-elmask        =5          # (deg)",
        "pos1-posmode       =kinematic",
    ]
    out = apply_overrides(base, {"pos1-elmask": "20", "brand-new-key": "abc"})
    assert out[0] == "# comment kept"
    patched = [l for l in out if l.strip().startswith("pos1-elmask")]
    assert len(patched) == 1
    key, val = patched[0].split("=", 1)
    assert val.split("#")[0].strip() == "20"
    assert any(l.strip().startswith("brand-new-key") and
               l.split("=", 1)[1].strip() == "abc" for l in out)
    # untouched key preserved verbatim
    assert "pos1-posmode       =kinematic" in out


# ---------------------------------------------------------------------------
# min-max normalization
# ---------------------------------------------------------------------------
def test_minmax_normalize_lower_is_better():
    n = minmax_normalize([1.0, 3.0, 2.0])
    assert n[0] == 0.0        # best (smallest) -> 0
    assert n[1] == 1.0        # worst (largest) -> 1
    assert n[2] == pytest.approx(0.5)


def test_minmax_normalize_higher_is_better_flips():
    n = minmax_normalize([10.0, 90.0, 50.0], higher_is_better=True)
    assert n[1] == 0.0        # highest fix% -> best -> 0
    assert n[0] == 1.0
    assert n[2] == pytest.approx(0.5)


def test_minmax_normalize_degenerate_all_equal_no_div0():
    n = minmax_normalize([7.0, 7.0, 7.0])
    assert all(math.isfinite(x) for x in n)
    assert n == [0.5, 0.5, 0.5]


def test_minmax_normalize_nan_is_worst():
    n = minmax_normalize([1.0, float("nan"), 2.0])
    assert n[1] == 1.0
    assert n[0] == 0.0 and n[2] == 1.0


def test_minmax_normalize_all_nan():
    assert minmax_normalize([float("nan")] * 3) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# weighted combiner
# ---------------------------------------------------------------------------
def test_combine_scores_better_metrics_lower_combined():
    # variant 0 best on everything (all normalized 0), variant 1 worst (all 1)
    c = combine_scores([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                       w_accuracy=0.5, w_multipath=0.5, have_accuracy=True)
    assert c[0] < c[1]
    assert c[0] == 0.0
    assert c[1] == pytest.approx(1.0)


def test_combine_scores_no_gt_uses_multipath_only():
    # accuracy channels say v0 is terrible, but with no GT they must be ignored
    c = combine_scores([1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
                       w_accuracy=0.5, w_multipath=0.5, have_accuracy=False)
    assert c[0] == 0.0
    assert c[1] == pytest.approx(1.0)
    assert c[0] < c[1]


def test_combine_scores_weights_renormalized():
    # pure accuracy weighting: environment noise channel ignored
    c = combine_scores([0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0],
                       w_accuracy=1.0, w_multipath=0.0, have_accuracy=True)
    assert c == [0.0, pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# ranking end-to-end (synthetic winner)
# ---------------------------------------------------------------------------
def _metric(variant, two_sigma, mx, p95, fix, med=0.5, snr=40.0):
    return {"variant": variant, "elmask": 10.0, "snr": "33", "status": "ok",
            "two_sigma_m": two_sigma, "max_m": mx,
            "p_resid_med": med, "p_resid_p95": p95, "fix_pct": fix,
            "mean_snr": snr}


def test_rank_variants_picks_synthetic_winner():
    metrics = [
        _metric("mediocre", two_sigma=1.0, mx=3.0, p95=2.0, fix=50.0),
        _metric("winner",   two_sigma=0.2, mx=1.0, p95=0.5, fix=90.0),
        _metric("loser",    two_sigma=2.0, mx=6.0, p95=4.0, fix=10.0),
    ]
    ranked = rank_variants(metrics, 0.5, 0.5, have_accuracy=True)
    assert ranked[0]["variant"] == "winner"
    assert ranked[-1]["variant"] == "loser"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["combined_score"] < ranked[1]["combined_score"] \
        < ranked[2]["combined_score"]
    # input not mutated
    assert "combined_score" not in metrics[0]


def test_rank_variants_no_gt_ignores_accuracy():
    # "acc_champ" dominates accuracy but has awful environment noise/fix;
    # without GT the environment noise champion must win.
    metrics = [
        _metric("acc_champ", two_sigma=0.1, mx=0.5, p95=5.0, fix=5.0),
        _metric("mp_champ",  two_sigma=3.0, mx=9.0, p95=0.4, fix=95.0),
    ]
    ranked = rank_variants(metrics, 0.5, 0.5, have_accuracy=False)
    assert ranked[0]["variant"] == "mp_champ"


def test_rank_variants_nan_accuracy_never_wins():
    # failed run -> NaN everywhere ranks last
    metrics = [
        _metric("ok", two_sigma=1.0, mx=2.0, p95=1.0, fix=60.0),
        _metric("failed", two_sigma=float("nan"), mx=float("nan"),
                p95=float("nan"), fix=float("nan")),
    ]
    ranked = rank_variants(metrics, 0.5, 0.5, have_accuracy=True)
    assert ranked[0]["variant"] == "ok"


# ---------------------------------------------------------------------------
# stale-output guard: clear_variant_outputs + run_failure_status
# ---------------------------------------------------------------------------
def test_clear_variant_outputs_deletes_stale_pos_and_stat(tmp_path):
    pos = tmp_path / "v" / "v.pos"
    pos.parent.mkdir()
    pos.write_text("stale pos from previous sweep", encoding="utf-8")
    stat = Path(str(pos) + ".stat")
    stat.write_text("stale stat", encoding="utf-8")

    returned_stat = clear_variant_outputs(pos)

    assert returned_stat == stat  # same .stat path the scorer reads
    assert not pos.exists()       # a failed rerun cannot re-parse these
    assert not stat.exists()


def test_clear_variant_outputs_noop_when_absent(tmp_path):
    pos = tmp_path / "fresh.pos"
    stat = clear_variant_outputs(pos)  # must not raise
    assert stat == Path(str(pos) + ".stat")


def test_run_failure_status_nonzero_rc_fails_even_if_pos_exists():
    # Simulated solver crash: rc!=0 but a (stale/partial) .pos exists.
    # The variant MUST be marked failed, never parsed/scored.
    status = run_failure_status(1, True, 4096, "segfault")
    assert status is not None
    assert "rc=1" in status
    assert "segfault" in status


def test_run_failure_status_missing_or_empty_pos_fails():
    assert run_failure_status(0, False, 0, "") is not None
    assert run_failure_status(0, True, 0, "") is not None  # empty file


def test_run_failure_status_ok_run_passes():
    assert run_failure_status(0, True, 1234, "") is None


# ---------------------------------------------------------------------------
# median_offset_m surfaced (bias-removed accuracy made visible)
# ---------------------------------------------------------------------------
def test_median_off_column_in_results_csv_schema():
    assert "median_off_m" in RESULT_COLS


def test_rank_variants_passes_median_off_through():
    metrics = [
        dict(_metric("a", two_sigma=0.2, mx=1.0, p95=0.5, fix=90.0),
             median_off_m=3.25),
        dict(_metric("b", two_sigma=1.0, mx=3.0, p95=2.0, fix=50.0),
             median_off_m=0.1),
    ]
    ranked = rank_variants(metrics, 0.5, 0.5, have_accuracy=True)
    by_name = {m["variant"]: m for m in ranked}
    # a big constant offset is surfaced but does NOT change scatter ranking
    assert by_name["a"]["median_off_m"] == 3.25
    assert by_name["a"]["rank"] == 1


# ---------------------------------------------------------------------------
# GT-misalignment guard: --max-dt-s flag + all-NaN accuracy detection
# ---------------------------------------------------------------------------
_REQ_ARGS = ["--rover-obs", "r.obs", "--base-obs", "b.obs",
             "--nav", "n.nav", "--out", "out"]


def test_parse_args_max_dt_s_default():
    assert parse_args(_REQ_ARGS).max_dt_s == pytest.approx(0.05)


def test_parse_args_max_dt_s_override():
    assert parse_args(_REQ_ARGS + ["--max-dt-s", "0.5"]).max_dt_s \
        == pytest.approx(0.5)


def test_accuracy_all_nan_detects_total_mismatch():
    nan = float("nan")
    all_bad = [_metric("a", two_sigma=nan, mx=nan, p95=0.5, fix=90.0),
               _metric("b", two_sigma=nan, mx=nan, p95=0.6, fix=80.0)]
    assert accuracy_all_nan(all_bad) is True


def test_accuracy_all_nan_false_when_any_variant_scored():
    nan = float("nan")
    some_good = [_metric("a", two_sigma=nan, mx=nan, p95=0.5, fix=90.0),
                 _metric("b", two_sigma=0.3, mx=1.0, p95=0.6, fix=80.0)]
    assert accuracy_all_nan(some_good) is False


def test_accuracy_all_nan_empty_metrics():
    assert accuracy_all_nan([]) is False


# ---------------------------------------------------------------------------
# percentile helper
# ---------------------------------------------------------------------------
def test_percentile_basic():
    vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile(vals, 50.0) == pytest.approx(3.0)
    assert percentile(vals, 0.0) == pytest.approx(1.0)
    assert percentile(vals, 100.0) == pytest.approx(5.0)


def test_percentile_empty_and_single():
    assert math.isnan(percentile([], 95.0))
    assert percentile([2.5], 95.0) == 2.5


# ---------------------------------------------------------------------------
# Variant dataclass sanity
# ---------------------------------------------------------------------------
def test_variant_is_frozen():
    v = Variant(name="x", elmask_deg=5.0, snr="off")
    with pytest.raises(Exception):
        v.name = "y"  # type: ignore[misc]
