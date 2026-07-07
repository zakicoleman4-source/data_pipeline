"""Tests for data_pipeline.trust_formula_v2 — v2 trust scoring pipeline."""

import numpy as np
import pytest

from data_pipeline.parsers import PosRow
from data_pipeline.trust_formula_v2 import (
    SIGNAL_NAMES,
    TrustConfigV2,
    assign_labels,
    composite_score,
    compute_trust_v2,
    extract_signals,
    normalize_signals,
)


# ---- Test helpers ----

def _make_pos_rows(n=20):
    rows = []
    for i in range(n):
        rows.append(PosRow(
            utc_s=1700000000.0 + i,
            lat_deg=32.0 + i * 0.0001,
            lon_deg=34.8 + i * 0.0001,
            h_m=100.0 + i * 0.1,
            quality=2,
            vn=1.0 + 0.1 * i, ve=0.5, vu=0.0,
            ns=15,
            sd_n=0.5 + 0.01 * i, sd_e=0.3, sd_u=1.0,
            ratio=2.5,
        ))
    return rows


def _make_v2_result(n=20):
    from data_pipeline.epoch_weight_v2 import EpochWeightV2Result
    rng = np.random.RandomState(42)
    return EpochWeightV2Result(
        E_smooth=np.linspace(0, 10, n),
        N_smooth=np.linspace(0, 5, n),
        U_smooth=np.zeros(n),
        vE_smooth=np.ones(n) * 0.5,
        vN_smooth=np.ones(n) * 1.0,
        vU_smooth=np.zeros(n),
        n_nhc_updates=5, n_zupt_updates=3,
        n_doppler_gated=2, n_doppler_vel_filtered=0,
        fwd_bwd_disagree_h=rng.uniform(0, 3, n),
        fwd_bwd_disagree_3d=rng.uniform(0, 5, n),
        innovation_h=rng.uniform(0, 2, n),
        innovation_norm=rng.uniform(0, 4, n),
    )


# ---- Tests ----

class TestExtractSignals:
    def test_extract_signals_shape(self):
        """Shape is (n, 10) and all values are finite."""
        n = 20
        pos = _make_pos_rows(n)
        v2 = _make_v2_result(n)
        signals = extract_signals(pos, v2)
        assert signals.shape == (n, 10)
        assert np.all(np.isfinite(signals))

    def test_extract_signals_names(self):
        """SIGNAL_NAMES has exactly 10 entries matching the spec."""
        expected = [
            "eff_sig", "disagree", "innovation_h", "fwd_bwd_disagree_h",
            "innovation_norm", "q_penalty", "ns_penalty", "sd_h",
            "speed_mps", "ratio_inv",
        ]
        assert SIGNAL_NAMES == expected
        assert len(SIGNAL_NAMES) == 10


class TestNormalize:
    def test_normalize_signals_range(self):
        """All normalized values are in [0, 1]."""
        n = 20
        pos = _make_pos_rows(n)
        v2 = _make_v2_result(n)
        raw = extract_signals(pos, v2)
        normed = normalize_signals(raw)
        assert normed.shape == raw.shape
        assert np.all(normed >= 0.0)
        assert np.all(normed <= 1.0)


class TestCompositeScore:
    def test_composite_score_shape(self):
        """Composite score has shape (n,), all finite, in [0, 1]."""
        n = 20
        pos = _make_pos_rows(n)
        v2 = _make_v2_result(n)
        raw = extract_signals(pos, v2)
        normed = normalize_signals(raw)
        from data_pipeline.trust_formula_v2 import SIGNAL_WEIGHTS_POS
        score = composite_score(normed, SIGNAL_WEIGHTS_POS)
        assert score.shape == (n,)
        assert np.all(np.isfinite(score))
        assert np.all(score >= 0.0)
        assert np.all(score <= 1.0)


class TestAssignLabels:
    def test_assign_labels_four_categories(self):
        """Four score combos produce all four label values."""
        cfg = TrustConfigV2(threshold_pos=0.5, threshold_vel=0.5)
        # pos < thresh AND vel < thresh -> high
        # pos < thresh AND vel >= thresh -> pos_only
        # pos >= thresh AND vel < thresh -> vel_only
        # pos >= thresh AND vel >= thresh -> low
        pos_score = np.array([0.2, 0.2, 0.8, 0.8])
        vel_score = np.array([0.2, 0.8, 0.2, 0.8])
        labels = assign_labels(pos_score, vel_score, cfg)
        assert labels == ["high", "pos_only", "vel_only", "low"]


class TestComputeTrustV2:
    def test_compute_trust_v2_end_to_end(self):
        """Full pipeline on synthetic data returns valid TrustResultV2."""
        n = 20
        pos = _make_pos_rows(n)
        v2 = _make_v2_result(n)
        result = compute_trust_v2(pos, v2)

        # Structural checks.
        assert result.pos_trusted.shape == (n,)
        assert result.vel_trusted.shape == (n,)
        assert result.pos_score.shape == (n,)
        assert result.vel_score.shape == (n,)
        assert result.signals.shape == (n, 10)
        assert len(result.labels) == n

        # All labels are valid.
        valid_labels = {"high", "pos_only", "vel_only", "low"}
        assert all(l in valid_labels for l in result.labels)

        # Counts add up.
        assert (result.n_high + result.n_pos_only
                + result.n_vel_only + result.n_low) == n

        # bool arrays consistent with labels.
        for i, label in enumerate(result.labels):
            if label == "high":
                assert result.pos_trusted[i] and result.vel_trusted[i]
            elif label == "pos_only":
                assert result.pos_trusted[i] and not result.vel_trusted[i]
            elif label == "vel_only":
                assert not result.pos_trusted[i] and result.vel_trusted[i]
            else:
                assert not result.pos_trusted[i] and not result.vel_trusted[i]

    def test_compute_trust_v2_empty(self):
        """Empty input returns empty TrustResultV2 without errors."""
        from data_pipeline.epoch_weight_v2 import EpochWeightV2Result
        empty = np.array([])
        v2 = EpochWeightV2Result(
            E_smooth=empty, N_smooth=empty, U_smooth=empty,
            vE_smooth=empty, vN_smooth=empty, vU_smooth=empty,
            n_nhc_updates=0, n_zupt_updates=0,
            n_doppler_gated=0, n_doppler_vel_filtered=0,
            fwd_bwd_disagree_h=empty, fwd_bwd_disagree_3d=empty,
            innovation_h=empty, innovation_norm=empty,
        )
        result = compute_trust_v2([], v2)
        assert len(result.labels) == 0
        assert result.pos_trusted.shape == (0,)
        assert result.vel_trusted.shape == (0,)
        assert result.pos_score.shape == (0,)
        assert result.vel_score.shape == (0,)
        assert result.signals.shape == (0, 10)
        assert result.n_high == 0
        assert result.n_pos_only == 0
        assert result.n_vel_only == 0
        assert result.n_low == 0
