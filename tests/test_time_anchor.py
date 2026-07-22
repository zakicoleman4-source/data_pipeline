"""Regression tests for TimeAnchor fitting and uncertainty calculation.

Tests the OLS regression that maps media timestamps to UTC. With sub-ms
accuracy requirements, any regression here immediately impacts sample accuracy.
"""

import pytest
import math
from pathlib import Path

from data_pipeline.time_sync import fit_time_anchor_from_pairs, TimeAnchor


class TestTimeAnchorBasic:
    """Basic TimeAnchor construction and properties."""

    def test_simple_fit(self):
        """Test a simple linear fit with no noise."""
        pairs = [(float(i), 1000.0 + 0.001 * i) for i in range(100)]
        anchor = fit_time_anchor_from_pairs(pairs, robust=False)

        assert anchor.n == 100
        assert abs(anchor.slope - 0.001) < 1e-6
        assert abs(anchor.rmse_s) < 1e-6  # Perfect fit.

    def test_fit_with_jitter(self):
        """Test fit with realistic per-anchor jitter (~30 ms)."""
        import random
        random.seed(42)

        pairs = []
        for i in range(100):
            # Use 1e12 ns steps (1000 s each) so the span is wide enough
            # for OLS to resolve 1 ppm drift against 30 ms jitter.
            x = float(i) * 1e12  # video_ns, ~27.5 hour span
            y_true = 1000.0 + (1e-9 + 1e-15) * x  # 1 ns/ns + 1 ppm drift
            y = y_true + random.gauss(0, 0.030)  # 30 ms jitter
            pairs.append((x, y))

        anchor = fit_time_anchor_from_pairs(pairs)
        assert anchor.n == 100
        # RMSE should be ~30 ms from jitter.
        assert 0.020 < anchor.rmse_s < 0.050
        # Drift should be ~1 ppm.
        assert abs(anchor.drift_ppm - 1.0) < 2.0

    def test_uncertainty_at_centroid(self):
        """Test fit_uncertainty_s at the centroid."""
        pairs = [(float(i) * 1e9, 1000.0 + i * 1e-9) for i in range(100)]
        anchor = fit_time_anchor_from_pairs(pairs, robust=False)

        # At the centroid, uncertainty = sigma / sqrt(n).
        centroid_unc = anchor.fit_uncertainty_s
        assert centroid_unc > 0
        assert centroid_unc < 1e-3  # Should be small for a perfect fit.

    def test_uncertainty_at_edges(self):
        """Test fit_uncertainty_s_at() away from centroid."""
        pairs = [(float(i) * 1e9, 1000.0 + i * 1e-9) for i in range(100)]
        anchor = fit_time_anchor_from_pairs(pairs, robust=False)

        x_min = pairs[0][0]
        x_max = pairs[-1][0]
        x_mid = (x_min + x_max) / 2

        unc_min = anchor.fit_uncertainty_s_at(x_min)
        unc_mid = anchor.fit_uncertainty_s_at(x_mid)
        unc_max = anchor.fit_uncertainty_s_at(x_max)

        # Uncertainty should grow away from centroid.
        assert unc_mid <= unc_min and unc_mid <= unc_max
        # But all should be reasonable (< 1 ms for 100 anchors).
        assert unc_max < 1e-3


class TestTimeAnchorRobustness:
    """Robustness against outliers."""

    def test_robust_outlier_rejection(self):
        """Test that robust=True rejects outliers."""
        pairs = [(float(i) * 1e9, 1000.0 + i * 1e-9) for i in range(100)]
        # Add a large outlier.
        pairs[50] = (pairs[50][0], pairs[50][1] + 1.0)  # +1 second outlier.

        anchor_robust = fit_time_anchor_from_pairs(pairs, robust=True)
        anchor_naive = fit_time_anchor_from_pairs(pairs, robust=False)

        # Robust fit should reject the outlier.
        assert anchor_robust.n_rejected > 0
        # RMSE should be lower with robust fit.
        assert anchor_robust.rmse_s < anchor_naive.rmse_s

    def test_cubic_improvement(self):
        """Test cubic RMSE improvement diagnostic."""
        pairs = [(float(i) * 1e9, 1000.0 + i * 1e-9) for i in range(100)]
        anchor = fit_time_anchor_from_pairs(pairs, robust=False)

        # For a truly linear fit, cubic improvement should be negligible.
        assert anchor.cubic_rmse_improvement_s < 1e-9


class TestTimeAnchorEdgeCases:
    """Edge cases and error handling."""

    def test_insufficient_points(self):
        """Test with too few points."""
        with pytest.raises(ValueError):
            fit_time_anchor_from_pairs([(1.0, 2.0)])  # Only 1 point.

    def test_identical_x_values(self):
        """Test with non-varying x (cannot fit)."""
        pairs = [(100.0, 1000.0 + i * 0.001) for i in range(10)]
        with pytest.raises(ValueError):
            fit_time_anchor_from_pairs(pairs, robust=False)

    def test_nan_handling(self):
        """Test that NaN pairs are handled (skipped by robust fitting)."""
        pairs = [
            (float(i) * 1e9, 1000.0 + i * 1e-9) if i != 50
            else (float(i) * 1e9, float("nan"))
            for i in range(100)
        ]
        anchor = fit_time_anchor_from_pairs(pairs, robust=True)
        # Should skip the NaN pair.
        assert anchor.n == 99
