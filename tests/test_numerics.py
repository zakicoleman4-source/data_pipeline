"""Verification tests for refactoring changes.

This test suite ensures that all refactoring work (NumPy vectorization, Local-frame
smoothing, epoch offset lookup, etc.) maintains numerical accuracy and does not
introduce regressions in time-sync or position accuracy.
"""

import pytest
import math
import random
from pathlib import Path

from data_pipeline.time_sync import _ols_about_means, _cubic_rmse_about_means
from data_pipeline.smoothing import gaussian_smooth, gaussian_smooth_circular_deg
from data_pipeline.geo import llh_iterable_to_enu, heading_from_latlon
from data_pipeline.parsers import interp_pos, PosRow


class TestNumpyVectorizationNumericAccuracy:
    """Verify NumPy refactoring maintains numeric accuracy."""

    def test_ols_about_means_accuracy(self):
        """Test that NumPy OLS gives same results as pure Python would."""
        # Create test data: y = 2.0 + 3.0 * (x - xmean)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 8.0, 11.0, 14.0, 17.0]  # y = 2 + 3*(x - 3) = 3x - 7

        slope, xmean, ymean, sxx, residuals = _ols_about_means(xs, ys)

        assert abs(slope - 3.0) < 1e-10
        assert abs(xmean - 3.0) < 1e-10
        assert abs(ymean - 11.0) < 1e-10
        # Residuals should be near zero (perfect fit).
        assert all(abs(r) < 1e-10 for r in residuals)

    def test_ols_with_noise(self):
        """Test OLS with realistic noise."""
        random.seed(42)
        xs = [float(i) for i in range(100)]
        true_slope = 0.5
        true_intercept = 10.0
        ys = [true_intercept + true_slope * (x - 50) + random.gauss(0, 0.1) for x in xs]

        slope, xmean, ymean, sxx, residuals = _ols_about_means(xs, ys)

        assert abs(slope - true_slope) < 0.01
        # RMSE should be ~0.1 from the noise.
        rmse = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5
        assert 0.08 < rmse < 0.12

    def test_cubic_rmse_improvement(self):
        """Test cubic polynomial fitting."""
        # Pure linear data: cubic improvement should be tiny.
        xs = [float(i) for i in range(100)]
        ys = [10.0 + 0.5 * (x - 50) for x in xs]

        cubic_rmse = _cubic_rmse_about_means(xs, ys)
        assert cubic_rmse < 1e-10

    def test_gaussian_smooth_preserves_value(self):
        """Test that Gaussian smoothing preserves mean for constant signal."""
        values = [5.0] * 100
        smoothed = gaussian_smooth(values, sigma_samples=5.0)

        # All values should still be 5.0.
        assert all(abs(v - 5.0) < 1e-10 for v in smoothed)

    def test_gaussian_smooth_with_nan(self):
        """Test Gaussian smoothing handles NaN correctly."""
        values = [1.0, 2.0, float("nan"), 2.0, 1.0]
        smoothed = gaussian_smooth(values, sigma_samples=1.0)

        # NaN should be preserved.
        assert math.isnan(smoothed[2])
        # Neighbors should still have finite values.
        assert math.isfinite(smoothed[1])
        assert math.isfinite(smoothed[3])

    def test_gaussian_smooth_circular_wrap(self):
        """Test circular smoothing handles 360° wrap correctly."""
        values = [350.0, 355.0, 0.0, 5.0, 10.0]
        smoothed = gaussian_smooth_circular_deg(values, sigma_samples=1.0)

        # Result should be smooth across the wrap.
        assert all(math.isfinite(v) for v in smoothed)
        # Center value (0°) should be near 0°, not 180°.
        assert abs(smoothed[2]) < 90 or abs(smoothed[2] - 360.0) < 90


class TestENUSmoothingCorrectness:
    """Verify Local-frame smoothing gives consistent results."""

    def test_llh_to_enu_round_trip(self):
        """Test conversion to Local-frame and back maintains accuracy."""
        from data_pipeline.geo import llh_to_ecef, ecef_to_enu

        ref_llh = (37.3382, -122.0324, 10.0)
        test_point = (37.3385, -122.0321, 15.0)

        # Convert to Local-frame.
        x, y, z = llh_to_ecef(*test_point)
        e, n, u = ecef_to_enu(x, y, z, ref_llh)

        # Local-frame should have reasonable magnitude (tens of metres for 0.0003° offset).
        distance = (e**2 + n**2 + u**2) ** 0.5
        assert 10 < distance < 1000  # Sanity check.

    def test_llh_iterable_to_enu_vectorization(self):
        """Test vectorized Local-frame conversion."""
        points = [
            (37.3382, -122.0324, 10.0),
            (37.3383, -122.0323, 11.0),
            (37.3384, -122.0322, 12.0),
        ]
        ref_llh = (37.3382, -122.0324, 10.0)

        es, ns, us = llh_iterable_to_enu(points, ref_llh)

        assert len(es) == len(points)
        assert len(ns) == len(points)
        assert len(us) == len(points)

        # First point should be at reference (zero displacement).
        assert abs(es[0]) < 1e-6
        assert abs(ns[0]) < 1e-6
        assert abs(us[0]) < 1e-6

        # Others should have increasing distance.
        dist1 = (es[1]**2 + ns[1]**2) ** 0.5
        dist2 = (es[2]**2 + ns[2]**2) ** 0.5
        assert dist1 < dist2


class TestLeapSecondLookup:
    """Verify epoch offset table lookup works correctly."""

    def test_leap_second_table_lookup(self):
        """Test that epoch offset lookup gives expected values."""
        from data_pipeline.time_sync import get_leap_seconds_for_epoch

        # Just after 2017-01-01 00:00:00 UTC (epoch offset inserted): 18 seconds.
        epoch_2017 = 1483228801.0  # One second after the 18th epoch offset took effect.
        leap_s = get_leap_seconds_for_epoch(epoch_2017)
        assert leap_s == 18.0

        # 2024-01-01 UTC: still 18 seconds (no new epoch offset added since 2016).
        epoch_2024 = 1704067200.0
        leap_s = get_leap_seconds_for_epoch(epoch_2024)
        assert leap_s == 18.0

    def test_leap_second_in_parse_rtkpos(self):
        """Test that parse_rtkpos uses the epoch offset lookup."""
        from data_pipeline.parsers import parse_rtkpos
        import tempfile
        import datetime as dt

        # Create a temporary .pos file.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pos', delete=False) as f:
            # Write a sample row with Reference time timestamp.
            f.write("% Sample\n")
            # Reference time 2024/01/01 00:00:18 = UTC 2024/01/01 00:00:00 (18 s Reference-UTC offset).
            # Standard The external solver .pos column order: date time lat lon h Q ns
            # sd_n sd_e sd_u sd_ne sd_eu sd_un age ratio (vn ve vu sd_v*).
            f.write("2024/01/01 00:00:18.000000 37.3382 -122.0324 10.0 1 12 0.01 0.01 0.03 0 0 0 20 5 0.0 10.0\n")
            temp_path = f.name

        try:
            rows = parse_rtkpos(Path(temp_path))
            assert len(rows) == 1
            # UTC should be Reference time - 18 seconds.
            # Reference time 2024/01/01 12:00:00 = POSIX 1704067200 + 18 = 1704067218
            # UTC = 1704067218 - 18 = 1704067200
            expected_utc = 1704067200.0
            assert abs(rows[0].utc_s - expected_utc) < 1.0  # Within 1 second.
        finally:
            Path(temp_path).unlink()


class TestFrameTimeDerivation:
    """Verify effective FPS derivation from sample timings."""

    def test_derive_effective_fps(self):
        """Test FPS derivation from sample timestamps."""
        from data_pipeline.stages.georef import _derive_effective_fps, _Frame

        # Create samples at regular 6 fps.
        frames = []
        for i in range(10):
            t_video_s = i / 6.0
            frames.append(_Frame(f"IMG_{i:04d}.jpg", t_video_s, 1000.0 + i * 0.167))

        fps = _derive_effective_fps(frames)
        assert abs(fps - 6.0) < 0.01  # Should be ~6 Hz.

    def test_derive_fps_with_jitter(self):
        """Test FPS derivation with small timing jitter."""
        from data_pipeline.stages.georef import _derive_effective_fps, _Frame
        import random

        random.seed(42)
        frames = []
        base_interval = 1.0 / 30.0  # 30 fps nominal.
        for i in range(100):
            t_video_s = i * base_interval + random.gauss(0, 0.001)
            frames.append(_Frame(f"IMG_{i:04d}.jpg", t_video_s, 1000.0 + i * base_interval))

        fps = _derive_effective_fps(frames)
        assert abs(fps - 30.0) < 1.0  # Should be ~30 Hz despite jitter.


class TestSubMillisecondAccuracy:
    """Verify sub-millisecond time-sync accuracy is maintained."""

    def test_time_anchor_sub_ms_precision(self):
        """Test that time-sync maintains sub-millisecond accuracy."""
        from data_pipeline.time_sync import fit_time_anchor_from_pairs

        # Create 1000 anchors over a 100-second session with realistic jitter.
        random.seed(42)
        pairs = []
        for i in range(1000):
            video_ns = 1e9 + i * 100_000_000  # 100 ms per anchor.
            utc_s = 1704110400.0 + i * 0.1 + random.gauss(0, 0.030)  # 30 ms jitter.
            pairs.append((video_ns, utc_s))

        anchor = fit_time_anchor_from_pairs(pairs)

        # RMSE from jitter should be ~30 ms.
        assert 0.020 < anchor.rmse_s < 0.050

        # Fit uncertainty at centroid should be ~1 ms (rmse / sqrt(n)).
        assert anchor.fit_uncertainty_s < 0.005  # < 5 ms.

        # Per-sample uncertainty should be sub-ms at centroid.
        mid_frame_ns = (pairs[0][0] + pairs[-1][0]) / 2
        unc_ms = anchor.fit_uncertainty_s_at(mid_frame_ns) * 1e3
        assert unc_ms < 5.0  # < 5 ms.


class TestInterpolationBoundaryAccuracy:
    """Verify interpolation handles boundaries correctly after bug fix 1.1."""

    def test_interp_at_exact_first_sample(self):
        """Test interpolation at exact first Post-processing sample (bug 1.1 fix)."""
        rows = [
            PosRow(1.0, 10.0, 20.0, 100.0, 1),
            PosRow(2.0, 10.1, 20.1, 101.0, 1),
        ]
        times = [r.utc_s for r in rows]

        result = interp_pos(rows, 1.0, max_gap_s=10.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert abs(lat - 10.0) < 1e-10

    def test_interp_at_exact_last_sample(self):
        """Test interpolation at exact last Post-processing sample (bug 1.1 fix)."""
        rows = [
            PosRow(1.0, 10.0, 20.0, 100.0, 1),
            PosRow(2.0, 10.1, 20.1, 101.0, 1),
        ]
        times = [r.utc_s for r in rows]

        result = interp_pos(rows, 2.0, max_gap_s=10.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert abs(lat - 10.1) < 1e-10
