"""Regression tests for yaw derivation from velocity vs. path.

Tests that yaw computed from Post-processing velocity and from path position
differences agree within expected bounds, and that the merge logic works.
"""

import pytest
import math

from data_pipeline.geo import heading_from_latlon, heading_from_enu


class TestHeadingComputation:
    """Tests for heading/yaw computation methods."""

    def test_heading_from_latlon_great_circle(self):
        """Test great-circle heading formula against known values."""
        # North: moving from (0, 0) to (1, 0).
        h = heading_from_latlon(0.0, 0.0, 1.0, 0.0)
        assert abs(h - 0.0) < 0.1  # Should be ~0° (north).

        # East: moving from (0, 0) to (0, 1).
        h = heading_from_latlon(0.0, 0.0, 0.0, 1.0)
        assert abs(h - 90.0) < 0.1  # Should be ~90° (east).

        # South: moving from (0, 0) to (-1, 0).
        h = heading_from_latlon(0.0, 0.0, -1.0, 0.0)
        assert abs(h - 180.0) < 0.1  # Should be ~180° (south).

        # West: moving from (0, 0) to (0, -1).
        h = heading_from_latlon(0.0, 0.0, 0.0, -1.0)
        assert abs(h - 270.0) < 0.1  # Should be ~270° (west).

    def test_heading_from_latlon_same_point(self):
        """Test heading for same start/end point."""
        h = heading_from_latlon(10.0, 20.0, 10.0, 20.0)
        assert math.isnan(h)

    def test_heading_from_latlon_nan_inputs(self):
        """Test heading with NaN inputs."""
        h = heading_from_latlon(float("nan"), 0.0, 1.0, 0.0)
        assert math.isnan(h)

        h = heading_from_latlon(0.0, float("nan"), 0.0, 1.0)
        assert math.isnan(h)

    def test_heading_from_enu_cardinal_directions(self):
        """Test Local-frame heading for cardinal directions."""
        # North: (0, 1, 0)
        h = heading_from_enu(0.0, 1.0)
        assert abs(h - 0.0) < 0.1

        # East: (1, 0, 0)
        h = heading_from_enu(1.0, 0.0)
        assert abs(h - 90.0) < 0.1

        # South: (0, -1, 0)
        h = heading_from_enu(0.0, -1.0)
        assert abs(h - 180.0) < 0.1

        # West: (-1, 0, 0)
        h = heading_from_enu(-1.0, 0.0)
        assert abs(h - 270.0) < 0.1

    def test_heading_from_enu_zero_vector(self):
        """Test Local-frame heading for zero/small vector."""
        h = heading_from_enu(0.0, 0.0)
        assert math.isnan(h)

        # Very small vector should also return NaN.
        h = heading_from_enu(1e-10, 1e-10)
        assert math.isnan(h)


class TestVelocityVsTrajectory:
    """Test consistency between velocity-derived and path-derived yaw."""

    def test_velocity_accuracy_vs_trajectory(self):
        """Velocity-derived yaw should be lower-noise than path finite-diff.

        This is a qualitative test: we expect velocity-derived yaw to have
        better SNR because it comes from Rate-signal (lower noise) rather than
        position differences (position noise amplified by differentiation).
        """
        # This is hard to test without access to real Post-processing data.
        # For now, we just ensure the functions exist and return sensible values.
        from data_pipeline.geo import heading_from_enu
        from data_pipeline.parsers import PosRow

        # Simulated velocity: 10 m/s north, 5 m/s east.
        # Heading should be atan2(east, north) = atan2(5, 10) ≈ 26.6°.
        h = heading_from_enu(5.0, 10.0)
        assert 25.0 < h < 30.0

    def test_yaw_merge_logic(self):
        """Test that velocity yaw is preferred, path yaw is fallback."""
        from data_pipeline.stages.georef import _merge_yaw_streams

        yaws_vel = [0.0, 90.0, float("nan"), 270.0]
        yaws_traj = [10.0, 100.0, 180.0, 280.0]

        merged = _merge_yaw_streams(yaws_vel, yaws_traj)

        assert merged[0] == 0.0  # Velocity preferred.
        assert merged[1] == 90.0
        assert merged[2] == 180.0  # Fallback to path where velocity is NaN.
        assert merged[3] == 270.0
