"""Regression tests for position interpolation (interp_pos).

Tests boundary handling, gap rejection, and NaN handling.
"""

import pytest
import math
from dataclasses import dataclass

from data_pipeline.parsers import PosRow, interp_pos, _bisect_pair


class TestInterpolation:
    """Position interpolation tests."""

    def test_exact_boundary_timestamps(self):
        """Test that exact boundary timestamps are handled correctly.

        This was bug 1.1: timestamps exactly at the first/last sample
        were incorrectly rejected.
        """
        rows = [
            PosRow(1.0, 10.0, 20.0, 100.0, 1),
            PosRow(2.0, 10.1, 20.1, 101.0, 1),
            PosRow(3.0, 10.2, 20.2, 102.0, 1),
        ]
        times = [r.utc_s for r in rows]

        # Query exactly at first sample.
        result = interp_pos(rows, 1.0, max_gap_s=1.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert abs(lat - 10.0) < 1e-6
        assert abs(lon - 20.0) < 1e-6

        # Query exactly at last sample.
        result = interp_pos(rows, 3.0, max_gap_s=1.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert abs(lat - 10.2) < 1e-6
        assert abs(lon - 20.2) < 1e-6

    def test_interior_interpolation(self):
        """Test interpolation at interior points.

        ``max_gap_s`` is the bracket-span ceiling, so it must be >= the
        actual gap between bracketing rows (2.0 s here) for the query to
        be admitted.
        """
        rows = [
            PosRow(0.0, 0.0, 0.0, 0.0, 1),
            PosRow(2.0, 10.0, 20.0, 100.0, 1),
        ]
        times = [r.utc_s for r in rows]

        # Query at t=1.0, midpoint.
        result = interp_pos(rows, 1.0, max_gap_s=2.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert abs(lat - 5.0) < 1e-6
        assert abs(lon - 10.0) < 1e-6

    def test_bracket_span_rejection_at_midpoint(self):
        """A query in the middle of a long bracket must reject even when
        each side individually fits under max_gap_s.

        Earlier policy used a per-side check (``query-a < gap and b-query < gap``)
        which silently admitted tunnels whenever the query source near the
        midpoint. The bracket-span check (``b - a > max_gap_s``) closes that
        hole.
        """
        rows = [
            PosRow(0.0, 0.0, 0.0, 0.0, 1),
            PosRow(4.0, 10.0, 20.0, 100.0, 1),
        ]
        times = [r.utc_s for r in rows]
        # Bracket span = 4 s, query at midpoint (each side = 2 s),
        # max_gap_s = 2 s — must reject.
        assert interp_pos(rows, 2.0, max_gap_s=2.0, times=times) is None

    def test_gap_rejection(self):
        """Test that queries outside the gap threshold are rejected."""
        rows = [
            PosRow(0.0, 0.0, 0.0, 0.0, 1),
            PosRow(10.0, 10.0, 20.0, 100.0, 1),
        ]
        times = [r.utc_s for r in rows]

        # Gap is 10 seconds, threshold is 2 seconds.
        result = interp_pos(rows, 5.0, max_gap_s=2.0, times=times)
        assert result is None

    def test_nan_position_propagates(self):
        """Test that NaN positions in a bracketing row propagate to the result.

        interp_pos does not skip NaN rows; callers are expected to only pass
        valid Post-processing rows (NaN lat/lon won't appear in real The external solver .pos files).
        """
        rows = [
            PosRow(0.0, 0.0, 0.0, 0.0, 1),
            PosRow(1.0, float("nan"), float("nan"), float("nan"), 1),
            PosRow(2.0, 10.0, 20.0, 100.0, 1),
        ]
        times = [r.utc_s for r in rows]

        result = interp_pos(rows, 1.5, max_gap_s=2.0, times=times)
        assert result is not None
        lat, lon, h = result
        assert not math.isfinite(lat)  # NaN propagates from bracketing row

    def test_outside_data_range(self):
        """Test queries outside the data time range."""
        rows = [
            PosRow(1.0, 10.0, 20.0, 100.0, 1),
            PosRow(3.0, 10.2, 20.2, 102.0, 1),
        ]
        times = [r.utc_s for r in rows]

        # Before first sample.
        result = interp_pos(rows, 0.5, max_gap_s=1.0, times=times)
        assert result is None

        # After last sample.
        result = interp_pos(rows, 4.0, max_gap_s=1.0, times=times)
        assert result is None


class TestBisectPair:
    """Test the _bisect_pair helper function."""

    def test_bracketing_pair(self):
        """Test finding a bracketing pair for an interior query."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _bisect_pair(times, 3.5)
        assert result == (2, 3)  # Indices, 0-indexed.

    def test_exact_match_first(self):
        """Test exact match at first sample."""
        times = [1.0, 2.0, 3.0]
        result = _bisect_pair(times, 1.0)
        assert result == (0, 0)  # Degenerate pair at boundary.

    def test_exact_match_last(self):
        """Test exact match at last sample."""
        times = [1.0, 2.0, 3.0]
        result = _bisect_pair(times, 3.0)
        assert result == (2, 2)

    def test_outside_range(self):
        """Test query outside data range."""
        times = [1.0, 2.0, 3.0]
        result = _bisect_pair(times, 0.5)
        assert result is None

        result = _bisect_pair(times, 3.5)
        assert result is None
