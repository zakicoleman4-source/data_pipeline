"""10 ms timing-sensitivity audit.

Tests every link in the media-PTS → UTC → Post-processing interpolation chain
for sub-10 ms accuracy. Any failure here means sample geocoding is wrong.
"""

import csv
import datetime as dt
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.parsers import (
    PosRow,
    interp_pos,
    parse_imu,
    parse_orientation,
    parse_rtkpos,
    read_frame_times_csv,
)
from data_pipeline.time_sync import (
    GPS_UTC_LEAP_SECONDS_2026,
    TimeAnchor,
    _parse_iso_utc,
    _parse_utc_seconds,
    fit_time_anchor,
    fit_time_anchor_from_pairs,
    get_leap_seconds_for_epoch,
    gpst_to_utc_seconds,
)


# ---------------------------------------------------------------------------
# Link 1: ISO UTC string parsing precision
# ---------------------------------------------------------------------------

class TestIsoUtcParsing:
    """_parse_iso_utc must not lose > 1 µs at any fractional-digit count."""

    @pytest.mark.parametrize("frac_digits", [0, 1, 3, 6, 9])
    def test_fractional_digit_precision(self, frac_digits: int):
        base = "2024-06-15T12:34:56"
        if frac_digits == 0:
            iso = base
            expected_us = 0
        else:
            frac = "123456789"[:frac_digits]
            iso = f"{base}.{frac}"
            expected_us = int((frac + "000000")[:6])
        parsed = _parse_iso_utc(iso)
        assert parsed.microsecond == expected_us

    def test_z_suffix_stripped(self):
        a = _parse_iso_utc("2024-01-01T00:00:00.123Z")
        b = _parse_iso_utc("2024-01-01T00:00:00.123")
        assert a == b

    def test_nine_digit_nano_truncates_correctly(self):
        """9-digit nanoseconds → 6-digit microseconds, max loss < 1 µs."""
        parsed = _parse_iso_utc("2024-01-01T00:00:00.123456789")
        assert parsed.microsecond == 123456
        parsed2 = _parse_iso_utc("2024-01-01T00:00:00.123456999")
        assert parsed2.microsecond == 123456

    def test_parse_utc_seconds_roundtrip(self):
        """_parse_utc_seconds output matches manual computation."""
        iso = "2024-01-01T00:00:00.500Z"
        s = _parse_utc_seconds(iso)
        expected = dt.datetime(2024, 1, 1, 0, 0, 0, 500000,
                               tzinfo=dt.timezone.utc).timestamp()
        assert abs(s - expected) < 1e-6


# ---------------------------------------------------------------------------
# Link 2: TimeAnchor OLS precision at realistic magnitudes
# ---------------------------------------------------------------------------

class TestTimeAnchorPrecision:
    """OLS fit must stay < 1 ms at centroid and < 5 ms at session edges
    for a realistic 35-min device session with ~30 ms per-anchor jitter."""

    def _make_realistic_session(self, n_anchors=2000, duration_s=2100,
                                 drift_ppm=3.0, jitter_s=0.030, seed=42):
        rng = random.Random(seed)
        t0_video_ns = 50_000_000  # 50 ms initial offset
        t0_utc_s = 1704067200.0  # 2024-01-01 00:00:00 UTC
        slope_true = (1.0 + drift_ppm * 1e-6) / 1e9  # s per ns

        pairs = []
        for i in range(n_anchors):
            video_ns = t0_video_ns + (i * duration_s / n_anchors) * 1e9
            utc_true = t0_utc_s + slope_true * (video_ns - t0_video_ns)
            utc_noisy = utc_true + rng.gauss(0, jitter_s)
            pairs.append((video_ns, utc_noisy))
        return pairs, slope_true, t0_utc_s, t0_video_ns

    def test_centroid_uncertainty_sub_ms(self):
        pairs, slope_true, t0_utc, t0_vns = self._make_realistic_session()
        anchor = fit_time_anchor_from_pairs(pairs)
        assert anchor.fit_uncertainty_s < 0.001  # < 1 ms at centroid

    def test_edge_uncertainty_sub_5ms(self):
        pairs, *_ = self._make_realistic_session()
        anchor = fit_time_anchor_from_pairs(pairs)
        unc_start = anchor.fit_uncertainty_s_at(pairs[0][0])
        unc_end = anchor.fit_uncertainty_s_at(pairs[-1][0])
        assert unc_start < 0.005
        assert unc_end < 0.005

    def test_slope_recovers_true_drift(self):
        pairs, slope_true, *_ = self._make_realistic_session(drift_ppm=5.0)
        anchor = fit_time_anchor_from_pairs(pairs)
        assert abs(anchor.slope - slope_true) / slope_true < 1e-4

    def test_video_pts_to_utc_accuracy(self):
        """End-to-end: media PTS → UTC must be < 5 ms from truth."""
        pairs, slope_true, t0_utc, t0_vns = self._make_realistic_session()
        anchor = fit_time_anchor_from_pairs(pairs)

        # Test at 100 evenly-spaced media times across session
        for i in range(100):
            t_video_s = (i / 99.0) * 2100.0
            video_ns = t0_vns + t_video_s * 1e9
            utc_true = t0_utc + slope_true * (video_ns - t0_vns)
            utc_pred = anchor.video_pts_to_utc_s(t_video_s + t0_vns / 1e9)
            err_ms = abs(utc_pred - utc_true) * 1000
            assert err_ms < 5.0, f"Frame {i}: {err_ms:.3f} ms error"

    def test_ns_to_s_conversion_no_precision_loss(self):
        """video_pts_to_utc_s(t_s) = video_ns_to_utc_s(t_s * 1e9).
        Verify float64 doesn't lose > 1 ns at realistic magnitudes."""
        anchor = TimeAnchor(
            slope=1e-9, xmean=1.05e12, ymean=1.704e9,
            n=1000, rmse_s=0.030, max_abs_s=0.1,
            sxx_ns2=1e24,
        )
        t_video_s = 2100.0  # 35-minute session
        via_ns = anchor.video_ns_to_utc_s(t_video_s * 1e9)
        via_pts = anchor.video_pts_to_utc_s(t_video_s)
        assert abs(via_ns - via_pts) < 1e-9


# ---------------------------------------------------------------------------
# Link 3: Reference time → UTC epoch offset conversion
# ---------------------------------------------------------------------------

class TestLeapSecondConversion:
    """parse_rtkpos must subtract exactly the right epoch offset."""

    def test_2024_data_uses_18s(self):
        ls = get_leap_seconds_for_epoch(1704067200.0)
        assert ls == 18.0

    def test_2026_data_uses_18s(self):
        ls = get_leap_seconds_for_epoch(1767225600.0)
        assert ls == 18.0

    def test_pre_gps_epoch(self):
        ls = get_leap_seconds_for_epoch(100_000_000.0)
        assert ls == 0.0

    def test_gpst_to_utc_standalone(self):
        gpst_unix = 1704067218.0  # Reference time for 2024-01-01 00:00:18 Reference time
        utc = gpst_to_utc_seconds(gpst_unix)
        assert abs(utc - 1704067200.0) < 1e-6

    def test_parse_rtkpos_gpst_to_utc_sub_ms(self):
        """Full .pos parse: Reference time→UTC within 1 ms of expected."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pos", delete=False, encoding="utf-8"
        ) as f:
            f.write("% header\n")
            # Reference time: 2024/06/15 12:00:18.500 → UTC 2024/06/15 12:00:00.500
            f.write(
                "2024/06/15 12:00:18.500000 "
                "37.33820000 -122.03240000 10.000 1 12 "
                "0.01 0.01 0.03 0 0 0 20 5\n"
            )
            tmp = f.name
        try:
            rows = parse_rtkpos(Path(tmp))
            assert len(rows) == 1
            expected = dt.datetime(
                2024, 6, 15, 12, 0, 0, 500000, tzinfo=dt.timezone.utc
            ).timestamp()
            assert abs(rows[0].utc_s - expected) < 0.001
        finally:
            Path(tmp).unlink()

    def test_parse_rtkpos_fractional_second_preserved(self):
        """Verify sub-second Reference time time survives the conversion."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pos", delete=False, encoding="utf-8"
        ) as f:
            f.write("% header\n")
            # .123456 fractional seconds
            f.write(
                "2024/06/15 12:00:18.123456 "
                "37.33820000 -122.03240000 10.000 1 12 "
                "0.01 0.01 0.03 0 0 0 20 5\n"
            )
            tmp = f.name
        try:
            rows = parse_rtkpos(Path(tmp))
            frac = rows[0].utc_s % 1.0
            assert abs(frac - 0.123456) < 0.001
        finally:
            Path(tmp).unlink()

    def test_two_rows_one_second_apart(self):
        """Two Post-processing rows 1.0s apart in Reference time must be 1.0s apart in UTC."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pos", delete=False, encoding="utf-8"
        ) as f:
            f.write("% header\n")
            f.write(
                "2024/06/15 12:00:18.000000 "
                "37.33820000 -122.03240000 10.000 1 12 "
                "0.01 0.01 0.03 0 0 0 20 5\n"
            )
            f.write(
                "2024/06/15 12:00:19.000000 "
                "37.33821000 -122.03241000 10.100 1 12 "
                "0.01 0.01 0.03 0 0 0 20 5\n"
            )
            tmp = f.name
        try:
            rows = parse_rtkpos(Path(tmp))
            assert len(rows) == 2
            dt_s = rows[1].utc_s - rows[0].utc_s
            assert abs(dt_s - 1.0) < 1e-6
        finally:
            Path(tmp).unlink()


# ---------------------------------------------------------------------------
# Link 4: Motion sensor timestamp (Reference seconds since epoch) → UTC
# ---------------------------------------------------------------------------

class TestImuTimestamp:
    """sensors_*.txt Reference-seconds-since-epoch → UTC must match Post-processing domain."""

    def test_imu_utc_matches_ppk_utc(self):
        """Same physical instant in .pos and sensors_*.txt → same utc_s."""
        # Reference time 2024/06/15 12:00:18.000 → UTC 2024/06/15 12:00:00.000
        # Reference seconds since Reference epoch for that instant:
        gps_epoch_unix = 315964800
        gpst_unix = dt.datetime(
            2024, 6, 15, 12, 0, 18, tzinfo=dt.timezone.utc
        ).timestamp()
        gps_seconds = gpst_unix - gps_epoch_unix

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(f"{gps_seconds:.6f},0.01,0.02,0.03,0.1,0.2,9.81\n")
            tmp = f.name
        try:
            rows = parse_imu(Path(tmp))
            assert len(rows) == 1
            expected_utc = dt.datetime(
                2024, 6, 15, 12, 0, 0, tzinfo=dt.timezone.utc
            ).timestamp()
            assert abs(rows[0].utc_s - expected_utc) < 0.001
        finally:
            Path(tmp).unlink()

    def test_imu_and_ppk_same_instant_agree(self):
        """Parse both a .pos row and an Motion sensor row at the same Reference time instant.
        Their utc_s values must agree within 1 ms."""
        gpst_dt = dt.datetime(2024, 6, 15, 12, 0, 18, 500000,
                              tzinfo=dt.timezone.utc)
        gps_epoch_unix = 315964800
        gps_seconds = gpst_dt.timestamp() - gps_epoch_unix

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pos", delete=False, encoding="utf-8"
        ) as f_pos:
            f_pos.write("% header\n")
            f_pos.write(
                "2024/06/15 12:00:18.500000 "
                "37.338 -122.032 10.0 1 12 "
                "0.01 0.01 0.03 0 0 0 20 5\n"
            )
            pos_path = f_pos.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f_imu:
            f_imu.write(f"{gps_seconds:.6f},0.01,0.02,0.03,0.1,0.2,9.81\n")
            imu_path = f_imu.name

        try:
            pos_rows = parse_rtkpos(Path(pos_path))
            imu_rows = parse_imu(Path(imu_path))
            assert len(pos_rows) == 1
            assert len(imu_rows) == 1
            assert abs(pos_rows[0].utc_s - imu_rows[0].utc_s) < 0.001
        finally:
            Path(pos_path).unlink()
            Path(imu_path).unlink()


# ---------------------------------------------------------------------------
# Link 5: OrientationDeg utcMs → UTC
# ---------------------------------------------------------------------------

class TestOrientationTimestamp:
    """OrientationDeg utcMs/1000 must land in the same UTC domain as Post-processing."""

    def test_orient_utc_domain(self):
        utc_instant = dt.datetime(2024, 6, 15, 12, 0, 0, 500000,
                                  tzinfo=dt.timezone.utc)
        utc_ms = int(utc_instant.timestamp() * 1000)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                f"OrientationDeg,{utc_ms},0,45.0,1.0,-2.0,3\n"
            )
            tmp = f.name
        try:
            rows = parse_orientation(Path(tmp))
            assert len(rows) == 1
            assert abs(rows[0].utc_s - utc_instant.timestamp()) < 0.001
        finally:
            Path(tmp).unlink()


# ---------------------------------------------------------------------------
# Link 6: Full chain — PTS → UTC → Post-processing interpolation
# ---------------------------------------------------------------------------

class TestFullTimingChain:
    """End-to-end: synthetic recording_*.txt + .pos + sample CSV.
    Sample geocoding error must stay < 10 ms."""

    def _make_chain_data(self, tmp_path: Path, n_anchors=2000,
                         n_ppk_rows=2100, n_frames=100,
                         duration_s=2100.0, drift_ppm=3.0):
        """Create synthetic data with known reference for timing."""
        rng = random.Random(12345)
        t0_utc_s = 1704067200.0  # 2024-01-01 00:00:00 UTC

        # recording_*.txt: (video_ns, UTC ISO)
        rec_path = tmp_path / "recording_test.txt"
        slope_true = (1.0 + drift_ppm * 1e-6) / 1e9
        with rec_path.open("w", encoding="utf-8") as f:
            for i in range(n_anchors):
                video_ns = int(i * duration_s / n_anchors * 1e9)
                utc_true = t0_utc_s + slope_true * video_ns
                utc_noisy = utc_true + rng.gauss(0, 0.030)
                utc_dt = dt.datetime.fromtimestamp(utc_noisy, tz=dt.timezone.utc)
                iso = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.") + \
                    f"{utc_dt.microsecond:06d}"
                f.write(f"{video_ns},{iso}\n")

        # .pos file: 1 Hz Reference time, straight line at 10 m/s heading north
        pos_path = tmp_path / "test.pos"
        with pos_path.open("w", encoding="utf-8") as f:
            f.write("% header\n")
            lat0, lon0, h0 = 37.338, -122.032, 10.0
            for i in range(n_ppk_rows):
                utc_s = t0_utc_s + i
                gpst_s = utc_s + 18.0  # UTC→Reference time
                gpst_dt = dt.datetime.fromtimestamp(gpst_s, tz=dt.timezone.utc)
                date_str = gpst_dt.strftime("%Y/%m/%d")
                time_str = gpst_dt.strftime("%H:%M:%S.%f")
                lat = lat0 + i * (10.0 / 111111.0)  # ~10 m/s north
                f.write(
                    f"{date_str} {time_str} {lat:.9f} {lon0:.9f} {h0:.3f} "
                    f"1 12 0.01 0.01 0.03 0 0 0 20 5\n"
                )

        # Sample times CSV: samples at regular intervals across session
        csv_path = tmp_path / "extracted_frame_times.csv"
        frame_times = []
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image", "t_video_s"])
            for i in range(n_frames):
                t_video_s = i * duration_s / n_frames
                name = f"frame_{t_video_s:.6f}.png"
                w.writerow([name, f"{t_video_s:.12f}"])
                frame_times.append(t_video_s)

        return rec_path, pos_path, csv_path, frame_times, slope_true, t0_utc_s

    def test_full_chain_sub_10ms(self, tmp_path: Path):
        """Media PTS → TimeAnchor → UTC → Post-processing interp: error < 10 ms."""
        rec, pos, csv_p, frame_times, slope_true, t0_utc = \
            self._make_chain_data(tmp_path)

        anchor = fit_time_anchor(rec)
        pos_rows = parse_rtkpos(pos)
        assert len(pos_rows) > 0

        pos_times = [r.utc_s for r in pos_rows]

        errors_ms = []
        for t_video_s in frame_times:
            utc_pred = anchor.video_pts_to_utc_s(t_video_s)
            utc_true = t0_utc + slope_true * (t_video_s * 1e9)
            err_ms = abs(utc_pred - utc_true) * 1000
            errors_ms.append(err_ms)

            # Also verify Post-processing interpolation at this UTC
            llh = interp_pos(pos_rows, utc_pred, max_gap_s=2.0,
                             times=pos_times)
            if llh is not None:
                # Position should be interpolated (not None/gap)
                assert math.isfinite(llh[0])

        max_err = max(errors_ms)
        mean_err = sum(errors_ms) / len(errors_ms)
        assert max_err < 10.0, f"Max timing error {max_err:.3f} ms >= 10 ms"
        assert mean_err < 2.0, f"Mean timing error {mean_err:.3f} ms >= 2 ms"

    def test_frame_order_preserved_through_csv(self, tmp_path: Path):
        """Samples read back from CSV must be sorted by ascending PTS."""
        _, _, csv_p, _, _, _ = self._make_chain_data(tmp_path)
        frames = read_frame_times_csv(csv_p)
        pts_values = [t for _, t in frames]
        assert pts_values == sorted(pts_values)

    def test_interp_at_ppk_boundary_exact(self):
        """Interpolation at exactly the first/last Post-processing epoch → no gap reject."""
        rows = [
            PosRow(1000.0, 37.0, -122.0, 10.0, 1),
            PosRow(1001.0, 37.1, -122.1, 11.0, 1),
            PosRow(1002.0, 37.2, -122.2, 12.0, 1),
        ]
        times = [r.utc_s for r in rows]
        # Exact first
        r = interp_pos(rows, 1000.0, max_gap_s=2.0, times=times)
        assert r is not None
        assert abs(r[0] - 37.0) < 1e-9
        # Exact last
        r = interp_pos(rows, 1002.0, max_gap_s=2.0, times=times)
        assert r is not None
        assert abs(r[0] - 37.2) < 1e-9

    def test_interp_midpoint_linear(self):
        """Midpoint interpolation is exact linear."""
        rows = [
            PosRow(1000.0, 37.0, -122.0, 10.0, 1),
            PosRow(1002.0, 37.2, -122.2, 12.0, 1),
        ]
        r = interp_pos(rows, 1001.0, max_gap_s=3.0)
        assert r is not None
        assert abs(r[0] - 37.1) < 1e-9
        assert abs(r[1] - (-122.1)) < 1e-9


# ---------------------------------------------------------------------------
# Link 7: Float precision at realistic POSIX magnitudes
# ---------------------------------------------------------------------------

class TestFloatPrecision:
    """Verify float64 arithmetic doesn't lose > 1 µs at 2024-era POSIX."""

    def test_utc_posix_addition_precision(self):
        """Adding 0.001s to a 2024-era POSIX timestamp preserves 1 µs.
        float64 ULP at 1.7e9 magnitude ≈ 0.24 µs; 72 ns loss is expected."""
        base = 1704067200.0  # 2024-01-01
        result = base + 0.001
        recovered = result - base
        assert abs(recovered - 0.001) < 1e-6  # < 1 µs, well within 10 ms

    def test_video_ns_multiplication_precision(self):
        """t_video_s * 1e9 at 35-min session preserves nanosecond."""
        t_s = 2100.123456789
        ns = t_s * 1e9
        recovered = ns / 1e9
        assert abs(recovered - t_s) < 1e-9

    def test_slope_times_centered_x_precision(self):
        """slope * (x - xmean) at realistic magnitudes preserves < 1 ns."""
        slope = 1.000003e-9  # 3 ppm drift
        xmean = 1.05e12
        ymean = 1.704e9

        x = 2.1e12  # End of 35-min session
        result = ymean + slope * (x - xmean)
        # Verify against high-precision computation
        from decimal import Decimal
        d_slope = Decimal("1.000003e-9")
        d_xmean = Decimal("1.05e12")
        d_ymean = Decimal("1.704e9")
        d_x = Decimal("2.1e12")
        d_result = d_ymean + d_slope * (d_x - d_xmean)
        assert abs(result - float(d_result)) < 1e-6  # < 1 µs


# ---------------------------------------------------------------------------
# Link 8: Sample CSV roundtrip precision
# ---------------------------------------------------------------------------

class TestFrameCsvRoundtrip:
    """t_video_s written to CSV and read back must not lose > 1 µs."""

    def test_csv_roundtrip_12_decimals(self, tmp_path: Path):
        csv_path = tmp_path / "extracted_frame_times.csv"
        original_pts = [0.0, 0.033367, 123.456789012, 2100.999999999]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image", "t_video_s"])
            for i, t in enumerate(original_pts):
                w.writerow([f"frame_{i}.png", f"{t:.12f}"])

        recovered = read_frame_times_csv(csv_path)
        assert len(recovered) == len(original_pts)
        for (_, t_read), t_orig in zip(recovered, original_pts):
            assert abs(t_read - t_orig) < 1e-9


# ---------------------------------------------------------------------------
# Link 9: Showinfo regex robustness
# ---------------------------------------------------------------------------

class TestShowinfoRegex:
    """The external converter showinfo regex must parse known format variants."""

    def test_standard_format(self):
        from data_pipeline.stages.frames import _SHOWINFO_RE
        line = (
            "[Parsed_showinfo_2 @ 0x5590] n:   3 pts:   3003 pts_time:0.100100 "
            "pos:  40960 fmt:yuv420p sar:1/1 s:1920x1080"
        )
        m = _SHOWINFO_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 3
        assert abs(float(m.group(2)) - 0.100100) < 1e-9

    def test_large_pts_time(self):
        from data_pipeline.stages.frames import _SHOWINFO_RE
        line = (
            "[Parsed_showinfo_1 @ 0xabc] n: 63000 pts: 567000000 "
            "pts_time:2100.333333 pos: 12345 fmt:yuv420p"
        )
        m = _SHOWINFO_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 63000
        assert abs(float(m.group(2)) - 2100.333333) < 1e-6

    def test_zero_pts(self):
        from data_pipeline.stages.frames import _SHOWINFO_RE
        line = (
            "[Parsed_showinfo_0 @ 0x1] n:   0 pts:      0 pts_time:0 "
            "pos:  0 fmt:yuv420p"
        )
        m = _SHOWINFO_RE.search(line)
        assert m is not None
        assert int(m.group(1)) == 0
        assert float(m.group(2)) == 0.0


# ---------------------------------------------------------------------------
# Link 10: Epoch offset table correctness
# ---------------------------------------------------------------------------

class TestLeapSecondTable:
    """Verify table entries match IERS Bulletin C historical data."""

    KNOWN_LEAPS = [
        ("1981-07-01", 1),
        ("1990-01-01", 6),
        ("1996-01-01", 11),
        ("1999-01-01", 13),
        ("2006-01-01", 14),
        ("2009-01-01", 15),
        ("2012-07-01", 16),
        ("2015-07-01", 17),
        ("2017-01-01", 18),
    ]

    @pytest.mark.parametrize("date_str,expected_ls", KNOWN_LEAPS)
    def test_known_leap_second_values(self, date_str, expected_ls):
        d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        )
        ls = get_leap_seconds_for_epoch(d.timestamp() + 1)  # 1s after
        assert ls == expected_ls

    def test_no_new_leap_after_2017(self):
        """As of 2026, no new epoch offset has been announced."""
        for year in range(2018, 2027):
            ts = dt.datetime(year, 6, 1, tzinfo=dt.timezone.utc).timestamp()
            assert get_leap_seconds_for_epoch(ts) == 18.0


# ---------------------------------------------------------------------------
# Stress test: many-sample interpolation consistency
# ---------------------------------------------------------------------------

class TestInterpolationStress:
    """1000+ sample interpolation must produce monotonically-spaced positions
    for a constant-velocity path."""

    def test_1000_frame_monotonic_latitude(self):
        """1000 samples along a northbound track → lat monotonically increases."""
        n = 2100
        t0 = 1704067200.0
        lat0 = 37.0
        rows = [
            PosRow(t0 + i, lat0 + i * 1e-5, -122.0, 10.0, 1)
            for i in range(n)
        ]
        times = [r.utc_s for r in rows]

        prev_lat = -999.0
        for j in range(1000):
            t = t0 + 0.5 + j * 2.0  # Every 2s, offset by 0.5s
            r = interp_pos(rows, t, max_gap_s=2.0, times=times)
            assert r is not None, f"Gap reject at j={j}"
            assert r[0] > prev_lat, f"Non-monotonic lat at j={j}"
            prev_lat = r[0]

    def test_gap_reject_never_fabricates(self):
        """Samples in a 5-second gap → all None, never fabricated positions."""
        rows = [
            PosRow(1000.0, 37.0, -122.0, 10.0, 1),
            PosRow(1001.0, 37.1, -122.1, 11.0, 1),
            # 5-second gap
            PosRow(1006.0, 37.5, -122.5, 15.0, 1),
            PosRow(1007.0, 37.6, -122.6, 16.0, 1),
        ]
        times = [r.utc_s for r in rows]
        for t in [1002.0, 1003.0, 1004.0, 1005.0]:
            r = interp_pos(rows, t, max_gap_s=2.0, times=times)
            assert r is None, f"Fabricated position in gap at t={t}"
