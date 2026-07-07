"""Time conversions for the pipeline.

Maps a per-sample device timestamp to absolute UTC. The timing file provides
(device-tick, UTC) pairs; these are noisy and drift slightly, so an
ordinary-least-squares fit over all pairs is used rather than any single pair.
Residual diagnostics are reported so callers can sanity-check the fit.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Reference-UTC epoch offset offset valid for 2017-... (no leap added by 2026).
GPS_UTC_LEAP_SECONDS_2026: float = 18.0

# Historical Reference-UTC epoch offset offsets: (POSIX timestamp of insertion, Reference-UTC after).
# Reference-UTC = 0 at Reference epoch (1980-01-06); increments by 1 for each epoch offset added to UTC.
# As of 2026 no new epoch offset has been announced beyond the 2017-01-01 insertion.
# Updated when IERS announces new epoch offset (typically Dec 31 or Jun 30).
# See https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.php
_LEAP_SECOND_TABLE: list[tuple[float, float]] = [
    (315964800.0,  0),   # 1980-01-06  Reference epoch, Reference-UTC = 0
    (362793600.0,  1),   # 1981-07-01
    (394329600.0,  2),   # 1982-07-01
    (425865600.0,  3),   # 1983-07-01
    (489024000.0,  4),   # 1985-07-01
    (567993600.0,  5),   # 1988-01-01
    (631152000.0,  6),   # 1990-01-01
    (662688000.0,  7),   # 1991-01-01
    (709948800.0,  8),   # 1992-07-01
    (741484800.0,  9),   # 1993-07-01
    (773020800.0, 10),   # 1994-07-01
    (820454400.0, 11),   # 1996-01-01
    (867715200.0, 12),   # 1997-07-01
    (915148800.0, 13),   # 1999-01-01
    (1136073600.0, 14),  # 2006-01-01
    (1230768000.0, 15),  # 2009-01-01
    (1341100800.0, 16),  # 2012-07-01
    (1435708800.0, 17),  # 2015-07-01
    (1483228800.0, 18),  # 2017-01-01  last known epoch offset
]


def get_leap_seconds_for_epoch(posix_timestamp_s: float) -> float:
    """Return the Reference-UTC epoch offset offset for a given POSIX timestamp.

    Args:
        posix_timestamp_s: POSIX timestamp (seconds since 1970-01-01 00:00:00 UTC)

    Returns:
        Epoch offset offset (float, typically 18-28).
        Defaults to the last known value if timestamp is beyond the table.

    Note:
        The epoch offset table is maintained manually. When IERS announces
        a new epoch offset, this table must be updated.
    """
    # Binary search for the largest timestamp <= the query.
    for ts, leap_s in reversed(_LEAP_SECOND_TABLE):
        if posix_timestamp_s >= ts:
            return leap_s
    # If before the earliest entry, use the first known value (shouldn't happen in practice).
    return _LEAP_SECOND_TABLE[0][1]


# For backward compatibility and convenience, export a function that accepts
# Python datetime objects as well.
def get_leap_seconds_for_datetime(dt_utc: dt.datetime) -> float:
    """Return the epoch offset offset for a given UTC datetime.

    Args:
        dt_utc: A datetime object in UTC.

    Returns:
        Epoch offset offset (float, typically 18-28).
    """
    return get_leap_seconds_for_epoch(dt_utc.timestamp())


# -----------------------------
# UTC parsing helpers
# -----------------------------


def _parse_iso_utc(iso: str) -> dt.datetime:
    """Parse an ISO 8601 UTC string with up to 9 fractional digits."""
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        base, frac = s.split(".", 1)
        s = f"{base}.{(frac + '000000')[:6]}"
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def _parse_utc_seconds(iso: str) -> float:
    """Parse an ISO 8601 UTC string into POSIX seconds (float)."""
    return _parse_iso_utc(iso).timestamp()


# -----------------------------
# TimeAnchor (regression fit)
# -----------------------------


@dataclass(frozen=True)
class TimeAnchor:
    """Best-fit affine map from media time (nanoseconds) to UTC (POSIX seconds).

    Internally we store the fit around the mean to stay numerically stable
    when ``video_ns`` values are O(1e12) and UTC seconds are O(1.7e9):

        utc_s(video_ns) = ymean + slope * (video_ns - xmean)

    ``slope`` has units of seconds-per-nanosecond and is essentially
    ``1e-9 * (1 + drift_ratio)``. ``rmse_s`` and ``max_abs_s`` are diagnostics
    of the per-anchor jitter against the fit, NOT the fit's own
    uncertainty - that is :attr:`fit_uncertainty_s`.

    ``cubic_rmse_improvement_s`` measures how much a degree-3 polynomial fit
    (over the same data) lowers RMSE compared to the linear fit. If this is
    much smaller than ``rmse_s`` then the source/system relationship really
    is linear and our linear model is sufficient; otherwise the linear fit
    is biased on top of jitter.
    """

    slope: float
    xmean: float
    ymean: float
    n: int
    rmse_s: float
    max_abs_s: float
    n_rejected: int = 0
    cubic_rmse_improvement_s: float = 0.0
    # Sum of squared centred x: sxx = sum((x_i - xmean)^2). Stored so the
    # fit's per-query uncertainty can be computed exactly without keeping
    # the full anchor list around.
    sxx_ns2: float = 0.0
    # Motion sensor-Post-processing cross-correlation clock offset correction (seconds).
    # Positive = media was lagging Signal (timestamps shifted forward).
    # 0.0 when uncalibrated or no Motion sensor data available.
    clock_offset_s: float = 0.0
    clock_offset_confidence: float = 0.0  # peak/noise ratio; >3 = reliable

    @property
    def drift_ppm(self) -> float:
        """Drift of the source clock relative to the system clock, in ppm."""
        return (self.slope * 1e9 - 1.0) * 1e6

    @property
    def sigma_residual_s(self) -> float:
        """Unbiased estimate of the per-anchor residual sigma (OLS, dof = n-2)."""
        if self.n <= 2 or self.rmse_s <= 0:
            return self.rmse_s
        # rmse_s was computed with /n (biased); rescale to /(n-2).
        return self.rmse_s * (self.n / (self.n - 2)) ** 0.5

    @property
    def fit_uncertainty_s(self) -> float:
        """1-sigma uncertainty of the fitted UTC AT THE FIT CENTROID.

        Standard OLS result at x = xmean: ``sigma / sqrt(n)`` with the
        unbiased residual sigma. This is the *best-case* uncertainty; the
        per-sample value should use :meth:`fit_uncertainty_s_at`.
        """
        if self.n < 3:
            return float("inf")
        return self.sigma_residual_s / (self.n ** 0.5)

    def fit_uncertainty_s_at(self, video_ns: float) -> float:
        """1-sigma uncertainty of the fitted UTC at a specific ``video_ns``.

        Uses the closed-form OLS prediction-of-mean variance::

            var(yhat) = sigma^2 * (1/n + (x - xmean)^2 / sxx)

        which grows quadratically with distance from the centroid. For a
        well-distributed anchor set (~1000+ anchors spanning a 35-min
        session) this typically stays within a small multiple of
        :attr:`fit_uncertainty_s`.
        """
        if self.n < 3 or self.sxx_ns2 <= 0:
            return float("inf")
        sigma = self.sigma_residual_s
        dx = video_ns - self.xmean
        return sigma * (1.0 / self.n + (dx * dx) / self.sxx_ns2) ** 0.5

    def video_ns_to_utc_s(self, video_ns: float) -> float:
        return self.ymean + self.slope * (video_ns - self.xmean)

    def boottime_to_utc_s(self, x_ns: float) -> float:
        """Alias of :meth:`video_ns_to_utc_s` for the alternate session layout."""
        return self.video_ns_to_utc_s(x_ns)

    def video_pts_to_utc_s(self, t_video_s: float) -> float:
        return self.video_ns_to_utc_s(t_video_s * 1e9)

    def video_pts_to_utc(self, t_video_s: float) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            self.video_pts_to_utc_s(t_video_s), tz=dt.timezone.utc
        )


def _ols_about_means(
    xs: list[float], ys: list[float]
) -> tuple[float, float, float, float, list[float]]:
    """Plain OLS slope ``b`` for ``y = a + b*(x - xmean)`` plus residuals.

    Returns ``(slope, xmean, ymean, sxx, residuals)``. ``sxx`` is exposed
    so callers can compute prediction-of-mean variance without keeping all
    inputs around. Numerically stable for large-magnitude x by always
    working in centred coordinates.
    """
    n = len(xs)
    if n < 2:
        raise ValueError(f"Need >= 2 points for regression, got {n}")

    # Use NumPy for vectorized computation (10-30x faster).
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)

    xmean = float(np.mean(x_arr))
    ymean = float(np.mean(y_arr))

    x_centered = x_arr - xmean
    y_centered = y_arr - ymean

    sxx = float(np.sum(x_centered * x_centered))
    sxy = float(np.sum(x_centered * y_centered))

    if sxx <= 0.0:
        raise ValueError("All x values are identical; cannot fit.")

    slope = sxy / sxx
    residuals = (y_arr - (ymean + slope * x_centered)).tolist()
    return slope, xmean, ymean, sxx, residuals


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _cubic_rmse_about_means(
    xs: list[float], ys: list[float]
) -> float:
    """Fit y = a0 + a1*(x-xmean) + a2*(x-xmean)^2 + a3*(x-xmean)^3 and return RMSE.

    Used purely as a diagnostic - if a degree-3 fit lowers RMSE meaningfully
    vs. linear, the source-vs-system clock relationship has non-linear drift
    we should worry about.
    """
    n = len(xs)
    if n < 5:
        return float("nan")

    # Use NumPy polyfit for cubic regression (100x faster than Gaussian elimination).
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    xmean = float(np.mean(x_arr))
    x_centered = x_arr - xmean

    try:
        coef = np.polyfit(x_centered, y_arr, deg=3)
        # coef is [a3, a2, a1, a0] for a3*x^3 + a2*x^2 + a1*x + a0
        yhat = np.polyval(coef, x_centered)
        rss = float(np.sum((y_arr - yhat) ** 2))
        return float((rss / n) ** 0.5)
    except Exception:
        return float("nan")


def estimate_clock_offset(
    imu_rows: list,
    pos_rows: list,
    *,
    max_lag_s: float = 2.0,
    step_s: float = 0.005,
    resample_hz: float = 20.0,
) -> tuple[float, float]:
    """Estimate systematic clock offset between device system clock and Signal.

    Cross-correlates Motion sensor acceleration magnitude (device clock) with
    Post-processing-derived acceleration (Signal clock) to find the lag that maximises
    correlation.  Returns ``(offset_s, confidence)`` where positive offset
    means the device clock lags Signal (media timestamps are late).

    Confidence is peak-to-median ratio of the correlation function;
    values > 3.0 indicate a reliable detection.
    """
    if len(imu_rows) < 100 or len(pos_rows) < 20:
        return 0.0, 0.0

    from scipy.ndimage import uniform_filter1d

    imu_ts = np.array([r.utc_s for r in imu_rows])
    pos_ts = np.array([r.utc_s for r in pos_rows])
    pos_ve = np.array([r.ve for r in pos_rows], dtype=float)
    pos_vn = np.array([r.vn for r in pos_rows], dtype=float)

    # --- Signal A: Motion sensor rate sensor yaw rate (turns are sharp features) ---
    imu_gz = np.array([r.gz for r in imu_rows])  # rad/s yaw rate

    # --- Signal B: Post-processing heading rate from Rate-signal velocity ---
    pos_heading = np.arctan2(pos_ve, pos_vn)  # rad
    dh = np.diff(pos_heading)
    dh = np.arctan2(np.sin(dh), np.cos(dh))  # unwrap
    dt_pos = np.diff(pos_ts)
    dt_pos = np.where(dt_pos > 0, dt_pos, 1.0)
    heading_rate = dh / dt_pos  # rad/s
    heading_rate_ts = 0.5 * (pos_ts[:-1] + pos_ts[1:])

    # --- Signal C: Motion sensor dynamic linear sensor vs Post-processing speed change (backup) ---
    imu_amag = np.array([
        abs(math.hypot(math.hypot(r.ax, r.ay), r.az) - 9.81)
        for r in imu_rows
    ])
    pos_speed = np.sqrt(pos_ve ** 2 + pos_vn ** 2)
    pos_accel = np.abs(np.diff(pos_speed) / dt_pos)
    pos_accel_ts = heading_rate_ts

    # Shared time grid.
    t_start = max(float(imu_ts[0]), float(heading_rate_ts[0]))
    t_end = min(float(imu_ts[-1]), float(heading_rate_ts[-1]))
    if t_end - t_start < 30.0:
        return 0.0, 0.0
    grid = np.arange(t_start, t_end, 1.0 / resample_hz)
    if len(grid) < 200:
        return 0.0, 0.0

    valid_hr = np.isfinite(heading_rate)
    valid_pa = np.isfinite(pos_accel)
    if valid_hr.sum() < 10 or valid_pa.sum() < 10:
        return 0.0, 0.0

    # Resample onto common grid + smooth to match Post-processing bandwidth.
    imu_gyro_r = np.interp(grid, imu_ts, np.abs(imu_gz))
    ppk_hr_r = np.interp(grid, heading_rate_ts[valid_hr], np.abs(heading_rate[valid_hr]))
    imu_acc_r = np.interp(grid, imu_ts, imu_amag)
    ppk_speed_r = np.interp(grid, pos_ts, pos_speed)

    win = max(1, int(1.0 * resample_hz))
    imu_gyro_r = uniform_filter1d(imu_gyro_r, win)
    ppk_hr_r = uniform_filter1d(ppk_hr_r, win)
    imu_acc_r = uniform_filter1d(imu_acc_r, win)
    ppk_speed_r = uniform_filter1d(ppk_speed_r, win)

    from scipy.signal import fftconvolve

    def _xcorr_fft(a, b, hz, max_lag):
        a = a - np.mean(a)
        b = b - np.mean(b)
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-9 or sb < 1e-9:
            return 0.0, 0.0
        a, b = a / sa, b / sb
        corr = fftconvolve(a, b[::-1], mode='full') / len(a)
        mid = len(a) - 1
        max_samp = int(max_lag * hz)
        lo, hi = mid - max_samp, mid + max_samp + 1
        lo, hi = max(0, lo), min(len(corr), hi)
        region = corr[lo:hi]
        peak_idx = int(np.argmax(region))
        peak_lag_samples = peak_idx - (mid - lo)
        peak_lag_s = peak_lag_samples / hz
        # Parabolic interpolation for sub-sample precision.
        if 0 < peak_idx < len(region) - 1:
            y0 = region[peak_idx - 1]
            y1 = region[peak_idx]
            y2 = region[peak_idx + 1]
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            if abs(denom) > 1e-12:
                frac = (y0 - y2) / denom
                peak_lag_s = (peak_lag_samples + frac) / hz
        peak_val = float(region[peak_idx])
        noise = float(np.median(np.abs(region)))
        conf = peak_val / max(noise, 1e-9)
        return peak_lag_s, conf

    # Try multiple signal pairs.
    lag1, c1 = _xcorr_fft(imu_gyro_r, ppk_hr_r, resample_hz, max_lag_s)
    ppk_accel_from_speed = np.abs(np.gradient(ppk_speed_r, 1.0 / resample_hz))
    ppk_accel_from_speed = uniform_filter1d(ppk_accel_from_speed, win)
    lag2, c2 = _xcorr_fft(imu_acc_r, ppk_accel_from_speed, resample_hz, max_lag_s)
    ppk_acc_r = np.interp(grid, pos_accel_ts[valid_pa], pos_accel[valid_pa])
    ppk_acc_r = uniform_filter1d(ppk_acc_r, win)
    lag3, c3 = _xcorr_fft(imu_acc_r, ppk_acc_r, resample_hz, max_lag_s)

    candidates = [(lag1, c1), (lag2, c2), (lag3, c3)]
    best_lag, best_conf = max(candidates, key=lambda x: x[1])
    return best_lag, best_conf


def fit_time_anchor_from_pairs(
    pairs: Iterable[tuple[float, float]],
    *,
    robust: bool = True,
    mad_threshold: float = 5.0,
    max_iter: int = 3,
) -> TimeAnchor:
    """OLS fit of ``utc_s = a + b*video_ns`` from (x, y) pairs.

    With ``robust=True`` (the default), we iteratively remove anchors whose
    residuals exceed ``mad_threshold`` * MAD (median absolute deviation) of
    the residuals. This protects against rare scheduling outliers (we have
    seen single anchors as much as 400 ms off the fit) without throwing
    away data that is merely jittery.
    """
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pairs:
        xv, yv = float(x), float(y)
        if math.isnan(xv) or math.isnan(yv):
            continue
        xs.append(xv)
        ys.append(yv)
    n0 = len(xs)
    if n0 < 2:
        raise ValueError(f"Need >= 2 anchors for regression, got {n0}")

    # Initial fit on every anchor.
    slope, xmean, ymean, sxx, residuals = _ols_about_means(xs, ys)
    keep_xs, keep_ys = list(xs), list(ys)

    n_rejected = 0
    if robust:
        for _ in range(max_iter):
            mad = _median([abs(r) for r in residuals])
            if mad <= 0.0:
                break
            cutoff = mad_threshold * mad
            new_xs: list[float] = []
            new_ys: list[float] = []
            for x, y, r in zip(keep_xs, keep_ys, residuals):
                if abs(r) <= cutoff:
                    new_xs.append(x)
                    new_ys.append(y)
            # Guard: if a rejection round would drop us below the minimum
            # anchor count needed for OLS (n=2), bail without rejecting any
            # more so we keep the previous valid fit + matching residuals
            # rather than refitting on a degenerate set.
            if len(new_xs) < 2:
                break
            if len(new_xs) == len(keep_xs):
                break
            keep_xs, keep_ys = new_xs, new_ys
            slope, xmean, ymean, sxx, residuals = _ols_about_means(keep_xs, keep_ys)
        n_rejected = n0 - len(keep_xs)

    rss = sum(r * r for r in residuals)
    rmse = (rss / max(1, len(residuals))) ** 0.5
    max_abs = max((abs(r) for r in residuals), default=0.0)

    cubic_rmse = _cubic_rmse_about_means(keep_xs, keep_ys)
    cubic_improve = (
        rmse - cubic_rmse if not math.isnan(cubic_rmse) else 0.0
    )

    return TimeAnchor(
        slope=slope,
        xmean=xmean,
        ymean=ymean,
        n=len(keep_xs),
        rmse_s=rmse,
        max_abs_s=max_abs,
        n_rejected=n_rejected,
        cubic_rmse_improvement_s=max(0.0, cubic_improve),
        sxx_ns2=sxx,
    )


def fit_time_anchor(
    path: Path,
    *,
    robust: bool = True,
    imu_rows: list | None = None,
    pos_rows: list | None = None,
    min_offset_confidence: float = 3.0,
) -> TimeAnchor:
    """Fit a :class:`TimeAnchor` from the session timing file at ``path``.

    Each line provides a device-tick / UTC pair; any trailing columns are
    ignored. When ``imu_rows`` and ``pos_rows`` are provided, a residual
    systematic offset is estimated by cross-correlation and folded into the fit.
    """

    def _iter_pairs() -> Iterable[tuple[float, float]]:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    x_ns = int(parts[0])           # video_ns (legacy) or boottime_ns (current)
                    utc_s = _parse_utc_seconds(parts[1])
                except (ValueError, IndexError):
                    continue
                yield (float(x_ns), utc_s)

    pairs = list(_iter_pairs())
    if len(pairs) < 2:
        # An empty / single-line recording_*.txt has no usable time bridge —
        # the session cannot be coordinate-tagged (per the capture format spec, an
        # empty session file marks an unscorable session).
        from .errors import PipelineError
        raise PipelineError(
            "E-PP-305",
            f"recording time-anchor file has {len(pairs)} usable anchor row(s) "
            f"(need >= 2): {path}",
            hint="The capture has no time bridge between the device clock and "
                 "UTC, so frames cannot be georeferenced. Re-capture the "
                 "session, or supply a recording_*.txt with GNSS-UTC anchors.",
            context={"path": str(path), "n_anchors": len(pairs)},
        )

    anchor = fit_time_anchor_from_pairs(iter(pairs), robust=robust)

    if imu_rows and pos_rows:
        offset_s, confidence = estimate_clock_offset(imu_rows, pos_rows)
        if confidence >= min_offset_confidence and abs(offset_s) > 0.001:
            anchor = TimeAnchor(
                slope=anchor.slope,
                xmean=anchor.xmean,
                ymean=anchor.ymean + offset_s,
                n=anchor.n,
                rmse_s=anchor.rmse_s,
                max_abs_s=anchor.max_abs_s,
                n_rejected=anchor.n_rejected,
                cubic_rmse_improvement_s=anchor.cubic_rmse_improvement_s,
                sxx_ns2=anchor.sxx_ns2,
                clock_offset_s=offset_s,
                clock_offset_confidence=confidence,
            )

    return anchor


# -----------------------------
# Capture-format awareness
# -----------------------------
#
# The recording_*.txt time-anchor file ships in two on-disk dialects, both of
# which the OLS fit above consumes natively (column 0 = an integer device tick,
# column 1 = ISO-8601 UTC):
#
#   OLD ("video_ns")  : column 0 is the media presentation timestamp in ns,
#                       counted from the start of the session (small values,
#                       starting near 0). Sample timing maps a sample's PTS
#                       (seconds) directly:  utc = anchor.video_pts_to_utc_s(pts).
#
#   NEW ("boottime")  : column 0 is ABSOLUTE CLOCK_BOOTTIME in ns (large values,
#                       monotonic since boot). capture_meta.json carries
#                       anchor_format=2 and video_t0_boottime_ns. Sample timing
#                       maps via:  utc = anchor.boottime_to_utc_s(t0 + pts*1e9).
#
# Both dialects fit the same affine model; only the *sample->x* mapping differs,
# and that routing lives in stages.georef._load_frames (driven by capture_meta).
# This constant + helper exist so callers/tests can name the distinction.

ANCHOR_FORMAT_VIDEO_NS = 0   # legacy media-PTS layout (no capture_meta)
ANCHOR_FORMAT_BOOTTIME = 2   # current absolute-boottime layout


# -----------------------------
# Empty-session-anchor fallback (derive boot->UTC from measurements_*.txt)
# -----------------------------
#
# 7 of 17 DAY14 sessions ship a 0-byte recording_*.txt — the time bridge the
# The platform app normally writes was never flushed. Without (boot, UTC) pairs the
# session cannot be coordinate-tagged and fit_time_anchor raises E-PP-305.
#
# The measurements_*.txt Raw rows independently carry BOTH clocks:
#   * Signal time  = TimeNanos - (FullBiasNanos + BiasNanos)  [ns since the reference epoch]
#                  -> UTC = reference_epoch + signal_ns/1e9 - epoch_offset
#   * boottime   = ChipsetElapsedRealtimeNanos (LAST Raw column) [ns since boot]
#
# When the chipset boottime column is populated (some source models), every
# Raw row is a (boottime_ns, UTC_s) pair and we fit the SAME affine model
# fit_time_anchor uses for session.txt. When that column is all-zero (the
# "dodge" captures), no boot<->UTC bridge exists in measurements, so the
# fallback REPORTS the session as anchor-unrecoverable (raises) rather than
# fabricating a mapping — per the capture-format contract.

_GPS_EPOCH_UNIX_S_TS: float = 315964800.0  # 1980-01-06 00:00:00 UTC


def boot_utc_pairs_from_measurements(
    path: Path,
) -> list[tuple[float, float]]:
    """Extract (boottime_ns, UTC_s) pairs from a measurements_*.txt.

    Reads ``Raw,`` rows, derives UTC from the Signal clock columns and pairs it
    with the ``ChipsetElapsedRealtimeNanos`` boottime (last column). Rows with a
    zero/blank/non-numeric boottime are skipped (no usable bridge for that row).

    Header layout (canonical The logger app / current app):
        Raw, utcTimeMillis, TimeNanos, LeapSecond, TimeUncertaintyNanos,
        FullBiasNanos, BiasNanos, ...                      <-- cols 2,5,6
        ..., ChipsetElapsedRealtimeNanos                   <-- LAST col
    """
    pairs: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.startswith("Raw,"):
                continue
            parts = raw.rstrip("\n").split(",")
            if len(parts) < 7:
                continue
            try:
                time_nanos = float(parts[2])
                full_bias = float(parts[5])
                bias = float(parts[6]) if parts[6] not in ("", None) else 0.0
                boot_ns = float(parts[-1])
            except (ValueError, IndexError):
                continue
            if not (boot_ns > 0):
                continue  # chipset boottime not populated for this row
            gnss_ns = time_nanos - (full_bias + bias)
            utc_unadj = _GPS_EPOCH_UNIX_S_TS + gnss_ns / 1e9
            leap = get_leap_seconds_for_epoch(utc_unadj)
            utc_s = utc_unadj - leap
            pairs.append((boot_ns, utc_s))
    return pairs


# Source tags returned by the measurements fallback. The Raw-row bridge
# (Signal clock + ChipsetElapsedRealtimeNanos) has been cross-validated against
# recording_*.txt on sessions carrying both: it agrees to < 0.1 ms. The
# Fix-row bridge (UnixTimeMillis + Location.elapsedRealtimeNanos) is
# SYSTEMATICALLY BIASED by the Signal fix delivery latency: on the SM-S901B
# day14/s21 sessions with ground truth it maps boottime -> UTC 107-140 ms
# EARLY (mean -123 ms, sd 13 ms across 7 sessions). Callers MUST surface
# this to the user when the Fix-row source is used: absolute Signal<->media
# and Signal<->stream timing is then ~0.1-0.15 s off (well outside a 10 ms
# budget), while stream<->media RELATIVE sync is unaffected because both
# media timelines ride the same anchor.
ANCHOR_SOURCE_MEASUREMENTS_RAW = "measurements-fallback"
ANCHOR_SOURCE_MEASUREMENTS_FIX = "measurements-fix-fallback"


def _fit_time_anchor_from_measurements_src(
    measurements_path: Path,
    *,
    robust: bool = True,
) -> tuple[TimeAnchor, str]:
    """Build a boot->UTC anchor from measurements_*.txt, reporting the source.

    Returns ``(anchor, source)`` where ``source`` is
    :data:`ANCHOR_SOURCE_MEASUREMENTS_RAW` when the accurate Raw-row bridge
    was used, or :data:`ANCHOR_SOURCE_MEASUREMENTS_FIX` when only the
    latency-biased Fix-row bridge was available (see the note on the
    constants above). Raises :class:`PipelineError` (E-PP-306) when neither
    yields >= 2 usable (boottime, UTC) pairs.
    """
    pairs = boot_utc_pairs_from_measurements(measurements_path)
    source = ANCHOR_SOURCE_MEASUREMENTS_RAW
    if len(pairs) < 2:
        # Raw-row bridge is dead (e.g. DAY14 "dodge" captures where
        # ChipsetElapsedRealtimeNanos is 0/blank). Try the Fix-row bridge as a
        # second-chance source before giving up. Lazy import: capture_diag
        # imports TimeAnchor from this module, so a top-level import here
        # would be circular.
        try:
            from .capture_diag import boot_utc_pairs_from_fix_rows
            fix_pairs = boot_utc_pairs_from_fix_rows(measurements_path)
            if len(fix_pairs) >= 2:
                pairs = fix_pairs
                source = ANCHOR_SOURCE_MEASUREMENTS_FIX
        except Exception:
            pass
    if len(pairs) < 2:
        from .errors import PipelineError
        raise PipelineError(
            "E-PP-306",
            f"measurements time-anchor fallback found {len(pairs)} usable "
            f"(boottime, UTC) row(s) (need >= 2): {measurements_path}",
            hint="The recording_*.txt time bridge is empty AND the "
                 "measurements_*.txt Raw rows carry no ElapsedRealtimeNanos "
                 "(ChipsetElapsedRealtimeNanos column is 0/blank), so there is "
                 "no boot<->UTC mapping to recover. The Fix-row fallback "
                 "(UnixTimeMillis + elapsedRealtimeNanos) was also tried and "
                 "did not yield >= 2 usable pairs. This session's frames "
                 "cannot be georeferenced; re-capture with a populated "
                 "recording_*.txt.",
            context={
                "measurements_path": str(measurements_path),
                "n_pairs": len(pairs),
            },
        )
    return fit_time_anchor_from_pairs(iter(pairs), robust=robust), source


def fit_time_anchor_from_measurements(
    measurements_path: Path,
    *,
    robust: bool = True,
) -> TimeAnchor:
    """Build a boot->UTC :class:`TimeAnchor` from measurements_*.txt.

    Used as the fallback when recording_*.txt is empty/missing. Raises
    :class:`PipelineError` (E-PP-306) when the measurements file carries no
    usable ``ChipsetElapsedRealtimeNanos`` boottime AND no usable ``Fix,``-row
    boot<->UTC pairing either (so the session is anchor-unrecoverable and
    must be reported, not guessed).

    NOTE: prefer :func:`fit_time_anchor_with_fallback` (or the private
    ``_fit_time_anchor_from_measurements_src``) when the caller needs to know
    WHICH bridge was used -- the Fix-row bridge carries a ~0.1-0.15 s fix
    latency bias that users must be warned about.
    """
    anchor, _source = _fit_time_anchor_from_measurements_src(
        measurements_path, robust=robust
    )
    return anchor


def fit_time_anchor_with_fallback(
    recording_path: Path,
    measurements_path: Path | None = None,
    *,
    robust: bool = True,
    imu_rows: list | None = None,
    pos_rows: list | None = None,
) -> tuple[TimeAnchor, str]:
    """Fit a TimeAnchor, falling back to measurements when session is empty.

    Returns ``(anchor, source)`` where ``source`` is ``"session.txt"`` when
    the normal recording_*.txt bridge was usable,
    ``"measurements-fallback"`` when the boot->UTC anchor was derived from the
    measurements Raw rows (Signal clock + ChipsetElapsedRealtimeNanos --
    validated accurate to < 0.1 ms against session.txt), or
    ``"measurements-fix-fallback"`` when only the Fix-row bridge was usable
    (systematically ~0.1-0.15 s EARLY due to fix delivery latency; callers
    must warn -- see ANCHOR_SOURCE_MEASUREMENTS_FIX). Propagates the
    underlying PipelineError when NEITHER source yields a usable bridge.
    """
    rec_usable = False
    try:
        if Path(recording_path).is_file() and Path(recording_path).stat().st_size > 0:
            rec_usable = True
    except OSError:
        rec_usable = False

    if rec_usable:
        try:
            anchor = fit_time_anchor(
                recording_path, robust=robust,
                imu_rows=imu_rows, pos_rows=pos_rows,
            )
            return anchor, "recording.txt"
        except Exception:
            # session.txt present but unparsable -> try the fallback below.
            if measurements_path is None:
                raise

    if measurements_path is None:
        # No fallback available; surface the canonical empty-anchor error.
        return fit_time_anchor(recording_path, robust=robust), "recording.txt"

    return _fit_time_anchor_from_measurements_src(measurements_path, robust=robust)


def classify_anchor_x_semantics(path: Path, *, threshold_ns: float = 1e14) -> str:
    """Heuristically classify a recording_*.txt as ``"video_ns"`` or ``"boottime"``.

    Content sniff used only as a fallback when capture_meta.json is absent
    (capture_meta + anchor_format is the authoritative signal). Absolute
    CLOCK_BOOTTIME values are large (device uptime in ns); legacy video_ns
    values start near zero. Returns ``"video_ns"`` for an unreadable / empty
    file so the safe legacy path is taken.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    x_ns = int(parts[0])
                except ValueError:
                    continue
                return "boottime" if x_ns >= threshold_ns else "video_ns"
    except OSError:
        pass
    return "video_ns"


def per_anchor_residuals(
    path: Path, anchor: TimeAnchor
) -> list[tuple[float, float]]:
    """Return ``(video_s, residual_s)`` pairs for every anchor against ``anchor``.

    Useful for diagnostic plots: plotting residuals vs media time will show
    any structure (drift, jumps, outliers) the linear fit didn't absorb.
    """
    out: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                video_ns = int(parts[0])
                utc_s = _parse_utc_seconds(parts[1])
            except (ValueError, IndexError):
                continue
            r = utc_s - anchor.video_ns_to_utc_s(video_ns)
            out.append((video_ns / 1e9, r))
    return out


# -----------------------------
# Backwards-compatible single-anchor API (kept for callers that don't
# care about sub-ms accuracy).
# -----------------------------


def parse_recording_map_first_line(path: Path) -> tuple[int, dt.datetime]:
    """Read the first ``video_ns,UTC`` row from a the source app recording_*.txt.

    NOTE: prefer :func:`fit_time_anchor` for accurate-to-ms results.
    """
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty recording map: {path}")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Unexpected recording map line in {path}: {line!r}")
    video_ns = int(parts[0])
    return video_ns, _parse_iso_utc(parts[1])


def utc_at_video_zero(path: Path) -> dt.datetime:
    """Return the UTC datetime corresponding to media PTS=0 from the *first* anchor.

    NOTE: the first-anchor approach has up to ~30 ms of jitter and accumulates
    drift across the session. Prefer :func:`fit_time_anchor` for ms accuracy.
    """
    video_ns, utc_at_first = parse_recording_map_first_line(path)
    return utc_at_first - dt.timedelta(seconds=video_ns / 1e9)


def video_pts_to_utc(utc0: dt.datetime, t_video_s: float) -> dt.datetime:
    """Map a media PTS (seconds) to UTC, given a single utc0 anchor."""
    return utc0 + dt.timedelta(seconds=t_video_s)


def gpst_to_utc_seconds(
    gpst_unix_seconds: float, leap_seconds: float = GPS_UTC_LEAP_SECONDS_2026
) -> float:
    """Convert a unix-seconds-style Reference time value to UTC unix seconds."""
    return gpst_unix_seconds - leap_seconds
