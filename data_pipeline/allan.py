"""Allan-variance (overlapping) calibration of device Motion sensor noise.

Computes the OVERLAPPING Allan deviation ``sigma(tau)`` over log-spaced
averaging times for each axis of the rate sensor (3) and linear sensor (3),
from the Motion sensor stream parsed out of ``sensors_*.txt`` (see
:func:`data_pipeline.parsers.parse_imu` / :class:`data_pipeline.parsers.ImuRow`).

From each axis curve we extract the standard Motion sensor noise parameters used by
IEEE Std 952 / the Motion sensor-Signal fusion process-noise model:

* **ARW** (rate sensor Angle Random Walk) / **VRW** (linear sensor Velocity Random Walk):
  the value of ``sigma(tau)`` on the ``slope = -1/2`` line read at ``tau = 1 s``.
  Rate sensor units: rad/s/sqrt(Hz) (white angular-rate noise density). Linear sensor units:
  (m/s^2)/sqrt(Hz) (white acceleration noise density).
* **Bias instability**: the flat minimum of ``sigma(tau)`` divided by the
  scaling factor ``0.664`` (the ``B`` coefficient, IEEE Std 952).
* **Rate random walk (RRW)**: the value on the ``slope = +1/2`` line read at
  ``tau = 3 s`` (the ``K`` coefficient), i.e. how fast the bias itself walks.

We also report the sample rate and total duration, and warn when the record
is too short for a reliable bias-instability estimate (the flat minimum of an
Allan curve typically sits at tens of seconds of averaging time, so a record
that cannot reach a few times that cluster has an unreliable ``B``).

The math here is signal-agnostic: feed it any rate-like channel sampled at a
constant rate. The :func:`allan_from_imu_rows` convenience wires the six device
channels through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# IEEE Std 952 bias-instability scaling: B = min(sigma) / 0.664.
_BIAS_INSTAB_SCALE = 0.664

# A bias-instability estimate is only trustworthy if the Allan curve can be
# sampled out to at least this averaging time (the flat minimum usually sits
# in the tens-of-seconds region for consumer MEMS Motion sensors). Below this we still
# report a value but flag it as unreliable.
_MIN_TAU_FOR_BIAS_S = 10.0


@dataclass
class AxisAllan:
    """Allan-deviation curve + extracted params for one channel."""

    name: str                      # e.g. "gx" / "ax"
    tau_s: np.ndarray              # averaging times (s), log-spaced
    sigma: np.ndarray             # overlapping Allan deviation at each tau
    # White-noise random-walk coefficient (N): ARW for rate sensor, VRW for linear sensor.
    # Read off the -1/2 slope line at tau = 1 s. Same units as the channel
    # values * sqrt(s)  ==  units / sqrt(Hz).
    random_walk: float
    # Bias-instability coefficient B = min(sigma) / 0.664. Same units as the
    # channel (rad/s for rate sensor, m/s^2 for linear sensor).
    bias_instability: float
    # Rate-random-walk coefficient K: read off the +1/2 slope line at tau=3 s.
    rate_random_walk: float
    bias_instability_tau_s: float  # the tau where the flat minimum was found


@dataclass
class AllanResult:
    """Full Allan analysis for all six device Motion sensor channels."""

    sample_rate_hz: float
    duration_s: float
    n_samples: int
    axes: dict[str, AxisAllan] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Convenience aggregates expected by the calibration store / fusion.
    @property
    def gyro_arw(self) -> dict[str, float]:
        return {ax: self.axes[ax].random_walk for ax in ("gx", "gy", "gz") if ax in self.axes}

    @property
    def accel_vrw(self) -> dict[str, float]:
        return {ax: self.axes[ax].random_walk for ax in ("ax", "ay", "az") if ax in self.axes}

    @property
    def bias_instability(self) -> dict[str, float]:
        return {ax: a.bias_instability for ax, a in self.axes.items()}

    @property
    def rate_random_walk(self) -> dict[str, float]:
        return {ax: a.rate_random_walk for ax, a in self.axes.items()}

    def mean_gyro_arw(self) -> float:
        vals = list(self.gyro_arw.values())
        return float(np.mean(vals)) if vals else float("nan")

    def mean_accel_vrw(self) -> float:
        vals = list(self.accel_vrw.values())
        return float(np.mean(vals)) if vals else float("nan")


def _log_spaced_m(n: int, n_points: int = 50) -> np.ndarray:
    """Log-spaced averaging factors m in [1, (n-1)//2]."""
    m_max = max(1, (n - 1) // 2)
    if m_max <= 1:
        return np.array([1], dtype=int)
    m = np.unique(
        np.floor(np.logspace(0, math.log10(m_max), num=n_points)).astype(int)
    )
    m = m[(m >= 1) & (m <= m_max)]
    return m


def overlapping_allan_deviation(
    rate: Sequence[float],
    fs: float,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a uniformly-sampled rate signal.

    Implements IEEE Std 952 overlapping Allan variance by integrating the
    rate into an "angle"/"velocity" series ``theta`` and forming the
    second-difference estimator::

        sigma^2(tau) = 1 / (2 * tau^2 * (N - 2m)) *
                       sum_{k=1}^{N-2m} (theta[k+2m] - 2 theta[k+m] + theta[k])^2

    with ``tau = m / fs``.

    Parameters
    ----------
    rate : sequence of float
        Uniformly sampled rate samples (rate sensor rad/s or linear sensor m/s^2).
    fs : float
        Sample rate in Hz (> 0).
    n_points : int
        Number of log-spaced averaging factors to evaluate.

    Returns
    -------
    (tau_s, sigma) : two 1-D numpy arrays of equal length.
    """
    x = np.asarray(rate, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if fs <= 0 or n < 3:
        return np.array([]), np.array([])

    tau0 = 1.0 / fs
    # Integrate rate -> angle (cumulative; prepend 0 so theta has length n+1).
    theta = np.concatenate(([0.0], np.cumsum(x) * tau0))
    N = theta.size  # == n + 1

    ms = _log_spaced_m(n, n_points)
    taus: list[float] = []
    sigmas: list[float] = []
    for m in ms:
        if (N - 2 * m) < 1:
            continue
        tau = m * tau0
        # theta indices 0..N-1; second difference with lag m.
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        # d has length N - 2m.
        var = np.sum(d * d) / (2.0 * tau * tau * d.size)
        if var > 0 and math.isfinite(var):
            taus.append(tau)
            sigmas.append(math.sqrt(var))
    return np.asarray(taus), np.asarray(sigmas)


def _fit_slope_value(tau: np.ndarray, sigma: np.ndarray, slope: float,
                     read_tau: float) -> float:
    """Fit a fixed-slope line in log-log to the part of the curve where it
    dominates, then read its value at ``read_tau``.

    For ``slope = -1/2`` (random walk) we fit the short-tau side; for
    ``slope = +1/2`` (rate random walk) the long-tau side. Robust to the
    bias-instability flat region by restricting the fit window.
    """
    if tau.size == 0:
        return float("nan")
    lt = np.log10(tau)
    ls = np.log10(sigma)
    # local log-log slope between neighbouring points
    if tau.size >= 2:
        local = np.gradient(ls, lt)
    else:
        local = np.array([slope])
    # Keep points whose local slope is within 0.25 of the target slope.
    mask = np.abs(local - slope) < 0.25
    if mask.sum() < 2:
        # Fall back: for -1/2 use the lowest-tau third, for +1/2 the highest.
        k = max(2, tau.size // 3)
        if slope < 0:
            mask = np.zeros_like(tau, dtype=bool); mask[:k] = True
        else:
            mask = np.zeros_like(tau, dtype=bool); mask[-k:] = True
    # Fit intercept b for fixed slope: ls = slope * lt + b  ->  b = mean(ls - slope*lt)
    b = float(np.mean(ls[mask] - slope * lt[mask]))
    val_log = slope * math.log10(read_tau) + b
    return float(10.0 ** val_log)


def analyze_axis(name: str, rate: Sequence[float], fs: float,
                 n_points: int = 50) -> AxisAllan:
    """Compute the Allan curve and extract params for one channel."""
    tau, sigma = overlapping_allan_deviation(rate, fs, n_points=n_points)
    if tau.size == 0:
        return AxisAllan(name, tau, sigma, float("nan"), float("nan"),
                         float("nan"), float("nan"))

    # Random walk (ARW/VRW): -1/2 slope line read at tau = 1 s, expressed as
    # the coefficient N = sigma(1 s) on that line. Units = channel/sqrt(Hz).
    random_walk = _fit_slope_value(tau, sigma, slope=-0.5, read_tau=1.0)

    # Bias instability: flat minimum / 0.664.
    i_min = int(np.argmin(sigma))
    bias_instability = float(sigma[i_min] / _BIAS_INSTAB_SCALE)
    bias_tau = float(tau[i_min])

    # Rate random walk: +1/2 slope line read at tau = 3 s.
    rate_random_walk = _fit_slope_value(tau, sigma, slope=0.5, read_tau=3.0)

    return AxisAllan(
        name=name,
        tau_s=tau,
        sigma=sigma,
        random_walk=random_walk,
        bias_instability=bias_instability,
        rate_random_walk=rate_random_walk,
        bias_instability_tau_s=bias_tau,
    )


def estimate_sample_rate(utc_s: Sequence[float]) -> float:
    """Robust sample-rate estimate from timestamps (median dt)."""
    t = np.asarray(utc_s, dtype=float)
    if t.size < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return float("nan")
    med = float(np.median(dt))
    return 1.0 / med if med > 0 else float("nan")


def compute_allan(
    imu_rows: Sequence["object"],
    n_points: int = 50,
    fs: Optional[float] = None,
) -> AllanResult:
    """Run the overlapping Allan analysis over all six channels of an Motion sensor log.

    Parameters
    ----------
    imu_rows : sequence of ImuRow-like
        Objects with attributes ``utc_s, ax, ay, az, gx, gy, gz``.
    n_points : int
        Log-spaced averaging factors per axis.
    fs : float, optional
        Override sample rate; otherwise estimated from timestamps.
    """
    rows = list(imu_rows)
    n = len(rows)
    if n < 3:
        return AllanResult(
            sample_rate_hz=float("nan"), duration_s=0.0, n_samples=n,
            warnings=["IMU record too short for Allan analysis (need >= 3 samples)."],
        )

    utc = np.array([r.utc_s for r in rows], dtype=float)
    sample_rate = fs if (fs and fs > 0) else estimate_sample_rate(utc)
    duration = float(utc[-1] - utc[0])

    channels = {
        "gx": np.array([r.gx for r in rows], float),
        "gy": np.array([r.gy for r in rows], float),
        "gz": np.array([r.gz for r in rows], float),
        "ax": np.array([r.ax for r in rows], float),
        "ay": np.array([r.ay for r in rows], float),
        # linear sensor z carries gravity; remove its mean so the white-noise/bias
        # estimate is not biased by the ~9.81 offset.
        "az": np.array([r.az for r in rows], float),
    }
    channels["ax"] = channels["ax"] - np.nanmean(channels["ax"])
    channels["ay"] = channels["ay"] - np.nanmean(channels["ay"])
    channels["az"] = channels["az"] - np.nanmean(channels["az"])

    axes: dict[str, AxisAllan] = {}
    for name, data in channels.items():
        axes[name] = analyze_axis(name, data, sample_rate, n_points=n_points)

    warnings: list[str] = []
    max_tau = max(
        (float(a.tau_s[-1]) if a.tau_s.size else 0.0) for a in axes.values()
    )
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        warnings.append("Could not estimate a valid sample rate from timestamps.")
    if max_tau < _MIN_TAU_FOR_BIAS_S:
        warnings.append(
            f"Record too short for reliable bias-instability: longest "
            f"averaging time tau={max_tau:.1f}s < {_MIN_TAU_FOR_BIAS_S:.0f}s. "
            f"Record at least ~{int(_MIN_TAU_FOR_BIAS_S * 5)}s static for a "
            f"trustworthy bias-instability minimum."
        )

    return AllanResult(
        sample_rate_hz=float(sample_rate),
        duration_s=duration,
        n_samples=n,
        axes=axes,
        warnings=warnings,
    )


# Backwards-friendly alias matching the JOB description wording.
allan_from_imu_rows = compute_allan
