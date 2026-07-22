"""Tests for data_pipeline.allan — overlapping Allan deviation.

Validates extracted noise parameters against a SYNTHETIC signal of known
white noise + bias instability (random-walk bias), so the truth values are
known and we can assert recovery within tolerance.
"""

import math

import numpy as np
import pytest

from data_pipeline.allan import (
    compute_allan,
    overlapping_allan_deviation,
    analyze_axis,
)
from data_pipeline.parsers import ImuRow


def _make_white_plus_rw(n, fs, white_density, rw_rate, seed=0):
    """Synthesize a rate signal = white noise + integrated white (random walk).

    white_density : N coefficient (units/sqrt(Hz)). White rate std per sample
        is white_density * sqrt(fs).
    rw_rate : K coefficient controlling the random-walk part. The increment
        per sample has std = rw_rate * sqrt(dt).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    white = rng.normal(0.0, white_density * math.sqrt(fs), n)
    rw_incr = rng.normal(0.0, rw_rate * math.sqrt(dt), n)
    rw = np.cumsum(rw_incr)
    return white + rw


def test_white_noise_slope_minus_half():
    """Pure white noise -> Allan deviation follows a -1/2 slope and the
    recovered random-walk coefficient matches the input density."""
    fs = 100.0
    n = 200_000
    density = 0.01  # units/sqrt(Hz)
    rng = np.random.default_rng(42)
    rate = rng.normal(0.0, density * math.sqrt(fs), n)

    tau, sigma = overlapping_allan_deviation(rate, fs)
    assert tau.size > 5

    # Log-log slope of the curve should be near -0.5 across the board.
    slope = np.polyfit(np.log10(tau), np.log10(sigma), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)

    ax = analyze_axis("gx", rate, fs)
    # ARW read at tau=1s should recover the density within 10%.
    assert ax.random_walk == pytest.approx(density, rel=0.10)


def test_recover_white_and_rate_random_walk():
    """White + random-walk synthetic: recover both N (ARW) and K (RRW)."""
    fs = 100.0
    n = 300_000
    density = 0.005
    k = 0.002
    rate = _make_white_plus_rw(n, fs, density, k, seed=7)

    ax = analyze_axis("gx", rate, fs)

    # White-noise coefficient within 15%.
    assert ax.random_walk == pytest.approx(density, rel=0.15)
    # Rate-random-walk coefficient: K is read off the +1/2 line at tau=3s.
    # The theoretical value on that line is K*sqrt(tau/3)... our reader returns
    # the coefficient K directly. Allow generous tolerance (RRW is noisy).
    assert ax.rate_random_walk == pytest.approx(k, rel=0.5)
    # Bias instability should be a positive finite minimum.
    assert math.isfinite(ax.bias_instability)
    assert ax.bias_instability > 0


def test_compute_allan_from_imu_rows():
    """End-to-end over six channels via ImuRow objects."""
    fs = 200.0
    n = 60_000  # 300 s
    dt = 1.0 / fs
    rng = np.random.default_rng(3)
    g_density = 0.001
    a_density = 0.02
    g = rng.normal(0.0, g_density * math.sqrt(fs), (3, n))
    a = rng.normal(0.0, a_density * math.sqrt(fs), (3, n))
    # linear sensor z has gravity offset which must be removed internally.
    a[2] += 9.81

    rows = [
        ImuRow(utc_s=i * dt, ax=a[0, i], ay=a[1, i], az=a[2, i],
               gx=g[0, i], gy=g[1, i], gz=g[2, i])
        for i in range(n)
    ]
    res = compute_allan(rows)

    assert res.sample_rate_hz == pytest.approx(fs, rel=0.01)
    assert res.duration_s == pytest.approx(n * dt, rel=0.01)
    assert set(res.axes) == {"gx", "gy", "gz", "ax", "ay", "az"}

    for axn in ("gx", "gy", "gz"):
        assert res.axes[axn].random_walk == pytest.approx(g_density, rel=0.20)
    for axn in ("ax", "ay", "az"):
        assert res.axes[axn].random_walk == pytest.approx(a_density, rel=0.20)

    # Aggregates.
    assert res.mean_gyro_arw() == pytest.approx(g_density, rel=0.20)
    assert res.mean_accel_vrw() == pytest.approx(a_density, rel=0.20)


def test_short_record_warns():
    """A short record should still run but warn about bias instability."""
    fs = 200.0
    n = 400  # 2 s — far too short for a stable bias-instability minimum
    dt = 1.0 / fs
    rng = np.random.default_rng(1)
    rows = [
        ImuRow(utc_s=i * dt, ax=rng.normal(), ay=rng.normal(), az=9.81 + rng.normal(),
               gx=rng.normal() * 0.01, gy=rng.normal() * 0.01, gz=rng.normal() * 0.01)
        for i in range(n)
    ]
    res = compute_allan(rows)
    assert any("bias-instability" in w for w in res.warnings)


def test_empty_input():
    res = compute_allan([])
    assert res.n_samples == 0
    assert res.warnings
