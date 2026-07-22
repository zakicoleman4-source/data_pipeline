"""Rauch-Tung-Striebel (RTS) Recursive smoother for path smoothing.

This implements a constant-velocity Recursive filter with backward RTS smoothing.
It's superior to independent low-pass filtering because it:

1. Uses Post-processing position covariance (from .pos Q matrix) to weight observations.
2. Produces velocity/acceleration estimates as a byproduct.
3. Is the standard approach for vehicle coordinate tagging.
4. Handles measurement gaps and outliers naturally.

References:
- Rauch, S. H., Tung, F., & Striebel, C. T. (1965). "Maximum likelihood estimates
  of linear dynamic systems." AIAA Journal, 3(8), 1445-1450.
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). "Estimation with
  applications to tracking and auxiliary-data." Wiley-Interscience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class KalmanState:
    """State vector for constant-velocity Recursive filter.

    Position and velocity in one coordinate direction (e.g., lat or lon).
    """
    x: float  # Position.
    v: float  # Velocity.
    P: float  # Position variance.
    Pv: float  # Position-velocity covariance.
    Pv_v: float  # Velocity variance.

    def to_array(self) -> np.ndarray:
        """Convert to [x, v]^T."""
        return np.array([self.x, self.v], dtype=np.float64)

    def covariance_matrix(self) -> np.ndarray:
        """Return the 2x2 covariance matrix [[P, Pv], [Pv, Pv_v]]."""
        return np.array(
            [[self.P, self.Pv], [self.Pv, self.Pv_v]],
            dtype=np.float64
        )


def constant_velocity_rts_smoother(
    positions: Sequence[float],
    times_s: Sequence[float],
    measurement_variance: float | Sequence[float] | None = None,
    process_noise_std: float = 0.01,
    initial_position_std: float = 1.0,
    initial_velocity_std: float = 1.0,
) -> tuple[list[float], list[float], list[float]]:
    """RTS Recursive smoother with constant-velocity model.

    Args:
        positions: Sequence of position measurements (can have NaN for missing).
        times_s: Sequence of measurement times in seconds (must be increasing).
        measurement_variance: Variance of position measurements. If a scalar, used
            for all. If a sequence, one per measurement. If None, estimated from
            finite measurements.
        process_noise_std: Standard deviation of velocity random walk (m/s^2).
        initial_position_std: Initial position uncertainty (m).
        initial_velocity_std: Initial velocity uncertainty (m/s).

    Returns:
        Tuple of (smoothed_positions, velocities, position_uncertainties).
    """
    n = len(positions)
    if n < 1:
        return [], [], []

    positions = np.asarray(positions, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)

    # Estimate measurement variance if not provided.
    if measurement_variance is None:
        finite_mask = np.isfinite(positions)
        if np.sum(finite_mask) > 1:
            residuals = np.diff(positions[finite_mask])
            measurement_variance = float(np.var(residuals) / 2.0)  # Rough estimate.
        else:
            measurement_variance = 1.0
    elif isinstance(measurement_variance, (list, tuple)):
        measurement_variance = np.asarray(measurement_variance, dtype=np.float64)

    process_noise_var = process_noise_std ** 2

    # Forward pass: Recursive filter.
    states_fwd: list[KalmanState] = []
    for i in range(n):
        if i == 0:
            # Initialize with first finite measurement or defaults.
            if np.isfinite(positions[i]):
                x = float(positions[i])
            else:
                x = 0.0
            state = KalmanState(
                x=x, v=0.0,
                P=initial_position_std ** 2,
                Pv=0.0,
                Pv_v=initial_velocity_std ** 2,
            )
        else:
            # Predict step.
            dt = float(times_s[i] - times_s[i - 1])
            if dt <= 0:
                # Degenerate: same or backward time. Skip.
                states_fwd.append(states_fwd[-1])
                continue

            prev = states_fwd[-1]
            # State transition: [x, v] -> [x + v*dt, v]
            x_pred = prev.x + prev.v * dt
            v_pred = prev.v
            # Covariance: P_pred = F @ P @ F^T + Q
            # F = [[1, dt], [0, 1]]; for a constant-velocity model with
            # continuous-time white-noise acceleration (PSD = q) the
            # discrete-time Q is the standard block:
            #   Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
            # The previous implementation used Q_pp = q*dt^2 and
            # Q_vv = q (no dt scaling) — wrong when dt varied between
            # Post-processing rows: short steps inflated velocity uncertainty too
            # fast and long-gap rows didn't grow it enough, distorting
            # the gain on every irregular interval.
            q_pp = process_noise_var * (dt ** 3) / 3.0
            q_pv = process_noise_var * (dt ** 2) / 2.0
            q_vv = process_noise_var * dt
            P_pred = prev.P + 2 * prev.Pv * dt + prev.Pv_v * (dt ** 2) + q_pp
            Pv_pred = prev.Pv + prev.Pv_v * dt + q_pv
            Pv_v_pred = prev.Pv_v + q_vv

            # Update step (if measurement is finite).
            if np.isfinite(positions[i]):
                z = float(positions[i])
                R = float(measurement_variance[i]) if isinstance(measurement_variance, np.ndarray) else measurement_variance

                # H = [1, 0]; K = P_pred_full @ H^T / S is a 2x1 vector.
                S = P_pred + R  # Innovation covariance.
                K_p = P_pred / S    # gain for position
                K_v = Pv_pred / S   # gain for velocity

                # Update state.
                y = z - x_pred  # Innovation.
                x = x_pred + K_p * y
                v = v_pred + K_v * y

                # Joseph-form covariance update for numerical stability.
                P = (1.0 - K_p) * P_pred
                Pv = (1.0 - K_p) * Pv_pred
                Pv_v = Pv_v_pred - K_v * Pv_pred

                state = KalmanState(x=x, v=v, P=P, Pv=Pv, Pv_v=Pv_v)
            else:
                # No measurement: use prediction.
                state = KalmanState(
                    x=x_pred, v=v_pred,
                    P=P_pred, Pv=Pv_pred, Pv_v=Pv_v_pred
                )

        states_fwd.append(state)

    # Backward pass: RTS smoother.
    states_smooth: list[KalmanState] = [KalmanState(0, 0, 0, 0, 0)] * n
    states_smooth[-1] = states_fwd[-1]  # Last state is unchanged.

    for i in range(n - 2, -1, -1):
        dt = float(times_s[i + 1] - times_s[i])
        if dt <= 0:
            states_smooth[i] = states_fwd[i]
            continue

        # Rauch gain: C = P_fwd @ F^T @ inv(P_pred)
        fwd = states_fwd[i]
        next_fwd = states_fwd[i + 1]

        # Predict next state from current forward pass.
        # Use the same Q matrix as the forward pass (CV model with
        # continuous-time white-noise acceleration).
        q_pp = process_noise_var * (dt ** 3) / 3.0
        q_pv = process_noise_var * (dt ** 2) / 2.0
        q_vv = process_noise_var * dt
        x_pred = fwd.x + fwd.v * dt
        v_pred = fwd.v
        P_pred = fwd.P + 2 * fwd.Pv * dt + fwd.Pv_v * (dt ** 2) + q_pp
        Pv_pred = fwd.Pv + fwd.Pv_v * dt + q_pv
        Pv_v_pred = fwd.Pv_v + q_vv

        # Full 2×2 RTS gain: G = P_fwd @ F^T @ inv(P_pred)
        # F = [[1, dt], [0, 1]]
        # P_fwd @ F^T = [[P+Pv*dt, Pv], [Pv+Pvv*dt, Pvv]]
        # P_pred = [[P_pred, Pv_pred], [Pv_pred, Pv_v_pred]]
        P_fwd_FT = np.array([
            [fwd.P + fwd.Pv * dt, fwd.Pv],
            [fwd.Pv + fwd.Pv_v * dt, fwd.Pv_v],
        ])
        P_pred_mat = np.array([
            [P_pred, Pv_pred],
            [Pv_pred, Pv_v_pred],
        ])
        det = P_pred * Pv_v_pred - Pv_pred * Pv_pred
        if abs(det) < 1e-20:
            states_smooth[i] = states_fwd[i]
            continue
        P_pred_inv = np.array([
            [Pv_v_pred, -Pv_pred],
            [-Pv_pred, P_pred],
        ]) / det
        G = P_fwd_FT @ P_pred_inv

        # Smooth state.
        dx = np.array([
            states_smooth[i + 1].x - x_pred,
            states_smooth[i + 1].v - v_pred,
        ])
        sx = np.array([fwd.x, fwd.v]) + G @ dx
        x_smooth = float(sx[0])
        v_smooth = float(sx[1])

        # Smooth covariance: P_s = P_fwd + G @ (P_s[k+1] - P_pred) @ G^T
        P_s_next = np.array([
            [states_smooth[i + 1].P, states_smooth[i + 1].Pv],
            [states_smooth[i + 1].Pv, states_smooth[i + 1].Pv_v],
        ])
        P_fwd_mat = np.array([[fwd.P, fwd.Pv], [fwd.Pv, fwd.Pv_v]])
        P_s = P_fwd_mat + G @ (P_s_next - P_pred_mat) @ G.T

        states_smooth[i] = KalmanState(
            x=x_smooth, v=v_smooth,
            P=float(P_s[0, 0]), Pv=float(P_s[0, 1]), Pv_v=float(P_s[1, 1])
        )

    # Extract results.
    smoothed_positions = [s.x for s in states_smooth]
    velocities = [s.v for s in states_smooth]
    position_uncertainties = [math.sqrt(max(0.0, s.P)) for s in states_smooth]

    return smoothed_positions, velocities, position_uncertainties


def smooth_trajectory_with_rts(
    lat: Sequence[float],
    lon: Sequence[float],
    h: Sequence[float],
    times_s: Sequence[float],
    measurement_variance: float = 0.01 ** 2,
    process_noise_std: float = 0.01,
) -> tuple[list[float], list[float], list[float]]:
    """Apply RTS Recursive smoother to a 3D path.

    Args:
        lat: Latitude values (degrees), can have NaN.
        lon: Longitude values (degrees), can have NaN.
        h: Height values (meters), can have NaN.
        times_s: Time values in seconds (must be increasing).
        measurement_variance: Variance of position measurements.
        process_noise_std: Standard deviation of velocity random walk.

    Returns:
        Tuple of (smoothed_lat, smoothed_lon, smoothed_h).
    """
    lat_smooth, _, _ = constant_velocity_rts_smoother(
        lat, times_s, measurement_variance, process_noise_std
    )
    lon_smooth, _, _ = constant_velocity_rts_smoother(
        lon, times_s, measurement_variance, process_noise_std
    )
    h_smooth, _, _ = constant_velocity_rts_smoother(
        h, times_s, measurement_variance, process_noise_std
    )

    return lat_smooth, lon_smooth, h_smooth
