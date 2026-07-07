"""GT-free physical-plausibility filter for a Post-processing path (``list[PosRow]``).

Motivation (day14 post-processing audit, PP3/PP4/PP5)
=====================================================
The MAX horizontal error of the day14 solves is dominated by a handful of
*physically impossible* Post-processing epochs that the current pipeline does not reject:

* ``s21/101315`` — altitude oscillates to **160.2 m** (TLV ground ~30-60 m),
  vertical speed to **-73.1 m/s**, horizontal speed to **47.7 m/s (172 km/h)**
  on Q4/float epochs that nevertheless report 1-2 m sigma.
* ``s21/081922`` — altitude to **83.7 m**, vertical speed to **-39.9 m/s**.

These are self-evidently wrong **without any ground truth** — a car does not
climb 100 m in a few seconds nor drive at 170 km/h through Tel Aviv. Rejecting
or repairing them crushes the MAX error directly, and the 2-sigma improves where
a spike sits inside the 95.45th-percentile tail.

Design rules (mirrors :mod:`data_pipeline.stages.multimask_ppk`)
=================================================================
* **GT-FREE at runtime.** The gates use only car-plausible physical bounds, the
  path's own robust statistics (median altitude, MAD on position deltas),
  and an *optional* externally supplied multimask-disagreement series. Ground
  truth is never read here.
* **Prefer REPAIR over DROP.** A short bad run between two good neighbours is
  linearly interpolated (and flagged); only a long bad run is hard-dropped.
  This directly addresses PP5: a repaired/dropped span is *always* flagged, so a
  downstream consumer can never silently bridge an un-flagged hole.
* **No-harm default.** Clean epochs are returned untouched. The bounds are wide
  enough that a normally-driven car never trips them.

Public API
==========
``robust_filter(rows, cfg=None, disagreement=None, log=None) -> FilterResult``
returns the cleaned ``list[PosRow]`` plus a per-epoch reason/flag table. The
pure maths is importable and unit-testable without any solver.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .geo import ecef_to_enu, llh_to_ecef
from .parsers import PosRow
from .pipeline import LogFn, make_logger


# ===========================================================================
# Config — car-plausible physical bounds (GT-free)
# ===========================================================================
@dataclass
class RobustFilterConfig:
    """Physical-plausibility bounds for a *car* path. All GT-free.

    Defaults are deliberately generous so a normally-driven vehicle never trips
    them; they only catch the self-evidently-impossible Post-processing divergence spikes.
    """

    # Horizontal speed gate. 40 m/s = 144 km/h — above any legal urban/highway
    # drive in this dataset; an epoch implying a faster jump is a position spike.
    max_horiz_speed_mps: float = 40.0
    # Vertical speed gate. A car climbs/descends at most a few m/s; 10 m/s is a
    # very loose ceiling. The day14 spikes hit 39-73 m/s.
    max_vert_speed_mps: float = 10.0
    # Altitude window relative to the *robust session median* (GT-free).
    # Empirically separated on day14: physically-sane routes peak at +28.8 m
    # above their own median (s21/081922, itself partly spiked) and bottom at
    # -20.8 m; the s21/101315 blow-up sustains +112.6 m (p99 +80.7). +50/-35 m
    # cleanly isolates the impossible epochs while leaving every sane route
    # (dodge max +12, s20u +18) untouched.
    alt_below_median_m: float = 35.0
    alt_above_median_m: float = 50.0
    # Robust per-epoch jump gate: reject an epoch whose position delta from the
    # local fit exceeds ``jump_mad_k`` * MAD of the delta distribution (and at
    # least ``jump_floor_m`` so a very-clean run does not flag mm noise).
    jump_mad_k: float = 8.0
    jump_floor_m: float = 8.0
    # Repair vs drop: a contiguous rejected run no longer than this many epochs
    # *and* no longer than this many seconds is interpolated from good
    # neighbours; longer runs are hard-dropped (and the gap is flagged).
    max_repair_epochs: int = 10
    max_repair_seconds: float = 12.0
    # Optional multimask-disagreement gate (only used when a series is passed):
    # epochs whose inter-mask spread exceeds this are rejected as environment noise.
    disagreement_reject_m: float = 5.0
    # ------------------------------------------------------------------
    # Speed-aware jump gate (car kinematic envelope). GT-free, opt-in:
    # OFF by default so a bare ``RobustFilterConfig()`` behaves exactly as
    # before; ``car_preset()`` turns it on.
    #
    # An epoch's velocity (position delta / dt) must be reachable from the
    # last kinematically-consistent velocity within the car's acceleration
    # envelope: |v_i - v_ref| <= a_env * elapsed + margin/dt, where
    # a_env = hypot(max(long, brake), lateral). This catches implausible
    # jumps that the fixed MAD floor misses at low speed (a stopped car
    # cannot displace 7 m in 1 s) and stays quiet at high speed where a
    # large *step* is normal (30 m @ 30 m/s cruise).
    # ------------------------------------------------------------------
    speed_gate_enabled: bool = False
    accel_long_max_mps2: float = 4.0    # sustained forward linear sensor of a car
    accel_brake_max_mps2: float = 8.0   # hard braking (~0.8 g)
    accel_lat_max_mps2: float = 8.0     # lateral grip limit (~0.8 g)
    speed_gate_margin_m: float = 3.0    # Signal-noise slack on the step residual
    # ------------------------------------------------------------------
    # Turn-rate plausibility gate. A car at speed v cannot change heading
    # faster than ~ a_lat_max / v rad/s (lateral-grip limit); an epoch
    # implying a faster yaw rate is a position spike. Only applied when
    # speed exceeds ``turn_rate_min_speed_mps`` (a crawling car can pivot
    # sharply, and heading from Signal deltas is noise below that) and the
    # step is long enough for a reliable heading. OFF by default;
    # ``car_preset()`` turns it on.
    # ------------------------------------------------------------------
    turn_rate_enabled: bool = False
    turn_rate_min_speed_mps: float = 5.0
    turn_rate_safety: float = 2.5       # multiplier on a_lat/v before flagging
                                        # (>=2.5: a real grip-limit corner sits
                                        # ~1x the limit; 2.5x keeps it clear so
                                        # only impossible yaw rates flag)
    turn_rate_min_step_m: float = 2.0   # min displacement for a usable heading
    turn_rate_min_dt_s: float = 0.5     # skip sub-second epochs: heading noise
                                        # (~2*sigma_pos/step) dominates there and
                                        # false-flags straight driving
    # Master switch — when False the filter is a no-op identity (default
    # pipeline behaviour is unchanged unless a caller opts in).
    enabled: bool = True


# Per-epoch outcome flags.
KEEP = "keep"
REPAIR = "repair"          # interpolated across a short bad run
DROP = "drop"              # hard-removed (long bad run); creates a flagged gap


@dataclass
class EpochVerdict:
    """One epoch's filter outcome (parallel to the input ``rows``)."""

    index: int
    utc_s: float
    outcome: str                       # KEEP / REPAIR / DROP
    reasons: list[str] = field(default_factory=list)   # why it was rejected
    gap: bool = False                  # True if this epoch begins/ends a gap


@dataclass
class FilterResult:
    """Output of :func:`robust_filter`."""

    rows: list[PosRow]                 # cleaned path (repaired in place, drops removed)
    verdicts: list[EpochVerdict]       # one per *input* epoch
    n_input: int
    n_kept: int
    n_repaired: int
    n_dropped: int
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_flagged(self) -> int:
        return self.n_repaired + self.n_dropped


# ===========================================================================
# Robust statistics helpers
# ===========================================================================
def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (not scaled). Robust spread estimate."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _enu(rows: Sequence[PosRow], ref: tuple[float, float, float]):
    E = np.empty(len(rows)); N = np.empty(len(rows)); U = np.empty(len(rows))
    for i, r in enumerate(rows):
        e, n, u = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), ref)
        E[i] = e; N[i] = n; U[i] = u
    return E, N, U


# ===========================================================================
# Core detector — per-epoch physical-plausibility gates (GT-FREE)
# ===========================================================================
def detect(
    rows: Sequence[PosRow],
    cfg: Optional[RobustFilterConfig] = None,
    *,
    disagreement: Optional[Sequence[float]] = None,
) -> list[set]:
    """Return a per-epoch set of reject reasons (empty set == clean).

    Reasons:
      ``alt_high`` / ``alt_low``   — altitude outside the robust session window.
      ``vert_speed``               — |vertical speed| to a neighbour too high.
      ``horiz_speed``              — horizontal speed to a neighbour too high.
      ``pos_jump``                 — robust-MAD outlier in position delta.
      ``speed_jump``               — velocity change exceeds the car linear sensor
                                     envelope (speed-aware jump gate; only when
                                     ``cfg.speed_gate_enabled``).
      ``turn_rate``                — implied yaw rate exceeds a_lat/v (only
                                     when ``cfg.turn_rate_enabled``).
      ``disagreement``             — multimask spread above threshold (optional).

    Pure function — no mutation, no GT. ``disagreement`` (if given) must be
    index-aligned with ``rows``.
    """
    cfg = cfg or RobustFilterConfig()
    n = len(rows)
    reasons: list[set] = [set() for _ in range(n)]
    if n == 0:
        return reasons

    ts = np.array([r.utc_s for r in rows])
    h = np.array([r.h_m for r in rows])
    ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)
    E, N, _U = _enu(rows, ref)

    # --- altitude window vs robust session median ---
    h_med = float(np.median(h))
    lo = h_med - cfg.alt_below_median_m
    hi = h_med + cfg.alt_above_median_m
    for i in range(n):
        if h[i] > hi:
            reasons[i].add("alt_high")
        elif h[i] < lo:
            reasons[i].add("alt_low")

    # --- speed gates (per-epoch, against previous epoch with finite dt) ---
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        dh = abs(h[i] - h[i - 1])
        if dh / dt > cfg.max_vert_speed_mps:
            reasons[i].add("vert_speed")
        dEN = math.hypot(E[i] - E[i - 1], N[i] - N[i - 1])
        if dEN / dt > cfg.max_horiz_speed_mps:
            reasons[i].add("horiz_speed")

    # --- robust per-epoch position-jump gate (MAD on horizontal step) ---
    if n >= 3:
        steps = np.zeros(n)
        for i in range(1, n):
            steps[i] = math.hypot(E[i] - E[i - 1], N[i] - N[i - 1])
        med = float(np.median(steps[1:]))
        mad = _mad(steps[1:])
        thr = max(med + cfg.jump_mad_k * mad, cfg.jump_floor_m)
        for i in range(1, n):
            if steps[i] > thr:
                reasons[i].add("pos_jump")

    # --- speed-aware jump gate (car kinematic envelope) ---
    # Max plausible per-epoch displacement = v_ref*dt + linear sensor-envelope term:
    # equivalently the *velocity residual* against the last consistent step
    # must satisfy |v_i - v_ref| <= a_env*elapsed + margin/dt. Comparing
    # against the last *consistent* velocity (not blindly the previous step)
    # avoids echo false-positives on the epoch right after a spike, and lets
    # the gate loosen naturally (elapsed grows) while a run is rejected.
    if cfg.speed_gate_enabled and n >= 3:
        a_env = math.hypot(
            max(cfg.accel_long_max_mps2, cfg.accel_brake_max_mps2),
            cfg.accel_lat_max_mps2,
        )
        ref_v: Optional[tuple[float, float]] = None
        ref_t = 0.0
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            if dt <= 0 or not math.isfinite(dt):
                continue
            v_e = (E[i] - E[i - 1]) / dt
            v_n = (N[i] - N[i - 1]) / dt
            if ref_v is None:
                # Seed only from a kinematically-clean epoch. An epoch already
                # flagged by the alt / speed / pos-jump gates is a spike; using
                # it as the reference would mis-flag the following GOOD epoch.
                if not reasons[i]:
                    ref_v, ref_t = (v_e, v_n), ts[i]
                continue
            elapsed = max(ts[i] - ref_t, dt)
            allowed = a_env * elapsed + cfg.speed_gate_margin_m / dt
            dv = math.hypot(v_e - ref_v[0], v_n - ref_v[1])
            if dv > allowed:
                reasons[i].add("speed_jump")
            elif not reasons[i]:
                # advance the reference only to an otherwise-clean epoch
                ref_v, ref_t = (v_e, v_n), ts[i]

    # --- turn-rate plausibility gate (yaw rate vs lateral-grip limit) ---
    # A car at speed v cannot change heading faster than ~ a_lat_max/v rad/s.
    # Heading comes from consecutive position deltas; only checked when the
    # step is long/fast enough for the heading to be meaningful. Same
    # last-consistent-reference scheme as the speed gate (echo-safe).
    if cfg.turn_rate_enabled and n >= 3:
        ref_h: Optional[float] = None
        ref_ht = 0.0
        ref_spd = 0.0
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            if dt <= 0 or not math.isfinite(dt):
                continue
            if dt < cfg.turn_rate_min_dt_s:
                continue  # sub-second: heading noise dominates, do not judge
            dE = E[i] - E[i - 1]
            dN = N[i] - N[i - 1]
            step = math.hypot(dE, dN)
            spd = step / dt
            if step < cfg.turn_rate_min_step_m or spd < cfg.turn_rate_min_speed_mps:
                continue  # heading unreliable / car may pivot at crawl speed
            head = math.atan2(dE, dN)
            if ref_h is None:
                if not reasons[i]:      # seed only from a clean epoch
                    ref_h, ref_ht, ref_spd = head, ts[i], spd
                continue
            elapsed = max(ts[i] - ref_ht, dt)
            dhead = abs((head - ref_h + math.pi) % (2.0 * math.pi) - math.pi)
            v_eff = max(min(spd, ref_spd), cfg.turn_rate_min_speed_mps)
            max_rate = cfg.turn_rate_safety * cfg.accel_lat_max_mps2 / v_eff
            if dhead / elapsed > max_rate:
                reasons[i].add("turn_rate")
            elif not reasons[i]:
                ref_h, ref_ht, ref_spd = head, ts[i], spd

    # --- optional multimask-disagreement gate ---
    if disagreement is not None:
        dis = np.asarray(disagreement, dtype=float)
        if dis.shape[0] == n:
            for i in range(n):
                if np.isfinite(dis[i]) and dis[i] > cfg.disagreement_reject_m:
                    reasons[i].add("disagreement")

    return reasons


# ===========================================================================
# Repair / drop runs of rejected epochs
# ===========================================================================
def _interp_row(a: PosRow, b: PosRow, t: float) -> PosRow:
    """Linear interpolation of a PosRow between good neighbours ``a`` and ``b``.

    Interpolates lat/lon/h; carries forward the *worse* (more conservative)
    sigma and marks quality as float (2) since the epoch is synthesised.
    """
    span = b.utc_s - a.utc_s
    f = 0.0 if span <= 0 else (t - a.utc_s) / span
    f = min(1.0, max(0.0, f))

    def lerp(x: float, y: float) -> float:
        return x + (y - x) * f

    def worse(x: float, y: float) -> float:
        vals = [v for v in (x, y) if v == v]   # drop NaN
        return max(vals) if vals else float("nan")

    return PosRow(
        utc_s=t,
        lat_deg=lerp(a.lat_deg, b.lat_deg),
        lon_deg=lerp(a.lon_deg, b.lon_deg),
        h_m=lerp(a.h_m, b.h_m),
        quality=2,                       # synthesised -> float, never claim fix
        vn=float("nan"), ve=float("nan"), vu=float("nan"),
        ns=min(a.ns, b.ns),
        sd_n=worse(a.sd_n, b.sd_n),
        sd_e=worse(a.sd_e, b.sd_e),
        sd_u=worse(a.sd_u, b.sd_u),
        age_s=float("nan"), ratio=float("nan"),
    )


def robust_filter(
    rows: Sequence[PosRow],
    cfg: Optional[RobustFilterConfig] = None,
    *,
    disagreement: Optional[Sequence[float]] = None,
    log: Optional[LogFn] = None,
) -> FilterResult:
    """Clean a Post-processing path in place (repair short bad runs, drop long ones).

    GT-FREE. Returns a :class:`FilterResult` whose ``rows`` is the cleaned
    path and ``verdicts`` is one outcome per *input* epoch (so a caller
    can attach gap/interp flags downstream — addresses PP5).
    """
    log_ = make_logger(log)
    cfg = cfg or RobustFilterConfig()
    rows = list(rows)
    n = len(rows)

    verdicts = [EpochVerdict(i, rows[i].utc_s if i < n else 0.0, KEEP)
                for i in range(n)]

    if not cfg.enabled or n == 0:
        return FilterResult(
            rows=list(rows), verdicts=verdicts, n_input=n, n_kept=n,
            n_repaired=0, n_dropped=0, reason_counts={})

    reasons = detect(rows, cfg, disagreement=disagreement)
    bad = [bool(reasons[i]) for i in range(n)]
    for i in range(n):
        verdicts[i].reasons = sorted(reasons[i])

    # Walk contiguous runs of rejected epochs; repair short, drop long.
    out_rows: list[PosRow] = []
    reason_counts: dict[str, int] = {}
    for s in reasons:
        for r in s:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    n_repaired = 0
    n_dropped = 0
    i = 0
    while i < n:
        if not bad[i]:
            out_rows.append(rows[i])
            i += 1
            continue
        # contiguous bad run [i, j)
        j = i
        while j < n and bad[j]:
            j += 1
        run_len = j - i
        # good neighbours bracketing the run
        left = i - 1 if i - 1 >= 0 else None
        right = j if j < n else None
        run_secs = (rows[j - 1].utc_s - rows[i].utc_s) if run_len > 1 else 0.0

        repairable = (
            left is not None and right is not None
            and run_len <= cfg.max_repair_epochs
            and run_secs <= cfg.max_repair_seconds
        )
        if repairable:
            a, b = rows[left], rows[right]
            for k in range(i, j):
                out_rows.append(_interp_row(a, b, rows[k].utc_s))
                verdicts[k].outcome = REPAIR
                n_repaired += 1
            # flag the boundary epochs as a (repaired) gap edge
            verdicts[i].gap = True
            verdicts[j - 1].gap = True
        else:
            for k in range(i, j):
                verdicts[k].outcome = DROP
                verdicts[k].gap = True
                n_dropped += 1
        i = j

    n_kept = sum(1 for v in verdicts if v.outcome == KEEP)
    log_(
        f"[robust_filter] in={n} kept={n_kept} repaired={n_repaired} "
        f"dropped={n_dropped} reasons={reason_counts}"
    )
    return FilterResult(
        rows=out_rows, verdicts=verdicts, n_input=n, n_kept=n_kept,
        n_repaired=n_repaired, n_dropped=n_dropped, reason_counts=reason_counts,
    )


# ===========================================================================
# Pre-smoother cleaning hook — opt-in wrapper (default behaviour unchanged)
# ===========================================================================
def clean_before_smoothing(
    rows: Sequence[PosRow],
    cfg: Optional[RobustFilterConfig] = None,
    *,
    disagreement: Optional[Sequence[float]] = None,
    log: Optional[LogFn] = None,
) -> tuple[list[PosRow], FilterResult]:
    """Convenience pre-smoother step: ``clean -> (cleaned rows, result)``.

    A smoother caller does ``rows, res = clean_before_smoothing(rows, cfg)``
    then feeds ``rows`` to the existing smoother. When ``cfg`` is ``None`` or
    ``cfg.enabled`` is ``False`` the input passes through unchanged.
    """
    res = robust_filter(rows, cfg, disagreement=disagreement, log=log)
    return res.rows, res


# ===========================================================================
# Preset — the shipped, day14-tuned default
# ===========================================================================
def car_preset() -> RobustFilterConfig:
    """The shipped robust_filter preset (car-plausible bounds, repair-first).

    Tuned on day14: catches the s21 altitude/speed blow-ups (PP3/PP4) without
    touching the physically-sane dodge/s20ultra routes (no-harm verified by the
    accuracy harness).

    In addition to the classic threshold gates, this preset enables the two
    car-physics gates (both GT-free, both no-harm on a normally-driven car):

    * **speed-aware jump gate** — per-epoch displacement must be reachable
      from the last consistent velocity within the car linear sensor envelope
      (~4 m/s^2 forward, ~8 m/s^2 braking/lateral). Catches sideways
      teleports that slip under the fixed MAD floor at cruise speed and
      small-but-impossible jumps at standstill.
    * **turn-rate gate** — the implied yaw rate must satisfy the lateral-grip
      limit (<= ~1.5 * a_lat / v rad/s at speed v). A legitimate hard-braking
      or sharp-but-driveable turn stays well inside both envelopes.
    """
    return RobustFilterConfig(
        speed_gate_enabled=True,
        turn_rate_enabled=True,
    )
