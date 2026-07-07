"""Tests for the automotive (car-physics) path-filter improvements.

Covers the two new GT-free gates in :mod:`data_pipeline.robust_filter`
(speed-aware jump gate + turn-rate plausibility gate, both enabled by the
improved ``car_preset()``), the opt-in automotive tuning of
:mod:`data_pipeline.epoch_weight_v2`, and API/backward-compat stability:

* a physically-impossible LATERAL teleport at cruise speed slips past the old
  fixed MAD floor / speed thresholds but is caught by the new gates;
* a legitimate HARD-BRAKING event (large but physically valid deceleration)
  is preserved untouched;
* on a noisy straight+turn car track with injected outliers the improved
  ``car_preset()`` yields lower horizontal RMSE than the pre-existing bare
  config;
* all pre-existing public APIs remain callable with their old signatures.
"""
from __future__ import annotations

import math

import numpy as np

from data_pipeline.geo import ecef_to_enu, llh_to_ecef
from data_pipeline.parsers import PosRow
from data_pipeline.robust_filter import (
    DROP,
    KEEP,
    REPAIR,
    RobustFilterConfig,
    car_preset,
    clean_before_smoothing,
    detect,
    robust_filter,
)


LAT0, LON0, H0 = 32.06, 34.80, 47.0
MLAT = 111_320.0
MLON = 111_320.0 * math.cos(math.radians(LAT0))
REF = (LAT0, LON0, H0)


def _row(t, east_m, north_m, h=H0, q=1, ns=14):
    return PosRow(
        utc_s=float(t),
        lat_deg=LAT0 + north_m / MLAT,
        lon_deg=LON0 + east_m / MLON,
        h_m=float(h),
        quality=q, ns=ns,
        sd_n=0.3, sd_e=0.3, sd_u=0.5,
    )


def _en(r: PosRow) -> tuple[float, float]:
    e, n, _ = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m), REF)
    return e, n


def _speed_track(speeds, headings_deg=None, noise=0.0, rng=None):
    """1 Hz track: epoch i moves ``speeds[i]`` metres along ``headings[i]``.

    Returns (rows, truth) where truth is the noiseless (E, N) per epoch.
    """
    n = len(speeds)
    if headings_deg is None:
        headings_deg = [90.0] * n          # due east
    E = N = 0.0
    rows, truth = [], []
    for i in range(n):
        if i > 0:
            h = math.radians(headings_deg[i])
            E += speeds[i] * math.sin(h)
            N += speeds[i] * math.cos(h)
        truth.append((E, N))
        dE = dN = 0.0
        if noise and rng is not None:
            dE = noise * rng.standard_normal()
            dN = noise * rng.standard_normal()
        rows.append(_row(i, east_m=E + dE, north_m=N + dN))
    return rows, truth


def _var_speed_profile():
    """Accelerate 0->30, cruise 30, brake to 6, cruise 6 (all 1 Hz, eastward).

    The wide speed spread inflates the step-MAD so the OLD fixed jump gate's
    threshold (med + k*MAD) sits far above any single-epoch step — exactly the
    regime where the old config goes blind to lateral teleports at speed.
    """
    speeds = []
    speeds += [2.0 * t for t in range(15)]          # 0..28 linear sensor @ 2 m/s^2
    speeds += [30.0] * 45                            # cruise
    speeds += [30.0 - 2.0 * (t + 1) for t in range(12)]   # brake to 6
    speeds += [6.0] * 30                             # slow cruise
    return speeds


# ---------------------------------------------------------------------------
# 1) Impossible LATERAL teleport at speed: slips the old gates, caught by new
# ---------------------------------------------------------------------------
def test_lateral_teleport_caught_by_new_gates_missed_by_old():
    speeds = _var_speed_profile()
    rows, truth = _speed_track(speeds)
    bad_i = 40                          # mid-cruise @ 30 m/s
    # Teleport 25 m sideways while continuing east: step = hypot(30, 25)
    # = 39.05 m -> under the 40 m/s horiz_speed gate AND under the MAD
    # threshold (med + 8*MAD >> 39 on this variable-speed track).
    e, n = truth[bad_i]
    rows[bad_i] = _row(bad_i, east_m=e, north_m=n + 25.0)

    old_cfg = RobustFilterConfig()      # pre-existing bare config
    reasons_old = detect(rows, old_cfg)
    assert reasons_old[bad_i] == set(), (
        f"precondition: old config must miss the lateral teleport, "
        f"got {reasons_old[bad_i]}"
    )
    assert reasons_old[bad_i + 1] == set()   # recovery epoch also slips

    # End-to-end: the old filter keeps the 25 m error verbatim.
    res_old = robust_filter(rows, old_cfg)
    assert res_old.verdicts[bad_i].outcome == KEEP
    err_old = max(abs(_en(r)[1] - truth[i][1])
                  for i, r in enumerate(res_old.rows))
    assert err_old > 20.0

    # New car preset: both car-physics gates fire.
    reasons_new = detect(rows, car_preset())
    assert "speed_jump" in reasons_new[bad_i]
    assert "turn_rate" in reasons_new[bad_i]

    res_new = robust_filter(rows, car_preset())
    assert res_new.verdicts[bad_i].outcome in (REPAIR, DROP)
    # repaired back onto the straight line
    kept_err = max(abs(_en(r)[1] - 0.0) for r in res_new.rows)
    assert kept_err < 5.0


def test_low_speed_hop_missed_by_old_mad_floor_caught():
    """A 20 m sideways hop during the slow (6 m/s) segment of a mixed-speed
    session: the session's step-MAD threshold sits ~80 m up (blind), and the
    step (hypot(6, 20) = 20.9 m) is far under the 40 m/s speed gate — the old
    config keeps it. The speed-aware gate flags it: from 6 m/s a car cannot
    displace 20 m laterally in 1 s (envelope ~ a_env*dt + margin = 14.3 m)."""
    speeds = _var_speed_profile()
    rows, truth = _speed_track(speeds)
    bad_i = 85                          # inside the 6 m/s slow cruise
    e, n = truth[bad_i]
    rows[bad_i] = _row(bad_i, east_m=e, north_m=n + 20.0)

    old_cfg = RobustFilterConfig()
    assert detect(rows, old_cfg)[bad_i] == set(), "old config must miss the hop"

    reasons_new = detect(rows, car_preset())
    assert "speed_jump" in reasons_new[bad_i]
    res = robust_filter(rows, car_preset())
    assert res.verdicts[bad_i].outcome in (REPAIR, DROP)


# ---------------------------------------------------------------------------
# 2) Legitimate hard braking is preserved (no false positive)
# ---------------------------------------------------------------------------
def test_hard_braking_preserved():
    # Cruise 25 m/s, slam brakes at 7.5 m/s^2 for 3 s down to 2.5 m/s, crawl.
    speeds = [25.0] * 40
    speeds += [25.0 - 7.5 * (t + 0.5) for t in range(3)]   # 21.25, 13.75, 6.25
    speeds += [2.5] * 27
    rows, _ = _speed_track(speeds)

    reasons = detect(rows, car_preset())
    assert all(rs == set() for rs in reasons), (
        f"hard braking falsely flagged: "
        f"{[(i, rs) for i, rs in enumerate(reasons) if rs]}"
    )
    res = robust_filter(rows, car_preset())
    assert res.n_repaired == 0
    assert res.n_dropped == 0
    assert all(v.outcome == KEEP for v in res.verdicts)
    for a, b in zip(rows, res.rows):
        assert a.lat_deg == b.lat_deg and a.lon_deg == b.lon_deg


def test_legitimate_turn_preserved():
    # 90-degree city corner over 10 s at 7 m/s (a_lat ~ 1.1 m/s^2) — legal.
    n = 60
    speeds = [7.0] * n
    headings = [90.0] * 25 + [90.0 - 9.0 * (t + 1) for t in range(10)] + [0.0] * 25
    rows, _ = _speed_track(speeds, headings)
    reasons = detect(rows, car_preset())
    assert all(rs == set() for rs in reasons)
    res = robust_filter(rows, car_preset())
    assert res.n_flagged == 0


# ---------------------------------------------------------------------------
# 3) Noisy straight+turn track with outliers: car_preset beats old bare config
# ---------------------------------------------------------------------------
def test_car_preset_lower_rmse_than_old_config_on_noisy_track():
    rng = np.random.default_rng(42)
    # City straight (8 m/s), 90-deg turn, accelerate, highway straight (25 m/s).
    speeds = [8.0] * 30
    headings = [90.0] * 30
    speeds += [8.0] * 10                                     # turning
    headings += [90.0 - 9.0 * (t + 1) for t in range(10)]     # east -> north
    speeds += [8.0 + 1.7 * (t + 1) for t in range(10)]        # linear sensor to 25
    headings += [0.0] * 10
    speeds += [25.0] * 50                                     # highway
    headings += [0.0] * 50
    rows, truth = _speed_track(speeds, headings, noise=0.35, rng=rng)

    # Inject lateral teleports sized to slip the OLD gates (step < 40 m/s and
    # far under the variable-speed MAD threshold) at both speed regimes.
    outliers = {15: (0.0, 18.0), 70: (20.0, 0.0), 85: (-20.0, 0.0)}
    for i, (dE, dN) in outliers.items():
        e, n = truth[i]
        rows[i] = _row(i, east_m=e + dE, north_m=n + dN)

    old_cfg = RobustFilterConfig()
    for i in outliers:
        assert detect(rows, old_cfg)[i] == set(), (
            f"precondition: outlier at {i} must slip the old config"
        )

    def _hrmse(res):
        errs = []
        t2truth = {float(i): truth[i] for i in range(len(truth))}
        for r in res.rows:
            te, tn = t2truth[r.utc_s]
            e, n = _en(r)
            errs.append((e - te) ** 2 + (n - tn) ** 2)
        return math.sqrt(sum(errs) / len(errs))

    rmse_old = _hrmse(robust_filter(rows, old_cfg))
    res_new = robust_filter(rows, car_preset())
    rmse_new = _hrmse(res_new)

    # New preset caught + repaired the teleports the old config kept.
    for i in outliers:
        assert res_new.verdicts[i].outcome in (REPAIR, DROP)
    assert rmse_new < rmse_old, (
        f"car_preset RMSE {rmse_new:.3f} must beat old config {rmse_old:.3f}"
    )
    assert rmse_new < 0.7 * rmse_old


# ---------------------------------------------------------------------------
# 4) API stability / backward compatibility
# ---------------------------------------------------------------------------
def test_car_preset_returns_config_and_old_apis_still_work():
    cfg = car_preset()
    assert isinstance(cfg, RobustFilterConfig)
    # new car-physics gates are ON in the preset, OFF in the bare config
    assert cfg.speed_gate_enabled and cfg.turn_rate_enabled
    bare = RobustFilterConfig()
    assert not bare.speed_gate_enabled and not bare.turn_rate_enabled

    rows, _ = _speed_track([13.0] * 30)
    # old call shapes still work
    res = robust_filter(rows)
    assert res.n_kept == len(rows)
    res2 = robust_filter(rows, RobustFilterConfig(), disagreement=None, log=None)
    assert res2.n_kept == len(rows)
    cleaned, fr = clean_before_smoothing(rows, car_preset())
    assert len(cleaned) == len(rows)
    # the shipped export preset's kwargs (user_export.py) still construct
    RobustFilterConfig(
        max_horiz_speed_mps=45.0, max_vert_speed_mps=8.0,
        alt_below_median_m=30.0, alt_above_median_m=40.0,
        jump_mad_k=6.0, jump_floor_m=8.0, max_repair_epochs=10,
        max_repair_seconds=12.0, disagreement_reject_m=5.0, enabled=True,
    )


def _v2_rows(n=60, v_north=30.0, noise=0.0, rng=None):
    rows = []
    for i in range(n):
        north = v_north * i + (noise * rng.standard_normal() if rng else 0.0)
        east = noise * rng.standard_normal() if rng else 0.0
        rows.append(PosRow(
            utc_s=float(i),
            lat_deg=LAT0 + north / MLAT, lon_deg=LON0 + east / MLON,
            h_m=H0, quality=1, vn=v_north, ve=0.0, vu=0.0, ns=12,
            sd_n=0.3, sd_e=0.3, sd_u=0.5,
            sd_vn=0.1, sd_ve=0.1, sd_vu=0.1,
        ))
    return rows


def test_v2_smoother_old_args_and_automotive_optin():
    from data_pipeline.epoch_weight_v2 import (
        EpochWeightV2Options,
        automotive_v2_options,
        smooth_epoch_weighted_v2,
    )

    # New fields default to no-op values (existing behaviour unchanged).
    opts = EpochWeightV2Options()
    assert opts.accel_env_limit_mps2 is None
    assert opts.nhc_speed_adaptive is False

    rng = np.random.default_rng(7)
    rows = _v2_rows(n=60, v_north=30.0, noise=0.6, rng=rng)

    # Old call signature still works.
    res_def = smooth_epoch_weighted_v2(rows)
    assert len(res_def.E_smooth) == len(rows)
    assert all(math.isfinite(v) for v in res_def.E_smooth)
    res_kw = smooth_epoch_weighted_v2(rows, None, options=EpochWeightV2Options(),
                                      log=None)
    assert np.allclose(res_def.E_smooth, res_kw.E_smooth)

    # Automotive tuning: documented opt-in, same public entry point.
    auto = automotive_v2_options()
    assert isinstance(auto, EpochWeightV2Options)
    assert auto.accel_env_limit_mps2 == 9.0 and auto.nhc_speed_adaptive
    res_auto = smooth_epoch_weighted_v2(rows, options=auto)
    assert len(res_auto.E_smooth) == len(rows)
    assert all(math.isfinite(v) for v in res_auto.E_smooth)
    assert all(math.isfinite(v) for v in res_auto.N_smooth)
    assert res_auto.n_nhc_updates > 0
    # the opt-in path genuinely changes behaviour at highway speed...
    assert not np.allclose(res_auto.E_smooth, res_def.E_smooth)
    # ...and the tighter high-speed NHC does not degrade the lateral track
    # (truth east == 0 on this straight northward drive).
    lat_def = float(np.sqrt(np.mean(np.square(res_def.E_smooth))))
    lat_auto = float(np.sqrt(np.mean(np.square(res_auto.E_smooth))))
    assert lat_auto <= lat_def * 1.05


# ---------------------------------------------------------------------------
# Hardening (2026-07-05): clean-epoch reference seeding + turn-rate dt guard.
# ---------------------------------------------------------------------------
def _track(es, ns, dt=1.0, t0=0.0):
    rows = []
    for i, (e, n) in enumerate(zip(es, ns)):
        rows.append(PosRow(
            utc_s=t0 + i * dt, lat_deg=LAT0 + n / MLAT, lon_deg=LON0 + e / MLON,
            h_m=H0, quality=1, ns=10, vn=0.0, ve=0.0, vu=0.0,
            sd_n=0.1, sd_e=0.1, sd_u=0.2,
        ))
    return rows


def _arc(speed, dt, headings, t0=0.0):
    e = n = 0.0
    es = [0.0]; ns = [0.0]
    for h in headings:
        e += speed * dt * math.sin(h); n += speed * dt * math.cos(h)
        es.append(e); ns.append(n)
    return _track(es, ns, dt=dt, t0=t0)


def test_speed_gate_seed_skips_spike_so_next_good_epoch_not_flagged():
    # First step is a spike (30 m in 1 s); the FOLLOWING clean 10 m step must
    # NOT be flagged speed_jump — the reference may not be seeded from the spike.
    rows = _track(es=[0, 0, 0, 0, 0, 0], ns=[0, 30, 40, 50, 60, 70])
    reasons = detect(rows, RobustFilterConfig(speed_gate_enabled=True, enabled=True))
    assert "pos_jump" in reasons[1]              # the spike itself is caught
    assert "speed_jump" not in reasons[2]        # the good epoch after is clean


def test_turn_rate_skips_subsecond_epochs():
    # dt = 0.2 s straight drive with heading jitter: heading noise dominates,
    # the gate must not judge it (would false-flag straight driving).
    es = [0.3 if i % 2 else -0.3 for i in range(12)]
    ns = [i * 3.0 for i in range(12)]            # 15 m/s north
    rows = _track(es, ns, dt=0.2)
    reasons = detect(rows, car_preset())
    assert not any("turn_rate" in s for s in reasons)


def test_turn_rate_preserves_grip_limit_corner():
    # A real corner AT the lateral-grip limit (yaw rate = a_lat/v) at 1 Hz must
    # survive: straight-line repair of a real corner ships a position error.
    rows = _arc(speed=15.0, dt=1.0, headings=[i * (8.0 / 15.0) for i in range(8)])
    reasons = detect(rows, car_preset())
    assert not any("turn_rate" in s for s in reasons)


def test_turn_rate_flags_impossible_heading_reversal():
    # An instantaneous ~180 deg heading reversal at speed is physically
    # impossible and must still be flagged.
    rows = _arc(speed=15.0, dt=1.0, headings=[0.0, 0.0, 0.0, math.pi, 0.0])
    reasons = detect(rows, car_preset())
    assert any("turn_rate" in s for s in reasons)
