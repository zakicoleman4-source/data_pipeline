"""Motion model deep-dive ROUND 3: use Motion model SPEED (not Motion model position/direction) as a
bad-Post-processing-epoch detector.

Rounds 1-2 established:
  - Round 1: velocity-blend fusion using Motion model's *direction* made the
    path WORSE (MAX 11.39 m raw -> 32.0 m fused). Root cause: the
    source->body rotation solve had a 46 deg residual (WARN threshold is
    20 deg) -- Motion model's translation DIRECTION is unreliable on this session.
  - Round 2: swept the Wahba min_speed gate and a wide fixed-R
    yaw/pitch/roll grid trying to fix that rotation. Best MAX achieved was
    31.89 m -- still far worse than raw Post-processing. Motion model direction could not be
    salvaged in post from this footage (forward-driving => near-degenerate
    geometric-consistency geometry, ~43 deg noisy translation direction).
  - BUT: Motion model speed MAGNITUDE (independent of the bad rotation, since
    |R @ v| == |v| for any rotation R) correlated 0.899 with Post-processing-Rate-signal
    speed. Speed is the one trustworthy Motion model signal on this footage.

Round 3 idea: stop trying to use Motion model as a POSITION source. Instead use Motion model
speed as a cross-check ("sanity odometer") on Post-processing's own inter-epoch speed.
Where the two disagree sharply, Post-processing's position for that epoch is suspect
(more likely to be a spiked/degraded fix) -- flag it and repair by linear
interpolation from its (presumed-good) neighbors. This can only help if
the dominant MAX-vs-The reference unit error is a discrete SPIKE (a Post-processing epoch whose
position jumps abruptly and then jumps back) -- a speed-magnitude check is
blind to a smooth, sustained bias (e.g. a float ambiguity that drifts the
position gradually over many epochs at a normal, non-anomalous speed).

STEP 0 (mandatory first step, decides if round 3 can work at all):
  Score raw Post-processing vs The reference unit, find the single worst (MAX) epoch, and inspect
  its neighborhood: horizontal error, Post-processing quality/ns flag, and both Post-processing
  inter-epoch speed and Motion model speed at that epoch. If Post-processing speed is anomalous
  there (spike) -> a speed detector has something to catch. If Post-processing speed
  is normal (smooth bias) -> Motion model speed detection is blind to this failure
  mode by construction and round 3 is diagnosed infeasible up front.

STEP 1: build the detector. Per Post-processing epoch (2..N-1), compute
  ppk_speed[i] = |pos[i] - pos[i-1]| / dt   (position-derived, NOT the
  Rate-signal vn/ve columns -- those can be spiked by the same bad fix that
  spikes the position, so they're not an independent check).
  vio_speed[i] = magnitude of the Post-processing-Rate-signal-scaled Motion model Local-frame velocity
  nearest epoch i (from vio_to_enu_velocities; |R@v|==|v| for any
  rotation, so this magnitude is valid even though round 1-2 showed the
  Motion model *direction* is not).
Flag epoch i as suspect Post-processing when |ppk_speed[i] - vio_speed[i]| exceeds a
threshold (sweep 2, 3, 5 m/s). Repair flagged epochs by linear
interpolation of lat/lon from their nearest unflagged neighbors. Re-score
MAX vs The reference unit for each threshold.

STEP 2: table threshold | n_flagged | MAX_vs_Javad. Compare to raw Post-processing
MAX (11.39 m) and the round-goal 3.0 m gate.

STEP 3: this script and scripts/vio_deepdive_r2.py write ROUND 3's section
of docs/findings/motion model-deepdive.md together (this script does the writing).

This script only CALLS data_pipeline.vio / .traj_score / .parsers /
.stages.user_export -- no production module is edited. The 6-minute
run_vio_multiframe_v2 pass is NOT re-run: samples are loaded from
_vio_deepdive_out/vio_samples_cache.pkl (written by round 2), matching
frame_decim_hz=4.0.
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import vio_deepdive as r1  # round 1: session-map synthesis, CSV export, speed corr
import vio_deepdive_r2 as r2  # round 2: canonical R, cached-sample loading conventions

from data_pipeline.parsers import parse_rtkpos, PosRow
from data_pipeline.traj_score import score_trajectories

OUT_DIR = REPO / "_vio_deepdive_out"
SAMPLES_CACHE = OUT_DIR / "vio_samples_cache.pkl"
FINDINGS_MD = REPO / "docs" / "findings" / "vio-deepdive.md"
FRAME_DECIM_HZ = r1.FRAME_DECIM_HZ

RAW_PPK_MAX_ROUND1 = 11.39
GATE_M = 3.0
THRESHOLDS = (2.0, 3.0, 5.0)


def log(t0: float, m: str) -> None:
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# Post-processing inter-epoch speed (position-derived, independent of the vn/ve Rate-signal
# columns which can be corrupted by the same bad fix that spikes lat/lon).
# ---------------------------------------------------------------------------
def ppk_position_speeds(pos_rows: list[PosRow]) -> list[float]:
    """speeds[i] = |pos[i]-pos[i-1]|/dt for i>=1; speeds[0] = nan."""
    lat0 = pos_rows[0].lat_deg
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    speeds = [float("nan")]
    for i in range(1, len(pos_rows)):
        a, b = pos_rows[i - 1], pos_rows[i]
        dt = b.utc_s - a.utc_s
        if dt <= 0:
            speeds.append(float("nan"))
            continue
        dn = (b.lat_deg - a.lat_deg) * m_per_deg_lat
        de = (b.lon_deg - a.lon_deg) * m_per_deg_lon
        speeds.append(math.hypot(de, dn) / dt)
    return speeds


def vio_speed_at_epochs(pos_rows: list[PosRow], vio_vels: list) -> list[float]:
    """Nearest-in-time Motion model Local-frame speed magnitude for each Post-processing epoch (nan if no
    Motion model sample within 1.0 s)."""
    import bisect
    vio_t = [t for t, _ in vio_vels]
    vio_spd = [math.hypot(float(v[0]), float(v[1])) for _, v in vio_vels]
    out = []
    for r in pos_rows:
        i = bisect.bisect_left(vio_t, r.utc_s)
        best = None
        best_dt = None
        for j in (i - 1, i):
            if 0 <= j < len(vio_t):
                dt = abs(vio_t[j] - r.utc_s)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, vio_spd[j]
        if best is not None and best_dt <= 1.0:
            out.append(best)
        else:
            out.append(float("nan"))
    return out


# ---------------------------------------------------------------------------
# STEP 1 detector: flag + linear-interpolation repair.
# ---------------------------------------------------------------------------
def flag_and_repair(pos_rows: list[PosRow], ppk_spd: list[float],
                     vio_spd: list[float], threshold_mps: float):
    n = len(pos_rows)
    flagged = [False] * n
    for i in range(n):
        p, v = ppk_spd[i], vio_spd[i]
        if math.isfinite(p) and math.isfinite(v) and abs(p - v) > threshold_mps:
            flagged[i] = True

    repaired = list(pos_rows)
    n_flagged = sum(flagged)
    if n_flagged == 0:
        return repaired, 0

    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        # Flagged run is [i, j). Interpolate from nearest good neighbors.
        left = i - 1
        right = j
        if left < 0 and right >= n:
            i = j
            continue  # everything flagged; nothing to anchor to, leave as-is
        if left < 0:
            # No left anchor: hold at right neighbor's position.
            anchor = pos_rows[right]
            for k in range(i, j):
                repaired[k] = _with_latlon(pos_rows[k], anchor.lat_deg, anchor.lon_deg)
        elif right >= n:
            # No right anchor: hold at left neighbor's position.
            anchor = pos_rows[left]
            for k in range(i, j):
                repaired[k] = _with_latlon(pos_rows[k], anchor.lat_deg, anchor.lon_deg)
        else:
            a, b = pos_rows[left], pos_rows[right]
            t0, t1 = a.utc_s, b.utc_s
            span = t1 - t0
            for k in range(i, j):
                al = 0.0 if span <= 0 else (pos_rows[k].utc_s - t0) / span
                lat = a.lat_deg + al * (b.lat_deg - a.lat_deg)
                lon = a.lon_deg + al * (b.lon_deg - a.lon_deg)
                repaired[k] = _with_latlon(pos_rows[k], lat, lon)
        i = j
    return repaired, n_flagged


def _with_latlon(r: PosRow, lat: float, lon: float) -> PosRow:
    return PosRow(
        utc_s=r.utc_s, lat_deg=lat, lon_deg=lon, h_m=r.h_m, quality=r.quality,
        vn=r.vn, ve=r.ve, vu=r.vu, ns=r.ns, sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
    )


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    log(t0, "parsing rover.pos (device PPK) and Javad GT")
    pos_rows = parse_rtkpos(r1.ROVER_POS)
    javad_rows = parse_rtkpos(r1.JAVAD_POS)
    javad_csv = OUT_DIR / "javad_gt.csv"
    r1.export_posrows_csv(javad_rows, javad_csv)
    raw_ppk_csv = OUT_DIR / "raw_ppk.csv"
    r1.export_posrows_csv(pos_rows, raw_ppk_csv)
    raw_score = score_trajectories(javad_csv, raw_ppk_csv)
    log(t0, f"RAW PPK vs Javad: {raw_score}")

    # ---- STEP 0: diagnose the raw-Post-processing MAX epoch -------------------------
    step0 = diagnose_max_epoch(pos_rows, javad_rows, javad_csv, raw_ppk_csv, t0)

    # ---- load cached Motion model samples (written by round 2; DO NOT re-run) ----
    if not SAMPLES_CACHE.is_file():
        log(t0, f"BLOCKED: no cached VIO samples at {SAMPLES_CACHE} -- "
                f"run scripts/vio_deepdive_r2.py first (round 3 must not "
                f"re-run the 6 min VIO pass).")
        _write_blocked(raw_score, step0, "no cached VIO samples on disk")
        return 1
    with open(SAMPLES_CACHE, "rb") as f:
        cached = pickle.load(f)
    if cached.get("frame_decim_hz") != FRAME_DECIM_HZ:
        log(t0, f"BLOCKED: cached samples frame_decim_hz="
                f"{cached.get('frame_decim_hz')} != {FRAME_DECIM_HZ}")
        _write_blocked(raw_score, step0, "cached VIO samples frame_decim_hz mismatch")
        return 1
    samples = cached["samples"]
    log(t0, f"loaded {len(samples)} cached VIO samples from {SAMPLES_CACHE} "
            f"(SKIPPED the ~6min run_vio_multiframe_v2 call)")

    # ---- Motion model Local-frame velocities: Post-processing-Rate-signal-scaled magnitude is what we use;
    #      direction is known-bad (round 1-2), so auto_calibrate/R choice
    #      barely matters here EXCEPT for coverage -- use the round-2 best
    #      fixed canonical R (auto_calibrate=False, deterministic, cheap)
    #      so the magnitude sample set is stable and reproducible.
    from data_pipeline.vio import vio_to_enu_velocities
    R_canon = r2.canonical_R_body_from_cam()
    vio_vels = vio_to_enu_velocities(
        samples, pos_rows, R_body_from_cam=R_canon, auto_calibrate=False,
    )
    log(t0, f"VIO ENU velocity samples: {len(vio_vels)} "
            f"(magnitude only is used below -- direction is known-unreliable "
            f"per round 1-2, but |R@v|==|v| for any rotation R, so speed "
            f"magnitude does not depend on getting R right)")
    speed_corr = r1._speed_corr(pos_rows, vio_vels)
    log(t0, f"VIO-speed vs PPK-Doppler-speed correlation: {speed_corr:.3f}")

    # ---- STEP 1 + 2: threshold sweep -------------------------------------
    ppk_spd = ppk_position_speeds(pos_rows)
    vio_spd = vio_speed_at_epochs(pos_rows, vio_vels)

    results = []
    for thr in THRESHOLDS:
        repaired, n_flagged = flag_and_repair(pos_rows, ppk_spd, vio_spd, thr)
        out_csv = OUT_DIR / f"r3_repaired_thr{thr:g}.csv"
        r1.export_posrows_csv(repaired, out_csv)
        score = score_trajectories(javad_csv, out_csv)
        log(t0, f"threshold={thr:g} m/s: n_flagged={n_flagged} "
                f"MAX={score['max_m']:.3f} m (2sigma={score['two_sigma_m']:.3f})")
        results.append({
            "threshold": thr, "n_flagged": n_flagged,
            "max_m": score["max_m"], "two_sigma_m": score["two_sigma_m"],
            "median_offset_m": score["median_offset_m"], "n": score["n"],
        })

    finite = [r for r in results if math.isfinite(r["max_m"])]
    best = min(finite, key=lambda r: r["max_m"]) if finite else None

    _write_findings(
        raw_score=raw_score, step0=step0, results=results, best=best,
        speed_corr=speed_corr, n_vio_vels=len(vio_vels),
    )
    log(t0, "done")
    return 0


# ---------------------------------------------------------------------------
# STEP 0: worst raw-Post-processing epoch diagnosis.
# ---------------------------------------------------------------------------
def diagnose_max_epoch(pos_rows, javad_rows, javad_csv, raw_ppk_csv, t0) -> dict:
    import numpy as np
    import bisect

    tr = np.array([r.utc_s for r in javad_rows])
    order = np.argsort(tr)
    tr = tr[order]
    latr = np.array([javad_rows[i].lat_deg for i in order])
    lonr = np.array([javad_rows[i].lon_deg for i in order])
    lat0 = float(latr[0])
    mlat = 111320.0
    mlon = mlat * math.cos(math.radians(lat0))

    tt = np.array([r.utc_s for r in pos_rows])
    idx = np.searchsorted(tr, tt)
    de, dn, matched_i = [], [], []
    for k, ti in enumerate(tt):
        cands = [j for j in (idx[k] - 1, idx[k]) if 0 <= j < tr.size]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(tr[j] - ti))
        if abs(tr[j] - ti) > 0.05:
            continue
        dn.append((pos_rows[k].lat_deg - latr[j]) * mlat)
        de.append((pos_rows[k].lon_deg - lonr[j]) * mlon)
        matched_i.append(k)
    de = np.array(de); dn = np.array(dn)
    off_e = float(np.median(de)); off_n = float(np.median(dn))
    err = np.hypot(de - off_e, dn - off_n)
    worst_local = int(np.argmax(err))
    k = matched_i[worst_local]
    worst_row = pos_rows[k]
    worst_err = float(err[worst_local])

    # Post-processing inter-epoch speed around the worst epoch (position-derived).
    ppk_spd_all = ppk_position_speeds(pos_rows)
    spd_before = ppk_spd_all[k] if k < len(ppk_spd_all) else float("nan")
    spd_after = ppk_spd_all[k + 1] if k + 1 < len(ppk_spd_all) else float("nan")

    # Neighborhood median speed (excluding the epoch itself) as a
    # "normal driving speed here" baseline for comparison.
    lo, hi = max(0, k - 10), min(len(ppk_spd_all), k + 11)
    nbhd = [s for idx2, s in enumerate(ppk_spd_all[lo:hi], start=lo)
            if idx2 not in (k,) and math.isfinite(s)]
    nbhd_median = float(np.median(nbhd)) if nbhd else float("nan")

    is_spike = (
        math.isfinite(spd_before) and math.isfinite(nbhd_median)
        and nbhd_median > 0.5
        and (spd_before > 2.0 * nbhd_median or
             (math.isfinite(spd_after) and spd_after > 2.0 * nbhd_median))
    )

    step0 = {
        "worst_err_m": worst_err,
        "gpstime_utc_s": worst_row.utc_s,
        "quality": worst_row.quality,
        "ns": worst_row.ns,
        "sd_n": worst_row.sd_n,
        "sd_e": worst_row.sd_e,
        "ppk_speed_before_mps": spd_before,
        "ppk_speed_after_mps": spd_after,
        "neighborhood_median_speed_mps": nbhd_median,
        "is_spike": is_spike,
        "epoch_index": k,
    }
    log(t0, f"STEP0 worst raw-PPK-vs-Javad epoch: err={worst_err:.3f} m "
            f"gpstime={worst_row.utc_s:.3f} quality={worst_row.quality} "
            f"ns={worst_row.ns} sd_n={worst_row.sd_n:.3f} sd_e={worst_row.sd_e:.3f} "
            f"ppk_speed(before/after)={spd_before:.2f}/{spd_after:.2f} m/s "
            f"neighborhood_median={nbhd_median:.2f} m/s "
            f"-> {'SPIKE' if is_spike else 'SMOOTH (no speed anomaly)'}")
    return step0


def _write_blocked(raw_score, step0, error: str) -> None:
    lines = [
        "## ROUND 3 -- BLOCKED",
        "",
        f"- RAW PPK vs Javad MAX = **{raw_score['max_m']:.3f} m**",
        f"- STEP 0 diagnosis (worst epoch): {step0}",
        f"- ROUND 3: BLOCKED -- {error}",
        "",
        "See `scripts/vio_deepdive_r3.py` for the exact repro.",
    ]
    _append_or_write(lines)


def _append_or_write(new_lines: list[str]) -> None:
    existing = ""
    if FINDINGS_MD.is_file():
        existing = FINDINGS_MD.read_text(encoding="utf-8")
        marker = "\n## ROUND 3"
        idx = existing.find(marker)
        if idx != -1:
            existing = existing[:idx].rstrip() + "\n"
    FINDINGS_MD.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(new_lines) + "\n",
        encoding="utf-8",
    )


def _write_findings(*, raw_score, step0, results, best, speed_corr, n_vio_vels) -> None:
    gate_met = best is not None and best["max_m"] <= GATE_M
    is_spike = step0["is_spike"]

    lines = [
        "## ROUND 3 -- VIO-speed bad-PPK-epoch detector",
        "",
        "**Date:** 2026-07-02 · **Session:** day14 `dodge/20260628_190336_677` "
        "· **Tool:** `scripts/vio_deepdive_r3.py` (reuses round 1's "
        "recording_map synthesis / CSV export / speed-corr helpers and "
        "round 2's canonical `R_body_from_cam` via import; loads the "
        "**cached** VIO samples from `_vio_deepdive_out/vio_samples_cache.pkl` "
        "-- the ~6 min `run_vio_multiframe_v2` pass was NOT re-run) · "
        "`data_pipeline/vio.py` was NOT modified, only called.",
        "",
        "### Why round 3",
        "",
        "Rounds 1-2 showed VIO's translation **direction** cannot be trusted "
        "on this forward-driving footage (best camera->body rotation still "
        "left a >=40 deg residual; velocity-blend fusion using that "
        "direction made MAX worse, not better: 11.39 m raw -> 31.89 m best "
        "fused). But VIO **speed magnitude** correlated 0.899 with "
        "PPK-Doppler speed in round 1 -- and speed magnitude does not "
        "depend on `R_body_from_cam` at all (`|R @ v| == |v|` for any "
        "rotation `R`). Round 3 stops trying to use VIO as a position "
        "source and instead uses VIO speed purely as a **cross-check**: "
        "where PPK's own inter-epoch speed disagrees sharply with VIO's "
        "independently-measured speed, that PPK epoch is flagged as "
        "suspect and repaired by linear interpolation from its neighbors.",
        "",
        "### STEP 0 -- is the raw-PPK MAX epoch a spike or a bias?",
        "",
        "This is the gating question: a speed-magnitude detector can only "
        "ever catch a PPK epoch whose *speed* is anomalous (a discrete "
        "position spike / jump-and-return). It is blind by construction to "
        "a smooth, sustained float-ambiguity bias that drifts position "
        "gradually while the reported speed stays normal.",
        "",
        f"Worst raw-PPK-vs-Javad epoch (after median-offset removal, same "
        f"convention as `score_trajectories`): horizontal error = "
        f"**{step0['worst_err_m']:.3f} m** at gpstime (utc_s) = "
        f"{step0['gpstime_utc_s']:.3f}, PPK quality flag = "
        f"{step0['quality']}, ns (sats used) = {step0['ns']}, "
        f"sd_n={step0['sd_n']:.3f} m, sd_e={step0['sd_e']:.3f} m.",
        "",
        f"- PPK inter-epoch speed immediately before/after this epoch: "
        f"{step0['ppk_speed_before_mps']:.2f} / "
        f"{step0['ppk_speed_after_mps']:.2f} m/s.",
        f"- Neighborhood (+-10 epoch) median PPK speed: "
        f"{step0['neighborhood_median_speed_mps']:.2f} m/s.",
        "",
        f"**Diagnosis: {'SPIKE' if is_spike else 'SMOOTH BIAS'}.** "
        + (
            "The PPK speed at the worst epoch is >2x the local neighborhood "
            "median (or the recovery epoch immediately after is), i.e. the "
            "position jumps abruptly and then jumps back -- a speed anomaly "
            "a VIO cross-check CAN in principle catch."
            if is_spike else
            "PPK speed at the worst epoch is in line with the local "
            "neighborhood median -- the position error there is NOT "
            "accompanied by an anomalous inter-epoch speed. This means "
            "the MAX-error epoch is (most likely) a smooth, sustained "
            "float-ambiguity/multipath BIAS the receiver reports at a "
            "perfectly normal apparent speed, not a discrete jump. A "
            "speed-magnitude detector is blind to this failure mode by "
            "construction: nothing about |PPK_speed - VIO_speed| exceeding "
            "a threshold can fire on an epoch whose speed was never "
            "anomalous in the first place."
        ),
        "",
        "### STEP 1+2 -- detector sweep",
        "",
        "Per epoch: `ppk_speed[i] = |pos[i]-pos[i-1]| / dt` (position-"
        "derived, not the `vn`/`ve` Doppler columns -- those can be spiked "
        "by the same bad fix that spikes the position, so they aren't an "
        "independent check). `vio_speed[i]` = magnitude of the nearest "
        "PPK-Doppler-scaled VIO ENU velocity sample (round-2 canonical "
        f"`R_body_from_cam`, `auto_calibrate=False`; n={n_vio_vels} "
        f"samples, speed_corr={speed_corr:.3f} vs PPK Doppler, matching "
        "round 1). Epochs where `|ppk_speed[i] - vio_speed[i]| > threshold` "
        "are flagged and repaired by linear interpolation of lat/lon from "
        "their nearest unflagged neighbors.",
        "",
        "| threshold (m/s) | n_flagged | MAX vs Javad (m) | 2sigma (m) |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['threshold']:g} | {r['n_flagged']} | "
            f"{r['max_m']:.3f} | {r['two_sigma_m']:.3f} |"
        )
    lines += [
        "",
        f"Raw PPK MAX (no repair): **{raw_score['max_m']:.3f} m**.",
        "",
    ]
    if best is not None:
        beat_raw = best["max_m"] < raw_score["max_m"]
        lines += [
            f"Best detector: threshold={best['threshold']:g} m/s, "
            f"n_flagged={best['n_flagged']}, MAX = **{best['max_m']:.3f} m** "
            f"({'beats' if beat_raw else 'does NOT beat'} raw PPK MAX "
            f"{raw_score['max_m']:.3f} m, delta "
            f"{raw_score['max_m'] - best['max_m']:+.3f} m).",
            "",
        ]
    else:
        lines += ["No threshold produced a finite MAX (unexpected).", ""]

    lines += [
        "### Gate: MAX <= 3.0 m",
        "",
        f"**{'PASS' if gate_met else 'NOT MET'}** -- best detector MAX = "
        f"{best['max_m']:.3f} m {'<=' if gate_met else '>'} 3.0 m."
        if best is not None else "**NOT MET** -- no finite result.",
        "",
        "### Verdict",
        "",
    ]
    if is_spike and gate_met:
        lines.append(
            "The MAX-error epoch is a genuine PPK position spike, and the "
            "VIO-speed detector caught and repaired it (or a similarly-bad "
            "epoch) enough to bring MAX under the 3.0 m gate. Round 3 "
            "closes the loop for this session."
        )
    elif is_spike and not gate_met:
        lines.append(
            "STEP 0 confirms the MAX epoch IS a speed-detectable spike, but "
            "the sweep above did not bring MAX under 3.0 m -- either the "
            "detector's interpolation repair is too coarse (a straight-line "
            "interpolation over a multi-epoch flagged run does not "
            "reconstruct the true path), or a second, non-spike error "
            "source (e.g. a bias elsewhere in the trajectory) is now the "
            "binding constraint on MAX once the spike is fixed. Re-run "
            "STEP 0's diagnosis against the BEST-threshold repaired "
            "trajectory (not raw PPK) to see whether the new MAX epoch is "
            "itself a spike or a bias, before choosing the next lever."
        )
    else:
        lines.append(
            "**STEP 0 shows the raw-PPK MAX epoch is a smooth bias, not a "
            "speed-detectable spike.** By construction, a VIO-speed "
            "cross-check cannot fix this: it only fires when PPK's "
            "reported speed disagrees with VIO's, and this epoch's speed "
            "was never anomalous. This is stated plainly: **the 3.0 m gate "
            "is not reachable via VIO-speed bad-epoch detection on this "
            "session.** The lever this round set out to test (\"VIO speed "
            "is trustworthy even though VIO direction/position isn't\") is "
            "real and confirmed again here (speed_corr="
            f"{speed_corr:.3f}), but a speed cross-check is the wrong tool "
            "for a bias-dominated MAX error -- it needs an independent "
            "*position* or *heading* correction, which is exactly what "
            "rounds 1-2 already showed VIO cannot supply on this footage."
        )

    lines += [
        "",
        "### Next lever",
        "",
    ]
    if is_spike and not gate_met:
        lines.append(
            "1. Diagnose the post-repair MAX epoch (repeat STEP 0 against "
            "the best-threshold repaired CSV) to see if a second spike or "
            "a bias is now binding.\n"
            "2. Try a smaller flagged-run repair (single-epoch drop + "
            "interpolate) instead of repairing an entire contiguous "
            "flagged run at once, in case a multi-epoch run's endpoints "
            "are themselves mildly biased.\n"
            "3. Tighten `max_dt_s` / neighbor-matching in the VIO-speed "
            "lookup (`vio_speed_at_epochs` currently accepts up to 1.0 s "
            "away) to reduce false negatives in low-VIO-coverage stretches."
        )
    else:
        lines.append(
            "Bias-dominated PPK epochs need an independent absolute "
            "position or heading reference to correct, not a speed "
            "cross-check. Candidates: (a) wire VIO's PPK-Doppler-scaled "
            "speed (this round's validated signal) into `epoch_weight_v2` "
            "as a down-weighting factor for RTKLIB's own float/fix "
            "quality flag rather than an independent detector -- i.e. use "
            "it to decide how much to TRUST an epoch, not to relocate it; "
            "(b) investigate the raw RTKLIB solve config for this session "
            "(elevation mask, ambiguity-fix settings) since a smooth bias "
            "across many epochs points at the PPK solve itself rather than "
            "anything fixable in post-processing; (c) accept that for this "
            "specific dodge190336 session the 3.0 m MAX gate is not "
            "reachable without re-solving PPK with different settings or "
            "adding a genuinely independent sensor (not VIO speed alone)."
        )

    lines += [
        "",
        "### Artifacts",
        "",
        "- Script: `scripts/vio_deepdive_r3.py` (imports "
        "`scripts/vio_deepdive.py` and `scripts/vio_deepdive_r2.py`; does "
        "not duplicate their logic).",
        "- Reused cache: `_vio_deepdive_out/vio_samples_cache.pkl` "
        "(written by round 2; round 3 only loads it).",
        f"- Per-threshold repaired CSVs: "
        f"`_vio_deepdive_out/r3_repaired_thr<N>.csv` ({len(results)} "
        "thresholds).",
    ]
    _append_or_write(lines)


if __name__ == "__main__":
    raise SystemExit(main())
