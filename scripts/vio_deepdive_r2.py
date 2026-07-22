"""Motion model deep-dive ROUND 2: get the source->body rotation right.

Round 1 (`scripts/vio_deepdive.py`) established:
  - day14 dodge190336's `recording_*.txt` is 0 bytes; a synthesized
    recording_map (boot->UTC bridge from measurements Fix rows +
    video_anchor.txt + capture_meta.json) works around it.
  - `run_vio_multiframe_v2` + `vio_to_enu_velocities(auto_calibrate=True)`
    (default `min_speed_mps=3.0` inside `calibrate_R_body_from_cam`)
    produced a Wahba-solved R_body_from_cam with **46 deg** p50 residual
    (mount likely rotated/ambiguous at that speed gate) and a velocity-blend
    fused path that was WORSE than raw Post-processing: MAX 32 m vs 11.39 m raw.
    Motion model speed MAGNITUDE was good (corr 0.899 vs Post-processing Rate-signal) -- the problem
    is specifically the source->body ROTATION, not the Motion model pipeline itself.

Round 2 goal: fix R_body_from_cam so the fused MAX comes down, ideally to
<= 3.0 m. This script only CALLS data_pipeline.vio (run_vio_multiframe_v2,
vio_to_enu_velocities, calibrate_R_body_from_cam) -- vio.py is never edited.

KEY EFFICIENCY: run_vio_multiframe_v2 is run EXACTLY ONCE (~6 min) and the
resulting VioSample list is cached in memory (and pickled to disk so a
second invocation of this script skips the Motion model run entirely). Every
variant below (min_speed sweep, fixed canonical R + yaw/pitch/roll grid)
reuses the same cached samples -- only vio_to_enu_velocities + the
round-1 velocity-blend fusion + scorer are re-run per variant, which is
cheap (< 1s each).

Variants
--------
1. min_speed_mps sweep for calibrate_R_body_from_cam (Wahba / auto_calibrate
   path): {1.5, 2.5, 5.0, 8.0}. Report residual + fused MAX for each.
2. Fixed canonical mount R (forward-facing landscape device: source
   +Z=forward, +X=right, +Y=down; vehicle body +X=forward, +Y=left,
   +Z=up) passed to vio_to_enu_velocities(R_body_from_cam=R,
   auto_calibrate=False) -- bypasses the Wahba solve entirely. Also
   grid-search +-10 deg yaw/pitch/roll perturbations of that canonical R
   to minimize fused MAX.
3. Report the best variant overall (lowest fused MAX vs The reference unit).
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reuse round-1's data plumbing (recording_map synthesis, minimal-CSV
# export, velocity-blend fusion, speed-correlation diagnostic) by import
# -- no round-1 code is duplicated or edited.
sys.path.insert(0, str(REPO / "scripts"))
import vio_deepdive as r1  # round 1 script, read-only import

from data_pipeline.parsers import parse_rtkpos, parse_imu, PosRow
from data_pipeline.traj_score import score_trajectories

OUT_DIR = REPO / "_vio_deepdive_out"
SAMPLES_CACHE = OUT_DIR / "vio_samples_cache.pkl"
FINDINGS_MD = REPO / "docs" / "findings" / "vio-deepdive.md"

FRAME_DECIM_HZ = r1.FRAME_DECIM_HZ


def log(t0: float, m: str) -> None:
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# Rotation helpers (numpy-only; no scipy dependency needed for a 3x3 grid).
# ---------------------------------------------------------------------------
def _Rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def canonical_R_body_from_cam() -> np.ndarray:
    """Forward-facing landscape device mount, source -> vehicle body.

    Source axes (The feature library pinhole convention, matches vio.py's docstring):
        +X = right, +Y = down, +Z = forward

    Vehicle body axes (this repo's Local-frame-heading convention in
    vio_to_enu_velocities: body.X = forward, body.Y = right [NOTE: read
    the ve/vn formula in vio_to_enu_velocities -- it treats body.Y as
    RIGHT, not left: ve = sh*bx + ch*by, vn = ch*bx - sh*by, which is the
    standard forward/right heading-rotation, i.e. a right-handed
    NED-style body sample with X=forward, Y=right, Z=down]:
        +X = forward, +Y = right, +Z = down

    So for a forward-facing landscape device (source pointed out the
    windshield, device held landscape so source "up" in the image lines up
    with vehicle "up"):
        body.X (forward) = cam.Z (forward)      -> row0 = [0, 0, 1]
        body.Y (right)   = cam.X (right)        -> row1 = [1, 0, 0]
        body.Z (down)    = cam.Y (down)         -> row2 = [0, 1, 0]

    This is IDENTICAL to the "static dashcam fallback" already hard-coded
    in vio_to_enu_velocities (auto_calibrate=False path) -- confirms that
    fallback already assumes this exact canonical mount. We rebuild it
    here explicitly (rather than relying on the hidden default) so the
    perturbation grid below has a documented, inspectable base matrix.
    """
    return np.array([
        [0.0, 0.0, 1.0],   # body X (fwd)   = cam Z (fwd)
        [1.0, 0.0, 0.0],   # body Y (right) = cam X (right)
        [0.0, 1.0, 0.0],   # body Z (down)  = cam Y (down)
    ], dtype=np.float64)


def perturb(R_base: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Apply a small yaw/pitch/roll perturbation (body-sample Tait-Bryan,
    intrinsic Z-Y-X = yaw about body Z, pitch about body Y, roll about
    body X) on top of R_base: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ R_base.
    """
    dR = _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)
    return dR @ R_base


# ---------------------------------------------------------------------------
# Cheap per-variant pipeline: samples (fixed) -> vio_vels -> fused -> score.
# ---------------------------------------------------------------------------
def run_variant(
    *, name: str, samples, pos_rows, javad_csv: Path,
    R_body_from_cam: Optional[np.ndarray], auto_calibrate: bool,
    min_speed_mps: float, out_csv: Path,
) -> dict:
    from data_pipeline.vio import vio_to_enu_velocities, calibrate_R_body_from_cam

    residual_deg = float("nan")
    if R_body_from_cam is None and auto_calibrate:
        # Mirror vio_to_enu_velocities' internal auto-calibrate call but
        # with an explicit min_speed_mps so we can sweep it and read back
        # the residual (vio_to_enu_velocities itself only exposes the
        # default min_speed_mps=3.0 via its own internal call).
        try:
            R_body_from_cam, p50 = calibrate_R_body_from_cam(
                samples, pos_rows, min_speed_mps=min_speed_mps,
            )
            residual_deg = math.degrees(p50)
        except ValueError as exc:
            return {
                "name": name, "residual_deg": float("nan"),
                "fused_max_m": float("nan"), "error": str(exc),
            }
        vio_vels = vio_to_enu_velocities(
            samples, pos_rows, R_body_from_cam=R_body_from_cam,
            auto_calibrate=False,
        )
    else:
        vio_vels = vio_to_enu_velocities(
            samples, pos_rows, R_body_from_cam=R_body_from_cam,
            auto_calibrate=False,
        )
        # Still report the residual this fixed R would have scored, for
        # comparability with the Wahba variants (diagnostic only, not
        # used to pick R).
        try:
            _, p50 = calibrate_R_body_from_cam(
                samples, pos_rows, min_speed_mps=3.0,
            )
            # Residual of THIS R (not the Wahba-solved one): recompute
            # directly against the same high-speed pairs Wahba used.
            residual_deg = _residual_of_R_deg(samples, pos_rows, R_body_from_cam, min_speed_mps=3.0)
        except ValueError:
            residual_deg = _residual_of_R_deg(samples, pos_rows, R_body_from_cam, min_speed_mps=3.0)

    if len(vio_vels) < 3:
        return {
            "name": name, "residual_deg": residual_deg,
            "fused_max_m": float("nan"),
            "error": f"only {len(vio_vels)} VIO ENU velocity samples (<3)",
        }

    fused_rows = r1.fuse_vio_velocity_blend(pos_rows, vio_vels)
    r1.export_posrows_csv(fused_rows, out_csv)
    score = score_trajectories(javad_csv, out_csv)
    speed_corr = r1._speed_corr(pos_rows, vio_vels)
    return {
        "name": name, "residual_deg": residual_deg,
        "fused_max_m": score["max_m"], "two_sigma_m": score["two_sigma_m"],
        "median_offset_m": score["median_offset_m"], "n": score["n"],
        "n_vio_vels": len(vio_vels), "speed_corr": speed_corr,
        "R": R_body_from_cam,
    }


def _residual_of_R_deg(samples, pos_rows, R: np.ndarray, min_speed_mps: float) -> float:
    """p50 angular residual of a GIVEN (fixed) R against the same
    high-speed Post-processing-heading pairs calibrate_R_body_from_cam uses
    internally -- lets fixed-R variants report a residual comparable to
    the Wahba-solved ones without re-deriving R.
    """
    from bisect import bisect_left as _bisect
    pos_t = [r.utc_s for r in pos_rows]
    target = np.array([1.0, 0.0, 0.0])
    residuals = []
    for s in samples:
        if not math.isfinite(float(s.t_unit_cam[0])):
            continue
        i = _bisect(pos_t, s.utc_s)
        if i <= 0 or i >= len(pos_rows):
            continue
        a, b = pos_rows[i - 1], pos_rows[i]
        dt = b.utc_s - a.utc_s
        if dt <= 0 or dt > 1.5:
            continue
        al = (s.utc_s - a.utc_s) / dt
        ve = a.ve + al * (b.ve - a.ve)
        vn = a.vn + al * (b.vn - a.vn)
        if not (math.isfinite(ve) and math.isfinite(vn)):
            continue
        if math.hypot(ve, vn) < min_speed_mps:
            continue
        c = s.t_unit_cam / (np.linalg.norm(s.t_unit_cam) + 1e-9)
        rc = R @ c
        dot = float(np.clip(np.dot(rc, target), -1.0, 1.0))
        residuals.append(math.acos(dot))
    if not residuals:
        return float("nan")
    residuals.sort()
    return math.degrees(residuals[len(residuals) // 2])


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

    # ---- recording_map (synthesize if empty; identical to round 1) -----
    recording_map = r1.RECORDING_TXT
    if not r1.RECORDING_TXT.is_file() or r1.RECORDING_TXT.stat().st_size == 0:
        recording_map = OUT_DIR / "synthesized_recording_map.txt"
        if recording_map.is_file() and recording_map.stat().st_size > 0:
            log(t0, f"reusing existing synthesized recording_map: {recording_map}")
        else:
            log(t0, "synthesizing recording_map (empty recording_*.txt)")
            r1.synthesize_recording_map(recording_map, log=lambda m: log(t0, m))

    # ---- run_vio_multiframe_v2 ONCE, cache samples to disk + memory ----
    samples = None
    if SAMPLES_CACHE.is_file():
        try:
            with open(SAMPLES_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("frame_decim_hz") == FRAME_DECIM_HZ:
                samples = cached["samples"]
                log(t0, f"loaded cached VIO samples from {SAMPLES_CACHE} "
                        f"(n={len(samples)}, frame_decim_hz={FRAME_DECIM_HZ}) "
                        f"-- SKIPPING the ~6min run_vio_multiframe_v2 call")
        except Exception as exc:
            log(t0, f"cache load failed ({exc}); will re-run VIO")

    if samples is None:
        from data_pipeline.vio import run_vio_multiframe_v2
        imu_rows = parse_imu(r1.SENSORS_TXT)
        log(t0, f"sensors: {len(imu_rows)} rows")
        log(t0, f"running run_vio_multiframe_v2 (frame_decim_hz={FRAME_DECIM_HZ}) "
                f"-- ONE-TIME ~6 min run, then cached for all round-2 variants")
        try:
            samples = run_vio_multiframe_v2(
                video_path=r1.VIDEO_PATH,
                recording_map=recording_map,
                frame_decim_hz=FRAME_DECIM_HZ,
                imu_rows=imu_rows,
                log=lambda m: log(t0, m),
            )
        except Exception as exc:
            log(t0, f"BLOCKED: run_vio_multiframe_v2 raised: {type(exc).__name__}: {exc}")
            _write_blocked(raw_score, f"{type(exc).__name__}: {exc}")
            return 1
        if not samples:
            log(t0, "BLOCKED: run_vio_multiframe_v2 produced 0 samples")
            _write_blocked(raw_score, "0 VIO samples produced")
            return 1
        with open(SAMPLES_CACHE, "wb") as f:
            pickle.dump({"frame_decim_hz": FRAME_DECIM_HZ, "samples": samples}, f)
        log(t0, f"cached {len(samples)} VIO samples -> {SAMPLES_CACHE}")

    n_valid = sum(1 for s in samples if math.isfinite(float(s.t_unit_cam[0])))
    log(t0, f"VIO samples: {len(samples)} ({n_valid} with finite t_unit_cam)")

    results: list[dict] = []

    # ---- Variant set 1: min_speed_mps sweep (Wahba / auto_calibrate) ---
    for min_speed in (1.5, 2.5, 5.0, 8.0):
        name = f"wahba_min_speed_{min_speed:g}"
        log(t0, f"variant: {name}")
        res = run_variant(
            name=name, samples=samples, pos_rows=pos_rows, javad_csv=javad_csv,
            R_body_from_cam=None, auto_calibrate=True, min_speed_mps=min_speed,
            out_csv=OUT_DIR / f"fused_{name}.csv",
        )
        log(t0, f"  -> {res}")
        results.append(res)

    # ---- Variant set 2: fixed canonical R + yaw/pitch/roll grid --------
    R_canon = canonical_R_body_from_cam()
    grid_offsets = [-10.0, 0.0, 10.0]
    for yaw in grid_offsets:
        for pitch in grid_offsets:
            for roll in grid_offsets:
                name = f"fixedR_yaw{yaw:+.0f}_pitch{pitch:+.0f}_roll{roll:+.0f}"
                R = perturb(R_canon, yaw, pitch, roll) if (yaw or pitch or roll) else R_canon
                res = run_variant(
                    name=name, samples=samples, pos_rows=pos_rows,
                    javad_csv=javad_csv, R_body_from_cam=R, auto_calibrate=False,
                    min_speed_mps=3.0, out_csv=OUT_DIR / f"fused_{name}.csv",
                )
                log(t0, f"variant: {name} -> max_m={res.get('fused_max_m')} "
                        f"residual={res.get('residual_deg')}")
                results.append(res)

    # ---- Pick best (lowest finite fused MAX) ----------------------------
    finite = [r for r in results if math.isfinite(r.get("fused_max_m", float("nan")))]
    if not finite:
        log(t0, "BLOCKED: no variant produced a finite fused MAX")
        _write_blocked(raw_score, "no round-2 variant produced >=3 VIO ENU velocity samples")
        return 1
    best = min(finite, key=lambda r: r["fused_max_m"])
    log(t0, f"BEST variant: {best['name']} fused_max_m={best['fused_max_m']} "
            f"residual_deg={best['residual_deg']}")

    _write_findings(raw_score=raw_score, results=results, best=best,
                     n_vio_samples=len(samples), n_valid=n_valid)
    log(t0, "done")
    return 0


def _write_blocked(raw_score, error: str) -> None:
    lines = [
        "# VIO deep-dive -- day14 dodge190336 vs Javad",
        "",
        "## ROUND 2 -- BLOCKED",
        "",
        f"- RAW PPK vs Javad MAX = **{raw_score['max_m']:.3f} m**",
        f"- ROUND 2: BLOCKED -- {error}",
        "",
        "See `scripts/vio_deepdive_r2.py` for the exact repro.",
        "",
        "(Round-1 section preserved below if present; re-run "
        "`scripts/vio_deepdive.py` first if this file was overwritten "
        "before round 1 ran.)",
    ]
    _append_or_write(lines)


def _append_or_write(new_lines: list[str]) -> None:
    """Append a ROUND 2 section to the existing findings doc (keeps round
    1's content), or create the file if it doesn't exist yet."""
    existing = ""
    if FINDINGS_MD.is_file():
        existing = FINDINGS_MD.read_text(encoding="utf-8")
        # Strip a prior ROUND 2 section if this script already ran once
        # (idempotent re-run), keeping everything up to "## ROUND 2".
        marker = "\n## ROUND 2"
        idx = existing.find(marker)
        if idx != -1:
            existing = existing[:idx].rstrip() + "\n"
    FINDINGS_MD.write_text(
        existing.rstrip("\n") + "\n\n" + "\n".join(new_lines) + "\n",
        encoding="utf-8",
    )


def _write_findings(*, raw_score, results, best, n_vio_samples, n_valid) -> None:
    goal_met = best["fused_max_m"] <= 3.0
    round1_vio_max = 32.0  # from round-1 findings doc
    round1_raw_max = 11.39

    def _fmt_row(r: dict) -> str:
        if "error" in r:
            return (f"| `{r['name']}` | "
                    f"{r['residual_deg']:.2f} | ERROR: {r['error']} |")
        return (f"| `{r['name']}` | {r['residual_deg']:.2f} | "
                f"{r['fused_max_m']:.3f} |")

    wahba_rows = [r for r in results if r["name"].startswith("wahba_")]
    fixedr_rows = [r for r in results if r["name"].startswith("fixedR_")]

    lines = [
        "## ROUND 2 -- camera->body rotation calibration sweep",
        "",
        "**Date:** 2026-07-02 · **Session:** day14 `dodge/20260628_190336_677` "
        "· **Tool:** `scripts/vio_deepdive_r2.py` (reuses round 1's "
        "recording_map synthesis + velocity-blend fusion via import, no "
        "duplication) · `data_pipeline/vio.py` was NOT modified, only "
        "called.",
        "",
        "### Why round 2",
        "",
        f"Round 1's `auto_calibrate=True` (Wahba/Kabsch solve, default "
        f"`min_speed_mps=3.0`) produced a **46 deg** p50 residual -- far "
        f"above the 20 deg \"mount likely slipped\" warning threshold in "
        f"`calibrate_R_body_from_cam`'s own docstring -- and the resulting "
        f"fused trajectory was WORSE than raw PPK "
        f"(MAX {round1_raw_max:.2f} m raw -> {round1_vio_max:.2f} m fused). "
        f"VIO speed MAGNITUDE tracked PPK well (corr 0.899), so the VIO "
        f"pipeline itself (feature tracking, essential-matrix solve, "
        f"PPK-Doppler scaling) is healthy -- the problem is isolated to "
        f"the camera->body ROTATION used to project VIO's direction into "
        f"the vehicle frame.",
        "",
        "### run_vio_multiframe_v2 run once, cached",
        "",
        f"`run_vio_multiframe_v2` was invoked exactly once "
        f"(`frame_decim_hz={FRAME_DECIM_HZ}`) and its `VioSample` list "
        f"(n={n_vio_samples}, {n_valid} with a finite translation "
        f"direction) cached to `_vio_deepdive_out/vio_samples_cache.pkl`. "
        f"Every variant below reuses those cached samples -- only "
        f"`vio_to_enu_velocities` / `calibrate_R_body_from_cam` / the "
        f"round-1 velocity-blend fusion / scorer are re-run per variant.",
        "",
        "### Variant 1 -- min_speed_mps sweep (Wahba solve)",
        "",
        "`calibrate_R_body_from_cam(samples, pos_rows, min_speed_mps=X)` "
        "-> `R` -> `vio_to_enu_velocities(..., R_body_from_cam=R, "
        "auto_calibrate=False)` -> round-1 velocity-blend fusion -> score.",
        "",
        "| variant | residual (deg) | fused MAX vs Javad (m) |",
        "|---|---:|---:|",
    ]
    for r in wahba_rows:
        lines.append(_fmt_row(r))
    lines += [
        "",
        "### Variant 2 -- fixed canonical mount R + yaw/pitch/roll grid",
        "",
        "Canonical R_body_from_cam for a forward-facing landscape device "
        "(camera +X=right/+Y=down/+Z=forward; vehicle body "
        "+X=forward/+Y=right/+Z=down, matching the heading-rotation "
        "convention `vio_to_enu_velocities` uses internally): "
        "`body.X(fwd)=cam.Z`, `body.Y(right)=cam.X`, `body.Z(down)=cam.Y` "
        "-- i.e. the same matrix as `vio_to_enu_velocities`'s own "
        "`auto_calibrate=False` \"static dashcam fallback\", rebuilt "
        "explicitly here. A +-10 deg yaw/pitch/roll grid (27 combinations "
        "incl. 0/0/0) was applied on top via "
        "`Rz(yaw)@Ry(pitch)@Rx(roll)@R_canonical`, passed to "
        "`vio_to_enu_velocities(R_body_from_cam=R, auto_calibrate=False)`. "
        "This bypasses the Wahba solve entirely; the residual column here "
        "is diagnostic only (computed post-hoc against the same >=3 m/s "
        "PPK-heading pairs Wahba uses, NOT used to pick R).",
        "",
        "| variant | residual vs 3 m/s pairs (deg) | fused MAX vs Javad (m) |",
        "|---|---:|---:|",
    ]
    for r in fixedr_rows:
        lines.append(_fmt_row(r))

    lines += [
        "",
        "### Best variant",
        "",
        f"**`{best['name']}`** -- residual {best['residual_deg']:.2f} deg, "
        f"fused MAX vs Javad = **{best['fused_max_m']:.3f} m** "
        f"(2sigma={best.get('two_sigma_m', float('nan')):.3f} m, "
        f"n={best.get('n', 'n/a')}, n_vio_vels={best.get('n_vio_vels', 'n/a')}, "
        f"speed_corr={best.get('speed_corr', float('nan')):.3f}).",
        "",
        f"vs round-1 raw PPK MAX ({round1_raw_max:.2f} m): "
        f"{'BEAT raw PPK' if best['fused_max_m'] < round1_raw_max else 'still WORSE than raw PPK'} "
        f"(delta {round1_raw_max - best['fused_max_m']:+.3f} m). "
        f"vs round-1 VIO-fused MAX ({round1_vio_max:.2f} m): "
        f"improved by {round1_vio_max - best['fused_max_m']:.3f} m.",
        "",
        "### Gate: MAX <= 3.0 m",
        "",
        f"**{'PASS' if goal_met else 'NOT MET'}** -- best fused MAX = "
        f"{best['fused_max_m']:.3f} m {'<=' if goal_met else '>'} 3.0 m.",
        "",
    ]
    if goal_met:
        lines += [
            "Round 2 closes the loop for this session: a fixed, "
            "documented camera-mount rotation (bypassing the unreliable "
            "Wahba solve, which is sensitive to the min_speed_mps gate "
            "and can converge on a rotated/ambiguous solution when the "
            "high-speed sample set is small or the mount assumption is "
            "wrong) gets fused MAX under the 3.0 m target.",
        ]
    else:
        lines += [
            "### Round 3 lever",
            "",
            "Neither the Wahba min_speed sweep nor the fixed-canonical-R "
            "+-10 deg grid reached the 3.0 m gate. Candidates for round 3, "
            "roughly in order of expected leverage:",
            "",
            "1. **Widen/refine the fixed-R grid.** +-10 deg may not bracket "
            "the true mount angle if the device was held at an unusual "
            "angle (e.g. dashboard-mounted with a non-trivial pitch, or "
            "yaw offset from the vehicle's true forward axis on a "
            "car-mount clip). Re-run variant 2 with a coarser, wider "
            "sweep (e.g. +-45 deg in 15 deg steps) to bracket the true "
            "mount angle before fine-tuning.",
            "2. **Per-axis diagnosis.** Decompose the residual by fixing "
            "two axes at 0 and sweeping the third across a full 360 deg "
            "to find which single rotation axis is misaligned -- much "
            "cheaper than a full 3D grid and tells us whether this is a "
            "yaw (heading offset), pitch (tilt), or roll (mount rotated "
            "in-plane) problem.",
            "3. **Stop scaling by raw per-sample PPK-Doppler speed "
            "and instead validate the fusion math itself** (round-1 "
            "already flagged this as a separate lever): confirm the "
            "velocity-blend's trapezoidal integration and epoch "
            "re-anchoring aren't amplifying a still-imperfect direction "
            "error into large positional excursions between epochs -- "
            "even a well-calibrated R will produce large per-epoch MAX "
            "swings if a handful of VIO samples have a bad direction "
            "(e.g. near-degenerate low-parallax frame pairs) and nothing "
            "downweights them in the current unweighted "
            "trapezoidal blend.",
            "4. **Wire VIO into `epoch_weight_v2`** as originally flagged "
            "in round 1, rather than the ad-hoc velocity-blend -- lets a "
            "well-calibrated VIO velocity down-weight/override individual "
            "bad PPK epochs instead of only filling in the path shape "
            "between them.",
        ]
    lines += [
        "",
        "### Artifacts",
        "",
        "- Script: `scripts/vio_deepdive_r2.py` (imports "
        "`scripts/vio_deepdive.py` for recording-map synthesis + "
        "velocity-blend fusion; does not duplicate it).",
        "- Cached VIO samples: `_vio_deepdive_out/vio_samples_cache.pkl` "
        "(re-used across all round-2 variants and any future round-3 run "
        "at the same `frame_decim_hz`).",
        f"- Per-variant fused CSVs: `_vio_deepdive_out/fused_<variant>.csv` "
        f"({len(results)} variants).",
    ]
    _append_or_write(lines)


if __name__ == "__main__":
    raise SystemExit(main())
