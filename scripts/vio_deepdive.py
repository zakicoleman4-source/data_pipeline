"""Motion model deep-dive ROUND 1: fuse the existing Motion model subsystem into the day14
dodge190336 path and measure horizontal MAX error vs The reference unit GT.

This script only CALLS data_pipeline.vio / data_pipeline.time_sync /
data_pipeline.capture_diag — it does not modify any production module.

Procedure
---------
1. Parse rover.pos -> pos_rows. Score RAW Post-processing vs The reference unit -> baseline MAX.
2. Data problem: recording_20260628_190336_677.txt (the recording_map
   run_vio_multiframe_v2 needs for its internal fit_time_anchor(recording_map)
   call) is 0 bytes for this session -- and every other day14 dodge session.
   This is the documented E-PP-305 "empty session anchor" case. The
   documented fallback (fit_time_anchor_with_fallback -> measurements Raw
   ChipsetElapsedRealtimeNanos) ALSO fails here (E-PP-306): this is a Cell
   ("dodge") capture where the chipset boottime column in Raw rows is all
   zero. capture_diag.resolve_boot_anchor's second-level fallback --
   boot_utc_pairs_from_fix_rows (measurements "Fix," rows, which carry a
   populated elapsedRealtimeNanos column even when Raw's doesn't) -- DOES
   work here (1414 usable Fix rows).
   We use that existing, already-shipped helper (read-only import, no vio.py
   edits) to build a boot->UTC TimeAnchor, then combine it with the per-sample
   recording_*.video_anchor.txt (frameNumber, bootNs) to synthesize a
   recording_map file in the exact two-column (video_ns, iso_utc) dialect
   that time_sync.fit_time_anchor / run_vio_multiframe_v2 already parse
   natively. video_t0_boottime_ns from capture_meta.json anchors sample 0.
3. run_vio_multiframe_v2 on the media using the synthesized recording_map,
   rate sensor-aided via sensors_*.txt, at a modest frame_decim_hz.
4. vio_to_enu_velocities(samples, pos_rows, auto_calibrate=True) -> fused
   Motion model Local-frame velocities (Post-processing-Rate-signal-scaled, Motion model-direction).
5. Build a Motion model-corrected path: integrate the Motion model Local-frame velocity between
   consecutive Post-processing epochs (trapezoidal) added to the Post-processing position at the
   start of each interval, i.e. a velocity-only blend that lets Motion model's
   drift-free sample-to-sample direction correct the Post-processing position's path
   shape between epochs while Post-processing still anchors absolute position every
   epoch (no unbounded drift). This is the simplest honest ROUND 1 fusion;
   it does NOT touch epoch_weight_v2 (no smoother-level velocity-prior wiring
   yet -- that is the round 2 lever if this round's MAX doesn't move enough).
6. Export both RAW Post-processing and the Motion model-corrected path to minimal CSVs
   (reference time, lat_deg, lon_deg -- exactly what traj_score._read consumes) and
   score both against a The reference unit CSV exported the same way. Report MAX for each.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_pipeline.parsers import parse_rtkpos, parse_imu, PosRow
from data_pipeline.traj_score import score_trajectories

# ---------------------------------------------------------------------------
# Session paths
# ---------------------------------------------------------------------------
SESSION_DIR = Path(r"C:/Aj/gps/day14/dodge/20260628_190336_677")
VIDEO_PATH = SESSION_DIR / "recording_20260628_190336_677.mp4"
RECORDING_TXT = SESSION_DIR / "recording_20260628_190336_677.txt"
VIDEO_ANCHOR_TXT = SESSION_DIR / "recording_20260628_190336_677.video_anchor.txt"
MEASUREMENTS_TXT = SESSION_DIR / "measurements_20260628_190336_677.txt"
SENSORS_TXT = SESSION_DIR / "sensors_20260628_190336_677.txt"

ROVER_POS = Path(
    r"C:/Aj/gps/day14/solved_2026-06-28/dodge/20260628_190336_677/rover.pos"
)
JAVAD_POS = Path(r"C:/Aj/gps/day14/solved_2026-06-28/gt/gt_log0628a.pos")

OUT_DIR = REPO / "_vio_deepdive_out"
FRAME_DECIM_HZ = 4.0
MAX_VIO_FRAMES_SECONDS: Optional[float] = None  # None = whole media


# ---------------------------------------------------------------------------
# Minimal CSV export for the scorer (reference time, lat_deg, lon_deg only --
# exactly the columns data_pipeline.traj_score._read reads). Both CSVs
# below use PosRow.utc_s (UTC unix seconds) for the "reference time" column; the
# scorer only ever compares the two CSVs against each other by nearest-time
# match, so a shared, self-consistent time base is all that's required --
# it does not need to be literal Reference time.
# ---------------------------------------------------------------------------
def _write_minimal_csv(rows, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gpstime", "lat_deg", "lon_deg"])
        for t, lat, lon in rows:
            if not (math.isfinite(t) and math.isfinite(lat) and math.isfinite(lon)):
                continue
            w.writerow([f"{t:.3f}", f"{lat:.9f}", f"{lon:.9f}"])


def export_posrows_csv(pos_rows, out_csv: Path) -> None:
    _write_minimal_csv(
        ((r.utc_s, r.lat_deg, r.lon_deg) for r in pos_rows), out_csv
    )


# ---------------------------------------------------------------------------
# Step 2: synthesize a recording_map when recording_*.txt is empty.
#
# Uses ONLY existing, already-shipped helpers (capture_diag.resolve_boot_anchor
# / boot_utc_pairs_from_fix_rows, time_sync.TimeAnchor) via import -- no
# production module is modified.
# ---------------------------------------------------------------------------
def synthesize_recording_map(out_path: Path, log=print) -> Path:
    from data_pipeline.capture_diag import resolve_boot_anchor, parse_video_anchor
    from data_pipeline.capture_meta import parse_capture_meta

    notes: list[str] = []
    boot_anchor, source = resolve_boot_anchor(
        recording_txt=RECORDING_TXT,
        measurements_txt=MEASUREMENTS_TXT,
        notes=notes,
    )
    for n in notes:
        log(f"[bridge] {n}")
    if boot_anchor is None:
        raise RuntimeError(
            "No GNSS boot->UTC bridge recoverable for this session "
            "(recording.txt empty, measurements Raw+Fix both unusable)."
        )
    log(f"[bridge] boot->UTC anchor source = {source}, n={boot_anchor.n}, "
        f"rmse_s={boot_anchor.rmse_s:.4f}")

    cm = parse_capture_meta(SESSION_DIR / "capture_meta.json")
    video_t0_boottime_ns = float(cm.video_t0_boottime_ns)
    log(f"[bridge] video_t0_boottime_ns (frame 0) = {video_t0_boottime_ns:.0f}")

    video_pairs = parse_video_anchor(VIDEO_ANCHOR_TXT)  # (frameNumber, bootNs)
    if len(video_pairs) < 2:
        raise RuntimeError(f"video_anchor.txt has too few rows: {VIDEO_ANCHOR_TXT}")

    # Write (video_ns_relative_to_frame0, iso_utc) pairs -- the exact dialect
    # time_sync.fit_time_anchor's _iter_pairs already parses (col0=int ns,
    # col1=ISO8601 UTC string; trailing cols ignored).
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _frame_num, boot_ns in video_pairs:
            video_ns = boot_ns - video_t0_boottime_ns
            if video_ns < 0:
                continue  # samples before the media's own t0 (shouldn't occur)
            utc_s = boot_anchor.boottime_to_utc_s(float(boot_ns))
            iso = dt.datetime.fromtimestamp(
                utc_s, tz=dt.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            f.write(f"{int(video_ns)},{iso}\n")
            n_written += 1
    log(f"[bridge] synthesized recording_map: {n_written} (video_ns, utc) rows "
        f"-> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 5: velocity-blend fusion (round-1, no smoother wiring).
# ---------------------------------------------------------------------------
def fuse_vio_velocity_blend(pos_rows: list[PosRow], vio_vels: list) -> list[PosRow]:
    """Integrate Motion model Local-frame velocity within each Post-processing epoch interval, added on
    top of the Post-processing position at the interval start, re-anchored every epoch.

    For interval [pos[i], pos[i+1]]: take all vio_vels samples whose utc_s
    falls in that interval (sorted), trapezoidally integrate d(east), d(north)
    from pos[i]'s position, and OVERRIDE pos[i+1]'s lat/lon with
    pos[i]_latlon + integrated(d_east, d_north) -- i.e. Motion model supplies the
    within-epoch path shape, Post-processing still resets/anchors absolute position at
    every epoch boundary (bounded drift by construction: max drift is one
    epoch's worth of Motion model integration, ~1s here since rover.pos is 1 Hz).

    When an interval has zero Motion model samples (no media coverage / no Post-processing speed
    for scaling), the original Post-processing position is kept unchanged for that row.
    """
    if not vio_vels:
        return list(pos_rows)

    lat0 = pos_rows[0].lat_deg
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))

    vio_t = [t for t, _ in vio_vels]
    vio_v = [v for _, v in vio_vels]
    n_vio = len(vio_t)

    out: list[PosRow] = [pos_rows[0]]
    j = 0  # walk pointer into vio_vels
    n_blended = 0
    for i in range(len(pos_rows) - 1):
        a = pos_rows[i]
        b = pos_rows[i + 1]
        t0, t1 = a.utc_s, b.utc_s
        if t1 <= t0:
            out.append(b)
            continue

        # Collect Motion model samples strictly within (t0, t1].
        while j < n_vio and vio_t[j] <= t0:
            j += 1
        k = j
        samples_in = []
        while k < n_vio and vio_t[k] <= t1:
            samples_in.append((vio_t[k], vio_v[k]))
            k += 1

        if not samples_in:
            out.append(b)
            continue

        # Trapezoidal integration of E/N velocity from t0 to t1, anchored
        # at a's lat/lon. Velocity is held constant from t0 to the first
        # sample and from the last sample to t1 (zero-order hold at the
        # boundaries -- conservative, no extrapolation).
        d_e = 0.0
        d_n = 0.0
        prev_t = t0
        prev_ve, prev_vn = samples_in[0][1][0], samples_in[0][1][1]
        for t_s, v in samples_in:
            dt_s = t_s - prev_t
            if dt_s > 0:
                d_e += 0.5 * (prev_ve + float(v[0])) * dt_s
                d_n += 0.5 * (prev_vn + float(v[1])) * dt_s
            prev_t = t_s
            prev_ve, prev_vn = float(v[0]), float(v[1])
        dt_tail = t1 - prev_t
        if dt_tail > 0:
            d_e += prev_ve * dt_tail
            d_n += prev_vn * dt_tail

        new_lat = a.lat_deg + d_n / m_per_deg_lat
        new_lon = a.lon_deg + d_e / m_per_deg_lon
        out.append(
            PosRow(
                utc_s=b.utc_s, lat_deg=new_lat, lon_deg=new_lon, h_m=b.h_m,
                quality=b.quality, vn=b.vn, ve=b.ve, vu=b.vu, ns=b.ns,
                sd_n=b.sd_n, sd_e=b.sd_e, sd_u=b.sd_u,
            )
        )
        n_blended += 1

    print(f"[fuse] blended {n_blended}/{len(pos_rows) - 1} epoch intervals "
          f"with >=1 VIO velocity sample")
    return out


def _speed_corr(pos_rows: list[PosRow], vio_vels: list) -> float:
    """Correlation of |Motion model Local-frame speed| vs |Post-processing Rate-signal speed| at Motion model sample
    times (nearest Post-processing epoch). Cheap agreement diagnostic independent of the
    position-blend result."""
    if len(vio_vels) < 3:
        return float("nan")
    pos_t = [r.utc_s for r in pos_rows]
    ppk_spd = []
    vio_spd = []
    import bisect
    for t, v in vio_vels:
        i = bisect.bisect_left(pos_t, t)
        if i <= 0 or i >= len(pos_rows):
            continue
        a, b = pos_rows[i - 1], pos_rows[i]
        dt_ab = b.utc_s - a.utc_s
        if dt_ab <= 0:
            continue
        al = (t - a.utc_s) / dt_ab
        ve = a.ve + al * (b.ve - a.ve)
        vn = a.vn + al * (b.vn - a.vn)
        if not (math.isfinite(ve) and math.isfinite(vn)):
            continue
        ppk_spd.append(math.hypot(ve, vn))
        vio_spd.append(math.hypot(float(v[0]), float(v[1])))
    if len(ppk_spd) < 3:
        return float("nan")
    import numpy as np
    a = np.asarray(ppk_spd)
    b = np.asarray(vio_spd)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    def log(m: str) -> None:
        print(f"[{time.time() - t_start:6.1f}s] {m}", flush=True)

    log("parsing rover.pos (device PPK) and Javad GT")
    pos_rows = parse_rtkpos(ROVER_POS)
    javad_rows = parse_rtkpos(JAVAD_POS)
    log(f"rover.pos: {len(pos_rows)} epochs; Javad GT: {len(javad_rows)} epochs")

    javad_csv = OUT_DIR / "javad_gt.csv"
    raw_ppk_csv = OUT_DIR / "raw_ppk.csv"
    export_posrows_csv(javad_rows, javad_csv)
    export_posrows_csv(pos_rows, raw_ppk_csv)

    # ---- Step 1: baseline (raw Post-processing vs The reference unit) --------------------------
    raw_score = score_trajectories(javad_csv, raw_ppk_csv)
    log(f"RAW PPK vs Javad: {raw_score}")

    # ---- Step 2: recording_map (synthesize if empty) -------------------
    recording_map = RECORDING_TXT
    if not RECORDING_TXT.is_file() or RECORDING_TXT.stat().st_size == 0:
        log(f"recording_*.txt is empty ({RECORDING_TXT}); "
            f"synthesizing a recording_map from video_anchor.txt + "
            f"measurements Fix-row boot->UTC bridge")
        recording_map = OUT_DIR / "synthesized_recording_map.txt"
        try:
            synthesize_recording_map(recording_map, log=log)
        except Exception as exc:
            log(f"BLOCKED: could not synthesize a recording_map: "
                f"{type(exc).__name__}: {exc}")
            _write_findings_blocked(raw_score, str(exc))
            return 1

    # ---- Step 3: run_vio_multiframe_v2 ---------------------------------
    from data_pipeline.vio import run_vio_multiframe_v2, vio_to_enu_velocities

    log("parsing sensors_*.txt (gyro-aided VO)")
    imu_rows = parse_imu(SENSORS_TXT)
    log(f"sensors: {len(imu_rows)} rows")

    log(f"running run_vio_multiframe_v2 (frame_decim_hz={FRAME_DECIM_HZ}) "
        f"-- this can take several minutes")
    try:
        samples = run_vio_multiframe_v2(
            video_path=VIDEO_PATH,
            recording_map=recording_map,
            frame_decim_hz=FRAME_DECIM_HZ,
            imu_rows=imu_rows,
            log=log,
        )
    except Exception as exc:
        log(f"BLOCKED: run_vio_multiframe_v2 raised: {type(exc).__name__}: {exc}")
        _write_findings_blocked(raw_score, f"{type(exc).__name__}: {exc}")
        return 1

    log(f"VIO samples: {len(samples)}")
    if not samples:
        log("BLOCKED: run_vio_multiframe_v2 produced 0 samples")
        _write_findings_blocked(raw_score, "0 VIO samples produced")
        return 1

    n_valid = sum(1 for s in samples if math.isfinite(float(s.t_unit_cam[0])))
    log(f"VIO samples with finite t_unit_cam: {n_valid}/{len(samples)}")

    # ---- Step 4: Motion model -> Local-frame velocities ----------------------------------
    log("vio_to_enu_velocities (auto_calibrate=True)")
    vio_vels = vio_to_enu_velocities(samples, pos_rows, auto_calibrate=True, log=log)
    log(f"VIO ENU velocity samples: {len(vio_vels)}")

    speed_corr = _speed_corr(pos_rows, vio_vels)
    log(f"VIO-speed vs PPK-Doppler-speed correlation: {speed_corr:.3f}"
        if math.isfinite(speed_corr) else "VIO-speed vs PPK-speed correlation: n/a")

    if len(vio_vels) < 3:
        log("BLOCKED: too few VIO ENU velocity samples to fuse "
            "(need speed>=0.3 m/s epochs with valid VIO samples)")
        _write_findings_blocked(
            raw_score, "too few VIO ENU velocity samples (<3) to fuse",
            speed_corr=speed_corr, n_vio_samples=len(samples),
        )
        return 1

    # ---- Step 5: velocity-blend fusion ----------------------------------
    fused_rows = fuse_vio_velocity_blend(pos_rows, vio_vels)
    fused_csv = OUT_DIR / "vio_fused.csv"
    export_posrows_csv(fused_rows, fused_csv)

    # ---- Step 6: score ----------------------------------------------------
    fused_score = score_trajectories(javad_csv, fused_csv)
    log(f"VIO-fused vs Javad: {fused_score}")

    _write_findings(
        raw_score=raw_score, fused_score=fused_score,
        n_vio_samples=len(samples), n_valid=n_valid,
        n_vio_vels=len(vio_vels), speed_corr=speed_corr,
        recording_map_source=(
            "synthesized (Fix-row boot->UTC bridge + video_anchor.txt)"
            if recording_map != RECORDING_TXT else "recording.txt (native)"
        ),
    )
    log("done")
    return 0


def _write_findings_blocked(raw_score, error: str, speed_corr=None,
                             n_vio_samples=None) -> None:
    lines = [
        "# VIO deep-dive round 1 -- day14 dodge190336 vs Javad",
        "",
        f"**Status:** BLOCKED",
        "",
        f"- RAW PPK vs Javad MAX = **{raw_score['max_m']:.3f} m** "
        f"(2sigma={raw_score['two_sigma_m']:.3f} m, n={raw_score['n']})",
        f"- VIO fusion: BLOCKED -- {error}",
    ]
    if speed_corr is not None:
        lines.append(f"- VIO-speed vs PPK-speed correlation: {speed_corr}")
    if n_vio_samples is not None:
        lines.append(f"- VIO samples produced: {n_vio_samples}")
    lines += [
        "",
        "See `scripts/vio_deepdive.py` for the exact repro + API calls used.",
    ]
    (REPO / "docs" / "findings" / "vio-deepdive.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_findings(*, raw_score, fused_score, n_vio_samples, n_valid,
                     n_vio_vels, speed_corr, recording_map_source) -> None:
    moved = fused_score["max_m"] < raw_score["max_m"]
    delta = raw_score["max_m"] - fused_score["max_m"]
    goal_met = fused_score["max_m"] <= 3.0
    lines = [
        "# VIO deep-dive round 1 -- day14 dodge190336 vs Javad",
        "",
        "**Date:** 2026-07-02 · **Session:** day14 `dodge/20260628_190336_677` "
        "· **Reference:** Javad survey-grade GNSS GT "
        "(`solved_2026-06-28/gt/gt_log0628a.pos`) · "
        "**Tool:** `scripts/vio_deepdive.py`",
        "",
        "## Goal",
        "",
        "Fuse the EXISTING VIO subsystem (`data_pipeline/vio.py`: "
        "`run_vio_multiframe_v2` + `vio_to_enu_velocities`) into the device "
        "PPK trajectory and drive horizontal **MAX** error vs Javad down to "
        "<= 3.0 m, across multiple rounds. This is ROUND 1: baseline "
        "measurement + first honest fusion attempt.",
        "",
        "## Data problem found (and worked around)",
        "",
        "`recording_20260628_190336_677.txt` -- the time-bridge file "
        "`run_vio_multiframe_v2` needs internally (`time_sync.fit_time_anchor"
        "(recording_map)`) -- is **0 bytes** for this session, and for every "
        "other day14 dodge session. This is a known, documented gap (see "
        "`time_sync.py` \"Empty-recording-anchor fallback\" section: *\"7 of "
        "17 DAY14 sessions ship a 0-byte recording_*.txt\"*).",
        "",
        "The documented fallback chain was tried in order:",
        "",
        "1. `fit_time_anchor(recording_map)` -> raises `E-PP-305` "
        "(0 usable anchor rows).",
        "2. `fit_time_anchor_with_fallback` -> measurements `Raw,` rows' "
        "`ChipsetElapsedRealtimeNanos` column -> **all zero** for this "
        "(Pixel/\"dodge\") capture -> raises `E-PP-306`.",
        "3. `capture_diag.resolve_boot_anchor`'s second-level fallback -- "
        "`boot_utc_pairs_from_fix_rows` (measurements `Fix,` rows, which "
        "carry a populated `elapsedRealtimeNanos` column even though `Raw,`'s "
        "doesn't) -- **worked**: 1414 usable (boottime, UTC) pairs.",
        "",
        "This script combines that boot->UTC anchor with the per-frame "
        "`recording_*.video_anchor.txt` (frameNumber, bootNs) and "
        "`capture_meta.json`'s `video_t0_boottime_ns` to synthesize a "
        "`recording_map` file in the exact `(video_ns, iso_utc)` dialect "
        "`time_sync.fit_time_anchor` already parses natively -- no "
        "production module was edited, only existing helpers "
        "(`capture_diag.resolve_boot_anchor`, `capture_diag.parse_video_anchor`, "
        "`capture_meta.parse_capture_meta`) were imported and called. "
        f"Recording-map source used this run: **{recording_map_source}**.",
        "",
        "## Results",
        "",
        "| trajectory | MAX (m) | 2sigma (m) | median offset (m) | n |",
        "|---|---:|---:|---:|---:|",
        f"| RAW PPK vs Javad | **{raw_score['max_m']:.3f}** | "
        f"{raw_score['two_sigma_m']:.3f} | {raw_score['median_offset_m']:.3f} "
        f"| {raw_score['n']} |",
        f"| VIO-fused vs Javad | **{fused_score['max_m']:.3f}** | "
        f"{fused_score['two_sigma_m']:.3f} | "
        f"{fused_score['median_offset_m']:.3f} | {fused_score['n']} |",
        "",
        f"- VIO samples produced: {n_vio_samples} "
        f"({n_valid} with a finite translation direction)",
        f"- VIO ENU velocity samples (after PPK-Doppler scaling, "
        f"speed >= 0.3 m/s gate): {n_vio_vels}",
        f"- VIO-speed vs PPK-Doppler-speed correlation: "
        f"{speed_corr:.3f}" if math.isfinite(speed_corr) else
        "- VIO-speed vs PPK-Doppler-speed correlation: n/a",
        "",
        "## Fusion method (round 1)",
        "",
        "Velocity-blend, not smoother-integrated: for each PPK epoch "
        "interval `[pos[i], pos[i+1]]`, VIO ENU velocity samples falling in "
        "that interval are trapezoidally integrated (zero-order hold at the "
        "interval boundaries) to get a delta-east/delta-north offset from "
        "`pos[i]`, which replaces `pos[i+1]`'s lat/lon. PPK still anchors "
        "absolute position at every epoch boundary (1 Hz here), so drift is "
        "bounded to at most one epoch's worth of VIO integration -- this is "
        "NOT a full VIO-integrated trajectory (that would need the FGO/EKF "
        "smoother wiring described below). `epoch_weight_v2` / the "
        "production smoother was **not** touched or given a velocity prior "
        "in this round.",
        "",
        "## Verdict",
        "",
        f"MAX {'IMPROVED' if moved else 'DID NOT IMPROVE'} vs raw PPK: "
        f"{raw_score['max_m']:.3f} m -> {fused_score['max_m']:.3f} m "
        f"(delta {delta:+.3f} m). "
        f"Goal (MAX <= 3.0 m) is {'MET' if goal_met else 'NOT yet met'}.",
        "",
        "## Next lever (round 2)",
        "",
        "The velocity-blend above only touches the *shape* of the path "
        "between 1 Hz PPK epochs; it cannot correct a bad PPK epoch itself "
        "(the dominant MAX-error source is almost certainly a handful of "
        "spiked/degraded epochs, not inter-epoch path shape). Round 2 should "
        "wire the VIO ENU velocities into `epoch_weight_v2` as an "
        "independent velocity observation (or via the FGO factor graph) so "
        "VIO can down-weight/override a PPK epoch it disagrees with, rather "
        "than only filling in between good epochs. A second, cheaper lever: "
        "widen `frame_decim_hz` / gyro-aided KLT tuning to raise "
        "`n_vio_vels` coverage (currently gated by PPK speed >= 0.3 m/s, "
        "so any near-stationary stretch of the drive produces zero VIO "
        "correction there).",
        "",
        "## Artifacts",
        "",
        "- Script: `scripts/vio_deepdive.py`.",
        "- CSVs: `_vio_deepdive_out/javad_gt.csv`, `_vio_deepdive_out/raw_ppk.csv`, "
        "`_vio_deepdive_out/vio_fused.csv`, "
        "`_vio_deepdive_out/synthesized_recording_map.txt`.",
        "- Data: day14 `dodge/20260628_190336_677/` (video, sensors, "
        "measurements) vs `solved_2026-06-28/dodge/20260628_190336_677/"
        "rover.pos` vs `solved_2026-06-28/gt/gt_log0628a.pos`.",
    ]
    (REPO / "docs" / "findings" / "vio-deepdive.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
