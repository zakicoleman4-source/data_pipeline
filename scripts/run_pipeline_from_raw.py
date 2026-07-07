"""End-to-end pipeline from a capture folder + base.obs + nav files.

User provides:
  --raw       capture session folder (files auto-detected)
  --base-obs  reference input Interchange-format .obs
  --nav       one or more auxiliary-data/auxiliary data files (.nav / .26N / .26G / .26L / .26C ...)
  --out       output directory
  [--config]  The external solver config (default: data_pipeline/configs/fgo_seed.conf)
  [--fps]     sample extraction fps (default 4)
  [--smoother] {gaussian,ns_adaptive,epoch_weighted,fgo}  (default: epoch_weighted)
  [--base-cartesian XYZ] optional surveyed base Cartesian XYZ X,Y,Z (m,m,m) — patches config

Pipeline produces:
  out/rover.obs                        — Interchange-format conversion of measurements_*.txt
  out/rover.pos                        — The external solver Post-processing solution
  out/rover.pos.stat                   — The external solver per-source residual stats (for epoch weighting)
  out/samples/frame_<NNNNNN>.png        — extracted PNG samples (dot-free seq index)
  out/extracted_frame_times.csv        — sample index <-> PTS map
  out/Georef.csv          — final per-sample lat/lon[/alt][/yaw,pitch,roll]
  out/pos_metadata.csv                 — full per-epoch The external solver metadata + calibrated sigmas
  out/trajectory_user.csv + .export format       — client path (honest per-epoch 2-sigma + flags)
  out/sync_player.html                 — synced media + path + Motion sensor trust panel
  out/trajectory_viewer.html           — diagnostic viewer
  out/comparison_viewer.html           — diagnostic viewer
  out/accuracy_dashboard.html, trust_viewer.html, routes_viewer.html,
  out/trajectory_compare.html          — bonus engineer viewers
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _export_source_choices() -> list[str]:
    """'raw' + every registered smoother name (for --export-source)."""
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from data_pipeline.smoothers import list_smoothers
        return ["raw"] + list_smoothers()
    except Exception:
        return ["raw"]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    configs_dir = repo / "data_pipeline" / "configs"
    default_conf = configs_dir / "javad.conf"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, type=Path,
                   help="Capture session folder (files auto-detected)")
    p.add_argument("--base-obs", required=True, type=Path, help="Base station RINEX .obs")
    p.add_argument("--nav", required=True, type=Path, nargs="+",
                   help="One or more nav/ephemeris files (.nav .26N .26G .26L .26C .sp3 etc.)")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--config", type=Path, default=default_conf,
                   help=("RTKLIB config (.conf). Default: javad.conf (preferred). "
                         "Elevation variants: javad_el05/10/15/20/25.conf "
                         "(or use --elmask). Also: fgo_seed.conf, handsetbase.conf, "
                         "javad_avg_sp.conf."))
    p.add_argument("--elmask", type=int, choices=[5, 10, 15, 20, 25], default=None,
                   help=("Elevation-mask preset: selects configs/javad_el<NN>.conf "
                         "(overrides --config). 5/10/15/20/25 deg."))
    p.add_argument("--fps", type=float, default=4.0, help="Frame extraction fps (default %(default)s)")
    p.add_argument("--smoother",
                   choices=["gaussian", "ns_adaptive", "epoch_weighted",
                            "epoch_weighted_v2", "fgo", "fused_bent"],
                   default="epoch_weighted_v2",
                   help="Trajectory smoother (default: %(default)s — IMU-aware 6D Kalman + ZUPT + NHC, no-video champion)")
    # Three ways to override base coords (mutually exclusive):
    p.add_argument("--base-ecef", type=str, default=None,
                   help="Surveyed base ECEF X,Y,Z (comma-separated m). Patches config.")
    p.add_argument("--base-llh", type=str, default=None,
                   help="Surveyed base lat,lon,h (decimal deg + m). Patches config.")
    p.add_argument("--base-from-rinex", action="store_true",
                   help="Read APPROX POSITION XYZ from --base-obs header. Patches config.")
    p.add_argument("--add-altitude", action="store_true", help="Emit altitude column in Georef CSV")
    p.add_argument("--add-ypr", action="store_true", help="Emit yaw/pitch/roll columns")
    p.add_argument("--coord-systems", nargs="+", default=None,
                   choices=["geodetic", "ecef", "utm", "enu"],
                   help="Which coordinate blocks to write in trajectory_user.csv "
                        "(default: geodetic ecef). Choose any of: geodetic ecef utm enu.")
    p.add_argument("--time-basis", dest="time_basis", nargs="+", default=["gpst"],
                   choices=["gpst", "utc", "audio", "iso"],
                   help="TIME column(s) for trajectory_user.csv, emitted first in "
                        "the given order (default: gpst). gpst->gpstime (GPST s), "
                        "utc->utc_s (absolute UTC unix s), audio->t_audio_s "
                        "(seconds from audio sample 0, needs the session's audio "
                        "anchor), iso->utc_iso (ISO-8601 UTC string).")
    p.add_argument("--export-source", default=None,
                   choices=_export_source_choices(),
                   help="Which trajectory trajectory_user.csv/.kml serialize, "
                        "INDEPENDENT of --smoother (which still drives georef "
                        "and the other stages). 'raw' = raw PPK rows exactly; "
                        "any smoother name = run that smoother on the raw PPK "
                        "rows for the export only. Default: unset = current "
                        "behavior (the pipeline smoother's rows when an epoch "
                        "smoother ran, else raw PPK).")
    p.add_argument("--emit-final-velocity", action="store_true", default=False,
                   help="Add final_vn/ve/vu_mps + final_speed_mps (raw PPK "
                        "Doppler) + vel_disagree_mps + coords_dropped columns "
                        "to trajectory_user.csv.")
    p.add_argument("--vel-disagree-threshold", type=float, default=None,
                   help="Coord/Doppler velocity disagreement gate (m/s). When "
                        "set (implies --emit-final-velocity), rows where "
                        "|coord-derived vel - Doppler vel| exceeds this get "
                        "EMPTY final_v* AND EMPTY coordinate columns, with "
                        "coords_dropped=1 (row kept, omission visible).")
    p.add_argument("--no-smooth-z", dest="smooth_z", action="store_false", default=True,
                   help="Disable height/Z smoothing on export (default: Z is smoothed).")
    p.add_argument("--z-sigma-s", type=float, default=3.0,
                   help="Height-smoothing gaussian sigma in seconds (default %(default)s).")
    p.add_argument("--include-viewers", action="store_true", default=True, help="Build HTML viewers")
    p.add_argument("--rnx2rtkp-exe", type=Path, default=None, help="Override rnx2rtkp binary path")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))

    from data_pipeline.pipeline import RawInputs
    from data_pipeline.stages import frames, georef, ppk, rinex, viewers
    from data_pipeline.pos_metadata import to_metadata_csv
    from data_pipeline.parsers import parse_rtkpos

    # ---- validate inputs ----
    raw_dir = args.raw.resolve()
    if not raw_dir.is_dir():
        print(f"FAIL: --raw not a directory: {raw_dir}", file=sys.stderr); return 2
    base_obs = args.base_obs.resolve()
    if not base_obs.is_file():
        print(f"FAIL: --base-obs not a file: {base_obs}", file=sys.stderr); return 2
    nav_files = [n.resolve() for n in args.nav]
    for n in nav_files:
        if not n.is_file():
            print(f"FAIL: --nav file missing: {n}", file=sys.stderr); return 2
    # --elmask preset selects a the reference unit elevation variant, overriding --config.
    if args.elmask is not None:
        _cfgdir = Path(__file__).resolve().parent.parent / "data_pipeline" / "configs"
        args.config = _cfgdir / f"javad_el{args.elmask:02d}.conf"
    config_file = args.config.resolve()
    if not config_file.is_file():
        print(f"FAIL: --config missing: {config_file}", file=sys.stderr); return 2

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    log = print

    # ---- discover RAW inputs ----
    try:
        raw = RawInputs.from_folder(raw_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"FAIL: RAW folder inspection: {e}", file=sys.stderr); return 3
    log(f"[raw] measurements: {raw.measurements_txt}")
    log(f"[raw] recording_map: {raw.recording_txt}")
    log(f"[raw] video: {raw.recording_mp4}")
    log(f"[raw] sensors: {raw.sensors_txt}")

    base_ecef: Optional[tuple[float, float, float]] = None
    n_base_opts = sum(1 for x in (args.base_ecef, args.base_llh, args.base_from_rinex) if x)
    if n_base_opts > 1:
        print("FAIL: --base-ecef, --base-llh, --base-from-rinex are mutually exclusive",
              file=sys.stderr); return 2
    if args.base_ecef:
        try:
            parts = [float(x) for x in args.base_ecef.split(",")]
            if len(parts) != 3:
                raise ValueError("expected X,Y,Z")
            base_ecef = (parts[0], parts[1], parts[2])
            log(f"[ppk] surveyed base ECEF override: {base_ecef}")
        except ValueError as e:
            print(f"FAIL: --base-ecef parse: {e}", file=sys.stderr); return 2
    elif args.base_llh:
        from data_pipeline.base_pos import base_xyz_from_llh
        try:
            parts = [float(x) for x in args.base_llh.split(",")]
            if len(parts) != 3:
                raise ValueError("expected lat,lon,h")
            base_ecef = base_xyz_from_llh(parts[0], parts[1], parts[2])
            log(f"[ppk] surveyed base LLH override: lat={parts[0]} lon={parts[1]} h={parts[2]} -> ECEF {base_ecef}")
        except ValueError as e:
            print(f"FAIL: --base-llh parse: {e}", file=sys.stderr); return 2
    elif args.base_from_rinex:
        from data_pipeline.base_pos import read_rinex_approx_xyz
        xyz = read_rinex_approx_xyz(base_obs)
        if xyz is None:
            print(f"FAIL: --base-from-rinex: no usable APPROX POSITION XYZ in {base_obs}",
                  file=sys.stderr); return 2
        base_ecef = xyz
        log(f"[ppk] base ECEF read from RINEX header: {base_ecef}")

    # ---- Stage 1: RAW -> interchange-format rover.obs ----
    log("=" * 60); log("[1/4] RINEX")
    rover_obs = out / "rover.obs"
    try:
        rinex.run(
            measurements_txt=raw.measurements_txt,
            output_obs=rover_obs,
            android_rinex_src=None,
            log=log,
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"FAIL: RINEX stage: {type(e).__name__}: {e}\n"
              "Verify the measurements_*.txt is a the capture app raw GNSS file "
              "(must contain Raw,... lines) and that the vendored "
              "android_rinex is intact at vendor/android_rinex/src.",
              file=sys.stderr); return 4
    if not rover_obs.is_file() or rover_obs.stat().st_size < 100:
        print(f"FAIL: RINEX produced empty/missing .obs at {rover_obs}. "
              "the capture app session may have zero usable epochs (e.g. "
              "all-zero FullBiasNanos on SM-A205U-family devices).",
              file=sys.stderr); return 4
    log(f"[rinex] OK -> {rover_obs} ({rover_obs.stat().st_size} bytes)")

    # ---- Stage 2: Post-processing -> rover.pos + rover.pos.stat ----
    log("=" * 60); log("[2/4] PPK (RTKLIB rnx2rtkp)")
    pos_path = out / "rover.pos"
    try:
        ppk_res = ppk.run(
            rover_obs=rover_obs,
            base_obs=base_obs,
            nav_files=nav_files,
            config_file=config_file,
            output_pos=pos_path,
            rnx2rtkp_exe=args.rnx2rtkp_exe,
            base_ecef_xyz=base_ecef,
            log=log,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"FAIL: PPK stage: {e}", file=sys.stderr); return 5
    if not pos_path.is_file():
        print(f"FAIL: PPK produced no .pos: {pos_path}", file=sys.stderr); return 5
    # Validate .pos has solution rows, not just headers.
    pos_validation = parse_rtkpos(pos_path)
    if not pos_validation:
        print(f"FAIL: PPK produced a .pos with header only / Q=0 every "
              f"epoch (file: {pos_path}, size {pos_path.stat().st_size} B).\n"
              "Causes: config too strict (try a looser .conf — "
              "javad_avg_sp.conf is more permissive than fgo_seed); "
              "base + rover obs do not overlap in time; or wrong nav files.",
              file=sys.stderr); return 5
    log(f"[ppk] OK -> {pos_path} ({len(pos_validation)} solution rows)")
    if ppk_res.stat_path and ppk_res.stat_path.is_file():
        log(f"[ppk] stat -> {ppk_res.stat_path}")

    # ---- Stage 3: media -> samples + frame_times.csv ----
    log("=" * 60); log("[3/4] Frames")
    try:
        fr = frames.run(video=raw.recording_mp4, out_dir=out, fps=args.fps, fmt="png", log=log)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"FAIL: frame extraction: {type(e).__name__}: {e}\n"
              "Verify the video is a valid mp4 and ffmpeg is on PATH "
              "(or vendor/ffmpeg/bin/ffmpeg.exe exists).",
              file=sys.stderr); return 6
    if fr.frame_count < 10:
        print(f"FAIL: too few frames extracted ({fr.frame_count}) from "
              f"{raw.recording_mp4}. Verify the video is not corrupted "
              "and --fps is reasonable for its length.",
              file=sys.stderr); return 6
    log(f"[frames] OK {fr.frame_count} frames -> {fr.frame_times_csv}")

    # ---- Stage 4: Coordinate output CSV ----
    log("=" * 60); log(f"[4/4] Georef CSV (smoother={args.smoother})")
    smoothing_profile = "car"
    use_ns_adaptive = (args.smoother == "ns_adaptive")
    use_fgo = (args.smoother == "fgo")
    use_epoch_weighted = (args.smoother == "epoch_weighted")
    use_epoch_weighted_v2 = (args.smoother == "epoch_weighted_v2")
    if args.smoother == "fused_bent":
        smoothing_profile = "fused-bent"

    csv_opts = georef.CsvOptions(
        smoothing=smoothing_profile,
        include_altitude=args.add_altitude,
        add_ypr=args.add_ypr,
        use_ns_adaptive_smoothing=use_ns_adaptive,
        use_fgo_smoothing=use_fgo,
    )

    csv_path = out / "Georef.csv"
    # If a per-epoch smoother was requested, run it before sample interp.
    if use_epoch_weighted or use_epoch_weighted_v2:
        from data_pipeline.parsers import PosRow
        import math
        pos_rows = parse_rtkpos(pos_path)
        if not pos_rows:
            print(f"FAIL: PPK produced no rows in {pos_path}; cannot run "
                  "epoch-weighted smoother. Re-run PPK with a longer "
                  "observation window or check the .obs/.nav files.",
                  file=sys.stderr)
            return 5
        stat_p = ppk_res.stat_path if (ppk_res.stat_path and ppk_res.stat_path.is_file()) else None
        if use_epoch_weighted_v2:
            from data_pipeline.epoch_weight_v2 import (
                EpochWeightV2Options, smooth_epoch_weighted_v2,
            )
            from data_pipeline.parsers import parse_imu as _parse_imu_v2
            try:
                imu_rows = _parse_imu_v2(raw.sensors_txt)
            except (FileNotFoundError, OSError, ValueError) as _e:
                log(f"[csv] v2 IMU parse failed ({_e}); running without IMU.")
                imu_rows = None
            v2_opts = EpochWeightV2Options(
                stat_path=stat_p, zupt_enabled=True,
                nhc_enabled=True, nhc_heading_source="doppler",
                sigma_a_base=0.10,
            )
            v2_res = smooth_epoch_weighted_v2(
                pos_rows, imu_rows=imu_rows, options=v2_opts, log=log,
            )
            Es, Ns_, Us = v2_res.E_smooth, v2_res.N_smooth, v2_res.U_smooth
            log(f"[csv] v2: doppler_gated={v2_res.n_doppler_gated} "
                f"zupt={v2_res.n_zupt_updates} nhc={v2_res.n_nhc_updates}")
        else:
            from data_pipeline.epoch_weight import smooth_epoch_weighted
            Es, Ns_, Us = smooth_epoch_weighted(pos_rows, stat_path=stat_p)
        # Reconstruct LLH from Local-frame using same ref as inside the helper
        from data_pipeline.geo import _A, _E2, ecef_to_enu, llh_to_ecef
        ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
        rx, ry, rz = llh_to_ecef(*ref)
        rlat = math.radians(ref[0]); rlon = math.radians(ref[1])
        sl, cl = math.sin(rlat), math.cos(rlat); so, co = math.sin(rlon), math.cos(rlon)
        smoothed_rows: list[PosRow] = []
        for i, r in enumerate(pos_rows):
            e, n, u = float(Es[i]), float(Ns_[i]), float(Us[i])
            x = rx + (-so * e - sl * co * n + cl * co * u)
            y = ry + (co * e - sl * so * n + cl * so * u)
            z = rz + (cl * n + sl * u)
            p_xy = math.hypot(x, y); lon_r = math.atan2(y, x)
            lat_r = math.atan2(z, p_xy * (1 - _E2))
            for _ in range(6):
                sinl = math.sin(lat_r)
                Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
                h_iter = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
                lat_r = math.atan2(z, p_xy * (1 - _E2 * Nrad / (Nrad + h_iter)))
            sinl = math.sin(lat_r)
            Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
            h_m = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
            smoothed_rows.append(PosRow(
                utc_s=r.utc_s,
                lat_deg=math.degrees(lat_r),
                lon_deg=math.degrees(lon_r),
                h_m=h_m,
                quality=r.quality, ns=r.ns,
                vn=r.vn, ve=r.ve, vu=r.vu,
                sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
            ))
        # Write smoothed-pos to a new file
        smoothed_pos = out / "rover.smoothed.pos"
        _write_pos_minimal(smoothed_pos, smoothed_rows)
        log(f"[csv] using epoch-weighted smoothed pos -> {smoothed_pos}")
        pos_for_georef = smoothed_pos
        csv_opts = georef.CsvOptions(
            smoothing="none",
            include_altitude=args.add_altitude,
            add_ypr=args.add_ypr,
        )
    else:
        pos_for_georef = pos_path

    try:
        csv_res = georef.run(
            frame_times_csv=fr.frame_times_csv,
            recording_map=raw.recording_txt,
            pos_file=pos_for_georef,
            data_log=raw.measurements_txt,
            sensors_txt=raw.sensors_txt,
            out_csv=csv_path,
            fps=args.fps,
            options=csv_opts,
            capture_meta=raw.capture_meta_json,
            chop_video_anchor=(raw.chop_video_anchor if raw.is_chop else None),
            video_path=(raw.recording_mp4 if raw.is_chop else None),
            log=log,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"FAIL: Georef CSV stage: {type(e).__name__}: {e}",
              file=sys.stderr); return 7
    log(f"[csv] OK positioned {csv_res.n_with_position}/{csv_res.n_frames} -> {csv_path}")

    # ---- Stage 5: per-epoch metadata sidecar ----
    log("=" * 60); log("[bonus] pos metadata CSV")
    try:
        pos_rows = parse_rtkpos(pos_path)
        meta_csv = out / "pos_metadata.csv"
        to_metadata_csv(pos_rows, meta_csv)
        log(f"[meta] OK {len(pos_rows)} epochs -> {meta_csv}")
    except Exception as e:
        log(f"[meta] WARN failed to export pos metadata: {e}")

    # ---- Stage 5b: user-facing path export ----
    log("=" * 60); log("[bonus] user trajectory export")
    try:
        from data_pipeline.stages.user_export import (
            export_trajectory, export_kml, resolve_export_rows,
        )
        # --- Coordinate-source chooser: --export-source picks the exported
        # trajectory INDEPENDENTLY of --smoother (which still drives georef).
        if args.export_source is not None:
            imu_for_export = None
            try:
                from data_pipeline.parsers import parse_imu as _parse_imu_exp
                imu_for_export = _parse_imu_exp(raw.sensors_txt)
            except Exception as _e:
                log(f"[export] IMU parse for --export-source failed ({_e}); "
                    "running the export smoother without IMU.")
            export_rows = resolve_export_rows(
                parse_rtkpos(pos_path),
                source=args.export_source,
                imu_rows=imu_for_export,
                stat_path=(ppk_res.stat_path
                           if (ppk_res.stat_path and ppk_res.stat_path.is_file())
                           else None),
                log=log,
            )
            tag = ("raw_ppk" if args.export_source == "raw"
                   else args.export_source)
            log(f"[export] source={args.export_source} "
                f"({len(export_rows)} rows; independent of --smoother="
                f"{args.smoother})")
        # Otherwise: export the SMOOTHED rows when an epoch smoother ran;
        # else raw Post-processing (historical behavior).
        elif use_epoch_weighted or use_epoch_weighted_v2:
            export_rows = smoothed_rows  # built above
            tag = args.smoother
        else:
            export_rows = parse_rtkpos(pos_path)
            tag = f"raw_ppk_{args.smoother}"
        # --- TIME-basis chooser: resolve the stream-start UTC anchor when the
        # 'stream' basis is requested (t_audio_s = utc_s - audio_start_utc_s).
        time_bases = list(args.time_basis)
        audio_start_utc_s: Optional[float] = None
        if "audio" in time_bases:
            try:
                from data_pipeline.audio_frame_export import resolve_session_anchors
                anchors = resolve_session_anchors(
                    raw_dir, inputs=raw, need_utc=True, log=log,
                )
                if anchors.boot_anchor is None:
                    raise ValueError("no boot->UTC anchor for this session")
                audio_start_utc_s = float(
                    anchors.boot_anchor.boottime_to_utc_s(anchors.audio_start_boot_ns)
                )
                log(f"[export] audio time basis: audio_start_utc_s="
                    f"{audio_start_utc_s:.6f} "
                    f"(boot->UTC source: {anchors.boot_anchor_source or 'n/a'})")
            except Exception as _e:
                log(f"[export] WARN: 'audio' time basis requested but the "
                    f"session's audio anchor could not be resolved "
                    f"({type(_e).__name__}: {_e}); dropping 'audio' from "
                    f"--time-basis.")
                time_bases = [b for b in time_bases if b != "audio"]
                if not time_bases:
                    log("[export] WARN: no time basis left; falling back to gpst.")
                    time_bases = ["gpst"]
        user_csv = out / "trajectory_user.csv"
        ue = export_trajectory(
            export_rows, user_csv, source_tag=tag,
            coord_systems=args.coord_systems,
            smooth_z=args.smooth_z, z_sigma_s=args.z_sigma_s,
            time_bases=tuple(time_bases),
            audio_start_utc_s=audio_start_utc_s,
            emit_final_velocity=args.emit_final_velocity,
            vel_disagree_threshold_mps=args.vel_disagree_threshold,
        )
        log(f"[export] {ue.n_rows}/{ue.n_input_rows} rows "
            f"(dropped={ue.n_dropped_rows}, over_bar_flagged={ue.n_flagged_over_bar}, "
            f"vel_untrusted={ue.n_vel_untrusted}), "
            f"inflation={ue.inflation:.2f} -> {user_csv}")
        for _ln in ue.summary_text().splitlines():
            log(f"[export] {_ln}")
        user_kml = out / "trajectory_user.kml"
        export_kml(export_rows, user_kml, name=f"trajectory_{tag}",
                   smooth_z=args.smooth_z, z_sigma_s=args.z_sigma_s)
        log(f"[export] KML -> {user_kml}")
    except Exception as e:
        log(f"[export] WARN user export failed: {e}")

    # ---- Stage 5c: trust-painted smoothed path viewer ----
    if use_epoch_weighted or use_epoch_weighted_v2:
        log("=" * 60); log("[bonus] trust-painted trajectory viewer")
        try:
            raw_for_trust = parse_rtkpos(pos_path)
            viewers.build_smoothed_trust_viewer(
                raw_pos_rows=raw_for_trust,
                smoothed_pos_rows=smoothed_rows,
                out_html=out / "trust_viewer.html",
                log=log,
            )
        except Exception as e:
            log(f"[trust] WARN trust viewer failed: {e}")

    # ---- Stage 5d: multi-route comparison viewer (all smoothers) ----
    log("=" * 60); log("[bonus] running ALL smoothers + accuracy dashboard")
    raw_rows_for_routes = parse_rtkpos(pos_path)
    all_smoother_outputs: dict[str, list] = {}  # label -> list[PosRow]

    if use_epoch_weighted:
        all_smoother_outputs["epoch_weighted"] = smoothed_rows
    elif use_epoch_weighted_v2:
        all_smoother_outputs["epoch_weighted_v2"] = smoothed_rows

    # ---- Run gaussian / ns_adaptive / fgo / fused_bent for the dashboard ----
    def _enu_to_pos_rows(Es, Ns_, Us_, source_rows):
        """Convert Local-frame arrays back to PosRow list (preserve sigmas / Q / ns / vel)."""
        from data_pipeline.geo import _A, _E2, llh_to_ecef
        from data_pipeline.parsers import PosRow
        ref = (source_rows[0].lat_deg, source_rows[0].lon_deg, source_rows[0].h_m)
        rx, ry, rz = llh_to_ecef(*ref)
        rlat = math.radians(ref[0]); rlon = math.radians(ref[1])
        sl, cl = math.sin(rlat), math.cos(rlat); so, co = math.sin(rlon), math.cos(rlon)
        out_rows = []
        for i, r in enumerate(source_rows):
            e, n, u = float(Es[i]), float(Ns_[i]), float(Us_[i])
            x = rx + (-so*e - sl*co*n + cl*co*u)
            y = ry + (co*e - sl*so*n + cl*so*u)
            z = rz + (cl*n + sl*u)
            p_xy = math.hypot(x, y); lon_r = math.atan2(y, x)
            lat_r = math.atan2(z, p_xy*(1 - _E2))
            for _ in range(6):
                sinl = math.sin(lat_r)
                Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
                h_iter = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
                lat_r = math.atan2(z, p_xy*(1 - _E2*Nrad/(Nrad + h_iter)))
            sinl = math.sin(lat_r)
            Nrad = _A / math.sqrt(1 - _E2 * sinl * sinl)
            h_m = p_xy / max(1e-12, math.cos(lat_r)) - Nrad
            out_rows.append(PosRow(
                utc_s=r.utc_s, lat_deg=math.degrees(lat_r), lon_deg=math.degrees(lon_r),
                h_m=h_m, quality=r.quality, ns=r.ns,
                vn=r.vn, ve=r.ve, vu=r.vu,
                sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                sd_vn=r.sd_vn, sd_ve=r.sd_ve, sd_vu=r.sd_vu,
                ratio=r.ratio, age_s=r.age_s,
            ))
        return out_rows

    # gaussian xy=2s z=10s
    try:
        from data_pipeline.smoothing import gaussian_smooth
        from data_pipeline.geo import ecef_to_enu, llh_to_ecef
        ref_g = (raw_rows_for_routes[0].lat_deg, raw_rows_for_routes[0].lon_deg, raw_rows_for_routes[0].h_m)
        def _enu_g(r):
            x,y,z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m); return ecef_to_enu(x,y,z,ref_g)
        E0 = np.array([_enu_g(r)[0] for r in raw_rows_for_routes])
        N0 = np.array([_enu_g(r)[1] for r in raw_rows_for_routes])
        U0 = np.array([_enu_g(r)[2] for r in raw_rows_for_routes])
        ts0 = np.array([r.utc_s for r in raw_rows_for_routes])
        rate = 1.0 / float(np.median(np.diff(ts0))) if len(ts0) > 1 else 1.0
        Eg = np.array(gaussian_smooth(E0.tolist(), 2.0 * rate))
        Ng = np.array(gaussian_smooth(N0.tolist(), 2.0 * rate))
        Ug = np.array(gaussian_smooth(U0.tolist(), 10.0 * rate))
        all_smoother_outputs["gaussian_xy2_z10"] = _enu_to_pos_rows(Eg, Ng, Ug, raw_rows_for_routes)
        log(f"[multi] gaussian xy=2 z=10 OK")
    except Exception as _e:
        log(f"[multi] gaussian skipped: {_e}")

    # ns_adaptive (when ns is informative)
    try:
        from data_pipeline.ns_sigma import (
            AdaptiveBwParams, ns_is_informative, sigma_samples_from_ns,
        )
        from data_pipeline.smoothing import gaussian_smooth_adaptive_bw
        NS_arr = np.array([r.ns for r in raw_rows_for_routes], dtype=float)
        if ns_is_informative(NS_arr):
            bw = AdaptiveBwParams()
            sh = sigma_samples_from_ns(NS_arr, rate, bw, "h")
            sv = sigma_samples_from_ns(NS_arr, rate, bw, "v")
            Ea = np.array(gaussian_smooth_adaptive_bw(E0.tolist(), sh.tolist()))
            Na = np.array(gaussian_smooth_adaptive_bw(N0.tolist(), sh.tolist()))
            Ua = np.array(gaussian_smooth_adaptive_bw(U0.tolist(), sv.tolist()))
            all_smoother_outputs["ns_adaptive"] = _enu_to_pos_rows(Ea, Na, Ua, raw_rows_for_routes)
            log(f"[multi] ns_adaptive OK")
        else:
            log(f"[multi] ns_adaptive skipped (ns column not informative)")
    except Exception as _e:
        log(f"[multi] ns_adaptive skipped: {_e}")

    # epoch_weighted (if not already produced)
    if "epoch_weighted" not in all_smoother_outputs:
        try:
            from data_pipeline.epoch_weight import smooth_epoch_weighted
            stat_p = ppk_res.stat_path if (ppk_res.stat_path and ppk_res.stat_path.is_file()) else None
            Ee, Ne, Ue = smooth_epoch_weighted(raw_rows_for_routes, stat_path=stat_p)
            all_smoother_outputs["epoch_weighted"] = _enu_to_pos_rows(Ee, Ne, Ue, raw_rows_for_routes)
            log(f"[multi] epoch_weighted OK")
        except Exception as _e:
            log(f"[multi] epoch_weighted skipped: {_e}")

    # epoch_weighted_v2 (if not already produced)
    if "epoch_weighted_v2" not in all_smoother_outputs:
        try:
            from data_pipeline.epoch_weight_v2 import (
                EpochWeightV2Options, smooth_epoch_weighted_v2,
            )
            try:
                from data_pipeline.parsers import parse_imu as _parse_imu
                imu_rows_v2 = _parse_imu(raw.sensors_txt)
            except Exception:
                imu_rows_v2 = None
            stat_p = ppk_res.stat_path if (ppk_res.stat_path and ppk_res.stat_path.is_file()) else None
            v2_opts = EpochWeightV2Options(
                stat_path=stat_p, zupt_enabled=True,
                nhc_enabled=True, nhc_heading_source="doppler",
                sigma_a_base=0.10,
            )
            v2r = smooth_epoch_weighted_v2(
                raw_rows_for_routes, imu_rows=imu_rows_v2, options=v2_opts,
            )
            all_smoother_outputs["epoch_weighted_v2"] = _enu_to_pos_rows(
                v2r.E_smooth, v2r.N_smooth, v2r.U_smooth, raw_rows_for_routes,
            )
            log(f"[multi] epoch_weighted_v2 OK "
                f"(zupt={v2r.n_zupt_updates}, nhc={v2r.n_nhc_updates}, "
                f"doppler_gated={v2r.n_doppler_gated})")
        except Exception as _e:
            log(f"[multi] epoch_weighted_v2 skipped: {_e}")

    # FGO (when the factor library + Motion sensor available)
    try:
        from data_pipeline.parsers import parse_imu
        from data_pipeline.fgo import FgoOptions, run_fgo
        imu_rows = parse_imu(raw.sensors_txt)
        if imu_rows:
            fgo_res = run_fgo(raw_rows_for_routes, imu_rows, options=FgoOptions(), log=log)
            # Build PosRow list from fgo result lat/lon/h directly
            from data_pipeline.parsers import PosRow
            fgo_rows_for_dash = []
            for i, r in enumerate(raw_rows_for_routes):
                if (math.isfinite(fgo_res.lat_deg[i])
                        and math.isfinite(fgo_res.lon_deg[i])
                        and math.isfinite(fgo_res.h_m[i])):
                    fgo_rows_for_dash.append(PosRow(
                        utc_s=r.utc_s, lat_deg=fgo_res.lat_deg[i],
                        lon_deg=fgo_res.lon_deg[i], h_m=fgo_res.h_m[i],
                        quality=r.quality, ns=r.ns,
                        vn=r.vn, ve=r.ve, vu=r.vu,
                        sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
                    ))
                else:
                    fgo_rows_for_dash.append(r)
            all_smoother_outputs["fgo"] = fgo_rows_for_dash
            log(f"[multi] fgo OK")
    except ImportError as _e:
        log(f"[multi] fgo skipped (gtsam not installed): {_e}")
    except Exception as _e:
        log(f"[multi] fgo skipped: {_e}")

    log(f"[multi] total smoothers: {len(all_smoother_outputs)}")

    # ---- Routes viewer (multi-line path comparison) ----
    try:
        routes_dict = {"raw_ppk": raw_rows_for_routes, **all_smoother_outputs}
        viewers.build_routes_viewer(
            routes=routes_dict,
            out_html=out / "routes_viewer.html",
            log=log,
        )
    except Exception as e:
        log(f"[routes] WARN routes viewer failed: {e}")

    # ---- Path comparison viewer (pairwise + per-axis + speed) ----
    try:
        routes_dict = {"raw_ppk": raw_rows_for_routes, **all_smoother_outputs}
        viewers.build_trajectory_compare_viewer(
            routes=routes_dict,
            out_html=out / "trajectory_compare.html",
            log=log,
        )
    except Exception as e:
        log(f"[traj-compare] WARN failed: {e}")

    # ---- Accuracy dashboard (single-page engineer panel) ----
    try:
        from data_pipeline.stages.accuracy_dashboard import build_accuracy_dashboard
        build_accuracy_dashboard(
            raw_pos_rows=raw_rows_for_routes,
            filter_outputs=all_smoother_outputs,
            out_html=out / "accuracy_dashboard.html",
            log=log,
        )
    except Exception as e:
        log(f"[dashboard] WARN accuracy dashboard failed: {e}")

    # ---- Stage 5e: sync player (media + path + Motion sensor trust panel) ----
    if args.include_viewers:
        build_sync_player_step(
            raw=raw, pos_path=pos_path,
            frame_times_csv=fr.frame_times_csv,
            stat_path=(ppk_res.stat_path
                       if (ppk_res.stat_path and ppk_res.stat_path.is_file())
                       else None),
            out_html=out / "sync_player.html",
            log=log,
        )

    # ---- Stage 6: viewers ----
    if args.include_viewers:
        log("=" * 60); log("[bonus] viewers")
        try:
            viewers.build_trajectory_viewer(
                data_log=raw.measurements_txt,
                georef_csv=csv_path,
                out_html=out / "trajectory_viewer.html",
                recording_map=raw.recording_txt,
                log=log,
            )
            viewers.build_comparison_viewer(
                data_log=raw.measurements_txt,
                pos_file=pos_path,
                frame_times_csv=fr.frame_times_csv,
                recording_map=raw.recording_txt,
                out_html=out / "comparison_viewer.html",
                fps=args.fps,
                # Segment clip: its own anchor's min bootNs overrides the parent
                # capture_meta t0 (segment PTS are rebased to 0).
                chop_video_anchor=(getattr(raw, "chop_video_anchor", None)
                                   if getattr(raw, "is_chop", False) else None),
                log=log,
            )
            log(f"[viewers] OK")
        except Exception as e:
            log(f"[viewers] WARN failed: {e}")

    log("=" * 60)
    log(f"PIPELINE COMPLETE -> {out}")
    log(f"Key outputs:")
    log(f"  {csv_path.name}")
    log(f"  rover.pos")
    log(f"  pos_metadata.csv")
    if args.include_viewers:
        log(f"  trajectory_viewer.html (+ comparison_viewer.html, plotly.min.js)")
    return 0


def build_sync_player_step(
    *,
    raw,
    pos_path: Path,
    frame_times_csv: Path,
    stat_path: Optional[Path],
    out_html: Path,
    log=print,
) -> bool:
    """Best-effort sync-player build for the CLI (issue B, client-readiness).

    Ships the same media + path + Motion sensor-trust panel the GUI builds
    (``gui.py::_run_sync_player``) to CLI clients. Wired identically: pos +
    sample times + session map + sensors + stream/capture-meta anchors.

    Guarded: any failure logs a WARN and returns False -- it never crashes
    the pipeline run (parity with the other bonus viewers). Skips (False)
    when the session has no media.
    """
    if getattr(raw, "recording_mp4", None) is None:
        log("[sync] skipped (session has no video)")
        return False
    log("=" * 60); log("[bonus] sync player (video + trajectory + IMU trust)")
    try:
        from data_pipeline.stages import viewers as viewers_stage
        viewers_stage.build_sync_player(
            video=raw.recording_mp4,
            pos_file=pos_path,
            frame_times_csv=frame_times_csv,
            recording_map=raw.recording_txt,
            out_html=out_html,
            sensors_txt=raw.sensors_txt,
            data_log=raw.measurements_txt,
            stat_file=stat_path,
            # Stream + anchors (new-format sessions; None-safe for old format).
            wav=getattr(raw, "audio_wav", None),
            audio_anchor=getattr(raw, "audio_anchor_txt", None),
            capture_meta=getattr(raw, "capture_meta_json", None),
            video_anchor=getattr(raw, "video_anchor_txt", None),
            # Segment clip: its own anchor's min bootNs overrides the parent
            # capture_meta t0 (segment PTS are rebased to 0).
            chop_video_anchor=(getattr(raw, "chop_video_anchor", None)
                               if getattr(raw, "is_chop", False) else None),
            show_spectrogram=True,
            mux_audio=True,
            log=log,
        )
        log(f"[sync] OK -> {out_html}")
        return True
    except Exception as e:
        log(f"[sync] WARN sync player failed ({type(e).__name__}): {e}")
        return False


def _write_pos_minimal(path: Path, rows) -> None:
    """Write a minimal The external solver-style .pos file for downstream parsers.

    Format mirrors what parse_rtkpos expects. Sigmas / Rate-signal / off-diagonal
    covariance from PosRow when finite; preserved verbatim so downstream
    filters (epoch_weight) see the same data they'd see from raw The external solver.

    Atomic write: writes to ``<path>.tmp`` then renames.
    """
    import datetime as dt
    import math
    import os

    from data_pipeline.time_sync import get_leap_seconds_for_epoch

    def fmt(v, w):
        return f"{v:>{w}.6f}" if (v is not None and math.isfinite(v)) else " " * w

    def fmt_or_default(v, w, default):
        return f"{v:>{w}.4f}" if (v is not None and math.isfinite(v)) else f"{default:>{w}.4f}"

    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("% program   : data_pipeline epoch_weighted smoothed\n")
        f.write("% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp,ns=# of satellites)\n")
        f.write("%  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)      sdvn     sdve     sdvu    sdvne    sdveu    sdvun\n")
        for r in rows:
            # Per-row epoch offset (correct across epoch offset boundaries).
            gpst_unix = r.utc_s + get_leap_seconds_for_epoch(r.utc_s)
            t = dt.datetime.fromtimestamp(gpst_unix, tz=dt.timezone.utc)
            date_s = t.strftime("%Y/%m/%d")
            time_s = t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"
            f.write(
                f"{date_s} {time_s}  {r.lat_deg:.9f}   {r.lon_deg:.9f}  {r.h_m:>10.4f}  "
                f"{r.quality:>2d} {r.ns:>3d}  "
                f"{fmt(r.sd_n, 7)} {fmt(r.sd_e, 7)} {fmt(r.sd_u, 7)} "
                f"{fmt_or_default(r.sd_ne, 8, 0.0)} {fmt_or_default(r.sd_eu, 8, 0.0)} {fmt_or_default(r.sd_un, 8, 0.0)} "
                f"{fmt_or_default(r.age_s, 7, 0.0)} {fmt_or_default(r.ratio, 7, 0.0)} "
                f"{fmt(r.vn, 10)} {fmt(r.ve, 10)} {fmt(r.vu, 10)} "
                f"{fmt(r.sd_vn, 8)} {fmt(r.sd_ve, 8)} {fmt(r.sd_vu, 8)} "
                f"{fmt(r.sd_vne, 8)} {fmt(r.sd_veu, 8)} {fmt(r.sd_vun, 8)}"
                f"\n"
            )
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
