"""End-to-end Post-processing pipeline — one entry point, minimum user inputs.

The user provides:
    * `raw_folder`         — the source app session folder containing
                              measurements_*.txt, recording_*.txt,
                              recording_*.container file, sensors_*.txt
    * `base_obs`           — The reference unit (or other survey-grade) base .obs
    * `nav_files`          — list of nav/auxiliary data .26N/.26G/.26L/.26C
                              (or a directory; auto-detected by
                              :func:`data_pipeline.stages.ppk.detect_nav_files`)
    * `out_dir`            — where outputs go

The pipeline does everything else:

    1. Detect the four RAW files (RawInputs.from_folder).
    2. Convert measurements_*.txt → subject.obs via vendored android_rinex.
    3. Run the solver binary with the shipped Post-processing conf
       (or a user-supplied conf if provided).
    4. Filter the resulting .pos with `apply_kalman_smart` (Recipe 2,
       conditional ADAPTIVE second pass).
    5. Write the cleaned .pos next to the raw output.
    6. Build a per-epoch feature CSV from the (.pos, .pos.stat) pair.

All outputs land in ``out_dir`` with predictable names so the GUI / CLI
can wire each stage to a button or progress bar.

Quick CLI usage:

    python -m data_pipeline.pipeline_full \
        --raw     C:\\data\\session\\RAW \
        --base    C:\\data\\session\\base\\base_obs.26o \
        --nav-dir C:\\data\\session\\base \
        --out     C:\\data\\session\\out
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .base_pos import read_rinex_approx_xyz
from .epoch_features import build_epoch_features, write_features_csv
from .kalman_sigma import SmartKalmanOptions, apply_kalman_smart
from .parsers import PosRow, parse_rtkpos
from .pipeline import LogFn, RawInputs, make_logger
from .stages import ppk as ppk_stage
from .stages import rinex as rinex_stage
from .time_sync import GPS_UTC_LEAP_SECONDS_2026


_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
# Site-agnostic default: auto-averages base from Interchange-format header. When the
# header carries surveyed coords, run_full() auto-overrides ant2-pos*
# via base_pos.read_rinex_approx_xyz — closing the ~50 cm gap vs the
# single-point auto-average.
DEFAULT_CONF = _CONFIGS_DIR / "javad_avg_sp.conf"


@dataclass(frozen=True)
class AccuracyReport:
    """User-facing accuracy summary attached to every :func:`run_full` result.

    Numbers are derived from the cleaned The external solver output:
      - ``ci95_h_m`` / ``ci95_v_m`` = 2 * mean per-epoch The external solver sigma
        (The external solver sigmas are known to be 2-11x overconfident on device Post-processing;
        we conservatively inflate by 4x so the reported CI lands in the
        right ballpark for phase-data sessions).
      - ``fix_pct`` / ``float_pct`` / ``single_pct`` from the Q column.
      - ``mean_speed_mps`` / ``mean_sat_count`` for context.

    See :func:`format_accuracy_report` for the human-readable rendering.
    """

    n_epochs:           int
    duration_min:       float
    mean_sigma_h_m:     float
    mean_sigma_v_m:     float
    ci95_h_m:           float        # 95% horizontal CI (m), inflated x4
    ci95_v_m:           float        # 95% vertical CI (m), inflated x4
    fix_pct:            float
    float_pct:          float
    single_pct:         float
    mean_sat_count:     float
    mean_speed_mps:     float
    source_chain:       str          # e.g. "The external solver(javad_avg_sp) -> K_smart [K -> A [mixed]]"


@dataclass(frozen=True)
class FullPipelineResult:
    """Pointers to every artifact produced by :func:`run_full`."""

    raw_inputs:     RawInputs
    rover_obs:      Path
    raw_pos:        Path
    raw_pos_stat:   Optional[Path]
    cleaned_pos:    Path
    features_csv:   Optional[Path]
    n_epochs:       int
    smart_branch:   str       # "K-only" or "K -> A [regime]"
    accuracy:       "AccuracyReport"
    base_source:    str       # "Interchange-format-header" or "auto-average (avg_sp)" etc.


def _write_pos_like(
    src_template: Path, rows: Iterable[PosRow], dst: Path,
) -> None:
    """Write a solver-style .pos: keep ``%`` header from src_template,
    re-emit numeric rows from ``rows``.

    UTC→Reference time uses :data:`time_sync.GPS_UTC_LEAP_SECONDS_2026` so the
    conversion stays correct when epoch offset change. NaN sigmas / vel
    write as ``0.0000`` (The external solver convention for "unknown").
    """
    import datetime as dt
    import math
    header_lines: list[str] = []
    try:
        with src_template.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("%"):
                    header_lines.append(line)
                else:
                    break
    except OSError as ex:
        raise OSError(
            f"Could not read header template {src_template}: {ex}. "
            "Cannot emit cleaned .pos without it."
        ) from ex

    def _f(v: float) -> float:
        return v if math.isfinite(v) else 0.0

    rows = list(rows)
    with dst.open("w", encoding="utf-8") as f:
        for h in header_lines:
            f.write(h)
        for r in rows:
            t = dt.datetime.fromtimestamp(
                r.utc_s + GPS_UTC_LEAP_SECONDS_2026, tz=dt.timezone.utc,
            )
            ds = t.strftime("%Y/%m/%d %H:%M:%S.")
            us = t.strftime("%f")
            f.write(
                f"{ds}{us}  {r.lat_deg:.9f}   {r.lon_deg:.9f}    "
                f"{r.h_m:.4f}   {r.quality}   {r.ns}   "
                f"{_f(r.sd_n):.4f}   {_f(r.sd_e):.4f}   {_f(r.sd_u):.4f}   "
                f"0.0000   0.0000   0.0000   0.00    0.0    "
                f"{_f(r.vn):.5f}   {_f(r.ve):.5f}   {_f(r.vu):.5f}\n"
            )


def run_full(
    *,
    raw_folder: Path,
    base_obs: Path,
    nav_dir: Optional[Path] = None,
    nav_files: Optional[Iterable[Path]] = None,
    out_dir: Path,
    config_file: Optional[Path] = None,
    rinex_options: Optional[rinex_stage.RinexOptions] = None,
    smart_options: Optional[SmartKalmanOptions] = None,
    skip_features: bool = False,
    log: Optional[LogFn] = None,
) -> FullPipelineResult:
    """Run the four-stage pipeline. See module docstring."""
    log_ = make_logger(log)
    raw_folder = Path(raw_folder).resolve()
    base_obs = Path(base_obs).resolve()
    out_dir = Path(out_dir).resolve()

    # ---- 0. Argument validation (fail fast with actionable messages) ----
    if not raw_folder.is_dir():
        raise FileNotFoundError(
            f"RAW folder not found: {raw_folder}. "
            "Pass --raw <path> pointing at a the capture app session folder "
            "containing measurements_*.txt, recording_*.txt/.mp4, sensors_*.txt."
        )
    if not base_obs.is_file():
        raise FileNotFoundError(
            f"Base .obs not found: {base_obs}. "
            "Pass --base <path> to your survey-grade base station RINEX OBS."
        )
    if nav_files is None and nav_dir is None:
        raise ValueError(
            "Either --nav-dir or --nav must be supplied (RINEX nav files "
            ".26N/.26G/.26L/.26C or a folder containing them)."
        )
    conf_path = Path(config_file).resolve() if config_file else DEFAULT_CONF
    if not conf_path.is_file():
        raise FileNotFoundError(
            f"RTKLIB config not found: {conf_path}. "
            "Pass --conf <path> to a valid RTKLIB .conf, or ensure the "
            f"shipped default {DEFAULT_CONF.name} is present in "
            f"data_pipeline/configs/."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. RawInputs ----
    raw = RawInputs.from_folder(raw_folder)
    log_(f"[full] RAW folder: {raw_folder}")
    log_(f"[full]   measurements: {raw.measurements_txt.name}")
    log_(f"[full]   recording:    {raw.recording_txt.name}")
    log_(f"[full]   video:        {raw.recording_mp4.name}")
    log_(f"[full]   sensors:      {raw.sensors_txt.name}")

    # ---- 2. Interchange-format conversion ----
    rover_obs = out_dir / f"{raw.measurements_txt.stem}.obs"
    log_(f"[full] -> RINEX: {rover_obs.name}")
    rinex_stage.run(
        measurements_txt=raw.measurements_txt,
        output_obs=rover_obs,
        options=rinex_options,
        log=log_,
    )

    # ---- 3. Resolve nav files ----
    if nav_files is None:
        nav_dir_p = Path(nav_dir)
        if not nav_dir_p.is_dir():
            raise FileNotFoundError(
                f"--nav-dir not a directory: {nav_dir_p}. "
                "Point it at a folder containing your RINEX nav files."
            )
        nav_files_list = ppk_stage.detect_nav_files(nav_dir_p, recursive=False)
        if not nav_files_list:
            raise FileNotFoundError(
                f"No nav/ephemeris files auto-detected in {nav_dir_p}. "
                "Expected RINEX nav (e.g. .26N / .26G / .26L / .26C) or "
                ".sp3 / .clk. Pass --nav <files...> explicitly if your "
                "files use a non-standard extension."
            )
    else:
        nav_files_list = [Path(p) for p in nav_files]
        if not nav_files_list:
            raise ValueError(
                "--nav was supplied with zero files. Pass at least one "
                "RINEX nav / ephemeris file."
            )
        missing = [p for p in nav_files_list if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "These nav files do not exist: "
                + ", ".join(str(p) for p in missing)
            )
    log_(f"[full] nav files ({len(nav_files_list)}): "
         + ", ".join(p.name for p in nav_files_list[:6])
         + (" ..." if len(nav_files_list) > 6 else ""))

    # ---- 4. Post-processing ----
    # Auto-detect surveyed base coords from the Interchange-format header. The external solver's
    # auto-average (ant2-postype=single) is ~50 cm worse than surveyed
    # coords. If the base Interchange-format carries APPROX POSITION XYZ that isn't
    # the (0,0,0) placeholder, override.
    base_ecef = read_rinex_approx_xyz(base_obs)
    if base_ecef is not None:
        base_source = "RINEX-header"
        log_(f"[full] base coords from RINEX header: "
             f"X={base_ecef[0]:.2f} Y={base_ecef[1]:.2f} Z={base_ecef[2]:.2f}")
    else:
        base_source = f"{conf_path.stem} default (ant2-postype)"
        log_(f"[full] no surveyed base in RINEX header; using conf default")

    log_(f"[full] PPK conf: {conf_path.name}")
    raw_pos = out_dir / f"{raw.measurements_txt.stem}.pos"
    ppk_result = ppk_stage.run(
        rover_obs=rover_obs,
        base_obs=base_obs,
        nav_files=nav_files_list,
        config_file=conf_path,
        output_pos=raw_pos,
        base_ecef_xyz=base_ecef,
        log=log_,
    )
    raw_pos_stat = ppk_result.stat_path

    # ---- 5. Smart Recursive filter ----
    log_("[full] -> apply_kalman_smart")
    rows = parse_rtkpos(raw_pos)
    if not rows:
        raise RuntimeError(
            f"rnx2rtkp produced an empty .pos ({raw_pos}). "
            "Common causes: wrong base coordinates in conf (check ant2-pos*), "
            "no overlapping time between rover and base obs, or no usable "
            "satellite data. Check the log above for RTKLIB warnings."
        )
    smart = apply_kalman_smart(rows, options=smart_options, log=log_)
    cleaned_pos = out_dir / f"{raw.measurements_txt.stem}_clean.pos"
    _write_pos_like(raw_pos, smart.rows_out, cleaned_pos)
    log_(f"[full]    {smart.branch_label}: sigma_h={smart.mean_sigma_h_m:.2f}m, "
         f"{len(smart.rows_out)} rows -> {cleaned_pos.name}")

    # ---- 6. Per-epoch feature CSV ----
    features_csv: Optional[Path] = None
    if not skip_features and raw_pos_stat is not None and raw_pos_stat.is_file():
        log_("[full] -> per-epoch features CSV")
        # Narrow except: features CSV is optional, but only swallow the
        # parse/IO failures we expect. Programmer errors (AttributeError,
        # NameError, TypeError) still bubble up so they don't hide.
        try:
            feats = build_epoch_features(raw_pos, raw_pos_stat)
            features_csv = out_dir / f"{raw.measurements_txt.stem}_features.csv"
            n = write_features_csv(feats, features_csv)
            log_(f"[full]    wrote {n} epochs -> {features_csv.name}")
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as ex:
            features_csv = None
            log_(f"[full]    features CSV skipped: {type(ex).__name__}: {ex}")

    # ---- 7. User-facing accuracy report ----
    # Use the RAW The external solver rows for sigma stats (downstream smoothing zeroes
    # the sd_* columns) and the smoothed rows for everything else.
    accuracy = _build_accuracy_report(
        smart.rows_out, smart.mean_sigma_h_m, raw_rows=rows,
        source_chain=f"RTKLIB({conf_path.stem}, base={base_source}) -> "
                     f"K_smart [{smart.branch_label}]",
    )
    log_("")
    for line in format_accuracy_report(accuracy).splitlines():
        log_(f"[full] {line}")
    log_(f"[full] DONE: outputs in {out_dir}")
    return FullPipelineResult(
        raw_inputs=raw, rover_obs=rover_obs,
        raw_pos=raw_pos, raw_pos_stat=raw_pos_stat,
        cleaned_pos=cleaned_pos, features_csv=features_csv,
        n_epochs=len(smart.rows_out),
        smart_branch=smart.branch_label,
        accuracy=accuracy,
        base_source=base_source,
    )


def _build_accuracy_report(
    rows: list[PosRow], mean_sigma_h_kalman_m: float, *,
    source_chain: str, raw_rows: Optional[list[PosRow]] = None,
) -> "AccuracyReport":
    """Summarise the cleaned path into an :class:`AccuracyReport`.

    ``rows`` are the (possibly-smoothed) output rows used for Q-flag /
    velocity / source-count stats. ``raw_rows`` (optional) are the
    pre-smoothing The external solver rows used for sigma stats — smoothing drops
    the per-epoch sd_* columns so the cleaned rows would mis-report
    accuracy. Defaults to ``rows`` if not supplied.
    """
    import math
    import statistics
    if not rows:
        return AccuracyReport(
            n_epochs=0, duration_min=0.0,
            mean_sigma_h_m=float("nan"), mean_sigma_v_m=float("nan"),
            ci95_h_m=float("nan"), ci95_v_m=float("nan"),
            fix_pct=0.0, float_pct=0.0, single_pct=0.0,
            mean_sat_count=0.0, mean_speed_mps=0.0,
            source_chain=source_chain,
        )
    n = len(rows)
    duration_s = max(0.0, rows[-1].utc_s - rows[0].utc_s)
    sigma_src = raw_rows if raw_rows else rows
    sd_h_vals = [math.hypot(r.sd_n, r.sd_e)
                 for r in sigma_src
                 if math.isfinite(r.sd_n) and math.isfinite(r.sd_e)]
    sd_v_vals = [r.sd_u for r in sigma_src if math.isfinite(r.sd_u)]
    mean_sd_h = statistics.fmean(sd_h_vals) if sd_h_vals else mean_sigma_h_kalman_m
    mean_sd_v = statistics.fmean(sd_v_vals) if sd_v_vals else float("nan")
    # The external solver sigmas overconfident by 2-11x on device Post-processing per pos_metadata
    # calibration. Use 4x as a conservative inflation; multiply by 1.96
    # for ~95% CI from 1-sigma. Together: ~7.84x mean sd_h.
    INFLATION = 4.0
    Z95 = 1.96
    ci95_h = INFLATION * Z95 * mean_sd_h if math.isfinite(mean_sd_h) else float("nan")
    ci95_v = INFLATION * Z95 * mean_sd_v if math.isfinite(mean_sd_v) else float("nan")
    q_fix = sum(1 for r in rows if r.quality == 1)
    q_flt = sum(1 for r in rows if r.quality == 2)
    q_sng = sum(1 for r in rows if r.quality >= 4)
    speeds = [math.hypot(r.vn, r.ve)
              for r in rows
              if math.isfinite(r.vn) and math.isfinite(r.ve)]
    sat_counts = [r.ns for r in rows if r.ns > 0]
    return AccuracyReport(
        n_epochs=n,
        duration_min=duration_s / 60.0,
        mean_sigma_h_m=mean_sd_h,
        mean_sigma_v_m=mean_sd_v,
        ci95_h_m=ci95_h, ci95_v_m=ci95_v,
        fix_pct=100.0 * q_fix / n,
        float_pct=100.0 * q_flt / n,
        single_pct=100.0 * q_sng / n,
        mean_sat_count=statistics.fmean(sat_counts) if sat_counts else 0.0,
        mean_speed_mps=statistics.fmean(speeds) if speeds else 0.0,
        source_chain=source_chain,
    )


def format_accuracy_report(r: "AccuracyReport") -> str:
    """Render an :class:`AccuracyReport` as a multi-line human-readable
    block (no unicode glyphs — safe on Hebrew cp1255)."""
    import math
    def _f(v: float, w: int = 5, p: int = 2) -> str:
        return f"{v:{w}.{p}f}" if math.isfinite(v) else "  n/a"
    return (
        "===================  ACCURACY REPORT  ===================\n"
        f"Source chain: {r.source_chain}\n"
        f"Epochs:       {r.n_epochs}  over  {r.duration_min:.1f} min\n"
        f"Quality:      Fix {r.fix_pct:5.1f}%  Float {r.float_pct:5.1f}%  "
        f"Single {r.single_pct:5.1f}%\n"
        f"Mean sats:    {r.mean_sat_count:.1f}    Mean speed: "
        f"{r.mean_speed_mps:.2f} m/s\n"
        f"Mean RTKLIB sigma:   horiz={_f(r.mean_sigma_h_m)} m   "
        f"vert={_f(r.mean_sigma_v_m)} m\n"
        f"Assumed accuracy (95% CI, x4 inflation vs RTKLIB sigma):\n"
        f"     horizontal   +/- {_f(r.ci95_h_m)} m\n"
        f"     vertical     +/- {_f(r.ci95_v_m)} m\n"
        "Note: device PPK floor is ~3 m horiz / 6 m vert (1-sigma) on "
        "phase-data sessions.\n"
        "      Single-fix-heavy sessions exceed this. Supply GT track to "
        "calibrate the\n"
        "      inflation factor exactly (see data_pipeline.pos_metadata).\n"
        "========================================================="
    )


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement so unicode
    glyphs in log strings (sigma, arrows) don't crash on non-UTF-8
    Windows consoles (e.g. Hebrew cp1255)."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, type=Path,
                    help="the capture app RAW folder")
    ap.add_argument("--base", required=True, type=Path,
                    help="Base-station .obs file")
    ap.add_argument("--nav-dir", type=Path, default=None,
                    help="Folder containing nav/ephemeris .26N/.26G/etc")
    ap.add_argument("--nav", type=Path, nargs="*", default=None,
                    help="Explicit list of nav files (alternative to --nav-dir)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output folder")
    ap.add_argument("--conf", type=Path, default=None,
                    help="Override RTKLIB conf (default: shipped optimal)")
    ap.add_argument("--no-features", action="store_true",
                    help="Skip the per-epoch features CSV")
    args = ap.parse_args()
    res = run_full(
        raw_folder=args.raw,
        base_obs=args.base,
        nav_dir=args.nav_dir,
        nav_files=args.nav,
        out_dir=args.out,
        config_file=args.conf,
        skip_features=args.no_features,
        log=print,
    )
    print(f"\nclean .pos:    {res.cleaned_pos}")
    print(f"raw .pos:      {res.raw_pos}")
    if res.features_csv:
        print(f"features csv:  {res.features_csv}")
    print(f"base coords:   {res.base_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
