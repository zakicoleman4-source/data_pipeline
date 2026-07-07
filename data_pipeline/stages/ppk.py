"""Post-processing via the external solver's command-line binary.

This stage is a thin, well-instrumented wrapper around the external solver
binary. The processing options (frequency, refinement mode, propagation
models, etc.) are taken entirely from a solver-format config file
(``.conf``) so the user can drop in any config prepared with the vendor's
other tools without translation.

Command pattern (mirrors the sister automation project)::

    <solver>.exe -k <config.conf> -o <out.pos> <subject.obs> <base.obs> <aux1> [<aux2> ...]

Auxiliary inputs may be any of the interchange-format variants (.nav / .??n
/ .??g / .??p / .??l / .rnx), GNAV, precise auxiliary data (.sp3 / .eph),
clock (.clk), or any other file the solver binary understands. Wildcard
expressions are honoured by the solver itself when wrapped in quotes
(``"path/*.nav"``).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..pipeline import LogFn, make_logger


# The solver binary location is resolved at runtime in this order:
#   1. ``RNX2RTKP`` env var (explicit override)
#   2. vendor/rtklib/rnx2rtkp.exe (shipped with the project)
#   3. anywhere on PATH
# No hard-coded developer install path is assumed.
DEFAULT_RTKLIB_DIR: Path | None = None
_RNX2RTKP_NAME = "rnx2rtkp.exe" if os.name == "nt" else "rnx2rtkp"

# Common extensions for nav/auxiliary data auto-detection alongside an .obs file.
NAV_EXTENSIONS: tuple[str, ...] = (
    ".nav", ".gnav", ".hnav", ".lnav", ".qnav", ".cnav",
    ".sp3", ".eph", ".clk", ".sbs", ".ionex", ".inx",
    ".rnx",  # Interchange-format 3.x mixed nav (the solver binary parses type from header).
)
# Hatanaka-style three-letter suffixes are detected separately because they
# are file *name* patterns rather than extensions (e.g. "subject.24n").
_RINEX_NAV_NAME_PATTERN = ("[0-9][0-9][nglp]", "[0-9][0-9][NGLP]")


# rnx2rtkp emits a per-epoch progress line to stderr terminated by a bare
# "\r" (unconditionally -- it never checks isatty). A long / high-rate session
# therefore captures hundreds of thousands of progress "lines". Anything that
# re-logs or retains that stream verbatim must collapse it first.
_LINE_BREAK_RE = re.compile(r"[\r\n]+")

# Cap on how much of the raw stdout/stderr streams PpkResult retains.
# Full retention tripled the memory footprint of a long session for no
# benefit -- every consumer only ever wants the tail.
_RESULT_STREAM_MAX_CHARS = 4000


def _tail_lines(text: str, n: int = 50) -> str:
    """Return the last ``n`` non-empty lines of ``text`` joined by newlines.

    Treats both ``\\r`` and ``\\n`` as line breaks so rnx2rtkp's
    carriage-return progress stream collapses to its meaningful tail
    instead of being handled as one enormous line.
    """
    if not text:
        return ""
    lines = [ln for ln in _LINE_BREAK_RE.split(text) if ln.strip()]
    return "\n".join(lines[-n:])


def resolve_rnx2rtkp(override: Optional[Path] = None) -> Path:
    """Locate ``the solver binary`` via the central :mod:`data_pipeline.lab_tools` resolver.

    Honours, in order: ``override`` argument → ``RNX2RTKP`` env var →
    ``data_to_frames.config.json`` next to the package → the
    lab-developer default install path → system ``PATH``. Raises
    :class:`FileNotFoundError` listing every probed path on miss.
    """
    from ..lab_tools import resolve_tool
    return resolve_tool("rnx2rtkp", override)


def list_config_files(rtklib_dir: Optional[Path] = None) -> list[Path]:
    """Return ``*.conf`` files in the external solver install directory (preset menu)."""
    if rtklib_dir is not None:
        base: Optional[Path] = Path(rtklib_dir)
    else:
        base = DEFAULT_RTKLIB_DIR
    if base is None or not base.is_dir():
        return []
    return sorted(base.glob("*.conf"))


def detect_nav_files(*search_dirs: Path, recursive: bool = True) -> list[Path]:
    """Heuristically find nav/auxiliary data files alongside the .obs files.

    Search is performed in each provided directory; results are de-duplicated
    while preserving discovery order. Extension matching is case-insensitive.
    With ``recursive=True`` (default) every subdirectory is searched too,
    which lets users dump subject + base + nav into one folder tree without
    flattening it first.
    """
    seen: dict[Path, None] = {}
    glob_method = "rglob" if recursive else "glob"
    for d in search_dirs:
        if d is None:
            continue
        d = Path(d)
        if not d.is_dir():
            continue
        candidates: list[Path] = []
        for ext in NAV_EXTENSIONS:
            candidates.extend(getattr(d, glob_method)(f"*{ext}"))
            candidates.extend(getattr(d, glob_method)(f"*{ext.upper()}"))
        # Interchange-format 2.xx nav files (e.g. base.24n, base.24g, base.24p, base.24l).
        iter_method = d.rglob("*") if recursive else d.iterdir()
        for stem in iter_method:
            if not stem.is_file():
                continue
            sfx = stem.suffix.lower()
            if len(sfx) == 4 and sfx[1:3].isdigit() and sfx[3] in "ngplh":
                candidates.append(stem)
        for c in candidates:
            seen.setdefault(c.resolve(), None)
    return list(seen.keys())


# The external solver config keys that pin the reference input's position.
_BASE_POS_KEYS = ("ant2-postype", "ant2-pos1", "ant2-pos2", "ant2-pos3")
# Pattern allowing inline comments like ``# (deg|m)`` after the value.
_CONF_LINE_RE = re.compile(r"^(\s*[\w\-]+\s*=\s*)([^#\r\n]*)(#.*)?$")


def write_patched_config(
    src_conf: Path,
    dst_conf: Path,
    *,
    base_ecef_xyz: tuple[float, float, float],
    log: Optional[LogFn] = None,
) -> Path:
    """Copy ``src_conf`` to ``dst_conf`` with the base position overwritten.

    The four The external solver keys ``ant2-postype``, ``ant2-pos1``, ``ant2-pos2`` and
    ``ant2-pos3`` are forced to Cartesian XYZ XYZ in metres regardless of what the
    source config used. Lines absent from the source are appended at the
    bottom so the resulting file is always self-contained. Inline comments
    (``# ...``) are preserved.
    """
    log_ = make_logger(log)
    src_conf = Path(src_conf)
    dst_conf = Path(dst_conf)
    x, y, z = base_ecef_xyz
    replacements = {
        "ant2-postype": "xyz",
        "ant2-pos1": f"{x:.4f}",
        "ant2-pos2": f"{y:.4f}",
        "ant2-pos3": f"{z:.4f}",
    }
    seen: set[str] = set()
    out_lines: list[str] = []

    with src_conf.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            m = _CONF_LINE_RE.match(line)
            if not m:
                out_lines.append(line)
                continue
            prefix = m.group(1)
            key = prefix.split("=", 1)[0].strip().lower()
            if key in replacements:
                comment = m.group(3) or ""
                new_val = replacements[key]
                pad = " " if comment else ""
                out_lines.append(f"{prefix}{new_val}{pad}{comment}".rstrip())
                seen.add(key)
            else:
                out_lines.append(line)

    missing = [k for k in _BASE_POS_KEYS if k not in seen]
    if missing:
        out_lines.append("")
        out_lines.append("# Base position injected by data_pipeline.stages.ppk")
        for key in missing:
            out_lines.append(f"{key:<18}={replacements[key]}")

    dst_conf.parent.mkdir(parents=True, exist_ok=True)
    dst_conf.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log_(
        f"[ppk] base position patched into {dst_conf.name}: "
        f"X={x:.3f} Y={y:.3f} Z={z:.3f}"
    )
    return dst_conf


@dataclass
class PpkResult:
    """Outcome of a successful Post-processing run.

    ``clean_pos_path`` is the sibling ``.pos`` written by the outlier
    filter when ``outlier_filter`` was passed to :func:`run`. It has the
    same column layout as the raw The external solver output so downstream consumers
    can use either file interchangeably; the cleaned one is the safer
    default for Export format / viewers / Coordinate output CSV.
    """
    pos_path: Path
    stat_path: Optional[Path]
    command: List[str]
    # ``stdout`` / ``stderr`` hold only the *tail* of the captured streams
    # (last ~50 meaningful lines, capped at _RESULT_STREAM_MAX_CHARS chars).
    # rnx2rtkp emits a "\r" progress line per epoch, so retaining the full
    # streams on a long session costs hundreds of MB for nothing.
    stdout: str
    stderr: str
    returncode: int
    rnx2rtkp_exe: Path
    nav_files: List[Path] = field(default_factory=list)
    clean_pos_path: Optional[Path] = None
    clean_result: Optional["CleanResult"] = None


def run(
    *,
    rover_obs: Path,
    base_obs: Path,
    nav_files: Sequence[Path],
    config_file: Path,
    output_pos: Path,
    rnx2rtkp_exe: Optional[Path] = None,
    base_ecef_xyz: Optional[tuple[float, float, float]] = None,
    timeout_s: float = 3600.0,
    outlier_filter: Optional["OutlierFilterOptions"] = None,
    log: Optional[LogFn] = None,
) -> PpkResult:
    """Run ``the solver binary`` and return a :class:`PpkResult`.

    All four input paths must exist and ``nav_files`` must be non-empty.
    The output parent directory is created if needed. The function blocks
    until ``the solver binary`` exits or ``timeout_s`` elapses.

    When ``base_ecef_xyz`` is given, a temporary patched copy of
    ``config_file`` is generated next to ``output_pos`` with the base
    position keys forced to Cartesian XYZ XYZ, and that copy is passed to
    ``the solver binary`` instead of the original.
    """
    log_ = make_logger(log)
    exe = resolve_rnx2rtkp(rnx2rtkp_exe)
    rover_obs = Path(rover_obs)
    base_obs = Path(base_obs)
    config_file = Path(config_file)
    output_pos = Path(output_pos)

    from ..errors import PipelineError
    if not rover_obs.is_file():
        raise PipelineError(
            "E-PP-101",
            f"Rover .obs not found at {rover_obs}",
            hint="Pass --rover-obs in CLI or pick a Rover file in GUI Inputs tab. "
                 "Expected RINEX 2.10/3.x OBS produced by the device (via RINEX stage).",
            context={"rover_obs": str(rover_obs)},
        )
    if not base_obs.is_file():
        raise PipelineError(
            "E-PP-100",
            f"Base .obs not found at {base_obs}",
            hint="Pass --base in CLI or pick a Base file in GUI Inputs tab. "
                 "Expected RINEX 2.10/3.x OBS produced by your survey receiver.",
            context={"base_obs": str(base_obs)},
        )
    if not config_file.is_file():
        raise PipelineError(
            "E-PP-103",
            f"RTKLIB config file not found at {config_file}",
            hint="Pick a preset from the GUI PPK tab (handsetbase, "
                 "javad_avg_sp) or supply your own .conf via --conf.",
            context={"config_file": str(config_file)},
        )
    nav_list = [Path(n) for n in nav_files]
    if not nav_list:
        raise PipelineError(
            "E-PP-102",
            "At least one navigation/ephemeris file is required (got 0).",
            hint="Drop a .nav / .rnx / .sp3 / .qnav / etc. next to your base "
                 ".obs OR pass --nav. The pipeline auto-detects extensions "
                 "from `data_pipeline.stages.ppk.NAV_EXTENSIONS`.",
            context={"obs_dir": str(base_obs.parent)},
        )
    for n in nav_list:
        if not n.is_file():
            raise PipelineError(
                "E-PP-102",
                f"Navigation file not found at {n}",
                hint="Check the path and re-run; nav file glob is detected "
                     "relative to the base .obs directory.",
                context={"nav_file": str(n)},
            )

    # Fine measurements pre-check: a duty-cycled device obs has every L* value
    # zeroed, which silently degrades the whole Post-processing run to code Differential (Q=4).
    # Warn loudly up front but never abort (Differential may be all they can get),
    # and never let a parse hiccup break the run.
    try:
        from ..obs_check import check_carrier_phase
        phase_report = check_carrier_phase(rover_obs)
        if not phase_report.has_phase:
            log_("[ppk] " + "!" * 70)
            log_(f"[ppk] WARNING: {phase_report.message}")
            log_("[ppk] Continuing anyway - expect code-DGPS quality only.")
            log_("[ppk] " + "!" * 70)
        else:
            log_(f"[ppk] rover obs phase pre-check: {phase_report.message}")
    except Exception as e:  # pragma: no cover - defensive: never fatal
        log_(f"[ppk] rover obs phase pre-check skipped ({e})")

    output_pos.parent.mkdir(parents=True, exist_ok=True)

    if base_ecef_xyz is not None:
        # Patched-config name includes the output stem so parallel runs
        # don't clobber each other's patched .conf when they share an
        # output directory.
        patched = output_pos.parent / f"{output_pos.stem}.patched.conf"
        config_file = write_patched_config(
            config_file, patched, base_ecef_xyz=base_ecef_xyz, log=log_,
        )

    # Deduplicate nav list while preserving caller-supplied order; The external solver
    # otherwise re-reads the same file and wastes a few seconds per dup.
    seen_nav: set[Path] = set()
    nav_list_dedup: list[Path] = []
    for n in nav_list:
        rp = n.resolve()
        if rp in seen_nav:
            continue
        seen_nav.add(rp)
        nav_list_dedup.append(n)
    nav_list = nav_list_dedup

    cmd: list[str] = [
        str(exe),
        "-k", str(config_file),
        "-o", str(output_pos),
        str(rover_obs),
        str(base_obs),
    ]
    cmd.extend(str(n) for n in nav_list)

    log_(f"[ppk] exe = {exe}")
    log_(f"[ppk] rover = {rover_obs.name}")
    log_(f"[ppk] base  = {base_obs.name}")
    log_(f"[ppk] nav   = {[n.name for n in nav_list]}")
    log_(f"[ppk] config = {config_file.name}")
    log_(f"[ppk] output = {output_pos}")
    if base_ecef_xyz is not None:
        log_(f"[ppk] base ECEF = "
             f"X={base_ecef_xyz[0]:.4f} Y={base_ecef_xyz[1]:.4f} "
             f"Z={base_ecef_xyz[2]:.4f}")
    log_(f"[ppk] cmd = {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PipelineError(
            "E-PP-103",
            f"rnx2rtkp timed out after {timeout_s:.0f}s",
            hint="Bump `timeout_s` if the session is large (multi-hour PPK "
                 "on a slow disk), or shorten the rover/base time window.",
            context={"command": cmd, "stdout_tail": _tail_lines(e.stdout or "", 30)},
        ) from e
    except OSError as e:
        raise PipelineError(
            "E-PP-004",
            f"Could not launch rnx2rtkp at {exe}: {e}",
            hint="The resolver returned a path but the OS refused to exec it. "
                 "Check antivirus / permissions / file-corruption; reinstall "
                 "RTKLIB or repoint via the RNX2RTKP env var.",
            context={"exe": str(exe), "command": cmd},
        ) from e

    # Do NOT re-log the full captured streams: rnx2rtkp's stderr contains one
    # "\r"-terminated progress line per epoch, so a long session yields
    # 100k-500k lines. Dumping them all into the GUI log queue at once blocks
    # the Tk mainloop for minutes and can panic Tk's text B-tree allocator.
    # Log only a one-line summary plus the last ~50 meaningful lines.
    tail = _tail_lines((proc.stdout or "") + "\n" + (proc.stderr or ""), 50)
    log_(f"[ppk] rnx2rtkp finished rc={proc.returncode}; last lines:")
    for line in tail.splitlines():
        # Guard against a single break-free mega-line flooding the log.
        log_(line if len(line) <= 2000 else line[:2000] + " ...[truncated]")

    if proc.returncode != 0:
        tail = _tail_lines(proc.stderr, 30) or _tail_lines(proc.stdout, 30)
        raise PipelineError(
            "E-PP-103",
            f"rnx2rtkp exited with code {proc.returncode}",
            hint="Inspect the captured stderr in the error report's context. "
                 "Common causes: time-range mismatch between base and rover, "
                 "wrong nav file, base position not surveyed.",
            context={
                "returncode": proc.returncode,
                "command": cmd,
                "stderr_tail": tail,
                "rover_obs": str(rover_obs),
                "base_obs": str(base_obs),
                "config": str(config_file),
            },
        )
    if not output_pos.is_file():
        tail = _tail_lines(proc.stdout, 30) or _tail_lines(proc.stderr, 30)
        raise PipelineError(
            "E-PP-104",
            f"rnx2rtkp returned 0 but produced no .pos file at {output_pos}",
            hint="Check the config file (.conf) and the input observation "
                 "windows overlap. RTKLIB silently writes nothing when the "
                 "rover/base time ranges don't intersect.",
            context={
                "expected_pos": str(output_pos),
                "stdout_tail": tail,
                "command": cmd,
            },
        )

    # Content check: The external solver can exit 0 but emit a header-only .pos when
    # no epochs survive (time-window mismatch / all sources masked). Count
    # non-comment lines and fail fast with an actionable message.
    n_data_rows = 0
    with output_pos.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("%"):
                n_data_rows += 1
                if n_data_rows >= 1:
                    break
    if n_data_rows == 0:
        tail = _tail_lines(proc.stdout, 30) or _tail_lines(proc.stderr, 30)
        raise PipelineError(
            "E-PP-104",
            f"rnx2rtkp wrote {output_pos.name} but it has 0 data epochs",
            hint="Common causes: rover/base time windows do not overlap; "
                 "elevation mask too high; SNR mask too tight; missing "
                 "constellation in nav files. Inspect the RTKLIB stderr "
                 "tail in the error report for the real complaint.",
            context={
                "pos_path": str(output_pos),
                "stdout_tail": tail,
                "command": cmd,
            },
        )

    size = output_pos.stat().st_size
    log_(f"[ppk] wrote {output_pos} ({size:,} bytes)")
    stat_path = output_pos.with_suffix(output_pos.suffix + ".stat")

    # Optional outlier-filter post-process. Defaults are conservative so
    # the cleaned file is always safe to drop into the viewers / CSV / Export format
    # batch without polluting them with bad fixes or environment noise jumps.
    clean_pos_path: Optional[Path] = None
    clean_result: Optional[CleanResult] = None
    if outlier_filter is not None:
        clean_pos_path = output_pos.with_name(output_pos.stem + "_clean.pos")
        clean_result = clean_pos(
            output_pos, clean_pos_path,
            options=outlier_filter, log=log_,
        )

    return PpkResult(
        pos_path=output_pos,
        stat_path=stat_path if stat_path.is_file() else None,
        command=cmd,
        # Retain only the tail of each stream -- see _RESULT_STREAM_MAX_CHARS.
        stdout=_tail_lines(proc.stdout or "")[-_RESULT_STREAM_MAX_CHARS:],
        stderr=_tail_lines(proc.stderr or "")[-_RESULT_STREAM_MAX_CHARS:],
        returncode=proc.returncode,
        rnx2rtkp_exe=exe,
        nav_files=nav_list,
        clean_pos_path=clean_pos_path,
        clean_result=clean_result,
    )


# ----------------------------------------------------------------------
# Convenience: javad_avg_sp.conf + user-supplied base coordinate
# ----------------------------------------------------------------------

def _packaged_conf_dir() -> Path:
    """Directory holding the configs shipped inside ``data_pipeline``.

    Returned even when the file isn't present so callers can list /
    error on the path directly.
    """
    return Path(__file__).resolve().parent.parent / "configs"


def list_packaged_configs() -> list[Path]:
    """All ``*.conf`` files shipped inside ``data_pipeline/configs/``.

    GUI dropdowns + CLI ``--conf`` autocomplete use this so they don't
    have to hard-code preset names.
    """
    d = _packaged_conf_dir()
    if not d.is_dir():
        return []
    return sorted(d.glob("*.conf"))


def run_with_user_base(
    *,
    rover_obs: Path,
    base_obs: Path,
    nav_files: Sequence[Path],
    output_pos: Path,
    base_spec: str,
    config_name: str = "javad_avg_sp.conf",
    rnx2rtkp_exe: Optional[Path] = None,
    timeout_s: float = 3600.0,
    outlier_filter: Optional["OutlierFilterOptions"] = None,
    log: Optional[LogFn] = None,
) -> PpkResult:
    """Run Post-processing with a packaged ``.conf`` template and a user-chosen base.

    Solves the most common operator request: "use *javad_avg_sp.conf*
    (or whichever preset) but pin the base to coordinates I know".

    Parameters
    ----------
    base_spec
        Free-form base position. Auto-detected via
        :func:`data_pipeline.base_pos.parse_base_spec`. Accepted forms::

            "lat,lon,h"                       e.g. "45.000000,0.000000,100.00"
            "llh:lat,lon,h"
            "cartesian XYZ:X,Y,Z"                      e.g. "cartesian XYZ:4517590.0,0.0,4487348.0"
            "X,Y,Z"                           bare Cartesian XYZ if all |val|>100 km
            "interchange-format:/path/to/base.obs"         pull APPROX POSITION XYZ from header

    config_name
        File name (no path) of a preset in ``data_pipeline/configs/``.
        Defaults to ``"javad_avg_sp.conf"``; pass another preset name to
        use it instead.

    Raises
    ------
    PipelineError E-PP-103
        If ``config_name`` isn't a shipped preset.
    PipelineError E-PP-105
        If ``base_spec`` can't be parsed.
    """
    log_ = make_logger(log)
    from ..base_pos import parse_base_spec
    from ..errors import PipelineError

    conf_dir = _packaged_conf_dir()
    conf_path = conf_dir / config_name
    if not conf_path.is_file():
        available = ", ".join(p.name for p in list_packaged_configs())
        raise PipelineError(
            "E-PP-103",
            f"Packaged config {config_name!r} not found in {conf_dir}.",
            hint=f"Choose one of: {available or '(none shipped)'}. "
                 "Or pass `config_file=...` to ppk.run() directly.",
            context={"config_name": config_name, "search_dir": str(conf_dir)},
        )

    ecef = parse_base_spec(base_spec)
    if ecef is None:
        raise PipelineError(
            "E-PP-105",
            f"Could not parse base position spec {base_spec!r}.",
            hint=(
                "Use one of: 'lat,lon,h' (decimal degrees, metres) | "
                "'llh:lat,lon,h' | 'ecef:X,Y,Z' (metres) | 'X,Y,Z' "
                "(bare ECEF if all |val|>100km) | 'rinex:/path/base.obs' "
                "(reads APPROX POSITION XYZ from the .obs header)."
            ),
            context={"base_spec": base_spec},
        )
    log_(f"[ppk-user-base] using config {conf_path.name}; "
         f"base ECEF X={ecef[0]:.4f} Y={ecef[1]:.4f} Z={ecef[2]:.4f}")
    return run(
        rover_obs=rover_obs,
        base_obs=base_obs,
        nav_files=nav_files,
        config_file=conf_path,
        output_pos=output_pos,
        rnx2rtkp_exe=rnx2rtkp_exe,
        base_ecef_xyz=ecef,
        timeout_s=timeout_s,
        outlier_filter=outlier_filter,
        log=log,
    )


# ----------------------------------------------------------------------
# Outlier rejection (post-process of the .pos the solver binary produced)
# ----------------------------------------------------------------------


@dataclass
class OutlierFilterOptions:
    """Thresholds for :func:`clean_pos`. Defaults trade a small amount of
    coverage for a hard guarantee that every kept epoch is usable
    downstream (no bad fixes leaking into the Coordinate output CSV, viewers,
    or Export format overlays).

    A row is rejected when **any** of the following fires:

    * its quality flag is in ``drop_q_values`` (Single + Degraded by default)
    * its quality flag is greater than ``max_q`` (keep Fix + Float by default)
    * its ``ns`` (sources solved) is below ``min_ns``
    * its horizontal standard deviation ``hypot(sdn, sde)`` exceeds
      ``max_sigma_h_m``
    * its vertical sd ``sdu`` exceeds ``max_sigma_v_m``
    * its differential ``age`` exceeds ``max_age_s``
    * ``hypot(vn, ve)`` exceeds ``max_speed_mps`` (defends against rare
      velocity-blowup outliers)
    * its 3D Cartesian XYZ position is more than ``position_jump_m`` from the
      **median** of the previous ``jump_window`` kept rows — catches
      environment noise jumps, cycle-slip wobbles, etc.

    Set ``enabled=False`` to make :func:`clean_pos` a no-op pass-through.
    """

    enabled: bool = True
    max_q: int = 2                          # 1=Fix, 2=Float, 4=Differential, 5=Single
    drop_q_values: tuple[int, ...] = (5, 6)
    min_ns: int = 5
    max_sigma_h_m: float = 5.0
    max_sigma_v_m: float = 10.0
    max_age_s: float = 30.0
    max_speed_mps: float = 50.0
    position_jump_m: float = 30.0
    jump_window: int = 5                    # rolling neighbours for median test


@dataclass(frozen=True)
class CleanResult:
    """Outcome of a :func:`clean_pos` call.

    ``out_path`` is the cleaned ``.pos`` file (drop-in replacement for the
    raw one). ``rejected_by`` is a per-criterion histogram useful for
    quickly spotting which filter was the dominant cause of loss.
    """

    out_path: Path
    n_in: int
    n_out: int
    rejected_by: dict[str, int]
    summary: str


def _parse_pos_numeric(parts: list[str]) -> Optional[dict]:
    """Parse a solver .pos data row into numeric fields.

    Standard The external solver column layout (whitespace-separated)::

        0:date  1:time  2:lat  3:lon  4:h   5:Q   6:ns
        7:sdn   8:sde   9:sdu  10:sdne 11:sdeu 12:sdun
        13:age  14:ratio  15:vn  16:ve  17:vu

    Returns ``None`` if the row is too short or any required field fails
    to parse. Velocity / age columns are optional and stored as NaN when
    absent.
    """
    if len(parts) < 7:
        return None
    try:
        q = int(parts[5])
        ns = int(parts[6])
    except ValueError:
        return None
    out: dict = {"q": q, "ns": ns}
    # Position uncertainties (cols 7-9 are 1-sigma sd-NEU in metres).
    try:
        out["sdn"] = float(parts[7]) if len(parts) > 7 else float("nan")
        out["sde"] = float(parts[8]) if len(parts) > 8 else float("nan")
        out["sdu"] = float(parts[9]) if len(parts) > 9 else float("nan")
    except ValueError:
        out["sdn"] = out["sde"] = out["sdu"] = float("nan")
    # Differential age (col 13).
    try:
        out["age"] = float(parts[13]) if len(parts) > 13 else float("nan")
    except ValueError:
        out["age"] = float("nan")
    # Velocities (cols 15-17).
    try:
        out["vn"] = float(parts[15]) if len(parts) > 15 else float("nan")
        out["ve"] = float(parts[16]) if len(parts) > 16 else float("nan")
        out["vu"] = float(parts[17]) if len(parts) > 17 else float("nan")
    except ValueError:
        out["vn"] = out["ve"] = out["vu"] = float("nan")
    # Lat/lon/height for jump test.
    try:
        out["lat"] = float(parts[2])
        out["lon"] = float(parts[3])
        out["h"]   = float(parts[4])
    except ValueError:
        return None
    return out


def _llh_to_ecef_local(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    """Tiny The standard datum LLH→Cartesian XYZ (avoids importing geo.py from this stage)."""
    import math as _m
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    lat = _m.radians(lat_deg)
    lon = _m.radians(lon_deg)
    sl, cl = _m.sin(lat), _m.cos(lat)
    N = a / _m.sqrt(1 - e2 * sl * sl)
    x = (N + h_m) * cl * _m.cos(lon)
    y = (N + h_m) * cl * _m.sin(lon)
    z = (N * (1 - e2) + h_m) * sl
    return x, y, z


def clean_pos(
    in_pos: Path,
    out_pos: Path,
    *,
    options: Optional[OutlierFilterOptions] = None,
    log: Optional[LogFn] = None,
) -> CleanResult:
    """Drop outlier epochs from a The external solver ``.pos`` file.

    Header lines (``% ...``) and any line not parseable as a data row are
    copied through verbatim. Data rows are kept only when they pass every
    filter in :class:`OutlierFilterOptions`.

    The position-jump test uses the median (not mean) of the last
    ``jump_window`` *kept* rows in Cartesian XYZ. Median is robust to a single
    outlier creeping into the window before the filter catches up; this
    matters when a environment noise jump precedes the algorithm noticing.

    The cleaned file has the **same** column layout as the input so any
    existing ``.pos`` parser (rtkplot, our :func:`parsers.parse_rtkpos`,
    Coordinate output import scripts, etc.) reads it without changes.
    """
    import math as _m
    log_ = make_logger(log)
    options = options or OutlierFilterOptions()
    in_pos = Path(in_pos)
    out_pos = Path(out_pos)
    out_pos.parent.mkdir(parents=True, exist_ok=True)

    if not options.enabled:
        shutil.copyfile(in_pos, out_pos)
        return CleanResult(
            out_path=out_pos, n_in=0, n_out=0,
            rejected_by={}, summary="outlier filter disabled (file copied)",
        )

    reasons = {
        "q_dropvalue":      0,
        "q_above_max":      0,
        "low_ns":           0,
        "high_sigma_h":     0,
        "high_sigma_v":     0,
        "high_age":         0,
        "high_speed":       0,
        "position_jump":    0,
        "unparseable":      0,
    }
    kept_ecef: list[tuple[float, float, float]] = []
    n_in = 0
    n_out = 0

    with in_pos.open("r", encoding="utf-8", errors="replace") as fin, \
         out_pos.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\r\n")
            if not line:
                fout.write(raw)
                continue
            if line.startswith("%"):
                fout.write(raw)
                continue

            parts = line.split()
            d = _parse_pos_numeric(parts)
            if d is None:
                reasons["unparseable"] += 1
                # Drop silently — most likely a malformed mid-file row.
                continue
            n_in += 1
            q   = d["q"]
            ns  = d["ns"]
            # NaN sigmas treated as "unknown — fail the sanity filter".
            # Defaulting to 0.0 would let The external solver-blind epochs slip through
            # the sdh/sdv > max threshold check (0.0 > anything is False).
            sdh = _m.hypot(d["sdn"], d["sde"]) if (
                _m.isfinite(d["sdn"]) and _m.isfinite(d["sde"])
            ) else float("inf")
            sdv = d["sdu"] if _m.isfinite(d["sdu"]) else float("inf")
            age = d["age"] if _m.isfinite(d["age"]) else 0.0
            spd = _m.hypot(d["vn"], d["ve"]) if (
                _m.isfinite(d["vn"]) and _m.isfinite(d["ve"])
            ) else 0.0

            if q in options.drop_q_values:
                reasons["q_dropvalue"] += 1
                continue
            if q > options.max_q:
                reasons["q_above_max"] += 1
                continue
            if ns < options.min_ns:
                reasons["low_ns"] += 1
                continue
            if sdh > options.max_sigma_h_m:
                reasons["high_sigma_h"] += 1
                continue
            if sdv > options.max_sigma_v_m:
                reasons["high_sigma_v"] += 1
                continue
            if age > options.max_age_s:
                reasons["high_age"] += 1
                continue
            if spd > options.max_speed_mps:
                reasons["high_speed"] += 1
                continue

            # Position-jump test (against median of last N kept rows).
            xyz = _llh_to_ecef_local(d["lat"], d["lon"], d["h"])
            if len(kept_ecef) >= options.jump_window:
                xs = sorted(p[0] for p in kept_ecef[-options.jump_window:])
                ys = sorted(p[1] for p in kept_ecef[-options.jump_window:])
                zs = sorted(p[2] for p in kept_ecef[-options.jump_window:])
                m = options.jump_window // 2
                med = (xs[m], ys[m], zs[m])
                dist = _m.sqrt(
                    (xyz[0] - med[0]) ** 2
                    + (xyz[1] - med[1]) ** 2
                    + (xyz[2] - med[2]) ** 2
                )
                if dist > options.position_jump_m:
                    reasons["position_jump"] += 1
                    continue

            kept_ecef.append(xyz)
            fout.write(raw)
            n_out += 1

    rej_total = n_in - n_out
    pct = (100.0 * rej_total / n_in) if n_in else 0.0
    summary = (
        f"input epochs: {n_in}  kept: {n_out}  rejected: {rej_total} ({pct:.1f}%)\n"
        + "\n".join(f"  - {k}: {v}" for k, v in reasons.items() if v > 0)
    )
    log_("[ppk] outlier filter:")
    for line in summary.splitlines():
        log_(f"[ppk]   {line}")
    return CleanResult(
        out_path=out_pos, n_in=n_in, n_out=n_out,
        rejected_by={k: v for k, v in reasons.items() if v > 0},
        summary=summary,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rover", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path, nargs="+")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rnx2rtkp", type=Path, default=None,
                    help="Override path to rnx2rtkp executable.")
    ap.add_argument("--no-clean", action="store_true",
                    help="Skip the post-process outlier filter.")
    args = ap.parse_args()
    res = run(
        rover_obs=args.rover,
        base_obs=args.base,
        nav_files=args.nav,
        config_file=args.config,
        output_pos=args.out,
        rnx2rtkp_exe=args.rnx2rtkp,
        outlier_filter=None if args.no_clean else OutlierFilterOptions(),
    )
    if res.clean_pos_path is not None:
        print(f"cleaned .pos -> {res.clean_pos_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
