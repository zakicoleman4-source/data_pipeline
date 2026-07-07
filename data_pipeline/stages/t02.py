"""Signal binary → Interchange-format conversion (Trimble T02 / The reference unit JPS).

Three converters are wrapped here, picked automatically by file extension
but always overridable by the caller:

* **Trimble T02 / T01 / T00 / R00 → runpkr00 + teqc** (default for ``.t02``
  family). ``runpkr00 -g -d in.T02`` unpacks the proprietary container
  into a ``.tgd`` (or ``.dat``); ``teqc +obs out.obs +nav out.21n,out.21g
  in.tgd`` converts that into Interchange-format OBS + per-source group NAV files.

* **The reference unit JPS → jps2rin** (default for ``.jps``). Single-step convert
  written by The reference unit themselves; emits OBS + per-source group NAV files
  named after the input stem.

* **JPS → convbin** (fallback, also supports a handful of
  unit formats jps2rin doesn't). Invoked with ``-r the reference unit``.

The previous revision of this module incorrectly defaulted ``.t02`` to
``jps2rin``. T02 is Trimble's binary format, not The reference unit's — they share
nothing structurally. This revision routes by extension so the right
tool runs for the right file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..pipeline import LogFn, make_logger

# Default install locations are NOT shipped — user must set env vars
# (JPS2RIN, CONVBIN, RUNPKR00, TEQC) or place binaries on PATH.
DEFAULT_JPS2RIN: Path | None = None
DEFAULT_CONVBIN: Path | None = None
DEFAULT_RUNPKR00: Path | None = None
DEFAULT_TEQC: Path | None = None
_TEQC_FALLBACKS: tuple[Path, ...] = ()

# Standard nav/auxiliary data extensions emitted by the supported converters.
NAV_EXTENSIONS: tuple[str, ...] = (
    ".nav", ".gnav", ".hnav", ".lnav", ".qnav", ".cnav", ".inav",
    ".sp3", ".eph", ".clk",
)
# Interchange-format 2.xx single-letter nav suffixes (case-insensitive third char).
_RINEX2_NAV_LETTERS = "nglphqc"

SUPPORTED_RINEX_VERSIONS: tuple[str, ...] = (
    "2.10", "2.11", "2.12", "3.00", "3.01", "3.02", "3.03", "3.04", "3.05",
)

# File extensions handled by the Trimble pipeline.
TRIMBLE_EXTS: tuple[str, ...] = (".t02", ".t01", ".t00", ".r00", ".t04")
# File extensions handled by the reference unit pipeline.
JAVAD_EXTS: tuple[str, ...] = (".jps", ".tps", ".tpd")

VALID_CONVERTERS: tuple[str, ...] = ("trimble", "jps2rin", "convbin")


def auto_pick_converter(input_file: Path) -> str:
    """Return the default converter for ``input_file`` based on its extension.

    ``.t02 / .t01 / .t00 / .r00 / .t04`` → ``"trimble"`` (runpkr00 + teqc).
    ``.jps / .tps / .tpd``               → ``"jps2rin"``.
    Anything else                        → ``"jps2rin"`` as a permissive
    default (the caller will see a clear failure if the file is not
    actually JPS).
    """
    ext = Path(input_file).suffix.lower()
    if ext in TRIMBLE_EXTS:
        return "trimble"
    if ext in JAVAD_EXTS:
        return "jps2rin"
    return "jps2rin"


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------

def resolve_jps2rin(override: Optional[Path] = None) -> Path:
    """Locate ``jps2rin`` via the central :mod:`data_pipeline.lab_tools` resolver."""
    from ..lab_tools import resolve_tool
    return resolve_tool("jps2rin", override)


def resolve_convbin(override: Optional[Path] = None) -> Path:
    """Locate ``convbin`` via the central :mod:`data_pipeline.lab_tools` resolver."""
    from ..lab_tools import resolve_tool
    return resolve_tool("convbin", override)


def resolve_runpkr00(override: Optional[Path] = None) -> Path:
    """Locate ``runpkr00`` via the central :mod:`data_pipeline.lab_tools` resolver."""
    from ..lab_tools import resolve_tool
    return resolve_tool("runpkr00", override)


def resolve_teqc(override: Optional[Path] = None) -> Path:
    """Locate ``teqc`` via the central :mod:`data_pipeline.lab_tools` resolver."""
    from ..lab_tools import resolve_tool
    return resolve_tool("teqc", override)


# ---------------------------------------------------------------------------
# Output discovery
# ---------------------------------------------------------------------------

@dataclass
class T02ConvertResult:
    """Outcome of a successful binary → Interchange-format conversion."""
    obs_files: List[Path]
    nav_files: List[Path]
    output_dir: Path
    rinex_version: str
    converter: str
    command: List[str]
    stdout: str
    stderr: str
    returncode: int
    tool_exe: Path
    input_file: Path = field(default_factory=Path)


def _discover_outputs(output_dir: Path, stem: str) -> tuple[list[Path], list[Path]]:
    """Glob ``output_dir`` for OBS + NAV files matching ``stem``.

    Covers Interchange-format 3.x (``.obs / .nav / .gnav / .sp3 / ...``) and Interchange-format 2.xx
    (``.NNo / .NNn / .NNg / .NNl / .NNp / .NNc / .NNh / .NNq`` where the
    last char is case-insensitive).
    """
    obs: list[Path] = []
    nav: list[Path] = []
    if not output_dir.is_dir():
        return obs, nav
    stem_lower = stem.lower()
    for p in sorted(output_dir.iterdir()):
        if not p.is_file():
            continue
        if not p.stem.lower().startswith(stem_lower):
            continue
        sfx = p.suffix.lower()
        if sfx in (".obs", ".rnx"):
            obs.append(p)
            continue
        if sfx in NAV_EXTENSIONS:
            nav.append(p)
            continue
        if len(sfx) == 4 and sfx[1:3].isdigit():
            tag = sfx[3]
            if tag == "o":
                obs.append(p)
            elif tag in _RINEX2_NAV_LETTERS:
                nav.append(p)
    return obs, nav


# ---------------------------------------------------------------------------
# Converter-specific runners
# ---------------------------------------------------------------------------

def _run_jps2rin(
    *, input_file: Path, output_dir: Path, rinex_version: str,
    tool_exe: Path, timeout_s: float, log_: LogFn,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    cmd = [
        str(tool_exe),
        f"/v={rinex_version}",
        f"/o={output_dir}",
        f"/of={input_file.stem}",
        str(input_file),
    ]
    log_(f"[t02] jps2rin = {tool_exe}")
    log_(f"[t02] cmd     = {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    return cmd, proc


def _run_convbin(
    *, input_file: Path, output_dir: Path, rinex_version: str,
    tool_exe: Path, include_doppler: bool, include_snr: bool,
    timeout_s: float, log_: LogFn,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    stem = input_file.stem
    cmd: list[str] = [
        str(tool_exe),
        "-r", "javad",
        "-v", rinex_version,
        "-o", str(output_dir / f"{stem}.obs"),
        "-n", str(output_dir / f"{stem}.nav"),
        "-g", str(output_dir / f"{stem}.gnav"),
        "-h", str(output_dir / f"{stem}.hnav"),
        "-l", str(output_dir / f"{stem}.lnav"),
        "-q", str(output_dir / f"{stem}.qnav"),
        "-b", str(output_dir / f"{stem}.cnav"),
    ]
    if include_doppler:
        cmd.append("-od")
    if include_snr:
        cmd.append("-os")
    cmd.append(str(input_file))
    log_(f"[t02] convbin = {tool_exe}")
    log_(f"[t02] cmd     = {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    return cmd, proc


def _run_trimble(
    *, input_file: Path, output_dir: Path, rinex_version: str,
    runpkr00_exe: Path, teqc_exe: Path, gps_week: Optional[int],
    timeout_s: float, log_: LogFn,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    """Run the Trimble two-step pipeline: ``runpkr00`` then ``teqc``.

    ``runpkr00`` writes its outputs into the current working directory, so
    the file is staged into ``output_dir`` first; ``teqc`` then converts the
    ``.tgd`` / ``.dat`` to Interchange-format OBS + NAV files named after the input stem.
    Both steps are run sequentially; their combined stdout/stderr is bundled
    into the returned ``CompletedProcess`` for surface in the GUI log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem

    # 1) Stage the input into output_dir so runpkr00's CWD-relative outputs
    #    land where we want them.
    staged = output_dir / input_file.name
    if staged.resolve() != input_file.resolve():
        # Stream-copy preserves disk-efficient streaming for multi-GB T02s
        # — read_bytes() + write_bytes() would load the entire file into
        # the Python heap.
        shutil.copy2(input_file, staged)

    # 2) runpkr00 -g -d <staged.T02> → <stem.tgd>
    runpkr_cmd = [str(runpkr00_exe), "-g", "-d", str(staged.name)]
    log_(f"[t02] runpkr00 = {runpkr00_exe}")
    log_(f"[t02] cmd #1   = {' '.join(runpkr_cmd)}")
    r1 = subprocess.run(
        runpkr_cmd, cwd=str(output_dir),
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    if r1.returncode != 0:
        return runpkr_cmd, r1

    # Find the produced raw file in output_dir.
    raw_candidates = sorted(output_dir.glob(f"{stem}.tgd")) \
                   + sorted(output_dir.glob(f"{stem}.dat"))
    if not raw_candidates:
        # Fabricate a "failed" CompletedProcess so the caller sees an error.
        r1 = subprocess.CompletedProcess(
            args=runpkr_cmd, returncode=1,
            stdout=r1.stdout,
            stderr=(r1.stderr or "") + f"\nrunpkr00 produced no .tgd/.dat in {output_dir}",
        )
        return runpkr_cmd, r1
    raw = raw_candidates[0]

    # 3) teqc +obs <out.obs> +nav <out.21n>,<out.21g> [-week N] <raw>
    #    Use Interchange-format 2.11 naming for nav files (teqc 2019Feb25 only emits
    #    Interchange-format 2.xx reliably; user can opt out via rinex_version).
    yy = "21"  # filename suffix only; auxiliary data time is in the file header
    obs_out = output_dir / f"{stem}.{yy}o"
    nav_gps = output_dir / f"{stem}.{yy}n"
    nav_glo = output_dir / f"{stem}.{yy}g"
    nav_gal = output_dir / f"{stem}.{yy}l"
    teqc_cmd: list[str] = [str(teqc_exe)]
    if gps_week is not None:
        teqc_cmd += ["-week", str(int(gps_week))]
    teqc_cmd += [
        "+obs", str(obs_out),
        "+nav", f"{nav_gps},{nav_glo},{nav_gal}",
        str(raw),
    ]
    log_(f"[t02] teqc     = {teqc_exe}")
    log_(f"[t02] cmd #2   = {' '.join(teqc_cmd)}")
    r2 = subprocess.run(
        teqc_cmd, cwd=str(output_dir),
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )

    # Combine stdout/stderr from both stages so the caller sees the full story.
    merged = subprocess.CompletedProcess(
        args=runpkr_cmd + ["&&"] + teqc_cmd,
        returncode=r2.returncode,
        stdout=(r1.stdout or "") + (r2.stdout or ""),
        stderr=(r1.stderr or "") + (r2.stderr or ""),
    )
    return merged.args, merged


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(
    *,
    input_file: Path,
    output_dir: Path,
    rinex_version: str = "3.05",
    converter: Optional[str] = None,
    tool_exe: Optional[Path] = None,
    teqc_exe: Optional[Path] = None,
    include_doppler: bool = True,
    include_snr: bool = True,
    gps_week: Optional[int] = None,
    timeout_s: float = 1800.0,
    log: Optional[LogFn] = None,
) -> T02ConvertResult:
    """Convert ``input_file`` to Interchange-format OBS + NAV in ``output_dir``.

    ``converter`` is ``"trimble"`` (runpkr00 + teqc) / ``"jps2rin"`` /
    ``"convbin"``. When ``None`` (default) the converter is auto-picked
    from the input extension.

    For the Trimble pipeline ``tool_exe`` overrides ``runpkr00`` and
    ``teqc_exe`` overrides ``teqc``. ``gps_week`` is an optional override
    forwarded to ``teqc`` as ``-week N`` for files where the embedded
    week number rolled over (teqc emits a warning when this happens).
    """
    log_ = make_logger(log)
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    if not input_file.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")
    if rinex_version not in SUPPORTED_RINEX_VERSIONS:
        raise ValueError(
            f"unsupported RINEX version {rinex_version!r}. "
            f"Supported: {', '.join(SUPPORTED_RINEX_VERSIONS)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if converter is None:
        converter = auto_pick_converter(input_file)
        log_(f"[t02] auto-picked converter = {converter!r} "
             f"(based on {input_file.suffix} extension)")
    converter = converter.lower()
    if converter not in VALID_CONVERTERS:
        raise ValueError(
            f"unknown converter {converter!r}. "
            f"Use one of: {', '.join(VALID_CONVERTERS)}"
        )

    if converter == "trimble":
        runpkr00_exe = resolve_runpkr00(tool_exe)
        teqc_path = resolve_teqc(teqc_exe)
        cmd, proc = _run_trimble(
            input_file=input_file, output_dir=output_dir,
            rinex_version=rinex_version,
            runpkr00_exe=runpkr00_exe, teqc_exe=teqc_path,
            gps_week=gps_week, timeout_s=timeout_s, log_=log_,
        )
        # The "tool_exe" surfaced on the result is the unpacker; teqc lives
        # in the merged command line.
        primary_exe = runpkr00_exe
    elif converter == "jps2rin":
        primary_exe = resolve_jps2rin(tool_exe)
        cmd, proc = _run_jps2rin(
            input_file=input_file, output_dir=output_dir,
            rinex_version=rinex_version, tool_exe=primary_exe,
            timeout_s=timeout_s, log_=log_,
        )
    else:  # convbin
        primary_exe = resolve_convbin(tool_exe)
        cmd, proc = _run_convbin(
            input_file=input_file, output_dir=output_dir,
            rinex_version=rinex_version, tool_exe=primary_exe,
            include_doppler=include_doppler, include_snr=include_snr,
            timeout_s=timeout_s, log_=log_,
        )

    if proc.stdout:
        for line in proc.stdout.splitlines():
            log_(line)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log_(line)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{converter} pipeline failed with exit code {proc.returncode}. "
            f"Command: {' '.join(cmd)}"
        )

    obs, nav = _discover_outputs(output_dir, input_file.stem)
    if not obs:
        raise RuntimeError(
            f"{converter} returned 0 but no RINEX OBS file appeared in "
            f"{output_dir}. Check the input file and try the alternative "
            f"converter."
        )

    log_(f"[t02] OBS: {[p.name for p in obs]}")
    log_(f"[t02] NAV: {[p.name for p in nav]}")
    return T02ConvertResult(
        obs_files=obs,
        nav_files=nav,
        output_dir=output_dir,
        rinex_version=rinex_version,
        converter=converter,
        command=cmd,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        tool_exe=primary_exe,
        input_file=input_file,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to .t02 / .jps / .tps file.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for RINEX OBS + NAV.")
    ap.add_argument("--converter", choices=VALID_CONVERTERS, default=None,
                    help="Force a converter (default: auto by extension).")
    ap.add_argument("--rinex", default="3.05",
                    choices=SUPPORTED_RINEX_VERSIONS)
    ap.add_argument("--tool", type=Path, default=None,
                    help="Override primary converter (or runpkr00 for Trimble).")
    ap.add_argument("--teqc", type=Path, default=None,
                    help="Override teqc path (Trimble converter only).")
    ap.add_argument("--week", type=int, default=None,
                    help="Force GPS week (teqc -week, Trimble only).")
    ap.add_argument("--no-doppler", action="store_true",
                    help="Drop Doppler column (convbin only).")
    ap.add_argument("--no-snr", action="store_true",
                    help="Drop SNR column (convbin only).")
    args = ap.parse_args()
    run(
        input_file=args.input,
        output_dir=args.out,
        rinex_version=args.rinex,
        converter=args.converter,
        tool_exe=args.tool,
        teqc_exe=args.teqc,
        gps_week=args.week,
        include_doppler=not args.no_doppler,
        include_snr=not args.no_snr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
