"""Stage 1: convert the source app ``measurements_*.txt`` to Interchange-format OBS.

This is a thin, fully-configurable wrapper around the upstream
``android_rinex/src/gnsslogger_to_rnx.py`` script. We mirror its CLI surface
so the GUI can expose every flag the user might want to tweak, while keeping
sensible defaults that match the canonical command:

    python gnsslogger_to_rnx.py --skip-edit --fix-bias -o output.obs input.txt
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..pipeline import LogFn, make_logger


@dataclass
class RinexOptions:
    """All flags the gnsslogger_to_rnx.py wrapper exposes.

    Flag names mirror upstream so the wrapper stays a transparent passthrough.
    """

    # Behaviour flags (defaults match the canonical command).
    skip_edit: bool = True
    fix_bias: bool = True

    # Identity / metadata.
    marker_name: str = "UNKN"
    observer: str = "UNKN"
    agency: str = "UNKN"
    receiver_number: str = "UNKN"
    receiver_type: str = "UNKN"
    receiver_version: str = "AndroidOS >7.0"
    antenna_number: str = "UNKN"
    antenna_type: str = "internal"

    # Numeric tuning.
    pseudorange_bias: float = 0.0
    time_adj: float = 1e-7
    slip_mask: int = 3
    filter_mode: str = "sync"
    # Signal-quality strictness preset (see KEEP_LEVEL_PRESETS in
    # vendor/android_rinex/src/the logger app.py).
    #   strict     = Google decimeter-challenge defaults (default)
    #   relaxed    = drop CNo floor to 15, ignore environment noise flag
    #   permissive = drop CNo floor to 10, no environment noise / slip checks
    # Hard filters (code lock, TOW/TOD, source group) always enforced.
    keep_level: str = "relaxed"

    # Extra flags the user may add ad-hoc (rare).
    extra: list[str] = field(default_factory=list)

    def to_cli_args(self) -> list[str]:
        """Materialise this options object into ``gnsslogger_to_rnx.py`` flags."""
        args: list[str] = []
        if self.skip_edit:
            args.append("--skip-edit")
        if self.fix_bias:
            args.append("--fix-bias")
        args += ["--marker-name", self.marker_name]
        args += ["--observer", self.observer]
        args += ["--agency", self.agency]
        args += ["--receiver-number", self.receiver_number]
        args += ["--receiver-type", self.receiver_type]
        args += ["--receiver-version", self.receiver_version]
        args += ["--antenna-number", self.antenna_number]
        args += ["--antenna-type", self.antenna_type]
        args += ["--pseudorange-bias", str(self.pseudorange_bias)]
        args += ["--time-adj", str(self.time_adj)]
        args += ["--slip-mask", str(self.slip_mask)]
        args += ["--filter-mode", self.filter_mode]
        args += ["--keep-level", self.keep_level]
        args += list(self.extra)
        return args


VENDORED_ANDROID_RINEX_SRC = (
    Path(__file__).resolve().parent.parent.parent / "vendor" / "android_rinex" / "src"
)


def find_gnsslogger_script(android_rinex_src: Path | None = None) -> Path:
    """Locate ``gnsslogger_to_rnx.py``.

    If ``android_rinex_src`` is ``None`` or doesn't contain the script, we
    fall back to the **vendored** copy under ``vendor/android_rinex/src/``.
    This keeps the repo fully self-contained for offline / air-gapped use.
    """
    candidates: list[Path] = []
    if android_rinex_src is not None:
        candidates += [
            android_rinex_src / "gnsslogger_to_rnx.py",
            android_rinex_src / "src" / "gnsslogger_to_rnx.py",
        ]
    candidates.append(VENDORED_ANDROID_RINEX_SRC / "gnsslogger_to_rnx.py")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Could not locate gnsslogger_to_rnx.py - vendored copy missing under "
        f"{VENDORED_ANDROID_RINEX_SRC} and no usable user-supplied path."
    )


def run(
    *,
    measurements_txt: Path,
    output_obs: Path,
    android_rinex_src: Path | None = None,
    options: Optional[RinexOptions] = None,
    python_exe: Optional[str] = None,
    log: Optional[LogFn] = None,
) -> Path:
    """Run android_rinex's converter to produce ``output_obs``.

    The script is invoked from the android_rinex ``src`` directory because
    upstream uses bare module imports (``from the logger app import ...``).
    """
    log_ = make_logger(log)
    options = options or RinexOptions()
    measurements_txt = Path(measurements_txt)
    if not measurements_txt.is_file():
        raise FileNotFoundError(
            f"measurements_*.txt not found: {measurements_txt}. "
            "Expected a the capture app raw measurements file (the one that starts "
            "with the 'Raw' header)."
        )
    output_obs = output_obs.resolve()
    output_obs.parent.mkdir(parents=True, exist_ok=True)

    script = find_gnsslogger_script(android_rinex_src)
    cwd = script.parent

    cmd: list[str] = [
        python_exe or sys.executable,
        str(script),
        *options.to_cli_args(),
        "-o",
        str(output_obs),
        str(measurements_txt.resolve()),
    ]
    log_(f"[rinex] cwd={cwd}")
    log_("[rinex] cmd=" + " ".join(shlex.quote(c) for c in cmd))

    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True,
        text=True, encoding='utf-8', errors='replace',
    )
    for line in (proc.stdout or "").splitlines():
        log_(f"[rinex/out] {line}")
    for line in (proc.stderr or "").splitlines():
        log_(f"[rinex/err] {line}")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            f"gnsslogger_to_rnx.py failed (exit {proc.returncode}).\n"
            f"--- last stderr ---\n{tail}\n"
            f"Common causes: device model {measurements_txt.stem} uses an "
            f"unsupported FullBiasNanos format (try --keep-level=permissive), "
            f"or the raw measurements file is incomplete."
        )
    if not output_obs.exists():
        raise RuntimeError(
            f"gnsslogger_to_rnx.py reported success but {output_obs} is missing. "
            "Check disk space + write permissions on the output folder."
        )
    # Detect header-only output (every raw measurement rejected by filter).
    has_data = _count_obs_epochs(output_obs)
    if has_data == 0:
        raise RuntimeError(
            f"{output_obs.name} contains no observation data after the header. "
            "Every raw measurement was rejected. Try a looser --keep-level "
            f"(currently '{options.keep_level}'; try 'permissive') or check "
            "that the device actually has GNSS-raw permission granted."
        )
    log_(f"[rinex] wrote {output_obs} ({output_obs.stat().st_size:,} bytes)")
    return output_obs


def _count_obs_epochs(obs_path: Path) -> int:
    """Detect whether the .obs has any observation data past the
    'END OF HEADER' marker. Returns 0 for header-only (empty) output,
    >=1 when any data line is present, -1 on read error."""
    try:
        with obs_path.open("r", encoding="utf-8", errors="replace") as f:
            in_header = True
            for line in f:
                if in_header:
                    if "END OF HEADER" in line:
                        in_header = False
                    continue
                if line.strip():
                    return 1
    except OSError:
        return -1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements", required=True, type=Path)
    ap.add_argument(
        "--android-rinex-src",
        type=Path,
        default=None,
        help=(
            "Path to an external android_rinex/src. Optional - the vendored "
            "copy under vendor/android_rinex/src is used when omitted."
        ),
    )
    ap.add_argument("--out", required=True, type=Path, help="Output .obs path.")
    ap.add_argument("--no-skip-edit", action="store_true")
    ap.add_argument("--no-fix-bias", action="store_true")
    ap.add_argument("--marker-name", default="UNKN")
    ap.add_argument("--observer", default="UNKN")
    ap.add_argument("--agency", default="UNKN")
    ap.add_argument("--filter-mode", default="sync", choices=["sync", "trck"])
    args = ap.parse_args()

    options = RinexOptions(
        skip_edit=not args.no_skip_edit,
        fix_bias=not args.no_fix_bias,
        marker_name=args.marker_name,
        observer=args.observer,
        agency=args.agency,
        filter_mode=args.filter_mode,
    )
    run(
        measurements_txt=args.measurements,
        output_obs=args.out,
        android_rinex_src=args.android_rinex_src,
        options=options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
