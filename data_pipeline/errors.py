"""Central error registry + user-reportable error codes.

Why this exists
===============
End users hit failures in the field. A bare Python traceback is
useless: they screenshot it, send it, we can't tell which of 50
possible causes was real, and reproduction takes a round-trip.

This module gives every failure point a STABLE CODE
(``E-PP-NNN`` — "PipelinePython, number") plus a structured
``PipelineError`` carrying:

* ``code``     — stable identifier, e.g. ``"E-PP-101"``
* ``message``  — what went wrong, in operator English
* ``hint``     — actionable fix, e.g. "set --base in CLI or pick a
                  Base file in the GUI Inputs tab"
* ``context``  — dict of relevant paths / values / version strings

Workers in the GUI catch ``PipelineError`` and:
1. Show the code + message + hint in a messagebox
2. Append a structured record to ``~/.data_pipeline_last_error.json``
3. Tell the user "If this keeps happening, send last_error.json to
   support@..."

When support gets ``last_error.json`` they can map the code straight
to the failing call site without guessing.

How to add a new code
=====================
1. Pick the next free number in the relevant range below.
2. Add a constant in :data:`ERROR_CODES` with a one-line
   description.
3. ``raise PipelineError("E-PP-NNN", "operator message",
   hint="how to fix", context={...})`` at the failure point.
4. (Optional) Add a test that the code fires.

Code ranges
===========
* ``E-PP-001..099`` reserved for environment + install failures
  (missing binaries, wrong Python version, missing deps)
* ``E-PP-100..199`` Post-processing / Interchange-format stage
* ``E-PP-200..299`` Motion sensor / sensors parsing
* ``E-PP-300..399`` Media / samples stage
* ``E-PP-400..499`` Smoothers (cv_rts, ekf_smoothed, epoch_weight, FGO)
* ``E-PP-500..599`` Motion model
* ``E-PP-600..699`` Coordinate output CSV / outputs
* ``E-PP-700..799`` GUI / Tk
* ``E-PP-900..999`` Reserved for internal-invariant violations
                    ("a bug — please report")
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code registry — keep one-line descriptions here so the operator-facing
# tooling can look up any code without reading the raising code.
# ---------------------------------------------------------------------------
ERROR_CODES: dict[str, str] = {
    # Environment + install -----------------------------------------------
    "E-PP-001": "Python version below minimum (3.10).",
    "E-PP-002": "Required package missing (numpy / scipy / opencv-python / ...).",
    "E-PP-003": "Optional dep gtsam missing — FGO smoother disabled.",
    "E-PP-004": "External binary not found (rnx2rtkp / convbin / jps2rin / runpkr00 / teqc).",
    "E-PP-005": "ffmpeg binary not found on PATH or in vendor/ffmpeg/bin.",
    # Post-processing / Interchange-format ---------------------------------------------------------
    "E-PP-100": "Base .obs file not found.",
    "E-PP-101": "Rover .obs file not found.",
    "E-PP-102": "No navigation/ephemeris file alongside the .obs.",
    "E-PP-103": "rnx2rtkp non-zero exit (see captured stderr in `context`).",
    "E-PP-104": "RTKLIB produced an empty .pos (no usable epochs).",
    "E-PP-105": "Could not parse user-supplied base position spec.",
    "E-PP-110": "the capture app raw file has zero usable measurements "
                "(check device model + FullBiasNanos availability).",
    "E-PP-106": "RINEX header malformed or unsupported version.",
    # Motion sensor / sensors -------------------------------------------------------
    "E-PP-200": "sensors_*.txt not found.",
    "E-PP-201": "sensors_*.txt has zero parseable IMU rows.",
    "E-PP-202": "measurements_*.txt (data log) parse failed.",
    # Media / samples ------------------------------------------------------
    "E-PP-300": "Video file not found.",
    "E-PP-301": "Video cannot be opened (codec missing / corrupt).",
    "E-PP-302": "Video ended before requested frame index.",
    "E-PP-303": "Frame write failed (disk full / permissions / path too long).",
    "E-PP-304": "recording_*.txt time-anchor file not found.",
    "E-PP-305": "Time-anchor fit failed (need at least 2 anchor rows).",
    "E-PP-306": "Empty recording_*.txt AND no usable boottime in measurements_*.txt (anchor unrecoverable).",
    # Smoothers -----------------------------------------------------------
    "E-PP-400": "Smoother received empty input.",
    "E-PP-401": "Smoother got mismatched-length input arrays.",
    "E-PP-402": "Smoother sigma misconfigured (negative / non-finite).",
    "E-PP-403": "Smoother chi-2 gate non-positive.",
    "E-PP-404": "EKF ZUPT threshold > static threshold (impossible config).",
    "E-PP-405": "FGO LM optimizer diverged — returned raw PPK trajectory.",
    "E-PP-406": "FGO requires gtsam — not installed (E-PP-003).",
    # Motion model -----------------------------------------------------------------
    "E-PP-500": "VIO video could not be opened.",
    "E-PP-501": "VIO produced zero valid samples "
                "(check video has motion + good lighting).",
    "E-PP-502": "R_body_from_cam calibration failed "
                "(need at least 20 PPK rows with horizontal speed > 3 m/s).",
    # Outputs -------------------------------------------------------------
    "E-PP-600": "Output directory not writable.",
    "E-PP-601": "PPK interpolation at frame times produced zero points.",
    # GUI -----------------------------------------------------------------
    "E-PP-700": "GUI worker raised an uncaught exception (see traceback in `context`).",
    "E-PP-701": "Drag-and-drop path contains characters the encoder can't handle.",
    # Internal-invariant violations --------------------------------------
    "E-PP-900": "Internal invariant violated — this is a bug. Please report.",
}


@dataclass
class PipelineError(Exception):
    """Structured error for every operator-visible failure.

    Raise from a failure point with a stable code so support can map
    the code straight to the call site without guessing.
    """

    code: str
    message: str
    hint: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            # Don't crash inside an error path — just log + flag.
            _log.warning(
                "PipelineError raised with unregistered code %r — "
                "add it to data_pipeline/errors.py::ERROR_CODES",
                self.code,
            )
        super().__init__(self.format())

    def format(self) -> str:
        """One-line summary suitable for messagebox / log line."""
        parts = [f"[{self.code}] {self.message}"]
        if self.hint:
            parts.append(f"Fix: {self.hint}")
        return " — ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "context": self.context,
            "description": ERROR_CODES.get(self.code, "(unregistered code)"),
        }


# ---------------------------------------------------------------------------
# Last-error report file — operator sends this to support
# ---------------------------------------------------------------------------

def _default_report_path() -> Path:
    """Per-user file used by :func:`save_error_report`.

    Lives in the user's home directory so the same path is found on
    every OS and survives across pipeline runs.
    """
    return Path.home() / ".data_pipeline_last_error.json"


def _version_info() -> dict[str, str]:
    """Lightweight env snapshot embedded into every error report."""
    from . import __version__ as pp_version
    return {
        "data_pipeline_version": pp_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }


def save_error_report(
    err: BaseException,
    *,
    path: Optional[Path] = None,
    stage: str = "",
) -> Path:
    """Persist a JSON record of ``err`` for the operator to mail support.

    Returns the file path. Never raises — last-resort error path must
    not itself throw. ``stage`` is a free-text tag (e.g. ``"post-processing"``,
    ``"samples"``) so support can map the failure to the GUI tab.
    """
    out_path = path or _default_report_path()
    record: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "env": _version_info(),
    }
    if isinstance(err, PipelineError):
        record.update(err.to_dict())
        record["traceback"] = "".join(
            traceback.format_exception(type(err), err, err.__traceback__)
        )
    else:
        record.update({
            "code": "E-PP-900",
            "message": f"{type(err).__name__}: {err}",
            "hint": "Unhandled exception — likely a bug. Please report.",
            "context": {},
            "description": ERROR_CODES["E-PP-900"],
            "traceback": "".join(
                traceback.format_exception(type(err), err, err.__traceback__)
            ),
        })
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except OSError as oe:
        # Disk full / permissions — log + give up silently.
        _log.warning("could not save error report to %s: %s", out_path, oe)
    return out_path


# ---------------------------------------------------------------------------
# Decorators / helpers for wrapping worker functions
# ---------------------------------------------------------------------------

def report_user_message(err: BaseException) -> str:
    """Format ``err`` as a one-screen message for a messagebox.

    Adds the standard tail telling the user where the JSON report
    lives so they can attach it to a bug mail.
    """
    if isinstance(err, PipelineError):
        head = err.format()
    else:
        head = f"[E-PP-900] {type(err).__name__}: {err}"
    return (
        f"{head}\n\n"
        f"If this keeps happening, send the file\n"
        f"  {_default_report_path()}\n"
        f"to support."
    )
