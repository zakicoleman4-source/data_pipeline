"""Motion sensor calibration store: compute, save, load Allan-derived noise params.

Calibration is keyed by a USER-ENTERED DEVICE LABEL. The source app hardcodes
the header device model (``SM-S901B``) regardless of the real device, so the
auto device-id is unreliable and must NOT be used as the key. The operator
types a human label (e.g. "Eli's S23 Ultra") and that string is the identity.

Two calibration sources, in priority order (decision = BOTH):

1. **dedicated static session** — a ``sensors_*.txt`` logged while the
   device source still on a desk. Preferred: the entire record is stationary, so
   the Allan curve is clean and reaches long averaging times.
2. **mined ZUPT segments** — when only a drive is available, reuse the
   existing stationary detection (:func:`parsers.detect_static_periods` over
   the ``.pos`` velocity) to find stops, slice the Motion sensor rows inside those
   windows, and concatenate them into a synthetic "static" stream. Noisier and
   shorter, but works without a dedicated session.

The result is serialised to JSON and can be re-loaded by device label for
reuse across sessions.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

from .allan import AllanResult, compute_allan
from .parsers import ImuRow, PosRow, detect_static_periods, parse_imu

# Schema version so future readers can migrate old files.
CALIBRATION_SCHEMA_VERSION = 1

# Where calibrations live by default (next to other user configs).
DEFAULT_CALIB_DIRNAME = "imu_calibrations"


@dataclass
class AxisParams:
    """Per-axis Allan params, JSON-friendly."""

    random_walk: float          # ARW (rate sensor) or VRW (linear sensor), units/sqrt(Hz)
    bias_instability: float     # B coefficient, channel units
    rate_random_walk: float     # K coefficient, channel units


@dataclass
class ImuCalibration:
    """A saved Motion sensor calibration keyed by user-entered device label."""

    device_label: str
    date: str                    # ISO-8601 date the calibration was computed
    sample_rate_hz: float
    duration_s: float
    source: str                  # "dedicated_static" | "mined_zupt"
    # Per-axis maps keyed by channel name (gx/gy/gz/ax/ay/az).
    axes: dict[str, AxisParams] = field(default_factory=dict)
    n_samples: int = 0
    n_static_segments: int = 0   # only meaningful for mined_zupt
    warnings: list[str] = field(default_factory=list)
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    # ---- aggregate accessors used by the fusion mapping ----
    def mean_gyro_arw(self) -> float:
        vals = [self.axes[a].random_walk for a in ("gx", "gy", "gz") if a in self.axes]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def mean_accel_vrw(self) -> float:
        vals = [self.axes[a].random_walk for a in ("ax", "ay", "az") if a in self.axes]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def mean_accel_bias_instability(self) -> float:
        vals = [self.axes[a].bias_instability for a in ("ax", "ay", "az") if a in self.axes]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def mean_gyro_bias_instability(self) -> float:
        vals = [self.axes[a].bias_instability for a in ("gx", "gy", "gz") if a in self.axes]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    # ---- serialisation ----
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ImuCalibration":
        axes = {
            name: AxisParams(**vals) if isinstance(vals, dict) else vals
            for name, vals in (d.get("axes") or {}).items()
        }
        return cls(
            device_label=d["device_label"],
            date=d.get("date", ""),
            sample_rate_hz=float(d.get("sample_rate_hz", float("nan"))),
            duration_s=float(d.get("duration_s", 0.0)),
            source=d.get("source", "unknown"),
            axes=axes,
            n_samples=int(d.get("n_samples", 0)),
            n_static_segments=int(d.get("n_static_segments", 0)),
            warnings=list(d.get("warnings", [])),
            schema_version=int(d.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
        )


def _allan_to_calibration(
    res: AllanResult,
    device_label: str,
    source: str,
    n_static_segments: int = 0,
    when: Optional[dt.date] = None,
) -> ImuCalibration:
    axes = {
        name: AxisParams(
            random_walk=float(a.random_walk),
            bias_instability=float(a.bias_instability),
            rate_random_walk=float(a.rate_random_walk),
        )
        for name, a in res.axes.items()
    }
    return ImuCalibration(
        device_label=device_label.strip(),
        date=(when or dt.date.today()).isoformat(),
        sample_rate_hz=float(res.sample_rate_hz),
        duration_s=float(res.duration_s),
        source=source,
        axes=axes,
        n_samples=int(res.n_samples),
        n_static_segments=n_static_segments,
        warnings=list(res.warnings),
    )


def compute_calibration(
    device_label: str,
    *,
    static_imu_path: Optional[Path] = None,
    drive_imu_path: Optional[Path] = None,
    drive_pos_rows: Optional[Sequence[PosRow]] = None,
    imu_rows: Optional[Sequence[ImuRow]] = None,
    max_static_speed_mps: float = 0.4,
    min_static_duration_s: float = 2.0,
) -> ImuCalibration:
    """Compute an Motion sensor calibration, session which source was used.

    Priority (decision = BOTH, prefer dedicated static):

    1. ``static_imu_path`` (or pre-parsed ``imu_rows`` flagged as static):
       a dedicated static session -> use the whole stream.
    2. else mine ZUPT segments: requires ``drive_imu_path`` (or its parsed
       rows) plus ``drive_pos_rows`` carrying velocity, run
       :func:`detect_static_periods`, slice Motion sensor rows inside the stops.

    Raises ``ValueError`` if neither source is usable.
    """
    if not device_label or not device_label.strip():
        raise ValueError("device_label is required (user-entered identity).")

    # --- source 1: dedicated static ---
    if static_imu_path is not None:
        rows = parse_imu(Path(static_imu_path))
        if not rows:
            raise ValueError(f"No IMU rows parsed from {static_imu_path}")
        res = compute_allan(rows)
        return _allan_to_calibration(res, device_label, source="dedicated_static")

    if imu_rows is not None and drive_pos_rows is None:
        # Caller hands us pre-parsed rows and asserts they are static.
        rows = list(imu_rows)
        if not rows:
            raise ValueError("imu_rows empty.")
        res = compute_allan(rows)
        return _allan_to_calibration(res, device_label, source="dedicated_static")

    # --- source 2: mine ZUPT from a drive ---
    if drive_pos_rows is None:
        raise ValueError(
            "No calibration source: supply static_imu_path (dedicated static "
            "recording) OR drive_pos_rows + drive_imu_path/imu_rows to mine "
            "stationary ZUPT segments."
        )
    drive_rows = list(imu_rows) if imu_rows is not None else (
        parse_imu(Path(drive_imu_path)) if drive_imu_path is not None else []
    )
    if not drive_rows:
        raise ValueError("No drive IMU rows available to mine ZUPT segments.")

    periods = detect_static_periods(
        list(drive_pos_rows),
        min_duration_s=min_static_duration_s,
        max_speed_mps=max_static_speed_mps,
    )
    if not periods:
        raise ValueError(
            "No stationary ZUPT segments found in the drive (need stops with "
            "velocity < %.2f m/s for >= %.1fs). Record a dedicated static log "
            "instead." % (max_static_speed_mps, min_static_duration_s)
        )

    # Slice Motion sensor rows inside each static window and concatenate.
    static_rows: list[ImuRow] = []
    for (t0, t1) in periods:
        static_rows.extend(r for r in drive_rows if t0 <= r.utc_s <= t1)
    static_rows.sort(key=lambda r: r.utc_s)
    if len(static_rows) < 3:
        raise ValueError(
            "Mined ZUPT segments contained too few IMU samples for Allan "
            "analysis. Record a dedicated static log instead."
        )

    res = compute_allan(static_rows)
    cal = _allan_to_calibration(
        res, device_label, source="mined_zupt", n_static_segments=len(periods),
    )
    cal.warnings.append(
        f"Calibration mined from {len(periods)} stationary ZUPT segment(s) of "
        f"a drive; a dedicated static recording is more reliable."
    )
    return cal


# ---------------------------------------------------------------------------
# Save / load by device label
# ---------------------------------------------------------------------------

def _safe_filename(label: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label.strip())
    return keep or "calibration"


def save_calibration(cal: ImuCalibration, path: Path) -> Path:
    """Write a calibration to a JSON file. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cal.to_dict(), f, indent=2, sort_keys=False)
    return path


def export_calibration(cal: ImuCalibration, directory: Path) -> Path:
    """Save under ``directory`` with a filename derived from the device label."""
    fname = f"imu_calib_{_safe_filename(cal.device_label)}.json"
    return save_calibration(cal, Path(directory) / fname)


def load_calibration(path: Path) -> ImuCalibration:
    """Load a calibration JSON. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ImuCalibration.from_dict(data)


def find_calibration_by_label(directory: Path, device_label: str) -> Optional[ImuCalibration]:
    """Search ``directory`` for a saved calibration matching ``device_label``.

    Match is case-insensitive on the stored ``device_label`` field (not the
    filename), so re-typing the same label finds the calibration. Returns the
    most recent match, or None.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    target = device_label.strip().lower()
    matches: list[ImuCalibration] = []
    for jf in directory.glob("*.json"):
        try:
            cal = load_calibration(jf)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if cal.device_label.strip().lower() == target:
            matches.append(cal)
    if not matches:
        return None
    matches.sort(key=lambda c: c.date, reverse=True)
    return matches[0]


def compute_calibration_via_plugin(
    backend: str,
    device_label: str,
    sensors_rows,
    options: Optional[dict] = None,
) -> "ImuCalibration":
    """Compute a calibration using a registered CalibrationPlugin backend.

    Lets an external (drop-in or entry-point) calibration method stand in for
    the built-in Allan computation. The plugin's output dict is already schema-
    validated at register time, so ``from_dict`` round-trips cleanly.
    """
    from .plugins_api import get_calibration_plugin

    plugin = get_calibration_plugin(backend)
    opts = dict(options or {})
    opts.setdefault("device_label", device_label)
    d = plugin.compute(list(sensors_rows), opts)
    return ImuCalibration.from_dict(d)
