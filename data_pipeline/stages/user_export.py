"""User-facing path export.

Writes a single CSV per chosen path with the columns clients ask for:

    reference time, lat, lon, h, x, y, z, vn, ve, vu,
    std_xy, std_xy_smart, std_vn, std_ve, std_vu,
    trust_class, source, trust_label_v2, gap,
    pos_within_bar, vel_trusted

Where:
  - reference time         : Reference time timestamp (seconds since 1980-01-06 00:00:00 UTC, Reference).
                      Computed = utc_s + leap_seconds(utc_s).
  - lat,lon,h       : The standard datum ellipsoidal degrees + metres.
  - x,y,z           : Cartesian XYZ metres (computed from lat/lon/h).
  - vn,ve,vu        : Rate-signal (or smoother-derived) velocity, m/s.
  - std_xy          : The external solver-only horizontal 1-sigma (sd_n,sd_e quadrature
                      × inflation_factor from local-variance calibration).
                      Tends to under-report by 1.5–7× — see std_xy_smart.
  - std_xy_smart    : ``accuracy_predictor.smart_session_std`` —
                      validated against 14 GT sessions (the reference set/reference site/session 2-6),
                      med < 2×smart on 14/14, p95 < 3×smart on 12/14.
                      Adds Q-mix ambig bias + environment noise spike bonus +
                      hidden-bias detection on top of raw_sd × inflation.
  - std_vn/ve/vu    : Velocity 1-sigma in m/s (The external solver sd_vn/sd_ve/sd_vu).
  - trust_class     : per-session label — 'trustworthy', 'tight',
                      'spike_risk'. Client report should explicitly warn
                      on 'spike_risk' that occasional epochs may exceed
                      the predicted envelope.
  - source          : provenance tag (smoother name) for downstream audit.

No reference assumed — std is the *predicted* error from session
features + calibration; ``std_xy_smart`` is the validated number to
use; ``std_xy`` is shipped for back-compat.

2026-07-05 additions:
  - ``time_bases`` chooser (backward-compatible by default): 'reference time' /
    'utc' / 'stream' / 'iso' TIME columns, anchored to the stream timeline
    (default = the single historical ``reference time`` column, byte-for-byte).
  - ``coord_systems`` chooser (backward-compatible by default): 'datum-based' /
    'cartesian XYZ' / 'grid' / 'local-frame' coordinate blocks (default = datum-based+cartesian XYZ, the
    historical column set unchanged).
  - Z (height) smoothing -- NOT backward-compatible, INTENTIONALLY DEFAULT ON
    (explicit client product decision): ``smooth_z=True`` gaussian-smooths
    ``h_m`` over the time axis (``z_sigma_s=3.0`` s) for every caller, which
    changes the exported height AND every height-derived coordinate
    (Cartesian XYZ x/y/z, Grid h, Local-frame u, Export format altitude) plus downstream trust/sigma
    inputs relative to pre-2026-07-05 exports. Horizontal lat/lon are never
    touched. Pass ``smooth_z=False`` to restore raw (unsmoothed) heights.

Writes atomically (.tmp -> os.replace) so an aborted run leaves no
half-written file.
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from ..accuracy_predictor import predicted_epoch_std, smart_session_std
from ..epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
from ..geo import ecef_to_enu, llh_to_ecef
from ..parsers import PosRow
from ..pos_metadata import calibrate_sigma_inflation
from ..robust_filter import DROP as _FILTER_DROP, RobustFilterConfig, robust_filter
from ..time_sync import get_leap_seconds_for_epoch
from ..trust_formula_v2 import compute_trust_v2


# ---------------------------------------------------------------------------
# Winning robust_filter preset (P2/P3, 2026-06-29) — the export DEFAULT.
# ---------------------------------------------------------------------------
# Baked from accuracy_2sigma_2026-06-29/winning_preset.json. This is the proven
# FILTER-ONLY win: it crushes the physically-impossible Post-processing divergence spikes
# (s21/101315 altitude 160 m / 172 km/h teleport -> MAX error) and is a strict
# no-op on clean routes (clean-route 2-sigma guard regress <= 0.001 m in P2).
# NOTE (P3 correction): the Motion sensor-calibrated *fusion* stage regressed MAX on some
# spiked sessions, so fusion is NOT part of the default — only robust_filter is.
def winning_export_filter() -> RobustFilterConfig:
    """The shipped robust_filter preset run by ``export_trajectory`` by default."""
    return RobustFilterConfig(
        max_horiz_speed_mps=45.0,
        max_vert_speed_mps=8.0,
        alt_below_median_m=30.0,
        alt_above_median_m=40.0,
        jump_mad_k=6.0,
        jump_floor_m=8.0,
        max_repair_epochs=10,
        max_repair_seconds=12.0,
        disagreement_reject_m=5.0,
        # Automotive car-physics gates (2026-07-05): speed-aware jump gate rejects
        # physically-impossible lateral jumps a fixed MAD floor misses while
        # preserving hard-braking. Turn-rate gate (dt>=0.5 s guard + 2.5x grip
        # safety, hardened 2026-07-05) catches impossible yaw spikes; verified
        # false-flag-free on the real day14 dodge track (0 lone corner flags on
        # 1412 epochs). Both seed their reference only from clean epochs.
        # (tests/test_automotive_filter.py)
        speed_gate_enabled=True,
        turn_rate_enabled=True,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Export coordinate systems (client chooser, 2026-07-05).
# ---------------------------------------------------------------------------
# ``coord_systems`` selects which coordinate blocks the CSV carries. The
# default reproduces the historical column set exactly (datum-based + Cartesian XYZ).
# Column blocks per system (emitted after ``reference time``, in request order,
# duplicate column names emitted once — e.g. ``h_m`` when both 'datum-based'
# and 'grid' are requested):
#   datum-based : lat_deg, lon_deg, h_m               (The standard datum deg / ellipsoidal m)
#   cartesian XYZ     : x_ecef_m, y_ecef_m, z_ecef_m        (The standard datum Cartesian XYZ metres)
#   grid      : utm_easting_m, utm_northing_m,      (auto-picked Grid zone from
#              utm_zone, h_m                        the FIRST valid fix; EPSG
#                                                   in a '#' header comment)
#   local-frame      : e_m, n_m, u_m                       (local Local-frame, origin = first
#                                                   valid fix of the export)
DEFAULT_COORD_SYSTEMS: tuple[str, ...] = ("geodetic", "ecef")
SUPPORTED_COORD_SYSTEMS: tuple[str, ...] = ("geodetic", "ecef", "utm", "enu")
DEFAULT_Z_SIGMA_S: float = 3.0

_COORD_COLUMNS: dict[str, tuple[str, ...]] = {
    "geodetic": ("lat_deg", "lon_deg", "h_m"),
    "ecef": ("x_ecef_m", "y_ecef_m", "z_ecef_m"),
    "utm": ("utm_easting_m", "utm_northing_m", "utm_zone", "h_m"),
    "enu": ("e_m", "n_m", "u_m"),
}


def _normalize_coord_systems(
    coord_systems: Optional[Sequence[str]],
) -> list[str]:
    """Validate + normalise the requested coordinate systems (order kept,
    duplicates collapsed). ``None`` -> the backward-compatible default."""
    if coord_systems is None:
        return list(DEFAULT_COORD_SYSTEMS)
    out: list[str] = []
    for cs in coord_systems:
        key = str(cs).strip().lower()
        if key not in _COORD_COLUMNS:
            raise ValueError(
                f"export_trajectory: unknown coord system {cs!r}. "
                f"Supported: {', '.join(SUPPORTED_COORD_SYSTEMS)}."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError(
            "export_trajectory: coord_systems is empty. Pass None for the "
            f"default ({'+'.join(DEFAULT_COORD_SYSTEMS)}) or at least one of "
            f"{', '.join(SUPPORTED_COORD_SYSTEMS)}."
        )
    return out


def _utm_zone_from_lonlat(lon: float, lat: float) -> tuple[int, bool, int]:
    """Auto-pick the Grid zone for a path: (zone 1..60, northern, EPSG).

    Called with the FIRST valid fix of the export (deterministic and safe at
    the antimeridian, unlike an arithmetic mean of longitudes).
    """
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(60, max(1, zone))
    northern = lat >= 0.0
    epsg = (32600 if northern else 32700) + zone
    return zone, northern, epsg


def _utm_transformer(epsg: int):
    """The standard datum lon/lat -> Grid transformer via pyproj (same dep as base_pos.py)."""
    try:
        import pyproj
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "UTM export needs pyproj. Run: pip install pyproj>=3.4.0"
        ) from e
    return pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )


# ---------------------------------------------------------------------------
# Export TIME bases (client chooser, 2026-07-05).
# ---------------------------------------------------------------------------
# ``time_bases`` selects which TIME column(s) the CSV carries, anchored to
# the stream timeline. The selected columns are emitted FIRST (before the
# coordinate blocks), in request order. The default reproduces the
# historical single ``reference time`` column byte-for-byte.
#   reference time  : reference time   (Reference time seconds = utc_s + leap; historical default)
#   utc   : utc_s     (absolute UTC unix seconds — the true global clock
#                      the stream rides)
#   stream : t_audio_s (utc_s - audio_start_utc_s; seconds from the stream
#                      sample-0 origin — the "synced to stream" column.
#                      Needs ``audio_start_utc_s``.)
#   iso   : utc_iso   (ISO-8601 UTC string YYYY-MM-DDThh:mm:ss.sssZ)
DEFAULT_TIME_BASES: tuple[str, ...] = ("gpst",)
SUPPORTED_TIME_BASES: tuple[str, ...] = ("gpst", "utc", "audio", "iso")

_TIME_COLUMNS: dict[str, str] = {
    "gpst": "gpstime",
    "utc": "utc_s",
    "audio": "t_audio_s",
    "iso": "utc_iso",
}


def _normalize_time_bases(
    time_bases: Optional[Sequence[str]],
) -> list[str]:
    """Validate + normalise the requested time bases (order kept,
    duplicates collapsed). ``None`` -> the backward-compatible default."""
    if time_bases is None:
        return list(DEFAULT_TIME_BASES)
    out: list[str] = []
    for tb in time_bases:
        key = str(tb).strip().lower()
        if key not in _TIME_COLUMNS:
            raise ValueError(
                f"export_trajectory: unknown time basis {tb!r}. "
                f"Supported: {', '.join(SUPPORTED_TIME_BASES)}."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError(
            "export_trajectory: time_bases is empty. Pass None for the "
            f"default ({'+'.join(DEFAULT_TIME_BASES)}) or at least one of "
            f"{', '.join(SUPPORTED_TIME_BASES)}."
        )
    return out


def _iso_utc(utc_s: float) -> str:
    """ISO-8601 UTC string ``YYYY-MM-DDThh:mm:ss.sssZ`` (millisecond,
    half-up rounded) for a POSIX timestamp; empty string when not finite."""
    if utc_s is None or not math.isfinite(utc_s):
        return ""
    ms_total = math.floor(utc_s * 1000.0 + 0.5)  # half-up at ms precision
    secs, ms = divmod(ms_total, 1000)
    t = _dt.datetime.fromtimestamp(secs, tz=_dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


# ---------------------------------------------------------------------------
# Z (height) smoothing (client request, 2026-07-05; DEFAULT ON).
# ---------------------------------------------------------------------------
def smooth_heights_time_gaussian(
    ts: Sequence[float],
    hs: Sequence[float],
    sigma_s: float,
    gap_break_s: Optional[float] = None,
) -> list[float]:
    """Time-weighted gaussian smoothing of a height series.

    Mirrors the gaussian kernel of ``georef._smooth_trajectory`` but weights
    by the *actual* epoch time deltas, so non-uniform sampling is handled
    exactly (no samples-per-second assumption). Two extra guards:

    * segments split at time gaps > ``gap_break_s`` (default
      ``max(5 * median_dt, 2 * sigma_s)``) are smoothed independently, so
      smoothing never bridges a data hole;
    * non-finite heights are excluded from every kernel and passed through
      unchanged.

    The kernel is truncated at 4 sigma. Series shorter than 3 samples (or a
    non-positive sigma) are returned unchanged.
    """
    n = len(ts)
    if n != len(hs):
        raise ValueError("smooth_heights_time_gaussian: ts/hs length mismatch")
    if n < 3 or not math.isfinite(sigma_s) or sigma_s <= 0.0:
        return list(hs)

    dts = [ts[i + 1] - ts[i] for i in range(n - 1) if ts[i + 1] > ts[i]]
    if not dts:
        return list(hs)
    median_dt = sorted(dts)[len(dts) // 2]
    if gap_break_s is None:
        gap_break_s = max(5.0 * median_dt, 2.0 * sigma_s)

    # Split into contiguous segments at gaps.
    segments: list[tuple[int, int]] = []
    seg_start = 0
    for i in range(1, n):
        if (ts[i] - ts[i - 1]) > gap_break_s:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, n))

    out = list(hs)
    half_window_s = 4.0 * sigma_s
    for a, b in segments:
        if b - a < 3:
            continue  # too short to smooth meaningfully
        lo = a
        hi = a
        for i in range(a, b):
            if not math.isfinite(hs[i]):
                continue  # keep NaN epochs NaN
            while lo < i and (ts[i] - ts[lo]) > half_window_s:
                lo += 1
            if hi < i + 1:
                hi = i + 1
            while hi < b and (ts[hi] - ts[i]) <= half_window_s:
                hi += 1
            wsum = 0.0
            hsum = 0.0
            for j in range(lo, hi):
                hj = hs[j]
                if not math.isfinite(hj):
                    continue
                u = (ts[j] - ts[i]) / sigma_s
                w = math.exp(-0.5 * u * u)
                wsum += w
                hsum += w * hj
            if wsum > 0.0:
                out[i] = hsum / wsum
    return out


def _apply_z_smoothing(rows: list[PosRow], z_sigma_s: float) -> list[PosRow]:
    """Return rows with ``h_m`` gaussian-smoothed over the time axis.

    The smoothed height feeds the datum-based ``h_m`` AND every derived sample
    (Cartesian XYZ / Grid / Local-frame), so all exported coordinate systems agree.
    """
    if len(rows) < 3 or not math.isfinite(z_sigma_s) or z_sigma_s <= 0.0:
        return rows
    ts = [r.utc_s for r in rows]
    hs = [r.h_m for r in rows]
    smoothed = smooth_heights_time_gaussian(ts, hs, z_sigma_s)
    return [replace(r, h_m=h) for r, h in zip(rows, smoothed)]


# --- Accuracy bar (project-wide, see overnight spec) ---------------------
# horizontal <= 6 m @ 2 sigma  AND  speed <= 3 km/h @ 2 sigma.
# The 1-sigma fields (std_xy_smart, std_v*) are doubled to get 2-sigma.
HORIZ_BAR_2SIGMA_M = 6.0
SPEED_BAR_2SIGMA_KMH = 3.0
SPEED_BAR_2SIGMA_MPS = SPEED_BAR_2SIGMA_KMH / 3.6  # 0.8333 m/s


@dataclass
class DroppedSection:
    """A contiguous run of epochs either suppressed from the export
    (``reason="no_sigma"`` — position cannot be certified at all) or kept but
    flagged over the accuracy bar (``reason="horizontal"`` — see
    ``UserExportResult.flagged_sections``). Times are UTC POSIX seconds
    (start/end inclusive).
    """
    start_utc_s: float
    end_utc_s: float
    n_epochs: int
    reason: str  # "horizontal" | "no_sigma"
    worst_h_2sigma_m: float = float("nan")
    worst_speed_2sigma_kmh: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "start_utc_s": round(self.start_utc_s, 3),
            "end_utc_s": round(self.end_utc_s, 3),
            "duration_s": round(self.end_utc_s - self.start_utc_s, 3),
            "n_epochs": self.n_epochs,
            "reason": self.reason,
            "worst_h_2sigma_m": (
                round(self.worst_h_2sigma_m, 3)
                if math.isfinite(self.worst_h_2sigma_m) else None
            ),
            "worst_speed_2sigma_kmh": (
                round(self.worst_speed_2sigma_kmh, 3)
                if math.isfinite(self.worst_speed_2sigma_kmh) else None
            ),
        }


@dataclass
class UserExportResult:
    csv_path: Path
    n_rows: int
    inflation: float
    smart_std_m: float = 0.0
    trust_class: str = "tight"
    # Accuracy gating (client-ready 2026-07-02 semantics):
    n_input_rows: int = 0          # epochs considered before suppression
    n_dropped_rows: int = 0        # epochs removed (no certifiable position sigma)
    dropped_sections: list = None  # list[DroppedSection] (dropped epochs only)
    coverage_pct: float = 100.0    # exported epochs / input epochs * 100
    # Over-bar epochs are KEPT + flagged (pos_within_bar=0), never deleted:
    n_flagged_over_bar: int = 0    # kept epochs whose honest 2-sigma > bar
    flagged_sections: list = None  # list[DroppedSection] (kept, reason="horizontal")
    # Velocity trust (position row is never dropped over velocity):
    n_vel_untrusted: int = 0       # kept epochs with missing/over-bar speed sigma
    # Robust filter (PP3/PP4/PP5):
    n_filter_repaired: int = 0     # impossible epochs interpolated by robust_filter
    n_filter_dropped: int = 0      # impossible epochs hard-dropped by robust_filter
    # Z (height) smoothing (intentional product default, 2026-07-05):
    z_smoothed: bool = False       # True when h_m (+ derived coords) was smoothed
    z_sigma_s_used: float = float("nan")  # gaussian sigma (s) actually applied
    # TIME-basis chooser (2026-07-05):
    time_bases: tuple = DEFAULT_TIME_BASES  # normalised bases actually emitted
    audio_start_utc_s: Optional[float] = None  # UTC of stream sample 0 (stream basis)

    def summary_text(self) -> str:
        """One-paragraph user-facing coverage + honesty summary."""
        secs = self.dropped_sections or []
        flagged = self.flagged_sections or []
        lines = [
            "Accuracy export summary",
            f"  bar: horizontal <= {HORIZ_BAR_2SIGMA_M:.0f} m @ 2 sigma "
            f"AND speed <= {SPEED_BAR_2SIGMA_KMH:.0f} km/h @ 2 sigma",
            f"  input epochs : {self.n_input_rows}",
            f"  exported     : {self.n_rows} ({self.coverage_pct:.1f}% coverage)",
            f"  dropped      : {self.n_dropped_rows} epoch(s) in "
            f"{len(secs)} section(s) (position sigma missing - cannot certify)",
            f"  over-bar     : {self.n_flagged_over_bar} epoch(s) in "
            f"{len(flagged)} section(s) KEPT + flagged pos_within_bar=0 "
            f"(honest 2-sigma exceeds the bar)",
            f"  vel untrusted: {self.n_vel_untrusted} epoch(s) "
            f"(velocity sigma missing or over speed bar; vel_trusted=0, row kept)",
        ]
        if tuple(self.time_bases) != DEFAULT_TIME_BASES:
            note = f"  time basis   : {'+'.join(self.time_bases)}"
            if self.audio_start_utc_s is not None:
                note += f" (audio_start_utc_s={self.audio_start_utc_s:.6f})"
            lines.append(note)
        if self.z_smoothed:
            lines.append(
                f"  note: heights gaussian-smoothed (sigma="
                f"{self.z_sigma_s_used:.1f}s; h_m + derived ECEF/UTM/ENU; "
                f"intentional default -- smooth_z=False for raw heights)"
            )
        for i, s in enumerate(secs, 1):
            d = s.to_dict()
            lines.append(
                f"    dropped [{i}] {d['duration_s']:.1f}s "
                f"({d['n_epochs']} epochs) reason={d['reason']}"
            )
        for i, s in enumerate(flagged, 1):
            d = s.to_dict()
            lines.append(
                f"    over-bar [{i}] {d['duration_s']:.1f}s "
                f"({d['n_epochs']} epochs) worst_h2s={d['worst_h_2sigma_m']}m"
            )
        return "\n".join(lines)


def export_trajectory(
    rows: list[PosRow],
    out_csv: Path,
    *,
    source_tag: str = "smoothed",
    inflation: Optional[float] = None,
    raw_rows: Optional[list[PosRow]] = None,
    suppress_inaccurate: bool = True,
    horiz_bar_2sigma_m: float = HORIZ_BAR_2SIGMA_M,
    speed_bar_2sigma_kmh: float = SPEED_BAR_2SIGMA_KMH,
    robust_filter_enabled: bool = True,
    filter_preset: Optional[RobustFilterConfig] = None,
    hard_drop_over_bar: bool = False,
    coord_systems: Optional[Sequence[str]] = None,
    smooth_z: bool = True,
    z_sigma_s: float = DEFAULT_Z_SIGMA_S,
    time_bases: Sequence[str] = DEFAULT_TIME_BASES,
    audio_start_utc_s: Optional[float] = None,
) -> UserExportResult:
    """Write the user-facing path CSV.

    Suppression semantics (client-ready, 2026-07-02)
    ------------------------------------------------
    Position validity is judged ONLY on the horizontal sigma:

    * an epoch is DROPPED only when its horizontal 2-sigma cannot be
      computed at all (``no_sigma`` -- nothing certifiable to report);
    * an epoch whose honest horizontal 2-sigma exceeds the bar is KEPT and
      flagged ``pos_within_bar=0`` (the client gets the full path with
      an honest error bar, instead of losing the epoch). Set
      ``hard_drop_over_bar=True`` to restore the legacy hard-drop behaviour;
    * velocity NEVER drops a row. A missing velocity sigma (common when the
      smoother path does not propagate ``sd_v*``) or a speed 2-sigma over the
      bar marks the row ``vel_trusted=0`` ("velocity untrusted") -- the
      position, whose sigma bar is independent, still ships.

    This fixes the over-suppression that previously deleted ~2/3 of a valid
    path whose honest (inflated) sigma source marginally above the 6 m bar.

    ``rows`` should be the smoother's output (epoch_weighted /
    fused_bent / FGO / raw Post-processing / hybrid). Each :class:`PosRow` must
    carry per-epoch sd_n/sd_e/sd_u and (when available) sd_vn/sd_ve/sd_vu
    + ve/vn/vu. NaN fields render as empty CSV cells; std_xy falls back
    to NaN if The external solver sigmas are missing.

    Robust filter (PP3/PP4/PP5, default ON)
    ---------------------------------------
    When ``robust_filter_enabled`` is True (the shipped default for the "best"
    export profile), the GT-free physical-plausibility :func:`robust_filter` runs
    *before* sigma prediction and suppression, using ``filter_preset`` (defaults
    to :func:`winning_export_filter`). It repairs short impossible runs and drops
    long ones, crushing the divergence-spike MAX error. It is a strict no-op on
    clean data (only physically-impossible epochs trip the gates), so leaving it
    on never regresses a clean route. Every repaired/dropped epoch is reflected
    in the new ``gap`` CSV column (1 at a repaired/dropped boundary) so a
    downstream consumer can never silently bridge a hole (fixes PP5). Set
    ``robust_filter_enabled=False`` to restore the pre-filter behaviour exactly.

    The Motion sensor-calibrated *fusion* path is intentionally NOT part of this default
    (it regressed MAX on spiked sessions in P3); it stays opt-in upstream.

    Coordinate systems (client chooser, 2026-07-05)
    -----------------------------------------------
    ``coord_systems`` is an ordered selection from 'datum-based', 'cartesian XYZ', 'grid',
    'local-frame' (see :data:`SUPPORTED_COORD_SYSTEMS`). ``None`` (the default) emits
    exactly the historical column set (datum-based + Cartesian XYZ, unchanged order).
    The selected blocks are emitted right after ``reference time`` in request order;
    the metadata columns (velocities, sigmas, trust, flags) are untouched.
    Grid auto-picks the zone from the FIRST valid fix's lon/lat (deterministic,
    antimeridian-safe; The standard datum / EPSG:326xx-327xx, logged in the ``utm_zone``
    column and a leading ``#`` header comment).
    Local-frame is anchored at the first valid fix of the exported path.

    Time bases (client chooser, 2026-07-05)
    ---------------------------------------
    ``time_bases`` is an ordered selection from 'reference time', 'utc', 'stream', 'iso'
    (see :data:`SUPPORTED_TIME_BASES`); the default ``("reference time",)`` emits
    exactly the historical single ``reference time`` column. The selected TIME
    columns are emitted FIRST (before the coordinate blocks), in request
    order (duplicates collapsed):

    * reference time  -> ``reference time``   = utc_s + leap_seconds(utc_s)  [Reference time seconds]
    * utc   -> ``utc_s``     = absolute UTC unix seconds
    * stream -> ``t_audio_s`` = utc_s - ``audio_start_utc_s`` (seconds from
      the stream sample-0 origin; requires ``audio_start_utc_s`` — the UTC
      time of stream sample 0, derivable from the session's stream anchor via
      ``audio_frame_export.resolve_session_anchors``)
    * iso   -> ``utc_iso``   = ISO-8601 UTC string YYYY-MM-DDThh:mm:ss.sssZ

    Z (height) smoothing (client request, 2026-07-05; DEFAULT ON)
    -------------------------------------------------------------
    ``smooth_z=True`` (default) gaussian-smooths ``h_m`` over the time axis
    with ``z_sigma_s`` seconds (default 3.0), time-weighted so non-uniform
    epoch spacing and gaps are handled (segments split at large gaps; see
    :func:`smooth_heights_time_gaussian`). Runs after the robust filter and
    before sigma/trust computation; the smoothed height feeds the datum-based
    ``h_m`` and every derived sample (Cartesian XYZ/Grid/Local-frame) consistently. Horizontal
    coordinates are never touched.

    NOTE: this default is intentionally NOT backward-compatible -- the client
    explicitly asked for smoothed Z as the shipped product default. Every
    existing caller therefore gets a smoothed ``h_m`` and correspondingly
    changed height-derived outputs (Cartesian XYZ x/y/z, Grid h_m, Local-frame u_m) and
    downstream trust/sigma inputs, compared with pre-2026-07-05 exports.
    Pass ``smooth_z=False`` to disable and export raw heights.

    Returns the row count + inflation factor used.
    """
    if not rows:
        raise ValueError(
            "export_trajectory: empty rows list. Run a smoother first "
            "and pass its PosRow list."
        )

    # --- TIME-basis chooser (validated up-front so a bad request fails fast).
    bases = _normalize_time_bases(time_bases)
    if "audio" in bases and audio_start_utc_s is None:
        raise ValueError(
            "audio-relative time requested but no audio anchor; the "
            "session's audio_anchor is required."
        )
    time_cols = [_TIME_COLUMNS[b] for b in bases]

    # --- PP3/PP4/PP5: GT-free robust filter before sigma + suppression. ---
    # ``gap_by_t`` maps a surviving epoch's rounded utc_s to its gap-edge flag so
    # the CSV ``gap`` column marks repaired/dropped boundaries (PP5).
    gap_by_t: dict[float, bool] = {}
    n_filter_repaired = 0
    n_filter_dropped = 0
    if robust_filter_enabled:
        cfg = filter_preset if filter_preset is not None else winning_export_filter()
        if cfg.enabled:
            fr = robust_filter(rows, cfg)
            rows = fr.rows
            n_filter_repaired = fr.n_repaired
            n_filter_dropped = fr.n_dropped
            # verdicts are indexed against the *input* rows; the surviving epochs
            # keep their utc_s, so key the gap flag by rounded time. This
            # covers REPAIRED runs (the repaired rows survive with the same
            # utc_s, and the filter flags the run's boundary epochs).
            for v in fr.verdicts:
                if getattr(v, "gap", False):
                    gap_by_t[round(v.utc_s, 3)] = True
            # HARD-DROPPED runs are different: the filter sets gap=True only
            # on the dropped epochs themselves, whose utc_s never appear in
            # the surviving rows -- so the flags above can never match a
            # written row and the client would silently bridge the hole
            # (PP5 violation). Make the hole visible by flagging the
            # surviving neighbours immediately BEFORE and AFTER each dropped
            # run instead. (Drop policy unchanged; visibility only.)
            verdicts = fr.verdicts
            nv = len(verdicts)
            iv = 0
            while iv < nv:
                if verdicts[iv].outcome != _FILTER_DROP:
                    iv += 1
                    continue
                jv = iv
                while jv < nv and verdicts[jv].outcome == _FILTER_DROP:
                    jv += 1
                if iv - 1 >= 0:
                    gap_by_t[round(verdicts[iv - 1].utc_s, 3)] = True
                if jv < nv:
                    gap_by_t[round(verdicts[jv].utc_s, 3)] = True
                iv = jv
            if not rows:
                raise ValueError(
                    "export_trajectory: robust_filter removed all rows "
                    "(trajectory was entirely physically implausible). "
                    "Pass robust_filter_enabled=False to bypass."
                )

    # --- Z (height) smoothing: after the robust filter, before sigma/trust.
    # The smoothed h_m feeds datum-based AND all derived samples (Cartesian XYZ/Grid/Local-frame).
    # DEFAULT ON by explicit client product decision (see docstring).
    z_smoothed_applied = False
    if smooth_z:
        smoothed_rows = _apply_z_smoothing(rows, z_sigma_s)
        z_smoothed_applied = smoothed_rows is not rows
        rows = smoothed_rows

    # --- Coordinate-system chooser setup. ---
    systems = _normalize_coord_systems(coord_systems)
    coord_cols: list[str] = []
    for cs in systems:
        for col_name in _COORD_COLUMNS[cs]:
            if col_name not in coord_cols:
                coord_cols.append(col_name)

    utm_zone_str = ""
    utm_epsg: Optional[int] = None
    utm_xform = None
    if "utm" in systems:
        # Zone from the FIRST valid fix (deterministic + antimeridian-safe).
        # A plain arithmetic mean of longitudes breaks on a track straddling
        # +/-180 deg (e.g. 179.9 and -179.9 average to ~0 -> zone 31, putting
        # coordinates hundreds of km off); the first fix cannot.
        first_fix: Optional[tuple[float, float]] = None
        for r in rows:
            if math.isfinite(r.lat_deg) and math.isfinite(r.lon_deg):
                first_fix = (r.lat_deg, r.lon_deg)
                break
        if first_fix is None:
            raise ValueError(
                "export_trajectory: 'utm' requested but no epoch has a "
                "finite lat/lon to pick a UTM zone from."
            )
        zone, northern, utm_epsg = _utm_zone_from_lonlat(
            first_fix[1], first_fix[0]
        )
        utm_zone_str = f"{zone}{'N' if northern else 'S'}"
        utm_xform = _utm_transformer(utm_epsg)

    enu_origin: Optional[tuple[float, float, float]] = None
    if "enu" in systems:
        for r in rows:
            if (math.isfinite(r.lat_deg) and math.isfinite(r.lon_deg)
                    and math.isfinite(r.h_m)):
                enu_origin = (r.lat_deg, r.lon_deg, r.h_m)
                break
        if enu_origin is None:
            raise ValueError(
                "export_trajectory: 'enu' requested but no epoch has a "
                "finite lat/lon/h to anchor the ENU origin."
            )

    inflation = inflation if inflation is not None else calibrate_sigma_inflation(rows)
    profile = smart_session_std(rows)
    smart_arr = predicted_epoch_std(rows, profile)

    trust_input = raw_rows if raw_rows is not None else rows
    v2 = smooth_epoch_weighted_v2(
        trust_input, imu_rows=None,
        options=EpochWeightV2Options(
            zupt_enabled=True, nhc_enabled=True,
            nhc_heading_source="doppler", sigma_a_base=0.10,
        ),
    )
    trust_v2 = compute_trust_v2(trust_input, v2)
    if raw_rows is not None and len(raw_rows) != len(rows):
        trust_by_t = {}
        for i_t, r_t in enumerate(trust_input):
            trust_by_t[round(r_t.utc_s, 3)] = trust_v2.labels[i_t]
    else:
        trust_by_t = None

    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_csv) + ".tmp")

    cols = [
        *time_cols, *coord_cols,
        "vn_mps", "ve_mps", "vu_mps",
        "speed_mps", "vel_error_pct_speed",
        "std_xy_m", "std_xy_smart_m",
        # --- explicit 2-sigma error columns (overnight Track 4 parity) ---
        "err_horiz_2sigma_m", "err_speed_2sigma_mps", "err_speed_2sigma_kmh",
        "std_vn_mps", "std_ve_mps", "std_vu_mps",
        "trust_class", "source", "trust_label_v2", "gap",
        # --- honesty flags (client-ready 2026-07-02) ---
        # pos_within_bar : 1 when err_horiz_2sigma_m <= the 6 m bar, else 0.
        # vel_trusted    : 1 when err_speed_2sigma is present and <= the speed
        #                  bar; 0 = velocity untrusted (row still valid).
        "pos_within_bar", "vel_trusted",
    ]

    def _fmt(v: float, w: int = 4) -> str:
        if v is None or not math.isfinite(v):
            return ""
        return f"{v:.{w}f}"

    speed_bar_mps = float(speed_bar_2sigma_kmh) / 3.6

    # First pass: compute per-epoch metrics + accept/reject decision, so we
    # can group rejected epochs into contiguous "dropped sections".
    @dataclass
    class _Epoch:
        idx: int
        row: PosRow
        gpst: float
        x: float
        y: float
        z: float
        std_xy: float
        speed: float
        vel_pct: float
        h_2sigma_m: float
        speed_2sigma_mps: float
        accept: bool
        reason: str
        over_bar: bool
        vel_ok: bool
        utm_e: float = float("nan")
        utm_n: float = float("nan")
        enu_e: float = float("nan")
        enu_n: float = float("nan")
        enu_u: float = float("nan")

    epochs: list[_Epoch] = []
    for i, r in enumerate(rows):
        ls = get_leap_seconds_for_epoch(r.utc_s)
        gpst = r.utc_s + ls
        try:
            x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        except Exception:
            x = y = z = float("nan")
        utm_e = utm_n = float("nan")
        if utm_xform is not None and math.isfinite(r.lat_deg) and math.isfinite(r.lon_deg):
            try:
                utm_e, utm_n = utm_xform.transform(r.lon_deg, r.lat_deg)
            except Exception:
                utm_e = utm_n = float("nan")
        enu_e = enu_n = enu_u = float("nan")
        if enu_origin is not None and math.isfinite(x):
            enu_e, enu_n, enu_u = ecef_to_enu(x, y, z, enu_origin)
        if math.isfinite(r.sd_n) and math.isfinite(r.sd_e):
            std_xy = math.hypot(r.sd_n, r.sd_e) * inflation
        else:
            std_xy = float("nan")
        speed = math.hypot(r.vn, r.ve) if (math.isfinite(r.vn) and math.isfinite(r.ve)) else float("nan")
        if math.isfinite(speed) and math.isfinite(r.sd_vn) and math.isfinite(r.sd_ve):
            vel_err_abs = math.hypot(r.sd_vn, r.sd_ve)
            vel_pct = (vel_err_abs / max(speed, 0.01)) * 100.0
        else:
            vel_pct = float("nan")

        # 2-sigma error: prefer the validated smart 1-sigma for horizontal.
        smart_1s = float(smart_arr[i]) if i < len(smart_arr) else float("nan")
        h_1sigma = smart_1s if math.isfinite(smart_1s) else std_xy
        h_2sigma = 2.0 * h_1sigma if math.isfinite(h_1sigma) else float("nan")
        # speed 2-sigma from velocity 1-sigma (horizontal quadrature).
        if math.isfinite(r.sd_vn) and math.isfinite(r.sd_ve):
            speed_2sigma = 2.0 * math.hypot(r.sd_vn, r.sd_ve)
        else:
            speed_2sigma = float("nan")

        # Position validity is judged ONLY on the horizontal sigma bar.
        # * no horizontal sigma at all  -> cannot certify -> drop (honesty).
        # * over the bar                -> KEEP + flag pos_within_bar=0
        #                                  (hard-drop only in legacy mode).
        # * velocity sigma missing/over -> vel_trusted=0, row always kept.
        reasons: list[str] = []
        over_bar = False
        if not math.isfinite(h_2sigma):
            reasons.append("no_sigma")
        elif h_2sigma > horiz_bar_2sigma_m:
            over_bar = True
            if hard_drop_over_bar:
                reasons.append("horizontal")
        vel_ok = math.isfinite(speed_2sigma) and speed_2sigma <= speed_bar_mps
        accept = (not reasons) if suppress_inaccurate else True
        reason = "+".join(reasons) if reasons else ""
        epochs.append(_Epoch(
            idx=i, row=r, gpst=gpst, x=x, y=y, z=z, std_xy=std_xy,
            speed=speed, vel_pct=vel_pct, h_2sigma_m=h_2sigma,
            speed_2sigma_mps=speed_2sigma, accept=accept, reason=reason,
            over_bar=over_bar, vel_ok=vel_ok,
            utm_e=utm_e, utm_n=utm_n,
            enu_e=enu_e, enu_n=enu_n, enu_u=enu_u,
        ))

    # Group rejected epochs into contiguous sections.
    dropped: list[DroppedSection] = []
    run: list[_Epoch] = []

    def _flush_run() -> None:
        if not run:
            return
        reasons = set()
        for e in run:
            reasons.update(e.reason.split("+") if e.reason else [])
        reasons.discard("")
        worst_h = max(
            (e.h_2sigma_m for e in run if math.isfinite(e.h_2sigma_m)),
            default=float("nan"),
        )
        worst_v = max(
            (e.speed_2sigma_mps for e in run if math.isfinite(e.speed_2sigma_mps)),
            default=float("nan"),
        )
        dropped.append(DroppedSection(
            start_utc_s=run[0].row.utc_s,
            end_utc_s=run[-1].row.utc_s,
            n_epochs=len(run),
            reason="+".join(sorted(reasons)) if reasons else "unknown",
            worst_h_2sigma_m=worst_h,
            worst_speed_2sigma_kmh=worst_v * 3.6 if math.isfinite(worst_v) else float("nan"),
        ))

    for e in epochs:
        if e.accept:
            _flush_run()
            run = []
        else:
            run.append(e)
    _flush_run()

    # Group KEPT-but-over-bar epochs into contiguous flagged sections
    # (informational: the epochs ship, with pos_within_bar=0).
    flagged: list[DroppedSection] = []
    frun: list[_Epoch] = []

    def _flush_frun() -> None:
        if not frun:
            return
        worst_h = max(
            (e.h_2sigma_m for e in frun if math.isfinite(e.h_2sigma_m)),
            default=float("nan"),
        )
        flagged.append(DroppedSection(
            start_utc_s=frun[0].row.utc_s,
            end_utc_s=frun[-1].row.utc_s,
            n_epochs=len(frun),
            reason="horizontal",
            worst_h_2sigma_m=worst_h,
        ))

    for e in epochs:
        if e.accept and e.over_bar:
            frun.append(e)
        else:
            _flush_frun()
            frun = []
    _flush_frun()

    # Second pass: write only accepted rows.
    n = 0
    with tmp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if utm_epsg is not None:
            # Record the auto-picked zone/CRS; readers skip '#' lines.
            f.write(f"# utm_zone={utm_zone_str} utm_epsg=EPSG:{utm_epsg}\n")
        wr.writerow(cols)
        for e in epochs:
            if not e.accept:
                continue
            r = e.row
            label = (
                trust_by_t.get(round(r.utc_s, 3), "low")
                if trust_by_t is not None else trust_v2.labels[e.idx]
            )
            speed_2sigma_kmh = (
                e.speed_2sigma_mps * 3.6
                if math.isfinite(e.speed_2sigma_mps) else float("nan")
            )
            coord_vals = {
                "lat_deg": _fmt(r.lat_deg, 9),
                "lon_deg": _fmt(r.lon_deg, 9),
                "h_m": _fmt(r.h_m, 4),
                "x_ecef_m": _fmt(e.x, 4),
                "y_ecef_m": _fmt(e.y, 4),
                "z_ecef_m": _fmt(e.z, 4),
                "utm_easting_m": _fmt(e.utm_e, 4),
                "utm_northing_m": _fmt(e.utm_n, 4),
                "utm_zone": utm_zone_str,
                "e_m": _fmt(e.enu_e, 4),
                "n_m": _fmt(e.enu_n, 4),
                "u_m": _fmt(e.enu_u, 4),
            }
            time_vals: list[str] = []
            for b in bases:
                if b == "gpst":
                    time_vals.append(_fmt(e.gpst, 6))
                elif b == "utc":
                    time_vals.append(_fmt(r.utc_s, 6))
                elif b == "audio":
                    time_vals.append(_fmt(r.utc_s - audio_start_utc_s, 6))
                else:  # iso
                    time_vals.append(_iso_utc(r.utc_s))
            wr.writerow([
                *time_vals,
                *[coord_vals[c] for c in coord_cols],
                _fmt(r.vn, 5), _fmt(r.ve, 5), _fmt(r.vu, 5),
                _fmt(e.speed, 4), _fmt(e.vel_pct, 2),
                _fmt(e.std_xy, 4),
                _fmt(float(smart_arr[e.idx]) if e.idx < len(smart_arr) else float("nan"), 4),
                _fmt(e.h_2sigma_m, 4),
                _fmt(e.speed_2sigma_mps, 5), _fmt(speed_2sigma_kmh, 4),
                _fmt(r.sd_vn, 5), _fmt(r.sd_ve, 5), _fmt(r.sd_vu, 5),
                profile.trust_class, source_tag, label,
                "1" if gap_by_t.get(round(r.utc_s, 3), False) else "0",
                "0" if e.over_bar else "1",
                "1" if e.vel_ok else "0",
            ])
            n += 1
    os.replace(tmp, out_csv)

    n_input = len(rows)
    n_dropped = sum(s.n_epochs for s in dropped)
    coverage = (100.0 * n / n_input) if n_input else 100.0
    n_flagged = sum(1 for e in epochs if e.accept and e.over_bar)
    n_vel_untrusted = sum(1 for e in epochs if e.accept and not e.vel_ok)
    return UserExportResult(
        csv_path=out_csv, n_rows=n, inflation=inflation,
        smart_std_m=profile.smart_std_m, trust_class=profile.trust_class,
        n_input_rows=n_input, n_dropped_rows=n_dropped,
        dropped_sections=dropped, coverage_pct=coverage,
        n_flagged_over_bar=n_flagged, flagged_sections=flagged,
        n_vel_untrusted=n_vel_untrusted,
        n_filter_repaired=n_filter_repaired, n_filter_dropped=n_filter_dropped,
        z_smoothed=z_smoothed_applied,
        z_sigma_s_used=z_sigma_s if z_smoothed_applied else float("nan"),
        time_bases=tuple(bases),
        audio_start_utc_s=audio_start_utc_s if "audio" in bases else None,
    )


def export_kml(
    rows: list[PosRow],
    out_kml: Path,
    *,
    name: str = "trajectory",
    color_by_trust: bool = False,
    trust_arr: Optional[list[float]] = None,
    smooth_z: bool = True,
    z_sigma_s: float = DEFAULT_Z_SIGMA_S,
) -> Path:
    """Lightweight Export format writer for user-export. Defers heavy multi-style
    output to :mod:`data_pipeline.stages.kml_export` for the batch path.

    When ``color_by_trust`` is True and ``trust_arr`` is provided, the
    path is broken into segments coloured green (trust=1) to
    blue (trust=0).

    ``smooth_z`` / ``z_sigma_s`` mirror :func:`export_trajectory` (DEFAULT ON,
    same time-weighted gaussian) so the Export format altitude stays consistent with
    the CSV's smoothed ``h_m``. Pass ``smooth_z=False`` for raw heights.
    """
    if not rows:
        raise ValueError("export_kml: empty rows list.")

    if smooth_z:
        rows = _apply_z_smoothing(rows, z_sigma_s)

    out_kml = Path(out_kml).resolve()
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_kml) + ".tmp")

    def _color(trust: float) -> str:
        # Export format uses ABGR. Green=ff00ff00, blue=ffff0000. Interpolate.
        t = max(0.0, min(1.0, float(trust)))
        g = int(round(255 * t))
        b = int(round(255 * (1 - t)))
        return f"ff{b:02x}{g:02x}00"

    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        f'<name>{name}</name>\n'
    )
    tail = '</Document></kml>\n'

    with tmp.open("w", encoding="utf-8") as f:
        f.write(head)
        if color_by_trust and trust_arr and len(trust_arr) == len(rows):
            # Per-segment colour. Each placemark has two endpoints.
            for i in range(1, len(rows)):
                a, b = rows[i - 1], rows[i]
                if not (math.isfinite(a.lat_deg) and math.isfinite(b.lat_deg)):
                    continue
                col = _color(0.5 * (trust_arr[i - 1] + trust_arr[i]))
                f.write(
                    f'<Placemark><Style><LineStyle><color>{col}</color>'
                    '<width>3</width></LineStyle></Style><LineString>'
                    '<altitudeMode>absolute</altitudeMode><coordinates>'
                    f'{a.lon_deg},{a.lat_deg},{a.h_m} '
                    f'{b.lon_deg},{b.lat_deg},{b.h_m}'
                    '</coordinates></LineString></Placemark>\n'
                )
        else:
            # Single track placemark.
            coords = " ".join(
                f"{r.lon_deg},{r.lat_deg},{r.h_m}" for r in rows
                if math.isfinite(r.lat_deg) and math.isfinite(r.lon_deg)
            )
            f.write(
                '<Placemark><name>track</name><Style><LineStyle>'
                '<color>ff00ffff</color><width>3</width></LineStyle></Style>'
                '<LineString><altitudeMode>absolute</altitudeMode>'
                f'<coordinates>{coords}</coordinates></LineString></Placemark>\n'
            )
        f.write(tail)
    os.replace(tmp, out_kml)
    return out_kml
