"""Parsers for the file formats the pipeline ingests.

We avoid loading entire files into memory beyond what's needed and we always
sort outputs by time so downstream code can binary-search.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from .time_sync import GPS_UTC_LEAP_SECONDS_2026, get_leap_seconds_for_datetime

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PosRow:
    """One row of a solver ``.pos`` file, normalised to UTC unix seconds."""

    utc_s: float
    lat_deg: float
    lon_deg: float
    h_m: float
    quality: int
    # vn / ve / vu in m/s, NaN when the .pos lacks velocity columns.
    vn: float = float("nan")
    ve: float = float("nan")
    vu: float = float("nan")
    # ``ns`` = number of sources used in the solution (The external solver col 7).
    # 0 when missing or malformed.
    ns: int = 0
    # Per-epoch 1-sigma uncertainties from The external solver output (cols 7-9 in the
    # standard subject-position .pos format). NaN when columns are absent.
    # Use these directly as the diagonal R for an adaptive Recursive-filter update.
    sd_n: float = float("nan")  # m, north  1-sigma
    sd_e: float = float("nan")  # m, east   1-sigma
    sd_u: float = float("nan")  # m, up     1-sigma

    # Off-diagonal position covariance (The external solver cols 10-12): sigma_ne, sigma_eu,
    # sigma_un. Sign-preserved (i.e. negative covariance is stored as negative).
    # Use with sd_n/sd_e/sd_u to build a full 3x3 R matrix for the position
    # update. NaN when absent.
    sd_ne: float = float("nan")
    sd_eu: float = float("nan")
    sd_un: float = float("nan")

    # The external solver AR validation: ``ratio`` is the second-best / best residual
    # ratio test. ratio > arthres (typically 3.0) triggers Q=1 fix. Higher
    # ratio means more confident fix; weight Q=2 epochs by ratio for
    # graceful trust scaling. NaN when absent. ``age`` is the differential
    # correction age in seconds (col 13).
    age_s: float = float("nan")
    ratio: float = float("nan")

    # Rate-signal velocity per-axis 1-sigma (The external solver cols 18-20). NaN when absent.
    sd_vn: float = float("nan")
    sd_ve: float = float("nan")
    sd_vu: float = float("nan")

    # Off-diagonal velocity covariance (The external solver cols 21-23).
    sd_vne: float = float("nan")
    sd_veu: float = float("nan")
    sd_vun: float = float("nan")


@dataclass(frozen=True)
class DataFix:
    """One ``Fix,...`` line from the source app.

    ``speed_mps`` / ``bearing_deg`` come from the source app's CSV columns 6 / 8.
    The FLP-blended provider reports them as Motion sensor-fused estimates; raw Reference
    rows often have only the position fields populated. NaN when absent.
    """

    utc_s: float
    provider: str
    lat: float
    lon: float
    h: float
    h_acc: float
    v_acc: float
    speed_mps: float = float("nan")
    bearing_deg: float = float("nan")


@dataclass(frozen=True)
class Orient:
    """One ``OrientationDeg,...`` line from the source app.

    ``yaw`` is in [0, 360); pitch / roll are in (-180, 180].
    """

    utc_s: float
    yaw: float
    roll: float
    pitch: float
    cal: int


# Column-header labels The external solver writes per out-solformat. The column-header
# line is the ``%`` line naming the coordinate columns, e.g.
# ``%  GPST   latitude(deg) longitude(deg)  height(m) ...`` (llh) or
# ``%  GPST   x-ecef(m) y-ecef(m) z-ecef(m) ...`` (xyz) or
# ``%  GPST   e-baseline(m) n-baseline(m) u-baseline(m) ...`` (enu).
_POS_COLUMN_LABELS: tuple[tuple[str, str], ...] = (
    ("latitude(deg)", "llh"),
    ("x-ecef(m)", "ecef"),
    ("e-baseline", "enu"),
    ("baseline(m)", "enu"),
    ("e-enu", "enu"),
)


def _pos_column_header_line(path: Path) -> str | None:
    """Return the coordinate column-header ``%`` line of a ``.pos`` file.

    That is the ``%`` line carrying one of the known coordinate column
    labels (``llh`` / ``ecef`` / ``enu`` solformats). When none matches, fall back to
    the LAST ``%`` line before the first data row (The external solver always writes the
    column header as the final comment line), so an unrecognised format can
    at least be named in a warning. ``None`` when the file has no ``%``
    header at all or cannot be read.
    """
    last_pct: str | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("%"):
                    break  # first data row → header section over
                for label, _fmt in _POS_COLUMN_LABELS:
                    if label in line:
                        return line
                last_pct = line
    except OSError:
        return None
    return last_pct


def detect_pos_solformat(path: Path) -> str:
    """Detect the coordinate solformat of a solver ``.pos`` file.

    Reads the column-header line (the ``%`` line naming the coordinate
    columns) and returns:

    * ``"llh"``  — ``latitude(deg) longitude(deg) height(m)`` (The external solver
      ``out-solformat=llh``, the pipeline's expected format);
    * ``"ecef"`` — ``x-ecef(m) y-ecef(m) z-ecef(m)`` (``out-solformat=xyz``);
    * ``"enu"``  — ``e-baseline(m) n-baseline(m) u-baseline(m)``
      (``out-solformat=enu``, a relative baseline — no absolute position);
    * ``"unknown"`` — no recognisable coordinate column label found
      (headerless / truncated / third-party files).
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("%"):
                    break  # reached data without a recognisable column header
                for label, fmt in _POS_COLUMN_LABELS:
                    if label in line:
                        return fmt
    except OSError:
        pass
    return "unknown"


def _detect_pos_time_system_ex(path: Path) -> tuple[str, bool]:
    """Return ``(time_system, token_found)`` for a ``.pos`` file.

    ``time_system`` is ``"GPST"`` or ``"UTC"``; ``token_found`` is True only
    when an explicit ``GPST`` / ``UTC`` token was actually present as the
    first token of the coordinate column-header line. When it is False the
    returned ``"GPST"`` is an *assumption* (the historical solver default),
    and callers may want to warn: if the file is really UTC, downstream
    media sync will be off by the epoch offset count (~18 s).
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line.startswith("%"):
                    break  # reached data before any column header → assume default
                if not any(label in line for label, _fmt in _POS_COLUMN_LABELS):
                    continue
                toks = line.lstrip("%").split()
                if toks:
                    first = toks[0].upper()
                    if first == "UTC":
                        return "UTC", True
                    if first == "GPST":
                        return "GPST", True
                return "GPST", False  # header found, token unrecognised
    except OSError:
        pass
    return "GPST", False


def _detect_pos_time_system(path: Path) -> str:
    """Return the time system of a ``.pos`` file: ``"GPST"`` or ``"UTC"``.

    The external solver writes the time system as the FIRST token of the
    column-label header line, e.g.::

        %  GPST                  latitude(deg) longitude(deg) ...
        %  UTC                   latitude(deg) longitude(deg) ...

    (controlled by the ``out-timesys`` / time-format option). The data rows
    that follow are in whatever system that token names. The default solver
    output is ``GPST``, so we only deviate from "subtract the epoch offset" when
    the header *explicitly* says UTC — a UTC-labelled file is already UTC, and
    subtracting the offset again would shift every epoch by ~18 s (≈180 m
    of along-track error at highway speed). When no recognisable header line
    is present we fall back to ``GPST`` (the historical solver-default
    assumption), so existing files are unaffected.
    """
    return _detect_pos_time_system_ex(path)[0]


@dataclass(frozen=True)
class PosHeader:
    """Config readout parsed from a solver ``.pos`` header (``outhead`` on).

    Every field is ``None`` when the corresponding ``%`` line is absent —
    old / minimal / third-party headers must never raise. ``time_system``
    and ``solformat`` always carry a value (defaults ``"GPST"`` / whatever
    :func:`detect_pos_solformat` reports) because downstream code always
    needs a decision for those two.
    """

    program: str | None = None
    pos_mode: str | None = None
    freqs: str | None = None
    elev_mask_deg: float | None = None
    amb_res: str | None = None
    nav_sys: str | None = None
    val_thres: float | None = None
    ref_pos: tuple[float, ...] | None = None
    obs_start: str | None = None
    obs_end: str | None = None
    time_system: str = "GPST"
    solformat: str = "unknown"

    def summary_lines(self) -> list[str]:
        """Human-readable summary for logging, one ``key = value`` per line."""
        def _fmt(v: object) -> str:
            if v is None:
                return "(not in header)"
            if isinstance(v, tuple):
                return " ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in v)
            return str(v)

        return [
            f"[pos] program    = {_fmt(self.program)}",
            f"[pos] pos mode   = {_fmt(self.pos_mode)}",
            f"[pos] freqs      = {_fmt(self.freqs)}",
            f"[pos] elev mask  = {_fmt(self.elev_mask_deg)}"
            + (" deg" if self.elev_mask_deg is not None else ""),
            f"[pos] amb res    = {_fmt(self.amb_res)}",
            f"[pos] navi sys   = {_fmt(self.nav_sys)}",
            f"[pos] val thres  = {_fmt(self.val_thres)}",
            f"[pos] ref pos    = {_fmt(self.ref_pos)}",
            f"[pos] obs start  = {_fmt(self.obs_start)}",
            f"[pos] obs end    = {_fmt(self.obs_end)}",
            f"[pos] time sys   = {self.time_system}",
            f"[pos] solformat  = {self.solformat}",
        ]


def parse_pos_header(path: Path) -> PosHeader:
    """Parse the ``% key : value`` header of a solver ``.pos`` file.

    Answers "what The external solver config produced this .pos" when The external solver wrote the
    file with ``outhead`` on. Tolerant of ANY missing line — a header-less
    file simply yields a :class:`PosHeader` full of ``None``.
    """
    kv: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("%"):
                    break  # data rows begin — header over
                body = line.lstrip("%").strip()
                if ":" not in body:
                    continue
                key, _, value = body.partition(":")
                key = key.strip().lower()
                value = value.strip()
                # First occurrence wins (e.g. many "inp file" lines).
                if key and key not in kv:
                    kv[key] = value
    except OSError:
        pass

    def _float_first(key: str) -> float | None:
        v = kv.get(key)
        if not v:
            return None
        try:
            return float(v.split()[0])
        except (ValueError, IndexError):
            return None

    ref_pos: tuple[float, ...] | None = None
    v = kv.get("ref pos")
    if v:
        try:
            vals = tuple(float(t) for t in v.split())
            ref_pos = vals if vals else None
        except ValueError:
            ref_pos = None

    def _str(key: str) -> str | None:
        v = kv.get(key)
        return v if v else None

    time_system, _found = _detect_pos_time_system_ex(path)
    return PosHeader(
        program=_str("program"),
        pos_mode=_str("pos mode"),
        freqs=_str("freqs"),
        elev_mask_deg=_float_first("elev mask"),
        amb_res=_str("amb res"),
        nav_sys=_str("navi sys"),
        val_thres=_float_first("val thres"),
        ref_pos=ref_pos,
        obs_start=_str("obs start"),
        obs_end=_str("obs end"),
        time_system=time_system,
        solformat=detect_pos_solformat(path),
    )


def parse_rtkpos(
    path: Path,
    leap_seconds: float | None = None,
) -> list[PosRow]:
    """Parse a solver ``.pos`` file into a list of :class:`PosRow`.

    The ``.pos`` time system is read from the column-header line
    (``%  GPST ...`` vs ``%  UTC ...``):

    * **GPST** (the solver default, and what every observed solver-output file
      uses): converted to UTC by subtracting ``leap_seconds``.
    * **UTC**: already UTC — the epoch offset is **not** subtracted, avoiding a
      spurious ~18 s double-subtraction (~180 m of error).

    When the header has no recognisable time-system token, ``GPST`` is assumed
    (the historical behaviour), so existing files parse identically.

    If ``leap_seconds`` is not provided, it is automatically determined
    from the first timestamp in the file using the built-in epoch offset table.
    An explicit ``leap_seconds`` is honoured for ``GPST`` files; for UTC-labelled
    files no offset subtraction is applied regardless.

    Velocity columns (vn, ve, vu) are read when present (The external solver writes them
    when the solution mode is set to NEU).

    Coordinate solformat handling (see :func:`detect_pos_solformat`):

    * ``llh`` — parsed directly (the historical, unchanged path).
    * ``ecef`` — the x/y/z columns are converted to lat/lon/h
      via :func:`data_pipeline.geo.ecef_to_llh`; the resulting rows are
      identical in meaning to the llh path.
    * ``enu`` — raises :class:`ValueError`: a baseline-relative ``.pos`` is
      relative to the reference input and carries no absolute position.
    * ``unknown`` — assumed llh (back-compat) with a WARN naming the
      unrecognised column header.
    """
    solformat = detect_pos_solformat(path)
    if solformat == "enu":
        raise ValueError(
            f"{path} is an ENU/baseline .pos (e/n/u-baseline columns): it "
            "contains only the rover-minus-base vector, so there is no "
            "absolute position without the base coordinate. Re-run RTKLIB "
            "with out-solformat=llh (RTKPost: Options > Output > Solution "
            "Format = Lat/Lon/Height) and use that .pos instead."
        )
    is_ecef = solformat == "ecef"
    if is_ecef:
        from . import geo as _geo
        _log.info("[pos] input is ECEF solformat -> converted to lat/lon/h")
    elif solformat == "unknown":
        hdr = _pos_column_header_line(path)
        _log.warning(
            "[pos] unrecognized .pos column header %s; assuming "
            "lat/lon/h (llh) columns.",
            repr(hdr) if hdr is not None else "(no % header line found)",
        )

    time_system, time_token_found = _detect_pos_time_system_ex(path)
    if not time_token_found:
        _log.warning(
            "[pos] no time-system token in header; assuming GPST (18 s from "
            "UTC). If this .pos is UTC, media sync will be ~18 s off."
        )
    is_utc = time_system == "UTC"
    rows: list[PosRow] = []
    # Epoch offset resolution. The reliable rule is "look up per row", but
    # that fires an iterative datetime + table lookup for every Post-processing epoch
    # (1 Hz × 35 min ≈ 2 100 rows). We resolve first + last rows up front
    # and, when they agree (the 100% case in practice — sessions don't
    # straddle a epoch offset insertion today), reuse that constant for the
    # whole file. The slow per-row iteration only runs in the constructed
    # boundary-crossing case.

    def _refine_ls(t_gpst: dt.datetime) -> float:
        ls = get_leap_seconds_for_datetime(t_gpst)
        t_unix = t_gpst.timestamp()
        for _ in range(2):
            refined_utc = dt.datetime.fromtimestamp(t_unix - ls, tz=dt.timezone.utc)
            new_ls = get_leap_seconds_for_datetime(refined_utc)
            if new_ls == ls:
                break
            ls = new_ls
        return ls

    def _parse_row_time(date_s: str, time_s: str) -> dt.datetime:
        return dt.datetime.strptime(
            f"{date_s} {time_s}", "%Y/%m/%d %H:%M:%S.%f"
        ).replace(tzinfo=dt.timezone.utc)

    constant_ls: float | None = None
    if leap_seconds is None and not is_utc:
        first_t: dt.datetime | None = None
        last_t: dt.datetime | None = None
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    t_curr = _parse_row_time(parts[0], parts[1])
                except ValueError:
                    continue
                if first_t is None:
                    first_t = t_curr
                last_t = t_curr
        if first_t is not None and last_t is not None:
            ls_first = _refine_ls(first_t)
            ls_last = _refine_ls(last_t)
            if ls_first == ls_last:
                constant_ls = ls_first

    n_skipped_malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 6:
                n_skipped_malformed += 1
                continue
            try:
                date_s, time_s = parts[0], parts[1]
                if is_ecef:
                    # x/y/z-cartesian XYZ occupy the same columns 2-4; convert to
                    # datum-based so PosRow is populated identically to llh.
                    lat, lon, h = _geo.ecef_to_llh(
                        float(parts[2]), float(parts[3]), float(parts[4])
                    )
                else:
                    lat = float(parts[2])
                    lon = float(parts[3])
                    h = float(parts[4])
                q = int(parts[5])
                ns = int(parts[6]) if len(parts) >= 7 else 0
                # Per-epoch sigmas (cols 7-9 in standard The external solver output).
                sd_n  = float(parts[7])  if len(parts) >= 8  else float("nan")
                sd_e  = float(parts[8])  if len(parts) >= 9  else float("nan")
                sd_u  = float(parts[9])  if len(parts) >= 10 else float("nan")
                sd_ne = float(parts[10]) if len(parts) >= 11 else float("nan")
                sd_eu = float(parts[11]) if len(parts) >= 12 else float("nan")
                sd_un = float(parts[12]) if len(parts) >= 13 else float("nan")
                age_s = float(parts[13]) if len(parts) >= 14 else float("nan")
                ratio = float(parts[14]) if len(parts) >= 15 else float("nan")
                vn = float(parts[15]) if len(parts) >= 16 else float("nan")
                ve = float(parts[16]) if len(parts) >= 17 else float("nan")
                vu = float(parts[17]) if len(parts) >= 18 else float("nan")
                sd_vn  = float(parts[18]) if len(parts) >= 19 else float("nan")
                sd_ve  = float(parts[19]) if len(parts) >= 20 else float("nan")
                sd_vu  = float(parts[20]) if len(parts) >= 21 else float("nan")
                sd_vne = float(parts[21]) if len(parts) >= 22 else float("nan")
                sd_veu = float(parts[22]) if len(parts) >= 23 else float("nan")
                sd_vun = float(parts[23]) if len(parts) >= 24 else float("nan")
            except ValueError:
                n_skipped_malformed += 1
                continue
            # The row timestamp is parsed as a naive-UTC datetime so
            # ``.timestamp()`` yields the labelled-system Unix-epoch second.
            # For Reference time files we then subtract epoch offset to reach true UTC;
            # for a UTC-labelled file the value is already UTC (ls = 0).
            t_row = _parse_row_time(date_s, time_s)
            if is_utc:
                ls = 0.0
            elif leap_seconds is not None:
                ls = leap_seconds
            elif constant_ls is not None:
                ls = constant_ls
            else:
                ls = _refine_ls(t_row)
            utc_s = t_row.timestamp() - ls
            rows.append(PosRow(
                utc_s, lat, lon, h, q, vn, ve, vu, ns,
                sd_n=sd_n, sd_e=sd_e, sd_u=sd_u,
                sd_ne=sd_ne, sd_eu=sd_eu, sd_un=sd_un,
                age_s=age_s, ratio=ratio,
                sd_vn=sd_vn, sd_ve=sd_ve, sd_vu=sd_vu,
                sd_vne=sd_vne, sd_veu=sd_veu, sd_vun=sd_vun,
            ))
    rows.sort(key=lambda r: r.utc_s)
    if n_skipped_malformed > 0 and not rows:
        raise RuntimeError(
            f"parse_rtkpos: every data row in {path} was malformed "
            f"({n_skipped_malformed} skipped). Verify the file is a "
            "valid RTKLIB .pos (lat/lon/h/Q/ns minimum)."
        )
    if n_skipped_malformed > 0:
        import warnings
        warnings.warn(
            f"parse_rtkpos: skipped {n_skipped_malformed} malformed row(s) "
            f"in {path.name}; {len(rows)} rows usable.",
            RuntimeWarning, stacklevel=2,
        )
    return rows


def parse_data_fix(path: Path) -> list[DataFix]:
    """Parse ``Fix,...`` lines from a source-app ``gnss_log_*.txt``.

    Only fixes with valid lat/lon and a millisecond UTC timestamp are kept.
    """
    rows: list[DataFix] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("Fix,"):
                continue
            parts = line.rstrip("\r\n").split(",")
            if len(parts) < 13:
                continue
            try:
                provider = parts[1]
                lat = float(parts[2])
                lon = float(parts[3])
                alt = float(parts[4]) if parts[4] != "" else float("nan")
                spd = float(parts[5]) if parts[5] != "" else float("nan")
                acc = float(parts[6]) if parts[6] != "" else float("nan")
                brg = float(parts[7]) if parts[7] != "" else float("nan")
                t_ms = int(parts[8])
                vacc = float(parts[12]) if parts[12] != "" else float("nan")
                t = dt.datetime.fromtimestamp(t_ms / 1000.0, tz=dt.timezone.utc)
            except (ValueError, OSError, OverflowError):
                # ValueError: malformed numeric. OSError/OverflowError:
                # fromtimestamp out of platform range (Windows: pre-1970).
                continue
            rows.append(DataFix(t.timestamp(), provider, lat, lon, alt, acc, vacc, spd, brg))
    rows.sort(key=lambda r: r.utc_s)
    return rows


def parse_orientation(path: Path) -> list[Orient]:
    """Parse ``OrientationDeg,...`` lines.

    the source app format::

        OrientationDeg,utcMs,elapsedNs,yawDeg,rollDeg,pitchDeg,calibrationAccuracy
    """
    rows: list[Orient] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("OrientationDeg,"):
                continue
            parts = line.rstrip("\r\n").split(",")
            if len(parts) < 7:
                continue
            try:
                t_ms = int(parts[1])
                yaw = float(parts[3])
                roll = float(parts[4])
                pitch = float(parts[5])
                cal = int(parts[6])
            except ValueError:
                continue
            rows.append(Orient(t_ms / 1000.0, yaw, roll, pitch, cal))
    rows.sort(key=lambda r: r.utc_s)
    return rows


def read_frame_times_csv(path: Path) -> list[tuple[str, float]]:
    """Read ``extracted_frame_times.csv`` (Image, t_video_s).

    Returns ``(image, t_video_s)`` tuples sorted by ascending PTS. Both
    extractor paths write sorted CSVs today, but every downstream consumer
    (``_load_frames``, the bracket-based interpolators, the time-windowed
    yaw helper) assumes monotonic ordering; sorting here makes that
    invariant immune to hand-edits or third-party tooling that might
    write the rows in a different order.
    """
    out: list[tuple[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = (ln for ln in f if not ln.lstrip().startswith("#"))
        for row in csv.DictReader(rows):
            name = (row.get("Image") or "").strip()
            try:
                t = float(row.get("t_video_s") or "")
            except ValueError:
                continue
            if name:
                out.append((name, t))
    out.sort(key=lambda r: r[1])
    return out


# -----------------------------
# Time-aware interpolation
# -----------------------------


def _bisect_pair(times: list[float], t: float) -> tuple[int, int] | None:
    """Return the indices ``(i-1, i)`` bracketing ``t``, or None outside data.

    Endpoints are admitted by collapsing to a degenerate pair (``i == j``) so
    a sample whose UTC equals exactly the first or last sample is interpolated
    rather than dropped. This matters for the very first / last extracted
    sample in tightly-cut sessions.
    """
    n = len(times)
    if n == 0 or t < times[0] or t > times[-1]:
        return None
    i = bisect_left(times, t)
    if i == 0:
        # t == times[0] exactly (we already filtered t < times[0]).
        return 0, 0
    if i >= n or (i == n - 1 and times[i] == t):
        # bisect_left returns n only when t > times[-1], already filtered.
        # Second condition: exact match at last sample → degenerate pair.
        return n - 1, n - 1
    return i - 1, i


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_angle_deg(a: float, b: float, t: float) -> float:
    """Linear interpolation of degrees that respects ±180 wrap."""
    diff = ((b - a + 540.0) % 360.0) - 180.0
    return a + diff * t


def _times_of(rows: list) -> list[float]:
    """Extract the ``utc_s`` field from any row sequence into a fresh list.

    Exposed so callers that loop over many query times can pre-compute it
    once and avoid the O(N) rebuild on every interp_* call.
    """
    return [r.utc_s for r in rows]


def interp_pos(
    rows: list[PosRow],
    utc_s: float,
    max_gap_s: float,
    times: list[float] | None = None,
) -> tuple[float, float, float] | None:
    """Linear interpolation of (lat,lon,h) at ``utc_s`` from sorted Post-processing rows.

    Returns ``None`` when ``utc_s`` is outside the data or when the bracketing
    pair is further apart than ``max_gap_s`` on either side (e.g. tunnel).

    Pre-compute ``times = [r.utc_s for r in rows]`` and pass it in when
    interpolating many query times against the same ``rows`` to avoid the
    quadratic cost of rebuilding the list on every call.
    """
    if not rows:
        return None
    if times is None:
        times = _times_of(rows)
    pair = _bisect_pair(times, utc_s)
    if pair is None:
        return None
    ai, bi = pair
    a, b = rows[ai], rows[bi]
    # Reject when:
    #  (1) the bracket itself spans more than ``max_gap_s`` (a tunnel: even
    #      a query at the midpoint would interpolate across the gap), or
    #  (2) the query is more than ``max_gap_s`` from the nearer endpoint
    #      (e.g. extrapolation right after a tunnel ends).
    if (b.utc_s - a.utc_s) > max_gap_s:
        return None
    if (utc_s - a.utc_s) > max_gap_s or (b.utc_s - utc_s) > max_gap_s:
        return None
    if b.utc_s == a.utc_s:
        return (a.lat_deg, a.lon_deg, a.h_m)
    t = (utc_s - a.utc_s) / (b.utc_s - a.utc_s)
    return (
        lerp(a.lat_deg, b.lat_deg, t),
        lerp(a.lon_deg, b.lon_deg, t),
        lerp(a.h_m, b.h_m, t),
    )


def interp_pos_with_velocity(
    rows: list[PosRow],
    utc_s: float,
    max_gap_s: float,
    times: list[float] | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Like :func:`interp_pos` but also returns interpolated (vn, ve, vu).

    Returns ``None`` outside the data or beyond ``max_gap_s``. When the
    underlying ``.pos`` lacks velocity columns the velocity components come
    back as ``NaN``s; the caller can detect this with ``math.isnan``.
    """
    if not rows:
        return None
    if times is None:
        times = _times_of(rows)
    pair = _bisect_pair(times, utc_s)
    if pair is None:
        return None
    ai, bi = pair
    a, b = rows[ai], rows[bi]
    if (b.utc_s - a.utc_s) > max_gap_s:
        return None
    if (utc_s - a.utc_s) > max_gap_s or (b.utc_s - utc_s) > max_gap_s:
        return None
    if b.utc_s == a.utc_s:
        return (a.lat_deg, a.lon_deg, a.h_m, a.vn, a.ve, a.vu)
    t = (utc_s - a.utc_s) / (b.utc_s - a.utc_s)
    return (
        lerp(a.lat_deg, b.lat_deg, t),
        lerp(a.lon_deg, b.lon_deg, t),
        lerp(a.h_m, b.h_m, t),
        lerp(a.vn, b.vn, t),
        lerp(a.ve, b.ve, t),
        lerp(a.vu, b.vu, t),
    )


def interp_data(
    rows: list[DataFix],
    utc_s: float,
    max_gap_s: float,
    times: list[float] | None = None,
) -> tuple[float, float, float, float, float] | None:
    """Linear interpolation of device Fix lat/lon/h/h_acc/v_acc at ``utc_s``."""
    if not rows:
        return None
    if times is None:
        times = _times_of(rows)
    pair = _bisect_pair(times, utc_s)
    if pair is None:
        return None
    ai, bi = pair
    a, b = rows[ai], rows[bi]
    if (b.utc_s - a.utc_s) > max_gap_s:
        return None
    if (utc_s - a.utc_s) > max_gap_s or (b.utc_s - utc_s) > max_gap_s:
        return None
    if b.utc_s == a.utc_s:
        return (a.lat, a.lon, a.h, a.h_acc, a.v_acc)
    t = (utc_s - a.utc_s) / (b.utc_s - a.utc_s)
    return (
        lerp(a.lat, b.lat, t),
        lerp(a.lon, b.lon, t),
        lerp(a.h, b.h, t),
        lerp(a.h_acc, b.h_acc, t),
        lerp(a.v_acc, b.v_acc, t),
    )


def interp_orient(
    rows: list[Orient],
    utc_s: float,
    max_gap_s: float,
    times: list[float] | None = None,
) -> Orient | None:
    """Interpolate device YPR at ``utc_s``, respecting angle wrap-around."""
    if not rows:
        return None
    if times is None:
        times = _times_of(rows)
    pair = _bisect_pair(times, utc_s)
    if pair is None:
        return None
    ai, bi = pair
    a, b = rows[ai], rows[bi]
    if (b.utc_s - a.utc_s) > max_gap_s:
        return None
    if (utc_s - a.utc_s) > max_gap_s or (b.utc_s - utc_s) > max_gap_s:
        return None
    if b.utc_s == a.utc_s:
        return a
    t = (utc_s - a.utc_s) / (b.utc_s - a.utc_s)
    return Orient(
        utc_s=utc_s,
        yaw=lerp_angle_deg(a.yaw, b.yaw, t) % 360.0,
        roll=lerp_angle_deg(a.roll, b.roll, t),
        pitch=lerp_angle_deg(a.pitch, b.pitch, t),
        cal=min(a.cal, b.cal),
    )


def decimate_orientation(rows: list[Orient], target_hz: float) -> list[Orient]:
    """Down-sample an orientation stream to roughly ``target_hz`` samples/s.

    The device records OrientationDeg at ~190 Hz which makes any pure-Python
    Gaussian convolution painfully slow. Decimating to 10 Hz before smoothing
    is more than enough for vehicle dynamics and turns the smoothing pass
    into a sub-second job.
    """
    if target_hz <= 0 or len(rows) <= 1:
        return rows
    dt_target = 1.0 / target_hz
    out: list[Orient] = [rows[0]]
    for r in rows[1:]:
        if r.utc_s - out[-1].utc_s >= dt_target:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Motion sensor (sensors_*.txt) — linear sensor + rate sensor at ~200 Hz
# ---------------------------------------------------------------------------

# Reference epoch as Unix timestamp (1980-01-06 00:00:00 UTC).
_GPS_EPOCH_UNIX_S: int = 315964800


@dataclass(frozen=True)
class ImuRow:
    """One row of sensors_*.txt: linear sensor + rate sensor sample."""

    utc_s: float
    ax: float  # m/s²
    ay: float  # m/s²
    az: float  # m/s²
    gx: float  # rad/s
    gy: float  # rad/s
    gz: float  # rad/s


def parse_imu(
    path: Path,
    leap_seconds: float | None = None,
) -> list[ImuRow]:
    """Parse sensors_*.txt to UTC.

    Two on-disk column layouts are supported and auto-detected per row by
    column count:

      LEGACY (7 cols, old app / DAY12):
        GPS_seconds, gx, gy, gz, ax, ay, az
        Columns 1-3 = rate sensor (rad/s); columns 4-6 = full linear sensor
        (m/s², magnitude ~9.81 when stationary).

      NEW (13 cols, current app / DAY14, anchor_format 2):
        GPS_seconds, gx, gy, gz, <linAccX,Y,Z>, ax, ay, az, <mag/other...>
        Columns 1-3 = rate sensor (rad/s); columns 4-6 = a near-zero triple
        (linear-linear sensor / uncalibrated-rate sensor residual, magnitude ~0.0);
        columns 7-9 = the full linear sensor WITH gravity (magnitude ~9.81
        when stationary). Reading 4-6 here would yield ~0 g and silently
        break ZUPT / Motion sensor calibration, so the linear sensor columns must shift to 7-9.

    The correct linear sensor columns are chosen by the row's field count: >=10 fields
    -> new layout (linear sensor 7-9); otherwise legacy (linear sensor 4-6). Rate sensor is always
    columns 1-3 in both layouts.

    Column 0 is Reference-epoch seconds (since 1980-01-06) in both capture formats,
    converted to UTC Unix seconds by adding the Reference epoch offset and subtracting
    the current epoch offset. (No device-boottime path: per the capture
    spec, sensors timestamps are always Reference-epoch seconds.)
    """
    # leap_seconds: if caller supplies a value we honour it for every row.
    # Otherwise compute per-row via ``get_leap_seconds_for_datetime``. As an
    # optimisation we resolve the epoch offset offset for the FIRST and LAST
    # rows of the file up front; when they agree (any session that does NOT
    # cross a epoch offset boundary — the 100% case today) we apply that
    # constant to every row, saving a datetime + table lookup per Motion sensor sample
    # (matters: 200 Hz × 35 min ≈ 420 k rows). The per-row iterative refine
    # is only used when first ≠ last, which only happens in the deliberately
    # constructed midnight-epoch offset-insertion case.
    explicit_leap = leap_seconds

    def _resolve_ls_for(gps_s: float) -> float:
        rough_utc = gps_s + _GPS_EPOCH_UNIX_S - GPS_UTC_LEAP_SECONDS_2026
        ls = get_leap_seconds_for_datetime(
            dt.datetime.fromtimestamp(rough_utc, tz=dt.timezone.utc)
        )
        for _ in range(2):
            refined_utc = gps_s + _GPS_EPOCH_UNIX_S - ls
            new_ls = get_leap_seconds_for_datetime(
                dt.datetime.fromtimestamp(refined_utc, tz=dt.timezone.utc)
            )
            if new_ls == ls:
                break
            ls = new_ls
        return ls

    constant_ls: float | None = None
    if explicit_leap is None:
        first_g: float | None = None
        last_g: float | None = None
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split(",")
                if len(parts) < 7:
                    continue
                try:
                    g = float(parts[0])
                except ValueError:
                    continue
                if first_g is None:
                    first_g = g
                last_g = g
        if first_g is not None and last_g is not None:
            ls_first = _resolve_ls_for(first_g)
            ls_last = _resolve_ls_for(last_g)
            if ls_first == ls_last:
                constant_ls = ls_first

    rows: list[ImuRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) < 7:
                continue
            try:
                gps_s = float(parts[0])
                if explicit_leap is not None:
                    ls = explicit_leap
                elif constant_ls is not None:
                    ls = constant_ls
                else:
                    ls = _resolve_ls_for(gps_s)
                utc_s = gps_s + _GPS_EPOCH_UNIX_S - ls
                # Linear sensor column offset depends on layout: the new 13-column
                # format (>=10 fields) carries the gravity-bearing linear sensor
                # in columns 7-9; the legacy 7-column format in columns 4-6.
                # Rate sensor is columns 1-3 in both.
                if len(parts) >= 10:
                    ax_i, ay_i, az_i = 7, 8, 9
                else:
                    ax_i, ay_i, az_i = 4, 5, 6
                rows.append(ImuRow(
                    utc_s=utc_s,
                    ax=float(parts[ax_i]),
                    ay=float(parts[ay_i]),
                    az=float(parts[az_i]),
                    # columns 1-3 are the rate sensor (angular velocity, rad/s)
                    gx=float(parts[1]),
                    gy=float(parts[2]),
                    gz=float(parts[3]),
                ))
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r.utc_s)
    return rows


def detect_static_periods(
    pos_rows: list[PosRow],
    min_duration_s: float = 1.5,
    max_speed_mps: float = 0.4,
) -> list[tuple[float, float]]:
    """Return (start_utc, end_utc) intervals where Post-processing velocity < max_speed_mps.

    Requires velocity columns (vn, ve) in the .pos file. Periods shorter than
    min_duration_s are discarded to avoid false positives at traffic lights.
    """
    periods: list[tuple[float, float]] = []
    static_start: float | None = None
    for r in pos_rows:
        if math.isfinite(r.vn) and math.isfinite(r.ve):
            speed = math.sqrt(r.vn * r.vn + r.ve * r.ve)
            is_static = speed < max_speed_mps
        else:
            is_static = False
        if is_static:
            if static_start is None:
                static_start = r.utc_s
        else:
            if static_start is not None:
                if (r.utc_s - static_start) >= min_duration_s:
                    periods.append((static_start, r.utc_s))
                static_start = None
    if static_start is not None and pos_rows:
        if (pos_rows[-1].utc_s - static_start) >= min_duration_s:
            periods.append((static_start, pos_rows[-1].utc_s))
    return periods


def gravity_pitch_roll_from_static(
    imu_rows: list[ImuRow],
    static_periods: list[tuple[float, float]],
    min_samples: int = 20,
) -> list[tuple[float, float, float]]:
    """Compute absolute pitch/roll from the gravity vector during static stops.

    When the vehicle is stopped the linear sensor measures only gravity.
    Averaging over the static window suppresses vibration noise and gives
    a gravity direction accurate to <0.1°, free of any rate sensor drift.

    Returns sorted list of ``(mid_utc_s, pitch_deg, roll_deg)`` anchor points.

    The platform sample convention (device face-up, Y toward top of screen):
      pitch = -atan2(ax, sqrt(ay²+az²))   — forward/backward tilt
      roll  =  atan2(ay, az)               — left/right tilt
    """
    if not imu_rows or not static_periods:
        return []
    imu_times = [r.utc_s for r in imu_rows]
    anchors: list[tuple[float, float, float]] = []
    for t_start, t_end in static_periods:
        i0 = bisect_left(imu_times, t_start)
        i1 = bisect_left(imu_times, t_end)
        window = imu_rows[i0:i1]
        if len(window) < min_samples:
            continue
        ax = sum(r.ax for r in window) / len(window)
        ay = sum(r.ay for r in window) / len(window)
        az = sum(r.az for r in window) / len(window)
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag < 5.0:  # sanity: gravity ≈ 9.81 m/s²; <5 means data is wrong
            continue
        ax /= mag
        ay /= mag
        az /= mag
        pitch = math.degrees(-math.atan2(ax, math.sqrt(ay * ay + az * az)))
        roll = math.degrees(math.atan2(ay, az))
        mid = (t_start + t_end) / 2.0
        anchors.append((mid, pitch, roll))
    return sorted(anchors, key=lambda x: x[0])
