"""Fine measurements pre-check for Interchange-format 3 observation files.

The platform devices with Signal duty-cycling enabled (or with ADR flagged invalid)
log coarse measurement but write every fine measurements (``L*``) observable as 0.000.
The external solver happily processes such a file — but without phase there is no Live-correction/Post-processing
solution refinement, so the whole run silently degrades to code Differential (Q=4,
metre-level). This module detects that failure mode *before* the expensive
``the solver binary`` run so the operator gets an actionable message.

Only reads the file; never raises on malformed records (they are skipped),
though a missing file / unreadable header still raises so callers can decide.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Interchange-format 3 observation record layout: 3-char source id, then one 16-char
# slot per observable: F14.3 value + 1 LLI char + 1 signal-strength char.
_SAT_ID_RE = re.compile(r"^[A-Z][ 0-9][0-9]")
_OBS_SLOT_W = 16
_OBS_VALUE_W = 14

# Fraction of source phase slots that must be non-zero for the file to
# count as "has usable fine measurements". Real files with phase sit at tens of
# percent; duty-cycled files sit at exactly 0.
_MIN_PHASE_FRACTION = 0.01


@dataclass
class ObsPhaseReport:
    """Result of :func:`check_carrier_phase`.

    ``n_sat_obs`` counts source-phase slots scanned (one per source
    per phase observable per epoch); ``n_phase_nonzero`` counts how many of
    those carried a non-zero phase value. ``per_system`` maps source group
    letter (G/R/E/C/J/I/S) to ``{"phase_types": [...], "n_sat_obs": int,
    "n_phase_nonzero": int}``.
    """

    has_phase: bool
    n_phase_nonzero: int
    n_sat_obs: int
    per_system: dict = field(default_factory=dict)
    message: str = ""


def _parse_obs_types_header(lines: list[str]) -> dict[str, list[str]]:
    """Parse ``SYS / # / OBS TYPES`` header lines.

    Returns ``{sys_letter: [obs_code, ...]}`` in column order. Handles the
    header's own continuation lines (first char blank, codes continue).
    """
    sys_types: dict[str, list[str]] = {}
    current_sys: str | None = None
    expected: int = 0
    for line in lines:
        label = line[60:].strip()
        if label != "SYS / # / OBS TYPES":
            continue
        body = line[:60]
        if body[:1].strip():  # new system line: "G    8 C1C L1C ..."
            toks = body.split()
            if len(toks) < 2:
                continue
            current_sys = toks[0]
            try:
                expected = int(toks[1])
            except ValueError:
                expected = 0
            sys_types[current_sys] = toks[2:]
        elif current_sys is not None:  # continuation: codes only
            sys_types[current_sys].extend(body.split())
        if current_sys is not None and expected:
            # Cut any accidental over-read (defensive).
            sys_types[current_sys] = sys_types[current_sys][:expected]
    return sys_types


def check_carrier_phase(obs_path: Path | str) -> ObsPhaseReport:
    """Scan a Interchange-format 3 obs file and report whether usable fine measurements exists.

    From the ``SYS / # / OBS TYPES`` header the per-source group phase
    observables (``L*``) and their column indices are determined; the data
    epochs are then scanned counting non-zero phase values (a real L1 phase
    is ~1e7-1e8 cycles; ``0.000`` or blank means the unit logged no ADR).

    ``has_phase`` is False when zero — or fewer than 1% — of the source
    phase slots carry a value: Live-correction/Post-processing is impossible and The external solver will only
    produce code-Differential (Q=4) solutions.
    """
    obs_path = Path(obs_path)

    header_lines: list[str] = []
    with obs_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            header_lines.append(raw.rstrip("\r\n"))
            if raw[60:].strip() == "END OF HEADER":
                break

    sys_types = _parse_obs_types_header(header_lines)
    # Per system: column indices of phase (L*) observables.
    phase_cols: dict[str, list[int]] = {
        sys: [i for i, code in enumerate(codes) if code.startswith("L")]
        for sys, codes in sys_types.items()
    }

    per_system: dict[str, dict] = {
        sys: {
            "phase_types": [sys_types[sys][i] for i in cols],
            "n_sat_obs": 0,
            "n_phase_nonzero": 0,
        }
        for sys, cols in phase_cols.items()
    }

    n_phase_nonzero = 0
    n_sat_obs = 0

    def _scan_sat_line(line: str) -> None:
        nonlocal n_phase_nonzero, n_sat_obs
        sys = line[0]
        cols = phase_cols.get(sys)
        if not cols:
            return
        stats = per_system[sys]
        for k in cols:
            start = 3 + _OBS_SLOT_W * k
            valstr = line[start:start + _OBS_VALUE_W].strip()
            n_sat_obs += 1
            stats["n_sat_obs"] += 1
            if not valstr:
                continue
            try:
                val = float(valstr)
            except ValueError:
                continue
            if val != 0.0:
                n_phase_nonzero += 1
                stats["n_phase_nonzero"] += 1

    with obs_path.open("r", encoding="utf-8", errors="replace") as f:
        in_header = True
        pending: str | None = None  # source line awaiting possible continuation
        skip_event_lines = 0
        for raw in f:
            line = raw.rstrip("\r\n")
            if in_header:
                if line[60:].strip() == "END OF HEADER":
                    in_header = False
                continue
            if skip_event_lines > 0:
                skip_event_lines -= 1
                continue
            if line.startswith(">"):
                if pending is not None:
                    _scan_sat_line(pending)
                    pending = None
                # Epoch flag > 1 marks an event block: the following
                # "record count" lines are header-style text, not obs.
                toks = line[1:].split()
                try:
                    flag = int(toks[6]) if len(toks) >= 7 else 0
                    count = int(toks[7]) if len(toks) >= 8 else 0
                except ValueError:
                    flag, count = 0, 0
                if flag > 1:
                    skip_event_lines = count
                continue
            if _SAT_ID_RE.match(line):
                if pending is not None:
                    _scan_sat_line(pending)
                pending = line
            elif pending is not None and line.strip():
                # Interchange-format 3 normally keeps a source's observations on one
                # line, but tolerate writers that wrap: a non-'>' non-source
                # line continues the previous record.
                pending = pending + line
        if pending is not None:
            _scan_sat_line(pending)

    frac = (n_phase_nonzero / n_sat_obs) if n_sat_obs else 0.0
    has_phase = n_sat_obs > 0 and frac >= _MIN_PHASE_FRACTION

    if has_phase:
        message = (
            f"carrier phase OK in {obs_path.name}: "
            f"{n_phase_nonzero}/{n_sat_obs} phase slots non-zero "
            f"({100.0 * frac:.1f}%)."
        )
    else:
        message = (
            f"no usable carrier phase in {obs_path.name} "
            f"({n_phase_nonzero}/{n_sat_obs} phase slots non-zero) -> "
            "RTK/PPK impossible; RTKLIB will only produce code DGPS (Q=4). "
            "Likely GNSS duty-cycling was on / ADR invalid. Recapture with "
            '"Force full GNSS measurements".'
        )

    return ObsPhaseReport(
        has_phase=has_phase,
        n_phase_nonzero=n_phase_nonzero,
        n_sat_obs=n_sat_obs,
        per_system=per_system,
        message=message,
    )


# ---------------------------------------------------------------------------
# Observation summary: SNR + source-count aggregation
# ---------------------------------------------------------------------------


@dataclass
class ObsSummary:
    """Result of :func:`summarize_obs` — SNR / source-count aggregation.

    ``avg_snr_db`` is the mean of every non-zero ``S*`` (signal-strength,
    dB-Hz) value in the file; ``snr_per_system`` maps source group letter
    (G/R/E/C/J/I/S) to that system's mean. The per-epoch lists
    (``times_s`` / ``sats_per_epoch`` / ``snr_per_epoch``) are aligned and
    suitable for time-series plots; ``times_s`` is the epoch timestamp in
    Unix-like seconds on the file's own time scale (Interchange-format obs epochs are
    unit time, typically Reference time — fine for relative plotting).
    ``interval_s`` is the median epoch spacing (NaN with < 2 epochs).
    """

    epoch_count: int
    interval_s: float
    avg_sats_per_epoch: float
    avg_snr_db: float
    snr_per_system: dict = field(default_factory=dict)
    times_s: list = field(default_factory=list)
    sats_per_epoch: list = field(default_factory=list)
    snr_per_epoch: list = field(default_factory=list)
    message: str = ""


def _parse_epoch_time(line: str) -> float:
    """Unix-like seconds from a ``> yyyy mm dd hh mm ss.sss`` epoch line.

    Returns NaN when the line is malformed (caller keeps counting the epoch
    anyway; only the time axis loses that point).
    """
    toks = line[1:].split()
    try:
        y, mo, d, h, mi = (int(t) for t in toks[:5])
        s = float(toks[5])
    except (ValueError, IndexError):
        return float("nan")
    try:
        base = _dt.datetime(y, mo, d, h, mi, tzinfo=_dt.timezone.utc)
    except ValueError:
        return float("nan")
    return base.timestamp() + s


def summarize_obs(obs_path: Path | str) -> ObsSummary:
    """Scan a Interchange-format 3 obs file: avg SNR (overall + per source group),
    avg sources-per-epoch, epoch count and epoch interval.

    Uses the same tolerant record scanner as :func:`check_carrier_phase`
    (event blocks skipped, wrapped source lines re-joined, malformed
    records ignored). SNR comes from the ``S*`` observables declared in the
    ``SYS / # / OBS TYPES`` header; zero / blank slots are not counted.
    """
    obs_path = Path(obs_path)

    header_lines: list[str] = []
    with obs_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            header_lines.append(raw.rstrip("\r\n"))
            if raw[60:].strip() == "END OF HEADER":
                break

    sys_types = _parse_obs_types_header(header_lines)
    snr_cols: dict[str, list[int]] = {
        sys: [i for i, code in enumerate(codes) if code.startswith("S")]
        for sys, codes in sys_types.items()
    }

    snr_sum: dict[str, float] = {sys: 0.0 for sys in snr_cols}
    snr_n: dict[str, int] = {sys: 0 for sys in snr_cols}

    times_s: list[float] = []
    sats_per_epoch: list[int] = []
    snr_per_epoch: list[float] = []

    # Per-epoch accumulators.
    ep_sats = 0
    ep_snr_sum = 0.0
    ep_snr_n = 0
    have_epoch = False

    def _close_epoch() -> None:
        nonlocal ep_sats, ep_snr_sum, ep_snr_n, have_epoch
        if not have_epoch:
            return
        sats_per_epoch.append(ep_sats)
        snr_per_epoch.append(ep_snr_sum / ep_snr_n if ep_snr_n else float("nan"))
        ep_sats = 0
        ep_snr_sum = 0.0
        ep_snr_n = 0
        have_epoch = False

    def _scan_sat_line(line: str) -> None:
        nonlocal ep_sats, ep_snr_sum, ep_snr_n
        ep_sats += 1
        sys = line[0]
        cols = snr_cols.get(sys)
        if not cols:
            return
        for k in cols:
            start = 3 + _OBS_SLOT_W * k
            valstr = line[start:start + _OBS_VALUE_W].strip()
            if not valstr:
                continue
            try:
                val = float(valstr)
            except ValueError:
                continue
            if val > 0.0:
                snr_sum[sys] = snr_sum.get(sys, 0.0) + val
                snr_n[sys] = snr_n.get(sys, 0) + 1
                ep_snr_sum += val
                ep_snr_n += 1

    with obs_path.open("r", encoding="utf-8", errors="replace") as f:
        in_header = True
        pending: str | None = None
        skip_event_lines = 0
        for raw in f:
            line = raw.rstrip("\r\n")
            if in_header:
                if line[60:].strip() == "END OF HEADER":
                    in_header = False
                continue
            if skip_event_lines > 0:
                skip_event_lines -= 1
                continue
            if line.startswith(">"):
                if pending is not None:
                    _scan_sat_line(pending)
                    pending = None
                toks = line[1:].split()
                try:
                    flag = int(toks[6]) if len(toks) >= 7 else 0
                    count = int(toks[7]) if len(toks) >= 8 else 0
                except ValueError:
                    flag, count = 0, 0
                if flag > 1:
                    skip_event_lines = count
                    continue
                _close_epoch()
                times_s.append(_parse_epoch_time(line))
                have_epoch = True
                continue
            if _SAT_ID_RE.match(line):
                if pending is not None:
                    _scan_sat_line(pending)
                pending = line
            elif pending is not None and line.strip():
                pending = pending + line
        if pending is not None:
            _scan_sat_line(pending)
        _close_epoch()

    epoch_count = len(sats_per_epoch)
    avg_sats = (sum(sats_per_epoch) / epoch_count) if epoch_count else float("nan")
    total_snr = sum(snr_sum.values())
    total_n = sum(snr_n.values())
    avg_snr = (total_snr / total_n) if total_n else float("nan")
    per_system = {
        sys: (snr_sum[sys] / snr_n[sys])
        for sys in snr_n
        if snr_n[sys] > 0
    }

    finite_t = [t for t in times_s if math.isfinite(t)]
    if len(finite_t) >= 2:
        diffs = sorted(
            b - a for a, b in zip(finite_t, finite_t[1:]) if b > a
        )
        interval = diffs[len(diffs) // 2] if diffs else float("nan")
    else:
        interval = float("nan")

    message = (
        f"{obs_path.name}: {epoch_count} epochs"
        + (f" @ {interval:g} s" if math.isfinite(interval) else "")
        + f", avg {avg_sats:.1f} sats/epoch"
        + (f", avg SNR {avg_snr:.1f} dB-Hz" if math.isfinite(avg_snr) else ", no SNR data")
    )

    return ObsSummary(
        epoch_count=epoch_count,
        interval_s=interval,
        avg_sats_per_epoch=avg_sats,
        avg_snr_db=avg_snr,
        snr_per_system=per_system,
        times_s=times_s,
        sats_per_epoch=sats_per_epoch,
        snr_per_epoch=snr_per_epoch,
        message=message,
    )
