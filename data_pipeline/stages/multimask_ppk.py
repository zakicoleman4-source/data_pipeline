"""Multi-elevation-mask Post-processing ensemble + GT-free environment noise-aware fusion.

This stage productizes the proven R&D from
``DAY12/source_to_external/multimask_ppk_2026-06-28`` (see
``multimask_ppk_report.html`` + ``docs/.../2026-06-28-multimask-post-processing-design.md``).

The idea
========
Run Post-processing at several elevation masks (``pos1-elmask``). Where the per-mask
solutions AGREE you are in open sky: trust the LOWEST mask (most sources,
best geometry). Where they DISAGREE you are in environment noise/high-noise environment: the
disagreement is itself a GT-free environment noise signal (same principle as the
validated v2 trust-disagreement model, r=0.873), so ESCALATE to the minimal
mask above which the higher masks converge — but never to a mask whose
solution is starved of sources / has a bad AR ratio (geometry guard).

Hard rule (honoured here and enforced by the R&D spec)
======================================================
The selector/fusion uses ONLY inter-mask disagreement + per-solution quality
(``ns``, AR ``ratio``, reported sigma, Q flag). **No ground truth at runtime.**
GT is used only for offline scoring, never inside this stage.

Two R&D conclusions baked in
============================
1. Inter-mask DISAGREEMENT predicts true error with precision ~1.0 -> it is
   shipped as the PRIMARY GT-free confidence / suppression flag and is ALWAYS
   emitted (independent of the fused selector).
2. Mask escalation tightens some routes (day4_dodge 23.9 -> 10.4 m) but
   OVER-tightens clean routes when applied blindly -> the fused selector is
   conservative with a geometry guard and defaults to no-harm.

Public API
==========
``run_multimask_ppk(rover_obs, base_obs, nav, base_conf, *, masks=..., workdir, log=None)``
returns a :class:`MultiMaskResult` with per-mask ``.pos`` paths, the fused
``.pos`` path, and a per-epoch disagreement/flag CSV.

The pure-Python fusion + disagreement maths are importable and unit-testable
WITHOUT invoking ``the solver binary`` — :func:`fuse`, :func:`epoch_spread`,
:func:`build_maskset_from_posfiles`, :func:`write_disagreement_csv`,
:func:`write_fused_pos`.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..geo import ecef_to_enu, enu_to_llh, llh_to_ecef
from ..parsers import parse_rtkpos
from ..pipeline import LogFn, make_logger

# Default elevation-mask sweep (degrees). Matches the proven R&D sweep.
DEFAULT_MASKS: tuple[int, ...] = (5, 10, 15, 20, 25, 30)

# Only pos1-elmask is overridden per mask (it is the axis of this stage).
_ELMASK_RE = re.compile(r"^(\s*pos1-elmask\s*=\s*)([^#\r\n]*)(#.*)?$")


# ===========================================================================
# Config patching (override ONLY pos1-elmask)
# ===========================================================================
def patch_elmask(src_conf: Path, dst_conf: Path, elmask: int,
                 *, log: Optional[LogFn] = None) -> Path:
    """Copy ``src_conf`` to ``dst_conf`` overriding only ``pos1-elmask``.

    Everything else in the base config (base position, AR mode, SNR masks,
    source groups, frequencies) is preserved verbatim. If the key is absent
    it is appended so the result is always self-contained.
    """
    log_ = make_logger(log)
    src_conf = Path(src_conf)
    dst_conf = Path(dst_conf)
    lines = src_conf.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    seen = False
    for line in lines:
        m = _ELMASK_RE.match(line)
        if m:
            comment = m.group(3) or ""
            pad = " " if comment else ""
            out.append(f"{m.group(1)}{elmask}{pad}{comment}".rstrip())
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"pos1-elmask        ={elmask}")
    dst_conf.parent.mkdir(parents=True, exist_ok=True)
    dst_conf.write_text("\n".join(out) + "\n", encoding="utf-8")
    log_(f"[multimask] patched elmask={elmask} -> {dst_conf.name}")
    return dst_conf


def _win_abs(p: Path) -> str:
    """Return an absolute path string with native (backslash on Windows)
    separators. the solver (EX 2.5.0 build) fails on forward-slash absolute paths, so
    on Windows we hand it backslash paths explicitly.
    """
    rp = Path(p).resolve()
    s = str(rp)
    if os.name == "nt":
        s = s.replace("/", "\\")
    return s


def _resolve_rnx2rtkp(override: Optional[Path]) -> Optional[Path]:
    """Locate the solver binary; return None if not found (so callers can skip)."""
    if override is not None:
        p = Path(override)
        return p if p.is_file() else None
    # Reuse the central resolver if available; fall back to PATH probe.
    try:
        from ..stages.ppk import resolve_rnx2rtkp  # type: ignore
        return resolve_rnx2rtkp(None)
    except Exception:
        pass
    name = "rnx2rtkp.exe" if os.name == "nt" else "rnx2rtkp"
    found = shutil.which(name)
    if found:
        return Path(found)
    # vendor location
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "vendor" / "rtklib" / name
        if cand.is_file():
            return cand
    return None


# ===========================================================================
# MaskSet: per-mask solutions aligned on common epochs (GT-free)
# ===========================================================================
@dataclass
class MaskSet:
    """Per-mask per-epoch Local-frame + quality, aligned on common Reference time/UTC epochs."""

    masks: list[int]
    ts: np.ndarray                       # (T,) common UTC seconds (sorted)
    ref_llh: tuple[float, float, float]
    E: dict[int, np.ndarray] = field(default_factory=dict)
    N: dict[int, np.ndarray] = field(default_factory=dict)
    U: dict[int, np.ndarray] = field(default_factory=dict)
    Q: dict[int, np.ndarray] = field(default_factory=dict)
    ns: dict[int, np.ndarray] = field(default_factory=dict)
    ratio: dict[int, np.ndarray] = field(default_factory=dict)
    sdh: dict[int, np.ndarray] = field(default_factory=dict)
    _spread_cache: Optional[np.ndarray] = None

    @property
    def T(self) -> int:
        return len(self.ts)


def build_maskset_from_posfiles(
    pos_by_mask: dict[int, Path],
    *,
    min_common: int = 10,
    log: Optional[LogFn] = None,
) -> Optional[MaskSet]:
    """Parse per-mask ``.pos`` files and align them on common epochs.

    ``pos_by_mask`` maps elevation-mask degree -> ``.pos`` path. Epochs are the
    INTERSECTION over masks that produced solutions so every kept timestamp has
    a position for every loaded mask (a fair spread comparison). Returns None
    when fewer than 2 masks are usable or fewer than ``min_common`` shared
    epochs exist.
    """
    log_ = make_logger(log)
    per_mask: dict[int, dict[float, object]] = {}
    for m, p in sorted(pos_by_mask.items()):
        p = Path(p)
        if not p.is_file():
            continue
        rows = parse_rtkpos(p)
        if not rows:
            continue
        per_mask[m] = {r.utc_s: r for r in rows}
    if len(per_mask) < 2:
        log_(f"[multimask] only {len(per_mask)} mask(s) usable; need >=2")
        return None

    common: Optional[set] = None
    for dct in per_mask.values():
        keys = set(dct.keys())
        common = keys if common is None else (common & keys)
    if not common or len(common) < min_common:
        log_(f"[multimask] only {len(common or [])} common epochs; "
             f"need >={min_common}")
        return None
    ts = np.array(sorted(common))

    base_mask = min(per_mask.keys())
    r0 = per_mask[base_mask][ts[0]]
    ref = (r0.lat_deg, r0.lon_deg, r0.h_m)

    ms = MaskSet(masks=sorted(per_mask.keys()), ts=ts, ref_llh=ref)
    for m, dct in per_mask.items():
        E = np.full(len(ts), np.nan)
        N = np.full(len(ts), np.nan)
        U = np.full(len(ts), np.nan)
        Q = np.zeros(len(ts), dtype=int)
        nsa = np.zeros(len(ts), dtype=int)
        rat = np.full(len(ts), np.nan)
        sdh = np.full(len(ts), np.nan)
        for i, t in enumerate(ts):
            r = dct[t]
            x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
            e, n, u = ecef_to_enu(x, y, z, ref)
            E[i], N[i], U[i] = e, n, u
            Q[i] = r.quality
            nsa[i] = r.ns
            rat[i] = r.ratio
            if math.isfinite(r.sd_n) and math.isfinite(r.sd_e):
                sdh[i] = math.hypot(r.sd_n, r.sd_e)
        ms.E[m] = E; ms.N[m] = N; ms.U[m] = U
        ms.Q[m] = Q; ms.ns[m] = nsa; ms.ratio[m] = rat; ms.sdh[m] = sdh
    log_(f"[multimask] aligned {len(ms.masks)} masks on {ms.T} common epochs")
    return ms


# ===========================================================================
# Fusion config + selector (GT-FREE) — ported from proven R&D, defaults
# from multimask_ppk_2026-06-28/tuning.json "chosen".
# ===========================================================================
@dataclass
class FuseConfig:
    tau_m: float = 0.6          # consensus spread threshold (horizontal, m)
    stab_m: float = 0.4         # "stops moving" threshold between adjacent masks (m)
    window_s: float = 10.0      # segment length for smoothing the decision
    hysteresis_m: float = 0.3   # extra spread needed to LEAVE consensus once in it
    min_ns: int = 5             # geometry guard: min sources (when ns reported)
    min_ratio: float = 1.5      # geometry guard: min AR ratio (only if reported>0)
    max_sdh_m: float = 1.0      # geometry guard: max reported horiz sigma to trust
    sigma_worse_m: float = 0.1  # refuse escalation if higher mask sigma worse by >this
    require_better_sigma: bool = True
    # Disagreement flag threshold (horizontal spread, m). The environment noise flag
    # fires when the inter-mask spread exceeds this. Default == tau so the flag
    # and the consensus boundary coincide; tune separately if desired.
    flag_threshold_m: float = 0.6


# ---------------------------------------------------------------------------
# Spread = the GT-free environment noise signal (max pairwise horizontal distance)
# ---------------------------------------------------------------------------
def epoch_spread(ms: MaskSet) -> np.ndarray:
    """Per-epoch horizontal spread across masks = max pairwise E/N distance.

    Cached on the MaskSet so a tuning grid reuses it.
    """
    if ms._spread_cache is not None:
        return ms._spread_cache
    Es = np.vstack([ms.E[m] for m in ms.masks])     # (M,T)
    Ns = np.vstack([ms.N[m] for m in ms.masks])
    M, T = Es.shape
    spread = np.zeros(T)
    for a in range(M):
        for b in range(a + 1, M):
            de = Es[a] - Es[b]
            dn = Ns[a] - Ns[b]
            dist = np.hypot(de, dn)
            dist = np.where(np.isfinite(dist), dist, 0.0)
            spread = np.maximum(spread, dist)
    ms._spread_cache = spread
    return spread


def _geom_ok(ms: MaskSet, m: int, i: int, cfg: FuseConfig) -> bool:
    """Geometry guard using whichever quality signals the .pos carries.

    On device data the AR ``ratio`` column is 0 for almost every epoch
    (continuous-AR floats / Differential never report it) and ``ns`` is 0 on code-only
    Differential routes, so the guard is layered: prefer ns, fall back to reported
    sigma, and never trust a Q>=4 (Differential/single) epoch that a lower mask could
    have given as float.
    """
    if not np.isfinite(ms.E[m][i]):
        return False
    q = ms.Q[m][i]
    ns = ms.ns[m][i]
    r = ms.ratio[m][i]
    sdh = ms.sdh[m][i]
    if q == 1:                      # fix: always good geometry
        return True
    if q >= 4:                      # Differential / single / source-group: weak by definition
        return np.isfinite(sdh) and sdh <= cfg.max_sdh_m
    # float (q==2): require ns when present, else sigma cap
    if ns > 0 and ns < cfg.min_ns:
        return False
    if np.isfinite(r) and r > 0 and r < cfg.min_ratio:
        return False
    if np.isfinite(sdh) and sdh > cfg.max_sdh_m:
        return False
    return True


def _escalate_mask(ms: MaskSet, i: int, cfg: FuseConfig) -> int:
    """Pick the minimal mask whose neighbours have stabilised, with a guard
    that REFUSES to escalate when the higher mask reports worse geometry.

    1. find the lowest mask m where moving to m+1 changes the answer < stab_m
       (the environment noise low-elevation sources are gone above m),
    2. only accept escalating above the lowest mask if that higher mask's
       reported horizontal sigma is not materially worse (require_better_sigma)
       AND it passes the geometry guard,
    3. otherwise keep the lowest mask (escalation would only add noise).
    """
    masks = ms.masks
    low = masks[0]
    sdh_low = ms.sdh[low][i]

    chosen = masks[-1]
    for k in range(len(masks) - 1):
        m = masks[k]; mnext = masks[k + 1]
        e0, n0 = ms.E[m][i], ms.N[m][i]
        e1, n1 = ms.E[mnext][i], ms.N[mnext][i]
        if not (np.isfinite(e0) and np.isfinite(e1)):
            continue
        if math.hypot(e1 - e0, n1 - n0) < cfg.stab_m:
            chosen = m
            break

    if chosen != low and cfg.require_better_sigma:
        sdh_ch = ms.sdh[chosen][i]
        if (np.isfinite(sdh_low) and np.isfinite(sdh_ch)
                and sdh_ch > sdh_low + cfg.sigma_worse_m):
            chosen = low   # higher mask noisier -> starved geometry, not environment noise

    if _geom_ok(ms, chosen, i, cfg):
        return chosen
    # fall back to the lowest mask with acceptable geometry (most sources)
    for m in masks:
        if _geom_ok(ms, m, i, cfg):
            return m
    # nothing passes guard -> lowest available with a finite position
    for m in masks:
        if np.isfinite(ms.E[m][i]):
            return m
    return low


@dataclass
class FuseResult:
    ts: np.ndarray
    ref_llh: tuple[float, float, float]
    E: np.ndarray
    N: np.ndarray
    U: np.ndarray
    chosen_mask: np.ndarray          # per epoch
    spread: np.ndarray               # per epoch horizontal disagreement (m)
    escalated: np.ndarray            # bool per epoch (True = disagree branch)
    flag: np.ndarray                 # bool per epoch (spread >= flag_threshold)


def fuse(ms: MaskSet, cfg: Optional[FuseConfig] = None) -> FuseResult:
    """Per-segment fusion with hysteresis + geometry guard. GT-FREE.

    Consensus (spread < tau): pick the LOWEST mask (most sources).
    Disagreement (spread >= tau): escalate via :func:`_escalate_mask`.
    Segment windowing + a majority/hold rule stop the chosen mask flipping
    every epoch.
    """
    cfg = cfg or FuseConfig()
    T = ms.T
    spread = epoch_spread(ms)
    low = ms.masks[0]

    # 1) per-epoch raw decision with hysteresis on the consensus boundary
    raw_choice = np.zeros(T, dtype=int)
    raw_escal = np.zeros(T, dtype=bool)
    in_consensus = True
    for i in range(T):
        s = spread[i]
        thr = cfg.tau_m + (cfg.hysteresis_m if in_consensus else 0.0)
        if s < thr:
            in_consensus = True
            raw_choice[i] = low if _geom_ok(ms, low, i, cfg) else _escalate_mask(ms, i, cfg)
            raw_escal[i] = False
        else:
            in_consensus = False
            raw_choice[i] = _escalate_mask(ms, i, cfg)
            raw_escal[i] = True

    # 2) segment the timeline; majority-vote the mask per window with a hold
    seg_choice = raw_choice.copy()
    seg_escal = raw_escal.copy()
    ts = ms.ts
    i = 0
    last_mask: Optional[int] = None
    while i < T:
        j = i
        while j < T and (ts[j] - ts[i]) < cfg.window_s:
            j += 1
        seg = slice(i, j)
        vals = raw_choice[seg]
        uniq, cnt = np.unique(vals, return_counts=True)
        win_mask = int(uniq[np.argmax(cnt)])
        if last_mask is not None and win_mask != last_mask:
            if cnt.max() <= (j - i) * 0.5:
                win_mask = last_mask
        seg_choice[seg] = win_mask
        seg_escal[seg] = (win_mask != low)
        last_mask = win_mask
        i = j

    # 3) gather fused E/N/U from the chosen mask; fall back to nearest mask
    E = np.full(T, np.nan); N = np.full(T, np.nan); U = np.full(T, np.nan)
    for i in range(T):
        m = int(seg_choice[i])
        if np.isfinite(ms.E[m][i]):
            E[i], N[i], U[i] = ms.E[m][i], ms.N[m][i], ms.U[m][i]
        else:
            for mm in ms.masks:
                if np.isfinite(ms.E[mm][i]):
                    E[i], N[i], U[i] = ms.E[mm][i], ms.N[mm][i], ms.U[mm][i]
                    seg_choice[i] = mm
                    break

    flag = spread >= cfg.flag_threshold_m
    return FuseResult(ms.ts, ms.ref_llh, E, N, U,
                      seg_choice, spread, seg_escal, flag)


# ===========================================================================
# Output writers
# ===========================================================================
def write_disagreement_csv(ms: MaskSet, fr: FuseResult, out_csv: Path,
                           *, log: Optional[LogFn] = None) -> Path:
    """Write the per-epoch disagreement / flag table.

    Columns: ``reference time`` (UTC unix s), per-mask ``lat_m<NN>`` / ``lon_m<NN>``,
    ``disagreement_m`` (horizontal inter-mask spread), ``chosen_mask``,
    ``escalated``, ``flag`` (environment noise flag = spread >= threshold).
    """
    log_ = make_logger(log)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ref = ms.ref_llh
    hdr = ["gpstime"]
    for m in ms.masks:
        hdr += [f"lat_m{m:02d}", f"lon_m{m:02d}"]
    hdr += ["disagreement_m", "chosen_mask", "escalated", "flag"]
    lines = [",".join(hdr)]
    for i in range(ms.T):
        row = [f"{ms.ts[i]:.3f}"]
        for m in ms.masks:
            e, n, u = ms.E[m][i], ms.N[m][i], ms.U[m][i]
            if np.isfinite(e):
                lat, lon, _h = enu_to_llh(e, n, u, ref)
                row += [f"{lat:.9f}", f"{lon:.9f}"]
            else:
                row += ["", ""]
        row += [
            f"{fr.spread[i]:.4f}",
            str(int(fr.chosen_mask[i])),
            "1" if fr.escalated[i] else "0",
            "1" if fr.flag[i] else "0",
        ]
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_flag = int(np.count_nonzero(fr.flag))
    log_(f"[multimask] wrote disagreement CSV {out_csv.name} "
         f"({ms.T} epochs, {n_flag} flagged)")
    return out_csv


def write_fused_pos(fr: FuseResult, out_pos: Path,
                    *, log: Optional[LogFn] = None) -> Path:
    """Write the fused path as a solver-style ``.pos`` file.

    The column layout mirrors the standard subject .pos so any existing parser
    (``parse_rtkpos``, rtkplot, the viewers) reads it unchanged. The Q column
    encodes the fused decision: 1 (fix) where consensus, 2 (float) where the
    selector escalated on disagreement.
    """
    log_ = make_logger(log)
    out_pos = Path(out_pos)
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    ref = fr.ref_llh
    # The file is UTC-labelled (`% UTC ...`) so parse_rtkpos does NOT subtract
    # epoch offset — fr.ts (UTC unix seconds) round-trips exactly. The column
    # header token after `%` must be `UTC` for the parser to detect it.
    out: list[str] = [
        "% program   : data_pipeline.stages.multimask_ppk (fused)",
        "% fused trajectory from multi-elevation-mask PPK ensemble (GT-free)",
        "%  UTC                lat(deg)      lon(deg)     height(m)   Q  ns",
    ]
    for i in range(len(fr.ts)):
        e, n, u = fr.E[i], fr.N[i], fr.U[i]
        if not np.isfinite(e):
            continue
        lat, lon, h = enu_to_llh(e, n, u, ref)
        q = 2 if fr.escalated[i] else 1
        t = fr.ts[i]
        # Emit the UTC wall-clock of t so the round-trip is identity (ls=0).
        utc = _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
        stamp = utc.strftime("%Y/%m/%d %H:%M:%S.") + f"{utc.microsecond // 1000:03d}"
        out.append(
            f"{stamp}  {lat:14.9f} {lon:14.9f} {h:10.4f}  {q:1d}  "
            f"{int(fr.chosen_mask[i]):2d}"
        )
    out_pos.write_text("\n".join(out) + "\n", encoding="utf-8")
    log_(f"[multimask] wrote fused .pos {out_pos.name} "
         f"({sum(1 for v in fr.E if np.isfinite(v))} epochs)")
    return out_pos


# ===========================================================================
# Top-level result + orchestrator
# ===========================================================================
@dataclass
class MaskRun:
    mask: int
    pos_path: Path
    conf_path: Path
    n_rows: int
    returncode: int
    seconds: float
    err_tail: str = ""


@dataclass
class MultiMaskResult:
    masks: list[int]
    per_mask: dict[int, Path]            # mask -> .pos path (may be empty if failed)
    runs: list[MaskRun]
    fused_pos: Optional[Path]
    disagreement_csv: Optional[Path]
    report_html: Optional[Path]
    maskset: Optional[MaskSet]
    fuse_result: Optional[FuseResult]


def _count_solrows(pos: Path) -> int:
    if not pos.is_file():
        return 0
    n = 0
    with pos.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("%"):
                n += 1
    return n


def run_multimask_ppk(
    rover_obs: Path,
    base_obs: Path,
    nav: Sequence[Path],
    base_conf: Path,
    *,
    masks: Sequence[int] = DEFAULT_MASKS,
    workdir: Path,
    cfg: Optional[FuseConfig] = None,
    rnx2rtkp_exe: Optional[Path] = None,
    timeout_s: float = 3600.0,
    make_report: bool = True,
    log: Optional[LogFn] = None,
) -> MultiMaskResult:
    """Solve Post-processing at several elevation masks, then fuse GT-free.

    For each mask in ``masks`` a patched copy of ``base_conf`` (only
    ``pos1-elmask`` overridden) is run through ``the solver binary``; the per-mask
    ``.pos`` is cached in ``workdir``. All per-mask solutions are aligned on
    common epochs, the inter-mask DISAGREEMENT + environment noise flag are computed
    and written to a CSV, and a conservative geometry-guarded fused path
    is written to ``fused.pos``.

    Raises ``FileNotFoundError`` if ``the solver binary`` cannot be located. No ground
    truth is read at any point.
    """
    log_ = make_logger(log)
    rover_obs = Path(rover_obs)
    base_obs = Path(base_obs)
    base_conf = Path(base_conf)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    masks = sorted(set(int(m) for m in masks))
    navs = [Path(n) for n in nav]

    exe = _resolve_rnx2rtkp(rnx2rtkp_exe)
    if exe is None:
        raise FileNotFoundError(
            "rnx2rtkp executable not found. Pass rnx2rtkp_exe=... or set the "
            "RNX2RTKP env var / install vendor/rtklib/rnx2rtkp.exe."
        )
    log_(f"[multimask] exe = {exe}")
    log_(f"[multimask] masks = {masks}")
    log_(f"[multimask] rover = {rover_obs.name}  base = {base_obs.name}")
    log_(f"[multimask] nav = {[n.name for n in navs]}")

    runs: list[MaskRun] = []
    per_mask: dict[int, Path] = {}
    for m in masks:
        conf = workdir / f"mask{m:02d}.conf"
        pos = workdir / f"mask{m:02d}.pos"
        patch_elmask(base_conf, conf, m, log=log_)
        cmd = [
            _win_abs(exe), "-k", _win_abs(conf), "-o", _win_abs(pos),
            _win_abs(rover_obs), _win_abs(base_obs),
        ] + [_win_abs(n) for n in navs]
        log_(f"[multimask] mask {m:2d}: {' '.join(cmd)}")
        t0 = time.time()
        rc = -1
        err_tail = ""
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, check=False,
            )
            rc = proc.returncode
            err_tail = "\n".join(
                ln for ln in (proc.stderr or "").splitlines() if ln.strip()
            )[-300:]
        except subprocess.TimeoutExpired:
            rc = -999
            err_tail = "TIMEOUT"
        dt = time.time() - t0
        n_rows = _count_solrows(pos)
        runs.append(MaskRun(m, pos, conf, n_rows, rc, round(dt, 1), err_tail))
        if n_rows > 0:
            per_mask[m] = pos
        log_(f"[multimask] mask {m:2d}: rc={rc} rows={n_rows} ({dt:.0f}s)"
             + ("" if n_rows else f" NO SOLUTION {err_tail}"))

    # Align + fuse (GT-free). Skip gracefully if too few masks solved.
    ms = build_maskset_from_posfiles(per_mask, log=log_)
    fused_pos: Optional[Path] = None
    disagreement_csv: Optional[Path] = None
    report_html: Optional[Path] = None
    fr: Optional[FuseResult] = None
    if ms is not None:
        fr = fuse(ms, cfg)
        disagreement_csv = write_disagreement_csv(
            ms, fr, workdir / "disagreement.csv", log=log_)
        fused_pos = write_fused_pos(fr, workdir / "fused.pos", log=log_)
        if make_report:
            try:
                report_html = build_multimask_report(
                    ms, fr, workdir / "multimask_ppk_report.html", log=log_)
            except Exception as e:   # report is best-effort, never fatal
                log_(f"[multimask] report skipped: {type(e).__name__}: {e}")
    else:
        log_("[multimask] fusion skipped: <2 masks with common epochs")

    return MultiMaskResult(
        masks=masks, per_mask=per_mask, runs=runs,
        fused_pos=fused_pos, disagreement_csv=disagreement_csv,
        report_html=report_html, maskset=ms, fuse_result=fr,
    )


# ===========================================================================
# Optional HTML report (per-mask overlay + disagreement timeline)
# ===========================================================================
def build_multimask_report(ms: MaskSet, fr: FuseResult, out_html: Path,
                           *, log: Optional[LogFn] = None) -> Path:
    """Self-contained Plotly report: per-mask E/N overlay + disagreement
    timeline with the environment noise flag shaded. Reuses ``_copy_plotly_next_to``
    so it works air-gapped.
    """
    log_ = make_logger(log)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .viewers import _copy_plotly_next_to
        _copy_plotly_next_to(out_html.parent)
        plotly_src = "plotly.min.js"
    except Exception:
        plotly_src = "https://cdn.plot.ly/plotly-2.27.0.min.js"

    import json
    traces_xy = []
    for m in ms.masks:
        traces_xy.append({
            "x": [None if not np.isfinite(v) else float(v) for v in ms.E[m]],
            "y": [None if not np.isfinite(v) else float(v) for v in ms.N[m]],
            "mode": "lines", "name": f"mask {m}",
        })
    # fused path
    traces_xy.append({
        "x": [None if not np.isfinite(v) else float(v) for v in fr.E],
        "y": [None if not np.isfinite(v) else float(v) for v in fr.N],
        "mode": "lines", "name": "FUSED",
        "line": {"width": 3, "color": "black"},
    })
    t0 = float(ms.ts[0]) if ms.T else 0.0
    rel_t = [float(t - t0) for t in ms.ts]
    spread_trace = {
        "x": rel_t, "y": [float(v) for v in fr.spread],
        "mode": "lines", "name": "disagreement (m)",
    }
    flag_x = [rel_t[i] for i in range(ms.T) if fr.flag[i]]
    flag_y = [float(fr.spread[i]) for i in range(ms.T) if fr.flag[i]]
    flag_trace = {
        "x": flag_x, "y": flag_y, "mode": "markers",
        "name": "multipath flag", "marker": {"color": "red", "size": 5},
    }
    payload = {
        "xy": traces_xy,
        "spread": [spread_trace, flag_trace],
        "thr": float(getattr(fr, "_flag_thr", 0.6)),
    }
    n_flag = int(np.count_nonzero(fr.flag))
    n_esc = int(np.count_nonzero(fr.escalated))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Multi-mask PPK report</title>
<script src="{plotly_src}"></script>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:16px;background:#0e1525;color:#e2e8f0}}
h1{{color:#7dd3fc}} .stat{{color:#94a3b8}}</style></head><body>
<h1>Multi-elevation-mask PPK — GT-free disagreement &amp; fusion</h1>
<p class="stat">{ms.T} common epochs · masks {ms.masks} ·
{n_flag} epochs flagged (multipath) · {n_esc} epochs escalated by selector.</p>
<div id="xy" style="height:520px"></div>
<div id="sp" style="height:340px"></div>
<script>
var D = {json.dumps(payload)};
Plotly.newPlot('xy', D.xy, {{title:'Per-mask ENU overlay (E vs N, m) + FUSED',
  xaxis:{{title:'East (m)'}}, yaxis:{{title:'North (m)', scaleanchor:'x'}},
  paper_bgcolor:'#0e1525', plot_bgcolor:'#0a1020', font:{{color:'#e2e8f0'}}}});
Plotly.newPlot('sp', D.spread, {{title:'Inter-mask disagreement timeline (GT-free multipath signal)',
  xaxis:{{title:'time since start (s)'}}, yaxis:{{title:'horizontal spread (m)'}},
  paper_bgcolor:'#0e1525', plot_bgcolor:'#0a1020', font:{{color:'#e2e8f0'}}}});
</script></body></html>"""
    out_html.write_text(html, encoding="utf-8")
    log_(f"[multimask] wrote report {out_html.name}")
    return out_html


# ===========================================================================
# CLI
# ===========================================================================
def _parse_masks(s: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Multi-elevation-mask PPK ensemble + GT-free fusion.")
    ap.add_argument("--rover", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path, nargs="+")
    ap.add_argument("--conf", required=True, type=Path,
                    help="Base RTKLIB .conf; only pos1-elmask is swept.")
    ap.add_argument("--masks", type=_parse_masks, default=list(DEFAULT_MASKS),
                    help="Comma/space separated elevation masks, e.g. 5,10,15,20,25,30")
    ap.add_argument("--out", required=True, type=Path, help="Output / work dir.")
    ap.add_argument("--rnx2rtkp", type=Path, default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    res = run_multimask_ppk(
        args.rover, args.base, args.nav, args.conf,
        masks=args.masks, workdir=args.out,
        rnx2rtkp_exe=args.rnx2rtkp, make_report=not args.no_report, log=_log,
    )
    print(f"\nper-mask .pos: {[str(p) for p in res.per_mask.values()]}")
    print(f"fused .pos:     {res.fused_pos}")
    print(f"disagreement:   {res.disagreement_csv}")
    print(f"report:         {res.report_html}")
    return 0 if res.fused_pos is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
