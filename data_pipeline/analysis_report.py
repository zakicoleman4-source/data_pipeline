"""One-file "Analysis" report for a Post-processing solution and its raw observations.

Given a subject ``.pos`` (required) and optionally the subject ``.obs``, base
``.obs`` and a ground-truth ``.pos``, :func:`build_analysis_report` writes a
single self-contained HTML file (plotly.min.js inlined, opens offline) with
simple client-readable panels:

1. Solution overview  — The external solver config readout + Q-quality distribution.
2. Sources         — ns over time + histogram, average; sources SEEN
                        in the raw obs (subject vs base) over time.
3. SNR                — average signal strength per source group
                        (subject vs base when both .obs given) + over time.
4. Fine measurements      — phase / NO-phase badge per source group.
5. Noise / precision  — sd_n / sd_e / sd_u over time + horizontal sigma.
6. Predicted accuracy — accuracy_predictor session sigma + per-epoch CDF;
                        when a ground-truth .pos is given, real error CDF and
                        error-vs-source-count scatter with bucket stats.

Optional inputs that are missing simply omit their panel with a short note —
the report never fails because an .obs was not kept.
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .accuracy_predictor import predicted_epoch_std, smart_session_std
from .obs_check import ObsSummary, check_carrier_phase, summarize_obs
from .parsers import PosRow, parse_pos_header, parse_rtkpos

LogFn = Callable[[str], None]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# The external solver Q flag -> label / colour (colours match usual The external solver plot habits).
_Q_LABELS = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}
_Q_COLORS = {
    1: "#2ca02c",   # green
    2: "#ff9900",   # orange
    3: "#17becf",
    4: "#d62ca8",   # magenta
    5: "#d62728",   # red
    6: "#1f77b4",
}

_SYS_NAMES = {
    "G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou",
    "J": "QZSS", "I": "NavIC", "S": "SBAS",
}

# Max points kept in any time-series trace; longer sessions are strided so
# the single-file HTML stays a reasonable size.
_MAX_TS_POINTS = 20000

# Nearest-time match window (s) for subject <-> ground-truth pairing.
_GT_MATCH_DT_S = 0.5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _load_plotly_js() -> str:
    """Vendored plotly.min.js source for inlining (offline single file)."""
    for cand in (_ASSETS_DIR / "plotly.min.js",
                 Path(__file__).resolve().parent / "plotly.min.js"):
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"plotly.min.js not found under {_ASSETS_DIR}. Re-install the package."
    )


def _jclean(seq: Sequence) -> list:
    """NaN/inf -> None so json.dumps emits null (plotly renders a gap)."""
    out = []
    for v in seq:
        if isinstance(v, float) and not math.isfinite(v):
            out.append(None)
        elif isinstance(v, (np.floating, np.integer)):
            f = float(v)
            out.append(f if math.isfinite(f) else None)
        else:
            out.append(v)
    return out


def _stride(n: int) -> int:
    return max(1, (n + _MAX_TS_POINTS - 1) // _MAX_TS_POINTS)


def _iso_times(utc_seconds: Sequence[float]) -> list:
    """UTC seconds -> 'YYYY-mm-dd HH:MM:SS.f' strings plotly reads natively."""
    out = []
    for t in utc_seconds:
        if t is None or (isinstance(t, float) and not math.isfinite(t)):
            out.append(None)
            continue
        d = dt.datetime.fromtimestamp(float(t), tz=dt.timezone.utc)
        out.append(d.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    return out


def _esc(v: object) -> str:
    return _html.escape(str(v))


def _fig(div_id: str, traces: list, layout: dict, height: int = 340) -> str:
    """A plotly chart: div + the newPlot call, data inlined as JS literals."""
    base_layout = {
        "margin": {"l": 55, "r": 20, "t": 36, "b": 45},
        "height": height,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#fafbfc",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 12},
        "legend": {"orientation": "h", "y": -0.22},
    }
    base_layout.update(layout)
    return (
        f'<div id="{div_id}" class="chart"></div>\n'
        f"<script>Plotly.newPlot({json.dumps(div_id)}, "
        f"{json.dumps(traces)}, {json.dumps(base_layout)}, "
        '{responsive:true, displaylogo:false});</script>\n'
    )


def _section(title: str, body: str, explainer: str = "") -> str:
    expl = f'<p class="explain">{explainer}</p>' if explainer else ""
    return (
        f'<section class="card"><h2>{_esc(title)}</h2>{expl}{body}</section>\n'
    )


def _note(text: str) -> str:
    return f'<p class="muted-note">{_esc(text)}</p>'


def _badge(text: str, kind: str) -> str:
    """kind: good / warn / bad / info."""
    return f'<span class="badge badge-{kind}">{_esc(text)}</span>'


def _cdf_xy(values: np.ndarray) -> tuple[list, list]:
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return [], []
    y = (np.arange(1, v.size + 1) / v.size)
    return v.tolist(), y.tolist()


# ---------------------------------------------------------------------------
# panel builders
# ---------------------------------------------------------------------------


def _headline(rows: list[PosRow]) -> tuple[str, str]:
    """(badge_text, badge_kind) for the top of the report."""
    qs = np.array([r.quality for r in rows], int)
    n = len(qs)
    counts = {q: int(np.sum(qs == q)) for q in sorted(set(qs.tolist()))}
    dominant = max(counts, key=counts.get)
    pct = 100.0 * counts[dominant] / n
    if all(q >= 4 for q in counts):
        if dominant == 4:
            return ("DGPS-only (Q=4) — no carrier phase", "bad")
        return (f"{_Q_LABELS.get(dominant, f'Q={dominant}')}-only — "
                "no differential carrier solution", "bad")
    label = _Q_LABELS.get(dominant, f"Q={dominant}")
    kind = "good" if dominant == 1 else ("warn" if dominant == 2 else "bad")
    return (f"Solved: {label} ({pct:.0f}%)", kind)


def _panel_overview(pos_path: Path, rows: list[PosRow]) -> str:
    hdr = parse_pos_header(pos_path)
    n = len(rows)
    dur_s = rows[-1].utc_s - rows[0].utc_s if n >= 2 else 0.0

    def _fmt(v: object) -> str:
        if v is None:
            return "(not in header)"
        if isinstance(v, tuple):
            return " ".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                            for x in v)
        return str(v)

    fields = [
        ("Positioning mode", hdr.pos_mode),
        ("Frequencies", hdr.freqs),
        ("Elevation mask", None if hdr.elev_mask_deg is None
         else f"{hdr.elev_mask_deg:g} deg"),
        ("Ambiguity resolution", hdr.amb_res),
        ("Navigation systems", hdr.nav_sys),
        ("Validation threshold", hdr.val_thres),
        ("Time system", hdr.time_system),
        ("Coordinate format", hdr.solformat),
        ("Reference position", hdr.ref_pos),
        ("Obs start", hdr.obs_start),
        ("Obs end", hdr.obs_end),
        ("Duration", f"{dur_s/60.0:.1f} min ({dur_s:.0f} s)"),
        ("Epochs in solution", n),
    ]
    trs = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(_fmt(v))}</td></tr>"
        for k, v in fields
    )
    table = f'<table class="kv">{trs}</table>'

    qs = np.array([r.quality for r in rows], int)
    q_order = [q for q in (1, 2, 3, 4, 5, 6) if np.any(qs == q)]
    labels = [_Q_LABELS.get(q, f"Q={q}") for q in q_order]
    pcts = [100.0 * float(np.mean(qs == q)) for q in q_order]
    colors = [_Q_COLORS.get(q, "#777") for q in q_order]
    pie = _fig(
        "q-dist",
        [{
            "type": "pie",
            "labels": labels,
            "values": [round(p, 2) for p in pcts],
            "marker": {"colors": colors},
            "textinfo": "label+percent",
            "hole": 0.45,
            "sort": False,
        }],
        {"title": {"text": "Q-distribution (solution quality share)"}},
        height=320,
    )
    q_txt = "&nbsp;&nbsp;".join(
        f"{_esc(lbl)}: <b>{p:.1f}%</b>" for lbl, p in zip(labels, pcts)
    )
    body = (
        f'<div class="two-col"><div>{table}</div><div>{pie}'
        f'<p class="stat-line">{q_txt}</p></div></div>'
    )
    return _section(
        "1 · Solution overview", body,
        "What RTKLIB was configured to do, and what share of epochs reached "
        "each quality level. Q=1 FIX is centimetre-grade, Q=2 FLOAT is "
        "decimetre-to-metre, Q=4 DGPS and Q=5 SINGLE are metre-grade code "
        "solutions.",
    )


def _panel_satellites(rows: list[PosRow],
                      rover_sum: Optional[ObsSummary] = None,
                      base_sum: Optional[ObsSummary] = None) -> str:
    ns = [int(r.ns) for r in rows]
    avg_ns = float(np.mean(ns)) if ns else float("nan")
    st = _stride(len(rows))
    times = _iso_times([r.utc_s for r in rows[::st]])
    line = _fig(
        "ns-time",
        [{
            "type": "scatter", "mode": "lines",
            "x": times, "y": _jclean(ns[::st]),
            "line": {"color": "#1f77b4", "width": 1.5},
            "name": "satellites used",
        }],
        {"title": {"text": "Satellites used per epoch (ns)"},
         "yaxis": {"title": {"text": "ns"}, "rangemode": "tozero"}},
    )
    hist = _fig(
        "ns-hist",
        [{
            "type": "histogram", "x": ns,
            "marker": {"color": "#1f77b4"},
            "xbins": {"size": 1},
        }],
        {"title": {"text": "Distribution of ns"},
         "xaxis": {"title": {"text": "satellites used"}},
         "yaxis": {"title": {"text": "epochs"}}},
        height=300,
    )
    # Sources SEEN in the raw observations (subject vs base), overlaid on
    # the same clock axis as the ns chart (both are absolute epoch seconds).
    seen_traces = []
    seen_stats = []
    for name, s, color in (("Rover", rover_sum, "#1f77b4"),
                           ("Base", base_sum, "#2ca02c")):
        if s is None or not s.times_s or not s.sats_per_epoch:
            continue
        st_o = _stride(len(s.times_s))
        seen_traces.append({
            "type": "scatter", "mode": "lines",
            "name": f"{name} (seen)",
            "x": _iso_times(s.times_s[::st_o]),
            "y": _jclean(s.sats_per_epoch[::st_o]),
            "line": {"color": color, "width": 1.5},
        })
        if math.isfinite(s.avg_sats_per_epoch):
            seen_stats.append(
                f"{name} seen: <b>{s.avg_sats_per_epoch:.1f}</b> sats/epoch"
            )
    if seen_traces:
        seen_chart = _fig(
            "sats-seen-time", seen_traces,
            {"title": {"text": "Satellites seen (raw observations) — "
                               "rover vs base"},
             "yaxis": {"title": {"text": "satellites seen"},
                       "rangemode": "tozero"}},
        )
        seen_note = ""
        if rover_sum is None or not rover_sum.times_s:
            seen_note = _note("Rover .obs not provided — rover line omitted.")
        elif base_sum is None or not base_sum.times_s:
            seen_note = _note("Base .obs not provided — base line omitted.")
        stat_txt = (f'<p class="stat-line">{" &nbsp;|&nbsp; ".join(seen_stats)}'
                    "</p>") if seen_stats else ""
        seen_html = (
            "<h3>Satellites seen vs used</h3>"
            f"{stat_txt}{seen_note}{seen_chart}"
            '<p class="explain">"Seen" counts every satellite present in the '
            "raw .obs at each epoch; ns above counts only those the solver "
            "actually USED (ns &le; seen — low-elevation, SNR-masked and "
            "cycle-slipped satellites are dropped).</p>"
        )
    else:
        seen_html = _note(
            "No .obs file provided — satellites-seen (raw observations) "
            "chart omitted; the ns chart above only needs the .pos."
        )

    body = (
        f'<p class="stat-line">Average satellites used: '
        f"<b>{avg_ns:.1f}</b> &nbsp; (min {min(ns) if ns else 0}, "
        f"max {max(ns) if ns else 0})</p>"
        f'<div class="two-col"><div>{line}</div><div>{hist}</div></div>'
        f"{seen_html}"
    )
    return _section(
        "2 · Satellites", body,
        "How many satellites contributed to each solution epoch. Sustained "
        "drops usually mean obstruction (trees, buildings) and line up with "
        "noisier positions.",
    )


def _panel_snr(rover_sum: Optional[ObsSummary],
               base_sum: Optional[ObsSummary]) -> str:
    if rover_sum is None and base_sum is None:
        return _section(
            "3 · Signal strength (SNR)",
            _note("No .obs file provided — SNR panel omitted. Point the "
                  "report at the rover (and optionally base) RINEX .obs to "
                  "see signal-strength statistics."),
        )

    sys_keys: list[str] = []
    for s in (rover_sum, base_sum):
        if s is not None:
            for k in s.snr_per_system:
                if k not in sys_keys:
                    sys_keys.append(k)
    sys_keys.sort()
    x_labels = [_SYS_NAMES.get(k, k) for k in sys_keys] + ["ALL"]

    traces = []
    for name, s, color in (("rover", rover_sum, "#1f77b4"),
                           ("base", base_sum, "#2ca02c")):
        if s is None:
            continue
        ys = [s.snr_per_system.get(k) for k in sys_keys] + [
            s.avg_snr_db if math.isfinite(s.avg_snr_db) else None]
        traces.append({
            "type": "bar", "name": name.upper(),
            "x": x_labels,
            "y": _jclean([None if y is None else round(float(y), 2)
                          for y in ys]),
            "marker": {"color": color},
        })
    bars = _fig(
        "snr-bars", traces,
        {"title": {"text": "Average SNR by constellation (dB-Hz)"},
         "yaxis": {"title": {"text": "dB-Hz"}, "rangemode": "tozero"},
         "barmode": "group"},
    )

    # SNR over time (cheap: per-epoch mean already computed in the scan).
    ts_traces = []
    for name, s, color in (("rover", rover_sum, "#1f77b4"),
                           ("base", base_sum, "#2ca02c")):
        if s is None or not s.times_s:
            continue
        st = _stride(len(s.times_s))
        ts_traces.append({
            "type": "scatter", "mode": "lines",
            "name": f"{name.upper()} avg SNR",
            "x": _iso_times(s.times_s[::st]),
            "y": _jclean([None if not math.isfinite(v) else round(v, 2)
                          for v in s.snr_per_epoch[::st]]),
            "line": {"color": color, "width": 1.2},
        })
    ts = _fig(
        "snr-time", ts_traces,
        {"title": {"text": "Average SNR over time (dB-Hz)"},
         "yaxis": {"title": {"text": "dB-Hz"}}},
        height=300,
    ) if ts_traces else ""

    stat_bits = []
    for name, s in (("Rover", rover_sum), ("Base", base_sum)):
        if s is None:
            continue
        snr_txt = (f"{s.avg_snr_db:.1f} dB-Hz"
                   if math.isfinite(s.avg_snr_db) else "no SNR data")
        intv = (f" @ {s.interval_s:g} s" if math.isfinite(s.interval_s) else "")
        stat_bits.append(
            f"{name}: <b>{snr_txt}</b> overall, "
            f"{s.avg_sats_per_epoch:.1f} sats/epoch, "
            f"{s.epoch_count} epochs{intv}"
        )
    missing = ""
    if rover_sum is None:
        missing = _note("Rover .obs not provided — rover SNR omitted.")
    elif base_sum is None:
        missing = _note("Base .obs not provided — base SNR omitted.")

    body = (
        f'<p class="stat-line">{" &nbsp;|&nbsp; ".join(stat_bits)}</p>'
        f"{missing}{bars}{ts}"
    )
    return _section(
        "3 · Signal strength (SNR)", body,
        "Carrier-to-noise density (dB-Hz) reported by the receiver. Above "
        "~40 dB-Hz is strong; a rover far below the base indicates antenna "
        "or sky-view problems on the device side.",
    )


def _panel_phase(rover_obs: Optional[Path], phase_report) -> str:
    if rover_obs is None or phase_report is None:
        return _section(
            "4 · Carrier phase",
            _note("Rover .obs not provided — carrier-phase check omitted."),
        )
    rep = phase_report
    rows_html = ""
    for sys, st in sorted(rep.per_system.items()):
        n_obs = st.get("n_sat_obs", 0)
        n_nz = st.get("n_phase_nonzero", 0)
        frac = (100.0 * n_nz / n_obs) if n_obs else 0.0
        ok = n_obs > 0 and (n_nz / n_obs) >= 0.01
        b = _badge("phase", "good") if ok else _badge("NO phase", "bad")
        types = " ".join(st.get("phase_types", []))
        rows_html += (
            f"<tr><td>{_esc(_SYS_NAMES.get(sys, sys))}</td>"
            f"<td>{_esc(types)}</td>"
            f"<td>{n_nz:,} / {n_obs:,} ({frac:.1f}%)</td><td>{b}</td></tr>"
        )
    table = (
        '<table class="kv"><tr><th>Constellation</th><th>Phase observables'
        "</th><th>Non-zero phase slots</th><th>Status</th></tr>"
        f"{rows_html}</table>"
    )
    if rep.has_phase:
        head = (
            f'<p class="stat-line">{_badge("carrier phase OK", "good")} '
            f"{rep.n_phase_nonzero:,} of {rep.n_sat_obs:,} phase slots carry "
            "a value — RTK/PPK ambiguity resolution is possible.</p>"
        )
    else:
        head = (
            '<div class="alert">'
            f'{_badge("NO carrier phase", "bad")} '
            "This observation file has no usable carrier phase: RTK/PPK is "
            "impossible, so the solution degrades to code DGPS (Q=4, "
            "metre-level). Likely GNSS duty-cycling was on / ADR invalid. "
            'Recapture with "Force full GNSS measurements" enabled.</div>'
        )
    return _section(
        "4 · Carrier phase", head + table,
        "RTK/PPK needs carrier-phase (L*) observations. Devices with GNSS "
        "duty-cycling write every phase value as 0.000 — pseudorange only.",
    )


def _panel_noise(rows: list[PosRow]) -> str:
    sd_h = np.array([
        math.hypot(r.sd_n, r.sd_e)
        if (math.isfinite(r.sd_n) and math.isfinite(r.sd_e)) else float("nan")
        for r in rows
    ])
    if not np.isfinite(sd_h).any():
        return _section(
            "5 · Noise / precision",
            _note("This .pos carries no sd_n/sd_e/sd_u columns — noise panel "
                  "omitted (enable full solution output in RTKLIB)."),
        )
    med_h = float(np.nanmedian(sd_h))
    p95_h = float(np.nanpercentile(sd_h, 95))
    if med_h < 0.05:
        verdict = "very quiet — centimetre-level reported precision"
    elif med_h < 0.30:
        verdict = "low noise — decimetre-level reported precision"
    elif med_h < 1.0:
        verdict = "moderate noise — sub-metre reported precision"
    else:
        verdict = "noisy — metre-level reported precision"

    st = _stride(len(rows))
    times = _iso_times([r.utc_s for r in rows[::st]])
    traces = []
    for attr, name, color in (("sd_n", "sd_north", "#1f77b4"),
                              ("sd_e", "sd_east", "#ff7f0e"),
                              ("sd_u", "sd_up", "#2ca02c")):
        traces.append({
            "type": "scatter", "mode": "lines", "name": name,
            "x": times,
            "y": _jclean([getattr(r, attr) for r in rows[::st]]),
            "line": {"color": color, "width": 1.1},
        })
    chart = _fig(
        "sd-time", traces,
        {"title": {"text": "Reported 1-sigma per axis (m)"},
         "yaxis": {"title": {"text": "sigma (m)"}, "rangemode": "tozero"}},
    )
    body = (
        f'<p class="stat-line">Average horizontal sigma '
        f"(median of &radic;(sd_n&sup2;+sd_e&sup2;)): <b>{med_h:.3f} m</b> "
        f"&nbsp; p95: <b>{p95_h:.3f} m</b> &nbsp;&mdash;&nbsp; "
        f"<i>{_esc(verdict)}</i></p>{chart}"
    )
    return _section(
        "5 · Noise / precision", body,
        "RTKLIB's own per-epoch uncertainty estimate. It describes solution "
        "noise, not absolute bias — see the predicted-accuracy panel for an "
        "honest error envelope.",
    )


def _match_gt(rows: list[PosRow], gt_rows: list[PosRow],
              max_dt: float = _GT_MATCH_DT_S) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-time subject<->GT pairing.

    Returns (horizontal_error_m, ns) arrays for the matched subject epochs.
    """
    gt_t = np.array([g.utc_s for g in gt_rows])
    order = np.argsort(gt_t)
    gt_t = gt_t[order]
    gt_lat = np.array([g.lat_deg for g in gt_rows])[order]
    gt_lon = np.array([g.lon_deg for g in gt_rows])[order]

    lat0 = math.radians(float(np.median([r.lat_deg for r in rows])))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat0)

    errs, nss = [], []
    for r in rows:
        i = int(np.searchsorted(gt_t, r.utc_s))
        best, best_dt = -1, max_dt
        for j in (i - 1, i):
            if 0 <= j < gt_t.size:
                d = abs(gt_t[j] - r.utc_s)
                if d <= best_dt:
                    best, best_dt = j, d
        if best < 0:
            continue
        dn = (r.lat_deg - gt_lat[best]) * m_per_deg_lat
        de = (r.lon_deg - gt_lon[best]) * m_per_deg_lon
        errs.append(math.hypot(dn, de))
        nss.append(int(r.ns))
    return np.array(errs), np.array(nss)


def _panel_predicted(rows: list[PosRow], gt_pos: Optional[Path],
                     log_: LogFn) -> tuple[str, dict]:
    """Predicted-accuracy panel (+ GT validation when a truth .pos is given).

    Returns (html, stats) where stats carries headline numbers for the log.
    """
    stats: dict = {}
    profile = smart_session_std(rows)
    eps = predicted_epoch_std(rows, profile)
    stats["session_sigma_m"] = profile.smart_std_m
    stats["trust_class"] = profile.trust_class

    x, y = _cdf_xy(eps)
    cdf = _fig(
        "pred-cdf",
        [{
            "type": "scatter", "mode": "lines",
            "x": _jclean([round(v, 3) for v in x]), "y": _jclean(y),
            "line": {"color": "#1f77b4", "width": 2},
            "name": "predicted sigma CDF",
        }],
        {"title": {"text": "Fraction of epochs with predicted horizontal "
                           "sigma below X"},
         "xaxis": {"title": {"text": "predicted horizontal 1-sigma (m)"}},
         "yaxis": {"title": {"text": "fraction of epochs"},
                   "range": [0, 1.02]}},
    )
    trust_note = {
        "trustworthy": ("good", "sigma envelope validated as reliable"),
        "tight": ("warn", "typical float session — treat sigma as a "
                          "~2x envelope"),
        "spike_risk": ("bad", "occasional outlier epochs may exceed the "
                              "predicted bounds"),
    }.get(profile.trust_class, ("info", ""))
    head = (
        f'<p class="stat-line">Session-level predicted horizontal 1-sigma: '
        f"<b>{profile.smart_std_m:.2f} m</b> &nbsp; "
        f"(2-sigma: {2 * profile.smart_std_m:.2f} m) &nbsp; "
        f"{_badge(profile.trust_class, trust_note[0])} "
        f"<i>{_esc(trust_note[1])}</i></p>"
    )

    gt_html = ""
    if gt_pos is not None:
        try:
            gt_rows = parse_rtkpos(Path(gt_pos))
        except (ValueError, RuntimeError, OSError) as ex:
            gt_rows = []
            gt_html = _note(f"Ground-truth .pos could not be parsed ({ex}) "
                            "— truth comparison omitted.")
        if gt_rows:
            errs, nss = _match_gt(rows, gt_rows)
            if errs.size == 0:
                gt_html = _note(
                    "No rover epoch matched the ground truth within "
                    f"{_GT_MATCH_DT_S} s — check the two files overlap in "
                    "time. Truth comparison omitted."
                )
            else:
                med = float(np.median(errs))
                p95 = float(np.percentile(errs, 95))
                stats["gt_matched"] = int(errs.size)
                stats["gt_err_med_m"] = med
                stats["gt_err_p95_m"] = p95
                ex_, ey = _cdf_xy(errs)
                ecdf = _fig(
                    "gt-cdf",
                    [{
                        "type": "scatter", "mode": "lines",
                        "x": _jclean([round(v, 3) for v in ex_]),
                        "y": _jclean(ey),
                        "line": {"color": "#d62728", "width": 2},
                        "name": "measured error CDF",
                    }],
                    {"title": {"text": "Fraction of epochs with horizontal "
                                       "error below X (vs ground truth)"},
                     "xaxis": {"title": {"text": "horizontal error (m)"}},
                     "yaxis": {"title": {"text": "fraction of epochs"},
                               "range": [0, 1.02]}},
                )
                # error vs ns scatter + per-ns-bucket aggregates.
                buckets = {}
                for e, s in zip(errs.tolist(), nss.tolist()):
                    buckets.setdefault(s, []).append(e)
                bx = sorted(k for k, v in buckets.items() if len(v) >= 5)
                bmed = [float(np.median(buckets[k])) for k in bx]
                bp95 = [float(np.percentile(buckets[k], 95)) for k in bx]
                sc_traces = [
                    {
                        "type": "scatter", "mode": "markers",
                        "x": _jclean(nss.tolist()),
                        "y": _jclean([round(float(e), 3)
                                      for e in errs.tolist()]),
                        "marker": {"color": "#888", "size": 4,
                                   "opacity": 0.35},
                        "name": "per-epoch error",
                    },
                ]
                if bx:
                    sc_traces += [
                        {"type": "scatter", "mode": "lines+markers",
                         "x": bx, "y": _jclean([round(v, 3) for v in bmed]),
                         "line": {"color": "#1f77b4", "width": 2},
                         "name": "median per ns"},
                        {"type": "scatter", "mode": "lines+markers",
                         "x": bx, "y": _jclean([round(v, 3) for v in bp95]),
                         "line": {"color": "#d62728", "width": 2,
                                  "dash": "dot"},
                         "name": "p95 per ns"},
                    ]
                sc = _fig(
                    "gt-ns",
                    sc_traces,
                    {"title": {"text": "Horizontal error vs satellites used"},
                     "xaxis": {"title": {"text": "satellites used (ns)"}},
                     "yaxis": {"title": {"text": "horizontal error (m)"},
                               "rangemode": "tozero"}},
                )
                gt_html = (
                    '<h3>Measured vs ground truth</h3>'
                    f'<p class="stat-line">Matched epochs: <b>{errs.size}</b>'
                    f" &nbsp; median error: <b>{med:.2f} m</b> &nbsp; "
                    f"p95: <b>{p95:.2f} m</b></p>"
                    f'<div class="two-col"><div>{ecdf}</div><div>{sc}</div>'
                    "</div>"
                )
                log_(f"[analysis] GT match: {errs.size} epochs, "
                     f"median {med:.2f} m, p95 {p95:.2f} m")
    else:
        gt_html = _note("No ground-truth .pos provided — the predicted "
                        "sigma above is the honest session estimate without "
                        "truth validation.")

    body = head + cdf + gt_html
    return _section(
        "6 · Predicted accuracy", body,
        "A calibrated per-session error model (noise + ambiguity bias + "
        "spike terms, quality-aware floors) — more honest than the raw "
        "RTKLIB sigma, which reports noise only.",
    ), stats


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0;
       background: #eef1f5; color: #1c2733; }
.wrap { max-width: 1150px; margin: 0 auto; padding: 18px; }
h1 { font-size: 1.5em; margin: 8px 0 2px; }
h2 { font-size: 1.15em; margin: 0 0 6px; color: #21344a; }
h3 { font-size: 1.0em; margin: 14px 0 4px; color: #21344a; }
.card { background: #fff; border-radius: 10px; padding: 16px 18px;
        margin: 14px 0; box-shadow: 0 1px 4px rgba(20,40,70,.08); }
.explain { color: #5c6b7c; font-size: .9em; margin: 2px 0 10px; }
.stat-line { font-size: 1.0em; margin: 6px 0; }
.muted-note { color: #7a8794; font-style: italic; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-weight: 600; font-size: .85em; }
.badge-good { background: #e3f5e5; color: #197a24; }
.badge-warn { background: #fff3d6; color: #8a6100; }
.badge-bad  { background: #fde3e3; color: #a11616; }
.badge-info { background: #e3edfa; color: #1c4d8f; }
.badge-headline { font-size: 1.05em; padding: 6px 16px; }
.alert { background: #fde3e3; border-left: 4px solid #a11616;
         padding: 10px 12px; border-radius: 6px; margin: 8px 0; }
table.kv { border-collapse: collapse; width: 100%; font-size: .9em; }
table.kv td, table.kv th { border-bottom: 1px solid #e5eaf0;
         padding: 4px 8px; text-align: left; vertical-align: top; }
table.kv td:first-child { color: #5c6b7c; white-space: nowrap; }
.two-col { display: flex; flex-wrap: wrap; gap: 14px; }
.two-col > div { flex: 1 1 440px; min-width: 320px; }
.chart { width: 100%; }
.subtitle { color: #5c6b7c; font-size: .9em; margin: 0 0 8px; }
"""


def build_analysis_report(
    pos_path: Path | str,
    out_html: Path | str,
    *,
    rover_obs: Path | str | None = None,
    base_obs: Path | str | None = None,
    gt_pos: Path | str | None = None,
    log: Optional[LogFn] = None,
) -> Path:
    """Build the one-file Post-processing analysis HTML report.

    ``pos_path`` (subject .pos) is required; ``rover_obs`` / ``base_obs`` /
    ``gt_pos`` are optional — panels that need a missing input are replaced
    by a short note. Returns the written HTML path.
    """
    log_: LogFn = log or (lambda s: None)
    pos_path = Path(pos_path)
    out_html = Path(out_html)

    rows = parse_rtkpos(pos_path)
    if not rows:
        raise ValueError(f"{pos_path} contains no parseable solution epochs.")
    log_(f"[analysis] {pos_path.name}: {len(rows)} epochs")

    rover_sum = base_sum = None
    phase_report = None
    if rover_obs is not None:
        rover_obs = Path(rover_obs)
        rover_sum = summarize_obs(rover_obs)
        phase_report = check_carrier_phase(rover_obs)
        log_(f"[analysis] rover obs: {rover_sum.message}")
        log_(f"[analysis] {phase_report.message}")
    if base_obs is not None:
        base_obs = Path(base_obs)
        base_sum = summarize_obs(base_obs)
        log_(f"[analysis] base obs: {base_sum.message}")

    headline_txt, headline_kind = _headline(rows)

    sections = [
        _panel_overview(pos_path, rows),
        _panel_satellites(rows, rover_sum, base_sum),
        _panel_snr(rover_sum, base_sum),
        _panel_phase(None if rover_obs is None else Path(rover_obs),
                     phase_report),
        _panel_noise(rows),
    ]
    pred_html, pred_stats = _panel_predicted(
        rows, None if gt_pos is None else Path(gt_pos), log_)
    sections.append(pred_html)

    generated = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    inputs_bits = [f"rover .pos: {pos_path.name}"]
    if rover_obs is not None:
        inputs_bits.append(f"rover .obs: {Path(rover_obs).name}")
    if base_obs is not None:
        inputs_bits.append(f"base .obs: {Path(base_obs).name}")
    if gt_pos is not None:
        inputs_bits.append(f"ground truth: {Path(gt_pos).name}")

    html_doc = (
        "<!DOCTYPE html>\n<html><head>\n<meta charset=\"utf-8\">\n"
        "<title>PPK Analysis Report</title>\n"
        f"<style>{_CSS}</style>\n"
        f"<script>{_load_plotly_js()}</script>\n"
        "</head><body>\n<div class=\"wrap\">\n"
        "<h1>PPK Solution Analysis</h1>\n"
        f'<p class="subtitle">{" | ".join(_esc(b) for b in inputs_bits)}'
        f" &nbsp;&mdash;&nbsp; generated {generated}</p>\n"
        f'<p><span class="badge badge-{headline_kind} badge-headline">'
        f"{_esc(headline_txt)}</span></p>\n"
        + "".join(sections)
        + "</div>\n</body></html>\n"
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    log_(
        f"[analysis] wrote {out_html} "
        f"(session sigma {pred_stats.get('session_sigma_m', float('nan')):.2f} m, "
        f"trust={pred_stats.get('trust_class', '?')})"
    )
    return out_html
