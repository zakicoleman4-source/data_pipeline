"""Single-page accuracy analysis dashboard for engineer review (no GT needed).

Aggregates:
  * All available smoother outputs (raw / gaussian / ns_adaptive /
    epoch_weighted / fgo / hybrid).
  * Per-epoch The external solver metadata (Q, ns, ratio, age, sd_n, sd_e).
  * Smart-std prediction + trust_class (validated on 14 GT sessions).
  * Motion model disagreement with Post-processing when Motion model velocities supplied.
  * Cross-smoother disagreement (filter-vs-filter, no GT).

Layout (top to bottom):
  1. Session summary pills (smart_std, trust_class, ns_med, Q distribution).
  2. Path panel — all filter outputs as legend-toggleable traces.
  3. Per-epoch quality strip — Q flag, ns, ratio over time.
  4. Predicted std vs filter disagreement — sanity check the std prediction.
  5. Filter-vs-filter cross-disagreement heatmap (when ≥ 2 smoothers).
  6. Per-epoch sd_h + smart_std envelope time-series.

All HTML+JS self-contained next to vendored ``plotly.min.js``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..accuracy_predictor import predicted_epoch_std, smart_session_std
from ..geo import ecef_to_enu, llh_to_ecef
from ..parsers import PosRow
from ..pipeline import LogFn, make_logger
from .viewers import _copy_plotly_next_to


@dataclass
class DashboardResult:
    html_path: Path
    n_epochs: int
    n_filters: int
    smart_std_m: float
    trust_class: str


def build_accuracy_dashboard(
    *,
    raw_pos_rows: list[PosRow],
    filter_outputs: dict,  # {label: list[PosRow]}
    out_html: Path,
    vio_velocities: Optional[list] = None,  # list of (utc_s, np.ndarray[3])
    log: Optional[LogFn] = None,
) -> DashboardResult:
    """Build engineer-facing accuracy analysis HTML.

    No GT required. Engineer flags problems by:
      - High filter-vs-filter disagreement -> regime mismatch.
      - Low ratio + Q=2 dominant -> weak fix.
      - Smart_std envelope smaller than filter-vs-filter spread -> std under-reports.
      - Motion model disagrees with Post-processing -> external observation says Post-processing is wrong.
    """
    log_ = make_logger(log)
    if not raw_pos_rows:
        raise ValueError("build_accuracy_dashboard: empty raw_pos_rows")

    ref_lat = raw_pos_rows[0].lat_deg
    ref_lon = raw_pos_rows[0].lon_deg
    ref_h = raw_pos_rows[0].h_m
    ref_llh = (ref_lat, ref_lon, ref_h)

    def _enu(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref_llh)

    n = len(raw_pos_rows)
    ts = np.array([r.utc_s for r in raw_pos_rows])
    Q = np.array([r.quality for r in raw_pos_rows])
    NS = np.array([r.ns for r in raw_pos_rows])
    RATIO = np.array([r.ratio for r in raw_pos_rows], float)
    SD_H = np.array([
        math.hypot(r.sd_n, r.sd_e)
        if (math.isfinite(r.sd_n) and math.isfinite(r.sd_e)) else float("nan")
        for r in raw_pos_rows
    ])

    profile = smart_session_std(raw_pos_rows)
    smart_arr = predicted_epoch_std(raw_pos_rows, profile)

    # Local-frame coords for each filter (incl. raw).
    routes = []
    palette = ["#9ca3af", "#22d3ee", "#22c55e", "#f59e0b", "#a855f7",
               "#3b82f6", "#10b981", "#ef4444"]
    all_rows = {"raw_ppk": raw_pos_rows, **filter_outputs}
    enu_by_route = {}
    for i, (label, rows) in enumerate(all_rows.items()):
        if len(rows) != n:
            log_(f"[dashboard] WARN filter '{label}' length {len(rows)} != raw {n}, skipping")
            continue
        es = []; ns = []
        for r in rows:
            e, nn, _u = _enu(r)
            es.append(float(e)); ns.append(float(nn))
        enu_by_route[label] = (np.array(es), np.array(ns))
        routes.append({
            "label": label,
            "color": palette[i % len(palette)],
            "e": es, "n": ns,
        })

    # Per-epoch filter-vs-raw disagreement (only for non-raw filters).
    disagreement = {}  # label -> list[float] m
    for label, (es, ns) in enu_by_route.items():
        if label == "raw_ppk":
            continue
        raw_e, raw_n = enu_by_route["raw_ppk"]
        d = np.sqrt((es - raw_e) ** 2 + (ns - raw_n) ** 2)
        disagreement[label] = d.tolist()

    # Motion model-vs-Post-processing disagreement when available. Cumulative Motion model position
    # delta vs raw Post-processing position delta from session start.
    vio_disagreement = None
    if vio_velocities:
        try:
            vio_ts = np.array([t for t, _v in vio_velocities])
            vio_vs = np.array([v for _t, v in vio_velocities])
            # Integrate Motion model velocity to position (cum trapezoidal).
            dt = np.diff(vio_ts, prepend=vio_ts[0])
            cum_vio = np.cumsum(vio_vs * dt[:, None], axis=0)
            # Sample at Post-processing timestamps via interp.
            cum_at_ppk = np.zeros((n, 3))
            for k in range(3):
                cum_at_ppk[:, k] = np.interp(ts, vio_ts, cum_vio[:, k],
                                              left=np.nan, right=np.nan)
            # Compare with Post-processing delta from raw[0].
            raw_e, raw_n = enu_by_route["raw_ppk"]
            ppk_de = raw_e - raw_e[0]
            ppk_dn = raw_n - raw_n[0]
            vio_de = cum_at_ppk[:, 0] - cum_at_ppk[0, 0] if not np.isnan(cum_at_ppk[0, 0]) else cum_at_ppk[:, 0]
            vio_dn = cum_at_ppk[:, 1] - cum_at_ppk[0, 1] if not np.isnan(cum_at_ppk[0, 1]) else cum_at_ppk[:, 1]
            vio_d = np.sqrt((ppk_de - vio_de) ** 2 + (ppk_dn - vio_dn) ** 2)
            vio_disagreement = vio_d.tolist()
        except Exception as e:
            log_(f"[dashboard] VIO disagreement compute failed: {e}")
            vio_disagreement = None

    # Pack JSON payload.
    payload = {
        "ref": {"lat": ref_lat, "lon": ref_lon, "h": ref_h},
        "ts_rel": (ts - ts[0]).tolist(),  # seconds since start
        "Q": Q.tolist(),
        "ns": NS.tolist(),
        "ratio": [float(r) if math.isfinite(r) else None for r in RATIO],
        "sd_h": [float(s) if math.isfinite(s) else None for s in SD_H],
        "smart_std": smart_arr.tolist(),
        "routes": routes,
        "disagreement": disagreement,
        "vio_disagreement": vio_disagreement,
        "session": {
            "smart_std_m": profile.smart_std_m,
            "trust_class": profile.trust_class,
            "inflation": profile.inflation,
            "raw_sd_med_m": profile.raw_sd_med_m,
            "raw_sd_spread": profile.raw_sd_spread,
            "ns_med": profile.ns_med,
            "q1_frac": profile.q1_frac,
            "q2_frac": profile.q2_frac,
            "q4_frac": profile.q4_frac,
            "q5_frac": profile.q5_frac,
            "ratio_med": profile.ratio_med,
            "components": profile.components,
            "n_epochs": n,
        },
    }

    out_html = Path(out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    js_path = out_html.with_suffix(".data.js")
    _copy_plotly_next_to(out_html.parent)
    js_path.write_text(
        "window.DASH = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Accuracy analysis dashboard</title>
<script src="plotly.min.js"></script>
<script src="__JS__"></script>
<style>
html,body{margin:0;background:#0b0f17;color:#d8e0ee;font-family:sans-serif;font-size:13px}
h1{margin:10px 14px;font-size:17px;color:#e5e7eb}
h2{margin:14px 14px 4px;font-size:14px;color:#cbd5e1;border-top:1px solid #1f2937;padding-top:10px}
.note{padding:4px 14px;color:#94a3b8;font-size:12px;max-width:1200px}
.row{display:flex;flex-wrap:wrap;gap:8px;padding:6px 14px}
.pill{background:#1f2937;border:1px solid #374151;border-radius:6px;padding:4px 10px;font-size:12px}
.pill b{color:#e5e7eb}
.pill.warn{background:#78350f;border-color:#92400e}
.pill.ok{background:#064e3b;border-color:#065f46}
.pill.danger{background:#7f1d1d;border-color:#991b1b}
.plot{width:100vw;height:48vh}
.plot.short{height:28vh}
.legend-help{padding:4px 14px;font-size:11px;color:#64748b}
</style></head><body>
<h1>Accuracy analysis dashboard</h1>
<div class="note">Engineer view — no reference required. Use filter
disagreement + std prediction + Q/ns/ratio profile to flag bad sessions.</div>

<h2>Session summary</h2>
<div class="row" id="summary"></div>
<div class="legend-help">
  trust_class: <b>trustworthy</b> = q1 dominant OR ratio_med &gt; 3 |
  <b>tight</b> = typical Q=2 float (use 2σ envelope) |
  <b>spike_risk</b> = multipath signature, occasional outliers exceed 3σ.
</div>

<h2>Trajectory — all filter outputs (toggle in legend)</h2>
<div id="plot_traj" class="plot"></div>

<h2>Per-epoch quality (Q flag + ns + ratio)</h2>
<div id="plot_quality" class="plot short"></div>

<h2>Predicted std vs filter-vs-raw disagreement</h2>
<div class="note">If a filter line consistently sits above the std envelope,
the std is under-reporting that filter's drift from raw. If raw matches
all filters, smart_std should bound the spread.</div>
<div id="plot_std" class="plot"></div>

<h2>Filter-vs-filter disagreement matrix</h2>
<div class="note">Cross-spread between smoothers. Tight cluster -> all
filters agree (likely good PPK). Wide spread -> regime ambiguity, engineer
should pick one + flag.</div>
<div id="plot_cross" class="plot short"></div>

<script>
const D = window.DASH;
const S = D.session;
const sumEl = document.getElementById('summary');
function pill(k, v, cls) {
  const el = document.createElement('div');
  el.className = 'pill' + (cls ? ' ' + cls : '');
  el.innerHTML = '<b>' + k + ':</b> ' + v;
  sumEl.appendChild(el);
}
const trustCls = (S.trust_class === 'trustworthy') ? 'ok' :
                 (S.trust_class === 'spike_risk') ? 'danger' : 'warn';
pill('trust_class', S.trust_class, trustCls);
pill('smart_std', S.smart_std_m.toFixed(2) + ' m');
pill('n epochs', S.n_epochs);
pill('inflation', S.inflation.toFixed(2) + 'x');
pill('raw_sd median', S.raw_sd_med_m.toFixed(3) + ' m');
pill('raw_sd spread', S.raw_sd_spread.toFixed(2) + 'x');
pill('ns median', S.ns_med.toFixed(1));
pill('Q1 / Q2 / Q4', (S.q1_frac*100).toFixed(1) + '% / ' + (S.q2_frac*100).toFixed(1) + '% / ' + (S.q4_frac*100).toFixed(1) + '%');
pill('ratio median', S.ratio_med.toFixed(2));
pill('ref lat', S.n_epochs > 0 ? D.ref.lat.toFixed(7) : '');

// ----- Trajectory panel -----
const trajTraces = D.routes.map(r => ({
  x: r.e, y: r.n, mode: 'lines',
  name: r.label, line: { color: r.color, width: 1.5 },
  hovertemplate: r.label + '<br>E=%{x:.2f} N=%{y:.2f}<extra></extra>',
}));
Plotly.newPlot('plot_traj', trajTraces, {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'East (m)', gridcolor:'#1f2937', scaleanchor:'y'},
  yaxis:{title:'North (m)', gridcolor:'#1f2937'},
  legend:{bgcolor:'rgba(11,15,23,0.7)', font:{color:'#d8e0ee'}},
  margin:{t:18,r:30,b:48,l:60},
});

// ----- Quality strip -----
Plotly.newPlot('plot_quality', [
  { x: D.ts_rel, y: D.Q, mode:'lines', name:'Q (1=fix 2=float 4=DGPS 5=single)',
    line:{color:'#fbbf24', width:1}, yaxis:'y1' },
  { x: D.ts_rel, y: D.ns, mode:'lines', name:'ns (sats in solution)',
    line:{color:'#22d3ee', width:1}, yaxis:'y2' },
  { x: D.ts_rel, y: D.ratio, mode:'lines', name:'ratio (AR test)',
    line:{color:'#a855f7', width:1}, yaxis:'y3' },
], {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'session time (s)', gridcolor:'#1f2937'},
  yaxis: {title:'Q', side:'left',  range:[0, 6], gridcolor:'#1f2937'},
  yaxis2:{title:'ns', side:'right', overlaying:'y', range:[0, 30], showgrid:false},
  yaxis3:{title:'ratio', side:'right', overlaying:'y', position:0.97, showgrid:false},
  legend:{orientation:'h', bgcolor:'rgba(11,15,23,0.7)'},
  margin:{t:18,r:80,b:48,l:60},
});

// ----- Predicted std vs disagreement -----
const stdTraces = [];
stdTraces.push({
  x: D.ts_rel, y: D.smart_std, mode:'lines', name:'smart_std (predicted)',
  line:{color:'#e5e7eb', width:2, dash:'dash'},
});
stdTraces.push({
  x: D.ts_rel, y: D.smart_std.map(v => v*2), mode:'lines', name:'2x smart_std',
  line:{color:'#94a3b8', width:1, dash:'dot'},
});
stdTraces.push({
  x: D.ts_rel, y: D.sd_h, mode:'lines', name:'raw RTKLIB sd_h',
  line:{color:'#475569', width:1},
});
let colorIdx = 0;
const colors = ['#22d3ee','#22c55e','#f59e0b','#a855f7','#3b82f6'];
for (const label in D.disagreement) {
  stdTraces.push({
    x: D.ts_rel, y: D.disagreement[label], mode:'lines',
    name: 'disagree(raw,' + label + ')',
    line:{color: colors[colorIdx++ % colors.length], width:1.2},
  });
}
if (D.vio_disagreement) {
  stdTraces.push({
    x: D.ts_rel, y: D.vio_disagreement, mode:'lines',
    name: 'disagree(raw, VIO_cum)',
    line:{color:'#ec4899', width:1.5},
  });
}
Plotly.newPlot('plot_std', stdTraces, {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'session time (s)', gridcolor:'#1f2937'},
  yaxis:{title:'metres', gridcolor:'#1f2937'},
  legend:{orientation:'h', bgcolor:'rgba(11,15,23,0.7)'},
  margin:{t:18,r:30,b:48,l:60},
});

// ----- Cross-filter spread -----
const filterLabels = D.routes.map(r => r.label).filter(l => l !== 'raw_ppk');
if (filterLabels.length >= 2) {
  // Per-epoch max disagreement across all filters vs raw.
  const ne = D.ts_rel.length;
  const maxSpread = new Array(ne).fill(0);
  for (let i = 0; i < ne; i++) {
    let m = 0;
    for (const lbl of filterLabels) {
      const v = D.disagreement[lbl] ? D.disagreement[lbl][i] : 0;
      if (v > m) m = v;
    }
    maxSpread[i] = m;
  }
  Plotly.newPlot('plot_cross', [{
    x: D.ts_rel, y: maxSpread, mode:'lines',
    name: 'max(disagreement vs raw)',
    line:{color:'#f87171', width:1.5},
  }, {
    x: D.ts_rel, y: D.smart_std, mode:'lines',
    name:'smart_std envelope',
    line:{color:'#e5e7eb', width:1.5, dash:'dash'},
  }], {
    paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
    xaxis:{title:'session time (s)', gridcolor:'#1f2937'},
    yaxis:{title:'metres', gridcolor:'#1f2937'},
    legend:{orientation:'h', bgcolor:'rgba(11,15,23,0.7)'},
    margin:{t:18,r:30,b:48,l:60},
  });
} else {
  document.getElementById('plot_cross').innerHTML =
    '<div class="note">Only one filter output supplied — add more smoothers to populate this panel.</div>';
}
</script>
</body></html>
""".replace("__JS__", js_path.name)

    tmp = Path(str(out_html) + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out_html)
    log_(
        f"[dashboard] wrote {out_html} "
        f"(n_epochs={n}, n_filters={len(filter_outputs)}, "
        f"smart_std={profile.smart_std_m:.2f}m, trust={profile.trust_class})"
    )
    return DashboardResult(
        html_path=out_html, n_epochs=n,
        n_filters=len(filter_outputs),
        smart_std_m=profile.smart_std_m,
        trust_class=profile.trust_class,
    )
