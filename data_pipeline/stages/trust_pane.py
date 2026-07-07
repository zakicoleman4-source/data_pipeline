"""Build trust pane HTML: path color-coded by position/velocity trust.

Colors:
  green  — both position (<10m) and velocity (<1.2 m/s) trusted
  blue   — velocity trusted only
  orange — position trusted only
  red    — neither trusted

Panels:
  1. Map scatter (lat/lon) colored by trust
  2. Time series: eff_sig + threshold, disagree + threshold
  3. Summary stats bar
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
from ..geo import ecef_to_enu, llh_to_ecef
from ..parsers import PosRow, parse_rtkpos, parse_imu
from ..pipeline import LogFn, make_logger
from ..trust_formula import TrustConfig, TrustResult, compute_trust


_COLORS = {
    "high": "#22c55e",      # green
    "vel_only": "#3b82f6",  # blue
    "pos_only": "#f97316",  # orange
    "low": "#ef4444",       # red
}

_LABELS = {
    "high": "Trusted (pos+vel)",
    "vel_only": "Velocity only",
    "pos_only": "Position only",
    "low": "Untrusted",
}


def build_trust_pane(
    pos_file: Path,
    out_html: Path,
    *,
    sensors_txt: Optional[Path] = None,
    config: Optional[TrustConfig] = None,
    log: Optional[LogFn] = None,
) -> Path:
    log_ = make_logger(log)
    cfg = config or TrustConfig()

    pos_rows = parse_rtkpos(pos_file)
    if not pos_rows:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    imu_rows = None
    if sensors_txt:
        try:
            imu_rows = parse_imu(sensors_txt)
        except Exception:
            pass

    v2_opts = EpochWeightV2Options(
        zupt_enabled=True, nhc_enabled=True,
        nhc_heading_source="doppler", sigma_a_base=0.10,
    )
    v2 = smooth_epoch_weighted_v2(pos_rows, imu_rows=imu_rows,
                                   options=v2_opts, log=log_)

    trust = compute_trust(pos_rows, v2, config=cfg)
    log_(f"[trust] high={trust.n_high} vel_only={trust.n_vel_only} "
         f"pos_only={trust.n_pos_only} low={trust.n_low}")

    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    n = len(pos_rows)
    ts = [r.utc_s for r in pos_rows]
    t0 = ts[0]
    t_rel = [t - t0 for t in ts]

    lats = [r.lat_deg for r in pos_rows]
    lons = [r.lon_deg for r in pos_rows]

    data = {
        "t": t_rel,
        "lat": lats,
        "lon": lons,
        "eff_sig": trust.eff_sig.tolist(),
        "disagree": trust.disagree.tolist(),
        "labels": trust.labels,
        "eff_sig_thresh": cfg.eff_sig_max,
        "disagree_thresh": cfg.disagree_max,
        "pos_validated_max": cfg.pos_validated_max_m,
        "vel_validated_p95": cfg.vel_validated_p95_mps,
        "n_total": n,
        "n_high": trust.n_high,
        "n_vel_only": trust.n_vel_only,
        "n_pos_only": trust.n_pos_only,
        "n_low": trust.n_low,
        "colors": _COLORS,
        "label_names": _LABELS,
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(data))

    # Copy plotly.min.js next to output
    from .viewers import _copy_plotly_next_to
    _copy_plotly_next_to(out_html.parent)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    log_(f"[trust] wrote {out_html}")
    return out_html


_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Trust Pane — Position &amp; Velocity Confidence</title>
<script src="plotly.min.js"></script>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee; }
  .header { padding:12px 20px; background:#1a1a2e; display:flex; align-items:center; gap:20px; }
  .header h1 { margin:0; font-size:18px; }
  .stat { display:inline-block; padding:4px 12px; border-radius:4px; font-size:13px; font-weight:600; }
  .legend { display:flex; gap:12px; margin-left:auto; font-size:12px; }
  .legend span { display:flex; align-items:center; gap:4px; }
  .legend .dot { width:10px; height:10px; border-radius:50%; }
  .panels { display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; height:calc(100vh - 50px); }
  .panel { background:#1a1a2e; margin:2px; }
  .formula { padding:8px 20px; background:#16213e; font-size:12px; font-family:monospace; }
  .formula b { color:#22c55e; }
</style>
</head><body>
<div class="header">
  <h1>Trust Pane</h1>
  <div id="stats"></div>
  <div class="legend">
    <span><span class="dot" style="background:#22c55e"></span>Trusted (pos+vel)</span>
    <span><span class="dot" style="background:#3b82f6"></span>Velocity only</span>
    <span><span class="dot" style="background:#f97316"></span>Position only</span>
    <span><span class="dot" style="background:#ef4444"></span>Untrusted</span>
  </div>
</div>
<div class="formula">
  Position trust: <b>eff_sig &lt; __EFF_THRESH__</b> &rarr; h &le; __POS_MAX__m &nbsp;|&nbsp;
  Velocity trust: <b>|raw-v2| &lt; __DIS_THRESH__</b> &rarr; v &le; __VEL_P95__ m/s @ 2&sigma;
</div>
<div class="panels">
  <div id="map" class="panel"></div>
  <div id="esig" class="panel"></div>
  <div id="summary" class="panel"></div>
  <div id="disagree" class="panel"></div>
</div>
<script>
const D = __DATA__;
const C = D.colors;
const L = D.label_names;

// Stats
document.getElementById('stats').innerHTML =
  `<span class="stat" style="background:${C.high}">${D.n_high} high (${(100*D.n_high/D.n_total).toFixed(1)}%)</span>` +
  `<span class="stat" style="background:${C.vel_only}">${D.n_vel_only} vel</span>` +
  `<span class="stat" style="background:${C.pos_only}">${D.n_pos_only} pos</span>` +
  `<span class="stat" style="background:${C.low}">${D.n_low} low</span>`;

// Color array
const colors = D.labels.map(l => C[l]);
const hoverText = D.labels.map((l,i) =>
  `t=${D.t[i].toFixed(1)}s<br>eff_sig=${D.eff_sig[i].toFixed(3)}<br>disagree=${D.disagree[i].toFixed(3)}<br>${L[l]}`);

// Map
Plotly.newPlot('map', [{
  x: D.lon, y: D.lat, mode:'markers',
  marker: { color: colors, size: 4 },
  text: hoverText, hoverinfo:'text', type:'scatter'
}], {
  title: {text:'Trajectory — Trust Map', font:{color:'#eee',size:14}},
  xaxis: {title:'Longitude', color:'#999', gridcolor:'#333'},
  yaxis: {title:'Latitude', color:'#999', gridcolor:'#333', scaleanchor:'x',
          scaleratio: 1/Math.cos(D.lat[0]*Math.PI/180)},
  plot_bgcolor:'#16213e', paper_bgcolor:'#1a1a2e',
  margin:{l:60,r:20,t:40,b:40}
});

// Eff sig time series
Plotly.newPlot('esig', [
  {x:D.t, y:D.eff_sig, mode:'markers', marker:{color:colors, size:3},
   text:hoverText, hoverinfo:'text', name:'eff_sig'},
  {x:[D.t[0],D.t[D.t.length-1]], y:[D.eff_sig_thresh,D.eff_sig_thresh],
   mode:'lines', line:{color:'#22c55e',dash:'dash',width:2}, name:'threshold'}
], {
  title:{text:'Position indicator: effective_sigma',font:{color:'#eee',size:14}},
  xaxis:{title:'Time (s)',color:'#999',gridcolor:'#333'},
  yaxis:{title:'eff_sig (m)',color:'#999',gridcolor:'#333',range:[0,Math.min(8,Math.max(...D.eff_sig)*1.1)]},
  plot_bgcolor:'#16213e', paper_bgcolor:'#1a1a2e',
  margin:{l:60,r:20,t:40,b:40}, showlegend:false,
  annotations:[{x:D.t[D.t.length-1],y:D.eff_sig_thresh,text:'pos trust',
    showarrow:false,font:{color:'#22c55e',size:11},yshift:10}]
});

// Disagree time series
Plotly.newPlot('disagree', [
  {x:D.t, y:D.disagree, mode:'markers', marker:{color:colors, size:3},
   text:hoverText, hoverinfo:'text', name:'|raw-v2|'},
  {x:[D.t[0],D.t[D.t.length-1]], y:[D.disagree_thresh,D.disagree_thresh],
   mode:'lines', line:{color:'#3b82f6',dash:'dash',width:2}, name:'threshold'}
], {
  title:{text:'Velocity indicator: |raw - v2| disagreement',font:{color:'#eee',size:14}},
  xaxis:{title:'Time (s)',color:'#999',gridcolor:'#333'},
  yaxis:{title:'disagreement (m)',color:'#999',gridcolor:'#333',range:[0,Math.min(20,Math.max(...D.disagree)*1.1)]},
  plot_bgcolor:'#16213e', paper_bgcolor:'#1a1a2e',
  margin:{l:60,r:20,t:40,b:40}, showlegend:false,
  annotations:[{x:D.t[D.t.length-1],y:D.disagree_thresh,text:'vel trust',
    showarrow:false,font:{color:'#3b82f6',size:11},yshift:10}]
});

// Summary donut
Plotly.newPlot('summary', [{
  values: [D.n_high, D.n_vel_only, D.n_pos_only, D.n_low],
  labels: [L.high, L.vel_only, L.pos_only, L.low],
  marker: {colors: [C.high, C.vel_only, C.pos_only, C.low]},
  type:'pie', hole:0.5,
  textinfo:'label+percent', textfont:{size:12, color:'#eee'},
  hoverinfo:'label+value+percent'
}], {
  title:{text:'Trust Distribution',font:{color:'#eee',size:14}},
  plot_bgcolor:'#16213e', paper_bgcolor:'#1a1a2e',
  margin:{l:20,r:20,t:40,b:20}, showlegend:false,
  annotations:[{text:`${D.n_total}<br>epochs`,showarrow:false,
    font:{size:16,color:'#eee'}}]
});
</script>
</body></html>""".replace(
    "__EFF_THRESH__", "0.85"
).replace(
    "__DIS_THRESH__", "4.0"
).replace(
    "__POS_MAX__", "9.53"
).replace(
    "__VEL_P95__", "1.18"
)
