"""Do the device linear sensor+rate sensor agree with the reference unit survey-grade Signal reference?

For each (session, reference) pair, run compute_imu_trust with the reference
path as pos_rows and the device sensors as imu_rows. corr = rate sensor agreement
(yaw-rate vs reference turn-rate); mount_conf = linear sensor agreement (linear linear sensor vs
reference along-track linear sensor). Prints a table and writes an HTML overlay."""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_pipeline.parsers import parse_imu, parse_rtkpos
from data_pipeline.imu_trust import compute_imu_trust

AFF = {"a": 0.0, "b": 1.0}
CASES = [
    ("day14 dodge190336", "vs JAVAD",
     r"C:/Aj/gps/day14/dodge/20260628_190336_677/sensors_20260628_190336_677.txt",
     r"C:/Aj/gps/day14/solved_2026-06-28/gt/gt_log0628a.pos"),
    ("day14 dodge190336", "vs device PPK",
     r"C:/Aj/gps/day14/dodge/20260628_190336_677/sensors_20260628_190336_677.txt",
     r"C:/Aj/gps/day14/solved_2026-06-28/dodge/20260628_190336_677/rover.pos"),
    ("day12 dodge1", "vs JAVAD",
     r"C:/Aj/gps/DAY12/dodge1/20260505_152247_472/sensors_20260505_152247_472.txt",
     r"C:/Aj/gps/DAY12/ppk/javad_gt12.pos"),
]

def run():
    out = REPO / "_javad_out"; out.mkdir(exist_ok=True)
    print(f"{'session':20}{'ref':14}{'gyro_corr':>10}{'accel_conf':>11}{'mount':>7}{'verdict':>8}")
    plot_payload = {}
    for label, ref, sens, pos in CASES:
        imu = parse_imu(Path(sens)); posr = parse_rtkpos(Path(pos))
        r = compute_imu_trust(imu, posr, AFF)
        f = r["flags"]
        print(f"{label:20}{ref:14}{str(f['corr']):>10}{str(f['mount_conf']):>11}"
              f"{str(f['mount_resolved']):>7}{f['verdict']:>8}")
        key = f"{label} {ref}"
        plot_payload[key] = {"t": r["t_video"], "gyro": r["yaw_meas_dps"],
                             "turn": r["turn_traj_dps"], "fwd": r["fwd_accel"],
                             "corr": f["corr"], "mount_conf": f["mount_conf"]}
    # HTML overlay (uses local plotly if present next to output, else CDN)
    plotly = "plotly.min.js" if (REPO / "data_pipeline" / "plotly.min.js").exists() else None
    src = ("../data_pipeline/plotly.min.js" if plotly
           else "https://cdn.plot.ly/plotly-2.35.2.min.js")
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<script src="{src}"></script><style>body{{background:#0b0b0b;color:#ddd;font-family:sans-serif;margin:0}}
.p{{height:280px}} h3{{margin:8px 10px 0}}</style></head><body>
<h2 style="padding:8px 10px">Device IMU vs Javad reference</h2>
<div id=root></div><script>
const D={json.dumps(plot_payload)};
const root=document.getElementById('root');
for(const [k,v] of Object.entries(D)){{
  const h=document.createElement('h3');h.textContent=k+`  (gyro corr ${{v.corr}}, accel conf ${{v.mount_conf}})`;root.appendChild(h);
  const d=document.createElement('div');d.className='p';root.appendChild(d);
  Plotly.newPlot(d,[
    {{x:v.t,y:v.gyro,name:'device gyro yaw-rate',line:{{color:'#7fffb3'}}}},
    {{x:v.t,y:v.turn,name:'Javad/ref turn-rate',line:{{color:'#f59e0b'}}}},
  ],{{paper_bgcolor:'#0b0b0b',plot_bgcolor:'#0b0b0b',font:{{color:'#bbb',size:10}},
     margin:{{l:44,r:8,t:6,b:24}},legend:{{orientation:'h'}},
     xaxis:{{title:'utc (s)',gridcolor:'#181818'}},yaxis:{{title:'deg/s',gridcolor:'#181818'}}}},
     {{responsive:true,displaylogo:false}});
}}
</script></body></html>"""
    (out / "imu_vs_javad.html").write_text(html, encoding="utf-8")
    print(f"wrote {out / 'imu_vs_javad.html'}")

if __name__ == "__main__":
    raise SystemExit(run())
