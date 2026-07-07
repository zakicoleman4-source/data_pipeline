"""Client-facing diagnostic viewers.

One module = one stop-shop for the visualisations the client cares about
after running ``pipeline_full.run_full``:

* :func:`make_smoother_comparison` -- one HTML overlaying every smoother
  variant (raw / Gaussian car / K_smart / ADAPTIVE / cv_rts_pv /
  epoch_weight when available). Toggleable traces in plotly legend.
* :func:`make_quality_panel`       -- multi-subplot HTML: ns + speed +
  sigma_h + sigma_v + Q over time. The "is my session any good?" panel.
* :func:`make_ppk_vs_kalman_diff`  -- per-epoch horizontal / vertical
  delta between raw Post-processing and the cleaned (smart-Recursive-filter) output. Shows
  WHERE the smoother had work to do.
* :func:`launch_rtkplot`           -- spawn bundled rtkplot.exe with
  the .obs / .pos / .stat files loaded so the user can drive the
  full The external solver-native viewer.

All HTML viewers are offline-portable (vendored ``plotly.min.js``
copied next to the emitted file).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _copy_plotly_next_to(html_dir: Path) -> Path:
    """Vendored plotly.min.js next to the HTML so the file works offline.

    Mirrors :func:`data_pipeline.stages.viewers._copy_plotly_next_to`
    but avoids the import dependency on session-motion model's viewers module.
    """
    html_dir.mkdir(parents=True, exist_ok=True)
    src = _ASSETS_DIR / "plotly.min.js"
    dst = html_dir / "plotly.min.js"
    if src.is_file() and (not dst.is_file()
                          or dst.stat().st_size != src.stat().st_size):
        import shutil
        shutil.copyfile(src, dst)
    return dst


_BASE_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="plotly.min.js"></script>
<style>
  body { margin:0; background:#0a1020; color:#e2e8f0;
         font-family: "Segoe UI", sans-serif; }
  h1   { padding:14px 18px 4px; font-weight:600; font-size:20px; }
  .sub { padding:0 18px 12px; color:#94a3b8; font-size:13px; }
  #plot{ width:100vw; height:calc(100vh - 70px); }
</style>
</head><body>
<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const LAYOUT = __LAYOUT__;
Plotly.newPlot('plot', DATA, LAYOUT,
               {responsive:true, displaylogo:false});
</script>
</body></html>
"""


def _write_html(
    out_html: Path, title: str, subtitle: str,
    data: list, layout: dict,
) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    _copy_plotly_next_to(out_html.parent)
    page = (_BASE_HTML
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__DATA__", json.dumps(data))
            .replace("__LAYOUT__", json.dumps(layout)))
    out_html.write_text(page, encoding="utf-8")
    return out_html


# ---------------------------------------------------------------------------
# Smoother comparison
# ---------------------------------------------------------------------------


def _enu_track(
    rows: Sequence["object"], ref: tuple[float, float, float],
) -> tuple[list[float], list[float], list[float]]:
    from .geo import llh_to_ecef, ecef_to_enu
    es, ns_, ts = [], [], []
    for r in rows:
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, n, _ = ecef_to_enu(x, y, z, ref)
        es.append(e); ns_.append(n); ts.append(r.utc_s)
    return es, ns_, ts


def _safe_smoother_call(name: str, fn, *args, **kwargs):
    """Run a smoother; on any failure return ``None`` and log."""
    try:
        return fn(*args, **kwargs)
    except (ImportError, RuntimeError, ValueError, OSError) as ex:
        print(f"[client_viewers] {name}: skipped ({type(ex).__name__}: {ex})")
        return None


def _gaussian_car(rows):
    """Gauss(xy=2s, z=10s) on rows. Returns same rows w/ smoothed lat/lon/h."""
    if len(rows) < 3:
        return list(rows)
    from .smoothing import gaussian_smooth
    t = [r.utc_s for r in rows]
    dts = [t[i+1] - t[i] for i in range(len(t)-1) if t[i+1] - t[i] > 1e-6]
    if not dts:
        return list(rows)
    median_dt = sorted(dts)[len(dts) // 2]
    fps = 1.0 / median_dt
    xs = max(1.0, 2.0 * fps)
    zs = max(1.0, 10.0 * fps)
    lat = gaussian_smooth([r.lat_deg for r in rows], xs)
    lon = gaussian_smooth([r.lon_deg for r in rows], xs)
    h   = gaussian_smooth([r.h_m for r in rows], zs)
    from .parsers import PosRow
    return [
        PosRow(
            utc_s=r.utc_s, lat_deg=lat[i], lon_deg=lon[i], h_m=h[i],
            quality=r.quality, vn=r.vn, ve=r.ve, vu=r.vu, ns=r.ns,
            sd_n=r.sd_n, sd_e=r.sd_e, sd_u=r.sd_u,
            sd_ne=r.sd_ne, sd_eu=r.sd_eu, sd_un=r.sd_un,
            age_s=r.age_s, ratio=r.ratio,
            sd_vn=r.sd_vn, sd_ve=r.sd_ve, sd_vu=r.sd_vu,
            sd_vne=r.sd_vne, sd_veu=r.sd_veu, sd_vun=r.sd_vun,
        )
        for i, r in enumerate(rows)
    ]


def make_smoother_comparison(
    raw_pos: Path, out_html: Path,
    *, stat_path: Optional[Path] = None,
) -> Path:
    """Render one HTML overlaying every available smoother on ``raw_pos``.

    Always-available baselines: raw Post-processing, Gaussian car (xy=2s/z=10s),
    K_smart (sigma-gated Recursive-filter+RTS), ADAPTIVE (regime-conditional).
    Opt-in (no-op if module missing): cv_rts_pv, epoch_weight.

    Output is a single HTML w/ plotly.min.js next to it. Each route is a
    legend-toggleable scatter trace in local Local-frame (anchored at the raw
    track's first epoch).
    """
    from .parsers import parse_rtkpos
    rows = parse_rtkpos(Path(raw_pos))
    if not rows:
        raise RuntimeError(
            f"No epochs in {raw_pos}. Cannot build comparison.")
    ref = (rows[0].lat_deg, rows[0].lon_deg, rows[0].h_m)

    traces: list[dict] = []

    def _add(name: str, rs, color: str, dash: str = "solid"):
        if not rs:
            return
        e, n, _ = _enu_track(rs, ref)
        traces.append({
            "x": e, "y": n, "mode": "lines",
            "name": name, "line": {"color": color, "width": 1.6,
                                   "dash": dash},
        })

    # 1) raw Post-processing (always)
    _add("raw PPK", list(rows), "#9ca3af")

    # 2) Gaussian car
    g = _safe_smoother_call("Gauss car", _gaussian_car, rows)
    _add("Gauss car (xy=2s, z=10s)", g, "#60a5fa")

    # 3) K_smart (Recipe-2 Recursive-filter + conditional ADAPTIVE)
    try:
        from .kalman_sigma import apply_kalman_smart
        r = apply_kalman_smart(rows)
        _add(f"K_smart [{r.branch_label}]", r.rows_out, "#a78bfa")
    except (ImportError, RuntimeError, ValueError) as ex:
        print(f"[client_viewers] K_smart: skipped ({ex})")

    # 4) ADAPTIVE alone
    try:
        from .nhc import adaptive_filter
        out, regime = adaptive_filter(list(rows))
        _add(f"ADAPTIVE [{regime}]", out, "#34d399")
    except (ImportError, RuntimeError, ValueError) as ex:
        print(f"[client_viewers] ADAPTIVE: skipped ({ex})")

    # 5) cv_rts_pv (session-audit)
    try:
        from .cv_rts import gate_then_cv  # noqa: F401
        # Common API guess: returns a list of PosRow.
        # Fall back to None if the call signature differs.
        try:
            from .cv_rts import gate_then_cv as _gtcv
            r = _safe_smoother_call("cv_rts_pv", _gtcv, list(rows))
            _add("cv_rts_pv", r, "#fbbf24", dash="dot")
        except (ImportError, TypeError):
            pass
    except ImportError:
        pass

    # 6) epoch_weight (session-audit)
    if stat_path is not None and Path(stat_path).is_file():
        try:
            from .epoch_weight import smooth_epoch_weighted
            r = _safe_smoother_call(
                "epoch_weighted", smooth_epoch_weighted,
                list(rows), stat_path=Path(stat_path),
            )
            # epoch_weight returns (E, N, U) numpy arrays in a local Local-frame
            # sample whose reference is the FIRST raw row -- same anchor
            # as our `ref`. So the arrays drop straight onto the plot.
            if r is not None and isinstance(r, tuple) and len(r) == 3:
                e_arr, n_arr, u_arr = r
                if len(e_arr) == len(rows):
                    traces.append({
                        "x": list(map(float, e_arr)),
                        "y": list(map(float, n_arr)), "mode": "lines",
                        "name": "epoch_weighted",
                        "line": {"color": "#f87171", "width": 1.6,
                                 "dash": "longdash"},
                    })
        except ImportError:
            pass

    layout = {
        "paper_bgcolor": "#0a1020",
        "plot_bgcolor":  "#0a1020",
        "font":          {"color": "#e2e8f0"},
        "xaxis": {"title": "East (m)",  "scaleanchor": "y",
                  "scaleratio": 1, "color": "#94a3b8"},
        "yaxis": {"title": "North (m)", "color": "#94a3b8"},
        "legend": {"bgcolor": "rgba(0,0,0,0.4)"},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    }
    subtitle = (f"{len(rows)} raw epochs &middot; ref={ref[0]:.6f}, "
                f"{ref[1]:.6f} &middot; toggle traces in legend")
    return _write_html(
        Path(out_html), "Smoother comparison", subtitle, traces, layout,
    )


# ---------------------------------------------------------------------------
# Quality panel: ns, speed, sigmas, Q
# ---------------------------------------------------------------------------


def make_quality_panel(pos_path: Path, out_html: Path) -> Path:
    """Render one HTML with stacked subplots over time:

    1. ns_solved (The external solver col 7)
    2. horizontal speed = hypot(vn, ve) from Rate-signal
    3. sigma_h = hypot(sd_n, sd_e), sigma_v = sd_u
    4. Quality flag (1=Fix, 2=Float, 4=Differential, 5=Single)
    """
    from .parsers import parse_rtkpos
    rows = parse_rtkpos(Path(pos_path))
    if not rows:
        raise RuntimeError(f"No epochs in {pos_path}.")
    t0 = rows[0].utc_s
    t = [r.utc_s - t0 for r in rows]
    ns_ = [r.ns for r in rows]
    spd = [math.hypot(r.vn, r.ve)
           if math.isfinite(r.vn) and math.isfinite(r.ve) else float("nan")
           for r in rows]
    sd_h = [math.hypot(r.sd_n, r.sd_e)
            if math.isfinite(r.sd_n) and math.isfinite(r.sd_e) else float("nan")
            for r in rows]
    sd_v = [r.sd_u if math.isfinite(r.sd_u) else float("nan") for r in rows]
    q    = [r.quality for r in rows]

    traces = [
        {"x": t, "y": ns_,  "mode": "lines", "name": "ns used",
         "yaxis": "y1", "line": {"color": "#7dd3fc", "width": 1.2}},
        {"x": t, "y": spd,  "mode": "lines", "name": "speed (m/s)",
         "yaxis": "y2", "line": {"color": "#34d399", "width": 1.2}},
        {"x": t, "y": sd_h, "mode": "lines", "name": "sigma_h (m)",
         "yaxis": "y3", "line": {"color": "#a78bfa", "width": 1.2}},
        {"x": t, "y": sd_v, "mode": "lines", "name": "sigma_v (m)",
         "yaxis": "y3", "line": {"color": "#fbbf24", "width": 1.2,
                                 "dash": "dot"}},
        {"x": t, "y": q,    "mode": "markers", "name": "Q",
         "yaxis": "y4",
         "marker": {"color": "#f87171", "size": 3}},
    ]
    layout = {
        "paper_bgcolor": "#0a1020",
        "plot_bgcolor":  "#0a1020",
        "font":  {"color": "#e2e8f0"},
        "xaxis": {"title": "time since start (s)", "color": "#94a3b8"},
        "yaxis":  {"title": "ns",         "domain": [0.78, 1.0],
                   "color": "#7dd3fc"},
        "yaxis2": {"title": "speed (m/s)", "domain": [0.52, 0.74],
                   "color": "#34d399"},
        "yaxis3": {"title": "sigma (m)",   "domain": [0.26, 0.48],
                   "color": "#a78bfa"},
        "yaxis4": {"title": "Q",           "domain": [0.0, 0.22],
                   "tickvals": [1, 2, 4, 5],
                   "ticktext": ["Fix=1", "Float=2", "DGPS=4", "Single=5"],
                   "color": "#f87171"},
        "legend": {"bgcolor": "rgba(0,0,0,0.4)"},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    }
    return _write_html(
        Path(out_html), "Session quality panel",
        f"{len(rows)} epochs &middot; {pos_path}",
        traces, layout,
    )


# ---------------------------------------------------------------------------
# Raw Post-processing vs Recursive-filter diff
# ---------------------------------------------------------------------------


def make_ppk_vs_kalman_diff(
    raw_pos: Path, clean_pos: Path, out_html: Path,
) -> Path:
    """Plot the per-epoch horizontal + vertical delta between raw .pos
    and cleaned (Recursive-filter-filtered) .pos. Tells the user where the
    smoother actually did work."""
    from .parsers import parse_rtkpos
    from .geo import llh_to_ecef, ecef_to_enu
    raw = parse_rtkpos(Path(raw_pos))
    cln = parse_rtkpos(Path(clean_pos))
    if not raw or not cln:
        raise RuntimeError(
            "raw or cleaned .pos is empty -- nothing to diff.")
    cmap = {round(r.utc_s, 3): r for r in cln}
    ref = (raw[0].lat_deg, raw[0].lon_deg, raw[0].h_m)
    t, dh, dv = [], [], []
    for r in raw:
        c = cmap.get(round(r.utc_s, 3))
        if c is None:
            continue
        xa, ya, za = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        ea, na, ua = ecef_to_enu(xa, ya, za, ref)
        xb, yb, zb = llh_to_ecef(c.lat_deg, c.lon_deg, c.h_m)
        eb, nb, ub = ecef_to_enu(xb, yb, zb, ref)
        t.append(r.utc_s - raw[0].utc_s)
        dh.append(math.hypot(ea - eb, na - nb))
        dv.append(abs(ua - ub))
    traces = [
        {"x": t, "y": dh, "mode": "lines", "name": "|horizontal diff|",
         "line": {"color": "#a78bfa", "width": 1.2}},
        {"x": t, "y": dv, "mode": "lines", "name": "|vertical diff|",
         "line": {"color": "#fbbf24", "width": 1.2}},
    ]
    layout = {
        "paper_bgcolor": "#0a1020",
        "plot_bgcolor":  "#0a1020",
        "font":  {"color": "#e2e8f0"},
        "xaxis": {"title": "time since start (s)", "color": "#94a3b8"},
        "yaxis": {"title": "magnitude (m)", "color": "#94a3b8"},
        "legend": {"bgcolor": "rgba(0,0,0,0.4)"},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    }
    if dh:
        mean_h = sum(dh) / len(dh)
        max_h = max(dh)
    else:
        mean_h = max_h = 0.0
    if dv:
        mean_v = sum(dv) / len(dv)
        max_v = max(dv)
    else:
        mean_v = max_v = 0.0
    sub = (f"matched {len(t)} epochs &middot; "
           f"mean horizontal diff {mean_h:.2f} m, max {max_h:.2f} m &middot; "
           f"mean vertical diff {mean_v:.2f} m, max {max_v:.2f} m")
    return _write_html(
        Path(out_html), "Raw PPK vs cleaned (Kalman) -- per-epoch diff",
        sub, traces, layout,
    )


# ---------------------------------------------------------------------------
# RTKPlot launcher
# ---------------------------------------------------------------------------


def make_vio_overlay(
    pos_path: Path, video_path: Path, recording_map: Path, out_html: Path,
    *, frame_decim_hz: float = 2.0, max_features: int = 300,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[object] = None,
) -> Path:
    """Run monocular Motion model on the media, overlay the Motion model-derived path
    on the raw Post-processing track. SLOW (3-5 min on 35-min media).

    No GT required — purely a visual sanity check that the Motion model velocity
    integration tracks the Post-processing shape.

    ``capture_meta`` / ``video_anchor`` / ``chop_video_anchor`` resolve the
    media sample-0 ``CLOCK_BOOTTIME`` t0 for boottime-anchor sessions (see
    :func:`data_pipeline.frame_time.resolve_video_t0_boottime_ns`). For a
    cut ("segment") clip, pass the segment's own anchor as
    ``chop_video_anchor`` — its min bootNs WINS over the parent
    capture_meta t0 (segment PTS are rebased to 0).
    """
    from .parsers import parse_rtkpos
    from .vio import (
        run_vio, vio_to_enu_velocities, calibrate_R_body_from_cam,
    )

    def _log(m):
        if log is not None:
            try:
                log(m)
            except TypeError:
                pass

    _log("[vio] parsing PPK")
    pos = parse_rtkpos(Path(pos_path))
    if not pos:
        raise RuntimeError(f"No PPK epochs in {pos_path}")
    _log("[vio] running VIO (this can take several minutes)...")
    samples = run_vio(
        video_path=Path(video_path),
        recording_map=Path(recording_map),
        frame_decim_hz=frame_decim_hz,
        max_features=max_features,
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        chop_video_anchor=chop_video_anchor,
        log=_log,
    )
    if not samples:
        raise RuntimeError(
            "VIO produced 0 usable samples. Likely causes: "
            "all-static video, very low light, or unsupported codec. "
            "Check that the .mp4 plays correctly.")
    _log(f"[vio] {len(samples)} samples; calibrating R_body_from_cam")
    Rbc, _ = calibrate_R_body_from_cam(samples, pos)
    vio_vels = vio_to_enu_velocities(samples, pos, R_body_from_cam=Rbc)
    _log(f"[vio] {len(vio_vels)} ENU velocity samples")

    # Integrate Motion model Local-frame velocities from the first Post-processing position as anchor.
    ref = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    es_ppk, ns_ppk, _ = _enu_track(pos, ref)
    t_ppk = [r.utc_s - pos[0].utc_s for r in pos]
    # vio_vels is list[tuple[float, np.ndarray]] from vio_to_enu_velocities
    t0 = pos[0].utc_s
    vio_t = [t - t0 for t, _ in vio_vels]
    vio_e = []
    vio_n = []
    cur_e = es_ppk[0]
    cur_n = ns_ppk[0]
    prev_t = vio_t[0] if vio_t else 0.0
    for t, v_enu in vio_vels:
        dt = (t - t0) - prev_t
        prev_t = t - t0
        cur_e += float(v_enu[0]) * dt
        cur_n += float(v_enu[1]) * dt
        vio_e.append(cur_e)
        vio_n.append(cur_n)

    traces = [
        {"x": es_ppk, "y": ns_ppk, "mode": "lines",
         "name": f"PPK ({len(pos)} epochs)",
         "line": {"color": "#9ca3af", "width": 1.4}},
        {"x": vio_e, "y": vio_n, "mode": "lines",
         "name": f"VIO integrated ({len(vio_vels)} samples)",
         "line": {"color": "#34d399", "width": 1.4, "dash": "dot"}},
    ]
    layout = {
        "paper_bgcolor": "#0a1020",
        "plot_bgcolor":  "#0a1020",
        "font":  {"color": "#e2e8f0"},
        "xaxis": {"title": "East (m)", "scaleanchor": "y",
                  "scaleratio": 1, "color": "#94a3b8"},
        "yaxis": {"title": "North (m)", "color": "#94a3b8"},
        "legend": {"bgcolor": "rgba(0,0,0,0.4)"},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
    }
    sub = (f"PPK ref={ref[0]:.6f}, {ref[1]:.6f} &middot; "
           "VIO trajectory integrated from PPK[0] using monocular "
           "essential-matrix translations scaled by PPK speed.")
    return _write_html(
        Path(out_html), "VIO trajectory overlay", sub, traces, layout,
    )


@dataclass(frozen=True)
class RtkPlotArgs:
    """Optional inputs passed to ``rtkplot.exe``.

    All fields are ``Optional[Path]``. ``rtkplot.exe`` opens whichever
    files are supplied; the rest the user loads via its File menu.
    """
    rover_obs:  Optional[Path] = None
    base_obs:   Optional[Path] = None
    nav_file:   Optional[Path] = None
    pos_file:   Optional[Path] = None
    stat_file:  Optional[Path] = None


def _resolve_rtkplot() -> Path:
    """Find ``rtkplot.exe`` -- bundle first, env var, lab default, PATH."""
    if os.name == "nt":
        # Bundle path (frozen exe) or source path
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                cand = Path(meipass) / "vendor" / "rtklib" / "rtkplot.exe"
                if cand.is_file():
                    return cand
        src = (Path(__file__).resolve().parent.parent
               / "vendor" / "rtklib" / "rtkplot.exe")
        if src.is_file():
            return src
        # env var
        env = os.environ.get("RTKPLOT")
        if env and Path(env).is_file():
            return Path(env)
        # Common default install location (Windows The external solver)
        cand = Path(r"C:\Program Files\RTKLIB\rtkplot.exe")
        if cand.is_file():
            return cand
    import shutil
    found = shutil.which("rtkplot.exe") or shutil.which("rtkplot")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "rtkplot.exe not found. Place rtkplot.exe in vendor/rtklib/ "
        "or set the RTKPLOT environment variable to its path."
    )


def launch_rtkplot(args: RtkPlotArgs) -> subprocess.Popen:
    """Spawn rtkplot.exe with files loaded. Does NOT wait for exit.

    rtkplot CLI accepts up to ~6 input files; we pass them in the
    order rtkplot expects: pos, then obs (subject/base), then nav.
    """
    exe = _resolve_rtkplot()
    cli: list[str] = [str(exe)]
    if args.pos_file and args.pos_file.is_file():
        cli.append(str(args.pos_file))
    if args.rover_obs and args.rover_obs.is_file():
        cli.append(str(args.rover_obs))
    if args.base_obs and args.base_obs.is_file():
        cli.append(str(args.base_obs))
    if args.nav_file and args.nav_file.is_file():
        cli.append(str(args.nav_file))
    if args.stat_file and args.stat_file.is_file():
        cli.append(str(args.stat_file))
    # creationflags so it detaches cleanly on Windows
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(cli, creationflags=flags)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def _cli_compare(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="viewers compare", description=__doc__)
    ap.add_argument("--pos",  required=True, type=Path)
    ap.add_argument("--stat", type=Path, default=None,
                    help="optional .pos.stat for epoch_weight overlay")
    ap.add_argument("--out",  required=True, type=Path,
                    help="output .html path")
    args = ap.parse_args(argv)
    out = make_smoother_comparison(args.pos, args.out, stat_path=args.stat)
    print(f"wrote {out}")
    return 0


def _cli_quality(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="viewers quality")
    ap.add_argument("--pos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    out = make_quality_panel(args.pos, args.out)
    print(f"wrote {out}")
    return 0


def _cli_diff(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="viewers diff")
    ap.add_argument("--raw",   required=True, type=Path,
                    help=".pos before smoothing")
    ap.add_argument("--clean", required=True, type=Path,
                    help=".pos after smoothing (e.g. *_clean.pos)")
    ap.add_argument("--out",   required=True, type=Path)
    args = ap.parse_args(argv)
    out = make_ppk_vs_kalman_diff(args.raw, args.clean, args.out)
    print(f"wrote {out}")
    return 0


def _cli_vio(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="viewers vio")
    ap.add_argument("--pos",           required=True, type=Path)
    ap.add_argument("--video",         required=True, type=Path)
    ap.add_argument("--recording-map", required=True, type=Path,
                    help="recording_*.txt with video-PTS<->UTC pairs")
    ap.add_argument("--out",           required=True, type=Path)
    ap.add_argument("--decim-hz",      type=float, default=2.0)
    args = ap.parse_args(argv)
    out = make_vio_overlay(
        args.pos, args.video, args.recording_map, args.out,
        frame_decim_hz=args.decim_hz, log=print,
    )
    print(f"wrote {out}")
    return 0


def _cli_rtkplot(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="viewers rtkplot")
    ap.add_argument("--rover-obs", type=Path, default=None)
    ap.add_argument("--base-obs",  type=Path, default=None)
    ap.add_argument("--nav",       type=Path, default=None)
    ap.add_argument("--pos",       type=Path, default=None)
    ap.add_argument("--stat",      type=Path, default=None)
    args = ap.parse_args(argv)
    p = launch_rtkplot(RtkPlotArgs(
        rover_obs=args.rover_obs, base_obs=args.base_obs,
        nav_file=args.nav, pos_file=args.pos, stat_file=args.stat,
    ))
    print(f"rtkplot.exe launched (pid={p.pid})")
    return 0


def main() -> int:
    """``data_to_frames-cli.exe viewers <subcmd> ...`` entry."""
    import argparse
    if len(sys.argv) < 2:
        print("usage: viewers compare|quality|diff|vio|rtkplot ...",
              file=sys.stderr)
        return 2
    sub = sys.argv[1]
    rest = sys.argv[2:]
    if sub == "compare":
        return _cli_compare(rest)
    if sub == "quality":
        return _cli_quality(rest)
    if sub == "diff":
        return _cli_diff(rest)
    if sub == "rtkplot":
        return _cli_rtkplot(rest)
    if sub == "vio":
        return _cli_vio(rest)
    print(f"unknown viewers subcommand: {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
