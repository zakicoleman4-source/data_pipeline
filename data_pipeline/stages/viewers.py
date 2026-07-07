"""Stage 4: build the offline HTML viewers.

Produces ``trajectory_viewer.html``, ``orientation_panel.html``,
``comparison_viewer.html``, and ``sync_player.html``. Each HTML loads a
vendored ``plotly.min.js`` next to itself so the output folder works
air-gapped.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil

import numpy as np
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..geo import llh_iterable_to_enu, llh_to_ecef, ecef_to_enu
from ..parsers import (
    decimate_orientation,
    interp_pos,
    parse_orientation,
    parse_data_fix,
    parse_rtkpos,
    read_frame_times_csv,
)
from ..pipeline import LogFn, make_logger
from ..smoothing import (
    estimate_rate_hz,
    gaussian_smooth,
    gaussian_smooth_circular_deg,
)
from ..stat_to_csv import StatRow, parse_stat
from ..sync_basemap import export_geotiff_basemap_wgs84
from ..time_sync import TimeAnchor, fit_time_anchor, per_anchor_residuals
from ..epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
from ..trust_formula_v2 import compute_trust_v2


_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


# -----------------------------
# Path viewer
# -----------------------------


@dataclass(frozen=True)
class TrajectoryResult:
    html_path: Path
    js_path: Path


def _noncomment_lines(f: Iterable[str]) -> Iterable[str]:
    """Yield lines, skipping those whose first non-space char is ``#``.

    The external tool/coordinate output CSV may carry a leading ``# ...`` comment line
    (``CsvOptions.emit_header_comment``). ``csv.DictReader`` does NOT skip
    such comments, so feeding it the raw file makes the comment become the
    header row -> ``Latitude``/``Longitude`` columns disappear. Filtering the
    comment lines first makes every reader robust whether or not the comment
    is present (it can appear anywhere, not just the first line).
    """
    for line in f:
        if line.lstrip().startswith("#"):
            continue
        yield line


def _read_georef_csv(path: Path) -> list[tuple[str, float, float, float]]:
    """Tolerant Coordinate output-CSV reader.

    The new default schema is lat/lon only; we fall back to ``Altitude=0``
    when the column is absent so the path viewer still works.

    Leading ``#`` comment lines (if present) are skipped before parsing.
    """
    rows: list[tuple[str, float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(_noncomment_lines(f)):
            try:
                lat = float(r.get("Latitude", ""))
                lon = float(r.get("Longitude", ""))
            except ValueError:
                continue
            alt_str = r.get("Altitude", "")
            try:
                h = float(alt_str) if alt_str not in (None, "") else 0.0
            except ValueError:
                h = 0.0
            rows.append((r.get("Image", ""), lat, lon, h))
    return rows


def _copy_plotly_next_to(html_dir: Path) -> Path:
    """Copy the vendored ``plotly.min.js`` next to ``html_dir`` (idempotent)."""
    src = _ASSETS_DIR / "plotly.min.js"
    if not src.is_file():
        raise FileNotFoundError(
            f"plotly.min.js missing from {src}. Re-install the package."
        )
    html_dir.mkdir(parents=True, exist_ok=True)
    dst = html_dir / "plotly.min.js"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    return dst


def build_trajectory_viewer(
    *,
    data_log: Path,
    georef_csv: Path,
    out_html: Path,
    recording_map: Optional[Path] = None,
    pos_file: Optional[Path] = None,
    log: Optional[LogFn] = None,
) -> TrajectoryResult:
    """Build a self-contained path viewer (device-only).

    The smoothed path comes from the Coordinate output CSV; the raw data Signal
    is overlaid for context. We deliberately don't include any base map so
    the viewer works on an air-gapped machine.

    When ``recording_map`` is set, the time-anchor regression stats (drift,
    fit uncertainty in ms, etc.) are written into the HTML header so they
    match the other diagnostic viewers.

    When ``pos_file`` is given, three Post-processing traces are added with markers
    coloured by **quality** (Q), **speed** (m/s from vn,ve), and
    **ns_solved** (source count). Only one is visible at a time; the
    HTML toolbar swaps which is shown.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    data_fixes = parse_data_fix(data_log)
    smoothed = _read_georef_csv(georef_csv)
    if not data_fixes:
        raise RuntimeError(f"No Fix lines parsed from {data_log}")
    if not smoothed:
        raise RuntimeError(f"No usable rows in {georef_csv}")

    ref_llh = (smoothed[0][1], smoothed[0][2], smoothed[0][3])
    log_(f"[viewer] reference: lat={ref_llh[0]:.7f} lon={ref_llh[1]:.7f} h={ref_llh[2]:.2f} m")

    e_p, n_p, u_p = llh_iterable_to_enu(
        ((p.lat, p.lon, p.h) for p in data_fixes), ref_llh
    )
    e_s, n_s, u_s = llh_iterable_to_enu(
        ((lat, lon, h) for _img, lat, lon, h in smoothed), ref_llh
    )

    data_text = [
        f"device {dt.datetime.fromtimestamp(p.utc_s, tz=dt.timezone.utc).isoformat()} "
        f"acc={p.h_acc:.1f} m"
        for p in data_fixes
    ]
    smoothed_text = [f"smoothed {img}" for img, *_ in smoothed]

    data3d = [
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "x": e_p,
            "y": n_p,
            "z": u_p,
            "name": "Device (raw FLP)",
            "line": {"color": "#ef4444", "width": 2},
            "marker": {"size": 2.0, "color": "#ef4444"},
            "text": data_text,
            "hovertemplate": "%{text}<br>E=%{x:.2f} N=%{y:.2f} U=%{z:.2f}<extra></extra>",
        },
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "x": e_s,
            "y": n_s,
            "z": u_s,
            "name": "Smoothed (Georef CSV)",
            "line": {"color": "#3b82f6", "width": 4},
            "marker": {"size": 2.5, "color": "#3b82f6"},
            "text": smoothed_text,
            "hovertemplate": "%{text}<br>E=%{x:.2f} N=%{y:.2f} U=%{z:.2f}<extra></extra>",
        },
    ]

    # Optional Post-processing colour-by-{Q,speed,ns} triple. Indices logged so the
    # HTML toolbar can flip visibility without restyling the data arrays.
    ppk_trace_indices: dict[str, int] = {}
    ppk_rows: list = []
    spd_vals: list[float] = []
    ns_vals:  list[int]   = []
    if pos_file is not None:
        try:
            ppk_rows = parse_rtkpos(pos_file)
        except Exception as ex:
            log_(f"[viewer] PPK parse failed ({ex}); skipping color-by traces")
            ppk_rows = []
        if ppk_rows:
            e_q, n_q, u_q = llh_iterable_to_enu(
                ((r.lat_deg, r.lon_deg, r.h_m) for r in ppk_rows), ref_llh
            )
            q_vals  = [r.quality for r in ppk_rows]
            ns_vals = [r.ns for r in ppk_rows]
            spd_vals = [
                math.sqrt((r.vn or 0.0)**2 + (r.ve or 0.0)**2)
                if math.isfinite(r.vn) and math.isfinite(r.ve)
                else float("nan")
                for r in ppk_rows
            ]
            ppk_hover_text = [
                f"PPK {dt.datetime.fromtimestamp(r.utc_s, tz=dt.timezone.utc).isoformat()}"
                f"<br>Q={r.quality} ns={r.ns} spd={spd_vals[i]:.2f} m/s"
                for i, r in enumerate(ppk_rows)
            ]
            # Q palette: 1=Fix green, 2=Float orange, 4=Differential grey, 5+=red.
            Q_PALETTE = {1: "#22c55e", 2: "#f59e0b", 4: "#9ca3af",
                         5: "#ef4444", 6: "#ef4444"}
            q_colors = [Q_PALETTE.get(q, "#888888") for q in q_vals]

            base_ppk = {
                "type": "scatter3d", "mode": "markers",
                "x": e_q, "y": n_q, "z": u_q,
                "text": ppk_hover_text,
                "hovertemplate": "%{text}<extra></extra>",
            }
            # 1) Q-colored (discrete swatches)
            ppk_trace_indices["q"] = len(data3d)
            data3d.append({
                **base_ppk,
                "name": "PPK epochs (by Q)",
                "marker": {"size": 3.0, "color": q_colors},
                "visible": True,
            })
            # 2) speed-colored (continuous Viridis)
            ppk_trace_indices["speed"] = len(data3d)
            data3d.append({
                **base_ppk,
                "name": "PPK epochs (by speed m/s)",
                "marker": {
                    "size": 3.0, "color": spd_vals,
                    "colorscale": "Viridis", "showscale": True,
                    "colorbar": {"title": "speed (m/s)", "x": 1.02,
                                 "thickness": 12, "len": 0.6},
                },
                "visible": False,
            })
            # 3) ns_solved (continuous, integer ticks)
            ppk_trace_indices["ns"] = len(data3d)
            ns_max = max(ns_vals) if ns_vals else 12
            data3d.append({
                **base_ppk,
                "name": "PPK epochs (by ns_solved)",
                "marker": {
                    "size": 3.0, "color": ns_vals,
                    "colorscale": "Plasma", "cmin": 0,
                    "cmax": max(12, ns_max), "showscale": True,
                    "colorbar": {"title": "ns_solved", "x": 1.02,
                                 "thickness": 12, "len": 0.6},
                },
                "visible": False,
            })
            log_(f"[viewer] PPK colour traces: {len(ppk_rows)} epochs")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    template = (_ASSETS_DIR / "trajectory.html").read_text(encoding="utf-8")
    ref_str = f"lat={ref_llh[0]:.7f} lon={ref_llh[1]:.7f} h={ref_llh[2]:.2f}m"
    template = template.replace(
        "__PPK_TRACE_INDICES__", json.dumps(ppk_trace_indices)
    )

    # Post-processing insight summary pills (epochs total, %Fix/Float/Single, median speed,
    # median ns). Renders to dashes when no .pos was supplied.
    def _dash() -> str: return "—"
    ppk_pill_html = ""
    if pos_file is not None and ppk_rows:
        _n = len(ppk_rows)
        _fixn = sum(1 for r in ppk_rows if r.quality == 1)
        _floatn = sum(1 for r in ppk_rows if r.quality == 2)
        _singn = sum(1 for r in ppk_rows if r.quality >= 4)
        _spd_finite = [s for s in spd_vals if math.isfinite(s)]
        _med_spd = (sorted(_spd_finite)[len(_spd_finite) // 2]
                    if _spd_finite else float("nan"))
        _max_spd = max(_spd_finite, default=float("nan"))
        _ns_finite = [n for n in ns_vals if n > 0]
        _med_ns = (sorted(_ns_finite)[len(_ns_finite) // 2]
                   if _ns_finite else 0)
        ppk_pill_html = (
            f'<br><span style="color:#888;font-size:11px;">'
            f'PPK INSIGHTS</span><br>'
            f'<span class="pill ok">epochs: {_n}</span>'
            f'<span class="pill ok">Fix: {_fixn}'
            f' ({100*_fixn/_n:.0f}%)</span>'
            f'<span class="pill warn">Float: {_floatn}'
            f' ({100*_floatn/_n:.0f}%)</span>'
            f'<span class="pill">Single/deg: {_singn}'
            f' ({100*_singn/_n:.0f}%)</span>'
            f'<span class="pill">median speed: '
            f'{_med_spd:.2f} m/s ({_med_spd*3.6:.1f} km/h)</span>'
            f'<span class="pill">max speed: '
            f'{_max_spd:.2f} m/s ({_max_spd*3.6:.1f} km/h)</span>'
            f'<span class="pill">median ns: {_med_ns}</span>'
        )
    template = template.replace("__PPK_PILLS__", ppk_pill_html)
    if recording_map is not None:
        anchor = fit_time_anchor(recording_map)
        log_(
            f"[viewer] time anchor: n={anchor.n} drift={anchor.drift_ppm:+.2f} ppm "
            f"fit ~{anchor.fit_uncertainty_s * 1e3:.2f} ms"
        )
        sub = (
            template.replace("__DATA3D__", json.dumps(data3d))
            .replace("__REF__", ref_str)
            .replace("__N__", str(anchor.n))
            .replace("__DRIFT__", f"{anchor.drift_ppm:+.2f}")
            .replace("__FIT_MS__", f"{anchor.fit_uncertainty_s * 1e3:.2f}")
            .replace("__RMSE_MS__", f"{anchor.rmse_s * 1e3:.1f}")
            .replace("__CUBIC_MS__", f"{anchor.cubic_rmse_improvement_s * 1e3:.2f}")
            .replace("__NREJ__", str(anchor.n_rejected))
        )
        html = sub
    else:
        dash = "\u2014"
        sub = (
            template.replace("__DATA3D__", json.dumps(data3d))
            .replace("__REF__", ref_str)
            .replace("__N__", dash)
            .replace("__DRIFT__", dash)
            .replace("__FIT_MS__", dash)
            .replace("__RMSE_MS__", dash)
            .replace("__CUBIC_MS__", dash)
            .replace("__NREJ__", dash)
        )
        html = sub
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[viewer] device={len(data_fixes)} smoothed={len(smoothed)} -> {out_html}"
    )
    return TrajectoryResult(html_path=out_html, js_path=js)


# -----------------------------
# Orientation panel
# -----------------------------


@dataclass(frozen=True)
class OrientationResult:
    html_path: Path
    js_path: Path


def _angle_diff_signed(a: float, b: float) -> float:
    return (a - b + 540.0) % 360.0 - 180.0


def _heading_from_velocity_deg(vn: float, ve: float) -> float:
    if (vn * vn + ve * ve) < 1e-6:
        return float("nan")
    h = math.degrees(math.atan2(ve, vn))
    return h + 360.0 if h < 0 else h


def _histogram(
    values: Iterable[float], n_bins: int = 60
) -> tuple[list[float], list[int]]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [], []
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        hi = lo + 1.0
    bin_w = (hi - lo) / n_bins
    centers = [lo + bin_w * (i + 0.5) for i in range(n_bins)]
    counts = [0] * n_bins
    for v in finite:
        idx = int((v - lo) / bin_w)
        if idx == n_bins:
            idx = n_bins - 1
        if 0 <= idx < n_bins:
            counts[idx] += 1
    return centers, counts


def _stats_row(name: str, unit: str, values: Iterable[float]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return f"<tr><td>{name}</td><td>{unit}</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
    n = len(finite)
    mn = min(finite)
    mx = max(finite)
    mean = sum(finite) / n
    var = sum((v - mean) ** 2 for v in finite) / n
    std = math.sqrt(var)
    return (
        f"<tr><td>{name}</td><td>{unit}</td>"
        f"<td>{mn:+.2f}</td><td>{mx:+.2f}</td>"
        f"<td>{mean:+.2f}</td><td>{std:.2f}</td></tr>"
    )


def build_orientation_panel(
    *,
    data_log: Path,
    pos_file: Path,
    out_html: Path,
    sensors_txt: Optional[Path] = None,
    smooth_sigma_seconds: float = 3.0,
    decimate_hz: float = 10.0,
    min_speed_mps: float = 2.0,
    log: Optional[LogFn] = None,
) -> OrientationResult:
    log_ = make_logger(log)
    out_html = out_html.resolve()

    orient_full = parse_orientation(data_log)
    pos = parse_rtkpos(pos_file)
    if not orient_full:
        # Fall back to Complementary-update Motion sensor attitude from sensors_*.txt when the device
        # log has no OrientationDeg lines (common with the source app >= v3.0).
        if sensors_txt is None or not sensors_txt.is_file():
            raise RuntimeError(
                f"No OrientationDeg lines in {data_log} and no sensors_*.txt available. "
                f"Enable sensor logging in the capture app, or provide sensors_*.txt."
            )
        log_(f"[panel] no OrientationDeg in data log — falling back to Mahony IMU attitude "
             f"from {sensors_txt.name}")
        from ..parsers import parse_imu
        from ..imu_gnss_fusion import fuse as _fuse
        from ..parsers import Orient as _Orient
        try:
            imu_rows = parse_imu(sensors_txt)
            _, att_samples = _fuse(imu_rows, pos, log=log_)
        except Exception as e:
            raise RuntimeError(
                f"No OrientationDeg in {data_log.name} and Mahony fusion failed: {e}"
            ) from e
        if not att_samples:
            raise RuntimeError(
                f"No OrientationDeg in {data_log.name} and Mahony produced no attitude samples."
            )
        orient_full = [
            _Orient(utc_s=s.utc_s, yaw=s.yaw_deg, roll=s.roll_deg,
                    pitch=s.pitch_deg, cal=3)
            for s in att_samples
        ]
        log_(f"[panel] Mahony attitude: {len(orient_full)} samples")
    if not pos:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    orient = decimate_orientation(orient_full, decimate_hz)
    log_(
        f"[panel] orientation {len(orient_full)} -> {len(orient)} samples "
        f"(decimate to ~{decimate_hz:g} Hz)"
    )

    rate_hz = estimate_rate_hz([o.utc_s for o in orient])
    sigma_samples = max(1.0, smooth_sigma_seconds * rate_hz)
    yaws = [o.yaw for o in orient]
    pitches = [o.pitch for o in orient]
    rolls = [o.roll for o in orient]
    yaws_smooth = gaussian_smooth_circular_deg(yaws, sigma_samples)
    pitches_smooth = gaussian_smooth(pitches, sigma_samples)
    rolls_smooth = gaussian_smooth(rolls, sigma_samples)
    log_(
        f"[panel] smoothing sigma={smooth_sigma_seconds:g}s "
        f"({sigma_samples:.1f} samples at {rate_hz:.1f} Hz)"
    )

    t_orient = [
        dt.datetime.fromtimestamp(o.utc_s, tz=dt.timezone.utc).isoformat()
        for o in orient
    ]

    headings: list[float] = []
    t_pos: list[str] = []
    for p in pos:
        if not (math.isfinite(p.vn) and math.isfinite(p.ve)):
            continue
        if math.hypot(p.vn, p.ve) < min_speed_mps:
            continue
        h = _heading_from_velocity_deg(p.vn, p.ve)
        headings.append(h)
        t_pos.append(
            dt.datetime.fromtimestamp(p.utc_s, tz=dt.timezone.utc).isoformat()
        )

    o_xs = [o.utc_s for o in orient]

    def _interp_yaw(series: list[float], t: float) -> float:
        i = bisect_left(o_xs, t)
        if i <= 0 or i >= len(o_xs):
            return float("nan")
        a_t, b_t = o_xs[i - 1], o_xs[i]
        # Reject if EITHER side of the bracket is more than 2 s from the
        # query (matches the gap policy used by interp_pos / interp_orient).
        if abs(t - a_t) > 2.0 or abs(b_t - t) > 2.0:
            return float("nan")
        a_v, b_v = series[i - 1], series[i]
        if not (math.isfinite(a_v) and math.isfinite(b_v)):
            return float("nan")
        d = ((b_v - a_v + 540.0) % 360.0) - 180.0
        frac = 0.0 if b_t == a_t else (t - a_t) / (b_t - a_t)
        return (a_v + d * frac) % 360.0

    diffs: list[float] = []
    diff_t: list[str] = []
    diffs_smooth: list[float] = []
    diff_t_smooth: list[str] = []
    for p in pos:
        if not (math.isfinite(p.vn) and math.isfinite(p.ve)):
            continue
        if math.hypot(p.vn, p.ve) < min_speed_mps:
            continue
        h_at = _heading_from_velocity_deg(p.vn, p.ve)
        if not math.isfinite(h_at):
            continue
        yaw_raw = _interp_yaw(yaws, p.utc_s)
        if math.isfinite(yaw_raw):
            diffs.append(_angle_diff_signed(yaw_raw, h_at))
            diff_t.append(
                dt.datetime.fromtimestamp(p.utc_s, tz=dt.timezone.utc).isoformat()
            )
        yaw_sm = _interp_yaw(yaws_smooth, p.utc_s)
        if math.isfinite(yaw_sm):
            diffs_smooth.append(_angle_diff_signed(yaw_sm, h_at))
            diff_t_smooth.append(
                dt.datetime.fromtimestamp(p.utc_s, tz=dt.timezone.utc).isoformat()
            )

    pitch_hist_x, pitch_hist_y = _histogram(pitches)
    roll_hist_x, roll_hist_y = _histogram(rolls)

    payload = {
        "t_orient": t_orient,
        "yaw": yaws,
        "yaw_smooth": yaws_smooth,
        "pitch": pitches,
        "pitch_smooth": pitches_smooth,
        "roll": rolls,
        "roll_smooth": rolls_smooth,
        "t_pos": t_pos,
        "heading": headings,
        "t_diff": diff_t,
        "yaw_minus_heading": diffs,
        "t_diff_s": diff_t_smooth,
        "yaw_minus_heading_smooth": diffs_smooth,
        "pitch_hist_x": pitch_hist_x,
        "pitch_hist_y": pitch_hist_y,
        "roll_hist_x": roll_hist_x,
        "roll_hist_y": roll_hist_y,
    }
    stats_rows = "".join(
        [
            _stats_row("device yaw (magnetometer)", "deg", yaws),
            _stats_row(
                f"device yaw (smoothed sigma={smooth_sigma_seconds:g}s)",
                "deg",
                yaws_smooth,
            ),
            _stats_row("pitch (gravity-based)", "deg", pitches),
            _stats_row(
                f"pitch (smoothed sigma={smooth_sigma_seconds:g}s)",
                "deg",
                pitches_smooth,
            ),
            _stats_row("roll (gravity-based)", "deg", rolls),
            _stats_row(
                f"roll (smoothed sigma={smooth_sigma_seconds:g}s)",
                "deg",
                rolls_smooth,
            ),
            _stats_row(
                f"heading from PPK velocity (>= {min_speed_mps:g} m/s)",
                "deg",
                headings,
            ),
            _stats_row("raw yaw - heading", "deg", diffs),
            _stats_row("smoothed yaw - heading", "deg", diffs_smooth),
        ]
    )

    template = (_ASSETS_DIR / "orientation.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload)).replace(
        "__STATS_ROWS__", stats_rows
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[panel] wrote {out_html} "
        f"(orient_samples={len(orient)} pos_moving={len(headings)})"
    )
    return OrientationResult(html_path=out_html, js_path=js)


# -----------------------------
# Comparison viewer (multiple smoothing profiles, togglable in legend)
# -----------------------------


@dataclass(frozen=True)
class _Profile:
    name: str
    color: str
    xy_sigma_s: float
    z_sigma_s: float


_DEFAULT_PROFILES: tuple[_Profile, ...] = (
    _Profile("raw PPK (1 Hz dots)", "#9ca3af", 0.0, 0.0),
    _Profile("linear interp (no smoothing)", "#60a5fa", 0.0, 0.0),
    _Profile("gentle (XY 0.5 s, Z 2 s)", "#34d399", 0.5, 2.0),
    _Profile("car (XY 2 s, Z 10 s)", "#f59e0b", 2.0, 10.0),
    _Profile("aggressive (XY 5 s, Z 20 s)", "#ef4444", 5.0, 20.0),
)


def _smooth(values: list[float], sigma_samples: float) -> list[float]:
    if sigma_samples <= 0:
        return list(values)
    return gaussian_smooth(values, sigma_samples)


def _first_boottime_ns_from_video_anchor(path: Path) -> Optional[float]:
    """Read the earliest ``bootNs`` from a per-sample ``video_anchor.txt``.

    Mirrors ``stages.georef._first_boottime_ns_from_video_anchor`` so the
    viewers resolve the boottime t0 the same way the coordinate output CSV stage does.
    Format: ``frameNumber,sensorTimestampNs(raw),bootNs,timestampSource``.
    """
    try:
        boots: list[float] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    boots.append(float(int(parts[2])))
                except ValueError:
                    continue
        if not boots:
            return None
        return min(boots)
    except OSError:
        return None


def _resolve_boottime_t0_ns(
    capture_meta: Optional[Path],
    video_anchor: Optional[Path],
    log_: LogFn,
    chop_video_anchor: Optional[Path] = None,
) -> Optional[float]:
    """Resolve ``video_t0_boottime_ns`` for boottime-format sessions.

    Same precedence as ``stages.georef._load_frames``: capture_meta manifest
    first, then the per-sample ``video_anchor.txt`` minimum bootNs. Returns
    ``None`` for legacy (video_ns) sessions, in which case sample PTS map
    directly via ``anchor.video_pts_to_utc_s``.

    ``chop_video_anchor``: for a cut ("segment") clip, the segment's own
    per-sample video_anchor.txt. When given, the sample-0 boottime t0 is the
    minimum bootNs of that file and OVERRIDES capture_meta's
    ``video_t0_boottime_ns`` — capture_meta carries the ORIGINAL full-session
    sample-0 boot, and the segment container file's PTS are rebased to 0, so using it would
    map every segment sample minutes early (same contract as
    ``stages.georef._load_frames``; see docs/findings/segment-time-contract.md).
    If the segment anchor is unreadable/empty a WARN is logged and the normal
    resolution runs (tolerant: viewers must not crash).
    """
    if chop_video_anchor is not None:
        chop_t0 = _first_boottime_ns_from_video_anchor(Path(chop_video_anchor))
        if chop_t0 is not None:
            log_("[viewer] chop clip: t0 from chop video_anchor min bootNs, "
                 "capture_meta t0 overridden")
            return chop_t0
        log_(f"[viewer] WARN: chop video_anchor "
             f"{Path(chop_video_anchor)} unreadable/empty; falling back to "
             "capture_meta/video_anchor t0 (frames of a trimmed clip may map "
             "to the full-session start)")
    boottime_t0_ns: Optional[float] = None
    if capture_meta is not None and Path(capture_meta).is_file():
        try:
            from ..capture_meta import parse_capture_meta
            cm = parse_capture_meta(Path(capture_meta))
            if cm.video_t0_boottime_ns is not None:
                boottime_t0_ns = float(cm.video_t0_boottime_ns)
                log_("[viewer] manifest timeline offset applied (boottime t0)")
        except FileNotFoundError:
            pass
        except Exception as e:  # tolerant: bad manifest must not kill the run
            log_(f"[viewer] manifest parse failed ({e}); using direct mapping")
    if boottime_t0_ns is None and video_anchor is not None and Path(video_anchor).is_file():
        t0 = _first_boottime_ns_from_video_anchor(Path(video_anchor))
        if t0 is not None:
            boottime_t0_ns = t0
            log_(f"[viewer] timeline offset recovered from {Path(video_anchor).name}")
    return boottime_t0_ns


def _make_frame_to_utc(
    anchor: TimeAnchor, boottime_t0_ns: Optional[float]
):
    """Return ``t_video_s -> utc_s``, boottime-aware.

    For ``anchor_format=2`` (boottime) sessions the session time-anchor maps
    ABSOLUTE bootNs to UTC, so a sample PTS must be lifted into bootNs first
    (``t0 + pts*1e9``). Feeding the raw PTS to ``video_pts_to_utc_s`` would map
    every sample ~hours away from the .pos window -> zero interpolation points
    (the Track-4/s21_1 "Post-processing interpolation produced no points" regression).
    """
    if boottime_t0_ns is not None:
        return lambda t: anchor.boottime_to_utc_s(boottime_t0_ns + t * 1e9)
    return lambda t: anchor.video_pts_to_utc_s(t)


def _interp_dense_at(
    pos_rows, frame_times: list[tuple[str, float]], anchor: TimeAnchor, max_gap_s: float,
    frame_to_utc=None,
) -> list[tuple[str, float, float, float, float]]:
    """For every sample, return (image, t_video_s, lat, lon, h) interpolated from Post-processing.

    ``frame_to_utc`` maps ``t_video_s -> utc_s``; when omitted it defaults to
    the legacy direct PTS mapping (``anchor.video_pts_to_utc_s``). Pass a
    boottime-aware mapping (see :func:`_make_frame_to_utc`) for
    ``anchor_format=2`` sessions.
    """
    if frame_to_utc is None:
        frame_to_utc = lambda t: anchor.video_pts_to_utc_s(t)
    out: list[tuple[str, float, float, float, float]] = []
    times = [r.utc_s for r in pos_rows]
    for image, t in frame_times:
        utc_s = frame_to_utc(t)
        llh = interp_pos(pos_rows, utc_s, max_gap_s, times=times)
        if llh is None:
            continue
        out.append((image, t, llh[0], llh[1], llh[2]))
    return out


def _time_window_overlap_msg(
    pos_rows, frame_times: list[tuple[str, float]], frame_to_utc
) -> str:
    """Build a human diagnostic comparing the .pos and sample UTC windows.

    Used when per-sample interpolation yields zero points so the failure
    reports the two time windows + overlap instead of a bare RuntimeError.
    """
    if not pos_rows:
        return "no PPK rows"
    pos_lo = min(r.utc_s for r in pos_rows)
    pos_hi = max(r.utc_s for r in pos_rows)

    def _iso(s: float) -> str:
        return dt.datetime.fromtimestamp(s, tz=dt.timezone.utc).isoformat()

    if not frame_times:
        return f".pos UTC [{_iso(pos_lo)} .. {_iso(pos_hi)}]; no frame times"
    fu = [frame_to_utc(t) for _img, t in frame_times]
    f_lo, f_hi = min(fu), max(fu)
    overlap = sum(1 for u in fu if pos_lo <= u <= pos_hi)
    gap_s = 0.0 if overlap else (pos_lo - f_hi if f_hi < pos_lo else f_lo - pos_hi)
    return (
        f".pos UTC window:   [{_iso(pos_lo)} .. {_iso(pos_hi)}]\n"
        f"frame UTC window:  [{_iso(f_lo)} .. {_iso(f_hi)}]\n"
        f"frames inside .pos window: {overlap}/{len(fu)}"
        + ("" if overlap else f"; windows disjoint by ~{abs(gap_s):.0f} s "
                              "(likely a boottime-vs-PTS or GPST-vs-UTC time-base mismatch)")
    )


def build_comparison_viewer(
    *,
    data_log: Path,
    pos_file: Path,
    frame_times_csv: Path,
    recording_map: Path,
    out_html: Path,
    fps: float | None = None,
    profiles: tuple[_Profile, ...] = _DEFAULT_PROFILES,
    max_gap_s: float = 2.0,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[LogFn] = None,
) -> TrajectoryResult:
    """Build a viewer with several smoothing profiles overlaid as togglable traces.

    Each profile is a separate Plotly trace; click in the legend to show /
    hide individually. The header pills surface the time-anchor diagnostics
    so the user has the same accuracy numbers in front of them.

    If fps is not provided, it is derived from the median interval between
    samples in the frame_times_csv.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    anchor = fit_time_anchor(recording_map)
    log_(
        f"[compare] anchor n={anchor.n} drift={anchor.drift_ppm:+.2f} ppm "
        f"fit ~{anchor.fit_uncertainty_s * 1e3:.2f} ms"
    )

    pos_rows = parse_rtkpos(pos_file)
    if not pos_rows:
        raise RuntimeError(f"No PPK rows in {pos_file}")
    data_fixes = parse_data_fix(data_log)
    frame_times = read_frame_times_csv(frame_times_csv)
    if not frame_times:
        raise RuntimeError(f"Empty frame times CSV: {frame_times_csv}")

    # Derive effective FPS if not supplied.
    if fps is None and frame_times:
        intervals_s = []
        for i in range(1, len(frame_times)):
            dt = frame_times[i][1] - frame_times[i - 1][1]
            if dt > 0:
                intervals_s.append(dt)
        if intervals_s:
            intervals_s.sort()
            median_interval = intervals_s[len(intervals_s) // 2]
            fps = 1.0 / median_interval if median_interval > 0 else 1.0
            log_(f"[compare] effective fps derived from frame timings: {fps:.3f} Hz")
    if fps is None:
        fps = 1.0  # Fallback.
    else:
        log_(f"[compare] fps: {fps:.3f} Hz")

    ref_llh = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)

    # Raw Post-processing (1 Hz)
    raw_e, raw_n, raw_u = llh_iterable_to_enu(
        ((r.lat_deg, r.lon_deg, r.h_m) for r in pos_rows), ref_llh
    )
    # Device FLP
    ph_e, ph_n, ph_u = llh_iterable_to_enu(
        ((p.lat, p.lon, p.h) for p in data_fixes), ref_llh
    )
    # Per-sample linear interpolation. Resolve the boottime t0 so
    # anchor_format=2 (boottime) sessions map sample PTS correctly; otherwise
    # all samples fall hours outside the .pos window -> zero points. For a
    # cut ("segment") clip, chop_video_anchor overrides capture_meta's
    # full-session t0 (segment PTS are rebased to 0).
    _t0_ns = _resolve_boottime_t0_ns(
        capture_meta, video_anchor, log_, chop_video_anchor=chop_video_anchor)
    _frame_to_utc = _make_frame_to_utc(anchor, _t0_ns)
    dense = _interp_dense_at(pos_rows, frame_times, anchor, max_gap_s, _frame_to_utc)
    if not dense:
        raise RuntimeError(
            "PPK interpolation produced no points; check time anchor / .pos.\n"
            + _time_window_overlap_msg(pos_rows, frame_times, _frame_to_utc)
        )
    lats = [r[2] for r in dense]
    lons = [r[3] for r in dense]
    hs = [r[4] for r in dense]

    data: list[dict] = [
        {
            "type": "scatter3d", "mode": "markers", "name": "Device FLP (raw)",
            "x": ph_e, "y": ph_n, "z": ph_u,
            "marker": {"color": "#a78bfa", "size": 1.5},
            "visible": "legendonly",
        },
        {
            "type": "scatter3d", "mode": "markers", "name": profiles[0].name,
            "x": raw_e, "y": raw_n, "z": raw_u,
            "marker": {"color": profiles[0].color, "size": 2.5},
        },
    ]
    # "linear interp" + smoothed variants share the same dense series; we
    # smooth in Local-frame metric space, not lat/lon, to avoid latitude bias.
    # Convert dense series to Local-frame first.
    e_dense: list[float] = []
    n_dense: list[float] = []
    u_dense: list[float] = []
    for la, lo, h in zip(lats, lons, hs):
        x, y, z = llh_to_ecef(la, lo, h)
        ee, nn, uu = ecef_to_enu(x, y, z, ref_llh)
        e_dense.append(ee)
        n_dense.append(nn)
        u_dense.append(uu)

    for p in profiles[1:]:
        sigma_xy = max(0.0, p.xy_sigma_s) * max(1.0, fps)
        sigma_z = max(0.0, p.z_sigma_s) * max(1.0, fps)

        # Smooth in Local-frame metric space.
        e_s = _smooth(e_dense, sigma_xy)
        n_s = _smooth(n_dense, sigma_xy)
        u_s = _smooth(u_dense, sigma_z)

        data.append({
            "type": "scatter3d", "mode": "lines", "name": p.name,
            "x": e_s, "y": n_s, "z": u_s,
            "line": {"color": p.color, "width": 3},
            "visible": True if p.name.startswith("car") else "legendonly",
        })

    template = (_ASSETS_DIR / "compare.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA3D__", json.dumps(data))
        .replace("__N__", str(anchor.n))
        .replace("__DRIFT__", f"{anchor.drift_ppm:+.2f}")
        .replace("__FIT_MS__", f"{anchor.fit_uncertainty_s * 1e3:.2f}")
        .replace("__RMSE_MS__", f"{anchor.rmse_s * 1e3:.1f}")
        .replace("__CUBIC_MS__", f"{anchor.cubic_rmse_improvement_s * 1e3:.2f}")
        .replace("__NREJ__", str(anchor.n_rejected))
    )
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    log_(f"[compare] wrote {out_html}")
    return TrajectoryResult(html_path=out_html, js_path=js)


# -----------------------------
# Sync media + path player
# -----------------------------


@dataclass(frozen=True)
class SyncPlayerResult:
    html_path: Path
    js_path: Path
    trajectory_count: int
    basemap_png: Optional[Path]
    # Stream extras (None when no WAV was supplied).
    audio_src: Optional[str] = None        # url referenced by the player
    spectrogram_bins: Optional[tuple] = None  # (n_time, n_freq) or None
    sync_stats: Optional[dict] = None      # cross-clock sync diagnostics
    av_mux_path: Optional[Path] = None     # recording_*_av.container file, if muxed
    imu_trust: Optional[dict] = None       # {flags, note} sensor-trust summary


def _render_imu_strip_token(html: str, imu_strip: "Optional[dict]") -> str:
    """Fill the __IMU_STRIP__ template token (null when there is no Motion sensor)."""
    return html.replace("__IMU_STRIP__", json.dumps(imu_strip))


def _build_mux_cmd(
    ffmpeg: str, video: Path, wav: Path, out_mp4: Path, offset_ms: float = 0.0,
) -> list:
    """Build the external converter argv for :func:`mux_audio_into_mp4` (no execution).

    Split out so the command construction — in particular the ``-itsoffset``
    placement and sign — is unit-testable without running the external converter.

    Sign convention (matches ``audio_sync.compute_sync_stats``):
    ``offset_ms = (audio_start_utc - video_start_utc) * 1000``. A positive
    offset means the stream session started AFTER the media (stream lags), so
    the stream input must be delayed (played later) to line up with media --
    achieved with a positive ``-itsoffset`` on the stream input. A negative
    offset means stream started BEFORE media, so a negative ``-itsoffset``
    cuts/skips the leading edge of the stream to align it.
    """
    offset_s = offset_ms / 1000.0
    cmd = [ffmpeg, "-y", "-i", str(video.resolve())]
    if offset_s != 0.0:
        cmd += ["-itsoffset", f"{offset_s:.6f}"]
    cmd += [
        "-i", str(wav.resolve()),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_mp4),
    ]
    return cmd


def mux_audio_into_mp4(
    video: Path, wav: Path, out_mp4: Path, *,
    offset_ms: float = 0.0, log: Optional[LogFn] = None,
) -> Optional[Path]:
    """Mux ``wav`` into ``media`` as a new ``out_mp4`` (non-destructive).

    The media stream is copied (``-c:v copy``) and the stream is AAC-encoded so
    the result plays in browsers. ``offset_ms`` is the measured stream<->media
    offset (``audio_sync`` sign convention: positive means stream started
    after media); it is applied to the stream input via ``-itsoffset`` so the
    muxed file matches what the sync_player HTML shows. The ORIGINAL media is
    never modified. Returns the output path on success, or ``None`` if the external converter
    is unavailable / fails (callers should then fall back to the side-car
    ``<stream>`` element).
    """
    log_ = make_logger(log)
    try:
        from ..ffmpeg_paths import resolve_ffmpeg
        ffmpeg = resolve_ffmpeg()
    except Exception as e:  # the external converter missing -> graceful fallback
        log_(f"[sync] audio mux skipped (ffmpeg unavailable: {e})")
        return None
    import subprocess
    out_mp4 = out_mp4.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_mux_cmd(ffmpeg, video, wav, out_mp4, offset_ms=offset_ms)
    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except Exception as e:
        log_(f"[sync] audio mux failed ({e}); using side-car <audio> instead")
        return None
    if offset_ms:
        log_(f"[sync] muxed audio -> {out_mp4} (offset_ms={offset_ms:+.1f})")
    else:
        log_(f"[sync] muxed audio -> {out_mp4}")
    return out_mp4


def build_sync_player(
    *,
    video: Path,
    pos_file: Path,
    frame_times_csv: Path,
    recording_map: Path,
    out_html: Path,
    rotation: int = 0,
    max_gap_s: float = 2.0,
    sensors_txt: Optional[Path] = None,
    data_log: Optional[Path] = None,
    flp_providers: tuple[str, ...] = (
        "fused", "FUSED", "FUSED_LOCATION_PROVIDER", "fused_location",
    ),
    basemap_geotiff: Optional[Path] = None,
    basemap_max_dim: int = 2048,
    stat_file: Optional[Path] = None,
    video_bias_ms: float = 0.0,
    wav: Optional[Path] = None,
    audio_anchor: Optional[Path] = None,
    show_spectrogram: bool = True,
    mux_audio: bool = False,
    capture_meta: Optional[Path] = None,
    video_anchor: Optional[Path] = None,
    chop_video_anchor: Optional[Path] = None,
    log: Optional[LogFn] = None,
) -> SyncPlayerResult:
    """Build an HTML player with the media on the left and a live path marker on the right.

    The marker tracks ``media.currentTime`` via the ``timeupdate`` event,
    using a binary search over the sample timestamps so it works for hours of
    footage at 60 fps without lag.

    Stream support (all optional, additive):

    * ``wav`` + ``audio_anchor`` enable a synced ``<stream>`` element, a stream
      feature map panel (``show_spectrogram``), and a cross-clock sync/drift
      stats panel. The stream anchor (stream sample -> BOOTTIME) is mapped to UTC
      through the same boot->UTC :class:`TimeAnchor` used for the path.
    * ``mux_audio=True`` writes a NEW ``recording_*_av.container file`` (media copied +
      AAC stream) next to the HTML and points the player at it; the original
      ``.container file`` is never overwritten. Falls back to the ``<stream>`` side-car if
      the external converter is unavailable.
    * ``capture_meta`` / ``video_anchor`` supply ``video_t0_boottime_ns`` so the
      stream<->media offset can be measured.
    * ``chop_video_anchor``: for a cut ("segment") clip, the segment's own
      per-sample video_anchor.txt. Its min bootNs OVERRIDES capture_meta's
      full-session ``video_t0_boottime_ns`` (segment PTS are rebased to 0; the
      parent t0 would map samples minutes early). Mirrors
      ``stages.georef._load_frames``.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Boot->UTC anchor with the same fallback chain coordinate output uses: day14-style
    # sessions ship a 0-byte recording_*.txt, so recover the bridge from
    # measurements_*.txt (Signal clock / Fix rows) when available (E-PP-305 fix).
    from ..time_sync import fit_time_anchor_with_fallback
    anchor, _anchor_src = fit_time_anchor_with_fallback(recording_map, data_log)
    if _anchor_src == "measurements-fallback":
        log_("[sync] recording_*.txt empty/missing -> time anchor recovered "
             "from measurements_*.txt (GNSS clock + ChipsetElapsedRealtimeNanos)")
    elif _anchor_src == "measurements-fix-fallback":
        log_("[sync] recording_*.txt empty/missing -> time anchor recovered "
             "from measurements_*.txt Fix rows (UnixTimeMillis + "
             "elapsedRealtimeNanos)")
        log_("[sync] WARN: the Fix-row bridge includes the GNSS fix delivery "
             "latency -- absolute UTC placement of video/audio is typically "
             "~0.10-0.15 s EARLY on this device class. Audio<->video relative "
             "sync is unaffected (both ride the same anchor).")
    pos_rows = parse_rtkpos(pos_file)
    frame_times = read_frame_times_csv(frame_times_csv)
    if not pos_rows or not frame_times:
        raise RuntimeError("Sync player needs both PPK and frame times.")

    # Resolve boottime t0 (anchor_format=2 sessions) so sample PTS map into the
    # .pos UTC window; legacy video_ns sessions fall back to direct PTS mapping.
    # For a segment clip, chop_video_anchor overrides capture_meta's stale
    # full-session t0.
    _t0_ns = _resolve_boottime_t0_ns(
        capture_meta, video_anchor, log_, chop_video_anchor=chop_video_anchor)
    _frame_to_utc = _make_frame_to_utc(anchor, _t0_ns)

    dense = _interp_dense_at(pos_rows, frame_times, anchor, max_gap_s, _frame_to_utc)
    if not dense:
        raise RuntimeError(
            "Per-frame PPK interpolation yielded zero points.\n"
            + _time_window_overlap_msg(pos_rows, frame_times, _frame_to_utc)
        )

    ref_llh = (dense[0][2], dense[0][3], dense[0][4])

    # Per-epoch speed + azimuth for velocity HUD (no interpolation — real epochs only).
    _pos_t: list[float] = [r.utc_s for r in pos_rows]

    _pe: list[float] = []
    _pn: list[float] = []
    for _r in pos_rows:
        _xi, _yi, _zi = llh_to_ecef(_r.lat_deg, _r.lon_deg, _r.h_m)
        _ei, _ni, _ = ecef_to_enu(_xi, _yi, _zi, ref_llh)
        _pe.append(_ei)
        _pn.append(_ni)

    # Rate-signal speed + azimuth per epoch
    _dop_spd: list[Optional[float]] = []
    _az_dop:  list[Optional[float]] = []
    for _r in pos_rows:
        if math.isfinite(_r.vn) and math.isfinite(_r.ve):
            _s = math.sqrt(_r.vn ** 2 + _r.ve ** 2)
            _dop_spd.append(_s)
            _az_dop.append(math.degrees(math.atan2(_r.ve, _r.vn)) % 360 if _s > 0.3 else None)
        else:
            _dop_spd.append(None)
            _az_dop.append(None)

    # Coords speed + azimuth per epoch (Local-frame position diff).
    # Centered finite-difference so the speed/azimuth assigned to epoch i
    # corresponds to t_i (not t_i - Δt/2 as backward-difference gives).
    # The half-Δt shift would have left a ~0.5 s time offset between the
    # Rate-signal azimuth (which is exactly at t_i) and the coords azimuth on
    # the per-sample HUD, biasing the "disagreement" metric.
    _crd_spd: list[Optional[float]] = [None] * len(pos_rows)
    _az_crd:  list[Optional[float]] = [None] * len(pos_rows)
    for _i in range(1, len(pos_rows) - 1):
        _dt_v = _pos_t[_i + 1] - _pos_t[_i - 1]
        if 0 < _dt_v <= 2.0:
            _de = _pe[_i + 1] - _pe[_i - 1]
            _dn = _pn[_i + 1] - _pn[_i - 1]
            _crd_spd[_i] = math.sqrt(_de ** 2 + _dn ** 2) / _dt_v
            if _de ** 2 + _dn ** 2 > 0.09:
                _az_crd[_i] = math.degrees(math.atan2(_de, _dn)) % 360

    # FLP speed timeline (optional, requires data_log with `fused` provider).
    # Prefers the speed_mps column the source app writes per Fix row; falls back
    # to position-derivative when the column is empty. The Motion sensor-blended FLP
    # estimate is the third speed source in the HUD speedometer.
    _flp_t:   list[float] = []
    _flp_spd: list[float] = []
    if data_log is not None and data_log.is_file():
        try:
            _all_fixes = parse_data_fix(data_log)
        except Exception as _ex:
            _all_fixes = []
            log_(f"[sync] FLP parse failed ({_ex})")
        _flp_set = {p.lower() for p in flp_providers}
        _flp_rows = [f for f in _all_fixes if (f.provider or "").lower() in _flp_set]
        for _f in _flp_rows:
            if math.isfinite(_f.speed_mps):
                _flp_t.append(_f.utc_s)
                _flp_spd.append(float(_f.speed_mps))
        if _flp_rows and not _flp_t:
            # Provider rows present but no speed column populated — derive
            # from successive Local-frame positions.
            _prev: Optional[tuple[float, float, float]] = None
            for _f in _flp_rows:
                _x, _y, _z = llh_to_ecef(_f.lat, _f.lon, _f.h if math.isfinite(_f.h) else 0.0)
                _e, _n, _ = ecef_to_enu(_x, _y, _z, ref_llh)
                if _prev is not None:
                    _pt, _pe2, _pn2 = _prev
                    _dt = _f.utc_s - _pt
                    if 0.05 < _dt < 5.0:
                        _flp_t.append(_f.utc_s)
                        _flp_spd.append(math.sqrt((_e - _pe2) ** 2 + (_n - _pn2) ** 2) / _dt)
                _prev = (_f.utc_s, _e, _n)
        log_(f"[sync] FLP speed samples: {len(_flp_t)} (providers={list(flp_providers)})")

    def _interp_flp(_t: float) -> Optional[float]:
        """Linear interpolation of FLP speed at sample UTC. None outside data."""
        if not _flp_t:
            return None
        if _t < _flp_t[0] or _t > _flp_t[-1]:
            return None
        _j = bisect_left(_flp_t, _t)
        if _j == 0:
            return _flp_spd[0]
        if _j >= len(_flp_t):
            return _flp_spd[-1]
        _t0, _t1 = _flp_t[_j - 1], _flp_t[_j]
        _s0, _s1 = _flp_spd[_j - 1], _flp_spd[_j]
        _dt = _t1 - _t0
        if _dt <= 0:
            return _s0
        _u = (_t - _t0) / _dt
        return _s0 + _u * (_s1 - _s0)

    # Motion sensor yaw via Complementary-update filter (optional, requires sensors_txt)
    _att_t:   list[float] = []
    _att_yaw: list[float] = []
    _imu_rows: list = []          # hoisted: also feeds the sensor-trust strip below
    if sensors_txt is not None and sensors_txt.is_file():
        try:
            from ..imu_gnss_fusion import fuse as _imu_fuse
            from ..parsers import parse_imu as _parse_imu
            _imu_rows = _parse_imu(sensors_txt)
            _, _att_samples = _imu_fuse(_imu_rows, pos_rows)
            _att_t   = [s.utc_s  for s in _att_samples]
            _att_yaw = [s.yaw_deg for s in _att_samples]
            log_(f"[sync] IMU attitude: {len(_att_samples)} samples")
        except Exception as _e:
            log_(f"[sync] IMU attitude skipped ({_e})")

    def _nearest_epoch(_t: float) -> Optional[int]:
        _j = bisect_left(_pos_t, _t)
        _best: Optional[int] = None
        _best_dt = 0.7
        for _k in (_j - 1, _j):
            if 0 <= _k < len(_pos_t):
                _dt = abs(_pos_t[_k] - _t)
                if _dt < _best_dt:
                    _best_dt = _dt
                    _best = _k
        return _best

    def _nearest_yaw(_t: float) -> Optional[float]:
        if not _att_t:
            return None
        _j = bisect_left(_att_t, _t)
        _best: Optional[int] = None
        _best_dt = 0.15
        for _k in (_j - 1, _j):
            if 0 <= _k < len(_att_t):
                _dt = abs(_att_t[_k] - _t)
                if _dt < _best_dt:
                    _best_dt = _dt
                    _best = _k
        return _att_yaw[_best] % 360 if _best is not None else None

    # Coverage plot: parse .stat for per-epoch source az/el/valid
    _sky_epochs: list[dict] = []
    _sky_epoch_t: list[float] = []
    if stat_file is not None and stat_file.is_file():
        try:
            stat_rows = parse_stat(stat_file)
            # Group by epoch (utc_s rounded to ms to merge freq duplicates)
            from collections import defaultdict
            _by_epoch: dict[float, list[StatRow]] = defaultdict(list)
            for sr in stat_rows:
                _by_epoch[round(sr.utc_s, 3)].append(sr)
            # Build epoch list sorted by time; deduplicate per PRN within epoch.
            # Keep max |res_p| across frequencies for environment noise indicator.
            for _et in sorted(_by_epoch):
                seen_prn: dict[str, dict] = {}
                for sr in _by_epoch[_et]:
                    key = sr.prn
                    _abs_res = round(abs(sr.res_p_m), 2) if sr.res_p_m != 0 else 0.0
                    if key not in seen_prn:
                        seen_prn[key] = {
                            "prn": sr.prn,
                            "az": round(sr.az_deg, 1),
                            "el": round(sr.el_deg, 1),
                            "v": sr.valid_flag,
                            "snr": round(sr.snr_db_hz, 1),
                            "mp": _abs_res,
                        }
                    else:
                        prev = seen_prn[key]
                        if sr.valid_flag > prev["v"]:
                            prev["v"] = sr.valid_flag
                            prev["az"] = round(sr.az_deg, 1)
                            prev["el"] = round(sr.el_deg, 1)
                            prev["snr"] = round(sr.snr_db_hz, 1)
                        prev["mp"] = max(prev["mp"], _abs_res)
                # Attach driving azimuth from nearest pos epoch
                _sky_ep = _nearest_epoch(_et)
                _sky_az: Optional[float] = None
                if _sky_ep is not None and _az_dop[_sky_ep] is not None:
                    _sky_az = round(_az_dop[_sky_ep], 1)
                elif _sky_ep is not None and _az_crd[_sky_ep] is not None:
                    _sky_az = round(_az_crd[_sky_ep], 1)
                _sky_epochs.append({
                    "sats": list(seen_prn.values()),
                    "drv": _sky_az,
                })
                _sky_epoch_t.append(_et)
            log_(f"[sync] skyplot: {len(_sky_epochs)} epochs, "
                 f"{len(stat_rows)} sat observations from .stat")
        except Exception as _ex:
            log_(f"[sync] skyplot skipped ({_ex})")

    def _nearest_sky_epoch(_t: float) -> Optional[int]:
        if not _sky_epoch_t:
            return None
        _j = bisect_left(_sky_epoch_t, _t)
        _best_i: Optional[int] = None
        _best_dt = 1.5
        for _k in (_j - 1, _j):
            if 0 <= _k < len(_sky_epoch_t):
                _dt = abs(_sky_epoch_t[_k] - _t)
                if _dt < _best_dt:
                    _best_dt = _dt
                    _best_i = _k
        return _best_i

    # Trust v2 labels (per Post-processing epoch)
    _trust_v2_labels: list[str] = ["low"] * len(pos_rows)
    try:
        _v2_res = smooth_epoch_weighted_v2(
            pos_rows, imu_rows=None,
            options=EpochWeightV2Options(
                zupt_enabled=True, nhc_enabled=True,
                nhc_heading_source="doppler", sigma_a_base=0.10,
            ),
        )
        _trust_res = compute_trust_v2(pos_rows, _v2_res)
        _trust_v2_labels = _trust_res.labels
        log_(f"[sync] trust-v2: high={_trust_res.n_high} pos={_trust_res.n_pos_only} "
             f"vel={_trust_res.n_vel_only} low={_trust_res.n_low}")
    except Exception as _ex:
        log_(f"[sync] trust-v2 skipped ({_ex})")

    payload: list[dict] = []
    for img, t, lat, lon, h in dense:
        x, y, z = llh_to_ecef(lat, lon, h)
        e, n, u = ecef_to_enu(x, y, z, ref_llh)
        _utc  = _frame_to_utc(t)
        _ep   = _nearest_epoch(_utc)
        _dop  = round(_dop_spd[_ep], 3)  if (_ep is not None and _dop_spd[_ep]  is not None) else None
        _crd  = round(_crd_spd[_ep], 3)  if (_ep is not None and _crd_spd[_ep]  is not None) else None
        _adop = round(_az_dop[_ep],  1)  if (_ep is not None and _az_dop[_ep]   is not None) else None
        _acrd = round(_az_crd[_ep],  1)  if (_ep is not None and _az_crd[_ep]   is not None) else None
        _yaw  = _nearest_yaw(_utc)
        _aimu = round(_yaw, 1) if _yaw is not None else None
        _dis  = round(abs(_dop - _crd), 3) if (_dop is not None and _crd is not None) else None
        _q    = int(pos_rows[_ep].quality) if _ep is not None else None
        _ns   = int(pos_rows[_ep].ns)      if _ep is not None else None
        _flp_v = _interp_flp(_utc)
        _flp_o = round(_flp_v, 3) if _flp_v is not None else None
        _sky_i = _nearest_sky_epoch(_utc)
        payload.append({
            "image":    img,
            "t_video_s": round(t, 6),
            "lat":  round(lat, 9),
            "lon":  round(lon, 9),
            "h":    round(h, 4),
            "e":    round(e, 3),
            "n":    round(n, 3),
            "u":    round(u, 3),
            "q":     _q,
            "ns":    _ns,
            "dop":   _dop,
            "crd":   _crd,
            "flp":   _flp_o,
            "disag": _dis,
            "az_dop": _adop,
            "az_crd": _acrd,
            "az_imu": _aimu,
            "sky":   _sky_i,
            "trust": _trust_v2_labels[_ep] if _ep is not None else "low",
        })

    # Use a relative URL when possible: if the media lives in (or under) the
    # output folder, reference it relatively so the HTML is portable.
    try:
        video_rel = video.resolve().relative_to(out_html.parent)
        video_src = str(video_rel).replace("\\", "/")
    except ValueError:
        video_src = video.resolve().as_uri()

    meta = {
        "anchor_n": anchor.n,
        "drift_ppm": round(anchor.drift_ppm, 3),
        "fit_uncertainty_ms": round(anchor.fit_uncertainty_s * 1e3, 3),
        "rmse_ms": round(anchor.rmse_s * 1e3, 2),
        "n_rejected": anchor.n_rejected,
    }

    basemap_json: Optional[dict] = None
    basemap_png: Optional[Path] = None
    if basemap_geotiff is not None:
        png_name = "sync_basemap.png"
        png_out = out_html.parent / png_name
        exp = export_geotiff_basemap_wgs84(
            basemap_geotiff.resolve(),
            png_out,
            max_dim=basemap_max_dim,
        )
        basemap_png = exp.png_path
        basemap_json = {
            "png": png_name,
            "west": exp.west,
            "south": exp.south,
            "east": exp.east,
            "north": exp.north,
        }
        log_(
            f"[sync] basemap WGS84 PNG -> {png_out} "
            f"bounds=({exp.west:.6f},{exp.south:.6f})-({exp.east:.6f},{exp.north:.6f})"
        )

    # media-PTS -> UTC affine (utc = a + b * t_video_s), matching the payload's
    # per-sample UTC mapping so the feature map playhead shares the path's
    # UTC axis. Robust to either anchor dialect (video_ns or boottime).
    _u0 = _frame_to_utc(0.0)
    _u1 = _frame_to_utc(1.0)
    video_utc_affine = {"a": _u0, "b": _u1 - _u0}

    # ── Sensor-trust strip (raw rate sensor yaw-rate vs path turn-rate) ─────────
    imu_strip: Optional[dict] = None
    if _imu_rows:
        try:
            from ..imu_trust import compute_imu_trust
            imu_strip = compute_imu_trust(_imu_rows, pos_rows, video_utc_affine)
            if imu_strip is not None:
                log_(f"[sync] imu-trust: {imu_strip['note']}")
        except Exception as _e:  # never let the trust strip kill the viewer
            log_(f"[sync] imu-trust skipped ({_e})")

    # ── Stream: feature map + cross-clock sync stats + synced playback ──────────
    audio_src_json: Optional[str] = None
    spectro_json: Optional[dict] = None
    sync_stats_json: Optional[dict] = None
    spectrogram_bins: Optional[tuple] = None
    av_mux_path: Optional[Path] = None
    audio_video_offset_ms: float = 0.0
    # Residual stream offset the LIVE player's JS must apply when seeking the
    # <stream> element from the media clock (audio_sync sign convention:
    # positive => stream started AFTER media). It equals the measured
    # stream<->media offset for the side-car WAV, and 0.0 once the offset has
    # been baked into the muxed AV container file via -itsoffset (double-applying it
    # would desync the muxed file by the same amount in the other direction).
    audio_offset_residual_ms: float = 0.0
    if wav is not None and Path(wav).is_file():
        if audio_anchor is None or not Path(audio_anchor).is_file():
            log_("[sync] WAV supplied without an audio_anchor; "
                 "audio playback enabled but spectrogram/stats unavailable.")
        try:
            from ..audio_sync import analyze_audio

            # Recover video_t0_boottime_ns for the stream<->media offset.
            # Segment clips: the segment anchor's min bootNs is the clip's real
            # sample-0 boot; the parent capture_meta t0 (full-session sample-0)
            # would skew the dual-stream offset by the cut amount (minutes).
            v_t0_boot: Optional[float] = None
            if chop_video_anchor is not None:
                v_t0_boot = _first_boottime_ns_from_video_anchor(
                    Path(chop_video_anchor))
                if v_t0_boot is not None:
                    log_("[sync] chop clip: a/v-offset video t0 from chop "
                         "video_anchor min bootNs, capture_meta t0 overridden")
                else:
                    log_("[sync] WARN: chop video_anchor unreadable/empty for "
                         "a/v offset; falling back to capture_meta t0")
            if v_t0_boot is None and capture_meta is not None \
                    and Path(capture_meta).is_file():
                try:
                    from ..capture_meta import parse_capture_meta
                    cm = parse_capture_meta(Path(capture_meta))
                    if cm.video_t0_boottime_ns is not None:
                        v_t0_boot = float(cm.video_t0_boottime_ns)
                except Exception as _e:
                    log_(f"[sync] capture_meta parse failed ({_e})")
            if v_t0_boot is None and video_anchor is not None \
                    and Path(video_anchor).is_file():
                from .georef import _first_boottime_ns_from_video_anchor
                v_t0_boot = _first_boottime_ns_from_video_anchor(Path(video_anchor))

            if audio_anchor is not None and Path(audio_anchor).is_file():
                ares = analyze_audio(
                    wav=Path(wav),
                    audio_anchor=Path(audio_anchor),
                    boot_anchor=anchor,
                    video_t0_boottime_ns=v_t0_boot,
                )
                if show_spectrogram:
                    spectro_json = ares.spectrogram.to_dict()
                    spectrogram_bins = (
                        ares.spectrogram.power_db.shape[1],
                        ares.spectrogram.power_db.shape[0],
                    )
                sync_stats_json = ares.stats.to_dict()
                if ares.stats.audio_video_offset_ms is not None:
                    audio_video_offset_ms = float(ares.stats.audio_video_offset_ms)
                    audio_offset_residual_ms = audio_video_offset_ms
                log_(
                    f"[sync] audio: rate={ares.audio.sample_rate} Hz "
                    f"dur={ares.audio.duration_s:.1f}s "
                    f"drift={ares.stats.audio_drift_ppm:+.1f}ppm "
                    f"a/v offset={ares.stats.audio_video_offset_ms}"
                )
        except Exception as _e:  # never let stream analysis kill the viewer
            log_(f"[sync] audio analysis skipped ({_e})")

        # Decide the stream source URL: muxed AV container file (preferred when requested)
        # or the wav side-car. The original media is never overwritten.
        if mux_audio:
            mux_name = Path(video).stem + "_av.mp4"
            mux_out = out_html.parent / mux_name
            muxed = mux_audio_into_mp4(
                Path(video), Path(wav), mux_out,
                offset_ms=audio_video_offset_ms, log=log,
            )
            if muxed is not None:
                av_mux_path = muxed
                # Offset is baked into the muxed file; the JS must not
                # re-apply it on top.
                audio_offset_residual_ms = 0.0
                try:
                    rel = muxed.resolve().relative_to(out_html.parent)
                    audio_src_json = str(rel).replace("\\", "/")
                except ValueError:
                    audio_src_json = muxed.resolve().as_uri()
        if audio_src_json is None:
            # Side-car <stream> pointing at the wav (copied next to HTML if
            # outside the output folder so the player stays portable).
            try:
                rel = Path(wav).resolve().relative_to(out_html.parent)
                audio_src_json = str(rel).replace("\\", "/")
            except ValueError:
                wav_dst = out_html.parent / Path(wav).name
                if not wav_dst.exists():
                    try:
                        shutil.copyfile(Path(wav), wav_dst)
                    except OSError as _e:
                        log_(f"[sync] could not copy wav next to html ({_e})")
                audio_src_json = (
                    Path(wav).name if wav_dst.exists()
                    else Path(wav).resolve().as_uri()
                )

    # Human-readable note for trimmed ("chop") clips so the time display can
    # explain why video 0:00 is not the original recording start. UI text only
    # -- the timing math above is untouched.
    clip_note: Optional[str] = None
    if chop_video_anchor is not None:
        clip_note = (
            "Trimmed clip: video 0:00.000 is the START OF THIS CLIP, not the "
            "start of the original recording. UTC times shown are absolute "
            "wall-clock times, so they do not start at the session start."
        )

    template = (_ASSETS_DIR / "sync_player.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__CLIP_NOTE__", json.dumps(clip_note))
        .replace("__VIDEO_SRC__", json.dumps(video_src))
        .replace("__TRAJECTORY__", json.dumps(payload))
        .replace("__META__", json.dumps(meta))
        .replace("__BASEMAP__", json.dumps(basemap_json))
        .replace("__SKYPLOT__", json.dumps(_sky_epochs if _sky_epochs else None))
        .replace("__ROT_DEG__", str(int(rotation)))
        .replace("__BIAS_MS__", str(int(round(video_bias_ms))))
        .replace("__BIAS_ABS__", str(abs(int(round(video_bias_ms)))))
        .replace("__AUDIO_SRC__", json.dumps(audio_src_json))
        .replace("__AUDIO_OFFSET_MS__", json.dumps(round(audio_offset_residual_ms, 3)))
        .replace("__SPECTRO__", json.dumps(spectro_json))
        .replace("__SYNC_STATS__", json.dumps(sync_stats_json))
        .replace("__VIDEO_UTC_AFFINE__", json.dumps(video_utc_affine))
        .replace("__IMU_STRIP__", json.dumps(imu_strip))
        .replace("__N__", str(anchor.n))
        .replace("__DRIFT__", f"{anchor.drift_ppm:+.2f}")
        .replace("__FIT_MS__", f"{anchor.fit_uncertainty_s * 1e3:.2f}")
        .replace("__RMSE_MS__", f"{anchor.rmse_s * 1e3:.1f}")
        .replace("__NREJ__", str(anchor.n_rejected))
    )
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    if video_bias_ms != 0:
        log_(f"[sync] video bias: {video_bias_ms:+.0f} ms")
    log_(
        f"[sync] wrote {out_html} (frames={len(payload)} video_src={video_src})"
    )
    return SyncPlayerResult(
        html_path=out_html,
        js_path=js,
        trajectory_count=len(payload),
        basemap_png=basemap_png,
        audio_src=audio_src_json,
        spectrogram_bins=spectrogram_bins,
        sync_stats=sync_stats_json,
        av_mux_path=av_mux_path,
        imu_trust=(None if imu_strip is None
                   else {"flags": imu_strip["flags"], "note": imu_strip["note"]}),
    )


# -----------------------------
# Geo viewer (2D source + 3D terrain)
# -----------------------------


@dataclass(frozen=True)
class GeoViewerResult:
    html_path: Path
    js_path: Path
    epoch_count: int
    has_basemap: bool
    has_dsm: bool


def build_geo_viewer(
    *,
    pos_file: Path,
    out_html: Path,
    georef_csv: Optional[Path] = None,
    basemap_tiff: Optional[Path] = None,
    dsm_tiff: Optional[Path] = None,
    basemap_max_dim: int = 1024,
    log: Optional[LogFn] = None,
) -> GeoViewerResult:
    """Build an offline HTML viewer: 2D source background layer + 3D DSM terrain with Post-processing path.

    Requires ``rasterio`` for Raster file input (miniconda or ``pip install rasterio``).
    Without rasterio the viewer still renders the path; background layer/DSM are omitted.
    """
    import base64
    import tempfile

    log_ = make_logger(log)
    out_html = out_html.resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    pos = parse_rtkpos(pos_file)
    if not pos:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    trajectory = [
        {
            "lat": round(r.lat_deg, 9),
            "lon": round(r.lon_deg, 9),
            "h":   round(r.h_m, 3),
            "q":   int(r.quality),
            "t":   round(r.utc_s, 3),
        }
        for r in pos
    ]

    # Smoothed path from Coordinate output CSV (per-sample, higher density than 1 Hz Post-processing)
    smoothed: list[dict] = []
    if georef_csv is not None and georef_csv.is_file():
        for _img, _lat, _lon, _h in _read_georef_csv(georef_csv):
            smoothed.append({
                "lat": round(_lat, 9),
                "lon": round(_lon, 9),
                "h":   round(_h, 3),
            })
        log_(f"[geo] smoothed trajectory: {len(smoothed)} frames from {georef_csv.name}")

    # Background layer: export → PNG → base64 inline data URL (works offline, no relative path)
    basemap_json: Optional[dict] = None
    if basemap_tiff is not None and basemap_tiff.is_file():
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tf:
            _tmp = Path(_tf.name)
        try:
            exp = export_geotiff_basemap_wgs84(basemap_tiff.resolve(), _tmp, max_dim=max(128, basemap_max_dim))
            _b64 = base64.b64encode(_tmp.read_bytes()).decode()
            basemap_json = {
                "png":   f"data:image/png;base64,{_b64}",
                "west":  exp.west,
                "south": exp.south,
                "east":  exp.east,
                "north": exp.north,
            }
            log_(f"[geo] basemap inline PNG ({len(_b64) // 1024} KB b64)")
        except Exception as _e:
            log_(f"[geo] basemap skipped ({_e})")
        finally:
            try:
                _tmp.unlink()
            except OSError:
                pass

    # DSM: rasterio → lon/lat 1-D grids + 2-D elevation matrix
    dsm_json: Optional[dict] = None
    if dsm_tiff is not None and dsm_tiff.is_file():
        try:
            import rasterio as _rio  # type: ignore[import]
            with _rio.open(dsm_tiff.resolve()) as _src:
                _z = _src.read(1).tolist()
                _b = _src.bounds
                _rows, _cols = _src.height, _src.width
            _lons = [
                round(_b.left + (_b.right - _b.left) * (_c + 0.5) / _cols, 6)
                for _c in range(_cols)
            ]
            _lats = [
                round(_b.top - (_b.top - _b.bottom) * (_r + 0.5) / _rows, 6)
                for _r in range(_rows)
            ]
            dsm_json = {
                "z":     _z,
                "lons":  _lons,
                "lats":  _lats,
                "west":  float(_b.left),
                "south": float(_b.bottom),
                "east":  float(_b.right),
                "north": float(_b.top),
            }
            log_(f"[geo] DSM {_rows}×{_cols} elev={min(min(r) for r in _z):.0f}–{max(max(r) for r in _z):.0f}m")
        except ImportError:
            log_("[geo] rasterio not available, DSM skipped")
        except Exception as _e:
            log_(f"[geo] DSM read skipped ({_e})")

    geo_data = json.dumps({
        "trajectory": trajectory,
        "smoothed":   smoothed,
        "basemap":    basemap_json,
        "dsm":        dsm_json,
    })
    template = (_ASSETS_DIR / "geo_viewer.html").read_text(encoding="utf-8")
    html = template.replace("__GEO_DATA__", geo_data)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    log_(
        f"[geo] wrote {out_html} ({len(pos)} epochs, "
        f"smoothed={len(smoothed)}, "
        f"basemap={'yes' if basemap_json else 'no'}, "
        f"dsm={'yes' if dsm_json else 'no'})"
    )
    return GeoViewerResult(
        html_path=out_html,
        js_path=js,
        epoch_count=len(pos),
        has_basemap=basemap_json is not None,
        has_dsm=dsm_json is not None,
    )


# -----------------------------
# Velocity viewer
# -----------------------------


@dataclass(frozen=True)
class VelocityViewerResult:
    html_path: Path
    js_path: Path
    epoch_count: int


def build_velocity_viewer(
    *,
    pos_file: Path,
    out_html: Path,
    data_log: Optional[Path] = None,
    flp_providers: tuple[str, ...] = (
        "fused", "FUSED", "FUSED_LOCATION_PROVIDER", "fused_location",
    ),
    log: Optional[LogFn] = None,
) -> VelocityViewerResult:
    """Build a self-contained Rate-signal speed + azimuth viewer from a .pos file.

    When ``data_log`` is given, an additional **FLP** speed trace is
    interpolated onto the Post-processing timeline so the user can compare three
    independent speed sources (Rate-signal / Coords-Δ / Fused-Location).
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    pos = parse_rtkpos(pos_file)
    if not pos:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    t0 = pos[0].utc_s
    ref_llh = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)

    enu: list[tuple[float, float, float]] = []
    for r in pos:
        xi, yi, zi = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        enu.append(ecef_to_enu(xi, yi, zi, ref_llh))

    # FLP speed source (optional). Same parse + interpolate trick as the
    # sync_player builder so the three speed sources share a definition.
    _flp_t:   list[float] = []
    _flp_spd: list[float] = []
    if data_log is not None and data_log.is_file():
        try:
            _all_fixes = parse_data_fix(data_log)
        except Exception as _ex:
            _all_fixes = []
            log_(f"[velocity] FLP parse failed ({_ex})")
        _flp_set = {p.lower() for p in flp_providers}
        for _f in _all_fixes:
            if ((_f.provider or "").lower() in _flp_set
                    and math.isfinite(_f.speed_mps)):
                _flp_t.append(_f.utc_s)
                _flp_spd.append(float(_f.speed_mps))
        log_(f"[velocity] FLP speed rows: {len(_flp_t)}")

    def _interp_flp(_t: float) -> Optional[float]:
        if not _flp_t or _t < _flp_t[0] or _t > _flp_t[-1]:
            return None
        _j = bisect_left(_flp_t, _t)
        if _j == 0:
            return _flp_spd[0]
        if _j >= len(_flp_t):
            return _flp_spd[-1]
        _t0, _t1 = _flp_t[_j - 1], _flp_t[_j]
        _s0, _s1 = _flp_spd[_j - 1], _flp_spd[_j]
        _dt = _t1 - _t0
        return _s0 if _dt <= 0 else _s0 + (_t - _t0) / _dt * (_s1 - _s0)

    t_arr:     list[float] = []
    dop_arr:   list[object] = []
    crd_arr:   list[float] = []
    flp_arr:   list[object] = []
    disag_arr: list[object] = []
    az_arr:    list[object] = []
    q_arr:     list[int]   = []

    for i in range(1, len(pos)):
        rp, rc = pos[i - 1], pos[i]
        d_t = rc.utc_s - rp.utc_s
        if d_t <= 0 or d_t > 2.0:
            continue
        if not (math.isfinite(rc.vn) and math.isfinite(rc.ve)):
            continue

        de = enu[i][0] - enu[i - 1][0]
        dn = enu[i][1] - enu[i - 1][1]
        cs = math.sqrt(de * de + dn * dn) / d_t

        ds_p = math.sqrt(rp.vn ** 2 + rp.ve ** 2) if (math.isfinite(rp.vn) and math.isfinite(rp.ve)) else float("nan")
        ds_c = math.sqrt(rc.vn ** 2 + rc.ve ** 2)
        ds   = 0.5 * (ds_p + ds_c) if math.isfinite(ds_p) else ds_c

        az = math.degrees(math.atan2(rc.ve, rc.vn)) % 360.0
        _fv = _interp_flp(rc.utc_s)

        t_arr.append(round(rc.utc_s - t0, 3))
        dop_arr.append(round(ds, 4) if math.isfinite(ds) else None)
        crd_arr.append(round(cs, 4))
        flp_arr.append(round(_fv, 4) if _fv is not None else None)
        disag_arr.append(round(abs(ds - cs), 4) if math.isfinite(ds) else None)
        az_arr.append(round(az, 2) if math.isfinite(az) else None)
        q_arr.append(int(rc.quality))

    data_js = json.dumps({
        "t":     t_arr,
        "dop":   dop_arr,
        "crd":   crd_arr,
        "flp":   flp_arr,
        "disag": disag_arr,
        "az":    az_arr,
        "q":     q_arr,
    })

    template = (_ASSETS_DIR / "velocity_viewer.html").read_text(encoding="utf-8")
    html = template.replace("__VELOCITY_DATA__", data_js)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    log_(f"[velocity] wrote {out_html} ({len(t_arr)} epochs)")
    return VelocityViewerResult(html_path=out_html, js_path=js, epoch_count=len(t_arr))


# -----------------------------
# Media-PTS vs Signal-UTC viewer (recording_*.txt anchor regression)
# -----------------------------


@dataclass(frozen=True)
class VideoTimeViewerResult:
    html_path: Path
    js_path: Path
    n_anchors: int
    drift_ppm: float
    rmse_ms: float
    fit_uncertainty_ms: float


def build_video_time_viewer(
    *,
    recording_map: Path,
    out_html: Path,
    envelope_samples: int = 400,
    log: Optional[LogFn] = None,
) -> VideoTimeViewerResult:
    """Build ``video_time.html`` — regression diagnostics for the media PTS
    ↔ Signal UTC bridge encoded in ``recording_*.txt``.

    Plots per-anchor residuals (jitter), the OLS fit-σ envelope across the
    session span, and a residual histogram. Header pills surface drift_ppm,
    RMSE, max|res|, cubic-fit improvement, and the OLS centroid σ.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    anchor = fit_time_anchor(recording_map)
    residuals = per_anchor_residuals(recording_map, anchor)
    if not residuals:
        raise RuntimeError(f"No anchor rows in {recording_map}")

    t_arr      = [round(t, 3) for t, _ in residuals]
    resid_ms   = [round(r * 1e3, 4) for _, r in residuals]
    span_s     = t_arr[-1] - t_arr[0] if len(t_arr) >= 2 else 0.0

    # Fit-uncertainty envelope: sample the OLS σ(x) across the span.
    if envelope_samples < 2:
        envelope_samples = 2
    unc_t: list[float] = []
    unc_ms: list[float] = []
    if span_s > 0:
        v0 = t_arr[0] * 1e9
        v1 = t_arr[-1] * 1e9
        step = (v1 - v0) / (envelope_samples - 1)
        for i in range(envelope_samples):
            vns = v0 + i * step
            u = anchor.fit_uncertainty_s_at(vns)
            if not math.isfinite(u):
                continue
            unc_t.append(round(vns / 1e9, 3))
            unc_ms.append(round(u * 1e3, 6))

    data_js = json.dumps({
        "t":         t_arr,
        "resid_ms":  resid_ms,
        "unc_t":     unc_t,
        "unc_ms":    unc_ms,
        "rmse_ms":   anchor.rmse_s * 1e3,
    })

    template = (_ASSETS_DIR / "video_time.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA__", data_js)
        .replace("__N__", str(anchor.n))
        .replace("__NREJ__", str(anchor.n_rejected))
        .replace("__SPAN_S__", f"{span_s:.1f}")
        .replace("__DRIFT_PPM__", f"{anchor.drift_ppm:+.2f}")
        .replace("__RMSE_MS__", f"{anchor.rmse_s * 1e3:.2f}")
        .replace("__MAX_MS__", f"{anchor.max_abs_s * 1e3:.1f}")
        .replace("__FIT_MS__", f"{anchor.fit_uncertainty_s * 1e3:.3f}")
        .replace("__CUBIC_MS__", f"{anchor.cubic_rmse_improvement_s * 1e3:.2f}")
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[video-time] wrote {out_html}  n={anchor.n} rej={anchor.n_rejected}  "
        f"drift={anchor.drift_ppm:+.2f}ppm  rmse={anchor.rmse_s*1e3:.2f}ms  "
        f"max={anchor.max_abs_s*1e3:.1f}ms  fit_sigma={anchor.fit_uncertainty_s*1e3:.3f}ms  "
        f"cubic_gain={anchor.cubic_rmse_improvement_s*1e3:.2f}ms"
    )
    return VideoTimeViewerResult(
        html_path=out_html, js_path=js,
        n_anchors=anchor.n,
        drift_ppm=anchor.drift_ppm,
        rmse_ms=anchor.rmse_s * 1e3,
        fit_uncertainty_ms=anchor.fit_uncertainty_s * 1e3,
    )


# -----------------------------
# INS overlay viewer (raw Post-processing vs FLP vs forward-EKF vs RTS-smoothed)
# -----------------------------


@dataclass(frozen=True)
class InsViewerResult:
    html_path: Path
    js_path: Path
    n_ppk: int
    n_flp: int
    n_fwd: int
    n_rts: int
    horiz_rms_m: float
    vert_rms_m: float


def _decimate_indices(n: int, max_pts: int) -> list[int]:
    """Pick ~max_pts evenly-spaced indices from [0, n) including endpoints."""
    if n <= max_pts:
        return list(range(n))
    step = n / max_pts
    return sorted({int(round(i * step)) for i in range(max_pts)} | {n - 1})


def build_ins_viewer(
    *,
    sensors_txt: Path,
    pos_file: Path,
    out_html: Path,
    data_log: Optional[Path] = None,
    vehicle_mode: bool = False,
    chi2_gate_pos: float = 0.0,
    chi2_gate_vel: float = 0.0,
    max_pts_per_trace: int = 4000,
    log: Optional[LogFn] = None,
) -> InsViewerResult:
    """Build ``ins_viewer.html`` — overlays raw Post-processing, FLP (optional), forward
    EKF, and RTS-smoothed paths so the user can SEE where the INS
    pipeline beats vs. loses against Post-processing + FLP. ZUPT and gated-out Post-processing
    epochs are marker-overlaid for context.
    """
    log_ = make_logger(log)
    from ..parsers import parse_imu, parse_data_fix
    from .ekf_fusion import EkfOptions, run_ekf, rts_smooth

    out_html = out_html.resolve()
    imu = parse_imu(sensors_txt)
    pos = parse_rtkpos(pos_file)
    if not imu:
        raise RuntimeError(f"No IMU rows in {sensors_txt}")
    if not pos:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    opts = EkfOptions(nhc_enabled=vehicle_mode,
                      chi2_gate_pos=chi2_gate_pos,
                      chi2_gate_vel=chi2_gate_vel)
    fwd = run_ekf(imu, pos, options=opts, log=log_)
    ref_llh = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    sm = rts_smooth(fwd, ref_llh)
    log_(f"[ins-viewer] forward emitted {len(fwd.fused)} rows; "
         f"RTS smoothed {len(sm.fused)}")

    t0_utc = pos[0].utc_s

    def _series_from_posrows(rows):
        idxs = _decimate_indices(len(rows), max_pts_per_trace)
        out_t, out_e, out_n, out_u, out_s = [], [], [], [], []
        for i in idxs:
            r = rows[i]
            ex, ny, uz = ecef_to_enu(*llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m),
                                     ref_llh)
            spd = (math.hypot(r.ve, r.vn)
                   if (math.isfinite(r.ve) and math.isfinite(r.vn))
                   else float("nan"))
            out_t.append(round(r.utc_s - t0_utc, 3))
            out_e.append(round(ex, 3))
            out_n.append(round(ny, 3))
            out_u.append(round(uz, 3))
            out_s.append(round(spd, 4) if math.isfinite(spd) else None)
        return {
            "t": out_t, "e": out_e, "n": out_n, "u": out_u, "speed": out_s,
        }

    ppk_series = _series_from_posrows(pos)
    fwd_series = _series_from_posrows(fwd.fused)
    rts_series = _series_from_posrows(sm.fused)

    flp_series = None
    n_flp = 0
    if data_log is not None and data_log.is_file():
        try:
            fixes = parse_data_fix(data_log)
            flp_set = {"fused", "fused_location_provider", "fused_location"}
            flp_fixes = [f for f in fixes if (f.provider or "").lower() in flp_set]
            n_flp = len(flp_fixes)
            if flp_fixes:
                idxs = _decimate_indices(len(flp_fixes), max_pts_per_trace)
                t_, e_, n_, u_, s_ = [], [], [], [], []
                for i in idxs:
                    f = flp_fixes[i]
                    ex, ny, uz = ecef_to_enu(
                        *llh_to_ecef(f.lat, f.lon, f.h), ref_llh,
                    )
                    t_.append(round(f.utc_s - t0_utc, 3))
                    e_.append(round(ex, 3))
                    n_.append(round(ny, 3))
                    u_.append(round(uz, 3))
                    s_.append(round(float(f.speed_mps), 4)
                              if math.isfinite(f.speed_mps) else None)
                flp_series = {"t": t_, "e": e_, "n": n_, "u": u_, "speed": s_}
        except Exception as ex:
            log_(f"[ins-viewer] FLP parse failed ({ex})")

    # RTS↔Post-processing residual stats.
    rts_t = [r.utc_s for r in sm.fused]
    horiz_sq = 0.0; vert_sq = 0.0; n_res = 0
    for pr in pos:
        i = bisect_left(rts_t, pr.utc_s)
        if i <= 0 or i >= len(rts_t):
            continue
        a, b = sm.fused[i - 1], sm.fused[i]
        dt = b.utc_s - a.utc_s
        alpha = 0.0 if dt <= 0 else (pr.utc_s - a.utc_s) / dt
        la = a.lat_deg + alpha * (b.lat_deg - a.lat_deg)
        lo = a.lon_deg + alpha * (b.lon_deg - a.lon_deg)
        hh = a.h_m   + alpha * (b.h_m   - a.h_m)
        ex, ey, ez = ecef_to_enu(*llh_to_ecef(la, lo, hh), ref_llh)
        rx, ry, rz = ecef_to_enu(*llh_to_ecef(pr.lat_deg, pr.lon_deg, pr.h_m),
                                 ref_llh)
        horiz_sq += (ex - rx) ** 2 + (ey - ry) ** 2
        vert_sq  += (ez - rz) ** 2
        n_res += 1
    horiz_rms = math.sqrt(horiz_sq / n_res) if n_res else float("nan")
    vert_rms  = math.sqrt(vert_sq  / n_res) if n_res else float("nan")

    zupt_t = [round(t - t0_utc, 3) for t in fwd.zupt_t]
    rej_t  = [round(t - t0_utc, 3) for t in fwd.rejected_t]

    data_js = json.dumps({
        "ppk":        ppk_series,
        "flp":        flp_series,
        "fwd":        fwd_series,
        "rts":        rts_series,
        "zupt_t":     zupt_t,
        "rejected_t": rej_t,
    })

    template = (_ASSETS_DIR / "ins_viewer.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA__", data_js)
        .replace("__N_FRAMES__", str(len(rts_series["t"])))
        .replace("__N_POS_OK__",  str(fwd.n_pos_updates))
        .replace("__N_POS_TOT__", str(len(pos)))
        .replace("__N_POS_REJ__", str(fwd.n_pos_rejected))
        .replace("__N_ZUPT__",    str(fwd.n_zupt))
        .replace("__N_NHC__",     str(fwd.n_nhc))
        .replace("__HORIZ_RMS__", f"{horiz_rms:.2f}")
        .replace("__VERT_RMS__",  f"{vert_rms:.2f}")
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(f"[ins-viewer] wrote {out_html}  PPK={len(pos)}  FLP={n_flp}  "
         f"FWD={len(fwd.fused)}  RTS={len(sm.fused)}  "
         f"horiz_rms={horiz_rms:.2f}m  vert_rms={vert_rms:.2f}m")

    return InsViewerResult(
        html_path=out_html, js_path=js,
        n_ppk=len(pos), n_flp=n_flp,
        n_fwd=len(fwd.fused), n_rts=len(sm.fused),
        horiz_rms_m=horiz_rms, vert_rms_m=vert_rms,
    )


# -----------------------------
# Time-math explainer (annotated walkthrough of the OLS fit)
# -----------------------------


@dataclass(frozen=True)
class TimeMathViewerResult:
    html_path: Path
    js_path: Path
    n_anchors: int
    drift_ppm: float
    rmse_ms: float


def build_time_math_viewer(
    *,
    recording_map: Path,
    out_html: Path,
    envelope_samples: int = 400,
    log: Optional[LogFn] = None,
) -> TimeMathViewerResult:
    """Build ``time_math.html`` — single-page explainer that *shows the math*
    of :func:`time_sync.fit_time_anchor` on the user's own anchors.

    Each panel on the right is annotated with the actual numbers (xmean,
    ymean, slope, RMSE, cubic gain, σ at centroid) so the reader can verify
    step-by-step what the code did to their session. Main plot lays the
    centred anchors against the OLS line + ±RMSE band + fit-σ envelope.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    anchor = fit_time_anchor(recording_map)
    residuals = per_anchor_residuals(recording_map, anchor)
    if not residuals:
        raise RuntimeError(f"No anchor rows in {recording_map}")

    # Re-derive (x_centered, y_centered) from raw anchors for display.
    raw_pairs: list[tuple[float, float]] = []
    with recording_map.open("r", encoding="utf-8") as f:
        for ln in f:
            parts = [p.strip() for p in ln.strip().split(",")]
            if len(parts) < 2:
                continue
            try:
                video_ns = int(parts[0])
                from ..time_sync import _parse_utc_seconds  # type: ignore[attr-defined]
                utc_s = _parse_utc_seconds(parts[1])
            except (ValueError, IndexError):
                continue
            raw_pairs.append((float(video_ns), utc_s))

    x_centered_s: list[float] = []
    y_centered_ms: list[float] = []
    for vns, us in raw_pairs:
        x_centered_s.append(round((vns - anchor.xmean) / 1e9, 3))
        y_centered_ms.append(round((us - anchor.ymean) * 1e3, 4))

    span_s = (
        x_centered_s[-1] - x_centered_s[0]
        if len(x_centered_s) >= 2 else 0.0
    )

    # σ envelope, scaled ×1000 so it shows on the same axis as residuals.
    unc_t: list[float] = []
    unc_ms: list[float] = []
    if span_s > 0 and envelope_samples >= 2:
        v0 = raw_pairs[0][0]
        v1 = raw_pairs[-1][0]
        step = (v1 - v0) / (envelope_samples - 1)
        for i in range(envelope_samples):
            vns = v0 + i * step
            u = anchor.fit_uncertainty_s_at(vns)
            if not math.isfinite(u):
                continue
            unc_t.append(round((vns - anchor.xmean) / 1e9, 3))
            unc_ms.append(round(u * 1e3 * 1000.0, 4))  # ×1000 for visibility

    resid_ms = [round(r * 1e3, 4) for _, r in residuals]

    data_js = json.dumps({
        "n":               anchor.n,
        "x_centered_s":    x_centered_s,
        "y_centered_ms":   y_centered_ms,
        "resid_ms":        resid_ms,
        "unc_t":           unc_t,
        "unc_ms":          unc_ms,
        "slope_ns_per_ns": anchor.slope,
        "drift_ppm":       anchor.drift_ppm,
        "rmse_ms":         anchor.rmse_s * 1e3,
    })

    template = (_ASSETS_DIR / "time_math.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA__", data_js)
        .replace("__N__", str(anchor.n))
        .replace("__NREJ__", str(anchor.n_rejected))
        .replace("__SPAN_S__", f"{span_s:.1f}")
        .replace("__XMEAN__", f"{anchor.xmean:.3e}")
        .replace("__YMEAN__", f"{anchor.ymean:.3f}")
        .replace("__DRIFT_PPM__", f"{anchor.drift_ppm:+.3f}")
        .replace("__RMSE_MS__", f"{anchor.rmse_s * 1e3:.2f}")
        .replace("__MAX_MS__", f"{anchor.max_abs_s * 1e3:.1f}")
        .replace("__FIT_MS__", f"{anchor.fit_uncertainty_s * 1e3:.3f}")
        .replace("__CUBIC_MS__", f"{anchor.cubic_rmse_improvement_s * 1e3:.2f}")
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[time-math] wrote {out_html}  n={anchor.n}  drift={anchor.drift_ppm:+.3f}ppm  "
        f"rmse={anchor.rmse_s*1e3:.2f}ms  cubic_gain={anchor.cubic_rmse_improvement_s*1e3:.2f}ms"
    )
    return TimeMathViewerResult(
        html_path=out_html, js_path=js,
        n_anchors=anchor.n,
        drift_ppm=anchor.drift_ppm,
        rmse_ms=anchor.rmse_s * 1e3,
    )


# -----------------------------
# Clock-bias viewer (device system + Signal HW clock vs Reference time)
# -----------------------------


# Reference epoch in POSIX seconds (1980-01-06 00:00:00 UTC).
_GPS_EPOCH_POSIX_S: float = 315964800.0


@dataclass(frozen=True)
class ClockBiasViewerResult:
    html_path: Path
    js_path: Path
    n_epochs: int
    sys_minus_gps_median_ms: float
    sys_minus_gps_std_ms: float
    drift_median_ppb: float


def _parse_clock_bias_epochs(measurements_txt: Path) -> list[dict]:
    """One row per *unique epoch* (TimeNanos) from a the source app Raw stream.

    All sources in an epoch share TimeNanos / utcTimeMillis / FullBiasNanos
    / BiasNanos / Drift*, so we keep the first occurrence and skip the rest.
    """
    HEADER = (
        "utcTimeMillis", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos",
        "FullBiasNanos", "BiasNanos", "BiasUncertaintyNanos",
        "DriftNanosPerSecond", "DriftUncertaintyNanosPerSecond",
        "HardwareClockDiscontinuityCount",
    )
    idx_by_name: dict[str, int] = {}
    out: list[dict] = []
    seen: set[int] = set()
    with measurements_txt.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Header rows look like:
                # "# Raw,utcTimeMillis,TimeNanos,..."
                if "Raw," in line and not idx_by_name:
                    cols = [c.strip() for c in line.lstrip("#").strip().split(",")]
                    for k in HEADER:
                        if k in cols:
                            idx_by_name[k] = cols.index(k)
                continue
            if not line.startswith("Raw,"):
                continue
            if not idx_by_name:
                continue
            parts = line.split(",")
            try:
                tn = int(parts[idx_by_name["TimeNanos"]])
            except (ValueError, IndexError):
                continue
            if tn in seen:
                continue
            seen.add(tn)

            def _get(name: str, default: float = float("nan")) -> float:
                try:
                    s = parts[idx_by_name[name]].strip()
                    if not s:
                        return default
                    return float(s)
                except (KeyError, ValueError, IndexError):
                    return default

            def _geti(name: str, default: int = 0) -> int:
                try:
                    s = parts[idx_by_name[name]].strip()
                    if not s:
                        return default
                    return int(float(s))
                except (KeyError, ValueError, IndexError):
                    return default

            out.append({
                "utc_ms":     _get("utcTimeMillis"),
                "time_ns":    float(tn),
                "leap":       _geti("LeapSecond", 0),
                "fbn":        _get("FullBiasNanos"),
                "bn":         _get("BiasNanos", 0.0),
                "bun":        _get("BiasUncertaintyNanos", 0.0),
                "dnps":       _get("DriftNanosPerSecond", 0.0),
                "dunps":      _get("DriftUncertaintyNanosPerSecond", 0.0),
                "hw_disc":    _geti("HardwareClockDiscontinuityCount", 0),
            })
    out.sort(key=lambda r: r["utc_ms"])
    return out


def build_clock_bias_viewer(
    *,
    measurements_txt: Path,
    out_html: Path,
    leap_seconds: Optional[float] = None,
    log: Optional[LogFn] = None,
) -> ClockBiasViewerResult:
    """Build ``clock_bias.html`` from a the source app ``measurements_*.txt``.

    Decomposes the device↔Reference clock relationship into three signals:

      * **sys − Reference (ms)** — The platform system clock minus Reference-derived UTC,
        per Raw epoch. This is the offset any app stamping
        ``System.currentTimeMillis()`` carries vs. Reference truth.
      * **drift (ppb)** — ``DriftNanosPerSecond`` of the Signal HW counter
        relative to Reference time.
      * **HW discontinuity events** — overlaid as vertical dashes whenever
        the chip's ``HardwareClockDiscontinuityCount`` increments.

    ``leap_seconds`` defaults to the table in :mod:`time_sync` evaluated at
    the first epoch's UTC.
    """
    log_ = make_logger(log)
    out_html = out_html.resolve()
    rows = _parse_clock_bias_epochs(measurements_txt)
    if not rows:
        raise RuntimeError(
            f"No usable Raw rows in {measurements_txt} "
            "(header line missing or FullBiasNanos column absent)."
        )

    if leap_seconds is None:
        from ..time_sync import get_leap_seconds_for_epoch
        leap_seconds = get_leap_seconds_for_epoch(rows[0]["utc_ms"] / 1000.0)
    leap_s = float(leap_seconds)

    t0_utc_s = rows[0]["utc_ms"] / 1000.0

    t_arr:           list[float] = []
    sys_bias_ms:     list[float] = []
    sys_unc_ms:      list[float] = []
    drift_ppb:       list[float] = []
    drift_unc_ppb:   list[float] = []
    disc_t:          list[float] = []

    last_disc = rows[0]["hw_disc"]
    for r in rows:
        # Reference time (POSIX UTC) derived from the Signal HW counter:
        # GpsTimeNanos = TimeNanos − (FullBiasNanos + BiasNanos)
        # GPS_UTC      = GpsTimeNanos·1e-9 + GPS_EPOCH_POSIX − leap_s
        if not math.isfinite(r["fbn"]):
            continue
        gps_time_ns = r["time_ns"] - (r["fbn"] + r["bn"])
        gps_utc_s   = gps_time_ns * 1e-9 + _GPS_EPOCH_POSIX_S - leap_s
        sys_utc_s   = r["utc_ms"] / 1000.0

        bias_s = sys_utc_s - gps_utc_s
        unc_s  = abs(r["bun"]) * 1e-9

        # Drift normalised: DriftNanosPerSecond ÷ 1 s ⇒ ns / 1e9 s ⇒ ppb.
        drift_p = r["dnps"]
        drift_u = abs(r["dunps"])

        t_arr.append(round(sys_utc_s - t0_utc_s, 3))
        sys_bias_ms.append(round(bias_s * 1e3, 4))
        sys_unc_ms.append(round(unc_s * 1e3, 4))
        drift_ppb.append(round(drift_p, 3))
        drift_unc_ppb.append(round(drift_u, 3))

        if r["hw_disc"] != last_disc:
            disc_t.append(round(sys_utc_s - t0_utc_s, 3))
            last_disc = r["hw_disc"]

    if not t_arr:
        raise RuntimeError(
            f"All Raw rows in {measurements_txt} lacked FullBiasNanos."
        )

    n = len(sys_bias_ms)
    sorted_b = sorted(sys_bias_ms)
    med = sorted_b[n // 2]
    mean = sum(sys_bias_ms) / n
    var  = sum((v - mean) ** 2 for v in sys_bias_ms) / n
    std  = math.sqrt(var)

    sorted_d = sorted(drift_ppb)
    drift_med = sorted_d[len(sorted_d) // 2] if sorted_d else 0.0

    span_s = (rows[-1]["utc_ms"] - rows[0]["utc_ms"]) / 1000.0
    n_disc = len(disc_t)

    data_js = json.dumps({
        "t":               t_arr,
        "sys_bias_ms":     sys_bias_ms,
        "sys_unc_ms":      sys_unc_ms,
        "drift_ppb":       drift_ppb,
        "drift_unc_ppb":   drift_unc_ppb,
        "drift_median_ppb": drift_med,
        "disc_t":          disc_t,
    })

    template = (_ASSETS_DIR / "clock_bias.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA__", data_js)
        .replace("__N_EPOCHS__", str(n))
        .replace("__SPAN_S__", f"{span_s:.1f}")
        .replace("__LEAP_S__", f"{leap_s:g}")
        .replace("__SYS_MED_MS__", f"{med:+.2f}")
        .replace("__SYS_STD_MS__", f"{std:.2f}")
        .replace("__DRIFT_PPB__", f"{drift_med:+.1f}")
        .replace("__DISCS__", str(n_disc))
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)

    log_(
        f"[clock-bias] wrote {out_html}  epochs={n}  span={span_s:.1f}s  "
        f"sys-GPS med={med:+.2f}ms std={std:.2f}ms  drift med={drift_med:+.1f}ppb  "
        f"HW disc.={n_disc}  leap_s={leap_s:g}"
    )
    return ClockBiasViewerResult(
        html_path=out_html, js_path=js,
        n_epochs=n,
        sys_minus_gps_median_ms=med,
        sys_minus_gps_std_ms=std,
        drift_median_ppb=drift_med,
    )


@dataclass(frozen=True)
class TrustViewerResult:
    html_path: Path
    js_path: Path
    n_frames: int
    trust_median: float
    trust_p10: float


def _read_trust_sidecar(path: Path) -> list[tuple[str, float, float, float, float]]:
    """Parse the trust sidecar CSV written by georef.py."""
    out: list[tuple[str, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(_noncomment_lines(f)):
            try:
                la = float(row["Latitude"])
                lo = float(row["Longitude"])
                hh = float(row.get("Altitude", "0") or 0)
                tr = float(row["Trust"])
            except (KeyError, ValueError):
                continue
            out.append((row["Image"], la, lo, hh, tr))
    return out


def build_trust_viewer(
    *,
    trust_csv: Path,
    out_html: Path,
    log: Optional[LogFn] = None,
) -> TrustViewerResult:
    """Build ``trust_viewer.html`` painting the path red->green by
    per-sample Post-processing trust (from the fused-bent sidecar).

    Red = the bend ignored Post-processing at this sample (output is pure FLP shape).
    Green = the bend trusted Post-processing fully (output sits on the anchor cloud).
    """
    log_ = make_logger(log)
    rows = _read_trust_sidecar(trust_csv)
    if not rows:
        raise RuntimeError(f"No usable rows in {trust_csv}")

    ref_lat, ref_lon, ref_h = rows[0][1], rows[0][2], rows[0][3]
    es, ns, us = llh_iterable_to_enu(
        ((r[1], r[2], r[3]) for r in rows), (ref_lat, ref_lon, ref_h),
    )
    images = [r[0] for r in rows]
    trust = [r[4] for r in rows]

    trust_sorted = sorted(trust)
    n = len(trust_sorted)
    def _pct(p: float) -> float:
        if n == 0:
            return float("nan")
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        return trust_sorted[idx]
    med = _pct(50)
    p10 = _pct(10)
    high_frac = sum(1 for v in trust if v >= 0.5) / n if n else 0.0
    low_frac = sum(1 for v in trust if v <= 0.1) / n if n else 0.0

    data_js = json.dumps({
        "image": images,
        "trust": trust,
        "e": es,
        "n": ns,
        "u": us,
    })

    template = (_ASSETS_DIR / "trust.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__DATA__", data_js)
        .replace("__NFRAMES__", str(n))
        .replace("__TRUST_MED__", f"{med:.3f}")
        .replace("__TRUST_P10__", f"{p10:.3f}")
        .replace("__TRUST_HIGH__", f"{high_frac * 100:.1f}%")
        .replace("__TRUST_LOW__", f"{low_frac * 100:.1f}%")
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    log_(
        f"[trust] wrote {out_html}  frames={n}  trust median={med:.3f} "
        f"p10={p10:.3f}  >=0.5: {high_frac * 100:.1f}%  "
        f"<=0.1: {low_frac * 100:.1f}%"
    )
    return TrustViewerResult(
        html_path=out_html, js_path=js, n_frames=n,
        trust_median=med, trust_p10=p10,
    )


def build_smoothed_trust_viewer(
    *,
    raw_pos_rows,
    smoothed_pos_rows,
    out_html: Path,
    sigma_disagree_m: float = 2.0,
    log: Optional[LogFn] = None,
) -> TrustViewerResult:
    """Paint a smoothed-path path coloured by Motion model-Post-processing agreement.

    For each epoch, computes 2D horizontal disagreement
    ``d = ||smoothed_enu - raw_enu||`` then maps to a trust scalar
    ``t = exp(-d^2 / (2 sigma^2))``. Green (1.0) = filter trusted Post-processing
    fully (smoothed sits on raw Post-processing), blue (0.0) = filter disagreed
    with Post-processing at this epoch.

    Args:
      raw_pos_rows:      raw The external solver PosRow list (the Post-processing input).
      smoothed_pos_rows: filter output PosRow list (same length, same
                         utc_s ordering).
      out_html:          where to write the HTML.
      sigma_disagree_m:  controls the green->blue ramp. d=sigma -> trust=0.61.

    Returns: TrustViewerResult.
    """
    import math as _math
    from ..geo import ecef_to_enu, llh_to_ecef
    log_ = make_logger(log)

    if not raw_pos_rows or not smoothed_pos_rows:
        raise ValueError("build_smoothed_trust_viewer: empty input rows.")
    if len(raw_pos_rows) != len(smoothed_pos_rows):
        raise ValueError(
            f"length mismatch: raw={len(raw_pos_rows)} smoothed={len(smoothed_pos_rows)}; "
            "smoother must emit one row per PPK epoch."
        )

    ref_lat = raw_pos_rows[0].lat_deg
    ref_lon = raw_pos_rows[0].lon_deg
    ref_h = raw_pos_rows[0].h_m
    ref_llh = (ref_lat, ref_lon, ref_h)

    def _enu(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref_llh)

    es: list[float] = []; ns: list[float] = []; us: list[float] = []
    trust: list[float] = []
    disagree_m: list[float] = []
    for r_raw, r_sm in zip(raw_pos_rows, smoothed_pos_rows):
        e_s, n_s, u_s = _enu(r_sm)
        e_r, n_r, u_r = _enu(r_raw)
        d = _math.hypot(e_s - e_r, n_s - n_r)
        t = _math.exp(-d * d / (2.0 * max(1e-6, sigma_disagree_m) ** 2))
        es.append(float(e_s)); ns.append(float(n_s)); us.append(float(u_s))
        disagree_m.append(float(d)); trust.append(float(t))

    n_pts = len(trust)
    trust_sorted = sorted(trust)
    def _pct(p):
        if n_pts == 0: return float("nan")
        idx = max(0, min(n_pts - 1, int(round(p / 100 * (n_pts - 1)))))
        return trust_sorted[idx]
    med = _pct(50); p10 = _pct(10)

    data = {
        "e": es, "n": ns, "u": us,
        "trust": trust, "disagree_m": disagree_m,
        "ref": {"lat": ref_lat, "lon": ref_lon, "h": ref_h},
        "sigma_disagree_m": sigma_disagree_m,
        "trust_p10": p10, "trust_p50": med,
    }

    out_html = Path(out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    js_path = out_html.with_suffix(".data.js")
    _copy_plotly_next_to(out_html.parent)
    js_path.write_text(
        "window.TRUST_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Smoothed path: trust vs PPK</title>
<script src="plotly.min.js"></script>
<script src="__JS__"></script>
<style>
html,body{margin:0;background:#0b0f17;color:#d8e0ee;font-family:sans-serif;font-size:13px}
h1{margin:10px 14px;font-size:16px;color:#e5e7eb}
.note{padding:4px 14px;color:#94a3b8;font-size:12px;max-width:1200px}
.row{display:flex;flex-wrap:wrap;gap:8px;padding:6px 14px}
.pill{background:#1f2937;border:1px solid #374151;border-radius:6px;padding:4px 10px;font-size:12px}
.pill b{color:#e5e7eb}
.plot{width:100vw;height:50vh}
</style></head><body>
<h1>Smoothed trajectory — trust vs raw PPK</h1>
<div class="note">
Each epoch coloured by <b style="color:#22d3ee">trust = exp(&minus;d²/2σ²)</b> where d = ||smoothed&minus;raw|| in ENU.
Green = smoother agreed with PPK at this epoch. Blue = smoother disagreed
(it overruled PPK with prior / IMU / VIO / smoother dynamics).
</div>
<div class="row" id="stats"></div>
<div id="plot" class="plot"></div>
<div id="plot_dis" class="plot"></div>
<script>
const D = window.TRUST_DATA;
const stats = document.getElementById('stats');
function pill(k,v){const el=document.createElement('div');el.className='pill';el.innerHTML='<b>'+k+':</b> '+v;stats.appendChild(el);}
pill('n epochs', D.e.length);
pill('trust P50', D.trust_p50.toFixed(3));
pill('trust P10', D.trust_p10.toFixed(3));
pill('sigma (m)', D.sigma_disagree_m);
pill('ref lat', D.ref.lat.toFixed(7));

Plotly.newPlot('plot', [{
  x: D.e, y: D.n, mode: 'markers',
  marker: {
    size: 4,
    color: D.trust, cmin: 0, cmax: 1,
    colorscale: [[0,'#3b82f6'],[0.5,'#60a5fa'],[1,'#22c55e']],
    colorbar: {title:'trust', tickvals:[0,0.5,1], ticktext:['blue (disagrees)','','green (trusts PPK)']}
  },
  text: D.disagree_m.map(d => 'disagree=' + d.toFixed(2) + ' m'),
  hovertemplate: 'E=%{x:.1f}m N=%{y:.1f}m<br>%{text}<extra></extra>',
}], {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'East (m)', gridcolor:'#1f2937', scaleanchor:'y'},
  yaxis:{title:'North (m)', gridcolor:'#1f2937'},
  margin:{t:18,r:30,b:48,l:60},
});

// Histogram of disagreement.
Plotly.newPlot('plot_dis', [{
  x: D.disagree_m, type: 'histogram', nbinsx: 60,
  marker:{color:'#60a5fa'},
}], {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  title: {text:'Per-epoch ||smoothed - raw PPK|| (m)', font:{color:'#e5e7eb'}},
  xaxis:{title:'disagreement (m)', gridcolor:'#1f2937'},
  yaxis:{title:'count', gridcolor:'#1f2937'},
  margin:{t:40,r:30,b:48,l:60},
});
</script>
</body></html>
""".replace("__JS__", js_path.name)
    tmp = Path(str(out_html) + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    import os as _os
    _os.replace(tmp, out_html)
    log_(f"[trust-viewer] wrote {out_html} (n={n_pts}, p50={med:.3f}, p10={p10:.3f})")
    return TrustViewerResult(
        html_path=out_html, js_path=js_path, n_frames=n_pts,
        trust_median=med, trust_p10=p10,
    )


@dataclass
class TrajectoryCompareResult:
    html_path: Path
    n_routes: int
    n_epochs: int
    pairwise_stats: dict  # {"(a, b)": {"rmse": ..., "p95": ..., "max": ...}}


def build_trajectory_compare_viewer(
    *,
    routes: dict,
    out_html: Path,
    log: Optional[LogFn] = None,
) -> "TrajectoryCompareResult":
    """Comprehensive path comparison panel — no GT needed.

    ``routes`` is ``{label: list[PosRow]}``. All routes must share the
    same length (one row per Post-processing epoch). First route's first row sets
    the Local-frame origin.

    Panels:
      1. Session pills (route count, epoch count, ref lat/lon).
      2. 2D Local-frame path — all routes as legend-toggleable Plotly traces.
      3. Per-axis disagreement vs reference route over time (E, N, U).
      4. Speed profile per route (||v|| derived from finite-diff).
      5. Pairwise disagreement matrix (RMSE / P50 / P95 / MAX in metres).
      6. Time-series ||route_i − route_j||₂ for the most-divergent pair.

    Use to answer: "which two smoothers diverge most, when, and on what axis?"
    """
    from ..geo import ecef_to_enu, llh_to_ecef
    log_ = make_logger(log)
    if not routes:
        raise ValueError("build_trajectory_compare_viewer: empty routes dict")
    labels = list(routes.keys())
    first_label = labels[0]
    if not routes[first_label]:
        raise ValueError(f"first route '{first_label}' is empty.")
    n = len(routes[first_label])
    for lbl, rows in routes.items():
        if len(rows) != n:
            raise ValueError(
                f"all routes must have same length; '{lbl}' has {len(rows)} vs reference {n}."
            )

    ref_lat = routes[first_label][0].lat_deg
    ref_lon = routes[first_label][0].lon_deg
    ref_h = routes[first_label][0].h_m
    ref_llh = (ref_lat, ref_lon, ref_h)

    def _enu(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        return ecef_to_enu(x, y, z, ref_llh)

    enu_by_route: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lbl, rows in routes.items():
        es = np.array([_enu(r)[0] for r in rows])
        ns = np.array([_enu(r)[1] for r in rows])
        us = np.array([_enu(r)[2] for r in rows])
        enu_by_route[lbl] = (es, ns, us)

    ts = np.array([r.utc_s for r in routes[first_label]])
    ts_rel = (ts - ts[0]).tolist()

    palette = ["#9ca3af", "#22d3ee", "#22c55e", "#f59e0b", "#a855f7",
               "#3b82f6", "#10b981", "#ef4444", "#ec4899", "#06b6d4"]
    route_payload = []
    speed_payload = []
    for i, lbl in enumerate(labels):
        es, ns, us = enu_by_route[lbl]
        # Numerical speed from finite-diff position.
        if n > 1:
            dt = np.diff(ts, prepend=ts[0]); dt[0] = dt[1] if len(dt) > 1 else 1.0
            ve = np.diff(es, prepend=es[0]) / np.where(dt > 0, dt, 1.0)
            vn = np.diff(ns, prepend=ns[0]) / np.where(dt > 0, dt, 1.0)
            sp = np.sqrt(ve ** 2 + vn ** 2)
        else:
            sp = np.zeros(n)
        route_payload.append({
            "label": lbl, "color": palette[i % len(palette)],
            "e": es.tolist(), "n": ns.tolist(), "u": us.tolist(),
        })
        speed_payload.append({
            "label": lbl, "color": palette[i % len(palette)],
            "speed": sp.tolist(),
        })

    # Pairwise stats + per-axis disagreement vs first route.
    pairwise: dict[str, dict] = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            ea, na, ua = enu_by_route[la]
            eb, nb, ub = enu_by_route[lb]
            d2 = (ea - eb) ** 2 + (na - nb) ** 2
            d = np.sqrt(d2)
            pairwise[f"{la}__VS__{lb}"] = {
                "rmse_m": float(np.sqrt(np.mean(d2))),
                "p50_m": float(np.percentile(d, 50)),
                "p95_m": float(np.percentile(d, 95)),
                "max_m": float(np.max(d)),
            }

    # Per-axis disagreement vs FIRST route (reference).
    ref_e, ref_n, ref_u = enu_by_route[first_label]
    per_axis = []
    for lbl in labels[1:]:
        es, ns, us = enu_by_route[lbl]
        per_axis.append({
            "label": lbl, "color": palette[(labels.index(lbl)) % len(palette)],
            "dE": (es - ref_e).tolist(),
            "dN": (ns - ref_n).tolist(),
            "dU": (us - ref_u).tolist(),
        })

    # Most-divergent pair time-series (drives the "watch this pair" plot).
    if pairwise:
        worst_key = max(pairwise, key=lambda k: pairwise[k]["max_m"])
        worst_la, worst_lb = worst_key.split("__VS__")
        ea, na, _ = enu_by_route[worst_la]
        eb, nb, _ = enu_by_route[worst_lb]
        worst_d = np.sqrt((ea - eb) ** 2 + (na - nb) ** 2)
        worst_payload = {
            "pair": [worst_la, worst_lb],
            "d": worst_d.tolist(),
            "rmse": pairwise[worst_key]["rmse_m"],
            "max": pairwise[worst_key]["max_m"],
        }
    else:
        worst_payload = None

    out_html = Path(out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    js_path = out_html.with_suffix(".data.js")
    _copy_plotly_next_to(out_html.parent)
    payload = {
        "ref": {"lat": ref_lat, "lon": ref_lon, "h": ref_h, "n_epochs": n},
        "ts_rel": ts_rel,
        "routes": route_payload,
        "speeds": speed_payload,
        "per_axis": per_axis,
        "pairwise": pairwise,
        "worst": worst_payload,
        "first_label": first_label,
    }
    js_path.write_text(
        "window.TC = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Trajectory comparison — all smoothers</title>
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
.plot{width:100vw;height:48vh}
.plot.short{height:28vh}
table{border-collapse:collapse;margin:8px 14px;font-size:12px;color:#d8e0ee}
th,td{padding:5px 10px;border:1px solid #374151;text-align:right}
th{background:#1f2937;color:#e5e7eb}
td.lbl{background:#111827;text-align:left;color:#cbd5e1}
.bad{background:#7f1d1d;color:#fee2e2}
.warn{background:#78350f;color:#fde68a}
.ok{background:#064e3b;color:#bbf7d0}
</style></head><body>

<h1>Trajectory comparison — all smoothers (no GT)</h1>
<div class="note">
Reference route = first one supplied. Per-axis disagreement &amp; pairwise
table answer: which smoothers diverge, where, and on what axis.
</div>

<h2>Session summary</h2>
<div class="row" id="sum"></div>

<h2>2D ENU trajectory (legend-toggleable)</h2>
<div id="plot_xy" class="plot"></div>

<h2>Per-axis disagreement vs reference (E / N / U)</h2>
<div class="note">Each line = (route − reference) per epoch on one axis.
Flat → that smoother matches the reference on that axis.</div>
<div id="plot_axis" class="plot"></div>

<h2>Speed profile per route (finite-diff)</h2>
<div id="plot_speed" class="plot short"></div>

<h2>Pairwise disagreement table (metres)</h2>
<div class="note">RMSE / P50 / P95 / max of ‖route_a − route_b‖. Click table cells
to inspect; cells coloured by max (red = &gt;5 m).</div>
<div id="pair_table"></div>

<h2>Most-divergent pair over time</h2>
<div id="plot_worst" class="plot short"></div>

<script>
const D = window.TC;
const sum = document.getElementById('sum');
function pill(k, v, cls) {
  const el = document.createElement('div');
  el.className = 'pill' + (cls ? ' ' + cls : '');
  el.innerHTML = '<b>' + k + ':</b> ' + v;
  sum.appendChild(el);
}
pill('routes', D.routes.length);
pill('epochs', D.ref.n_epochs);
pill('reference route', D.first_label);
pill('ref lat', D.ref.lat.toFixed(7));
pill('ref lon', D.ref.lon.toFixed(7));

// 2D ENU
Plotly.newPlot('plot_xy', D.routes.map(r => ({
  x: r.e, y: r.n, mode:'lines', name: r.label,
  line:{color: r.color, width: 1.5},
  hovertemplate: r.label + '<br>E=%{x:.2f} N=%{y:.2f}<extra></extra>',
})), {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'East (m)', gridcolor:'#1f2937', scaleanchor:'y'},
  yaxis:{title:'North (m)', gridcolor:'#1f2937'},
  legend:{bgcolor:'rgba(11,15,23,0.7)'},
  margin:{t:18,r:30,b:48,l:60},
});

// Per-axis disagreement vs reference
const axisTraces = [];
for (const r of D.per_axis) {
  axisTraces.push({x: D.ts_rel, y: r.dE, mode:'lines', name: r.label + ' dE',
    line:{color: r.color, width:1, dash:'solid'}, legendgroup: r.label, yaxis:'y1'});
  axisTraces.push({x: D.ts_rel, y: r.dN, mode:'lines', name: r.label + ' dN',
    line:{color: r.color, width:1, dash:'dot'}, legendgroup: r.label, yaxis:'y2'});
  axisTraces.push({x: D.ts_rel, y: r.dU, mode:'lines', name: r.label + ' dU',
    line:{color: r.color, width:1, dash:'dash'}, legendgroup: r.label, yaxis:'y3'});
}
Plotly.newPlot('plot_axis', axisTraces, {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'session time (s)', gridcolor:'#1f2937', domain:[0, 0.95]},
  yaxis:{title:'dE (m)', gridcolor:'#1f2937', side:'left'},
  yaxis2:{title:'dN (m)', gridcolor:'#1f2937', overlaying:'y', side:'right'},
  yaxis3:{title:'dU (m)', overlaying:'y', side:'right', position:0.93, showgrid:false},
  legend:{orientation:'h', bgcolor:'rgba(11,15,23,0.7)'},
  margin:{t:18,r:90,b:48,l:60},
});

// Speed profile per route
Plotly.newPlot('plot_speed', D.speeds.map(s => ({
  x: D.ts_rel, y: s.speed, mode:'lines', name: s.label,
  line:{color: s.color, width: 1},
})), {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis:{title:'session time (s)', gridcolor:'#1f2937'},
  yaxis:{title:'speed (m/s)', gridcolor:'#1f2937'},
  legend:{orientation:'h', bgcolor:'rgba(11,15,23,0.7)'},
  margin:{t:18,r:30,b:48,l:60},
});

// Pairwise table
const labels = D.routes.map(r => r.label);
let html = '<table><tr><th></th>';
for (const lb of labels) html += '<th>' + lb + '</th>';
html += '</tr>';
function cls(v) {
  if (v > 5) return 'bad';
  if (v > 2) return 'warn';
  if (v > 0.5) return '';
  return 'ok';
}
for (const la of labels) {
  html += '<tr><td class="lbl">' + la + '</td>';
  for (const lb of labels) {
    if (la === lb) { html += '<td>—</td>'; continue; }
    const k1 = la + '__VS__' + lb;
    const k2 = lb + '__VS__' + la;
    const stat = D.pairwise[k1] || D.pairwise[k2];
    if (!stat) { html += '<td>—</td>'; continue; }
    const v = stat.max_m;
    html += '<td class="' + cls(v) + '" title="rmse=' + stat.rmse_m.toFixed(2)
            + 'm p95=' + stat.p95_m.toFixed(2) + 'm max=' + v.toFixed(2) + 'm">'
            + 'max ' + v.toFixed(2) + 'm</td>';
  }
  html += '</tr>';
}
html += '</table>';
html += '<div class="note">Hover cell → rmse/p95/max. Red=&gt;5m max, amber=&gt;2m, green=&lt;0.5m.</div>';
document.getElementById('pair_table').innerHTML = html;

// Worst pair time series
if (D.worst) {
  Plotly.newPlot('plot_worst', [{
    x: D.ts_rel, y: D.worst.d, mode:'lines',
    name: D.worst.pair[0] + ' vs ' + D.worst.pair[1],
    line:{color:'#f87171', width: 1.5},
  }], {
    paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
    title: {text: 'Worst pair: ' + D.worst.pair[0] + ' vs ' + D.worst.pair[1]
                  + '  (RMSE=' + D.worst.rmse.toFixed(2) + 'm, max='
                  + D.worst.max.toFixed(2) + 'm)', font:{color:'#e5e7eb'}},
    xaxis:{title:'session time (s)', gridcolor:'#1f2937'},
    yaxis:{title:'||a-b|| (m)', gridcolor:'#1f2937'},
    margin:{t:42,r:30,b:48,l:60},
  });
} else {
  document.getElementById('plot_worst').innerHTML =
    '<div class="note">Need ≥ 2 routes to show worst-pair time series.</div>';
}
</script>
</body></html>
""".replace("__JS__", js_path.name)
    tmp = Path(str(out_html) + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out_html)
    log_(
        f"[traj-compare] wrote {out_html} "
        f"({len(routes)} routes, {n} epochs, "
        f"{len(pairwise)} pairs)"
    )
    return TrajectoryCompareResult(
        html_path=out_html, n_routes=len(routes), n_epochs=n,
        pairwise_stats=pairwise,
    )


@dataclass
class RoutesViewerResult:
    html_path: Path
    n_routes: int
    n_total_points: int


def build_routes_viewer(
    *,
    routes: dict,
    out_html: Path,
    log: Optional[LogFn] = None,
) -> RoutesViewerResult:
    """Multi-route 2D path comparison viewer.

    ``routes`` is a dict ``{label: list[PosRow]}``. Each route renders as
    a Plotly scatter trace in Local-frame (about the FIRST route's first row).
    Legend entries toggle visibility. Useful for comparing raw Post-processing,
    gaussian, ns-adaptive, epoch-weighted, FGO, fused-bent, hybrid Motion model
    on a single panel.

    No reference required.
    """
    from ..geo import ecef_to_enu, llh_to_ecef
    log_ = make_logger(log)
    if not routes:
        raise ValueError("build_routes_viewer: empty routes dict.")
    first_label = next(iter(routes))
    first_rows = routes[first_label]
    if not first_rows:
        raise ValueError(
            f"build_routes_viewer: route '{first_label}' is empty. "
            "Need at least one row to set the ENU reference."
        )
    ref_lat = first_rows[0].lat_deg
    ref_lon = first_rows[0].lon_deg
    ref_h = first_rows[0].h_m
    ref_llh = (ref_lat, ref_lon, ref_h)

    def _enu_xy(r):
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, n, u = ecef_to_enu(x, y, z, ref_llh)
        return e, n, u

    palette = [
        "#22d3ee", "#22c55e", "#f59e0b", "#ef4444", "#a855f7",
        "#3b82f6", "#10b981", "#eab308", "#ec4899", "#06b6d4",
    ]
    payload = {"routes": [], "ref": {"lat": ref_lat, "lon": ref_lon, "h": ref_h}}
    n_total = 0
    for i, (label, rows) in enumerate(routes.items()):
        if not rows:
            log_(f"[routes-viewer] WARN route '{label}' is empty — skipping")
            continue
        es: list[float] = []; ns: list[float] = []; us: list[float] = []
        for r in rows:
            e, n, u = _enu_xy(r)
            es.append(float(e)); ns.append(float(n)); us.append(float(u))
        payload["routes"].append({
            "label": label,
            "color": palette[i % len(palette)],
            "e": es, "n": ns, "u": us,
            "n_points": len(es),
        })
        n_total += len(es)

    out_html = Path(out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)
    js_path = out_html.with_suffix(".data.js")
    _copy_plotly_next_to(out_html.parent)
    js_path.write_text(
        "window.ROUTES_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Trajectory routes comparison</title>
<script src="plotly.min.js"></script>
<script src="__JS__"></script>
<style>
html,body{margin:0;background:#0b0f17;color:#d8e0ee;font-family:sans-serif;font-size:13px}
h1{margin:10px 14px;font-size:16px;color:#e5e7eb}
.note{padding:4px 14px;color:#94a3b8;font-size:12px;max-width:1200px}
.row{display:flex;flex-wrap:wrap;gap:8px;padding:6px 14px}
.pill{background:#1f2937;border:1px solid #374151;border-radius:6px;padding:4px 10px;font-size:12px}
.pill b{color:#e5e7eb}
.plot{width:100vw;height:75vh}
</style></head><body>
<h1>Trajectory routes comparison</h1>
<div class="note">
Each route is a separate Plotly trace. Click legend entries to hide / show.
Equal-aspect ENU about the first route's first row.
</div>
<div class="row" id="stats"></div>
<div id="plot" class="plot"></div>
<script>
const D = window.ROUTES_DATA;
const stats = document.getElementById('stats');
function pill(k, v) {
  const el = document.createElement('div'); el.className = 'pill';
  el.innerHTML = '<b>' + k + ':</b> ' + v;
  stats.appendChild(el);
}
pill('routes', D.routes.length);
pill('ref lat', D.ref.lat.toFixed(7));
pill('ref lon', D.ref.lon.toFixed(7));
for (const r of D.routes) {
  pill(r.label, r.n_points + ' pts');
}
const traces = D.routes.map(r => ({
  x: r.e, y: r.n, mode: 'lines+markers',
  name: r.label,
  line: {color: r.color, width: 1.5},
  marker: {color: r.color, size: 3},
  hovertemplate: r.label + '<br>E=%{x:.1f} N=%{y:.1f}<extra></extra>',
}));
Plotly.newPlot('plot', traces, {
  paper_bgcolor:'#0b0f17', plot_bgcolor:'#0b0f17', font:{color:'#d8e0ee'},
  xaxis: {title:'East (m)', gridcolor:'#1f2937', scaleanchor:'y'},
  yaxis: {title:'North (m)', gridcolor:'#1f2937'},
  legend: {bgcolor:'rgba(11,15,23,0.7)', font:{color:'#d8e0ee'}},
  margin: {t:18, r:30, b:48, l:60},
});
</script>
</body></html>
""".replace("__JS__", js_path.name)
    tmp = Path(str(out_html) + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    import os as _os
    _os.replace(tmp, out_html)
    log_(
        f"[routes-viewer] wrote {out_html} "
        f"({len(payload['routes'])} routes, {n_total} total points)"
    )
    return RoutesViewerResult(
        html_path=out_html, n_routes=len(payload["routes"]),
        n_total_points=n_total,
    )


# -----------------------------
# Skyline viewer (building silhouette from Signal obstruction)
# -----------------------------


@dataclass(frozen=True)
class SkylineViewerResult:
    html_path: Path
    js_path: Path
    n_epochs: int
    n_observations: int
    blocked_pct: float


def build_skyline_viewer(
    *,
    stat_file: Path,
    out_html: Path,
    pos_file: Optional[Path] = None,
    az_bin_deg: float = 5.0,
    el_bin_deg: float = 5.0,
    multipath_thresh_m: float = 5.0,
    log: Optional[LogFn] = None,
) -> SkylineViewerResult:
    """Build a standalone Signal obstruction / building-silhouette viewer.

    Bins all source observations from a solver .stat file into an
    azimuth/elevation grid and infers sky visibility vs obstruction.  The
    boundary between open-sky and blocked zones approximates the building
    skyline around the unit.
    """
    log_ = make_logger(log)
    out_html = Path(out_html).resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    stat_rows = parse_stat(Path(stat_file))
    if not stat_rows:
        raise RuntimeError(f"No $SAT rows in {stat_file}")
    log_(f"[skyline] parsed {len(stat_rows)} satellite observations")

    # --- Group by epoch ---
    from collections import defaultdict
    by_epoch: dict[float, list[StatRow]] = defaultdict(list)
    for sr in stat_rows:
        by_epoch[round(sr.utc_s, 3)].append(sr)
    epoch_times = sorted(by_epoch.keys())
    t0 = epoch_times[0]

    # --- Build az/el grid ---
    n_az = int(360 / az_bin_deg)
    n_el = int(90 / el_bin_deg)
    az_centers = [i * az_bin_deg + az_bin_deg / 2 for i in range(n_az)]
    el_centers = [i * el_bin_deg + el_bin_deg / 2 for i in range(n_el)]

    # Counters per cell: how many times a source was observed there
    obs_count = [[0] * n_el for _ in range(n_az)]
    solved_count = [[0] * n_el for _ in range(n_az)]
    multipath_count = [[0] * n_el for _ in range(n_az)]
    snr_sum = [[0.0] * n_el for _ in range(n_az)]
    snr_count = [[0] * n_el for _ in range(n_az)]

    for rows in by_epoch.values():
        for sr in rows:
            ai = min(int(sr.az_deg / az_bin_deg), n_az - 1)
            ei = min(int(sr.el_deg / el_bin_deg), n_el - 1)
            obs_count[ai][ei] += 1
            if sr.valid_flag == 1:
                solved_count[ai][ei] += 1
            if abs(sr.res_p_m) > multipath_thresh_m:
                multipath_count[ai][ei] += 1
            if sr.snr_db_hz > 0:
                snr_sum[ai][ei] += sr.snr_db_hz
                snr_count[ai][ei] += 1

    # --- Compute visibility per cell ---
    # For each elevation band, compute the median observation count across
    # azimuths.  Cells with observations are scored by solved fraction;
    # empty cells surrounded by observed neighbors are marked blocked.
    # Normalize per-elevation to handle the natural drop in source
    # density near the horizon.
    el_median_obs = []
    for ei in range(n_el):
        counts_at_el = [obs_count[ai][ei] for ai in range(n_az) if obs_count[ai][ei] > 0]
        el_median_obs.append(float(np.median(counts_at_el)) if counts_at_el else 0.0)

    visibility = [[None] * n_el for _ in range(n_az)]
    for ai in range(n_az):
        for ei in range(n_el):
            if obs_count[ai][ei] == 0:
                # No observations: check neighbors (wider radius for sparse data)
                n_observed_neighbors = 0
                for dai in range(-2, 3):
                    for dei in range(-2, 3):
                        if dai == 0 and dei == 0:
                            continue
                        nai = (ai + dai) % n_az
                        nei = ei + dei
                        if 0 <= nei < n_el and obs_count[nai][nei] > 0:
                            n_observed_neighbors += 1
                if n_observed_neighbors >= 4:
                    visibility[ai][ei] = 0.0
            else:
                v = solved_count[ai][ei] / obs_count[ai][ei]
                mp_frac = multipath_count[ai][ei] / obs_count[ai][ei]
                v = v * (1 - 0.5 * mp_frac)
                visibility[ai][ei] = round(v, 3)

    # --- Extract skyline contour ---
    # For each azimuth bin, find lowest elevation where visibility > 0.3
    skyline_pts = []
    panorama_el_clear = []
    panorama_el_mp = []
    panorama_snr_mean = []

    for ai in range(n_az):
        # Find lowest el with reasonable visibility
        clear_el = 0.0
        for ei in range(n_el):
            v = visibility[ai][ei]
            if v is not None and v > 0.3:
                clear_el = el_centers[ei]
                break
        else:
            clear_el = 90.0  # fully blocked column

        # Find highest el with environment noise
        mp_el = 0.0
        for ei in range(n_el):
            if multipath_count[ai][ei] > 0:
                mp_el = max(mp_el, el_centers[ei])

        # Mean SNR for this azimuth slice
        total_snr = sum(snr_sum[ai])
        total_cnt = sum(snr_count[ai])
        mean_snr = round(total_snr / total_cnt, 1) if total_cnt > 0 else 0.0

        skyline_pts.append({"az": az_centers[ai], "el": round(clear_el, 1)})
        panorama_el_clear.append(round(clear_el, 1))
        panorama_el_mp.append(round(mp_el, 1))
        panorama_snr_mean.append(mean_snr)

    # Smooth the skyline contour with a 3-bin circular moving average
    raw_el = list(panorama_el_clear)
    for i in range(n_az):
        im1 = (i - 1) % n_az
        ip1 = (i + 1) % n_az
        panorama_el_clear[i] = round((raw_el[im1] + raw_el[i] + raw_el[ip1]) / 3, 1)
    skyline_pts = [{"az": az_centers[i], "el": panorama_el_clear[i]} for i in range(n_az)]

    # Close the skyline loop
    skyline_pts.append(skyline_pts[0])

    # --- SNR heatmap (el rows × az cols) ---
    snr_heatmap = []
    for ei in range(n_el):
        row = []
        for ai in range(n_az):
            if snr_count[ai][ei] > 0:
                row.append(round(snr_sum[ai][ei] / snr_count[ai][ei], 1))
            else:
                row.append(None)
        snr_heatmap.append(row)

    # --- Timeline data ---
    tl_t = []
    tl_solved = []
    tl_tracked = []
    tl_mp = []
    tl_blocked = []

    for et in epoch_times:
        rows = by_epoch[et]
        n_solved = sum(1 for r in rows if r.valid_flag == 1)
        n_tracked_only = sum(1 for r in rows if r.valid_flag == 0)
        n_mp = sum(1 for r in rows if abs(r.res_p_m) > multipath_thresh_m)
        tl_t.append(round(et - t0, 2))
        tl_solved.append(n_solved)
        tl_tracked.append(n_tracked_only)
        tl_mp.append(n_mp)

        # Estimate blocked fraction: compare current visible count to session max
        total_this = len(rows)
        tl_blocked.append(total_this)

    # Normalize blocked to fraction of max
    max_visible = max(tl_blocked) if tl_blocked else 1
    tl_blocked_frac = [round(1.0 - v / max_visible, 3) for v in tl_blocked]

    # --- Stats ---
    total_obs = len(stat_rows)
    mean_solved = round(np.mean(tl_solved), 1) if tl_solved else 0
    mean_tracked = round(np.mean(tl_tracked), 1) if tl_tracked else 0
    total_mp = sum(1 for r in stat_rows if abs(r.res_p_m) > multipath_thresh_m)
    mp_pct = round(100 * total_mp / total_obs, 1) if total_obs else 0

    # Blocked cells fraction
    n_blocked = sum(
        1 for ai in range(n_az) for ei in range(n_el)
        if visibility[ai][ei] is not None and visibility[ai][ei] < 0.1
    )
    n_total_cells = sum(
        1 for ai in range(n_az) for ei in range(n_el)
        if visibility[ai][ei] is not None
    )
    blocked_pct = round(100 * n_blocked / n_total_cells, 1) if n_total_cells else 0

    # Dominant obstruction direction
    worst_az_idx = max(range(n_az), key=lambda i: panorama_el_clear[i])
    compass = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    compass_idx = int(round(az_centers[worst_az_idx] / 22.5)) % 16
    dominant_dir = f"{compass[compass_idx]} ({az_centers[worst_az_idx]:.0f}°)"

    # --- Assemble payload ---
    payload = {
        "stats": {
            "total_epochs": len(epoch_times),
            "total_sats": total_obs,
            "mean_visible": mean_solved,
            "mean_tracked": mean_tracked,
            "blocked_pct": blocked_pct,
            "multipath_pct": mp_pct,
            "dominant_obstruction": dominant_dir,
        },
        "polar_grid": {
            "az_bins": az_centers,
            "el_bins": el_centers,
            "visibility": visibility,
        },
        "skyline": skyline_pts,
        "panorama": {
            "az": az_centers,
            "el_clear": panorama_el_clear,
            "el_multipath": panorama_el_mp,
            "snr_mean": panorama_snr_mean,
        },
        "snr_heatmap": {
            "az_bins": az_centers,
            "el_bins": el_centers,
            "snr": snr_heatmap,
        },
        "timeline": {
            "t_s": tl_t,
            "n_solved": tl_solved,
            "n_tracked": tl_tracked,
            "n_multipath": tl_mp,
            "blocked_frac": tl_blocked_frac,
        },
    }

    template = (_ASSETS_DIR / "skyline_viewer.html").read_text(encoding="utf-8")
    html = template.replace("__SKYLINE_DATA__", json.dumps(payload))
    out_html.write_text(html, encoding="utf-8")
    js = _copy_plotly_next_to(out_html.parent)
    log_(f"[skyline] wrote {out_html} ({len(epoch_times)} epochs, "
         f"{total_obs} observations, {blocked_pct}% sky blocked)")
    return SkylineViewerResult(
        html_path=out_html, js_path=js,
        n_epochs=len(epoch_times), n_observations=total_obs,
        blocked_pct=blocked_pct,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)

    pt = sp.add_parser("trajectory", help="Build trajectory_viewer.html")
    pt.add_argument("--data-log", required=True, type=Path)
    pt.add_argument("--georef-csv", required=True, type=Path)
    pt.add_argument(
        "--recording-map",
        type=Path,
        default=None,
        help="Optional recording_*.txt — time-sync stats in page header when set.",
    )
    pt.add_argument("--out", required=True, type=Path)

    po = sp.add_parser("orientation", help="Build orientation_panel.html")
    po.add_argument("--data-log", required=True, type=Path)
    po.add_argument("--pos", required=True, type=Path)
    po.add_argument("--out", required=True, type=Path)
    po.add_argument("--sensors-txt", type=Path, default=None,
                    help="sensors_*.txt fallback when data log has no OrientationDeg lines.")
    po.add_argument("--smooth-sigma-s", type=float, default=3.0)
    po.add_argument("--decimate-hz", type=float, default=10.0)
    po.add_argument("--min-speed-mps", type=float, default=2.0)

    pc = sp.add_parser("compare", help="Build comparison_viewer.html (multi-profile)")
    pc.add_argument("--data-log", required=True, type=Path)
    pc.add_argument("--pos", required=True, type=Path)
    pc.add_argument("--frame-times-csv", required=True, type=Path)
    pc.add_argument("--recording-map", required=True, type=Path)
    pc.add_argument(
        "--fps", type=float, default=None,
        help="Frame rate in Hz (optional; derived from frame timings if not supplied)"
    )
    pc.add_argument("--out", required=True, type=Path)

    pv2 = sp.add_parser("velocity", help="Build velocity_viewer.html (Doppler + coords speed)")
    pv2.add_argument("--pos", required=True, type=Path)
    pv2.add_argument("--out", required=True, type=Path)

    pg = sp.add_parser("geo", help="Build geo_viewer.html (2D satellite + 3D terrain)")
    pg.add_argument("--pos", required=True, type=Path)
    pg.add_argument("--out", required=True, type=Path)
    pg.add_argument("--georef-csv", type=Path, default=None)
    pg.add_argument("--basemap-tiff", type=Path, default=None)
    pg.add_argument("--dsm-tiff", type=Path, default=None)
    pg.add_argument("--basemap-max-dim", type=int, default=1024)

    ps = sp.add_parser("sync", help="Build sync_player.html (video + trajectory)")
    ps.add_argument("--video", required=True, type=Path)
    ps.add_argument("--pos", required=True, type=Path)
    ps.add_argument("--frame-times-csv", required=True, type=Path)
    ps.add_argument("--recording-map", required=True, type=Path)
    ps.add_argument(
        "--basemap-tiff",
        type=Path,
        default=None,
        help="Optional GeoTIFF (warped to WGS84 PNG next to sync_player.html; pip install rasterio).",
    )
    ps.add_argument(
        "--basemap-max-dim",
        type=int,
        default=2048,
        help="Pixel size of exported basemap PNG (longest side).",
    )
    ps.add_argument(
        "--rotation",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Clockwise rotation applied to video display (matches frame extraction rotation).",
    )
    ps.add_argument(
        "--stat",
        type=Path,
        default=None,
        help="RTKLIB .stat file for satellite skyplot panel (az/el per epoch).",
    )
    ps.add_argument(
        "--bias-ms", type=float, default=0.0,
        help="Video-GNSS time bias in milliseconds (+ = video ahead of GNSS). "
             "Adjustable live in the player UI.",
    )
    ps.add_argument(
        "--wav", type=Path, default=None,
        help="audio_*.wav (48 kHz PCM) for synced playback + spectrogram.",
    )
    ps.add_argument(
        "--audio-anchor", type=Path, default=None,
        help="audio_anchor_*.txt (audio frame -> BOOTTIME) for spectrogram UTC "
             "alignment + cross-clock sync stats.",
    )
    ps.add_argument(
        "--capture-meta", type=Path, default=None,
        help="capture_meta.json (provides video_t0_boottime_ns for the "
             "audio<->video offset).",
    )
    ps.add_argument(
        "--video-anchor", type=Path, default=None,
        help="recording_*.video_anchor.txt (per-frame boot/pts; fallback source "
             "of video_t0_boottime_ns). NOTE: --capture-meta takes precedence "
             "over this — for a trimmed clip pass its anchor as "
             "--chop-video-anchor instead, which always wins.",
    )
    ps.add_argument(
        "--chop-video-anchor", type=Path, default=None,
        help="trimmed ('chop') clip's own *.video_anchor.txt — REQUIRED when "
             "--video is a chop: its min bootNs overrides the parent "
             "capture_meta video_t0_boottime_ns (chop PTS are rebased to 0). "
             "Do NOT pass the chop anchor as --video-anchor; it would lose "
             "to --capture-meta and every frame would map minutes early.",
    )
    ps.add_argument(
        "--mux-audio", action="store_true",
        help="Mux the WAV into a NEW recording_*_av.mp4 (original .mp4 "
             "untouched); falls back to a side-car <audio> if ffmpeg is missing.",
    )
    ps.add_argument(
        "--no-spectrogram", action="store_true",
        help="Disable the audio spectrogram panel.",
    )
    ps.add_argument("--out", required=True, type=Path)

    pin = sp.add_parser(
        "ins",
        help="Build ins_viewer.html (raw PPK vs FLP vs forward EKF vs RTS).",
    )
    pin.add_argument("--sensors-txt", required=True, type=Path)
    pin.add_argument("--pos",         required=True, type=Path)
    pin.add_argument("--data-log",   type=Path, default=None,
                     help="Optional measurements_*.txt for FLP overlay.")
    pin.add_argument("--vehicle-mode", action="store_true",
                     help="Enable non-holonomic constraint (cars only).")
    pin.add_argument("--chi2-gate-pos", type=float, default=0.0)
    pin.add_argument("--chi2-gate-vel", type=float, default=0.0)
    pin.add_argument("--out", required=True, type=Path)

    ptm = sp.add_parser(
        "time-math",
        help="Build time_math.html (annotated walkthrough of the OLS fit).",
    )
    ptm.add_argument("--recording-map", required=True, type=Path)
    ptm.add_argument("--envelope-samples", type=int, default=400)
    ptm.add_argument("--out", required=True, type=Path)

    pvt = sp.add_parser(
        "video-time",
        help="Build video_time.html (video PTS vs GNSS UTC regression).",
    )
    pvt.add_argument(
        "--recording-map", required=True, type=Path,
        help="the capture app recording_*.txt (video_ns,UTC anchor pairs).",
    )
    pvt.add_argument(
        "--envelope-samples", type=int, default=400,
        help="Number of points along the σ-envelope curve.",
    )
    pvt.add_argument("--out", required=True, type=Path)

    pcb = sp.add_parser(
        "clock-bias",
        help="Build clock_bias.html (device clock vs GPS time from Raw rows).",
    )
    pcb.add_argument(
        "--measurements-txt", required=True, type=Path,
        help="the capture app measurements_*.txt (must contain Raw lines).",
    )
    pcb.add_argument(
        "--leap-seconds", type=float, default=None,
        help="Override leap seconds (default: auto from time_sync table).",
    )
    pcb.add_argument("--out", required=True, type=Path)

    ptr = sp.add_parser(
        "trust",
        help="Build trust_viewer.html from the fused-bent trust sidecar.",
    )
    ptr.add_argument(
        "--trust-csv", required=True, type=Path,
        help="Path to <georef_csv_stem>_trust.csv (written by fused-bent run).",
    )
    ptr.add_argument("--out", required=True, type=Path)

    psk = sp.add_parser(
        "skyline",
        help="Build skyline_viewer.html (building silhouette from GNSS obstruction).",
    )
    psk.add_argument(
        "--stat", required=True, type=Path,
        help="RTKLIB .stat file (rnx2rtkp -x 1 output).",
    )
    psk.add_argument("--pos", type=Path, default=None,
                     help="Optional .pos file (for position context).")
    psk.add_argument("--az-bin", type=float, default=5.0,
                     help="Azimuth bin width in degrees (default: 5).")
    psk.add_argument("--el-bin", type=float, default=5.0,
                     help="Elevation bin width in degrees (default: 5).")
    psk.add_argument("--mp-thresh", type=float, default=5.0,
                     help="Pseudorange residual threshold for multipath (m, default: 5).")
    psk.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "trajectory":
        build_trajectory_viewer(
            data_log=args.data_log,
            georef_csv=args.georef_csv,
            out_html=args.out,
            recording_map=args.recording_map,
        )
    elif args.cmd == "orientation":
        build_orientation_panel(
            data_log=args.data_log,
            pos_file=args.pos,
            out_html=args.out,
            sensors_txt=args.sensors_txt,
            smooth_sigma_seconds=args.smooth_sigma_s,
            decimate_hz=args.decimate_hz,
            min_speed_mps=args.min_speed_mps,
        )
    elif args.cmd == "compare":
        build_comparison_viewer(
            data_log=args.data_log,
            pos_file=args.pos,
            frame_times_csv=args.frame_times_csv,
            recording_map=args.recording_map,
            out_html=args.out,
            fps=args.fps,
        )
    elif args.cmd == "velocity":
        build_velocity_viewer(pos_file=args.pos, out_html=args.out)
    elif args.cmd == "geo":
        build_geo_viewer(
            pos_file=args.pos,
            out_html=args.out,
            georef_csv=args.georef_csv,
            basemap_tiff=args.basemap_tiff,
            dsm_tiff=args.dsm_tiff,
            basemap_max_dim=args.basemap_max_dim,
        )
    elif args.cmd == "ins":
        build_ins_viewer(
            sensors_txt=args.sensors_txt,
            pos_file=args.pos,
            out_html=args.out,
            data_log=args.data_log,
            vehicle_mode=args.vehicle_mode,
            chi2_gate_pos=args.chi2_gate_pos,
            chi2_gate_vel=args.chi2_gate_vel,
        )
    elif args.cmd == "time-math":
        build_time_math_viewer(
            recording_map=args.recording_map,
            out_html=args.out,
            envelope_samples=args.envelope_samples,
        )
    elif args.cmd == "video-time":
        build_video_time_viewer(
            recording_map=args.recording_map,
            out_html=args.out,
            envelope_samples=args.envelope_samples,
        )
    elif args.cmd == "clock-bias":
        build_clock_bias_viewer(
            measurements_txt=args.measurements_txt,
            out_html=args.out,
            leap_seconds=args.leap_seconds,
        )
    elif args.cmd == "trust":
        build_trust_viewer(
            trust_csv=args.trust_csv,
            out_html=args.out,
        )
    elif args.cmd == "skyline":
        build_skyline_viewer(
            stat_file=args.stat,
            out_html=args.out,
            pos_file=args.pos,
            az_bin_deg=args.az_bin,
            el_bin_deg=args.el_bin,
            multipath_thresh_m=args.mp_thresh,
        )
    else:
        build_sync_player(
            video=args.video,
            pos_file=args.pos,
            frame_times_csv=args.frame_times_csv,
            recording_map=args.recording_map,
            out_html=args.out,
            rotation=args.rotation,
            basemap_geotiff=args.basemap_tiff,
            basemap_max_dim=max(128, args.basemap_max_dim),
            stat_file=args.stat,
            video_bias_ms=args.bias_ms,
            wav=args.wav,
            audio_anchor=args.audio_anchor,
            capture_meta=args.capture_meta,
            video_anchor=args.video_anchor,
            chop_video_anchor=args.chop_video_anchor,
            mux_audio=args.mux_audio,
            show_spectrogram=not args.no_spectrogram,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
