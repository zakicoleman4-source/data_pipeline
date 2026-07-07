"""Standalone the source app time-anchor corrector.

Given a the source app session folder (recording_*.txt + recording_*.container file) and a
list of relative-time floats (e.g. stream event picks in a recording_*.wav),
produce:

1. A CSV with for each pick:
   - relative_time_s    : input pick time (seconds since session start).
   - naive_utc_iso      : video_start_utc + relative_time_s.
                          What you'd guess if you only trusted the very first
                          anchor in recording_*.txt. NO drift correction.
   - anchored_utc_iso   : OLS-corrected UTC using all anchors in
                          recording_*.txt. The recommended value.
   - correction_s       : anchored - naive. Shows what the regression fixed.
   - anchor_uncertainty_s : 1-sigma uncertainty at this query time.
   - plus matching Reference time columns (UTC + epoch offset) for The external solver/.pos compare.

2. An HTML drift report with multiple plots: residual vs time (write-latency
   proxy), residual histogram, cumulative drift (naive vs anchored) with the
   user's pickings overlaid, rolling RMSE, residual Q-Q vs normal.

Standalone: only depends on Python 3.10+ and numpy. No data_pipeline import.

Usage:
    python correct_relative_times.py <session_dir> --times <floats_file> \
        [--out <csv>] [--html <html>] [--offset-s 0.0] [--no-html]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------
# Epoch offset table (Reference-UTC offset by epoch).
# Update when IERS announces a new epoch offset.
# -----------------------------

_LEAP_SECOND_TABLE: list[tuple[float, float]] = [
    (315964800.0,  0),   # 1980-01-06  Reference epoch
    (362793600.0,  1),   # 1981-07-01
    (394329600.0,  2),   # 1982-07-01
    (425865600.0,  3),   # 1983-07-01
    (489024000.0,  4),   # 1985-07-01
    (567993600.0,  5),   # 1988-01-01
    (631152000.0,  6),   # 1990-01-01
    (662688000.0,  7),   # 1991-01-01
    (709948800.0,  8),   # 1992-07-01
    (741484800.0,  9),   # 1993-07-01
    (773020800.0, 10),   # 1994-07-01
    (820454400.0, 11),   # 1996-01-01
    (867715200.0, 12),   # 1997-07-01
    (915148800.0, 13),   # 1999-01-01
    (1136073600.0, 14),  # 2006-01-01
    (1230768000.0, 15),  # 2009-01-01
    (1341100800.0, 16),  # 2012-07-01
    (1435708800.0, 17),  # 2015-07-01
    (1483228800.0, 18),  # 2017-01-01
]


def get_leap_seconds_for_epoch(utc_posix_s: float) -> float:
    for ts, leap in reversed(_LEAP_SECOND_TABLE):
        if utc_posix_s >= ts:
            return leap
    return _LEAP_SECOND_TABLE[0][1]


# -----------------------------
# ISO parsing / formatting
# -----------------------------


def _parse_iso_utc(iso: str) -> dt.datetime:
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        base, frac = s.split(".", 1)
        s = f"{base}.{(frac + '000000')[:6]}"
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def _parse_utc_seconds(iso: str) -> float:
    return _parse_iso_utc(iso).timestamp()


def _format_utc_iso(utc_s: float) -> str:
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _format_gpst_iso(gpst_s: float) -> str:
    return dt.datetime.fromtimestamp(gpst_s, tz=dt.timezone.utc).replace(
        tzinfo=None
    ).isoformat(timespec="microseconds") + " GPST"


# -----------------------------
# TimeAnchor (OLS regression)
# -----------------------------


@dataclass(frozen=True)
class TimeAnchor:
    """Best-fit affine map utc_s = ymean + slope * (video_ns - xmean)."""

    slope: float
    xmean: float
    ymean: float
    n: int
    rmse_s: float
    max_abs_s: float
    n_rejected: int
    sxx_ns2: float
    cubic_rmse_improvement_s: float

    @property
    def drift_ppm(self) -> float:
        return (self.slope * 1e9 - 1.0) * 1e6

    @property
    def sigma_residual_s(self) -> float:
        if self.n <= 2 or self.rmse_s <= 0:
            return self.rmse_s
        return self.rmse_s * (self.n / (self.n - 2)) ** 0.5

    @property
    def fit_uncertainty_s(self) -> float:
        if self.n < 3:
            return float("inf")
        return self.sigma_residual_s / (self.n ** 0.5)

    def fit_uncertainty_s_at(self, video_ns: float) -> float:
        if self.n < 3 or self.sxx_ns2 <= 0:
            return float("inf")
        sigma = self.sigma_residual_s
        dx = video_ns - self.xmean
        return sigma * (1.0 / self.n + (dx * dx) / self.sxx_ns2) ** 0.5

    def video_ns_to_utc_s(self, video_ns: float) -> float:
        return self.ymean + self.slope * (video_ns - self.xmean)


def _ols_about_means(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float, float, float, np.ndarray]:
    xmean = float(xs.mean())
    ymean = float(ys.mean())
    xc = xs - xmean
    yc = ys - ymean
    sxx = float((xc * xc).sum())
    if sxx <= 0:
        raise ValueError("All x values identical; cannot fit.")
    slope = float((xc * yc).sum() / sxx)
    residuals = ys - (ymean + slope * xc)
    return slope, xmean, ymean, sxx, residuals


def _cubic_rmse(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 5:
        return float("nan")
    xc = xs - xs.mean()
    try:
        coef = np.polyfit(xc, ys, deg=3)
        yhat = np.polyval(coef, xc)
        return float(np.sqrt(np.mean((ys - yhat) ** 2)))
    except Exception:
        return float("nan")


def fit_time_anchor(
    pairs: List[Tuple[float, float]],
    *,
    robust: bool = True,
    mad_threshold: float = 5.0,
    max_iter: int = 3,
) -> TimeAnchor:
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    if len(xs) < 2:
        raise ValueError(f"Need >= 2 anchors, got {len(xs)}")
    n0 = len(xs)

    slope, xmean, ymean, sxx, residuals = _ols_about_means(xs, ys)
    keep_x, keep_y = xs, ys

    if robust:
        for _ in range(max_iter):
            mad = float(np.median(np.abs(residuals)))
            if mad <= 0:
                break
            cutoff = mad_threshold * mad
            mask = np.abs(residuals) <= cutoff
            if mask.sum() == len(keep_x):
                break
            if mask.sum() < 2:
                break
            keep_x = keep_x[mask]
            keep_y = keep_y[mask]
            slope, xmean, ymean, sxx, residuals = _ols_about_means(keep_x, keep_y)

    n_rej = n0 - len(keep_x)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    max_abs = float(np.max(np.abs(residuals))) if len(residuals) else 0.0
    cubic = _cubic_rmse(keep_x, keep_y)
    cubic_improve = max(0.0, rmse - cubic) if not math.isnan(cubic) else 0.0

    return TimeAnchor(
        slope=slope, xmean=xmean, ymean=ymean,
        n=len(keep_x), rmse_s=rmse, max_abs_s=max_abs,
        n_rejected=n_rej, sxx_ns2=sxx,
        cubic_rmse_improvement_s=cubic_improve,
    )


def _read_anchor_pairs(path: Path) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                video_ns = int(parts[0])
                utc_s = _parse_utc_seconds(parts[1])
            except (ValueError, IndexError):
                continue
            out.append((float(video_ns), utc_s))
    if len(out) < 2:
        raise ValueError(f"Need >= 2 anchors in {path}, got {len(out)}")
    return out


def _per_anchor_residuals(
    pairs: List[Tuple[float, float]], anchor: TimeAnchor
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for vn, us in pairs:
        r = us - anchor.video_ns_to_utc_s(vn)
        out.append((vn / 1e9, r))
    return out


# -----------------------------
# Session-folder discovery (RawInputs-equivalent, minimal).
# -----------------------------


def _pick_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(
            f"Multiple files match {pattern!r} in {folder}: {names}. "
            "Keep only one session per folder."
        )
    return matches[0]


# -----------------------------
# Relative-times reader
# -----------------------------

_HEADER_KEYS = (
    "relative_time_s", "t_s", "t_rel_s", "t_video_s",
    "time_s", "seconds", "pick_s",
)


def _looks_like_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_relative_times(path: Path) -> List[float]:
    rows: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_comma = "," in sample
        if has_comma:
            reader = csv.reader(f)
            col_idx = 0
            for i, row in enumerate(reader):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                if i == 0 and not _looks_like_float(first):
                    header = [c.strip().lower() for c in row]
                    for key in _HEADER_KEYS:
                        if key in header:
                            col_idx = header.index(key)
                            break
                    continue
                try:
                    rows.append(float(row[col_idx].strip()))
                except (ValueError, IndexError):
                    continue
        else:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    rows.append(float(s))
                except ValueError:
                    continue
    if not rows:
        raise ValueError(f"No numeric values parsed from {path}")
    return rows


# -----------------------------
# CSV writer
# -----------------------------


@dataclass(frozen=True)
class _Row:
    rel_s: float
    video_s: float
    naive_utc_s: float
    anch_utc_s: float
    naive_gpst_s: float
    anch_gpst_s: float
    leap_s: float
    correction_s: float
    unc_s: float


def _build_rows(
    relative_times_s: List[float],
    anchor: TimeAnchor,
    *,
    offset_s: float,
    video_start_utc_s: float,
) -> List[_Row]:
    out: List[_Row] = []
    for t_rel in relative_times_s:
        t_video = float(t_rel) + offset_s
        video_ns = t_video * 1e9
        naive_utc = video_start_utc_s + t_video
        anch_utc = anchor.video_ns_to_utc_s(video_ns)
        leap = get_leap_seconds_for_epoch(anch_utc)
        out.append(_Row(
            rel_s=float(t_rel),
            video_s=t_video,
            naive_utc_s=naive_utc,
            anch_utc_s=anch_utc,
            naive_gpst_s=naive_utc + get_leap_seconds_for_epoch(naive_utc),
            anch_gpst_s=anch_utc + leap,
            leap_s=leap,
            correction_s=anch_utc - naive_utc,
            unc_s=anchor.fit_uncertainty_s_at(video_ns),
        ))
    return out


def write_corrected_csv(
    out_csv: Path,
    rows: List[_Row],
    *,
    video_start_utc_s: float,
    mp4_basename: str,
    session: str,
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    video_start_iso = _format_utc_iso(video_start_utc_s)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "relative_time_s",
            "video_pts_s",
            "video_start_utc_iso",
            "naive_utc_iso",
            "naive_utc_posix_s",
            "anchored_utc_iso",
            "anchored_utc_posix_s",
            "correction_s",
            "anchor_uncertainty_s",
            "naive_gpst_iso",
            "naive_gpst_posix_s",
            "anchored_gpst_iso",
            "anchored_gpst_posix_s",
            "leap_seconds",
            "mp4_basename",
            "session",
        ])
        for r in rows:
            w.writerow([
                f"{r.rel_s:.9f}",
                f"{r.video_s:.9f}",
                video_start_iso,
                _format_utc_iso(r.naive_utc_s),
                f"{r.naive_utc_s:.9f}",
                _format_utc_iso(r.anch_utc_s),
                f"{r.anch_utc_s:.9f}",
                f"{r.correction_s:.9f}",
                f"{r.unc_s:.9f}",
                _format_gpst_iso(r.naive_gpst_s),
                f"{r.naive_gpst_s:.9f}",
                _format_gpst_iso(r.anch_gpst_s),
                f"{r.anch_gpst_s:.9f}",
                f"{r.leap_s:.0f}",
                mp4_basename,
                session,
            ])
    return len(rows)


# -----------------------------
# HTML drift / statistics report
# -----------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>TimeAnchor drift report - __SESSION__</title>
<script src=\"plotly.min.js\"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; color: #222; }
  h1 { font-size: 18px; margin: 0 0 6px; }
  h3 { font-size: 14px; margin: 4px 0 8px; }
  .meta { font-size: 13px; margin-bottom: 12px; color: #555; }
  .row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
  .card { flex: 1 1 540px; border: 1px solid #ddd; border-radius: 6px;
          padding: 8px; background: #fafafa; }
  table { border-collapse: collapse; font-size: 13px; }
  td, th { padding: 3px 10px; border-bottom: 1px solid #eee; text-align: left; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .note { font-size: 12px; color: #666; margin: 8px 4px 0; }
</style>
</head>
<body>
<h1>TimeAnchor drift / write-latency report</h1>
<div class=\"meta\">Session: <b>__SESSION__</b> &middot; recording: <code>__RECORDING__</code></div>

<div class=\"card\">
<h3>Fit summary</h3>
<table>
  <tr><th>Anchors used (n)</th><td class=\"num\">__N__</td>
      <th>Anchors rejected</th><td class=\"num\">__N_REJ__</td></tr>
  <tr><th>RMSE (jitter)</th><td class=\"num\">__RMSE_MS__ ms</td>
      <th>Max abs residual</th><td class=\"num\">__MAX_MS__ ms</td></tr>
  <tr><th>Drift</th><td class=\"num\">__DRIFT_PPM__ ppm</td>
      <th>Sigma_hat (unbiased)</th><td class=\"num\">__SIGMA_MS__ ms</td></tr>
  <tr><th>Fit uncertainty (centroid)</th><td class=\"num\">__FIT_UNC_MS__ ms</td>
      <th>Cubic improvement vs linear</th><td class=\"num\">__CUBIC_MS__ ms</td></tr>
  <tr><th>Recording span</th><td class=\"num\">__SPAN_S__ s</td>
      <th>Accumulated drift over span</th><td class=\"num\">__ACCUM_MS__ ms</td></tr>
  <tr><th>Residual mean (bias)</th><td class=\"num\">__BIAS_MS__ ms</td>
      <th>Residual P95 abs</th><td class=\"num\">__P95_MS__ ms</td></tr>
</table>
<p class=\"note\">
RMSE = sqrt(mean(residual^2)). Captures Android writer scheduling jitter
(how long device takes to flush each anchor to storage).
Drift_ppm = (slope - 1) * 1e6: camera-vs-system clock skew the linear
fit absorbed. Accumulated drift = drift_ppm * span * 1e-6 s. Cubic
improvement: if comparable to RMSE the camera-vs-system relationship is
non-linear and the linear model is biased; otherwise linear is fine.
Sigma_hat = unbiased sigma with dof = n - 2.
</p>
</div>

<div class=\"row\">
  <div class=\"card\"><div id=\"residPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Per-anchor residual = utc_logged - utc_predicted.
       Orange diamonds = your pickings (interpolated). Watch for
       structure: a slope means residual drift the linear fit missed; a
       step means a clock reset.</p></div>
  <div class=\"card\"><div id=\"histPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Residual distribution. If roughly Gaussian, the
       OLS RMSE is a meaningful 1-sigma write-latency proxy. Long tails
       to the right = sporadic flush delays.</p></div>
</div>

<div class=\"row\">
  <div class=\"card\"><div id=\"correctionPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Cumulative correction = anchored_utc - naive_utc
       at each video-time query. The straight-line slope IS the drift_ppm
       (multiplied by 1e-6). Orange diamonds show your pickings — read
       directly how many ms the OLS shifted each one vs the naive
       first-anchor mapping.</p></div>
  <div class=\"card\"><div id=\"rollingPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Rolling RMSE in a __ROLL_WIN__ s window. Flat =
       stationary write-jitter. Bumps = device was under load (e.g.
       background sync, thermal throttle) and writer scheduling got
       worse.</p></div>
</div>

<div class=\"row\">
  <div class=\"card\"><div id=\"qqPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Q-Q plot vs standard normal. Straight line =
       Gaussian residuals. Tails curling up/down = heavy-tailed jitter
       (Android scheduler outliers).</p></div>
  <div class=\"card\"><div id=\"absSortPlot\" style=\"height:380px;\"></div>
       <p class=\"note\">Sorted |residual| - cumulative tail. Read
       directly: \"95% of anchors are within X ms of the fit.\"</p></div>
</div>

<script>
const DATA = __DATA__;

const residTrace = {
  x: DATA.video_s, y: DATA.resid_ms, mode: 'markers',
  type: 'scattergl', marker: { size: 4, color: '#1f77b4', opacity: 0.55 },
  name: 'anchor residual'
};
const pickResidTrace = {
  x: DATA.pick_video_s, y: DATA.pick_resid_ms, mode: 'markers',
  type: 'scatter',
  marker: { size: 11, color: '#ff7f0e', symbol: 'diamond',
            line: { color: '#cc4400', width: 1 } },
  name: 'pickings'
};
Plotly.newPlot('residPlot', [residTrace, pickResidTrace], {
  title: 'Per-anchor residual vs video time (write-latency proxy)',
  xaxis: { title: 'video time [s]' },
  yaxis: { title: 'utc_logged - utc_fit [ms]' },
  margin: { l:55, r:15, t:40, b:45 }
});

Plotly.newPlot('histPlot', [{
  x: DATA.resid_ms, type: 'histogram', nbinsx: 60,
  marker: { color: '#ff7f0e' }, name: 'residuals'
}], {
  title: 'Residual distribution',
  xaxis: { title: 'utc_logged - utc_fit [ms]' },
  yaxis: { title: 'count' },
  margin: { l:55, r:15, t:40, b:45 }
});

// Cumulative correction = anchored - naive over time at the *queries* + a
// dense reference line over the full span.
Plotly.newPlot('correctionPlot', [
  { x: DATA.dense_video_s, y: DATA.dense_correction_ms,
    mode: 'lines', line: { color: '#2ca02c', width: 2 },
    name: 'anchored - naive' },
  { x: DATA.pick_video_s, y: DATA.pick_correction_ms,
    mode: 'markers', type: 'scatter',
    marker: { size: 11, color: '#ff7f0e', symbol: 'diamond',
              line: { color: '#cc4400', width: 1 } },
    name: 'pickings' }
], {
  title: 'Cumulative correction (anchored - naive)',
  xaxis: { title: 'video time [s]' },
  yaxis: { title: 'correction [ms]' },
  margin: { l:55, r:15, t:40, b:45 }
});

Plotly.newPlot('rollingPlot', [{
  x: DATA.roll_video_s, y: DATA.roll_rmse_ms,
  mode: 'lines', line: { color: '#9467bd', width: 2 },
  name: 'rolling RMSE'
}], {
  title: 'Rolling RMSE (' + DATA.roll_win_s.toFixed(0) + ' s window)',
  xaxis: { title: 'video time [s]' },
  yaxis: { title: 'rolling RMSE [ms]' },
  margin: { l:55, r:15, t:40, b:45 }
});

Plotly.newPlot('qqPlot', [
  { x: DATA.qq_theory, y: DATA.qq_sample, mode: 'markers',
    type: 'scattergl',
    marker: { size: 4, color: '#1f77b4', opacity: 0.6 }, name: 'sample' },
  { x: DATA.qq_ref_x, y: DATA.qq_ref_y, mode: 'lines',
    line: { color: '#d62728', dash: 'dash' }, name: 'ideal Gaussian' }
], {
  title: 'Residual Q-Q vs standard normal',
  xaxis: { title: 'theoretical quantile' },
  yaxis: { title: 'residual quantile [ms]' },
  margin: { l:55, r:15, t:40, b:45 }
});

Plotly.newPlot('absSortPlot', [{
  x: DATA.abs_pct, y: DATA.abs_sorted_ms,
  mode: 'lines', line: { color: '#17becf', width: 2 }, name: '|residual|'
}], {
  title: 'Sorted |residual| vs percentile',
  xaxis: { title: 'percentile of anchors [%]' },
  yaxis: { title: '|residual| [ms]' },
  margin: { l:55, r:15, t:40, b:45 }
});
</script>
</body>
</html>
"""


def _rolling_rmse(
    video_s: np.ndarray, resid_s: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window RMSE centred on each anchor; O(n log n) via bisect."""
    if len(video_s) == 0:
        return video_s, resid_s
    order = np.argsort(video_s)
    vs = video_s[order]
    rs = resid_s[order]
    half = window_s / 2.0
    left = np.searchsorted(vs, vs - half, side="left")
    right = np.searchsorted(vs, vs + half, side="right")
    out = np.empty_like(vs)
    rs2 = rs * rs
    cumsum = np.concatenate(([0.0], np.cumsum(rs2)))
    cumcount = np.arange(len(vs) + 1, dtype=np.float64)
    for i in range(len(vs)):
        n = cumcount[right[i]] - cumcount[left[i]]
        if n <= 0:
            out[i] = 0.0
        else:
            s2 = cumsum[right[i]] - cumsum[left[i]]
            out[i] = math.sqrt(s2 / n)
    return vs, out


def _qq_theoretical_quantiles(n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    p = (np.arange(1, n + 1) - 0.5) / n
    # inverse normal cdf via erfinv (numpy>=1.21 has math.erfinv per scalar).
    return np.array([math.sqrt(2) * _erfinv(2 * pi - 1) for pi in p])


def _erfinv(x: float) -> float:
    # Winitzki approximation, accurate to ~1e-4 for |x|<1.
    a = 0.147
    ln = math.log(1 - x * x)
    sgn = 1.0 if x >= 0 else -1.0
    first = 2.0 / (math.pi * a) + ln / 2.0
    inside = first * first - ln / a
    return sgn * math.sqrt(math.sqrt(inside) - first)


def write_drift_html(
    out_html: Path,
    anchor: TimeAnchor,
    anchor_pairs: List[Tuple[float, float]],
    residuals: List[Tuple[float, float]],
    rows: List[_Row],
    *,
    session: str,
    recording_path: Path,
    plotly_src: Optional[Path],
    video_start_utc_s: float,
    roll_window_s: float = 30.0,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    video_s = np.array([r[0] for r in residuals], dtype=np.float64)
    resid_s = np.array([r[1] for r in residuals], dtype=np.float64)
    resid_ms = resid_s * 1000.0
    span = float(video_s.max() - video_s.min()) if len(video_s) else 0.0
    bias_ms = float(resid_ms.mean()) if len(resid_ms) else 0.0
    p95_ms = float(np.percentile(np.abs(resid_ms), 95)) if len(resid_ms) else 0.0
    accum_ms = anchor.drift_ppm * span * 1e-3  # ppm * s * 1e-6 -> s; *1000 -> ms

    # Pickings overlaid on residual plot and correction plot.
    pick_video_s = [r.video_s for r in rows]
    pick_resid_ms = [
        (r.anch_utc_s - r.naive_utc_s) * 1000.0 * 0.0  # placeholder, see below
        for r in rows
    ]
    # Residual for a picking is "what would the anchor at this media-time
    # have logged minus the fit" - but we don't have a logged value at the
    # pick. Instead show "0" plus a horizontal bracket of fit uncertainty.
    # Practical: plot the per-pick fit uncertainty as the y of the marker.
    pick_resid_ms = [r.unc_s * 1000.0 for r in rows]
    pick_correction_ms = [r.correction_s * 1000.0 for r in rows]

    # Dense correction curve: anchored - naive over full span.
    dense_t = np.linspace(0.0, max(span, 1.0), 256)
    naive_curve = video_start_utc_s + dense_t
    anch_curve = np.array([
        anchor.video_ns_to_utc_s(t * 1e9) for t in dense_t
    ])
    dense_correction_ms = (anch_curve - naive_curve) * 1000.0

    # Rolling RMSE.
    roll_x, roll_rmse_s = _rolling_rmse(video_s, resid_s, roll_window_s)
    roll_rmse_ms = roll_rmse_s * 1000.0

    # Q-Q: sample quantiles vs theoretical normal quantiles, ms.
    sample_sorted = np.sort(resid_ms)
    qq_theory = _qq_theoretical_quantiles(len(sample_sorted))
    # Reference line through 25th/75th sample-vs-theory quantiles.
    if len(sample_sorted) >= 4:
        q1_s, q3_s = np.percentile(sample_sorted, [25, 75])
        q1_t = np.percentile(qq_theory, 25)
        q3_t = np.percentile(qq_theory, 75)
        if q3_t != q1_t:
            slope_q = (q3_s - q1_s) / (q3_t - q1_t)
            intercept_q = q1_s - slope_q * q1_t
            ref_x = np.array([qq_theory.min(), qq_theory.max()])
            ref_y = intercept_q + slope_q * ref_x
        else:
            ref_x = qq_theory[:2]
            ref_y = sample_sorted[:2]
    else:
        ref_x = qq_theory[:2] if len(qq_theory) >= 2 else np.array([0.0, 1.0])
        ref_y = sample_sorted[:2] if len(sample_sorted) >= 2 else np.array([0.0, 1.0])

    # Sorted |residual| vs percentile.
    abs_sorted = np.sort(np.abs(resid_ms))
    abs_pct = (np.arange(1, len(abs_sorted) + 1) / max(1, len(abs_sorted))) * 100.0

    data = {
        "video_s": video_s.tolist(),
        "resid_ms": resid_ms.tolist(),
        "pick_video_s": pick_video_s,
        "pick_resid_ms": pick_resid_ms,
        "pick_correction_ms": pick_correction_ms,
        "dense_video_s": dense_t.tolist(),
        "dense_correction_ms": dense_correction_ms.tolist(),
        "roll_video_s": roll_x.tolist(),
        "roll_rmse_ms": roll_rmse_ms.tolist(),
        "roll_win_s": roll_window_s,
        "qq_theory": qq_theory.tolist(),
        "qq_sample": sample_sorted.tolist(),
        "qq_ref_x": ref_x.tolist(),
        "qq_ref_y": ref_y.tolist(),
        "abs_sorted_ms": abs_sorted.tolist(),
        "abs_pct": abs_pct.tolist(),
    }

    html = (
        _HTML_TEMPLATE
        .replace("__SESSION__", session)
        .replace("__RECORDING__", recording_path.name)
        .replace("__N__", str(anchor.n))
        .replace("__N_REJ__", str(anchor.n_rejected))
        .replace("__RMSE_MS__", f"{anchor.rmse_s*1000:.3f}")
        .replace("__MAX_MS__", f"{anchor.max_abs_s*1000:.3f}")
        .replace("__DRIFT_PPM__", f"{anchor.drift_ppm:.2f}")
        .replace("__SIGMA_MS__", f"{anchor.sigma_residual_s*1000:.3f}")
        .replace("__FIT_UNC_MS__", f"{anchor.fit_uncertainty_s*1000:.4f}")
        .replace("__CUBIC_MS__", f"{anchor.cubic_rmse_improvement_s*1000:.3f}")
        .replace("__SPAN_S__", f"{span:.1f}")
        .replace("__ACCUM_MS__", f"{accum_ms:.2f}")
        .replace("__BIAS_MS__", f"{bias_ms:.4f}")
        .replace("__P95_MS__", f"{p95_ms:.3f}")
        .replace("__ROLL_WIN__", f"{roll_window_s:.0f}")
        .replace("__DATA__", json.dumps(data))
    )
    out_html.write_text(html, encoding="utf-8")

    # Copy plotly.min.js next to HTML if available.
    if plotly_src is not None and plotly_src.exists():
        dst = out_html.parent / "plotly.min.js"
        if not dst.exists():
            shutil.copyfile(plotly_src, dst)


# -----------------------------
# Main
# -----------------------------


def _resolve_plotly_src() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent / "plotly.min.js",
        Path(__file__).resolve().parent.parent
            / "data_pipeline" / "assets" / "plotly.min.js",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Map relative-time floats to UTC/GPST via session "
                    "TimeAnchor; emit drift report."
    )
    ap.add_argument("session_dir", type=Path,
                    help="Folder with recording_*.txt and recording_*.mp4")
    ap.add_argument("--times", type=Path, required=True,
                    help="File of relative-time floats (one per line or CSV).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path (default: <times>.corrected.csv).")
    ap.add_argument("--html", type=Path, default=None,
                    help="Drift HTML path (default: <session>/drift_report.html).")
    ap.add_argument("--offset-s", type=float, default=0.0,
                    help="t_video = t_relative + offset_s (default 0.0).")
    ap.add_argument("--roll-window-s", type=float, default=30.0,
                    help="Rolling-RMSE window length [s] (default 30).")
    ap.add_argument("--no-html", action="store_true",
                    help="Skip drift HTML emission.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    session_dir: Path = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"error: session_dir not a directory: {session_dir}", file=sys.stderr)
        return 2

    recording_txt = _pick_one(session_dir, "recording_*.txt")
    recording_mp4 = _pick_one(session_dir, "recording_*.mp4")

    pairs = _read_anchor_pairs(recording_txt)
    times = _read_relative_times(args.times.resolve())

    # Fit twice: with MAD-based outlier rejection (robust) and without (raw).
    # Emit a CSV + HTML for each, suffixed `.robust` and `.raw`. Lets the
    # user compare directly what rejection is buying / costing.
    variants: List[tuple[str, TimeAnchor]] = [
        ("robust", fit_time_anchor(pairs, robust=True)),
        ("raw",    fit_time_anchor(pairs, robust=False)),
    ]

    default_csv_stem = args.out or args.times.with_suffix("").with_name(
        args.times.stem + ".corrected.csv"
    )
    default_html_path = args.html or (session_dir / "drift_report.html")
    plotly_src = _resolve_plotly_src()

    for tag, anchor in variants:
        video_start_utc_s = anchor.video_ns_to_utc_s(0.0)
        rows = _build_rows(
            times, anchor,
            offset_s=args.offset_s,
            video_start_utc_s=video_start_utc_s,
        )
        out_csv = default_csv_stem.with_suffix("").with_name(
            default_csv_stem.stem + f".{tag}" + default_csv_stem.suffix
        )
        n = write_corrected_csv(
            out_csv, rows,
            video_start_utc_s=video_start_utc_s,
            mp4_basename=recording_mp4.name,
            session=session_dir.name,
        )
        print(
            f"[{tag}] wrote {n} rows -> {out_csv}; "
            f"anchor n={anchor.n} (rej={anchor.n_rejected}) "
            f"rmse={anchor.rmse_s*1000:.2f}ms "
            f"max={anchor.max_abs_s*1000:.2f}ms "
            f"drift={anchor.drift_ppm:.2f}ppm "
            f"fit_unc={anchor.fit_uncertainty_s*1e6:.1f}us"
        )

        if not args.no_html:
            out_html = default_html_path.with_suffix("").with_name(
                default_html_path.stem + f".{tag}" + default_html_path.suffix
            )
            residuals = _per_anchor_residuals(pairs, anchor)
            write_drift_html(
                out_html, anchor, pairs, residuals, rows,
                session=session_dir.name,
                recording_path=recording_txt,
                plotly_src=plotly_src,
                video_start_utc_s=video_start_utc_s,
                roll_window_s=args.roll_window_s,
            )
            print(f"[{tag}] wrote drift report -> {out_html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
