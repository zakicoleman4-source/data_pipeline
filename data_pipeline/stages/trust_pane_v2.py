"""Build trust-v2 pane HTML: 5-panel viewer colored by multi-signal trust.

Panels:
  1. Header — title + KPI pills (green/blue/orange/red counts + %) + guarantee
  2. Top row — 3D scatter3d (Local-frame) left, 2D map (lat/lon) right
  3. Timeline strip — each epoch as thin colored bar
  4. Signal panel (2x2 grid) — top 4 signals with p95 threshold line
  5. Footer — guarantee statement

Colors:
  high     = #22c55e (green)
  vel_only = #3b82f6 (blue)
  pos_only = #f97316 (orange)
  low      = #ef4444 (red)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..epoch_weight_v2 import EpochWeightV2Options, smooth_epoch_weighted_v2
from ..geo import ecef_to_enu, llh_to_ecef
from ..parsers import parse_rtkpos, parse_imu
from ..pipeline import LogFn, make_logger
from ..trust_formula_v2 import (
    SIGNAL_NAMES, DISAGREE_GREEN, DISAGREE_RED,
    TrustConfigV2, compute_trust_v2,
)

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def build_trust_pane_v2(
    pos_file: Path,
    out_html: Path,
    *,
    sensors_txt: Optional[Path] = None,
    config: Optional[TrustConfigV2] = None,
    log: Optional[LogFn] = None,
) -> Path:
    """Build the v2 trust viewer HTML from a Post-processing .pos file.

    Parameters
    ----------
    pos_file : Path
        The external solver .pos file (Reference time timestamps, standard layout).
    out_html : Path
        Output HTML path.
    sensors_txt : Path, optional
        sensors_*.txt for Motion sensor data (enables Motion sensor-Q, NHC, ZUPT).
    config : TrustConfigV2, optional
        Override thresholds. Uses calibrated defaults when None.
    log : LogFn, optional
        Logging callback.

    Returns
    -------
    Path
        The written HTML file path.
    """
    log_ = make_logger(log)
    cfg = config or TrustConfigV2()

    # Step 1: parse Post-processing
    pos_rows = parse_rtkpos(pos_file)
    if not pos_rows:
        raise RuntimeError(f"No PPK rows in {pos_file}")

    # Step 2: optionally parse Motion sensor
    imu_rows = None
    if sensors_txt:
        try:
            imu_rows = parse_imu(sensors_txt)
        except Exception:
            pass

    # Step 3: run v2 smoother
    v2_opts = EpochWeightV2Options(
        zupt_enabled=True,
        nhc_enabled=True,
        nhc_heading_source="doppler",
        sigma_a_base=0.10,
    )
    v2 = smooth_epoch_weighted_v2(
        pos_rows, imu_rows=imu_rows, options=v2_opts, log=log_,
    )

    # Step 4: compute trust v2
    trust = compute_trust_v2(pos_rows, v2, config=cfg)
    log_(
        f"[trust-v2] high={trust.n_high} vel_only={trust.n_vel_only} "
        f"pos_only={trust.n_pos_only} low={trust.n_low}"
    )

    # Step 5: build Local-frame coordinates
    n = len(pos_rows)
    ref = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    t0 = pos_rows[0].utc_s

    t_rel: list[float] = []
    e_arr: list[float] = []
    n_arr: list[float] = []
    u_arr: list[float] = []
    lats: list[float] = []
    lons: list[float] = []

    for r in pos_rows:
        t_rel.append(r.utc_s - t0)
        x, y, z = llh_to_ecef(r.lat_deg, r.lon_deg, r.h_m)
        e, nn, uu = ecef_to_enu(x, y, z, ref)
        e_arr.append(e)
        n_arr.append(nn)
        u_arr.append(uu)
        lats.append(r.lat_deg)
        lons.append(r.lon_deg)

    # Step 6: top 4 signals (disagree-based labeling)
    top_signals = ["disagree", "fwd_bwd_disagree_h", "eff_sig", "innovation_h"]

    signal_thresholds: dict[str, float] = {
        "disagree": DISAGREE_GREEN,
        "fwd_bwd_disagree_h": DISAGREE_GREEN,
        "eff_sig": 2.0,
        "innovation_h": 2.5,
    }

    # Build signal_values for top signals
    signal_values: dict[str, list[float]] = {}
    for name in top_signals:
        idx = SIGNAL_NAMES.index(name)
        signal_values[name] = trust.signals[:, idx].tolist()

    # Step 7: assemble JSON data
    data = {
        "t": t_rel,
        "e": e_arr,
        "n": n_arr,
        "u": u_arr,
        "lat": lats,
        "lon": lons,
        "labels": trust.labels,
        "pos_score": trust.pos_score.tolist(),
        "vel_score": trust.vel_score.tolist(),
        "top_signals": top_signals,
        "signal_values": signal_values,
        "signal_thresholds": signal_thresholds,
        "n_total": n,
        "n_high": trust.n_high,
        "n_vel_only": trust.n_vel_only,
        "n_pos_only": trust.n_pos_only,
        "n_low": trust.n_low,
    }

    # Step 8: read template, inject data, write output
    template_path = _ASSETS / "trust_viewer_v2.html"
    template = template_path.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    log_(f"[trust-v2] wrote {out_html}")

    # Step 9: copy plotly.min.js next to output
    from .viewers import _copy_plotly_next_to
    _copy_plotly_next_to(out_html.parent)

    return out_html
