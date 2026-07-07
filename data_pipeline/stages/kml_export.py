"""Batch layered export for visual comparison in an external viewer.

Emits up to seven export files in one go so the user can drop the whole
folder into the viewer and toggle layers to compare smoothing variants side
by side. Each variant uses a distinct line colour so multiple tracks
overlay cleanly.

Layers
------
* ``ppk_raw``                 -- direct post-processed epochs, segmented and
                                 coloured by solver quality flag (Q1 green,
                                 Q2 orange, Q4 grey, Q5+ degraded red)
* ``ppk_gauss_gentle``        -- Gaussian-smoothed output with the
                                 ``gentle`` preset (xy_sigma=0.5 s,
                                 z_sigma=2 s)
* ``ppk_gauss_car``           -- ``car`` preset (2 s, 10 s)
* ``ppk_gauss_aggressive``    -- ``aggressive`` preset (5 s, 20 s)
* ``data_gnss_raw``           -- raw platform position fixes (provider=reference)
* ``data_flp``                -- fused-provider track
                                 (provider=fused / FUSED_LOCATION_PROVIDER)
* ``fused_bent``              -- fused-provider shape warped onto solver anchors via
                                 :func:`data_pipeline.fused_bend.bend_fused_to_ppk`

Any layer whose source data is empty (e.g. measurements file has no
fused-provider rows) is silently skipped and reported in the return value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..fused_bend import FusedBendOptions, bend_fused_to_ppk
from ..nhc import (
    AdaptiveFilterOptions, NhcOptions, adaptive_filter, apply_nhc,
)
from ..parsers import DataFix, PosRow, parse_data_fix, parse_rtkpos
from ..pipeline import LogFn, make_logger
from ..smoothing import gaussian_smooth


# The external solver Q -> human label + Export format colour (AABBGGRR = alpha, blue, green, red).
_Q_STYLE: dict[int, tuple[str, str]] = {
    1: ("Fix",       "ff00ff00"),   # green
    2: ("Float",     "ff00a5ff"),   # orange
    4: ("DGPS",      "ffaaaaaa"),   # grey
    5: ("Single",    "ff0000ff"),   # red
    6: ("Degraded",  "ff0000ff"),   # red
}
_Q_DEFAULT_STYLE = ("Unknown", "ff999999")

# Per-variant track colours (AABBGGRR).
# Only the variants proven ROBUST in the 8-dataset GT eval are exposed
# here. "Gaussian aggressive (5s/20s)" and "Gaussian gentle (0.5s/2s)"
# were dropped — aggressive over-smoothes Float-dominant data (+44 %
# worst-case), gentle is dominated by `car` on every dataset.
_TRACK_COLOR: dict[str, str] = {
    "adaptive":         "ff00ff00",  # bright green (champion)
    "gauss_car":        "ff00ffff",  # yellow (proven baseline)
    "data_gnss_raw":   "ffffaa00",  # light blue
    "data_flp":        "ff0080ff",  # orange-ish
    "fused_bent":       "ff00ff80",  # mint green
    "nhc_corrected":    "ffff80ff",  # pink-magenta (vehicle non-holonomic)
}


@dataclass
class KmlBatchOptions:
    """Per-layer toggles for :func:`export_all_kmls`."""

    # ROBUST layers only — every variant exposed here is proven to beat
    # raw Post-processing on every dataset in the 8-day GT eval (worst-case ≤ 0).
    # `gauss_gentle` and `gauss_aggressive` were intentionally dropped:
    # gentle is dominated by car on every dataset (-4.5 % vs -6.6 % mean);
    # aggressive can regress +44 % on Float-heavy data.
    include_raw_ppk:           bool = True
    include_adaptive:          bool = True   # NEW: data-conditional champion
    include_gaussian_car:      bool = True
    include_raw_gnss_data:    bool = True
    include_flp:               bool = True
    include_fused_bent:        bool = True
    # Vehicle non-holonomic constraint: kill the lateral Post-processing noise by
    # snapping each epoch's lateral residual to the local smoothed path
    # along the heading direction. See data_pipeline/nhc.py.
    include_nhc:               bool = True

    # FLP provider names (matched case-insensitively).
    flp_providers: tuple[str, ...] = (
        "fused", "FUSED", "FUSED_LOCATION_PROVIDER", "fused_location",
    )
    # Device-Signal provider names.
    raw_gnss_providers: tuple[str, ...] = ("gps", "GPS", "network", "NETWORK")
    line_width: float = 3.0
    fused_bend: FusedBendOptions = field(default_factory=FusedBendOptions)
    nhc: NhcOptions = field(default_factory=NhcOptions)
    adaptive: AdaptiveFilterOptions = field(default_factory=AdaptiveFilterOptions)


@dataclass(frozen=True)
class KmlBatchResult:
    """Paths actually written plus skipped layers."""

    written: list[Path]
    skipped: dict[str, str]   # variant -> reason


# ----------------------------------------------------------------------
# Export format writers
# ----------------------------------------------------------------------


def _kml_header(doc_name: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '<Document>\n'
        f'  <name>{_xml_escape(doc_name)}</name>\n'
    )


def _kml_footer() -> str:
    return '</Document>\n</kml>\n'


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _style_block(style_id: str, color_hex: str, width: float) -> str:
    return (
        f'  <Style id="{style_id}">\n'
        f'    <LineStyle><color>{color_hex}</color>'
        f'<width>{width:.1f}</width></LineStyle>\n'
        f'    <PolyStyle><color>{color_hex}</color></PolyStyle>\n'
        f'  </Style>\n'
    )


def _coords_block(coords: Sequence[tuple[float, float, float]]) -> str:
    # Export format order is lon,lat,alt
    parts = []
    for lon, lat, h in coords:
        if not (math.isfinite(lon) and math.isfinite(lat)):
            continue
        h_use = h if math.isfinite(h) else 0.0
        parts.append(f"{lon:.8f},{lat:.8f},{h_use:.3f}")
    if not parts:
        return ""
    return "      <coordinates>\n        " + "\n        ".join(parts) + "\n      </coordinates>\n"


def _linestring_placemark(
    name: str, coords: Sequence[tuple[float, float, float]],
    style_url: str,
) -> str:
    cb = _coords_block(coords)
    if not cb:
        return ""
    return (
        f'  <Placemark>\n'
        f'    <name>{_xml_escape(name)}</name>\n'
        f'    <styleUrl>#{style_url}</styleUrl>\n'
        f'    <LineString>\n'
        f'      <tessellate>1</tessellate>\n'
        f'      <altitudeMode>absolute</altitudeMode>\n'
        f'{cb}'
        f'    </LineString>\n'
        f'  </Placemark>\n'
    )


def _write_simple_kml(
    *, out_path: Path, doc_name: str,
    coords: Sequence[tuple[float, float, float]],
    color_hex: str, line_width: float,
) -> None:
    """One LineString in one colour."""
    parts = [_kml_header(doc_name)]
    parts.append(_style_block("sty", color_hex, line_width))
    parts.append(_linestring_placemark(doc_name, coords, "sty"))
    parts.append(_kml_footer())
    out_path.write_text("".join(parts), encoding="utf-8")


def _write_q_colored_kml(
    *, out_path: Path, doc_name: str,
    rows: Sequence[PosRow], line_width: float,
) -> None:
    """Multi-segment LineString, split where Q changes."""
    parts = [_kml_header(doc_name)]
    # Emit one style per Q seen plus a default fallback.
    seen_q: set[int] = {r.quality for r in rows}
    for q in sorted(seen_q):
        label, color = _Q_STYLE.get(q, _Q_DEFAULT_STYLE)
        parts.append(_style_block(f"q{q}", color, line_width))

    # Split into runs of identical Q. Bridge each run-end to the next
    # run-start so the line never has visual gaps even at colour changes.
    if rows:
        segments: list[tuple[int, list[tuple[float, float, float]]]] = []
        cur_q = rows[0].quality
        cur_pts: list[tuple[float, float, float]] = []
        for r in rows:
            pt = (r.lon_deg, r.lat_deg, r.h_m)
            if r.quality != cur_q and cur_pts:
                # Bridge: append current point to old run, then start new.
                cur_pts.append(pt)
                segments.append((cur_q, cur_pts))
                cur_q = r.quality
                cur_pts = [pt]
            else:
                cur_pts.append(pt)
                cur_q = r.quality
        if cur_pts:
            segments.append((cur_q, cur_pts))

        for i, (q, pts) in enumerate(segments):
            label, _ = _Q_STYLE.get(q, _Q_DEFAULT_STYLE)
            parts.append(_linestring_placemark(
                f"seg {i+1} Q={q} ({label}) n={len(pts)}",
                pts, f"q{q}",
            ))
    parts.append(_kml_footer())
    out_path.write_text("".join(parts), encoding="utf-8")


# ----------------------------------------------------------------------
# Smoothing helpers
# ----------------------------------------------------------------------


def _smooth_ppk(
    rows: Sequence[PosRow], xy_sigma_s: float, z_sigma_s: float,
) -> list[tuple[float, float, float]]:
    """Apply Gaussian smoothing in seconds (per-axis lat/lon/h)."""
    if not rows:
        return []
    if xy_sigma_s <= 0 and z_sigma_s <= 0:
        return [(r.lon_deg, r.lat_deg, r.h_m) for r in rows]

    # Estimate sample rate from median dt.
    ts = [r.utc_s for r in rows]
    dts = [b - a for a, b in zip(ts, ts[1:]) if (b - a) > 1e-6]
    if not dts:
        return [(r.lon_deg, r.lat_deg, r.h_m) for r in rows]
    median_dt = sorted(dts)[len(dts) // 2]
    fps = 1.0 / median_dt if median_dt > 0 else 1.0
    xy_samples = max(1.0, xy_sigma_s * fps) if xy_sigma_s > 0 else 0.0
    z_samples  = max(1.0, z_sigma_s  * fps) if z_sigma_s  > 0 else 0.0

    lats = [r.lat_deg for r in rows]
    lons = [r.lon_deg for r in rows]
    hs   = [r.h_m     for r in rows]
    lat_s = gaussian_smooth(lats, xy_samples) if xy_samples > 0 else lats
    lon_s = gaussian_smooth(lons, xy_samples) if xy_samples > 0 else lons
    h_s   = gaussian_smooth(hs,   z_samples ) if z_samples  > 0 else hs
    return list(zip(lon_s, lat_s, h_s))


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------


def export_all_kmls(
    *,
    pos_file: Path,
    measurements_txt: Optional[Path],
    out_dir: Path,
    options: Optional[KmlBatchOptions] = None,
    log: Optional[LogFn] = None,
) -> KmlBatchResult:
    """Generate up to seven Export format files in ``out_dir``.

    Parameters
    ----------
    pos_file
        The external solver ``.pos`` file (required for all Post-processing-derived layers).
    measurements_txt
        the source app ``measurements_*.txt`` (required for the FLP, raw-Signal,
        and fused-bent layers). Pass ``None`` to skip them.
    out_dir
        Folder receiving the ``.export format`` files. Created if missing.
    """
    log_ = make_logger(log)
    options = options or KmlBatchOptions()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: dict[str, str] = {}

    # Load Post-processing once.
    ppk: list[PosRow] = []
    if pos_file is not None and pos_file.is_file():
        ppk = parse_rtkpos(pos_file)
        log_(f"[kml] parsed .pos: {len(ppk)} rows")
    else:
        log_(f"[kml] no .pos provided — all PPK layers skipped")

    # Load device fixes lazily if needed.
    fixes: list[DataFix] = []
    if measurements_txt is not None and measurements_txt.is_file():
        fixes = parse_data_fix(measurements_txt)
        log_(f"[kml] parsed device Fix rows: {len(fixes)}")
    flp_set = {p.lower() for p in options.flp_providers}
    gnss_set = {p.lower() for p in options.raw_gnss_providers}
    flp = [f for f in fixes if (f.provider or "").lower() in flp_set]
    gnss = [f for f in fixes if (f.provider or "").lower() in gnss_set]

    def _emit(variant: str, fn: str, builder, *args, **kw):
        path = out_dir / fn
        try:
            builder(*args, **kw)
            written.append(path)
            log_(f"[kml]   wrote {fn}")
        except Exception as e:
            skipped[variant] = f"{type(e).__name__}: {e}"
            log_(f"[kml]   FAIL {fn}: {e}")

    # 1. Raw Post-processing, Q-coloured. Diagnostic only — NOT recommended for use.
    if options.include_raw_ppk:
        if not ppk:
            skipped["ppk_raw"] = "no .pos rows"
        else:
            _emit("ppk_raw", "ppk_raw.kml", _write_q_colored_kml,
                  out_path=out_dir / "ppk_raw.kml",
                  doc_name=("[diagnostic]  Raw PPK (segmented by Q)  "
                            "— use only to inspect quality; pick #1 ADAPTIVE for output"),
                  rows=ppk, line_width=options.line_width)

    # 2. Gaussian car — proven baseline (-6.60 % mean dRMS, never regresses).
    if options.include_gaussian_car:
        if not ppk:
            skipped["gauss_car"] = "no .pos rows"
        else:
            coords = _smooth_ppk(ppk, 2.0, 10.0)
            _emit("gauss_car", "ppk_gauss_car.kml", _write_simple_kml,
                  out_path=out_dir / "ppk_gauss_car.kml",
                  doc_name=("[#3 robust]  Gaussian car (2s/10s)  "
                            "GT eval: -6.60 % mean RMS, never regresses"),
                  coords=coords, color_hex=_TRACK_COLOR["gauss_car"],
                  line_width=options.line_width)

    # 3. ADAPTIVE — regime-conditional champion (-9.33 % mean, 8/8 wins).
    if options.include_adaptive:
        if not ppk:
            skipped["adaptive"] = "no .pos rows"
        else:
            ad_rows, regime = adaptive_filter(
                ppk, options=options.adaptive, log=log_,
            )
            coords = [(r.lon_deg, r.lat_deg, r.h_m) for r in ad_rows]
            _emit("adaptive", "ppk_adaptive.kml", _write_simple_kml,
                  out_path=out_dir / "ppk_adaptive.kml",
                  doc_name=(f"[#1 BEST]  ADAPTIVE [{regime}]  "
                            "GT eval: -9.33 % mean RMS, 8/8 datasets, never regresses"),
                  coords=coords, color_hex=_TRACK_COLOR["adaptive"],
                  line_width=options.line_width)

    # 5. Raw device-Signal.
    if options.include_raw_gnss_data:
        if not gnss:
            skipped["data_gnss_raw"] = "no GPS-provider Fix rows"
        else:
            coords = [(f.lon, f.lat, f.h) for f in gnss]
            _emit("data_gnss_raw", "data_gnss_raw.kml", _write_simple_kml,
                  out_path=out_dir / "data_gnss_raw.kml",
                  doc_name=f"Data GNSS raw (n={len(gnss)})",
                  coords=coords,
                  color_hex=_TRACK_COLOR["data_gnss_raw"],
                  line_width=options.line_width)

    # 6. FLP.
    if options.include_flp:
        if not flp:
            skipped["data_flp"] = "no FLP-provider Fix rows"
        else:
            coords = [(f.lon, f.lat, f.h) for f in flp]
            _emit("data_flp", "data_flp.kml", _write_simple_kml,
                  out_path=out_dir / "data_flp.kml",
                  doc_name=f"FLP track (n={len(flp)})",
                  coords=coords,
                  color_hex=_TRACK_COLOR["data_flp"],
                  line_width=options.line_width)

    # 7. NHC-corrected Post-processing (vehicle non-holonomic constraint).
    # Heading source: 'auto' picks Motion sensor if attitude provided to the caller,
    # else Rate-signal from .pos, else coords-Δ. The Export format batch entry-point
    # doesn't take Motion sensor input today so it falls through to Rate-signal — see
    # data_pipeline/nhc.py:apply_nhc for the standalone Motion sensor-driven call.
    if options.include_nhc:
        if not ppk:
            skipped["nhc"] = "no .pos rows"
        else:
            try:
                nhc_res = apply_nhc(ppk, options=options.nhc, log=log_)
            except Exception as e:
                skipped["nhc"] = f"{type(e).__name__}: {e}"
                nhc_res = None
            if nhc_res is not None and nhc_res.n_modified == 0:
                skipped["nhc"] = (
                    f"no epochs modified (source={nhc_res.heading_source_used})"
                )
            elif nhc_res is not None:
                coords = [(r.lon_deg, r.lat_deg, r.h_m) for r in nhc_res.rows_out]
                lat_before_cm = nhc_res.lat_resid_before_m * 100
                lat_after_cm  = nhc_res.lat_resid_after_m * 100
                _emit("nhc", "ppk_nhc_corrected.kml", _write_simple_kml,
                      out_path=out_dir / "ppk_nhc_corrected.kml",
                      doc_name=(f"[diagnostic]  NHC standalone (source={nhc_res.heading_source_used})  "
                                f"lateral wobble {lat_before_cm:.0f}cm -> {lat_after_cm:.0f}cm  "
                                "— GT eval: -2 % mean alone; pick #1 ADAPTIVE for production"),
                      coords=coords,
                      color_hex=_TRACK_COLOR["nhc_corrected"],
                      line_width=options.line_width)

    # 8. Fused-bent (needs both FLP and Post-processing).
    if options.include_fused_bent:
        if not flp:
            skipped["fused_bent"] = "no FLP rows"
        elif not ppk:
            skipped["fused_bent"] = "no .pos rows"
        else:
            # Query at every Post-processing epoch so the bent track has the same
            # temporal resolution as the anchors. Could alternatively
            # query at FLP rate for a smoother visual.
            q_times = [r.utc_s for r in ppk]
            lat_b, lon_b, h_b, has, trust, info = bend_fused_to_ppk(
                flp, ppk, q_times, options=options.fused_bend,
            )
            coords = [
                (lon, lat, h)
                for lat, lon, h, ok in zip(lat_b, lon_b, h_b, has)
                if ok
            ]
            if not coords:
                skipped["fused_bent"] = "bend produced no valid points"
            else:
                _emit("fused_bent", "fused_bent.kml", _write_simple_kml,
                      out_path=out_dir / "fused_bent.kml",
                      doc_name=(f"Fused-bent (n={len(coords)} of {len(q_times)}, "
                                f"anchors used={info.n_anchors_used}/{info.n_anchors_used+info.n_anchors_rejected})"),
                      coords=coords,
                      color_hex=_TRACK_COLOR["fused_bent"],
                      line_width=options.line_width)

    log_(f"[kml] done: wrote {len(written)} files, skipped {len(skipped)}")
    for k, v in skipped.items():
        log_(f"[kml]   skipped {k}: {v}")
    return KmlBatchResult(written=written, skipped=skipped)


# ----------------------------------------------------------------------
# All-smoothers Export format export
# ----------------------------------------------------------------------

_SMOOTHER_COLORS: dict[str, str] = {
    "raw_ppk":              "ff0000ff",  # red
    "gaussian_car":         "ff00ffff",  # yellow
    "gaussian_aggressive":  "ff00cccc",  # dark yellow
    "cv_rts":               "ffffff00",  # cyan
    "cv_rts_pv":            "ffff8000",  # blue
    "gate_then_cv":         "ffff00ff",  # magenta
    "epoch_weight":         "ff00ff00",  # green (champion)
    "ekf_smoothed":         "ff00a5ff",  # orange
    "fgo":                  "ffff80ff",  # pink
    "kalman_simple_cv":     "ffffffff",  # white
}
_SMOOTHER_COLOR_DEFAULT = "ffcccccc"


def export_smoother_kmls(
    *,
    pos_file: Path,
    out_dir: Path,
    sensors_txt: Optional[Path] = None,
    stat_file: Optional[Path] = None,
    only: Optional[Sequence[str]] = None,
    line_width: float = 3.0,
    log: Optional[LogFn] = None,
) -> KmlBatchResult:
    """Export one Export format per registered smoother.

    Parameters
    ----------
    pos_file
        The external solver ``.pos`` file.
    out_dir
        Folder receiving the ``.export format`` files.
    sensors_txt
        ``sensors_*.txt`` for Motion sensor-based smoothers (ekf_smoothed, fgo).
        Pass ``None`` to skip those.
    stat_file
        ``.pos.stat`` file for epoch_weight.
    only
        Restrict to these smoother names. ``None`` = all.
    """
    from ..parsers import ImuRow, parse_imu
    from ..smoothers import list_smoothers, run_smoother

    log_ = make_logger(log)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: dict[str, str] = {}

    if not pos_file.is_file():
        log_(f"[kml-all] .pos not found: {pos_file}")
        return KmlBatchResult(written=written, skipped={"all": "no .pos file"})

    ppk = parse_rtkpos(pos_file)
    log_(f"[kml-all] parsed .pos: {len(ppk)} rows")
    if not ppk:
        return KmlBatchResult(written=written, skipped={"all": "0 rows in .pos"})

    imu_rows: list[ImuRow] = []
    if sensors_txt is not None and sensors_txt.is_file():
        try:
            imu_rows = parse_imu(sensors_txt)
            log_(f"[kml-all] parsed IMU: {len(imu_rows)} rows")
        except Exception as e:
            log_(f"[kml-all] IMU parse failed: {e}")

    names = list(only) if only else list_smoothers()
    log_(f"[kml-all] running {len(names)} smoothers: {', '.join(names)}")

    for name in names:
        kwargs: dict = {}
        if stat_file is not None:
            kwargs["stat_path"] = stat_file
        res = run_smoother(
            name, ppk, imu_rows=imu_rows or None, log=log_, **kwargs,
        )
        if not res.ok:
            skipped[name] = res.error_message or "failed"
            continue
        if not res.fused:
            skipped[name] = "0 output rows"
            continue

        coords = [(r.lon_deg, r.lat_deg, r.h_m) for r in res.fused]
        color = _SMOOTHER_COLORS.get(name, _SMOOTHER_COLOR_DEFAULT)
        fn = f"smoother_{name}.kml"
        hrmse_str = f"  hRMSE={res.hrmse_m:.3f}m" if res.hrmse_m else ""
        doc_name = f"{name} (n={res.n_output}){hrmse_str}"

        path = out_dir / fn
        try:
            _write_simple_kml(
                out_path=path, doc_name=doc_name,
                coords=coords, color_hex=color,
                line_width=line_width,
            )
            written.append(path)
            log_(f"[kml-all]   wrote {fn}")
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            log_(f"[kml-all]   FAIL {fn}: {e}")

    log_(f"[kml-all] done: {len(written)} written, {len(skipped)} skipped")
    for k, v in skipped.items():
        log_(f"[kml-all]   skipped {k}: {v}")
    return KmlBatchResult(written=written, skipped=skipped)
