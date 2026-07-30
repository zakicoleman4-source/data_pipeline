"""Per-frame trajectory export: join extracted frame times + the per-frame
coordinate CSV into one client CSV with, per frame:

    Image, t_audio_s, t_audio_chop_s, utc_s, utc_iso,
    latitude, longitude, altitude_m,           (WGS84 geographic)

``t_audio_s`` is absolute (seconds from the FULL recording's audio sample 0);
``t_audio_chop_s`` is chop-relative (0 at THIS clip's first frame) — what you
want when exporting a chopped clip. The chop-relative column is derived from the
frames' own times, so it survives deletion of the original long recording (only
absolute UTC/audio need the source anchors).
    utm_zone, utm_easting, utm_northing,        (auto UTM zone for the capture)
    vE_mps, vN_mps, vU_mps,                     (geographic East/North/Up velocity)
    speed_mps, azimuth_deg                      (3D speed; geographic bearing 0=N)

Velocity is always in the geographic ENU frame (East/North/Up), so the azimuth
of the speed vector (``atan2(vE, vN)``) is a true geographic bearing regardless
of the position CRS. It is taken from the coordinate CSV's Doppler columns
(``DopplerVe_mps`` / ``DopplerVn_mps`` / ``DopplerVu_mps``) when present, and
otherwise derived from consecutive frame positions.

``t_audio_s`` / ``utc_s`` / ``utc_iso`` come from the repo's frame-time table,
which needs the raw session directory (for the boot->UTC and audio-origin
anchors). Without a session, the video-relative time (``t_video_s``) is emitted
in the ``t_audio_s`` column and UTC is left blank.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .geo import heading_from_enu, llh_to_ecef

LogFn = Callable[[str], None]

OUT_HEADER = (
    "Image", "t_audio_s", "t_audio_chop_s", "utc_s", "utc_iso",
    "latitude", "longitude", "altitude_m",
    "utm_zone", "utm_easting", "utm_northing",
    "vE_mps", "vN_mps", "vU_mps", "speed_mps", "azimuth_deg",
    # coordinate-delta velocity (from consecutive frame positions / Δt)
    "vE_cd_mps", "vN_cd_mps", "vU_cd_mps", "speed_cd_mps", "azimuth_cd_deg",
    # weighted_v2-smoothed position (WGS84 + UTM) + velocity (only when a phone
    # PPK .pos is given AND UTC is available to align the .pos to the frames)
    "sm_latitude", "sm_longitude", "sm_altitude_m",
    "sm_utm_easting", "sm_utm_northing",
    "sm_vE_mps", "sm_vN_mps", "sm_vU_mps", "sm_speed_mps", "sm_azimuth_deg",
)

_VE = ("DopplerVe_mps", "Ve_mps", "vE_mps", "ve_mps")
_VN = ("DopplerVn_mps", "Vn_mps", "vN_mps", "vn_mps")
_VU = ("DopplerVu_mps", "Vu_mps", "vU_mps", "vu_mps")

# Max time (s) a smoothed-velocity query point may sit from BOTH bracketing
# .pos epochs before we refuse to interpolate across the pair (see
# ``_smoothed_at_frames``: a real signal gap, e.g. a tunnel, can bracket a
# turn, and blending the two epochs' velocities would fabricate a heading
# that never happened).
_SMOOTHED_VEL_MAX_GAP_S = 1.5


def _enu_vec_to_ecef(ve: float, vn: float, vu: float,
                      lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Rotate a FREE vector (e.g. velocity) from local-ENU-at-(lat,lon) into
    ECEF. Same rotation as ``geo.enu_to_llh`` minus the ref-point translation
    (a velocity has no origin to translate)."""
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    vx = -so * ve - sl * co * vn + cl * co * vu
    vy = co * ve - sl * so * vn + cl * so * vu
    vz = cl * vn + sl * vu
    return vx, vy, vz


def _ecef_vec_to_enu(vx: float, vy: float, vz: float,
                      lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Rotate a FREE vector (e.g. velocity) from ECEF into local-ENU-at-(lat,
    lon). Same rotation as ``geo.ecef_to_enu`` minus the ref-point
    translation."""
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    ve = -so * vx + co * vy
    vn = -sl * co * vx - sl * so * vy + cl * vz
    vu = cl * co * vx + cl * so * vy + sl * vz
    return ve, vn, vu


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _find(headers, cands):
    low = {h.lower(): h for h in headers}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def utm_zone_for(lat: float, lon: float) -> tuple[int, str]:
    """(EPSG, 'zoneN'/'zoneS') for the UTM zone containing (lat, lon)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    north = lat >= 0.0
    return (32600 if north else 32700) + zone, f"{zone}{'N' if north else 'S'}"


def _read_georef(path: Path):
    """-> (order[stem], {stem: dict(lat, lon, alt, ve, vn, vu)}), Doppler NaN if absent."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        headers = rd.fieldnames or []
        name_c = _find(headers, ("Image", "Label", "name", "frame"))
        lat_c = _find(headers, ("Latitude", "lat"))
        lon_c = _find(headers, ("Longitude", "lon", "lng"))
        alt_c = _find(headers, ("Altitude", "alt", "height", "h"))
        ve_c, vn_c, vu_c = _find(headers, _VE), _find(headers, _VN), _find(headers, _VU)
        if not (name_c and lat_c and lon_c):
            raise ValueError(f"{path}: need Image + Latitude + Longitude columns "
                             f"(got {headers})")
        order, out = [], {}
        for r in rd:
            stem = Path((r.get(name_c) or "").strip()).stem
            if not stem:
                continue
            lat, lon = _f(r.get(lat_c)), _f(r.get(lon_c))
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if stem not in out:
                order.append(stem)
            out[stem] = dict(
                lat=lat, lon=lon, alt=_f(r.get(alt_c)) if alt_c else float("nan"),
                ve=_f(r.get(ve_c)) if ve_c else float("nan"),
                vn=_f(r.get(vn_c)) if vn_c else float("nan"),
                vu=_f(r.get(vu_c)) if vu_c else float("nan"),
            )
    return order, out


def _video_relative_times(frame_times_csv: Path):
    """{stem: (t_video_s, '', '')} — video-relative time in the t_audio slot, UTC blank."""
    out = {}
    with Path(frame_times_csv).open("r", newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        tcol = _find(rd.fieldnames or [], ("t_video_s", "t_audio_s", "time_s", "t"))
        ncol = _find(rd.fieldnames or [], ("Image", "name", "frame"))
        for r in rd:
            stem = Path((r.get(ncol) or "").strip()).stem
            if stem:
                out[stem] = ((r.get(tcol) or "").strip() if tcol else "", "", "")
    return out


def _time_table(frame_times_csv: Path, session_dir, anchors, log: LogFn):
    """{stem: (t_audio_s, utc_s, utc_iso)}; falls back to video-relative time.

    If a session is given but its anchors can't be resolved — e.g. the client
    chopped the clip and DELETED the long source recording, so the parent's
    audio/boot->UTC anchors are gone — we degrade to video-relative time (UTC
    blank) instead of failing. The chop-relative audio column is still produced
    from the frames' own times downstream.
    """
    if session_dir is None and anchors is None:
        log("[export] no session dir/anchors — emitting video-relative time (no UTC)")
        return _video_relative_times(frame_times_csv)
    from .frame_time_table import build_frame_time_table
    # BUG (timing/concurrency): this used to be a single fixed filename in the
    # shared temp dir. Two overlapping exports (e.g. the GUI running a batch,
    # or two sessions processed in parallel worker threads) would race on the
    # same path -- one export could read back the OTHER export's frame-time
    # table, silently joining a different session's UTC/audio times onto this
    # session's frames. Use a per-call unique file and clean it up afterward.
    fd, tmp_name = tempfile.mkstemp(prefix="_frame_time_table_export_", suffix=".csv")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        try:
            build_frame_time_table(Path(frame_times_csv), session_dir=session_dir,
                                   anchors=anchors, out_csv=tmp, write_html=False, log=log)
        except Exception as e:
            log(f"[export] could not resolve session anchors ({e}) — the source "
                "recording may be gone; falling back to video-relative time (no UTC)")
            return _video_relative_times(frame_times_csv)
        out = {}
        with tmp.open("r", newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                stem = Path((r.get("Image") or "").strip()).stem
                if stem:
                    out[stem] = (r.get("t_audio_s", "") or "", r.get("utc_s", "") or "",
                                 r.get("utc_iso", "") or "")
        return out
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _derive_velocity(order, geo, times):
    """ENU velocity per frame from consecutive positions (fallback when Doppler
    is absent). Centred difference where possible, one-sided at the ends."""
    # BUG (timing): the time base for dt used to be picked independently for
    # EACH of the two neighbour timestamps (utc_s if that specific frame has
    # it, else t_audio/video-s otherwise). utc_s is an absolute POSIX epoch
    # (~1.7e9 s) while t_audio_s/t_video_s are small clip-relative numbers, so
    # if one neighbour had utc_s and the other didn't, dt would subtract two
    # different clocks and silently produce a garbage velocity instead of a
    # NaN. Pick ONE basis for the whole table up front (utc_s only if every
    # frame that appears in the time table actually has it) so a dt is either
    # computed on a single consistent clock or -- if a specific neighbour is
    # missing that clock's value -- correctly yields NaN.
    have_all_utc = bool(times) and all(
        math.isfinite(_f(t[1])) for t in times.values()
    )
    idx = 1 if have_all_utc else 0   # utc_s column, else t_audio/video column

    def tval(stem):
        t = times.get(stem, ("", "", ""))
        return _f(t[idx])
    enu = {}
    if not order:
        return enu
    # BUG (frame-of-reference — same class as `_smoothed_at_frames`): this
    # used to convert EVERY frame's position into a SINGLE fixed-origin ENU
    # tangent plane (anchored at ``order[0]``) and difference there. A
    # position delta between two nearby frames is a free vector, so its
    # direction is only correct in the tangent plane it's expressed in; the
    # module docstring promises "azimuth ... is a true geographic bearing"
    # (i.e. each frame's OWN local-level frame, matching the Doppler
    # convention), but a fixed far-away origin's East/North axes rotate away
    # from a frame's own East/North as that frame's position moves away from
    # the origin (meridian convergence) — a synthetic 120 km track showed the
    # same 0.67 deg peak / 0.39 deg RMS azimuth bias as the smoothed-velocity
    # bug (negligible on day15's ~0.8 km track). Fix: work in ECEF (no
    # translation origin needed — differencing two ECEF positions cancels any
    # translation) and rotate the *delta* into each frame's OWN local ENU.
    alts = [geo[s]["alt"] for s in order if math.isfinite(geo[s]["alt"])]
    alt_fallback = float(np.median(alts)) if alts else 0.0
    xyz = {}
    for s in order:
        g = geo[s]
        a = g["alt"] if math.isfinite(g["alt"]) else alt_fallback
        xyz[s] = llh_to_ecef(g["lat"], g["lon"], a)
    for i, s in enumerate(order):
        j0, j1 = order[max(i - 1, 0)], order[min(i + 1, len(order) - 1)]
        dt = tval(j1) - tval(j0)
        if j0 == j1 or not math.isfinite(dt) or abs(dt) < 1e-6:
            enu[s] = (float("nan"), float("nan"), float("nan"))
            continue
        p0, p1 = xyz[j0], xyz[j1]
        dvx = (p1[0] - p0[0]) / dt
        dvy = (p1[1] - p0[1]) / dt
        dvz = (p1[2] - p0[2]) / dt
        g = geo[s]
        enu[s] = _ecef_vec_to_enu(dvx, dvy, dvz, g["lat"], g["lon"])
    return enu


def _smoothed_at_frames(pos_csv, times, order, log: LogFn):
    """weighted_v2-smoothed lat/lon/alt + ENU velocity per frame.

    Aligns the .pos epochs to the frames by absolute UTC (both the frame-time
    table's ``utc_s`` and ``parse_rtkpos``'s ``utc_s`` are POSIX UTC), so no
    GPST/leap ambiguity. Returns {stem: (lat, lon, alt, vE, vN, vU)} — empty
    (with a WARN) when UTC is unavailable or the .pos is too short.

    BUG (frame-of-reference): ``smooth_epoch_weighted_v2``'s Kalman state is
    ENU about a single FIXED origin (``pos[0]``) — see its ``ref = (pos[0]...)``
    — so ``res.vE_smooth/vN_smooth/vU_smooth`` are velocity components in
    THAT ONE tangent plane for every epoch. The Doppler columns this export
    also carries (``vE_mps`` etc, from the georef CSV) are the receiver's own
    local-level velocity AT EACH FRAME'S OWN lat/lon (that's what RTKLIB/a GNSS
    receiver reports per epoch) — a different, per-point frame. Emitting
    ``vE_smooth`` directly as ``sm_vE_mps`` therefore mixes two ENU
    conventions: fine right at ``pos[0]``, but the two tangent planes rotate
    apart (mainly meridian convergence) as a frame's own position moves away
    from ``pos[0]`` — same effect as `~/.claude` note on the analytic proof:
    a synthetic 120 km-east track showed 0.67 deg peak / 0.39 deg RMS azimuth
    bias from this alone (vs 0.005 deg on day15's ~0.8 km track, i.e. real but
    negligible at this session's scale — grows with track extent, so it will
    bite on long highway captures). Fix: rotate the smoothed velocity out of
    the ``pos[0]``-anchored frame and into each FRAME'S OWN local ENU (via
    ECEF, exact) before reporting it — the same convention as the Doppler and
    coordinate-delta velocity columns.

    BUG (gap interpolation): ``np.interp`` linearly blends velocity across
    ANY two bracketing .pos epochs, including a real signal gap (e.g. a
    tunnel/urban canyon). If the vehicle turns during the gap, the blended
    velocity is a fabricated average of the pre- and post-turn headings —
    wrong for the entire gap, not just imprecise. Mirror the codebase's
    existing ``interp_pos``/``interp_pos_with_velocity`` convention
    (parsers.py): refuse to report smoothed velocity when the bracketing
    .pos epochs are more than ``_SMOOTHED_VEL_MAX_GAP_S`` apart.
    """
    futc = {s: _f(times.get(s, ("", "", ""))[1]) for s in order}
    if not any(math.isfinite(v) for v in futc.values()):
        log("[export] smoothed: needs UTC (raw session folder) to align the .pos "
            "to the frames — smoothed columns left blank")
        return {}
    try:
        from .parsers import parse_rtkpos
        from .epoch_weight_v2 import smooth_epoch_weighted_v2
        from .geo import enu_to_llh
    except Exception as e:                       # pragma: no cover
        log(f"[export] smoothed: unavailable ({e})")
        return {}
    pos = parse_rtkpos(Path(pos_csv))
    if len(pos) < 5:
        log(f"[export] smoothed: only {len(pos)} .pos epochs — skipped")
        return {}
    res = smooth_epoch_weighted_v2(pos, None)    # GNSS-only (no IMU)
    ref = (pos[0].lat_deg, pos[0].lon_deg, pos[0].h_m)
    ep = np.array([p.utc_s for p in pos]); o = np.argsort(ep); ep = ep[o]
    E, N, U = res.E_smooth[o], res.N_smooth[o], res.U_smooth[o]
    vE, vN, vU = res.vE_smooth[o], res.vN_smooth[o], res.vU_smooth[o]
    ll = np.array([enu_to_llh(float(E[i]), float(N[i]), float(U[i]), ref)
                   for i in range(len(E))])
    lo_t, hi_t = ep[0], ep[-1]
    n_gap_skipped = 0
    out = {}
    for s in order:
        t = futc[s]
        if not math.isfinite(t) or t < lo_t - 1.0 or t > hi_t + 1.0:
            continue                             # frame outside the .pos span
        sla = float(np.interp(t, ep, ll[:, 0]))
        slo = float(np.interp(t, ep, ll[:, 1]))
        sal = float(np.interp(t, ep, ll[:, 2]))
        # Gap check: reject velocity (not position) when the bracketing .pos
        # epochs straddling `t` are further apart than the real-gap threshold.
        idx = int(np.searchsorted(ep, t))
        lo_i, hi_i = max(idx - 1, 0), min(idx, len(ep) - 1)
        bracket_gap = ep[hi_i] - ep[lo_i] if hi_i > lo_i else 0.0
        if bracket_gap > _SMOOTHED_VEL_MAX_GAP_S:
            n_gap_skipped += 1
            sve = svn = svu = float("nan")
        else:
            sve_ref = float(np.interp(t, ep, vE))
            svn_ref = float(np.interp(t, ep, vN))
            svu_ref = float(np.interp(t, ep, vU))
            # Rotate the pos[0]-anchored velocity into THIS frame's own local
            # ENU (via ECEF) so it matches the Doppler/coord-delta convention.
            vx, vy, vz = _enu_vec_to_ecef(sve_ref, svn_ref, svu_ref, ref[0], ref[1])
            sve, svn, svu = _ecef_vec_to_enu(vx, vy, vz, sla, slo)
        out[s] = (sla, slo, sal, sve, svn, svu)
    log(f"[export] smoothed (weighted_v2): {len(out)}/{len(order)} frames from "
        f"{len(pos)} .pos epochs ({res.n_zupt_updates} ZUPT, {res.n_nhc_updates} NHC)"
        + (f", {n_gap_skipped} velocity-blanked (>{_SMOOTHED_VEL_MAX_GAP_S:.1f}s .pos gap)"
           if n_gap_skipped else ""))
    return out


def build_frame_export(
    frame_times_csv: Path | str,
    georef_csv: Path | str,
    out_csv: Path | str,
    *,
    session_dir: Optional[Path | str] = None,
    anchors=None,
    pos_csv: Optional[Path | str] = None,
    log: Optional[LogFn] = None,
) -> Path:
    """Write the per-frame trajectory CSV (see module docstring). Returns the path.

    ``pos_csv`` (the phone PPK ``.pos``) enables the weighted_v2-smoothed
    position + velocity columns; it needs UTC (a session) to align to frames."""
    log_: LogFn = log or (lambda s: None)
    out_csv = Path(out_csv)
    order, geo = _read_georef(Path(georef_csv))
    if not geo:
        raise ValueError(f"{georef_csv}: no usable coordinate rows")
    times = _time_table(Path(frame_times_csv), session_dir, anchors, log_)
    # Chop-relative audio time: 0 at THIS clip's first frame. `t_audio_s` is
    # absolute (seconds from the FULL recording's audio sample 0); for a chopped
    # clip the client wants "seconds from the beginning of THIS clip's audio",
    # so subtract the chop's start (the min t_audio over its frames). Uses ONLY
    # the frames' own times — no dependency on the original recording, so it
    # still works after the client deletes the long source video.
    _ta_fin = [v for v in (_f(times.get(s, ("", "", ""))[0]) for s in order)
               if math.isfinite(v)]
    chop_t0 = min(_ta_fin) if _ta_fin else float("nan")
    if math.isfinite(chop_t0):
        log_(f"[export] chop-relative audio: t_audio_chop_s = t_audio_s - {chop_t0:.6f} "
             f"(this clip starts {chop_t0:.3f} s into the source audio)")
    # coordinate-delta velocity — ALWAYS computed (separate from Doppler)
    coord_delta = _derive_velocity(order, geo, times)
    smoothed = {}
    if pos_csv is not None and Path(pos_csv).is_file():
        try:
            smoothed = _smoothed_at_frames(pos_csv, times, order, log_)
        except Exception as e:
            log_(f"[export] smoothed skipped: {e}")

    # UTM zone from the median position of the capture (single zone per file).
    mid = geo[order[len(order) // 2]]
    epsg, zone = utm_zone_for(mid["lat"], mid["lon"])
    from pyproj import Transformer
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    log_(f"[export] {len(order)} frames · UTM zone {zone} (EPSG:{epsg})")

    have_doppler = any(math.isfinite(geo[s]["ve"]) and math.isfinite(geo[s]["vn"])
                       for s in order)
    all_doppler = all(math.isfinite(geo[s]["ve"]) and math.isfinite(geo[s]["vn"])
                       for s in order)
    # BUG (logic): this used to gate computing `derived` on `have_doppler`
    # (ANY row has Doppler) and then skip deriving entirely, leaving `derived`
    # an empty dict. In a file where only SOME rows carry Doppler, the
    # per-row fallback below (`derived.get(s, nan*3)`) would then find
    # nothing for the Doppler-less rows and silently emit NaN velocity for
    # them instead of falling back to position-derived velocity. Always
    # compute `derived` so the per-row fallback has something to fall back
    # to; skip only the (harmless, just wasted work) case where every row
    # already has Doppler.
    derived = {} if all_doppler else _derive_velocity(order, geo, times)
    log_("[export] velocity source: " + (
        "Doppler (coord CSV)" if all_doppler else
        "Doppler + derived (partial coverage)" if have_doppler else
        "derived from consecutive positions"))

    def c(x, nd=6):
        return "" if (isinstance(x, float) and not math.isfinite(x)) else \
            (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    n_utc = 0
    n_skipped = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_HEADER)
        for s in order:
            # BUG (logic): a single malformed/edge-case row (e.g. a transform
            # failure) used to raise out of the whole export, discarding every
            # row already written and every row still to come. Isolate each
            # row so one bad frame is skipped (and logged) instead of killing
            # the export.
            try:
                g = geo[s]
                ta, utc, iso = times.get(s, ("", "", ""))
                taf = _f(ta)
                ta_chop = (taf - chop_t0) if (math.isfinite(taf) and math.isfinite(chop_t0)) \
                    else float("nan")
                if str(utc).strip():
                    n_utc += 1
                E, N = to_utm.transform(g["lon"], g["lat"])
                ve, vn, vu = g["ve"], g["vn"], g["vu"]
                if not (math.isfinite(ve) and math.isfinite(vn)):
                    # No usable Doppler horizontal velocity for this row --
                    # fall back to the fully position-derived ENU triple.
                    ve, vn, vu = derived.get(s, (float("nan"),) * 3)
                elif not math.isfinite(vu):
                    # BUG (logic): Doppler gave ve/vn but not vu (common --
                    # many receivers only report horizontal Doppler). The old
                    # code left vu as NaN here and the speed_mps sum silently
                    # dropped it from the sum-of-squares, which is equivalent
                    # to assuming vu == 0 without saying so, and quietly turns
                    # the documented "3D speed" into a 2D one. Backfill vu
                    # alone from the position-derived estimate (if any) so
                    # speed_mps stays a genuine 3D magnitude when possible.
                    vu = derived.get(s, (float("nan"), float("nan"), float("nan")))[2]
                spd = math.sqrt(sum(v * v for v in (ve, vn, vu) if math.isfinite(v))) \
                    if math.isfinite(ve) and math.isfinite(vn) else float("nan")
                az = heading_from_enu(ve, vn) if (math.isfinite(ve) and math.isfinite(vn)
                                                  and (ve or vn)) else float("nan")
                # coordinate-delta velocity (position-derived), always emitted
                cve, cvn, cvu = coord_delta.get(s, (float("nan"),) * 3)
                cspd = math.sqrt(sum(v * v for v in (cve, cvn, cvu) if math.isfinite(v))) \
                    if math.isfinite(cve) and math.isfinite(cvn) else float("nan")
                caz = heading_from_enu(cve, cvn) if (math.isfinite(cve) and math.isfinite(cvn)
                                                     and (cve or cvn)) else float("nan")
                # weighted_v2-smoothed position (WGS84 + UTM) + velocity
                sm = smoothed.get(s)
                sla, slo, sal, sve, svn, svu = sm if sm else (float("nan"),) * 6
                # smoothed UTM in the SAME zone as the raw position
                if math.isfinite(sla) and math.isfinite(slo):
                    smE, smN = to_utm.transform(slo, sla)
                else:
                    smE = smN = float("nan")
                sspd = math.sqrt(sum(v * v for v in (sve, svn, svu) if math.isfinite(v))) \
                    if math.isfinite(sve) and math.isfinite(svn) else float("nan")
                saz = heading_from_enu(sve, svn) if (math.isfinite(sve) and math.isfinite(svn)
                                                     and (sve or svn)) else float("nan")

                w.writerow([
                    s, ta, c(ta_chop, 6), utc, iso,
                    c(g["lat"], 9), c(g["lon"], 9), c(g["alt"], 4),
                    zone, c(E, 3), c(N, 3),
                    c(ve, 4), c(vn, 4), c(vu, 4), c(spd, 4), c(az, 3),
                    c(cve, 4), c(cvn, 4), c(cvu, 4), c(cspd, 4), c(caz, 3),
                    c(sla, 9), c(slo, 9), c(sal, 4), c(smE, 3), c(smN, 3),
                    c(sve, 4), c(svn, 4), c(svu, 4), c(sspd, 4), c(saz, 3),
                ])
            except Exception as exc:  # noqa: BLE001 - isolate one bad row
                n_skipped += 1
                log_(f"[export] WARNING: skipped frame {s!r}: {exc!r}")
    log_(f"[export] wrote {out_csv} ({len(order)} rows, {n_utc} with UTC"
         + (f", {n_skipped} skipped" if n_skipped else "") + ")")
    return out_csv
