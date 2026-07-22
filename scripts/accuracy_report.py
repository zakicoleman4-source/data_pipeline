"""Deep, hardened accuracy report: camera-model-est &/or device GPS vs ground truth.

Reuses the client_pipeline repo's sigma_bands + geodesy so numbers match the
production tool. Robust to partial photogrammetry (only solved frames compared),
to a device georef with no time column (t0 recovered by correlating georef vs the
device .pos, ground-truth sampled at frame times), and to degenerate / missing inputs
(each source is computed independently; one failure does not sink the report).

    python scripts/accuracy_report.py         --gt ground_truth.pos --track device_track.pos         --georef georef.csv --ftimes frame_times.csv         --meta modelA=A/cameras_est.csv modelB=B/cameras_est.csv         --out report.html

Statistics: classic (sigma bands, CEP/DRMS/MRSE, percentiles), robust (median,
MAD, robust-sigma, trimmed RMS, outlier%), distribution shape (skew, kurtosis,
Anderson-Darling), inference (bootstrap 95% CI, lag-1 autocorrelation +
effective N), circular azimuth (Rayleigh test), error-ellipse geometry, scale &
drift, timing cross-check, and nominal-accuracy coverage.
"""
from __future__ import annotations
import argparse, csv, math, sys, json, warnings
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore")

# optional scipy — degrade gracefully if absent
try:
    from scipy import stats as _st
    from scipy.stats import bootstrap as _bootstrap
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ============================ CONFIG ============================
REPO_DEFAULT = Path(__file__).resolve().parents[1]   # repo root (this file is in scripts/)
OUT_DEFAULT  = "accuracy_report.html"

SPEED_FLOOR = 0.5
MOVING_MPS = 10.0 / 3.6      # 10 km/h — motion stats (speed/azimuth) reported only above this
MAX_STEP    = 1.0
THRESHOLDS  = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
PCTS        = (50, 68, 90, 95, 99)
SPEED_CLIP  = 8.0
BOOT_N      = 2000
SYNC_WARN_M = 10.0
RNG         = np.random.default_rng(0)   # deterministic bootstrap
# ==========================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=REPO_DEFAULT,
                   help="repo root providing the data_pipeline package (default: parent of scripts/)")
    p.add_argument("--ground-truth", "--gt", dest="gt", type=Path, required=True,
                   help="ground-truth trajectory .pos (survey-grade reference)")
    p.add_argument("--track", type=Path, required=True,
                   help="reference-device track .pos (used to recover the frame-time offset)")
    p.add_argument("--georef", type=Path, required=True,
                   help="per-frame device coordinates CSV (Image,Latitude,Longitude,Altitude,...)")
    p.add_argument("--ftimes", type=Path, required=True,
                   help="per-frame time CSV (Image,t_video_s)")
    p.add_argument("--meta", nargs="*", default=[],
                   help="one or more camera-model estimated-coordinate CSVs. Each becomes its "
                        "own source. Optional label: --meta run1=path1.csv run2=path2.csv")
    p.add_argument("--plotly", type=Path, default=None,
                   help="plotly.min.js to inline (default: auto-locate in the repo)")
    p.add_argument("--out", type=Path, default=Path(OUT_DEFAULT))
    p.add_argument("--no-meta", action="store_true", help="ignore all camera-model CSVs")
    p.add_argument("--epsg", type=int, default=None,
                   help="EPSG code of the camera-model CSVs when they are in a projected / "
                        "non-WGS84 CRS (e.g. 32636 for UTM 36N); reprojected to WGS84. "
                        "Per-source override: --meta label=path@<epsg>.")
    return p.parse_args()


def parse_meta_specs(items):
    """['label=path', 'path', 'label=path@epsg'] -> [(label, Path, epsg|None)].

    A trailing ``@<epsg>`` on the path forces reprojection of that source from
    the given projected CRS to WGS84 (overrides the global --epsg).
    """
    out, seen = [], {}
    for it in items:
        if "=" in it and not Path(it).is_file():
            label, path = it.split("=", 1)
        else:
            label, path = Path(it).parent.name or Path(it).stem, it
        epsg = None
        if "@" in path and not Path(path).is_file():
            path, _, ep = path.rpartition("@")
            try:
                epsg = int(ep)
            except ValueError:
                epsg = None
        label = label.strip() or "camera-model"
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}#{seen[label]}"
        out.append((label, Path(path), epsg))
    return out


A = parse_args()
sys.path.insert(0, str(A.repo))
try:
    from data_pipeline.photo_compare import sigma_bands, fit_similarity_robust
    from data_pipeline.geo import llh_to_ecef, ecef_to_enu, heading_from_enu
except Exception as e:
    sys.exit(f"FATAL: cannot import client_pipeline from {A.repo}: {e}")


def die(msg):
    sys.exit(f"FATAL: {msg}")


def need(path, what):
    if not Path(path).is_file():
        die(f"{what} not found: {path}")
    return Path(path)


def warn(msg):
    print(f"WARN: {msg}")


# ----------------------------- parsing -----------------------------
def _wrap(d):
    w = (np.asarray(d, float) + 180.0) % 360.0 - 180.0
    return np.where(w == -180.0, 180.0, w)


def parse_pos(path):
    """RTKLIB .pos -> (sod, lat, lon, h, vn, ve, vu). Velocity columns (16-18,
    1-based: vn ve vu m/s) are read when present, else NaN."""
    t, la, lo, h, vn, ve, vu = [], [], [], [], [], [], []
    for line in open(path):
        if line.startswith("%") or not line.strip():
            continue
        c = line.split()
        if len(c) < 5:
            continue
        try:
            hh, mm, ss = c[1].split(":")
            t.append(int(hh) * 3600 + int(mm) * 60 + float(ss))
            la.append(float(c[2])); lo.append(float(c[3])); h.append(float(c[4]))
        except ValueError:
            continue
        # velocity: standard llh .pos has 15 tokens; +9 => vn ve vu (+6 sd/cov)
        if len(c) >= 18:
            try:
                vn.append(float(c[15])); ve.append(float(c[16])); vu.append(float(c[17]))
            except ValueError:
                vn.append(float("nan")); ve.append(float("nan")); vu.append(float("nan"))
        else:
            vn.append(float("nan")); ve.append(float("nan")); vu.append(float("nan"))
    if not t:
        die(f"{path}: no epochs parsed")
    return map(np.array, (t, la, lo, h, vn, ve, vu))


def read_georef(path):
    fr, la, lo, al, acc = [], [], [], [], []
    ve, vn, vu = [], [], []          # phone Doppler ENU velocity per frame (NaN if absent)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        # Parse the required fields into locals first and only append once all
        # four lists can be extended together — appending field-by-field let a
        # mid-row failure (e.g. a blank Altitude) leave fr/la/lo one longer
        # than al, silently shifting every later row out of alignment.
        try:
            stem = Path(r["Image"]).stem
            lat_v = float(r["Latitude"]); lon_v = float(r["Longitude"]); alt_v = float(r["Altitude"])
        except (ValueError, KeyError, TypeError):
            continue
        # Accuracy is auxiliary — a blank/garbage AccuracyX/Y must not drop an
        # otherwise-valid position row.
        try:
            ax, ay = float(r.get("AccuracyX") or "nan"), float(r.get("AccuracyY") or "nan")
            acc_v = math.hypot(ax, ay) if math.isfinite(ax) and math.isfinite(ay) else float("nan")
        except (ValueError, TypeError):
            acc_v = float("nan")
        def _g(*names):
            for n in names:
                if n in r and (r.get(n) or "").strip():
                    try:
                        return float(r[n])
                    except ValueError:
                        return float("nan")
            return float("nan")
        fr.append(stem); la.append(lat_v); lo.append(lon_v); al.append(alt_v); acc.append(acc_v)
        ve.append(_g("DopplerVe_mps", "Ve_mps", "vE_mps"))
        vn.append(_g("DopplerVn_mps", "Vn_mps", "vN_mps"))
        vu.append(_g("DopplerVu_mps", "Vu_mps", "vU_mps"))
    if not fr:
        die(f"{path}: no rows parsed")
    return (fr, np.array(la), np.array(lo), np.array(al), np.array(acc),
            np.array(ve), np.array(vn), np.array(vu))


def read_meta(path):
    """Robust camera-model estimated-coordinate parse.

    Delegates to the repo's parse_metashape_cameras, which tolerates real-file
    quirks: name column Label / PhotoID / Camera / image; explicit
    Longitude/Latitude or the X(=lon)/Y(=lat)/Z(=alt) or X_est/Y_est/Z_est
    export order; comma / semicolon / tab delimiters; leading ``#`` comment
    headers; and a UTF-8 BOM. Returns {stem: (lat, lon, alt)} with a missing
    altitude as NaN (horizontal accuracy stays valid; vertical is flagged).
    """
    from data_pipeline.photo_compare import parse_metashape_cameras
    raw = parse_metashape_cameras(path)          # {stem: (lat, lon, h_or_None)}
    return {k: (lat, lon, (h if h is not None else float("nan")))
            for k, (lat, lon, h) in raw.items()}


def read_meta_projected(path, epsg):
    """Read a camera-model CSV whose coordinates are in a projected / non-WGS84
    CRS (EPSG code ``epsg``) and reproject to WGS84 -> {stem: (lat, lon, alt)}.

    Tolerant like read_meta (delimiter/name-column/comment/BOM), but treats the
    coordinate columns as Easting/Northing(/Height) in metres and reprojects
    with pyproj. Column order: X/Easting first, Y/Northing second (or the
    Metashape X_est/Y_est/Z_est export order).
    """
    from pyproj import Transformer
    tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)
    raw = [ln.rstrip("\r\n") for ln in open(path, encoding="utf-8-sig") if ln.strip()]
    body = [ln for ln in raw if not ln.lstrip().startswith("#")]
    header = raw[0].lstrip("#").strip() if raw and raw[0].lstrip().startswith("#") and not body else None
    lines = body if body else raw
    try:
        dialect = csv.Sniffer().sniff(lines[0], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    if header is None:
        header = lines[0]
        lines = lines[1:]
    cols = [c.strip().lstrip("#").strip() for c in next(csv.reader([header], dialect=dialect))]
    low = [c.lower() for c in cols]

    def find(cands):
        for i, c in enumerate(low):
            if c in cands:
                return i
        return None
    ni = find({"label", "photoid", "photo_id", "camera", "image", "name"})
    xi = find({"x_est", "x/easting", "x", "easting", "e"})
    yi = find({"y_est", "y/northing", "y", "northing", "n"})
    zi = find({"z_est", "z/height", "z", "height", "altitude", "elevation"})
    if ni is None or xi is None or yi is None:
        raise ValueError(f"{path}: need name + X/Easting + Y/Northing columns "
                         f"for a projected CSV (got {cols})")
    out = {}
    for row in csv.reader(lines, dialect=dialect):
        if not row or len(row) <= max(ni, xi, yi):
            continue
        name = row[ni].strip()
        if not name or name.startswith("#"):
            continue
        try:
            E, N = float(row[xi]), float(row[yi])
        except ValueError:
            continue
        lon, lat = tr.transform(E, N)
        alt = float("nan")
        if zi is not None and zi < len(row):
            try:
                alt = float(row[zi])
            except ValueError:
                pass
        out[Path(name).stem] = (lat, lon, alt)
    if not out:
        raise ValueError(f"{path}: no usable rows after reprojection")
    return out


# ----------------------------- load -----------------------------
need(A.gt, "ground-truth .pos"); need(A.track, "device track .pos")
need(A.georef, "per-frame georef CSV"); need(A.ftimes, "per-frame time CSV")
jt, jla, jlo, jh, jvn, jve, jvu = parse_pos(A.gt)
pt, pla, plo, ph, pvn, pve, pvu = parse_pos(A.track)
fr, gla, glo, gal, gacc, gdve, gdvn, gdvu = read_georef(A.georef)
tvmap = {}
_bad_ft = 0
for r in csv.DictReader(open(A.ftimes, encoding="utf-8-sig")):
    try:
        tvmap[Path(r["Image"]).stem] = float(r["t_video_s"])
    except (ValueError, KeyError, TypeError):
        _bad_ft += 1
if _bad_ft:
    warn(f"{_bad_ft} row(s) in {A.ftimes} had a missing/non-numeric t_video_s — skipped")
miss = [f for f in fr if f not in tvmap]
if miss:
    warn(f"{len(miss)}/{len(fr)} georef frames missing from frame-times; dropping them")
    keep = [i for i, f in enumerate(fr) if f in tvmap]
    fr = [fr[i] for i in keep]
    gla, glo, gal, gacc = gla[keep], glo[keep], gal[keep], gacc[keep]
    gdve, gdvn, gdvu = gdve[keep], gdvn[keep], gdvu[keep]
tvid = np.array([tvmap[f] for f in fr])
if len(fr) < 3:
    die(f"only {len(fr)} usable frames")
meta_specs = [] if A.no_meta else parse_meta_specs(A.meta)
metas = []  # [(label, {stem:(lat,lon,alt)})]
for label, path, epsg in meta_specs:
    epsg = epsg if epsg is not None else A.epsg
    if not path.is_file():
        warn(f"camera-model source '{label}' not found: {path} — skipped")
        continue
    try:
        d = read_meta_projected(path, epsg) if epsg else read_meta(path)
    except Exception as e:
        extra = ("" if epsg else
                 " (if this CSV is in a projected/UTM CRS, pass --epsg <code> "
                 "or --meta label=path@<code>)")
        warn(f"camera-model source '{label}' ({path}) could not be parsed: {e}{extra} — skipped")
        continue
    if d:
        metas.append((label, d))
    else:
        warn(f"camera-model source '{label}' ({path}) had no usable rows — skipped")
if not A.no_meta and not metas:
    warn("no camera-model sources — GPS-only report")

# ----------------------------- t0 sync -----------------------------
def cost(t0):
    q = t0 + tvid
    return ((np.interp(q, pt, plo) - glo) ** 2 + (np.interp(q, pt, pla) - gla) ** 2).mean()
grid = np.arange(pt.min() - tvid.max(), pt.max() + 1, 0.5)
t0 = min(grid, key=cost)
for step in (0.1, 0.02, 0.005):
    t0 = min(np.arange(t0 - step * 10, t0 + step * 10, step), key=cost)
frame_sod = t0 + tvid
sync_resid_m = math.sqrt(cost(t0)) * 111320.0
print(f"[sync] t0(sod)={t0:.3f} residual={sync_resid_m:.2f} m  n_frames={len(fr)}")
if sync_resid_m > SYNC_WARN_M:
    warn(f"sync residual {sync_resid_m:.1f} m > {SYNC_WARN_M} m — time alignment may be poor")

# ----------------------------- ENU -----------------------------
def sample_gt(t_query, tt, vv):
    """Sample the 10 Hz ground-truth track at frame times — cubic spline (chord-error
    free) with a linear fallback outside the data / when scipy is absent."""
    if HAVE_SCIPY:
        try:
            u, ui = np.unique(tt, return_index=True)
            from scipy.interpolate import CubicSpline
            s = CubicSpline(u, vv[ui], extrapolate=False)
            out = s(t_query)
            bad = ~np.isfinite(out)
            if bad.any():
                out[bad] = np.interp(t_query[bad], tt, vv)
            return out
        except Exception:
            pass
    return np.interp(t_query, tt, vv)
jla_f = sample_gt(frame_sod, jt, jla); jlo_f = sample_gt(frame_sod, jt, jlo); jh_f = sample_gt(frame_sod, jt, jh)
origin = (float(gla[0]), float(glo[0]), float(gal[0]))
def enu1(lat, lon, h):
    return np.array(ecef_to_enu(*llh_to_ecef(lat, lon, h), origin))
gps_enu = np.array([enu1(gla[i], glo[i], gal[i]) for i in range(len(fr))])
gt_enu = np.array([enu1(jla_f[i], jlo_f[i], jh_f[i]) for i in range(len(fr))])
gps_enu_all = gps_enu
meta_arrays = []  # [(label, enu(N,3), present_mask, garbage_bool)]
for label, d in metas:
    arr = np.full((len(fr), 3), np.nan)
    for i, f in enumerate(fr):
        if f in d:
            lat, lon, alt = d[f]
            no_alt = not math.isfinite(alt)
            if no_alt:                      # no altitude in the export
                alt = origin[2]             # placeholder so lat/lon still convert to ENU
            arr[i] = enu1(lat, lon, alt)
            if no_alt:
                arr[i, 2] = np.nan          # vertical genuinely unknown — must not silently
                                             # masquerade as a real (bogus) Up error downstream
    pres = np.isfinite(arr[:, 0])
    garbage = bool(pres.any() and np.median(np.linalg.norm(arr[pres][:, :2], axis=1)) > 1000.0)
    meta_arrays.append((label, arr, pres, garbage))

# ----------------------------- GT heading/speed -----------------------------
def series_speed(pos, t):
    s = np.full(len(t), np.nan)
    if len(t) < 2:
        return s
    dt = np.diff(t); dv = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s[1:] = np.where((dt > 0) & (dt <= MAX_STEP), dv / dt, np.nan)
    return s
gt_sp = series_speed(gt_enu, frame_sod)
that = np.full((len(fr), 2), np.nan); gt_head = np.full(len(fr), np.nan)
dp = np.diff(gt_enu[:, :2], axis=0); mag = np.linalg.norm(dp, axis=1)
for i in range(1, len(fr)):
    if mag[i - 1] > 1e-6:
        gt_head[i] = heading_from_enu(dp[i - 1, 0], dp[i - 1, 1]); that[i] = dp[i - 1] / mag[i - 1]
gt_cumdist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt_enu, axis=0), axis=1))])

# ----------------------------- velocity series -----------------------------
# Four per-frame speed sources for the velocity panel:
#  - GT Doppler  : |native velocity| from the GT .pos, sampled at frame times
#  - GT coord-Δ  : |Δposition|/Δt of the GT track            (= gt_sp)
#  - phone Doppler: |DopplerVe/Vn/Vu| from the georef        (per-frame)
#  - phone coord-Δ: |Δposition|/Δt of the device georef track
def _speed3(vn, ve, vu):
    # All three components must be finite -- vu was missing from this gate,
    # so a row with a real vn/ve but no vu (common: 2D-only Doppler) silently
    # came out as a *2D* speed (vu treated as 0 m/s) that looked like a valid
    # 3D value with no signal that the vertical term was missing.
    return np.sqrt(np.nan_to_num(vn) ** 2 + np.nan_to_num(ve) ** 2 + np.nan_to_num(vu) ** 2) \
        * np.where(np.isfinite(vn) & np.isfinite(ve) & np.isfinite(vu), 1.0, np.nan)
gt_dop_speed = _speed3(sample_gt(frame_sod, jt, jvn), sample_gt(frame_sod, jt, jve),
                       sample_gt(frame_sod, jt, jvu)) if np.isfinite(jve).any() else np.full(len(fr), np.nan)
gt_cd_speed = gt_sp
ph_dop_speed = _speed3(gdvn, gdve, gdvu) if np.isfinite(gdve).any() else np.full(len(fr), np.nan)
ph_cd_speed = series_speed(gps_enu, frame_sod)
# Speed error as % of current speed, gated to >= 10 km/h (2.778 m/s).
KMH10 = 10.0 / 3.6
_gt_ref = np.where(np.isfinite(gt_dop_speed), gt_dop_speed, gt_cd_speed)
_ph_ref = np.where(np.isfinite(ph_dop_speed), ph_dop_speed, ph_cd_speed)
_fast = _gt_ref >= KMH10
speed_pct_err = np.where(_fast & np.isfinite(_ph_ref),
                         100.0 * np.abs(_ph_ref - _gt_ref) / _gt_ref, np.nan)

# ----------------------------- smoothed device track (epoch_weight_v2) ------
smoothed_enu = None
try:
    from data_pipeline.parsers import parse_rtkpos, parse_pos_header
    from data_pipeline.time_sync import get_leap_seconds_for_datetime
    from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
    from data_pipeline.geo import enu_to_llh
    import datetime as _dt
    dpos = parse_rtkpos(A.track)
    if len(dpos) < 5:
        warn(f"device --track has only {len(dpos)} usable epoch(s) (<5) — "
             "skipping 'device GPS (smoothed v2)'")
    elif math.floor(dpos[0].utc_s / 86400.0) != math.floor(dpos[-1].utc_s / 86400.0):
        # sm_sod below is a %86400 seconds-of-day value like frame_sod; a
        # capture straddling UTC midnight would wrap and misorder samples.
        warn("device --track spans a UTC midnight boundary — seconds-of-day would wrap; "
             "skipping 'device GPS (smoothed v2)'")
    else:
        # parse_rtkpos normalises PosRow.utc_s to TRUE UTC (it subtracts leap
        # seconds off a GPST-labelled .pos). frame_sod/jt/pt above are each
        # source .pos's OWN raw, uncorrected clock (parse_pos does no leap
        # correction) -- GPST-SOD here, since this .pos is GPST-labelled, same
        # as the GT .pos, so the main "device GPS"/"GT" comparison is
        # internally consistent. Using utc_s%86400 directly against
        # frame_sod (as before) silently shifted every smoothed sample by the
        # leap-second offset (~18 s => ~100 m at driving speed -- this was the
        # entire ~157 m smoothed-vs-~3.4 m raw discrepancy). Convert utc_s
        # back to the SAME raw clock frame_sod uses before interpolating.
        _ts = parse_pos_header(A.track).time_system
        _ls = 0.0 if _ts == "UTC" else get_leap_seconds_for_datetime(
            _dt.datetime.fromtimestamp(dpos[0].utc_s, tz=_dt.timezone.utc))
        res = smooth_epoch_weighted_v2(dpos, None)          # GNSS-only (no IMU)
        ref = (dpos[0].lat_deg, dpos[0].lon_deg, dpos[0].h_m)
        sm_sod = np.array([(p.utc_s + _ls) % 86400.0 for p in dpos])
        sm_ll = np.array([enu_to_llh(float(res.E_smooth[i]), float(res.N_smooth[i]),
                                     float(res.U_smooth[i]), ref) for i in range(len(dpos))])
        o = np.argsort(sm_sod)
        sm_sod, sm_ll = sm_sod[o], sm_ll[o]
        sla = np.interp(frame_sod, sm_sod, sm_ll[:, 0])
        slo = np.interp(frame_sod, sm_sod, sm_ll[:, 1])
        shh = np.interp(frame_sod, sm_sod, sm_ll[:, 2])
        smoothed_enu = np.array([enu1(sla[i], slo[i], shh[i]) for i in range(len(fr))])
        print(f"[smooth] epoch_weight_v2 applied to device track "
              f"({len(dpos)} epochs, {res.n_zupt_updates} ZUPT, {res.n_nhc_updates} NHC, "
              f"leap={_ls:.0f}s)")
except Exception as e:
    warn(f"epoch_weight_v2 smoothing skipped: {e}")

# ----------------------------- stat helpers -----------------------------
def fin(a):
    a = np.asarray(a, float); return a[np.isfinite(a)]

def pctl(a, p):
    a = fin(a); return float(np.percentile(a, p)) if len(a) else float("nan")

def robust(a):
    a = fin(a)
    if not len(a):
        return dict(median=math.nan, mad=math.nan, rsigma=math.nan, trmrms=math.nan, out_pct=math.nan)
    med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
    rs = 1.4826 * mad
    lo, hi = np.percentile(a, [5, 95])
    tr = a[(a >= lo) & (a <= hi)]
    trmrms = float(math.sqrt(np.mean(tr ** 2))) if len(tr) else math.nan
    out = float(100.0 * np.mean(np.abs(a - med) > 3 * rs)) if rs > 0 else 0.0
    return dict(median=med, mad=mad, rsigma=rs, trmrms=trmrms, out_pct=out)

def shape(a):
    a = fin(a)
    if len(a) < 8:
        return dict(skew=math.nan, kurt=math.nan, ad=math.nan, normal=None)
    if HAVE_SCIPY:
        sk, ku = float(_st.skew(a)), float(_st.kurtosis(a))
        try:
            r = _st.anderson(a, "norm"); ad = float(r.statistic)
            crit = r.critical_values[2]  # 5%
            normal = bool(ad < crit)
        except Exception:
            ad, normal = math.nan, None
    else:
        m, s = a.mean(), a.std()
        sk = float(np.mean(((a - m) / s) ** 3)) if s > 0 else math.nan
        ku = float(np.mean(((a - m) / s) ** 4) - 3) if s > 0 else math.nan
        ad, normal = math.nan, None
    return dict(skew=sk, kurt=ku, ad=ad, normal=normal)

def boot_ci(a, func):
    a = fin(a)
    if len(a) < 8:
        return (math.nan, math.nan)
    if HAVE_SCIPY:
        try:
            r = _bootstrap((a,), func, n_resamples=BOOT_N, confidence_level=0.95,
                           method="percentile", random_state=RNG, vectorized=False)
            return (float(r.confidence_interval.low), float(r.confidence_interval.high))
        except Exception:
            pass
    idx = RNG.integers(0, len(a), size=(BOOT_N, len(a)))
    vals = np.array([func(a[i]) for i in idx])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def block_boot_ci(a, func, block):
    """Moving-block bootstrap CI — honours serial correlation (IID resampling
    would give too-narrow intervals when errors are autocorrelated)."""
    a = fin(a); n = len(a)
    if n < 8:
        return (math.nan, math.nan)
    block = int(max(1, min(block, n)))
    nblk = int(math.ceil(n / block)); smax = n - block
    vals = np.empty(BOOT_N)
    for b in range(BOOT_N):
        starts = RNG.integers(0, smax + 1, size=nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        vals[b] = func(a[idx])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def robust_bands(a):
    """sigma_bands after dropping |x-median|>5·(1.4826·MAD) outliers."""
    a = fin(a)
    if len(a) < 4:
        return sigma_bands(a), 0
    med = np.median(a); rs = 1.4826 * np.median(np.abs(a - med))
    mask = np.abs(a - med) <= 5 * rs if rs > 0 else np.ones(len(a), bool)
    return sigma_bands(a[mask]), int((~mask).sum())


def autocorr1(a):
    a = np.asarray(a, float); m = np.isfinite(a)
    a = a[m]
    if len(a) < 4:
        return math.nan
    a = a - a.mean()
    denom = np.sum(a * a)
    return float(np.sum(a[:-1] * a[1:]) / denom) if denom > 0 else math.nan

def acf(a, L):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 4:
        return []
    a = a - a.mean(); d = np.sum(a * a)
    return [1.0] + [float(np.sum(a[:-k] * a[k:]) / d) if d > 0 else 0.0 for k in range(1, L + 1)]

def circ_stats(deg):
    d = np.radians(fin(deg))
    if not len(d):
        return dict(cmean=math.nan, cstd=math.nan, R=math.nan, rayleigh_p=math.nan)
    C, S = np.mean(np.cos(d)), np.mean(np.sin(d)); R = math.hypot(C, S)
    cmean = math.degrees(math.atan2(S, C))
    cstd = math.degrees(math.sqrt(-2 * math.log(R))) if R > 1e-12 else math.nan
    n = len(d); Z = n * R * R
    p = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n)) if n else math.nan
    return dict(cmean=cmean, cstd=cstd, R=R, rayleigh_p=max(min(p, 1.0), 0.0))

def ellipse_geom(E, N):
    E, N = fin(E), fin(N)
    if len(E) < 3:
        return dict(a=math.nan, b=math.nan, theta=math.nan, ecc=math.nan, area95=math.nan)
    cov = np.cov(np.vstack([E, N]))
    w, V = np.linalg.eigh(cov); w = np.clip(w, 0, None)
    b, a = math.sqrt(w[0]), math.sqrt(w[1])
    vmaj = V[:, 1]; theta = math.degrees(math.atan2(vmaj[1], vmaj[0]))
    ecc = math.sqrt(1 - (b * b) / (a * a)) if a > 0 else math.nan
    k95 = math.sqrt(5.991)  # chi2, 2 dof, 95%
    return dict(a=a, b=b, theta=theta, ecc=ecc, area95=math.pi * k95 * k95 * a * b)

def ellipse_xy(E, N, k):
    E, N = fin(E), fin(N)
    if len(E) < 3:
        return [], []
    mE, mN = E.mean(), N.mean(); cov = np.cov(np.vstack([E, N])); w, V = np.linalg.eigh(cov)
    th = np.linspace(0, 2 * math.pi, 80)
    xy = V[:, 1:2] * (k * math.sqrt(max(w[1], 0)) * np.cos(th)) + V[:, 0:1] * (k * math.sqrt(max(w[0], 0)) * np.sin(th))
    return (mE + xy[0]).tolist(), (mN + xy[1]).tolist()

def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-9:
        return math.nan, math.nan
    slope = np.polyfit(x, y, 1)[0]
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), r

def xcorr_lag(a, b, maxlag=20):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < maxlag + 3:
        return math.nan
    a = a - a.mean(); b = b - b.mean()
    best, blag = -2, 0
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            x, y = a[:lag], b[-lag:]
        elif lag > 0:
            x, y = a[lag:], b[:-lag]
        else:
            x, y = a, b
        if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, blag = c, lag
    return blag


# ----------------------------- per-source compute -----------------------------
def compute(name, idx, src_enu, nominal_acc=None):
    idx = np.asarray(idx)
    d = src_enu[idx] - gt_enu[idx]
    E, N, U = d[:, 0], d[:, 1], d[:, 2]
    horiz = np.hypot(E, N); vert = np.abs(U); three = np.linalg.norm(d, axis=1)
    th = that[idx]; ok = np.isfinite(th[:, 0])
    along = np.where(ok, E * th[:, 0] + N * th[:, 1], np.nan)
    cross = np.where(ok, E * (-th[:, 1]) + N * th[:, 0], np.nan)
    ssp = series_speed(src_enu[idx], frame_sod[idx]); gsp = gt_sp[idx]
    sp_err = ssp - gsp; mv = gsp >= MOVING_MPS      # motion stats gated to >= 10 km/h
    sp_pct = np.where(mv & np.isfinite(sp_err), 100.0 * np.abs(sp_err) / gsp, np.nan)
    saz = np.full(len(idx), np.nan); dps = np.diff(src_enu[idx][:, :2], axis=0); ms = np.linalg.norm(dps, axis=1)
    for j in range(1, len(idx)):
        if ms[j - 1] > 1e-6:
            saz[j] = heading_from_enu(dps[j - 1, 0], dps[j - 1, 1])
    az_err = _wrap(saz - gt_head[idx]); az_err = np.where(mv & np.isfinite(az_err), az_err, np.nan)
    rmse = lambda a: float(math.sqrt(np.mean(np.square(fin(a))))) if len(fin(a)) else math.nan
    corr = {}
    for k, ax in enumerate("ENU"):
        s_, g_ = src_enu[idx][:, k], gt_enu[idx][:, k]
        corr[ax] = float(np.corrcoef(s_, g_)[0, 1]) if np.std(s_) > 0 and np.std(g_) > 0 else math.nan
    r1 = autocorr1(horiz); n = len(idx)
    neff = n * (1 - r1) / (1 + r1) if math.isfinite(r1) and r1 < 1 else float(n)
    blk = int(max(1, round((1 + r1) / (1 - r1)))) if math.isfinite(r1) and r1 < 1 else 1
    blk = min(blk, max(1, n // 4))
    # Aligned (7-param Umeyama fit src->GT): removes datum/control offset, scale
    # and rotation, leaving pure photogrammetric shape + noise. absolute - aligned
    # = the systematic (datum/control) component. camera-model used the device GPS as
    # control, so this separates "reproduced the biased GPS" from "true shape".
    aligned = None
    # Rows lacking altitude carry a NaN Z (see meta_arrays above) — feeding even one
    # into the 3D Umeyama SVD poisons scale/rotation/translation for every row (numpy's
    # SVD raises on NaN), silently wiping out the aligned block for the whole source.
    # Fit only on rows with a fully finite 3D point; still evaluate the resulting
    # transform against every row in idx (rows w/o altitude just come out NaN there,
    # dropped by sigma_bands as usual).
    fit_mask = np.isfinite(src_enu[idx][:, 2])
    if fit_mask.sum() >= 3:
        try:
            fit = fit_similarity_robust([str(i) for i in idx[fit_mask]],
                                         src_enu[idx][fit_mask], gt_enu[idx][fit_mask])
            da = fit.apply(src_enu[idx]) - gt_enu[idx]
            ah = np.hypot(da[:, 0], da[:, 1]); a3 = np.linalg.norm(da, axis=1)
            aligned = dict(horiz=sigma_bands(ah), err3d=sigma_bands(a3),
                           rmse_h=float(math.sqrt(np.nanmean(ah ** 2))) if np.isfinite(ah).any() else math.nan,
                           scale=float(fit.scale), scale_err_pct=100.0 * (fit.scale - 1.0),
                           rms_m=float(fit.rms_m), n_used=int(fit.n_used))
        except Exception as ex:
            print(f"WARN: {name} aligned fit failed: {ex}")
    sp_rob, sp_dropped = robust_bands(sp_err)
    drift, drift_r = linfit(gt_cumdist[idx], horiz)      # m error per m travelled
    espeed_slope, espeed_r = linfit(gsp, horiz)
    lag = xcorr_lag(ssp, gsp)
    if nominal_acc is not None:
        _nm = np.isfinite(nominal_acc[idx]) & np.isfinite(horiz)
        nom_cov = float(100.0 * np.mean(horiz[_nm] <= nominal_acc[idx][_nm])) if _nm.any() else math.nan
    else:
        nom_cov = math.nan
    return dict(
        name=name, n=int(n), coverage=100.0 * n / len(fr),
        dur=float(frame_sod[idx].max() - frame_sod[idx].min()),
        pathlen_src=float(np.nansum(np.linalg.norm(np.diff(src_enu[idx], axis=0), axis=1))),
        pathlen_gt=float(gt_cumdist[idx][-1] - gt_cumdist[idx][0]),
        horiz=sigma_bands(horiz), vert=sigma_bands(vert), err3d=sigma_bands(three),
        east=sigma_bands(E), north=sigma_bands(N), up=sigma_bands(U),
        along=sigma_bands(fin(along)), cross=sigma_bands(fin(cross)),
        speed=sigma_bands(fin(np.where(mv, sp_err, np.nan))),
        speed_pct=sigma_bands(fin(sp_pct)), azimuth=sigma_bands(fin(az_err)),
        rob_h=robust(horiz), rob_v=robust(vert), rob_3d=robust(three),
        shp_h=shape(horiz), shp_e=shape(E), shp_u=shape(U),
        ci_rmse_h=block_boot_ci(horiz, lambda x: math.sqrt(np.mean(x ** 2)), blk),
        ci_cep95=block_boot_ci(horiz, lambda x: np.percentile(x, 95), blk),
        ci_s2=block_boot_ci(horiz, lambda x: np.percentile(x, 95.45), blk),
        ci_iid=boot_ci(horiz, lambda x: math.sqrt(np.mean(x ** 2))),
        r1=r1, neff=float(neff), blk=blk, aligned=aligned,
        sp_rob=sp_rob, sp_dropped=sp_dropped,
        circ=circ_stats(az_err), ell=ellipse_geom(E, N),
        scale_err=100.0 * (float(np.nansum(np.linalg.norm(np.diff(src_enu[idx], axis=0), axis=1))) /
                           max(gt_cumdist[idx][-1] - gt_cumdist[idx][0], 1e-9) - 1.0),
        drift=drift, drift_r=drift_r, espeed_slope=espeed_slope, espeed_r=espeed_r, lag=lag,
        nom_cov=nom_cov, nominal=float(np.nanmedian(nominal_acc[idx])) if nominal_acc is not None else math.nan,
        rmse_h=rmse(horiz), rmse_3d=rmse(three), rmse_e=rmse(E), rmse_n=rmse(N), rmse_u=rmse(U),
        bias_e=float(np.nanmean(E)), bias_n=float(np.nanmean(N)), bias_u=float(np.nanmean(U)),
        cep={p: pctl(horiz, p) for p in (50, 90, 95, 99)}, r={p: pctl(three, p) for p in (95, 99)},
        pct_h={p: pctl(horiz, p) for p in PCTS}, pct_v={p: pctl(vert, p) for p in PCTS},
        pct_3d={p: pctl(three, p) for p in PCTS}, pct_along={p: pctl(np.abs(along), p) for p in PCTS},
        pct_cross={p: pctl(np.abs(cross), p) for p in PCTS},
        drms=math.sqrt(np.mean(E ** 2) + np.mean(N ** 2)), mrse=math.sqrt(np.mean(three ** 2)),
        corr=corr, mean_sp_gt=float(np.nanmean(gsp)), mean_sp_src=float(np.nanmean(ssp)),
        sp_corr=float(np.corrcoef(fin(ssp), gsp[np.isfinite(ssp)])[0, 1]) if np.isfinite(ssp).sum() > 2 else math.nan,
        cdf={t: float(100.0 * np.mean(horiz <= t)) for t in THRESHOLDS},
        acf=acf(horiz, 30),
        _E=E.tolist(), _N=N.tolist(), _horiz=horiz, _vert=vert.tolist(),
        _sp_err=sp_err, _t=frame_sod[idx], _gsp=gsp, _idx=idx,
    )


sources = {}
ENU_OF = {}   # source name -> full (N,3) ENU array (for chart traces)
garbage_labels = []
try:
    sources["device GPS"] = compute("device GPS", np.arange(len(fr)), gps_enu_all, nominal_acc=gacc)
    ENU_OF["device GPS"] = gps_enu_all
except Exception as e:
    warn(f"device GPS compute failed: {e}")
if smoothed_enu is not None:
    try:
        sources["device GPS (smoothed v2)"] = compute(
            "device GPS (smoothed v2)", np.arange(len(fr)), smoothed_enu)
        ENU_OF["device GPS (smoothed v2)"] = smoothed_enu
    except Exception as e:
        warn(f"smoothed-source compute failed: {e}")
for label, arr, pres, garbage in meta_arrays:
    if garbage:
        garbage_labels.append(label)
        warn(f"camera-model source '{label}' degenerate (median |ENU| huge) — excluded")
        continue
    if pres.sum() < 3:
        warn(f"camera-model source '{label}' has {int(pres.sum())} usable frames — skipped")
        continue
    try:
        sources[label] = compute(label, np.where(pres)[0], arr)
        ENU_OF[label] = arr
    except Exception as e:
        warn(f"camera-model source '{label}' compute failed: {e}")
if not sources:
    die("no source produced statistics")

for name, s in sources.items():
    al = s["aligned"]
    # The trailing "if al else ''" used to gate the WHOLE concatenated string
    # (n=, cov=, H2σ=, RMSE_h=, CI, robσ, r1, Neff -- everything), because
    # adjacent string literals join into one expression before the ternary is
    # applied. A source whose aligned (Umeyama) fit failed printed a blank
    # line instead of its always-available diagnostics. Gate only the
    # aligned-only tail.
    print(f"[{name}] n={s['n']} cov={s['coverage']:.0f}% H2σ={s['horiz']['sigma2']:.3f} "
          f"RMSE_h={s['rmse_h']:.3f} blockCI[{s['ci_rmse_h'][0]:.2f},{s['ci_rmse_h'][1]:.2f}] "
          f"iidCI[{s['ci_iid'][0]:.2f},{s['ci_iid'][1]:.2f}] blk={s['blk']} "
          f"robσ={s['rob_h']['rsigma']:.3f} r1={s['r1']:.2f} Neff={s['neff']:.0f}"
          + (f" | ABS RMSE_h={s['rmse_h']:.2f} vs ALIGNED RMSE_h={al['rmse_h']:.2f} "
             f"(fit scale={al['scale']:.4f}, {al['scale_err_pct']:+.2f}%)" if al else ""))
    print(f"       drift={s['drift']*100:.3f}m/100m lag={s['lag']} spd_rob2σ={s['sp_rob']['sigma2']:.3f} "
          f"(dropped {s['sp_dropped']}) nom_cov={s['nom_cov']:.0f}%")
for gl in garbage_labels:
    print(f"[{gl}] GARBAGE (degenerate reconstruction) -> excluded")

# vertical datum sanity: GPS up-bias near a geoid undulation => likely
# ellipsoidal-vs-orthometric mismatch (Israel geoid separation ~17 m).
datum_note = ""
if "device GPS" in sources:
    ub = abs(sources["device GPS"]["bias_u"])
    if 12.0 <= ub <= 25.0:
        datum_note = (f"vertical bias {sources['device GPS']['bias_u']:+.1f} m is near the local geoid "
                      "undulation — likely an ellipsoidal-vs-orthometric height mismatch; treat V/3D with care")
        warn(datum_note)
    else:
        print(f"[datum] GPS vertical bias {sources['device GPS']['bias_u']:+.2f} m — "
              "consistent with a shared (ellipsoidal) height datum")

# ----------------------------- HTML -----------------------------
def _load_plotly():
    if A.plotly and Path(A.plotly).is_file():
        return Path(A.plotly).read_text(encoding="utf-8")
    try:
        from data_pipeline.analysis_report import _load_plotly_js
        js = _load_plotly_js()
        if js:
            return js
    except Exception:
        pass
    for c in [A.repo / "data_pipeline" / "assets" / "plotly.min.js",
              A.georef.parent / "plotly.min.js"]:
        if Path(c).is_file():
            return Path(c).read_text(encoding="utf-8")
    warn("plotly.min.js not found — charts will be omitted (pass --plotly)")
    return ""
plotly = _load_plotly()
SESSION = A.georef.parent.name
_PALETTE = ["#2b8a3e", "#1c6fd6", "#8e44ad", "#d64500", "#0b8a8a", "#b5179e", "#5f7d00"]
COLOR = {"device GPS": "#e8720c"}
for _i, _name in enumerate(n for n in sources if n != "device GPS"):
    COLOR[_name] = _PALETTE[_i % len(_PALETTE)]
fmt = lambda v, d=3: ("—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}")
def yn(b):
    return "—" if b is None else ("normal" if b else "non-normal")


def T(head, rows):
    return f"<table><tr>{head}</tr>{rows}</table>"


def traj_tbl():
    h = ("<th>source</th><th>frames</th><th>cov%</th><th>dur(s)</th><th>GT path(m)</th>"
         "<th>src path(m)</th><th>scale err%</th><th>mean GT v</th><th>mean src v</th>")
    r = "".join("<tr><td>{}</td><td>{}</td><td>{:.0f}</td><td>{:.1f}</td><td>{:.1f}</td><td>{:.1f}</td>"
                "<td>{:+.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>".format(
        s["name"], s["n"], s["coverage"], s["dur"], s["pathlen_gt"], s["pathlen_src"],
        s["scale_err"], s["mean_sp_gt"], s["mean_sp_src"]) for s in sources.values())
    return T(h, r)


def position_tbl():
    h = ("<th>source</th><th>H1σ</th><th>H2σ</th><th>H3σ</th><th>H RMS</th><th>Hmax</th><th>V2σ</th>"
         "<th>V RMS</th><th>3D2σ</th><th>3Dmax</th><th>CEP50</th><th>CEP90</th><th>CEP95</th><th>CEP99</th>"
         "<th>R95</th><th>DRMS</th><th>2DRMS</th><th>MRSE</th><th>bias E/N/U</th>")
    r = ""
    for s in sources.values():
        H, V, E = s["horiz"], s["vert"], s["err3d"]
        r += ("<tr><td>{}</td>" + "<td>{:.3f}</td>" * 17 + "<td>{:+.2f}/{:+.2f}/{:+.2f}</td></tr>").format(
            s["name"], H["sigma1"], H["sigma2"], H["sigma3"], s["rmse_h"], H["max_abs"], V["sigma2"], s["rmse_u"],
            E["sigma2"], E["max_abs"], s["cep"][50], s["cep"][90], s["cep"][95], s["cep"][99], s["r"][95],
            s["drms"], 2 * s["drms"], s["mrse"], s["bias_e"], s["bias_n"], s["bias_u"])
    return T(h, r)


def robust_tbl():
    h = ("<th>source · metric</th><th>median</th><th>MAD</th><th>robust σ</th><th>trimmed RMS</th>"
         "<th>outlier%</th><th>skew</th><th>excess kurt</th><th>Anderson-Darling</th><th>normal?</th>")
    r = ""
    for s in sources.values():
        for lbl, rob, shp in (("horiz", s["rob_h"], s["shp_h"]), ("3D", s["rob_3d"], None),
                              ("vert", s["rob_v"], s["shp_u"])):
            sk = fmt(shp["skew"], 2) if shp else "—"
            ku = fmt(shp["kurt"], 2) if shp else "—"
            ad = fmt(shp["ad"], 2) if shp else "—"
            nm = yn(shp["normal"]) if shp else "—"
            r += ("<tr><td>{} · {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                  "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>").format(
                s["name"], lbl, fmt(rob["median"]), fmt(rob["mad"]), fmt(rob["rsigma"]),
                fmt(rob["trmrms"]), fmt(rob["out_pct"], 1), sk, ku, ad, nm)
    return T(h, r)


def infer_tbl():
    h = ("<th>source</th><th>RMSE H</th><th>RMSE 95% CI (block)</th><th>RMSE CI (IID, too tight)</th>"
         "<th>CEP95</th><th>CEP95 CI</th><th>H 2σ</th><th>2σ CI</th>"
         "<th>lag-1 ρ</th><th>block len</th><th>N</th><th>eff. N</th>")
    r = ""
    for s in sources.values():
        r += ("<tr><td>{}</td><td>{:.3f}</td><td>[{:.2f}, {:.2f}]</td><td>[{:.2f}, {:.2f}]</td>"
              "<td>{:.3f}</td><td>[{:.2f}, {:.2f}]</td><td>{:.3f}</td><td>[{:.2f}, {:.2f}]</td>"
              "<td>{:.3f}</td><td>{}</td><td>{}</td><td>{:.0f}</td></tr>").format(
            s["name"], s["rmse_h"], s["ci_rmse_h"][0], s["ci_rmse_h"][1], s["ci_iid"][0], s["ci_iid"][1],
            s["cep"][95], s["ci_cep95"][0], s["ci_cep95"][1], s["horiz"]["sigma2"], s["ci_s2"][0], s["ci_s2"][1],
            s["r1"], s["blk"], s["n"], s["neff"])
    return T(h, r)


def aligned_tbl():
    h = ("<th>source</th><th>ABS H 2σ</th><th>ALN H 2σ</th><th>ABS RMSE H</th><th>ALN RMSE H</th>"
         "<th>ABS 3D 2σ</th><th>ALN 3D 2σ</th><th>fit scale</th><th>scale err %</th>"
         "<th>fit RMS (m)</th><th>systematic = ABS−ALN RMSE</th>")
    r = ""
    for s in sources.values():
        al = s["aligned"]
        if not al:
            r += f"<tr><td>{s['name']}</td><td colspan=10>fit unavailable</td></tr>"
            continue
        r += ("<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>"
              "<td>{:.3f}</td><td>{:.3f}</td><td>{:.4f}</td><td>{:+.2f}</td><td>{:.3f}</td><td>{:.3f}</td></tr>").format(
            s["name"], s["horiz"]["sigma2"], al["horiz"]["sigma2"], s["rmse_h"], al["rmse_h"],
            s["err3d"]["sigma2"], al["err3d"]["sigma2"], al["scale"], al["scale_err_pct"], al["rms_m"],
            max(s["rmse_h"] - al["rmse_h"], 0.0))
    return T(h, r)


def pct_tbl():
    h = "<th>source · metric</th>" + "".join(f"<th>P{p}</th>" for p in PCTS)
    r = ""
    for s in sources.values():
        for lbl, key in (("horiz", "pct_h"), ("3D", "pct_3d"), ("vert", "pct_v"),
                         ("|along|", "pct_along"), ("|cross|", "pct_cross")):
            r += f"<tr><td>{s['name']} · {lbl}</td>" + "".join(f"<td>{fmt(s[key][p])}</td>" for p in PCTS) + "</tr>"
    return T(h, r)


def motion_tbl():
    h = ("<th>source</th><th>E2σ</th><th>N2σ</th><th>U2σ</th><th>along2σ</th><th>cross2σ</th>"
         "<th>speed2σ</th><th>speed3σ</th><th>speed%2σ</th><th>speed corr</th><th>speed lag</th>"
         "<th>corr E/N/U</th>")
    r = ""
    for s in sources.values():
        r += ("<tr><td>{}</td>" + "<td>{:.3f}</td>" * 5 + "<td>{:.3f}</td><td>{:.3f}</td><td>{:.1f}</td>"
              "<td>{:.3f}</td><td>{}</td><td>{:.3f}/{:.3f}/{:.3f}</td></tr>").format(
            s["name"], s["east"]["sigma2"], s["north"]["sigma2"], s["up"]["sigma2"], s["along"]["sigma2"],
            s["cross"]["sigma2"], s["speed"]["sigma2"], s["speed"]["sigma3"], s["speed_pct"]["sigma2"],
            s["sp_corr"], s["lag"], s["corr"]["E"], s["corr"]["N"], s["corr"]["U"])
    return T(h, r)


def geom_tbl():
    h = ("<th>source</th><th>ellipse a (m)</th><th>ellipse b (m)</th><th>orient (°)</th><th>eccentricity</th>"
         "<th>95% area (m²)</th><th>drift (m/100m)</th><th>drift r</th><th>err–speed slope</th><th>slope r</th>"
         "<th>nominal (m)</th><th>within nominal</th>")
    r = ""
    for s in sources.values():
        e = s["ell"]
        r += ("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
              "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>").format(
            s["name"], fmt(e["a"]), fmt(e["b"]), fmt(e["theta"], 1), fmt(e["ecc"]), fmt(e["area95"], 1),
            fmt(s["drift"] * 100), fmt(s["drift_r"], 2), fmt(s["espeed_slope"]), fmt(s["espeed_r"], 2),
            fmt(s["nominal"], 1), (fmt(s["nom_cov"], 0) + "%") if math.isfinite(s["nom_cov"]) else "—")
    return T(h, r)


def circ_tbl():
    h = ("<th>source</th><th>azimuth 2σ (°)</th><th>azimuth max (°)</th><th>circ mean (°)</th>"
         "<th>circ std (°)</th><th>resultant R</th><th>Rayleigh p</th><th>directional?</th>")
    r = ""
    for s in sources.values():
        c = s["circ"]
        r += ("<tr><td>{}</td><td>{:.2f}</td><td>{:.1f}</td><td>{:.2f}</td><td>{:.2f}</td>"
              "<td>{:.3f}</td><td>{:.1e}</td><td>{}</td></tr>").format(
            s["name"], s["azimuth"]["sigma2"], s["azimuth"]["max_abs"], c["cmean"], c["cstd"], c["R"],
            c["rayleigh_p"], "yes (biased)" if c["rayleigh_p"] < 0.05 else "no (uniform)")
    return T(h, r)


def cdf_tbl():
    h = "<th>source</th>" + "".join(f"<th>H&lt;{t}m</th>" for t in THRESHOLDS)
    r = "".join(f"<tr><td>{s['name']}</td>" + "".join(f"<td>{s['cdf'][t]:.0f}%</td>" for t in THRESHOLDS) + "</tr>"
                for s in sources.values())
    return T(h, r)


# charts
map_tr = [dict(x=gt_enu[:, 0].tolist(), y=gt_enu[:, 1].tolist(), mode="lines",
               name="ground truth", line=dict(color="#111", width=3))]
speed_tr = cdf_tr = None
speed_tr, cdf_tr, hist_tr, ell_tr, etime_tr, espeed_tr, acf_tr, qq_tr = [], [], [], [], [], [], [], []
for name, s in sources.items():
    idx = s["_idx"]; pe = ENU_OF[name][idx]; col = COLOR[name]
    map_tr.append(dict(x=pe[:, 0].tolist(), y=pe[:, 1].tolist(), mode="lines",
                       name=f"{name} (2σ={s['horiz']['sigma2']:.1f} m)", line=dict(color=col, width=1.5)))
    speed_tr.append(dict(x=s["_t"].tolist(), y=np.where(np.isfinite(s["_sp_err"]), s["_sp_err"], None).tolist(),
                         mode="lines", name=name, line=dict(color=col)))
    hs = np.sort(s["_horiz"])
    cdf_tr.append(dict(x=hs.tolist(), y=(100.0 * np.arange(1, len(hs) + 1) / len(hs)).tolist(),
                       mode="lines", name=name, line=dict(color=col)))
    hist_tr.append(dict(x=s["_horiz"].tolist(), type="histogram", name=name, opacity=0.55,
                        marker=dict(color=col), nbinsx=40))
    ell_tr.append(dict(x=s["_E"], y=s["_N"], mode="markers", name=f"{name} err",
                       marker=dict(color=col, size=3, opacity=0.35)))
    for k in (1, 2, 3):
        ex, ey = ellipse_xy(np.array(s["_E"]), np.array(s["_N"]), k)
        ell_tr.append(dict(x=ex, y=ey, mode="lines", name=f"{name} {k}σ",
                           line=dict(color=col, width=1, dash="dot"), showlegend=(k == 2)))
    etime_tr.append(dict(x=s["_t"].tolist(), y=s["_horiz"].tolist(), mode="lines", name=name, line=dict(color=col)))
    fmask = np.isfinite(s["_gsp"])
    espeed_tr.append(dict(x=s["_gsp"][fmask].tolist(), y=s["_horiz"][fmask].tolist(), mode="markers",
                          name=name, marker=dict(color=col, size=4, opacity=0.4)))
    if s["acf"]:
        acf_tr.append(dict(x=list(range(len(s["acf"]))), y=s["acf"], type="bar", name=name, marker=dict(color=col), opacity=0.6))
    if HAVE_SCIPY and len(fin(s["_E"])) > 8:
        (osm, osr), _ = _st.probplot(fin(s["_E"]), dist="norm")
        qq_tr.append(dict(x=osm.tolist(), y=osr.tolist(), mode="markers", name=name,
                          marker=dict(color=col, size=4, opacity=0.5)))

# --- velocity charts (data computed above) ---
def _line(y):
    return np.where(np.isfinite(np.asarray(y, float)), y, None).tolist()
_paint = np.where(np.isfinite(ph_dop_speed), ph_dop_speed, ph_cd_speed)
# nan_to_num(NaN)->0.0 used to paint frames with NO speed data (both Doppler
# and coord-Δ missing/filtered) the exact same color as a genuinely-stationary
# (0 m/s) frame -- indistinguishable on the Viridis scale, i.e. missing data
# silently read as "confirmed slow". Use the same NaN->None idiom as the
# other velocity charts below so those frames render with no color (gap),
# not a false 0 m/s.
vmap_tr = [dict(x=gps_enu[:, 0].tolist(), y=gps_enu[:, 1].tolist(), mode="markers",
                name="device track",
                marker=dict(color=_line(_paint), colorscale="Viridis",
                            size=5, showscale=True, colorbar=dict(title="speed<br>m/s")))]
vgraph_tr = [
    dict(x=frame_sod.tolist(), y=_line(gt_dop_speed), mode="lines", name="GT Doppler",
         line=dict(color="#111", width=2)),
    dict(x=frame_sod.tolist(), y=_line(gt_cd_speed), mode="lines", name="GT coord-Δ",
         line=dict(color="#888", width=1, dash="dot")),
    dict(x=frame_sod.tolist(), y=_line(ph_dop_speed), mode="lines", name="phone Doppler",
         line=dict(color="#e8720c", width=2)),
    dict(x=frame_sod.tolist(), y=_line(ph_cd_speed), mode="lines", name="phone coord-Δ",
         line=dict(color="#f2b077", width=1, dash="dot")),
]
spct_tr = [dict(x=frame_sod.tolist(), y=_line(speed_pct_err), mode="lines",
                name="speed %err (≥10 km/h)", line=dict(color="#e8720c"))]

band = sources["device GPS"]["speed"]["sigma2"] if "device GPS" in sources else list(sources.values())[0]["speed"]["sigma2"]
warnbox = ("".join(f"<p class=warn><b>camera-model source '{gl}' failed</b> — degenerate coords; excluded.</p>"
                   for gl in garbage_labels))
meta_names = [n for n in sources if n != "device GPS"]
mnote = ("" if not meta_names else "<p class=sub>camera-model sources: " + " · ".join(
    f"<b>{n}</b> {sources[n]['n']}/{len(fr)} frames ({sources[n]['coverage']:.0f}%)" for n in meta_names) +
    " — unaligned frames absent, not errors.</p>")
scipy_note = "" if HAVE_SCIPY else "<p class=warn>scipy not found — normality / bootstrap / QQ omitted.</p>"

CH = "<div id='{}' class=chart></div>"
html = f"""<!doctype html><meta charset=utf-8><title>Deep accuracy — camera-model/GPS vs ground truth</title>
<style>
:root{{--fg:#1a2230;--muted:#5b6676;--line:#e4e8ef;--bg:#f5f7fb;--card:#fff;
--accent:#e8720c;--head:#0f1b2d}}
*{{box-sizing:border-box}}
body{{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
color:var(--fg);background:var(--bg);margin:0}}
.wrap{{max-width:1340px;margin:0 auto;padding:22px}}
.banner{{background:linear-gradient(120deg,#0f1b2d,#1e3a5f);color:#fff;
padding:22px 26px;border-radius:14px;margin-bottom:18px;box-shadow:0 4px 18px rgba(15,27,45,.18)}}
.banner h1{{font-size:21px;font-weight:700;margin:0 0 6px}}
.banner .sub{{color:#b9c8de;font-size:12.5px;margin:0}}
.banner code{{background:rgba(255,255,255,.14);color:#fff}}
h2{{font-size:14.5px;font-weight:650;margin:30px 0 8px;padding:2px 0 2px 12px;
border-left:4px solid var(--accent);color:var(--head)}}
table{{border-collapse:separate;border-spacing:0;width:100%;margin:6px 0 2px;font-size:11.5px;
background:var(--card);border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(20,30,50,.07)}}
th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid var(--line)}}
tr:last-child td{{border-bottom:none}}
th:first-child,td:first-child{{text-align:left;font-weight:600}}
tr:first-child th{{background:#eef2f8;color:var(--head);font-weight:650}}
tr:nth-child(even) td{{background:#fafbfd}}
tr:hover td{{background:#eef6ff}}
.sub{{color:var(--muted);font-size:12px;margin:4px 0 2px}}
.warn{{background:#fff6e0;border:1px solid #f0c869;border-radius:8px;padding:9px 13px;margin:8px 0}}
.chart{{width:100%;height:460px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:8px;box-shadow:0 1px 3px rgba(20,30,50,.05);margin-top:4px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
code{{background:#eef2f8;padding:1px 5px;border-radius:4px;font-size:12px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
<div class=wrap>
<div class=banner>
<h1>Trajectory accuracy report</h1>
<p class=sub>Session <code>{SESSION}</code> · {len(fr)} frames · t0={t0:.3f} sod (sync residual {sync_resid_m:.2f} m)
· device GPS &amp; camera-model vs ground truth ({A.gt.name}) · ENU origin = georef[0] · sampled at frame times
· stats engine {'scipy' if HAVE_SCIPY else 'numpy-only'}</p>
</div>
{warnbox}{scipy_note}{mnote}
{('<p class=warn><b>Vertical datum:</b> ' + datum_note + '.</p>') if datum_note else ''}
<h2>1 · Trajectory &amp; scale</h2>{traj_tbl()}
<h2>2 · Position accuracy — absolute (m)</h2>{position_tbl()}
<p class=sub>σ = 68.27/95.45/99.73 pct of |error|. CEP=horizontal radial pct; R95=3D radial; DRMS/2DRMS horizontal; MRSE 3D; bias=mean signed. This is <b>absolute</b> error (no fit) — datum/control bias included.</p>
<h2>2b · Absolute vs aligned — systematic vs shape</h2>{aligned_tbl()}
<p class=sub>ALIGNED = after a 7-param Umeyama fit of source→GT (removes datum/control offset, scale, rotation), leaving pure geometric shape + noise. <b>ABS−ALN</b> = the systematic component. camera-model used the device GPS as control, so a large gap means it mostly reproduced the (biased) GPS; a small aligned error means the <i>shape</i> is good even where the datum is off. fit scale ≠ 1 quantifies scale error independently of path length.</p>
<h2>3 · Robust &amp; distribution stats</h2>{robust_tbl()}
<p class=sub>robust σ = 1.4826·MAD (outlier-immune). trimmed RMS = 5–95 pct. outlier% = |x−median|&gt;3·robustσ. skew/kurtosis of signed error; Anderson-Darling normality (5% crit).</p>
<h2>4 · Inference — CI honouring serial correlation</h2>{infer_tbl()}
<p class=sub>{BOOT_N}-resample bootstrap. Errors are serially correlated (lag-1 ρ), so an IID resample gives a <b>too-narrow</b> CI (shown for contrast); the headline CI is a <b>moving-block</b> bootstrap (block ≈ (1+ρ)/(1−ρ)) that preserves the correlation. eff. N = N·(1−ρ)/(1+ρ) is the real independent-sample count.</p>
<h2>5 · Percentiles of |error| (m)</h2>{pct_tbl()}
<h2>6 · Per-axis &amp; motion</h2>{motion_tbl()}
<p class=sub><b>Speed &amp; azimuth stats are computed over frames moving ≥ 10 km/h only.</b> along/cross = error on GT heading axes. speed lag = frame offset maximising src-vs-GT speed correlation (≈0 ⇒ good time sync). corr = Pearson of source vs GT position per axis.</p>
<h2>7 · Error-ellipse geometry, drift &amp; nominal coverage</h2>{geom_tbl()}
<p class=sub>ellipse a/b = 1σ semi-axes of the E–N error covariance; orient = major-axis bearing; ecc = eccentricity. drift = slope of horizontal error vs distance travelled. err–speed slope = error growth per m/s. within-nominal = % frames whose error ≤ the device's reported accuracy.</p>
<h2>8 · Circular azimuth statistics</h2>{circ_tbl()}
<p class=sub>heading-of-motion error. resultant R∈[0,1] (1=concentrated). Rayleigh p&lt;0.05 ⇒ a directional bias (not uniform noise).</p>
<h2>9 · Error CDF — % frames within horizontal threshold</h2>{cdf_tbl()}
<h2>10 · Top-view map (ENU)</h2>{CH.format('map')}
<h2>10b · Trajectory painted by speed</h2>
<p class=sub>Device track coloured by speed (m/s) — shows where the drive was fast/slow.</p>{CH.format('vmap')}
<h2>10c · Velocities — GT &amp; phone, Doppler vs coordinate-Δ</h2>
<p class=sub>Four speeds over time: GT Doppler, GT coordinate-delta, phone Doppler, phone coordinate-delta. Doppler is the receiver's native velocity; coordinate-Δ is Δposition/Δt (noisier).</p>{CH.format('vgraph')}
<h2>10d · 3D-speed error as % of speed (≥ 10 km/h)</h2>
<p class=sub>|phone speed − GT speed| / GT speed, only where GT ≥ 10 km/h (a 1 m/s error at 10 m/s = 10%).</p>{CH.format('spct')}
<h2>11 · Horizontal error over time · vs speed</h2><div class=grid>{CH.format('etime')}{CH.format('espeed')}</div>
<h2>12 · Error covariance ellipses (E–N) · horizontal-error histogram</h2><div class=grid>{CH.format('ellipse')}{CH.format('hist')}</div>
<h2>13 · Error autocorrelation (ACF) · normal QQ (E error)</h2><div class=grid>{CH.format('acf')}{CH.format('qq')}</div>
<h2>14 · 3D-speed accuracy (m/s)</h2>
<p class=sub>2σ band shaded (device {band:.2f} m/s); y clipped ±{SPEED_CLIP:.0f} m/s (outliers in §6).</p>{CH.format('speed')}
<h2>15 · Horizontal-error CDF</h2>{CH.format('cdf')}
</div>
<script>PLOTLY_SRC</script><script>SCRIPT_BLOCK</script>"""

def P(div, tr, layout):
    return f"Plotly.newPlot('{div}',{json.dumps(tr)},{layout},{{responsive:true}});"
LH = "{legend:{orientation:'h'},margin:{t:10}}"
script = (
    P("map", map_tr, "{xaxis:{title:'E [m]',scaleanchor:'y',scaleratio:1},yaxis:{title:'N [m]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("vmap", vmap_tr, "{xaxis:{title:'E [m]',scaleanchor:'y',scaleratio:1},yaxis:{title:'N [m]'},margin:{t:10}}")
    + P("vgraph", vgraph_tr, "{xaxis:{title:'time [sod]'},yaxis:{title:'speed [m/s]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("spct", spct_tr, "{xaxis:{title:'time [sod]'},yaxis:{title:'speed error [%]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("etime", etime_tr, "{xaxis:{title:'time [sod]'},yaxis:{title:'horizontal error [m]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("espeed", espeed_tr, "{xaxis:{title:'GT speed [m/s]'},yaxis:{title:'horizontal error [m]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("ellipse", ell_tr, "{xaxis:{title:'E error [m]',scaleanchor:'y',scaleratio:1},yaxis:{title:'N error [m]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("hist", hist_tr, "{barmode:'overlay',xaxis:{title:'horizontal error [m]'},yaxis:{title:'frames'},legend:{orientation:'h'},margin:{t:10}}")
    + P("acf", acf_tr, "{xaxis:{title:'lag (frames)'},yaxis:{title:'autocorrelation',range:[-0.2,1]},legend:{orientation:'h'},margin:{t:10}}")
    + P("qq", qq_tr, "{xaxis:{title:'theoretical quantiles'},yaxis:{title:'E error [m]'},legend:{orientation:'h'},margin:{t:10}}")
    + P("speed", speed_tr, "{xaxis:{title:'time [sod]'},yaxis:{title:'3D-speed error [m/s]',range:[" + f"{-SPEED_CLIP},{SPEED_CLIP}" + "]},legend:{orientation:'h'},shapes:[{type:'rect',xref:'paper',x0:0,x1:1,y0:" + f"{-band},y1:{band}" + ",fillcolor:'rgba(232,114,12,.10)',line:{width:0}}],margin:{t:10}}")
    + P("cdf", cdf_tr, "{xaxis:{title:'horizontal error [m]'},yaxis:{title:'% frames \\u2264 x',range:[0,100]},legend:{orientation:'h'},margin:{t:10}}")
)
html = html.replace("PLOTLY_SRC", plotly).replace("SCRIPT_BLOCK", script)
A.out.parent.mkdir(parents=True, exist_ok=True)
A.out.write_text(html, encoding="utf-8")
print(f"[out] {A.out}  ({A.out.stat().st_size // 1024} KB)")
