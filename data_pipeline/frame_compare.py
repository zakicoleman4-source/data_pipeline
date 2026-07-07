"""Compare refined per-sample coordinates against Signal-derived sample positions.

After the pipeline extracts samples and coordinate-tags them from the Post-processing
solution, the user may refine those samples in an external tool and obtain
refined per-sample coordinates. This module ingests that CSV (flexible column
naming; datum-based, projected/Grid or Cartesian XYZ inputs) and compares each sample's
external coordinate against the Signal-interpolated-at-sample-time position,
quantifying the delta between the image-derived geometry and the Reference-derived
geometry.

The headline question it answers: do the two geometries disagree by a
*systematic offset* (a bias vector -- e.g. a time-sync or lever-arm error, or
a datum shift in the external solution) or by *random scatter* (noise)?

Public API
----------
- :func:`load_external_frame_coords` -- read external sample coordinates CSV
  -> ``{image_stem: (lat_deg, lon_deg, h_m_or_None)}``.
- :func:`load_gnss_frame_coords_from_georef` -- read an existing Georef.csv
  -> ``{image_stem: (lat_deg, lon_deg, h_m_or_None)}``.
- :func:`compute_deltas` -- join by image stem, local-Local-frame deltas + summary.
- :func:`write_delta_csv` -- write per-sample delta records.
- :func:`format_summary` -- human-readable summary text.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence

from .geo import ecef_to_enu, heading_from_enu, llh_to_ecef

logger = logging.getLogger(__name__)

# The standard datum constants (mirror geo.py).
_A: float = 6_378_137.0
_F: float = 1.0 / 298.257_223_563
_E2: float = _F * (2.0 - _F)

# {image_stem: (lat_deg, lon_deg, h_m or None)}
FrameLLH = Mapping[str, tuple[float, float, Optional[float]]]


# -----------------------------
# Image key normalisation
# -----------------------------


def normalize_image_key(name: str) -> str:
    """Normalise an image reference to its extension-stripped stem.

    Matches how Georef.csv writes its ``Image`` labels (filename stem,
    extension stripped) so external rows join regardless of whether the
    external tool kept the extension or a directory prefix.
    """
    return Path(str(name).strip().strip('"')).stem


def warn_duplicate_stems(source: str, collided: Sequence[str]) -> int:
    """Warn (once) when normalised image stems collided while building a
    ``{stem: coords}`` dict, i.e. rows silently overwrote earlier ones.

    ``a/frame_1.png`` + ``b/frame_1.png`` (or ``frame_1.jpg`` +
    ``frame_1.png``) normalise to the same stem; last row wins. That
    behaviour is kept, but it must be VISIBLE: a collision means the join
    against the other side may use the wrong row's coordinates.

    Returns the collision count (0 = silent).
    """
    if not collided:
        return 0
    uniq = sorted(set(collided))
    shown = ", ".join(uniq[:5]) + (", ..." if len(uniq) > 5 else "")
    logger.warning(
        "%s: %d row(s) collided onto %d already-seen image stem(s) after "
        "extension/directory stripping (last row wins): %s. Coordinates for "
        "these frames may come from the wrong row.",
        source, len(collided), len(uniq), shown,
    )
    return len(collided)


# -----------------------------
# Cartesian XYZ -> datum-based (no external deps)
# -----------------------------


def _ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Iterative Cartesian XYZ -> datum-based (Bowring-style), sub-mm away from the poles.

    Same iteration as ``geo.enu_to_llh`` uses internally; factored here so an
    Cartesian XYZ external input can be converted without pyproj.
    """
    p = math.sqrt(x * x + y * y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
        n_rad = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        h = abs(z) - n_rad * (1.0 - _E2)
        return math.degrees(lat), 0.0, h
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(5):
        n_rad = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        lat = math.atan2(z + _E2 * n_rad * math.sin(lat), p)
    n_rad = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    lon = math.atan2(y, x)
    h = p / math.cos(lat) - n_rad
    return math.degrees(lat), math.degrees(lon), h


# -----------------------------
# Column auto-detection
# -----------------------------


def _norm_header(h: str) -> str:
    return h.strip().lstrip("#").strip().lower().replace(" ", "_")


def _find_col(headers: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the original header whose normalised form matches a candidate.

    Exact (normalised) matches only, tried in candidate priority order.
    """
    norm = {_norm_header(h): h for h in headers}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


_NAME_CANDS = (
    "image", "label", "name", "frame", "file", "filename",
    "image_name", "file_name", "camera", "photo",
)
_LAT_CANDS = ("latitude", "lat", "lat_deg", "latitude_deg", "lat_wgs84")
_LON_CANDS = ("longitude", "lon", "lng", "long", "lon_deg", "longitude_deg",
              "lon_wgs84")
_ALT_CANDS = ("altitude", "alt", "h", "height", "z", "ellipsoidal_height",
              "elevation", "alt_m", "h_m", "height_m")
_EAST_CANDS = ("easting", "east", "easting_m", "utm_e", "x")
_NORTH_CANDS = ("northing", "north", "northing_m", "utm_n", "y")
_ZONE_CANDS = ("zone", "utm_zone", "utmzone")
_EPSG_CANDS = ("epsg", "epsg_code", "crs", "srid")
_ECEF_X_CANDS = ("x_ecef", "ecef_x", "xecef", "ecefx")
_ECEF_Y_CANDS = ("y_ecef", "ecef_y", "yecef", "ecefy")
_ECEF_Z_CANDS = ("z_ecef", "ecef_z", "zecef", "ecefz")


def _to_float(v: object) -> Optional[float]:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _utm_zone_to_epsg(zone_str: str) -> int:
    """The external solver-style '10T' / '32U' / '32N' zone string -> EPSG code.

    The letter is interpreted as an MGRS *latitude band*: bands C-M are the
    southern hemisphere, N-X the northern (matching the convention used
    elsewhere in this codebase for Grid inputs).

    AMBIGUITY: a bare 'N' or 'S' letter could also be a *hemisphere* flag
    ('N' = north, 'S' = south) as emitted by some tools. Under the MGRS-band
    default used here, '32S' resolves to NORTH (band S covers ~32-40 deg N,
    EPSG 32632) -- the opposite hemisphere from what a hemisphere-convention
    tool means by '32S'. A loud warning is emitted for exactly-'N'/'S'
    letters documenting which interpretation was used; pass an explicit EPSG
    code (326xx north / 327xx south) via the ``epsg`` override to bypass the
    guess entirely.
    """
    import re

    m = re.match(r"^\s*(\d{1,2})\s*([A-Za-z]?)\s*$", str(zone_str))
    if not m:
        raise ValueError(
            f"invalid UTM zone: {zone_str!r}. Expected '<1-60><C-X>' "
            "e.g. '10T' or '32U'."
        )
    zone = int(m.group(1))
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be 1..60, got {zone} from {zone_str!r}.")
    letter = (m.group(2) or "N").upper()
    if letter in {"A", "B", "I", "O", "Y", "Z"}:
        raise ValueError(
            f"UTM band letter {letter!r} not allowed (reserved or polar)."
        )
    northern = letter >= "N"
    epsg = (32600 if northern else 32700) + zone
    if letter == "S":
        logger.warning(
            "UTM zone %r: letter 'S' is AMBIGUOUS. Interpreted as MGRS "
            "latitude band S (~32-40 deg N, NORTHERN hemisphere -> EPSG "
            "%d). If your tool means hemisphere 'S' (southern), this is the "
            "WRONG hemisphere: pass epsg=%d explicitly instead.",
            zone_str, epsg, 32700 + zone,
        )
    elif letter == "N":
        logger.warning(
            "UTM zone %r: letter 'N' is ambiguous (MGRS band N vs "
            "hemisphere flag N). Both mean the northern hemisphere here "
            "(EPSG %d); pass an explicit epsg= to silence this warning.",
            zone_str, epsg,
        )
    return epsg


class _ProjTransformCache:
    """Lazy per-EPSG pyproj transformer cache (projected -> The standard datum datum-based)."""

    def __init__(self) -> None:
        self._cache: dict[int, object] = {}

    def to_wgs84(self, epsg: int, easting: float, northing: float) -> tuple[float, float]:
        try:
            import pyproj
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "projected/UTM external coordinates need pyproj. "
                "Run: pip install pyproj>=3.4.0"
            ) from e
        tr = self._cache.get(epsg)
        if tr is None:
            tr = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)
            self._cache[epsg] = tr
        lon, lat = tr.transform(easting, northing)
        return lat, lon


def load_external_frame_coords(
    path: Path | str,
    *,
    epsg: Optional[int] = None,
    utm_zone: Optional[str] = None,
) -> dict[str, tuple[float, float, Optional[float]]]:
    """Read refined external per-sample coordinates from a CSV.

    Column names are matched case-insensitively. The image/name column is any
    of ``image|label|name|sample|file|filename|...``; coordinates may be:

    - datum-based: ``latitude|lat``, ``longitude|lon|lng``, optional
      ``altitude|alt|h|height|z`` (The standard datum assumed);
    - projected/Grid: ``easting|x``, ``northing|y`` plus a per-row
      ``zone``/``epsg`` column, or the ``epsg=`` / ``utm_zone=`` keyword;
    - Cartesian XYZ: ``x_ecef``, ``y_ecef``, ``z_ecef``.

    Leading ``#`` comment lines are skipped. Returns
    ``{image_stem: (lat_deg, lon_deg, h_m_or_None)}`` with image keys
    normalised to extension-stripped stems (matching Georef.csv labels).
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{p}: no data rows found")
    # Sniff the delimiter (comma / semicolon / tab are all seen in the wild).
    try:
        dialect = csv.Sniffer().sniff(lines[0], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(lines, dialect=dialect)
    headers = reader.fieldnames or []

    name_col = _find_col(headers, _NAME_CANDS)
    if name_col is None:
        raise ValueError(
            f"{p}: could not detect an image/name column. "
            f"Headers: {headers}. Expected one of {list(_NAME_CANDS)}."
        )

    ecef_x = _find_col(headers, _ECEF_X_CANDS)
    ecef_y = _find_col(headers, _ECEF_Y_CANDS)
    ecef_z = _find_col(headers, _ECEF_Z_CANDS)
    lat_col = _find_col(headers, _LAT_CANDS)
    lon_col = _find_col(headers, _LON_CANDS)
    alt_col = _find_col(headers, _ALT_CANDS)
    east_col = _find_col(headers, _EAST_CANDS)
    north_col = _find_col(headers, _NORTH_CANDS)
    zone_col = _find_col(headers, _ZONE_CANDS)
    epsg_col = _find_col(headers, _EPSG_CANDS)

    out: dict[str, tuple[float, float, Optional[float]]] = {}
    collided: list[str] = []
    proj_cache = _ProjTransformCache()
    default_epsg: Optional[int] = int(epsg) if epsg is not None else None
    if default_epsg is None and utm_zone:
        default_epsg = _utm_zone_to_epsg(utm_zone)

    if ecef_x and ecef_y and ecef_z:
        mode = "ecef"
    elif lat_col and lon_col:
        mode = "geodetic"
    elif east_col and north_col:
        mode = "projected"
        if default_epsg is None and zone_col is None and epsg_col is None:
            raise ValueError(
                f"{p}: projected coordinates detected "
                f"({east_col!r}/{north_col!r}) but no zone/EPSG column and no "
                "epsg=/utm_zone= override was given."
            )
    else:
        raise ValueError(
            f"{p}: could not detect coordinate columns. Headers: {headers}. "
            "Expected geodetic (latitude/longitude), projected "
            "(easting/northing + zone/epsg) or ECEF (x_ecef/y_ecef/z_ecef)."
        )

    for row in reader:
        raw_name = (row.get(name_col) or "").strip()
        if not raw_name:
            continue
        key = normalize_image_key(raw_name)
        if not key:
            continue

        if mode == "ecef":
            x = _to_float(row.get(ecef_x))
            y = _to_float(row.get(ecef_y))
            z = _to_float(row.get(ecef_z))
            if x is None or y is None or z is None:
                continue
            lat, lon, h = _ecef_to_llh(x, y, z)
            if key in out:
                collided.append(key)
            out[key] = (lat, lon, h)
            continue

        if mode == "geodetic":
            lat = _to_float(row.get(lat_col))
            lon = _to_float(row.get(lon_col))
            if lat is None or lon is None:
                continue
            h = _to_float(row.get(alt_col)) if alt_col else None
            if key in out:
                collided.append(key)
            out[key] = (lat, lon, h)
            continue

        # mode == "projected"
        e_v = _to_float(row.get(east_col))
        n_v = _to_float(row.get(north_col))
        if e_v is None or n_v is None:
            continue
        row_epsg = default_epsg
        if epsg_col:
            v = _to_float(row.get(epsg_col))
            if v is not None:
                row_epsg = int(v)
        if row_epsg is None and zone_col:
            zv = (row.get(zone_col) or "").strip()
            if zv:
                row_epsg = _utm_zone_to_epsg(zv)
        if row_epsg is None:
            continue
        lat, lon = proj_cache.to_wgs84(row_epsg, e_v, n_v)
        h = _to_float(row.get(alt_col)) if alt_col else None
        if key in out:
            collided.append(key)
        out[key] = (lat, lon, h)

    warn_duplicate_stems(str(p), collided)
    if not out:
        raise ValueError(f"{p}: no usable coordinate rows parsed")
    return out


def load_gnss_frame_coords_from_georef(
    georef_csv_path: Path | str,
) -> dict[str, tuple[float, float, Optional[float]]]:
    """Read the pipeline's Georef.csv -> ``{stem: (lat, lon, h_or_None)}``.

    Columns are ``Image, Latitude, Longitude, [Altitude], ...``; the Image
    label is already the extension-stripped stem. Leading ``#`` comment lines
    are skipped. Altitude is ``None`` when the file was written without it.
    """
    p = Path(georef_csv_path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{p}: no data rows found")
    reader = csv.DictReader(lines)
    headers = reader.fieldnames or []
    name_col = _find_col(headers, ("image",) + _NAME_CANDS)
    lat_col = _find_col(headers, _LAT_CANDS)
    lon_col = _find_col(headers, _LON_CANDS)
    alt_col = _find_col(headers, ("altitude", "alt"))
    if name_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"{p}: expected Image/Latitude/Longitude columns, got {headers}"
        )
    out: dict[str, tuple[float, float, Optional[float]]] = {}
    collided: list[str] = []
    for row in reader:
        raw_name = (row.get(name_col) or "").strip()
        lat = _to_float(row.get(lat_col))
        lon = _to_float(row.get(lon_col))
        if not raw_name or lat is None or lon is None:
            continue
        h = _to_float(row.get(alt_col)) if alt_col else None
        key = normalize_image_key(raw_name)
        if key in out:
            collided.append(key)
        out[key] = (lat, lon, h)
    warn_duplicate_stems(str(p), collided)
    if not out:
        raise ValueError(f"{p}: no usable coordinate rows parsed")
    return out


# -----------------------------
# Delta computation
# -----------------------------


@dataclass(frozen=True)
class FrameDelta:
    """Per-sample delta: external coordinate minus Signal coordinate (local Local-frame)."""

    image: str
    ext_lat: float
    ext_lon: float
    ext_h: Optional[float]
    gnss_lat: float
    gnss_lon: float
    gnss_h: Optional[float]
    d_east_m: float
    d_north_m: float
    d_up_m: float  # NaN when height missing on either side
    d_horiz_m: float
    d_vert_m: float  # NaN when height missing on either side


@dataclass(frozen=True)
class CompareResult:
    records: list[FrameDelta]
    summary: dict


def compute_deltas(external: FrameLLH, gnss: FrameLLH) -> CompareResult:
    """Join by image stem and compute local-Local-frame deltas + a summary.

    For each shared image stem, the delta is (external - Signal) expressed in
    the local East/North/Up sample anchored at the Signal point. The horizontal
    delta uses the Signal height on both sides when the external height is
    missing, so a missing altitude column never contaminates the horizontal
    numbers; up/vertical stats are only produced when heights are present on
    both sides.

    Summary keys::

        n_external, n_gnss, n_matched
        mean_horiz_m, median_horiz_m, two_sigma_horiz_m, max_horiz_m
        mean_east_m, mean_north_m         (the systematic offset vector)
        mean_offset_horiz_m               |mean offset vector|
        bearing_deg                       compass bearing of the mean offset
        std_horiz_m                       scatter of the horizontal deltas
                                          about the mean offset vector (RMS)
        classification                    'systematic_bias' | 'scattered'
                                          | 'no_match'
        n_vert, mean_up_m, median_vert_m, two_sigma_vert_m, max_abs_up_m
                                          (present when heights on both sides)
    """
    records: list[FrameDelta] = []
    shared = sorted(set(external) & set(gnss))
    for key in shared:
        e_lat, e_lon, e_h = external[key]
        g_lat, g_lon, g_h = gnss[key]
        ref_h = g_h if g_h is not None else 0.0
        both_h = e_h is not None and g_h is not None
        # Use the Signal height for the external point when its own height is
        # missing so the up component cannot leak into east/north.
        h_used = e_h if both_h else ref_h
        x, y, z = llh_to_ecef(e_lat, e_lon, h_used)
        de, dn, du = ecef_to_enu(x, y, z, (g_lat, g_lon, ref_h))
        d_horiz = math.hypot(de, dn)
        d_up = du if both_h else float("nan")
        records.append(
            FrameDelta(
                image=key,
                ext_lat=e_lat, ext_lon=e_lon, ext_h=e_h,
                gnss_lat=g_lat, gnss_lon=g_lon, gnss_h=g_h,
                d_east_m=de, d_north_m=dn, d_up_m=d_up,
                d_horiz_m=d_horiz, d_vert_m=d_up,
            )
        )

    summary = _summarize(records, n_external=len(external), n_gnss=len(gnss))
    return CompareResult(records=records, summary=summary)


def _empirical_two_sigma(values: list[float]) -> float:
    """Empirical 2-sigma (95.45th percentile, linear interpolation)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = 0.9545 * (len(s) - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 >= len(s):
        return s[-1]
    return s[i] + (s[i + 1] - s[i]) * frac


def _summarize(records: list[FrameDelta], *, n_external: int, n_gnss: int) -> dict:
    summary: dict = {
        "n_external": n_external,
        "n_gnss": n_gnss,
        "n_matched": len(records),
    }
    if not records:
        summary["classification"] = "no_match"
        return summary

    des = [r.d_east_m for r in records]
    dns = [r.d_north_m for r in records]
    dhs = [r.d_horiz_m for r in records]
    n = len(records)

    mean_e = sum(des) / n
    mean_n = sum(dns) / n
    mean_offset = math.hypot(mean_e, mean_n)
    # Scatter about the mean offset vector (RMS of the residual vectors).
    resid_sq = [
        (de - mean_e) ** 2 + (dn - mean_n) ** 2 for de, dn in zip(des, dns)
    ]
    std_horiz = math.sqrt(sum(resid_sq) / n)

    summary.update(
        mean_horiz_m=sum(dhs) / n,
        median_horiz_m=median(dhs),
        two_sigma_horiz_m=_empirical_two_sigma(dhs),
        max_horiz_m=max(dhs),
        mean_east_m=mean_e,
        mean_north_m=mean_n,
        mean_offset_horiz_m=mean_offset,
        bearing_deg=heading_from_enu(mean_e, mean_n),
        std_horiz_m=std_horiz,
    )

    # Systematic when the mean offset vector dominates the scatter about it.
    # The 1 mm floor keeps a perfectly rigid (zero-scatter) shift classified
    # as systematic without a divide-by-zero.
    summary["classification"] = (
        "systematic_bias"
        if mean_offset > 2.0 * max(std_horiz, 1e-3)
        else "scattered"
    )

    ups = [r.d_up_m for r in records if math.isfinite(r.d_up_m)]
    summary["n_vert"] = len(ups)
    if ups:
        abs_ups = [abs(u) for u in ups]
        summary.update(
            mean_up_m=sum(ups) / len(ups),
            median_vert_m=median(ups),
            two_sigma_vert_m=_empirical_two_sigma(abs_ups),
            max_abs_up_m=max(abs_ups),
        )
    return summary


# -----------------------------
# Output
# -----------------------------

_DELTA_COLUMNS = [
    "Image", "ext_lat", "ext_lon", "ext_h",
    "gnss_lat", "gnss_lon", "gnss_h",
    "d_east_m", "d_north_m", "d_up_m", "d_horiz_m", "d_vert_m",
]


def _fmt(v: Optional[float], nd: int) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def write_delta_csv(records: Sequence[FrameDelta], out_path: Path | str) -> Path:
    """Write per-sample delta records to ``out_path`` (CSV). Returns the path."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_DELTA_COLUMNS)
        for r in records:
            w.writerow([
                r.image,
                _fmt(r.ext_lat, 9), _fmt(r.ext_lon, 9), _fmt(r.ext_h, 4),
                _fmt(r.gnss_lat, 9), _fmt(r.gnss_lon, 9), _fmt(r.gnss_h, 4),
                _fmt(r.d_east_m, 4), _fmt(r.d_north_m, 4), _fmt(r.d_up_m, 4),
                _fmt(r.d_horiz_m, 4), _fmt(r.d_vert_m, 4),
            ])
    return p


def format_summary(summary: Mapping) -> str:
    """Render the summary dict as readable multi-line text for CLI output."""
    lines: list[str] = []
    lines.append(
        f"matched frames: {summary.get('n_matched', 0)} "
        f"(external={summary.get('n_external', '?')}, "
        f"gnss={summary.get('n_gnss', '?')})"
    )
    if summary.get("n_matched", 0) == 0:
        lines.append("no shared image labels -- nothing to compare")
        return "\n".join(lines)
    lines.append(
        "horizontal delta: "
        f"mean={summary['mean_horiz_m']:.3f} m  "
        f"median={summary['median_horiz_m']:.3f} m  "
        f"2sigma={summary['two_sigma_horiz_m']:.3f} m  "
        f"max={summary['max_horiz_m']:.3f} m"
    )
    bearing = summary.get("bearing_deg", float("nan"))
    bearing_txt = f"{bearing:.1f} deg" if math.isfinite(bearing) else "n/a"
    lines.append(
        "systematic offset vector (external - GNSS): "
        f"east={summary['mean_east_m']:+.3f} m  "
        f"north={summary['mean_north_m']:+.3f} m  "
        f"|offset|={summary['mean_offset_horiz_m']:.3f} m  "
        f"bearing={bearing_txt}"
    )
    lines.append(f"scatter about mean offset: std={summary['std_horiz_m']:.3f} m")
    if summary.get("n_vert", 0):
        lines.append(
            f"vertical (n={summary['n_vert']}): "
            f"mean_up={summary['mean_up_m']:+.3f} m  "
            f"median={summary['median_vert_m']:+.3f} m  "
            f"2sigma(|up|)={summary['two_sigma_vert_m']:.3f} m  "
            f"max|up|={summary['max_abs_up_m']:.3f} m"
        )
    cls = summary["classification"]
    if cls == "systematic_bias":
        lines.append(
            "verdict: SYSTEMATIC BIAS -- the external frame geometry is "
            "shifted as a block relative to GNSS (mean offset >> scatter). "
            "Likely a time-sync, lever-arm or datum/reference difference."
        )
    else:
        lines.append(
            "verdict: SCATTERED -- deltas look like zero-mean noise; no "
            "dominant block shift between external geometry and GNSS."
        )
    return "\n".join(lines)
