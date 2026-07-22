"""Camera-model vs GPS vs ground-truth accuracy comparison.

After the pipeline extracts and coordinate-tags frames, the user may build a
3D reconstruction of those frames in an external tool ("the camera model").
This module answers: which is closer to a survey-grade ground-truth ``.pos``
track — the camera-model per-frame positions, or the GPS post-processed
per-frame positions?

Per frame (present in ALL of camera model / GPS / ground truth) it computes,
in a shared local East/North/Up frame:

* horizontal error vs ground truth (camera model and GPS separately);
* azimuth (heading-of-motion) error vs the ground-truth motion bearing;
* 3D speed error and 3D velocity-vector error magnitude.

Aggregates follow the project convention: 1/2/3-sigma values are the
68.27 / 95.45 / 99.73 percentiles of the |error| distribution (the classic
standard deviation is also reported).

Reconstruction frame handling
-----------------------------
The reconstruction may or may not be georeferenced; :func:`detect_reconstruction_frame`
auto-detects:

* ``"llh"``   — camera centers look like (lon, lat, h) landing near the GPS
  track: converted directly to the common ENU frame.
* ``"local"`` — centers are already metric coordinates overlapping the GPS
  track's ENU extent: used as-is.
* ``"raw"``   — an arbitrary local frame: a 7-parameter similarity
  (Umeyama, scale+rotation+translation) is fitted from the shared frame
  names onto the GPS ENU positions, with a few reweighted iterations to
  reject outliers. The fitted scale and RMS residual are reported so the
  alignment quality is visible.

Public API
----------
- :func:`parse_colmap_images` — ``images.txt`` / ``images.bin`` ->
  ``{name_stem: (Cx, Cy, Cz)}`` camera centers (``C = -R(q)^T t``).
- :func:`frame_times_from_colmap_names` — ``frame_<t>_<idx>`` stems ->
  ``{stem: embedded_time_s}`` (self-timing camera-model->COLMAP exports).
- :func:`parse_metashape_cameras` — camera-model estimated-coordinates CSV
  (WGS84) -> ``{name_stem: (lat, lon, h)}``.
- :func:`camera_center_from_pose` — one (qw,qx,qy,qz,tx,ty,tz) -> center.
- :func:`umeyama_similarity` / :func:`fit_similarity_robust` — similarity fit.
- :func:`detect_reconstruction_frame` — georeference auto-detection.
- :func:`recover_frame_times_from_pos` — frame UTC recovery by projecting
  each frame's GPS position onto the GPS ``.pos`` track.
- :func:`sigma_bands` — the 1/2/3-sigma aggregation.
- :func:`build_report` — end-to-end: CSV + self-contained HTML + verdict.
"""

from __future__ import annotations

import csv
import datetime as _dt
import html as _html_mod
import json
import logging
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .frame_compare import (
    _NAME_CANDS,
    _find_col,
    _to_float,
    load_gnss_frame_coords_from_georef,
    normalize_image_key,
    warn_duplicate_stems,
)
from .geo import ecef_to_enu, heading_from_enu, llh_to_ecef
from .parsers import PosRow, interp_pos, interp_pos_with_velocity, parse_rtkpos

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

# {stem: (x, y, z)} camera centers in the reconstruction frame.
CamCenters = Mapping[str, tuple[float, float, float]]
# {stem: (lat_deg, lon_deg, h_m_or_None)}
FrameLLH = Mapping[str, tuple[float, float, Optional[float]]]

# Sigma-band percentiles (project convention: percentiles of |error|).
_SIGMA_PCTS = (68.27, 95.45, 99.73)

# Motion vectors shorter than this (m, horizontal) give no usable bearing.
_MIN_BEARING_DISP_M = 0.02

# Native-velocity horizontal speed (m/s) below which no bearing is derived.
_MIN_BEARING_SPEED_MPS = 0.02


# ---------------------------------------------------------------------------
# Reconstruction (COLMAP) parsing
# ---------------------------------------------------------------------------


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """(w,x,y,z) quaternion -> 3x3 rotation matrix (world -> camera).

    The quaternion is normalised first; a zero-norm quaternion raises.
    """
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion in reconstruction pose")
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def camera_center_from_pose(
    qvec: Sequence[float], tvec: Sequence[float],
) -> tuple[float, float, float]:
    """Camera CENTER in the reconstruction frame from a stored pose.

    The reconstruction stores the world->camera rotation ``R`` (as a
    (w,x,y,z) quaternion) and translation ``t``; the projection is
    ``p_cam = R p_world + t``, so the camera center (where ``p_cam = 0``)
    is ``C = -R^T t``.
    """
    r = _quat_to_rot(*qvec)
    t = np.asarray(tvec, dtype=float)
    c = -r.T @ t
    return (float(c[0]), float(c[1]), float(c[2]))


def _parse_images_txt(path: Path) -> dict[str, tuple[float, float, float]]:
    """Parse the text ``images.txt``: two lines per image, the second is the
    2D-point list (ignored; may be empty for images with no points)."""
    out: dict[str, tuple[float, float, float]] = {}
    collided: list[str] = []
    expecting_pose = True
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("#"):
                continue
            if not expecting_pose:
                # This slot is the 2D-points line (possibly empty). Safety:
                # points lines are (x, y, id) triplets, so a 10+-token line
                # whose count is NOT a multiple of 3 is really the next pose
                # (a writer that omitted the empty points line).
                toks = s.split()
                if toks and len(toks) >= 10 and len(toks) % 3 != 0:
                    expecting_pose = True  # fall through and parse as pose
                else:
                    expecting_pose = True
                    continue
            if not s:
                continue  # blank line where a pose was expected: skip
            parts = s.split()
            if len(parts) < 10:
                raise ValueError(
                    f"{path}: malformed image line (needs >=10 fields: "
                    f"ID QW QX QY QZ TX TY TZ CAMERA_ID NAME): {s[:120]!r}"
                )
            try:
                qvec = tuple(float(v) for v in parts[1:5])
                tvec = tuple(float(v) for v in parts[5:8])
            except ValueError as e:
                raise ValueError(
                    f"{path}: non-numeric pose fields in line {s[:120]!r}"
                ) from e
            name = " ".join(parts[9:])
            key = normalize_image_key(name)
            if key:
                if key in out:
                    collided.append(key)
                out[key] = camera_center_from_pose(qvec, tvec)
            expecting_pose = False
    warn_duplicate_stems(str(path), collided)
    return out


def _parse_images_bin(path: Path) -> dict[str, tuple[float, float, float]]:
    """Parse the binary ``images.bin`` (little-endian).

    Layout: num_reg_images (uint64); per image: image_id (uint32),
    qvec (4 x float64, w,x,y,z), tvec (3 x float64), camera_id (uint32),
    name (null-terminated bytes), num_points2D (uint64), then that many
    (float64 x, float64 y, uint64 point3D_id) records which are skipped.
    """
    out: dict[str, tuple[float, float, float]] = {}
    collided: list[str] = []
    with path.open("rb") as f:
        def _read(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise ValueError(f"{path}: truncated binary image file")
            return b

        (num_images,) = struct.unpack("<Q", _read(8))
        for _ in range(num_images):
            _read(4)  # image_id (uint32)
            qvec = struct.unpack("<4d", _read(32))
            tvec = struct.unpack("<3d", _read(24))
            _read(4)  # camera_id (uint32)
            name_bytes = bytearray()
            while True:
                c = _read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            (num_pts,) = struct.unpack("<Q", _read(8))
            f.seek(num_pts * 24, 1)  # skip (x f64, y f64, id u64) records
            name = name_bytes.decode("utf-8", errors="replace")
            key = normalize_image_key(name)
            if not key:
                continue
            if key in out:
                collided.append(key)
            out[key] = camera_center_from_pose(qvec, tvec)
    warn_duplicate_stems(str(path), collided)
    return out


def _resolve_images_file(path: Path) -> Path:
    """Accept an ``images.txt``/``images.bin`` file or a reconstruction
    directory (``sparse/``, ``sparse/0/`` etc.) containing one."""
    if path.is_file():
        return path
    if path.is_dir():
        for sub in (path, path / "0", path / "sparse", path / "sparse" / "0"):
            for fname in ("images.bin", "images.txt"):
                cand = sub / fname
                if cand.is_file():
                    return cand
        raise FileNotFoundError(
            f"{path}: no images.bin / images.txt found (looked in ., 0/, "
            "sparse/, sparse/0/)"
        )
    raise FileNotFoundError(f"{path}: not found")


def parse_colmap_images(path: Path | str) -> dict[str, tuple[float, float, float]]:
    """Read reconstruction camera poses -> ``{name_stem: (Cx, Cy, Cz)}``.

    ``path`` may be an ``images.txt``, an ``images.bin`` or a directory
    containing either. Image names are normalised to extension-stripped
    stems (matching the Georef.csv ``Image`` labels).
    """
    p = _resolve_images_file(Path(path))
    if p.suffix.lower() == ".bin":
        out = _parse_images_bin(p)
    else:
        out = _parse_images_txt(p)
    if not out:
        raise ValueError(f"{p}: no registered images parsed")
    return out


# ---------------------------------------------------------------------------
# camera-model estimated-coordinates CSV parsing
# ---------------------------------------------------------------------------

# Name-column candidates for the camera-model export (extends the shared list).
_MS_NAME_CANDS = _NAME_CANDS + ("photoid", "photo_id")
# camera-model geographic export column order is X=Longitude, Y=Latitude,
# Z=Altitude; some versions write compound headers like "X/Longitude".
_MS_X_LON_CANDS = ("x/longitude", "x_est", "x")
_MS_Y_LAT_CANDS = ("y/latitude", "y_est", "y")
_MS_Z_ALT_CANDS = ("z/altitude", "z_est", "z")
_MS_LAT_CANDS = ("latitude", "lat", "lat_deg", "latitude_deg")
_MS_LON_CANDS = ("longitude", "lon", "lng", "long", "lon_deg", "longitude_deg")
_MS_ALT_CANDS = ("altitude", "alt", "alt_m", "height", "h", "h_m",
                 "ellipsoidal_height")


def parse_metashape_cameras(
    path: Path | str,
) -> dict[str, tuple[float, float, Optional[float]]]:
    """Read an Agisoft camera-model "Export Reference / estimated coordinates"
    CSV (WGS84) -> ``{name_stem: (lat_deg, lon_deg, h_m_or_None)}``.

    Real-file quirks tolerated:

    * one or more leading ``#`` comment lines — the real header row is often
      itself ``#``-prefixed (e.g. ``#Label,X,Y,Z,...``); a single leading
      ``#`` is stripped from it before splitting;
    * the delimiter may be comma / semicolon / tab (sniffed);
    * the name column may be ``Label`` / ``PhotoID`` / ``Camera`` / ``image``;
    * geographic export column order is **X=Longitude, Y=Latitude,
      Z=Altitude**: explicit ``Longitude``/``Latitude``/``Altitude`` headers
      are used when present, otherwise X -> lon, Y -> lat, Z -> alt.

    Keys are normalised to extension-stripped stems (matching the Georef.csv
    ``Image`` labels).
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        raw = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    if not raw:
        raise ValueError(f"{p}: no data rows found")

    n_lead = 0
    while n_lead < len(raw) and raw[n_lead].lstrip().startswith("#"):
        n_lead += 1
    comments, body = raw[:n_lead], raw[n_lead:]

    # Sniff the delimiter from the most representative line available.
    sample = body[0] if body else comments[-1]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    def _cols(line: str) -> list[str]:
        return next(csv.reader([line], dialect=dialect))

    header_line: Optional[str] = None
    data_lines: list[str] = body
    if body and _find_col(_cols(body[0]), _MS_NAME_CANDS) is not None:
        header_line = body[0]
        data_lines = body[1:]
    elif comments:
        cand = comments[-1].lstrip()
        if cand.startswith("#"):
            cand = cand[1:]  # strip a single leading '#' from the header
        if _find_col(_cols(cand), _MS_NAME_CANDS) is not None:
            header_line = cand
    if header_line is None:
        raise ValueError(
            f"{p}: could not detect a header row with an image/name column "
            f"(expected one of {list(_MS_NAME_CANDS)}). First line: "
            f"{raw[0][:120]!r}"
        )

    reader = csv.DictReader([header_line] + data_lines, dialect=dialect)
    headers = reader.fieldnames or []
    name_col = _find_col(headers, _MS_NAME_CANDS)
    if name_col is None:  # pragma: no cover - guarded by the check above
        raise ValueError(f"{p}: no image/name column in {headers}")

    # Explicit Longitude/Latitude headers win; else camera-model X/Y/Z order.
    lat_col = _find_col(headers, _MS_LAT_CANDS)
    lon_col = _find_col(headers, _MS_LON_CANDS)
    alt_col = _find_col(headers, _MS_ALT_CANDS)
    if lat_col is None or lon_col is None:
        lon_col = _find_col(headers, _MS_X_LON_CANDS)
        lat_col = _find_col(headers, _MS_Y_LAT_CANDS)
        if alt_col is None:
            alt_col = _find_col(headers, _MS_Z_ALT_CANDS)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"{p}: could not detect coordinate columns in {headers}. "
            "Expected Longitude/Latitude(/Altitude) or the Metashape "
            "X(=lon)/Y(=lat)/Z(=alt) geographic export order."
        )

    out: dict[str, tuple[float, float, Optional[float]]] = {}
    collided: list[str] = []
    for row in reader:
        raw_name = (row.get(name_col) or "").strip()
        if not raw_name or raw_name.startswith("#"):
            continue
        key = normalize_image_key(raw_name)
        if not key:
            continue
        lat = _to_float(row.get(lat_col))
        lon = _to_float(row.get(lon_col))
        if lat is None or lon is None:
            continue
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            raise ValueError(
                f"{p}: row {raw_name!r} has (lat={lat}, lon={lon}) outside "
                "geographic range -- export the reference as WGS84 "
                "(geographic), not a projected CRS."
            )
        h = _to_float(row.get(alt_col)) if alt_col else None
        if key in out:
            collided.append(key)
        out[key] = (lat, lon, h)
    warn_duplicate_stems(str(p), collided)
    if not out:
        raise ValueError(f"{p}: no usable coordinate rows parsed")
    return out


# ---------------------------------------------------------------------------
# Similarity (7-parameter, Umeyama) fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityFit:
    """A fitted ``dst ~= scale * R @ src + t`` similarity transform."""

    scale: float
    rotation: tuple  # 3x3 nested tuple
    translation: tuple[float, float, float]
    rms_m: float
    n_used: int
    n_total: int
    rejected: tuple[str, ...] = ()

    def apply(self, pts: np.ndarray) -> np.ndarray:
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation, dtype=float)
        return self.scale * (np.asarray(pts, dtype=float) @ r.T) + t


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``dst ~= s * R @ src + t`` (Umeyama 1991).

    ``src`` / ``dst`` are (N,3) with N >= 3 non-degenerate correspondences.
    Returns ``(scale, R, t)`` with ``R`` a proper rotation (det = +1).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("umeyama_similarity needs matching (N,3) arrays")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"similarity fit needs >=3 correspondences, got {n}")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    var_s = float((xs ** 2).sum() / n)
    if var_s < 1e-18:
        raise ValueError("degenerate source points (zero spread)")
    cov = xd.T @ xs / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    r = u @ s_mat @ vt
    scale = float(np.trace(np.diag(d) @ s_mat) / var_s) if with_scale else 1.0
    t = mu_d - scale * (r @ mu_s)
    return scale, r, t


def fit_similarity_robust(
    names: Sequence[str],
    src: np.ndarray,
    dst: np.ndarray,
    *,
    iterations: int = 3,
    reject_k: float = 4.0,
    reject_floor_m: float = 0.02,
) -> SimilarityFit:
    """Similarity fit with a couple of reweighted (outlier-rejecting) passes.

    Each pass fits on the current inliers, then rejects points whose 3D
    residual exceeds ``max(reject_k * median_residual, reject_floor_m)``.
    Stops early when the inlier set is stable or would drop below 3.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = src.shape[0]
    if len(names) != n:
        raise ValueError("names/src length mismatch")
    mask = np.ones(n, dtype=bool)
    scale, r, t = umeyama_similarity(src, dst)
    for _ in range(max(0, iterations)):
        resid = np.linalg.norm(dst - (scale * (src @ r.T) + t), axis=1)
        med = float(np.median(resid[mask]))
        thr = max(reject_k * med, reject_floor_m)
        new_mask = resid <= thr
        if new_mask.sum() < 3 or bool(np.array_equal(new_mask, mask)):
            break
        mask = new_mask
        scale, r, t = umeyama_similarity(src[mask], dst[mask])
    resid = np.linalg.norm(dst - (scale * (src @ r.T) + t), axis=1)
    rms = float(math.sqrt(float((resid[mask] ** 2).mean()))) if mask.any() else float("nan")
    rejected = tuple(str(names[i]) for i in range(n) if not mask[i])
    return SimilarityFit(
        scale=scale,
        rotation=tuple(tuple(float(v) for v in row) for row in r),
        translation=(float(t[0]), float(t[1]), float(t[2])),
        rms_m=rms,
        n_used=int(mask.sum()),
        n_total=n,
        rejected=rejected,
    )


# ---------------------------------------------------------------------------
# Georeference auto-detection
# ---------------------------------------------------------------------------


def _llh_to_enu(
    lat: float, lon: float, h: float, origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    return ecef_to_enu(*llh_to_ecef(lat, lon, h), origin)


def detect_reconstruction_frame(
    cam_centers: CamCenters,
    gps_enu: Mapping[str, tuple[float, float, float]],
    origin_llh: tuple[float, float, float],
    *,
    llh_match_max_m: float = 10_000.0,
    local_match_frac: float = 0.25,
    local_match_abs_m: float = 30.0,
) -> str:
    """Classify the reconstruction frame: ``"llh"`` / ``"local"`` / ``"raw"``.

    * ``"llh"``: every center fits (|x| <= 180, |y| <= 90) AND, interpreted
      as (lon, lat, h) and converted to ENU about ``origin_llh``, the median
      horizontal distance to the matched GPS point is < ``llh_match_max_m``.
      (The range check alone is not enough — a small raw reconstruction also
      fits inside +-180/+-90.)
    * ``"local"``: centers treated as metric E/N/U about the same origin lie
      near the GPS track (median matched distance below
      ``max(local_match_frac * track_span, local_match_abs_m)``).
    * ``"raw"``: anything else — needs the similarity fit.
    """
    shared = sorted(set(cam_centers) & set(gps_enu))
    if not shared:
        raise ValueError("no shared frame names between the camera model and GPS")

    xs = [cam_centers[k][0] for k in shared]
    ys = [cam_centers[k][1] for k in shared]

    if all(abs(x) <= 180.0 for x in xs) and all(abs(y) <= 90.0 for y in ys):
        dists = []
        for k in shared:
            cx, cy, cz = cam_centers[k]
            h = cz if math.isfinite(cz) else origin_llh[2]
            e, n_, _u = _llh_to_enu(cy, cx, h, origin_llh)
            ge, gn, _gu = gps_enu[k]
            dists.append(math.hypot(e - ge, n_ - gn))
        if float(np.median(dists)) <= llh_match_max_m:
            return "llh"

    ge = np.array([gps_enu[k][0] for k in shared])
    gn = np.array([gps_enu[k][1] for k in shared])
    span = float(math.hypot(ge.max() - ge.min(), gn.max() - gn.min()))
    dists = [
        math.hypot(cam_centers[k][0] - gps_enu[k][0],
                   cam_centers[k][1] - gps_enu[k][1])
        for k in shared
    ]
    if float(np.median(dists)) <= max(local_match_frac * span, local_match_abs_m):
        return "local"
    return "raw"


# ---------------------------------------------------------------------------
# Frame UTC times
# ---------------------------------------------------------------------------

_TIME_NAME_CANDS = (
    "image", "label", "name", "frame", "file", "filename",
    "image_name", "file_name",
)
# Columns that explicitly mean UTC epoch seconds.
_UTC_COL_CANDS = (
    "utc_s", "utc", "time_utc", "utc_time", "timestamp_utc", "unix_s",
    "unix_time", "epoch_s",
)
# Ambiguous columns: accepted only when the values look like epoch seconds.
_AMBIG_TIME_CANDS = ("timestamp", "time", "t")

# Self-timing reconstruction stems: a camera-model->COLMAP export often names
# frames ``frame_<time>_<idx>`` (e.g. ``frame_1474.071944_0``): the float is
# the per-frame timestamp, the trailing int is the export index. The float
# must contain a decimal point so plain counters (``frame_000123``) and
# date-like stems never match.
_NAME_TIME_RE = re.compile(r"(?:^|_)(\d+\.\d+)_(\d+)$")


def frame_times_from_colmap_names(
    names: Sequence[str] | Mapping[str, object],
) -> dict[str, float]:
    """Extract per-frame times embedded in reconstruction image stems.

    Matches ``..._<float>_<idx>`` at the end of the stem (robust to a
    leading ``frame_`` and any prefix), e.g. ``frame_1474.071944_0`` ->
    ``1474.071944``. Stems that do not encode a time are skipped; returns
    ``{}`` when none match. The time base (video-relative / boottime / UTC)
    is up to the caller — see ``name_time_base`` in :func:`build_report`.
    """
    out: dict[str, float] = {}
    for name in names:
        m = _NAME_TIME_RE.search(str(name))
        if m:
            try:
                out[str(name)] = float(m.group(1))
            except ValueError:  # pragma: no cover - \d+\.\d+ always parses
                continue
    return out


def _parse_time_value(v: object) -> Optional[float]:
    """Numeric epoch seconds, or an ISO datetime string -> epoch seconds."""
    f = _to_float(v)
    if f is not None:
        return f
    s = str(v).strip()
    if not s:
        return None
    try:
        d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()


def load_frame_times(path: Path | str) -> Optional[dict[str, float]]:
    """Read per-frame UTC times from a CSV -> ``{stem: utc_s}``.

    The name column is auto-detected; the time column must be an explicit
    UTC column (``utc_s`` / ``utc`` / ``time_utc`` / ...), or an ambiguous
    ``timestamp``/``time`` column whose values look like Unix epoch seconds
    (median > 1e8) or ISO datetimes. Returns ``None`` when no usable UTC
    column is present (e.g. an ``Image, t_video_s`` file — that needs the
    session time anchor instead).
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{p}: no data rows found")
    reader = csv.DictReader(lines)
    headers = reader.fieldnames or []
    name_col = _find_col(headers, _TIME_NAME_CANDS)
    if name_col is None:
        raise ValueError(f"{p}: no image/name column detected in {headers}")
    time_col = _find_col(headers, _UTC_COL_CANDS)
    ambiguous = False
    if time_col is None:
        time_col = _find_col(headers, _AMBIG_TIME_CANDS)
        ambiguous = True
    if time_col is None:
        return None
    out: dict[str, float] = {}
    for row in reader:
        raw_name = (row.get(name_col) or "").strip()
        if not raw_name:
            continue
        t = _parse_time_value(row.get(time_col))
        if t is None:
            continue
        key = normalize_image_key(raw_name)
        if key:
            out[key] = t
    if not out:
        return None
    if ambiguous:
        med = float(np.median(list(out.values())))
        if med < 1e8:
            logger.warning(
                "%s: column %r values look video-relative (median %.3f), "
                "not UTC epoch seconds -- ignored. Provide the session so "
                "the time anchor can convert them.", p, time_col, med,
            )
            return None
    return out


def _discover_session_timing(
    session: Path,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Locate the timing files inside a raw session directory.

    Returns ``(recording_map, measurements, capture_meta, video_anchor)``
    (each ``None`` when absent).
    """
    def _pick(pattern: str, exclude: tuple[str, ...] = ()) -> Optional[Path]:
        cands = [
            c for c in sorted(session.glob(pattern))
            if not any(c.match(x) for x in exclude)
        ]
        return cands[0] if cands else None

    recording_map = _pick("recording_*.txt", exclude=("*.video_anchor.txt",))
    measurements = _pick("measurements_*.txt")
    capture_meta = _pick("capture_meta.json")
    video_anchor = _pick("recording_*.video_anchor.txt") or _pick("video_anchor.txt")
    return recording_map, measurements, capture_meta, video_anchor


def frame_times_via_session(
    frame_times_csv: Path | str,
    session_dir: Path | str,
    *,
    pos_rows: Optional[list[PosRow]] = None,
    log: Optional[LogFn] = None,
) -> dict[str, float]:
    """Frame UTC times via the pipeline time anchor.

    Reuses :func:`data_pipeline.stages.georef._load_frames` with the timing
    files discovered inside ``session_dir`` (``recording_*.txt``,
    ``measurements_*.txt``, ``capture_meta.json``,
    ``recording_*.video_anchor.txt``).
    """
    from .stages.georef import _load_frames

    log_ = log or (lambda s: None)
    session = Path(session_dir)
    ft_csv = Path(frame_times_csv)

    recording_map, measurements, capture_meta, video_anchor = (
        _discover_session_timing(session))
    if recording_map is None and measurements is None:
        raise FileNotFoundError(
            f"{session}: no recording_*.txt or measurements_*.txt found -- "
            "cannot build the boot->UTC time anchor"
        )
    frames, _anchor = _load_frames(
        ft_csv,
        recording_map or (session / "__no_recording_map__.txt"),
        log_,
        pos_rows=pos_rows,
        capture_meta=capture_meta,
        video_anchor=video_anchor,
        measurements_txt=measurements,
    )
    out: dict[str, float] = {}
    for f in frames:
        key = normalize_image_key(f.image)
        if key:
            out[key] = f.utc_s
    if not out:
        raise ValueError(f"{ft_csv}: no frame times resolved via the session anchor")
    return out


def _name_times_to_utc(
    name_times: Mapping[str, float],
    *,
    name_time_base: str,
    session_dir: Optional[Path],
    gps_pos_rows: Optional[list[PosRow]],
    log: LogFn,
) -> Optional[tuple[dict[str, float], str]]:
    """Convert stem-embedded frame times to UTC per ``name_time_base``.

    * ``"utc"``   — the embedded values already are absolute UTC seconds.
    * ``"video"`` — video-relative seconds: lifted through the session
      boot->UTC anchor with the SAME t0 resolution the pipeline uses
      (``capture_meta.video_t0_boottime_ns``, else the ``video_anchor.txt``
      min bootNs; a chop session's own anchor file IS that file).
    * ``"boot"``  — absolute ``CLOCK_BOOTTIME`` seconds: mapped straight
      through ``anchor.boottime_to_utc_s``.

    Returns ``(times, source_label)``, or ``None`` when conversion is not
    possible (video/boot base without a usable session anchor) so the
    caller can fall through to other tiers.
    """
    if name_time_base == "utc":
        log(f"[times] using embedded frame-name times as UTC "
            f"({len(name_times)} frames)")
        return dict(name_times), "embedded frame-name times (utc base)"

    if session_dir is None:
        log(f"[times] frame names encode times (base={name_time_base!r}) "
            "but no session dir was given for the boot->UTC anchor -- "
            "name times skipped")
        return None
    session = Path(session_dir)
    recording_map, measurements, capture_meta, video_anchor = (
        _discover_session_timing(session))
    if recording_map is None and measurements is None:
        log(f"[times] {session}: no recording_*.txt or measurements_*.txt "
            "-- cannot anchor the embedded frame-name times")
        return None
    try:
        from .time_sync import fit_time_anchor_with_fallback
        anchor, _src = fit_time_anchor_with_fallback(
            recording_map or (session / "__no_recording_map__.txt"),
            measurements,
            pos_rows=gps_pos_rows,
        )
        if name_time_base == "boot":
            out = {k: float(anchor.boottime_to_utc_s(t * 1e9))
                   for k, t in name_times.items()}
            label = "embedded frame-name times (boot base, session anchor)"
        else:  # "video"
            from .frame_time import make_frame_to_utc, resolve_video_t0_boottime_ns
            t0 = resolve_video_t0_boottime_ns(
                capture_meta=capture_meta, video_anchor=video_anchor, log=log)
            to_utc = make_frame_to_utc(anchor, t0)
            out = {k: float(to_utc(t)) for k, t in name_times.items()}
            label = "embedded frame-name times (video base, session anchor)"
    except Exception as e:
        log(f"[times] embedded frame-name times could not be anchored "
            f"({e}) -- skipped")
        return None
    log(f"[times] using embedded frame-name times "
        f"(base={name_time_base}, {len(out)} frames)")
    return out, label


def recover_frame_times_from_pos(
    gps_llh: FrameLLH,
    pos_rows: list[PosRow],
    *,
    max_match_dist_m: float = 25.0,
) -> tuple[dict[str, float], dict]:
    """Recover each frame's UTC by projecting its GPS position onto the GPS
    ``.pos`` track (the Georef positions were interpolated FROM that track,
    so each frame position lies on it and its along-track parameter recovers
    the frame time).

    Returns ``({stem: utc_s}, stats)`` where stats carries the match count
    and the median projection distance (a large median means the supplied
    ``.pos`` is not the one the Georef.csv was built from).
    """
    if len(pos_rows) < 2:
        raise ValueError("need >=2 .pos epochs to recover frame times")
    origin = (pos_rows[0].lat_deg, pos_rows[0].lon_deg, pos_rows[0].h_m)
    pe = np.empty(len(pos_rows))
    pn = np.empty(len(pos_rows))
    for i, r in enumerate(pos_rows):
        e, n_, _u = _llh_to_enu(r.lat_deg, r.lon_deg, r.h_m, origin)
        pe[i] = e
        pn[i] = n_
    pt = np.array([r.utc_s for r in pos_rows])

    out: dict[str, float] = {}
    dists: list[float] = []
    for stem, (lat, lon, h) in gps_llh.items():
        fe, fn, _fu = _llh_to_enu(lat, lon, h if h is not None else origin[2], origin)
        d2 = (pe - fe) ** 2 + (pn - fn) ** 2
        i = int(np.argmin(d2))
        best_d = math.sqrt(float(d2[i]))
        best_t = float(pt[i])
        for a, b in ((i - 1, i), (i, i + 1)):
            if a < 0 or b >= len(pos_rows):
                continue
            ax, ay = pe[a], pn[a]
            bx, by = pe[b], pn[b]
            seg2 = (bx - ax) ** 2 + (by - ay) ** 2
            if seg2 < 1e-12:
                continue
            u = ((fe - ax) * (bx - ax) + (fn - ay) * (by - ay)) / seg2
            u = min(1.0, max(0.0, float(u)))
            qx, qy = ax + u * (bx - ax), ay + u * (by - ay)
            d = math.hypot(fe - qx, fn - qy)
            if d < best_d:
                best_d = d
                best_t = float(pt[a] + u * (pt[b] - pt[a]))
        dists.append(best_d)
        if best_d <= max_match_dist_m:
            out[stem] = best_t
    stats = {
        "n_frames": len(gps_llh),
        "n_matched": len(out),
        "median_proj_dist_m": float(np.median(dists)) if dists else float("nan"),
    }
    return out, stats


# ---------------------------------------------------------------------------
# Sigma-band aggregation
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an ascending list (0..100)."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (n - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 >= n:
        return float(sorted_vals[-1])
    return float(sorted_vals[i] + (sorted_vals[i + 1] - sorted_vals[i]) * frac)


def sigma_bands(errors: Sequence[float]) -> dict:
    """Aggregate a (signed) error sample the project way.

    Returns ``n``, ``mean`` (of the signed values), ``mean_abs``, ``std``
    (classic population std of the signed values), ``rms``, ``sigma1`` /
    ``sigma2`` / ``sigma3`` (the 68.27 / 95.45 / 99.73 percentiles of the
    |error| distribution) and ``max_abs``. Non-finite inputs are dropped.
    """
    vals = [float(v) for v in errors if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "mean_abs": nan, "std": nan, "rms": nan,
                "sigma1": nan, "sigma2": nan, "sigma3": nan, "max_abs": nan}
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    rms = math.sqrt(sum(v * v for v in vals) / n)
    abs_sorted = sorted(abs(v) for v in vals)
    s1, s2, s3 = (_percentile(abs_sorted, p) for p in _SIGMA_PCTS)
    return {
        "n": n,
        "mean": mean,
        "mean_abs": sum(abs_sorted) / n,
        "std": std,
        "rms": rms,
        "sigma1": s1,
        "sigma2": s2,
        "sigma3": s3,
        "max_abs": abs_sorted[-1],
    }


def _wrap_deg(d: float) -> float:
    """Wrap an angle difference to (-180, 180]."""
    w = (d + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


# ---------------------------------------------------------------------------
# Per-frame records + comparison
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """One frame with all three sources in the common ENU frame.

    ``*_d*`` fields are (source - ground truth). Motion-derived fields
    (speeds, azimuths and their errors) are NaN on the first frame of a
    run or across a time gap larger than the step limit.
    """

    name: str
    utc_s: float
    gt_lat: float
    gt_lon: float
    gt_h: float
    cam_e: float
    cam_n: float
    cam_u: float
    gps_e: float
    gps_n: float
    gps_u: float
    gt_e: float
    gt_n: float
    gt_u: float
    cam_de: float = float("nan")
    cam_dn: float = float("nan")
    cam_du: float = float("nan")
    gps_de: float = float("nan")
    gps_dn: float = float("nan")
    gps_du: float = float("nan")
    cam_horiz_err_m: float = float("nan")
    gps_horiz_err_m: float = float("nan")
    cam_speed_mps: float = float("nan")
    gps_speed_mps: float = float("nan")
    gt_speed_mps: float = float("nan")
    cam_speed_err_mps: float = float("nan")
    gps_speed_err_mps: float = float("nan")
    # Speed error as a percentage of the ground-truth speed:
    # 100 * |source - gt| / max(gt, speed_floor).
    cam_speed_pct_err: float = float("nan")
    gps_speed_pct_err: float = float("nan")
    cam_vel3d_err_mps: float = float("nan")
    gps_vel3d_err_mps: float = float("nan")
    cam_az_deg: float = float("nan")
    gps_az_deg: float = float("nan")
    gt_az_deg: float = float("nan")
    cam_az_err_deg: float = float("nan")
    gps_az_err_deg: float = float("nan")
    # Method-vs-method heading: camera azimuth minus GPS azimuth (wrapped),
    # defined only when both bearings exist and GPS is actually moving.
    cam_vs_gps_az_err_deg: float = float("nan")


@dataclass
class Result:
    """Everything :func:`build_report` produced."""

    records: list[FrameRecord]
    summary: dict
    mode: str  # "llh" | "local" | "raw"
    fit: Optional[SimilarityFit]
    csv_path: Optional[Path]
    html_path: Optional[Path]
    verdict: str


def _pos_velocity_enu(
    rows: list[PosRow],
    times: list[float],
    utc_s: float,
    max_gap_s: float,
) -> Optional[tuple[float, float, float]]:
    """.pos-native (Doppler-grade) velocity at ``utc_s`` as an ENU
    ``(ve, vn, vu)`` tuple, or ``None`` when the epoch is outside the data
    or the ``.pos`` carries no velocity columns (NaN components)."""
    got = interp_pos_with_velocity(rows, utc_s, max_gap_s, times=times)
    if got is None:
        return None
    vn, ve, vu = got[3], got[4], got[5]
    if not (math.isfinite(vn) and math.isfinite(ve) and math.isfinite(vu)):
        return None
    return (ve, vn, vu)


def _motion_fields(
    recs: list[FrameRecord],
    *,
    speed_floor_mps: float,
    max_step_s: float,
    gps_pos_rows: Optional[list[PosRow]] = None,
    gt_pos_rows: Optional[list[PosRow]] = None,
    max_gap_s: float = 2.0,
) -> None:
    """Fill speed / azimuth fields (in place). ``recs`` must be sorted by
    ``utc_s``.

    The camera source always uses position finite-differences between
    consecutive frames. The GPS and ground-truth sources use the
    ``.pos``-native (Doppler-grade) 3D velocity interpolated from
    ``gps_pos_rows`` / ``gt_pos_rows`` when available, falling back to
    finite-differences otherwise (e.g. GPS positions from a Georef.csv, or
    a ``.pos`` without velocity columns).

    Azimuth errors are only produced where the ground-truth horizontal
    speed clears ``speed_floor_mps`` (a near-stationary motion vector has
    no meaningful bearing); the camera-vs-GPS heading difference is gated
    on the GPS horizontal speed the same way.
    """
    doppler_rows = {"gps": gps_pos_rows, "gt": gt_pos_rows}
    doppler_times = {
        src: ([r.utc_s for r in rows] if rows else None)
        for src, rows in doppler_rows.items()
    }
    for i, b in enumerate(recs):
        # Finite-difference velocities from the previous frame (m/s).
        fd: dict[str, tuple[float, float, float]] = {}
        fd_bearing_ok: dict[str, bool] = {}
        if i > 0:
            a = recs[i - 1]
            dt_s = b.utc_s - a.utc_s
            if 0.0 < dt_s <= max_step_s:
                for src in ("cam", "gps", "gt"):
                    de = getattr(b, f"{src}_e") - getattr(a, f"{src}_e")
                    dn = getattr(b, f"{src}_n") - getattr(a, f"{src}_n")
                    du = getattr(b, f"{src}_u") - getattr(a, f"{src}_u")
                    if not math.isfinite(du):
                        du = 0.0
                    fd[src] = (de / dt_s, dn / dt_s, du / dt_s)
                    fd_bearing_ok[src] = math.hypot(de, dn) >= _MIN_BEARING_DISP_M

        # Per-source ENU velocity: .pos-native for gps/gt when available.
        vel: dict[str, Optional[tuple[float, float, float]]] = {}
        for src in ("cam", "gps", "gt"):
            v: Optional[tuple[float, float, float]] = None
            bearing_ok = False
            rows = doppler_rows.get(src)
            if rows:
                v = _pos_velocity_enu(rows, doppler_times[src], b.utc_s,
                                      max_gap_s)
                if v is not None:
                    bearing_ok = math.hypot(v[0], v[1]) >= _MIN_BEARING_SPEED_MPS
            if v is None and src in fd:
                v = fd[src]
                bearing_ok = fd_bearing_ok[src]
            vel[src] = v
            if v is None:
                continue
            ve_, vn_, vu_ = v
            setattr(b, f"{src}_speed_mps",
                    math.sqrt(ve_ * ve_ + vn_ * vn_ + vu_ * vu_))
            if bearing_ok:
                setattr(b, f"{src}_az_deg", heading_from_enu(ve_, vn_))

        gt_v = vel["gt"]
        gt_hspeed = (math.hypot(gt_v[0], gt_v[1]) if gt_v is not None
                     else float("nan"))
        for src in ("cam", "gps"):
            v = vel[src]
            if v is None or gt_v is None:
                continue
            setattr(b, f"{src}_speed_err_mps",
                    getattr(b, f"{src}_speed_mps") - b.gt_speed_mps)
            # Percentage-of-GT speed error; the speed floor guards the
            # divide when the truth is (near) stationary.
            denom = max(b.gt_speed_mps, speed_floor_mps)
            if denom > 0.0:
                setattr(b, f"{src}_speed_pct_err",
                        100.0 * abs(getattr(b, f"{src}_speed_mps")
                                    - b.gt_speed_mps) / denom)
            dv = [v[k] - gt_v[k] for k in range(3)]
            setattr(b, f"{src}_vel3d_err_mps",
                    math.sqrt(dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2))
            az = getattr(b, f"{src}_az_deg")
            if (gt_hspeed >= speed_floor_mps
                    and math.isfinite(az) and math.isfinite(b.gt_az_deg)):
                setattr(b, f"{src}_az_err_deg", _wrap_deg(az - b.gt_az_deg))

        # Method-vs-method heading (needs no ground truth).
        gps_v = vel["gps"]
        gps_hspeed = (math.hypot(gps_v[0], gps_v[1]) if gps_v is not None
                      else float("nan"))
        if (gps_hspeed >= speed_floor_mps
                and math.isfinite(b.cam_az_deg)
                and math.isfinite(b.gps_az_deg)):
            b.cam_vs_gps_az_err_deg = _wrap_deg(b.cam_az_deg - b.gps_az_deg)


def _aggregate(recs: list[FrameRecord]) -> dict:
    def col(attr: str) -> list[float]:
        return [getattr(r, attr) for r in recs]

    return {
        "camera": {
            "horiz_m": sigma_bands(col("cam_horiz_err_m")),
            "azimuth_deg": sigma_bands(col("cam_az_err_deg")),
            "speed_mps": sigma_bands(col("cam_speed_err_mps")),
            "speed_pct": sigma_bands(col("cam_speed_pct_err")),
            "vel3d_mps": sigma_bands(col("cam_vel3d_err_mps")),
        },
        "gps": {
            "horiz_m": sigma_bands(col("gps_horiz_err_m")),
            "azimuth_deg": sigma_bands(col("gps_az_err_deg")),
            "speed_mps": sigma_bands(col("gps_speed_err_mps")),
            "speed_pct": sigma_bands(col("gps_speed_pct_err")),
            "vel3d_mps": sigma_bands(col("gps_vel3d_err_mps")),
        },
        "cam_vs_gps": {
            "azimuth_deg": sigma_bands(col("cam_vs_gps_az_err_deg")),
        },
    }


def _build_verdict(agg: dict) -> tuple[dict, str]:
    """Per-metric winner + a one-line overall verdict string."""
    cam, gps = agg["camera"], agg["gps"]

    def _pick(metric: str, key: str) -> dict:
        cv = cam[metric][key]
        gv = gps[metric][key]
        if not (math.isfinite(cv) and math.isfinite(gv)):
            winner = "n/a"
        elif cv < gv:
            winner = "camera model"
        elif gv < cv:
            winner = "GPS"
        else:
            winner = "tie"
        return {"winner": winner, "camera": cv, "gps": gv}

    detail = {
        "horizontal_2sigma_m": _pick("horiz_m", "sigma2"),
        "speed_2sigma_mps": _pick("speed_mps", "sigma2"),
        "azimuth_2sigma_deg": _pick("azimuth_deg", "sigma2"),
    }
    bits = []
    for label, unit, d in (
        ("horizontal 2-sigma", "m", detail["horizontal_2sigma_m"]),
        ("3D-speed error 2-sigma", "m/s", detail["speed_2sigma_mps"]),
        ("azimuth error 2-sigma", "deg", detail["azimuth_2sigma_deg"]),
    ):
        if d["winner"] == "n/a":
            bits.append(f"{label}: n/a")
        else:
            bits.append(
                f"{label}: {d['winner']} "
                f"(camera {d['camera']:.3f} {unit} vs GPS {d['gps']:.3f} {unit})"
            )
    wins = [d["winner"] for d in detail.values() if d["winner"] in ("camera model", "GPS")]
    if not wins:
        headline = "VERDICT: not enough matched data to compare."
    else:
        n_cam = wins.count("camera model")
        n_gps = wins.count("GPS")
        if n_cam > n_gps:
            headline = "VERDICT: the camera model is closer to ground truth overall."
        elif n_gps > n_cam:
            headline = "VERDICT: the GPS track is closer to ground truth overall."
        else:
            headline = "VERDICT: mixed -- each source wins some metrics."
    line = headline + " " + "; ".join(bits) + "."
    detail["headline"] = headline
    return detail, line


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "Image", "utc_s", "gt_lat", "gt_lon", "gt_h",
    "cam_e", "cam_n", "cam_u", "gps_e", "gps_n", "gps_u",
    "gt_e", "gt_n", "gt_u",
    "cam_dE", "cam_dN", "cam_dU", "gps_dE", "gps_dN", "gps_dU",
    "cam_horiz_err_m", "gps_horiz_err_m",
    "cam_speed_mps", "gps_speed_mps", "gt_speed_mps",
    "cam_speed_err_mps", "gps_speed_err_mps",
    "cam_speed_pct_err", "gps_speed_pct_err",
    "cam_vel3d_err_mps", "gps_vel3d_err_mps",
    "cam_az_deg", "gps_az_deg", "gt_az_deg",
    "cam_az_err_deg", "gps_az_err_deg", "cam_vs_gps_az_err_deg",
]


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def write_compare_csv(records: Sequence[FrameRecord], out_path: Path | str) -> Path:
    """Write the per-frame comparison rows -> ``camera_vs_gps_vs_gt.csv``."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        for r in records:
            w.writerow([
                r.name, _fmt(r.utc_s, 3),
                _fmt(r.gt_lat, 9), _fmt(r.gt_lon, 9), _fmt(r.gt_h, 4),
                _fmt(r.cam_e), _fmt(r.cam_n), _fmt(r.cam_u),
                _fmt(r.gps_e), _fmt(r.gps_n), _fmt(r.gps_u),
                _fmt(r.gt_e), _fmt(r.gt_n), _fmt(r.gt_u),
                _fmt(r.cam_de), _fmt(r.cam_dn), _fmt(r.cam_du),
                _fmt(r.gps_de), _fmt(r.gps_dn), _fmt(r.gps_du),
                _fmt(r.cam_horiz_err_m), _fmt(r.gps_horiz_err_m),
                _fmt(r.cam_speed_mps), _fmt(r.gps_speed_mps), _fmt(r.gt_speed_mps),
                _fmt(r.cam_speed_err_mps), _fmt(r.gps_speed_err_mps),
                _fmt(r.cam_speed_pct_err, 2), _fmt(r.gps_speed_pct_err, 2),
                _fmt(r.cam_vel3d_err_mps), _fmt(r.gps_vel3d_err_mps),
                _fmt(r.cam_az_deg, 2), _fmt(r.gps_az_deg, 2), _fmt(r.gt_az_deg, 2),
                _fmt(r.cam_az_err_deg, 2), _fmt(r.gps_az_err_deg, 2),
                _fmt(r.cam_vs_gps_az_err_deg, 2),
            ])
    return p


# ---------------------------------------------------------------------------
# HTML report (plotly-inline pattern shared with analysis_report)
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0;
       background: #eef1f5; color: #1c2733; }
.wrap { max-width: 1150px; margin: 0 auto; padding: 18px; }
h1 { font-size: 1.5em; margin: 8px 0 2px; }
h2 { font-size: 1.15em; margin: 0 0 6px; color: #21344a; }
.card { background: #fff; border-radius: 10px; padding: 16px 18px;
        margin: 14px 0; box-shadow: 0 1px 4px rgba(20,40,70,.08); }
.explain { color: #5c6b7c; font-size: .9em; margin: 2px 0 10px; }
.muted-note { color: #7a8794; font-style: italic; }
.badge { display: inline-block; padding: 6px 16px; border-radius: 12px;
         font-weight: 600; font-size: 1.0em; }
.badge-good { background: #e3f5e5; color: #197a24; }
.badge-info { background: #e3edfa; color: #1c4d8f; }
.badge-warn { background: #fff3d6; color: #8a6100; }
table.kv { border-collapse: collapse; width: 100%; font-size: .9em; }
table.kv td, table.kv th { border-bottom: 1px solid #e5eaf0;
         padding: 4px 8px; text-align: left; vertical-align: top; }
.chart { width: 100%; }
.subtitle { color: #5c6b7c; font-size: .9em; margin: 0 0 8px; }
"""

_CAM_COLOR = "#1f77b4"
_GPS_COLOR = "#d62728"


def _esc(v: object) -> str:
    return _html_mod.escape(str(v))


def _jclean(seq: Sequence[float]) -> list:
    return [
        (float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None)
        for v in seq
    ]


def _fig(div_id: str, traces: list, layout: dict, height: int = 340) -> str:
    base = {
        "margin": {"l": 55, "r": 20, "t": 36, "b": 45},
        "height": height,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#fafbfc",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 12},
        "legend": {"orientation": "h", "y": -0.22},
    }
    base.update(layout)
    return (
        f'<div id="{div_id}" class="chart"></div>\n'
        f"<script>Plotly.newPlot({json.dumps(div_id)}, "
        f"{json.dumps(traces)}, {json.dumps(base)}, "
        "{responsive:true, displaylogo:false});</script>\n"
    )


def _section(title: str, body: str, explainer: str = "") -> str:
    expl = f'<p class="explain">{_esc(explainer)}</p>' if explainer else ""
    return f'<section class="card"><h2>{_esc(title)}</h2>{expl}{body}</section>\n'


def _cdf_xy(values: Sequence[float]) -> tuple[list, list]:
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return [], []
    y = [(i + 1) / len(v) for i in range(len(v))]
    return v, y


def _sigma_table_html(agg: dict) -> str:
    metrics = [
        ("horiz_m", "Horizontal error vs truth (m)"),
        ("azimuth_deg", "Azimuth error vs truth (deg)"),
        ("speed_mps", "3D speed error vs truth (m/s)"),
        ("speed_pct", "Speed error (% of GT speed)"),
        ("vel3d_mps", "3D velocity-vector error (m/s)"),
    ]
    rows = ["<table class='kv'><tr><th>Metric</th><th>Source</th><th>N</th>"
            "<th>mean</th><th>mean |e|</th><th>std</th>"
            "<th>1&sigma; (68.27%)</th><th>2&sigma; (95.45%)</th>"
            "<th>3&sigma; (99.73%)</th><th>max |e|</th></tr>"]
    def _row(label: str, name: str, b: dict) -> str:
        cells = "".join(
            f"<td>{_fmt(b[k], 3) or 'n/a'}</td>"
            for k in ("mean", "mean_abs", "std", "sigma1", "sigma2",
                      "sigma3", "max_abs")
        )
        return (f"<tr><td>{_esc(label)}</td><td>{_esc(name)}</td>"
                f"<td>{b['n']}</td>{cells}</tr>")

    for key, label in metrics:
        for src, name in (("camera", "camera model"), ("gps", "GPS")):
            rows.append(_row(label, name, agg[src][key]))
    cvg = agg.get("cam_vs_gps", {}).get("azimuth_deg")
    if cvg is not None:
        rows.append(_row("Azimuth difference: camera vs GPS (deg)",
                         "camera model vs GPS", cvg))
    rows.append("</table>")
    return "".join(rows)


def _hist_traces(cam_vals: list, gps_vals: list) -> list:
    traces = []
    for vals, name, color in (
        (cam_vals, "camera model", _CAM_COLOR),
        (gps_vals, "GPS", _GPS_COLOR),
    ):
        clean = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
        traces.append({
            "type": "histogram", "x": _jclean(clean), "name": name,
            "opacity": 0.6, "marker": {"color": color},
        })
    return traces


def build_html_report(
    recs: list[FrameRecord],
    agg: dict,
    verdict_detail: dict,
    verdict_line: str,
    *,
    mode: str,
    fit: Optional[SimilarityFit],
    inputs: Sequence[str],
    out_html: Path | str,
    time_source: str = "",
) -> Path:
    """Write the single-file HTML report (plotly inlined; opens offline)."""
    try:
        from .analysis_report import _load_plotly_js
        plotly_src = _load_plotly_js()
    except Exception:  # pragma: no cover - asset missing in stripped installs
        plotly_src = ""
        logger.warning("plotly.min.js not found -- charts omitted from the report")

    sections: list[str] = []

    # Verdict banner + sigma table.
    kind = "info"
    if "camera model" in verdict_detail.get("headline", ""):
        kind = "good"
    elif "GPS" in verdict_detail.get("headline", ""):
        kind = "good"
    sections.append(_section(
        "1 - Verdict",
        f'<p><span class="badge badge-{kind}">{_esc(verdict_line)}</span></p>',
        "Which source is closer to the survey-grade ground truth, per metric "
        "(lower 2-sigma wins).",
    ))

    # Alignment stats.
    if mode == "raw" and fit is not None:
        fit_body = (
            "<table class='kv'>"
            f"<tr><td>fitted scale</td><td>{fit.scale:.6f}</td></tr>"
            f"<tr><td>RMS residual</td><td>{_fmt(fit.rms_m, 3)} m</td></tr>"
            f"<tr><td>correspondences used</td>"
            f"<td>{fit.n_used} / {fit.n_total}"
            + (f" (rejected: {_esc(', '.join(fit.rejected[:8]))}"
               + (" ..." if len(fit.rejected) > 8 else "") + ")"
               if fit.rejected else "") + "</td></tr>"
            "</table>"
        )
        expl = (
            "The reconstruction was in an arbitrary local frame, so a "
            "7-parameter similarity (scale + rotation + translation) was "
            "fitted from the shared frame names onto the GPS positions. "
            "NOTE: this aligns the camera model TO the GPS track, so any "
            "GPS block bias is inherited by the camera model; relative "
            "shape/scatter comparisons remain meaningful."
        )
        sections.append(_section("2 - Reconstruction alignment", fit_body, expl))
    else:
        label = {"llh": "georeferenced (lon/lat/h detected)",
                 "local": "georeferenced (metric local frame detected)"}.get(mode, mode)
        sections.append(_section(
            "2 - Reconstruction alignment",
            _note_html(f"Reconstruction frame: {label} -- used directly, "
                       "no similarity fit needed."),
        ))

    sections.append(_section(
        "3 - Error table (1/2/3-sigma)",
        _sigma_table_html(agg),
        "Sigma values are the 68.27 / 95.45 / 99.73 percentiles of the "
        "|error| distribution (project convention); the classic std is "
        "also shown.",
    ))

    if plotly_src:
        cam_x, cam_y = _cdf_xy([r.cam_horiz_err_m for r in recs])
        gps_x, gps_y = _cdf_xy([r.gps_horiz_err_m for r in recs])
        cdf = _fig("cdf_horiz", [
            {"type": "scatter", "mode": "lines", "x": _jclean(cam_x),
             "y": _jclean(cam_y), "name": "camera model",
             "line": {"color": _CAM_COLOR}},
            {"type": "scatter", "mode": "lines", "x": _jclean(gps_x),
             "y": _jclean(gps_y), "name": "GPS",
             "line": {"color": _GPS_COLOR}},
        ], {
            "title": "Horizontal error vs ground truth - CDF",
            "xaxis": {"title": "error (m)"},
            "yaxis": {"title": "fraction of frames", "range": [0, 1]},
        })
        sections.append(_section("4 - Horizontal error CDF", cdf))

        az = _fig("hist_az", _hist_traces(
            [r.cam_az_err_deg for r in recs],
            [r.gps_az_err_deg for r in recs],
        ), {
            "title": "Azimuth (heading) error vs ground truth",
            "xaxis": {"title": "error (deg, wrapped +-180)"},
            "yaxis": {"title": "frames"},
            "barmode": "overlay",
        })
        sections.append(_section(
            "5 - Azimuth error distribution", az,
            "Bearing of the per-frame motion vector vs the ground-truth "
            "bearing; only frames where the truth was moving are counted.",
        ))

        sp = _fig("hist_speed", _hist_traces(
            [r.cam_speed_err_mps for r in recs],
            [r.gps_speed_err_mps for r in recs],
        ), {
            "title": "3D speed error vs ground truth",
            "xaxis": {"title": "speed error (m/s)"},
            "yaxis": {"title": "frames"},
            "barmode": "overlay",
        })
        sections.append(_section("6 - 3D speed error distribution", sp))
    else:
        sections.append(_section(
            "4 - Charts", _note_html("plotly.min.js asset unavailable -- "
                                     "charts omitted; see the CSV.")))

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sub = " | ".join(_esc(b) for b in inputs)
    if time_source:
        sub += f" | frame times: {_esc(time_source)}"
    html_doc = (
        "<!DOCTYPE html>\n<html><head>\n<meta charset=\"utf-8\">\n"
        "<title>Camera model vs GPS vs ground truth</title>\n"
        f"<style>{_CSS}</style>\n"
        + (f"<script>{plotly_src}</script>\n" if plotly_src else "")
        + "</head><body>\n<div class=\"wrap\">\n"
        "<h1>Camera model vs GPS vs ground truth</h1>\n"
        f'<p class="subtitle">{sub} &nbsp;&mdash;&nbsp; '
        f"{len(recs)} matched frames &nbsp;&mdash;&nbsp; generated {generated}</p>\n"
        + "".join(sections)
        + "</div>\n</body></html>\n"
    )
    p = Path(out_html)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_doc, encoding="utf-8")
    return p


def _note_html(text: str) -> str:
    return f'<p class="muted-note">{_esc(text)}</p>'


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def _resolve_frame_times(
    *,
    frame_times: Optional[Path],
    georef_csv: Optional[Path],
    session_dir: Optional[Path],
    gps_llh: Optional[FrameLLH],
    gps_pos_rows: Optional[list[PosRow]],
    log: LogFn,
    name_times: Optional[Mapping[str, float]] = None,
    name_time_base: str = "video",
    use_name_times: bool = False,
) -> tuple[dict[str, float], str]:
    """Frame UTC times, trying the sources in reliability order.

    1. an explicit frame-times CSV that carries UTC;
    2. a time column inside the Georef.csv itself;
    3. the session time anchor (``_load_frames``) on an
       ``Image, t_video_s`` CSV;
    4. projection of the GPS frame positions onto the GPS ``.pos`` track;
    5. times embedded in the reconstruction image names
       (``frame_<t>_<idx>``), converted per ``name_time_base``.

    With ``use_name_times=True`` tier 5 is promoted to right after the
    explicit CSV/Georef tiers; otherwise it is the automatic last resort
    when nothing else resolves.
    """
    if frame_times is not None:
        tm = load_frame_times(frame_times)
        if tm:
            return tm, f"UTC column in {Path(frame_times).name}"

    if georef_csv is not None:
        try:
            tm = load_frame_times(georef_csv)
        except ValueError:
            tm = None
        if tm:
            return tm, f"time column in {Path(georef_csv).name}"

    if use_name_times and name_times:
        got = _name_times_to_utc(
            name_times, name_time_base=name_time_base,
            session_dir=session_dir, gps_pos_rows=gps_pos_rows, log=log)
        if got is not None:
            return got

    if session_dir is not None:
        ft = frame_times
        if ft is None:
            for base in (Path(session_dir),
                         *( [Path(georef_csv).parent] if georef_csv else [] )):
                cand = base / "extracted_frame_times.csv"
                if cand.is_file():
                    ft = cand
                    break
        if ft is not None:
            tm = frame_times_via_session(
                ft, session_dir, pos_rows=gps_pos_rows, log=log)
            return tm, "session time anchor"
        log("[times] session dir given but no extracted_frame_times.csv found")

    if gps_pos_rows and gps_llh:
        tm, stats = recover_frame_times_from_pos(gps_llh, gps_pos_rows)
        if tm:
            log(f"[times] recovered {stats['n_matched']}/{stats['n_frames']} "
                f"frame times by track projection (median distance "
                f"{stats['median_proj_dist_m']:.2f} m)")
            return tm, "recovered from the GPS .pos track"

    # Last resort: the reconstruction names themselves encode a time
    # (camera-model->COLMAP ``frame_<t>_<idx>`` exports are self-timing).
    if name_times and not use_name_times:
        got = _name_times_to_utc(
            name_times, name_time_base=name_time_base,
            session_dir=session_dir, gps_pos_rows=gps_pos_rows, log=log)
        if got is not None:
            return got

    hint = ""
    if name_times:
        hint = (
            " The reconstruction names DO encode times (frame_<t>_<idx>): "
            "pass a session directory so the video/boot anchor can convert "
            "them, or name_time_base='utc' if they already are UTC seconds."
        )
    raise ValueError(
        "cannot determine per-frame UTC times: provide a frame-times CSV "
        "with a UTC column, a session directory (for the time anchor), or "
        "the GPS .pos so times can be recovered by track projection."
        + hint
    )


def build_report(
    colmap_images: Path | str,
    gt_pos: Path | str,
    out_dir: Path | str,
    *,
    georef_csv: Path | str | None = None,
    pos: Path | str | None = None,
    frame_times: Path | str | None = None,
    session_dir: Path | str | None = None,
    speed_floor_mps: float = 0.5,
    max_gap_s: float = 2.0,
    max_step_s: float = 5.0,
    force_mode: Optional[str] = None,
    write_html: bool = True,
    log: Optional[LogFn] = None,
    camera_source: str = "colmap",
    metashape: Path | str | None = None,
    name_time_base: str = "video",
    use_name_times: bool = False,
) -> Result:
    """Build the camera-model vs GPS vs ground-truth comparison.

    Inputs: the reconstruction ``images.txt``/``images.bin`` (or its
    directory), the pipeline Georef.csv (GPS per-frame positions; or a GPS
    ``.pos`` to interpolate at frame times), and the ground-truth ``.pos``.
    Emits ``camera_vs_gps_vs_gt.csv`` and a self-contained HTML report in
    ``out_dir``. Returns the :class:`Result`.

    ``camera_source`` selects the camera-side parser: ``"colmap"``
    (default; ``colmap_images`` is an ``images.txt``/``images.bin``) or
    ``"metashape"`` (a camera-model estimated-coordinates CSV). A georeferenced
    WGS84 export auto-detects as ``"llh"`` and is compared directly; a local
    or un-georeferenced chunk frame auto-detects as ``"local"``/``"raw"`` and
    is fit to the GPS track (same detection as COLMAP). ``metashape`` is an
    optional explicit path to that CSV; passing it implies
    ``camera_source="metashape"``.

    Self-timing reconstructions: COLMAP image names of the form
    ``frame_<t>_<idx>`` (a camera-model->COLMAP export) embed a per-frame time.
    ``name_time_base`` says what that time is: ``"video"`` (video-relative
    seconds, converted through the session boot->UTC anchor — needs
    ``session_dir``), ``"boot"`` (``CLOCK_BOOTTIME`` seconds via the same
    anchor) or ``"utc"`` (already absolute UTC seconds). Name times are used
    automatically when no other frame-time source resolves;
    ``use_name_times=True`` promotes them over the session-anchor /
    track-projection tiers (an explicit UTC frame-times CSV / Georef time
    column still wins).
    """
    log_: LogFn = log or (lambda s: None)
    out_dir = Path(out_dir)

    if metashape is not None:
        camera_source = "metashape"
    if camera_source not in ("colmap", "metashape"):
        raise ValueError(
            f"unknown camera_source: {camera_source!r} "
            "(expected 'colmap' or 'metashape')"
        )
    if name_time_base not in ("video", "boot", "utc"):
        raise ValueError(
            f"unknown name_time_base: {name_time_base!r} "
            "(expected 'video', 'boot' or 'utc')"
        )

    if camera_source == "metashape":
        cam_input = Path(metashape) if metashape is not None else Path(colmap_images)
        ms_llh = parse_metashape_cameras(cam_input)
        # The llh branch consumes (x=lon, y=lat, z=h) tuples; a missing
        # altitude becomes NaN so that branch falls back to the ENU origin
        # height, exactly like a COLMAP llh reconstruction would.
        cam_centers: CamCenters = {
            k: (lon, lat, h if h is not None else float("nan"))
            for k, (lat, lon, h) in ms_llh.items()
        }
        # Georeferenced WGS84 exports auto-detect as 'llh' (compared directly).
        # A local / arbitrary chunk frame (metric coords not near the GPS
        # track, e.g. an indoor or un-georeferenced reconstruction) falls
        # through to 'local'/'raw' and is fit to GPS -- so we do NOT force
        # llh. An explicit force_mode still wins in the mode dispatch below.
        log_(f"[cam] {len(cam_centers)} camera positions from {cam_input} "
             f"(frame auto-detected: llh if georeferenced, else fit to GPS)")
    else:
        cam_input = Path(colmap_images)
        cam_centers = parse_colmap_images(colmap_images)
        log_(f"[cam] {len(cam_centers)} camera centers from {colmap_images}")

    gps_pos_rows: Optional[list[PosRow]] = None
    if pos is not None:
        gps_pos_rows = parse_rtkpos(Path(pos))
        if not gps_pos_rows:
            raise ValueError(f"{pos}: no epochs parsed")

    gps_llh: Optional[dict[str, tuple[float, float, Optional[float]]]] = None
    if georef_csv is not None:
        gps_llh = load_gnss_frame_coords_from_georef(georef_csv)
        log_(f"[gps] {len(gps_llh)} frame positions from {georef_csv}")

    # Times embedded in the reconstruction image names (COLMAP only:
    # camera-model->COLMAP exports name frames ``frame_<t>_<idx>``).
    name_times: dict[str, float] = {}
    if camera_source == "colmap":
        name_times = frame_times_from_colmap_names(cam_centers)
        if name_times:
            log_(f"[times] {len(name_times)}/{len(cam_centers)} image names "
                 f"encode a frame time (base={name_time_base})")

    times, time_source = _resolve_frame_times(
        frame_times=Path(frame_times) if frame_times else None,
        georef_csv=Path(georef_csv) if georef_csv else None,
        session_dir=Path(session_dir) if session_dir else None,
        gps_llh=gps_llh,
        gps_pos_rows=gps_pos_rows,
        log=log_,
        name_times=name_times,
        name_time_base=name_time_base,
        use_name_times=use_name_times,
    )
    log_(f"[times] frame UTC source: {time_source} ({len(times)} frames)")

    if gps_llh is None:
        if gps_pos_rows is None:
            raise ValueError(
                "need --georef-csv (GPS per-frame positions) or --pos (a GPS "
                ".pos to interpolate at frame times)"
            )
        ptimes = [r.utc_s for r in gps_pos_rows]
        gps_llh = {}
        for stem, t in times.items():
            llh = interp_pos(gps_pos_rows, t, max_gap_s, times=ptimes)
            if llh is not None:
                gps_llh[stem] = llh
        log_(f"[gps] {len(gps_llh)} frame positions interpolated from {pos}")

    gt_rows = parse_rtkpos(Path(gt_pos))
    if not gt_rows:
        raise ValueError(f"{gt_pos}: no ground-truth epochs parsed")
    gt_times = [r.utc_s for r in gt_rows]
    gt_llh: dict[str, tuple[float, float, float]] = {}
    for stem, t in times.items():
        llh = interp_pos(gt_rows, t, max_gap_s, times=gt_times)
        if llh is not None:
            gt_llh[stem] = llh

    matched = sorted(
        set(cam_centers) & set(gps_llh) & set(gt_llh) & set(times),
        key=lambda k: times[k],
    )
    log_(f"[join] cam={len(cam_centers)} gps={len(gps_llh)} "
         f"gt={len(gt_llh)} matched={len(matched)}")
    if len(matched) < 2:
        raise ValueError(
            f"only {len(matched)} frames are present in all of camera model / "
            "GPS / ground truth -- need at least 2 (and >=3 for a raw-frame "
            "alignment)."
        )

    # Common local ENU frame anchored at the first matched GPS fix.
    g0 = gps_llh[matched[0]]
    origin = (g0[0], g0[1], g0[2] if g0[2] is not None else gt_llh[matched[0]][2])

    gps_enu: dict[str, tuple[float, float, float]] = {}
    gps_h_ok: dict[str, bool] = {}
    for k in matched:
        lat, lon, h = gps_llh[k]
        ok = h is not None
        gps_h_ok[k] = ok
        gps_enu[k] = _llh_to_enu(lat, lon, h if ok else gt_llh[k][2], origin)
    gt_enu = {k: _llh_to_enu(*gt_llh[k], origin) for k in matched}

    mode = force_mode or detect_reconstruction_frame(cam_centers, gps_enu, origin)
    log_(f"[cam] reconstruction frame: {mode}")

    fit: Optional[SimilarityFit] = None
    cam_enu: dict[str, tuple[float, float, float]] = {}
    if mode == "llh":
        for k in matched:
            cx, cy, cz = cam_centers[k]
            h = cz if math.isfinite(cz) else origin[2]
            cam_enu[k] = _llh_to_enu(cy, cx, h, origin)
    elif mode == "local":
        for k in matched:
            cam_enu[k] = cam_centers[k]
    elif mode == "raw":
        if len(matched) < 3:
            raise ValueError("raw-frame alignment needs >=3 shared frames")
        src = np.array([cam_centers[k] for k in matched])
        dst = np.array([gps_enu[k] for k in matched])
        fit = fit_similarity_robust(matched, src, dst)
        log_(f"[fit] similarity: scale={fit.scale:.6f} rms={fit.rms_m:.3f} m "
             f"({fit.n_used}/{fit.n_total} used)")
        moved = fit.apply(src)
        for i, k in enumerate(matched):
            cam_enu[k] = (float(moved[i, 0]), float(moved[i, 1]), float(moved[i, 2]))
    else:
        raise ValueError(f"unknown reconstruction frame mode: {mode!r}")

    recs: list[FrameRecord] = []
    for k in matched:
        ce, cn, cu = cam_enu[k]
        ge, gn, gu = gps_enu[k]
        te, tn, tu = gt_enu[k]
        r = FrameRecord(
            name=k, utc_s=times[k],
            gt_lat=gt_llh[k][0], gt_lon=gt_llh[k][1], gt_h=gt_llh[k][2],
            cam_e=ce, cam_n=cn, cam_u=cu,
            gps_e=ge, gps_n=gn, gps_u=gu,
            gt_e=te, gt_n=tn, gt_u=tu,
        )
        r.cam_de, r.cam_dn, r.cam_du = ce - te, cn - tn, cu - tu
        r.gps_de, r.gps_dn = ge - te, gn - tn
        r.gps_du = (gu - tu) if gps_h_ok[k] else float("nan")
        r.cam_horiz_err_m = math.hypot(r.cam_de, r.cam_dn)
        r.gps_horiz_err_m = math.hypot(r.gps_de, r.gps_dn)
        recs.append(r)

    _motion_fields(
        recs, speed_floor_mps=speed_floor_mps, max_step_s=max_step_s,
        gps_pos_rows=gps_pos_rows, gt_pos_rows=gt_rows, max_gap_s=max_gap_s,
    )
    agg = _aggregate(recs)
    verdict_detail, verdict_line = _build_verdict(agg)

    summary = {
        "n_camera": len(cam_centers),
        "n_gps": len(gps_llh),
        "n_gt": len(gt_llh),
        "n_matched": len(matched),
        "mode": mode,
        "camera_source": camera_source,
        "time_source": time_source,
        "origin_llh": origin,
        "aggregates": agg,
        "verdict": verdict_detail,
        "verdict_line": verdict_line,
    }
    if fit is not None:
        summary["fit"] = {
            "scale": fit.scale, "rms_m": fit.rms_m,
            "n_used": fit.n_used, "n_total": fit.n_total,
            "rejected": list(fit.rejected),
        }

    csv_path = write_compare_csv(recs, out_dir / "camera_vs_gps_vs_gt.csv")
    html_path: Optional[Path] = None
    if write_html:
        inputs = [f"camera model: {cam_input.name}",
                  f"ground truth: {Path(gt_pos).name}"]
        if georef_csv is not None:
            inputs.append(f"GPS: {Path(georef_csv).name}")
        elif pos is not None:
            inputs.append(f"GPS: {Path(pos).name}")
        html_path = build_html_report(
            recs, agg, verdict_detail, verdict_line,
            mode=mode, fit=fit, inputs=inputs,
            out_html=out_dir / "camera_vs_gps_vs_gt.html",
            time_source=time_source,
        )
    log_(f"[out] wrote {csv_path}" + (f" and {html_path}" if html_path else ""))
    log_(verdict_line)

    return Result(
        records=recs, summary=summary, mode=mode, fit=fit,
        csv_path=csv_path, html_path=html_path, verdict=verdict_line,
    )
