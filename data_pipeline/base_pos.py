"""Reference input coordinate helpers.

Convert any user-friendly input (LLH, Grid, Cartesian XYZ, Interchange-format header) into a
canonical Cartesian XYZ (X, Y, Z) tuple for `stages.ppk.write_patched_config`.

Used by:
  - GUI reference input-position panel (Read-from-Interchange-format button)
  - scripts/run_pipeline_from_raw.py --base-llh / --base-from-interchange-format
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from .geo import llh_to_ecef


_APPROX_POS_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+APPROX POSITION XYZ", re.I,
)


def read_rinex_approx_xyz(obs_path: Path) -> Optional[tuple[float, float, float]]:
    """Read APPROX POSITION XYZ from a Interchange-format 3 .obs header.

    Returns (X, Y, Z) in metres, or None when the line is missing OR the
    coordinates are all zero (some converters emit placeholder zeros for
    unknown positions).
    """
    p = Path(obs_path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _APPROX_POS_RE.match(line)
                if m:
                    try:
                        x = float(m.group(1)); y = float(m.group(2)); z = float(m.group(3))
                    except ValueError:
                        return None
                    if abs(x) < 1.0 and abs(y) < 1.0 and abs(z) < 1.0:
                        return None  # placeholder zeros
                    return (x, y, z)
                if "END OF HEADER" in line:
                    break
    except OSError:
        return None
    return None


def base_xyz_from_llh(lat_deg: float, lon_deg: float, h_m: float = 0.0) -> tuple[float, float, float]:
    """Decimal-degree LLH to Cartesian XYZ XYZ (m)."""
    x, y, z = llh_to_ecef(lat_deg, lon_deg, h_m)
    return (float(x), float(y), float(z))


def parse_dms(s: str) -> float:
    """Parse degrees-minutes-seconds with optional hemisphere letter.

    Accepts forms:
      "47 07 24.4 N"
      "47:07:24.4N"
      "47d07'24.4\"N"
      "-122 39 15.5"
      "122 39 15.5 W"
    """
    s = s.strip()
    if not s:
        raise ValueError("empty DMS string")
    hemi = ""
    if s[-1].upper() in {"N", "S", "E", "W"}:
        hemi = s[-1].upper()
        s = s[:-1].strip()
    parts = re.split(r"[^\d.\-]+", s)
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"could not split DMS: {s!r}")
    try:
        d = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError as e:
        raise ValueError(f"DMS parse: {e}") from e
    sign = -1.0 if d < 0 else 1.0
    val = sign * (abs(d) + m / 60.0 + sec / 3600.0)
    # When a hemisphere letter is present, it is authoritative: a typed
    # minus sign on the degrees field does not override 'N' or 'E'.
    if hemi in {"S", "W"}:
        val = -abs(val)
    elif hemi in {"N", "E"}:
        val = abs(val)
    return val


def base_xyz_from_utm(
    easting_m: float, northing_m: float, h_m: float, zone_str: str,
) -> tuple[float, float, float]:
    """Grid (zone + letter) to Cartesian XYZ XYZ via pyproj.

    ``zone_str`` is The external solver-style like '10T' / '32U'; letter selects N/S
    hemisphere (>=N is northern).
    """
    try:
        import pyproj
    except ImportError as e:
        raise ImportError(
            "UTM conversion needs pyproj. Run: pip install pyproj>=3.4.0"
        ) from e
    m = re.match(r"^\s*(\d{1,2})\s*([A-Za-z]?)\s*$", zone_str)
    if not m:
        raise ValueError(
            f"invalid UTM zone: {zone_str!r}. "
            "Expected '<1-60><C-X>' e.g. '10T' or '32U'."
        )
    zone = int(m.group(1))
    if not (1 <= zone <= 60):
        raise ValueError(
            f"UTM zone must be 1..60, got {zone} from {zone_str!r}."
        )
    letter = m.group(2).upper() or "N"
    if letter in {"A", "B", "I", "O", "Y", "Z"}:
        raise ValueError(
            f"UTM band letter {letter!r} not allowed (reserved or polar). "
            "Valid bands are C..X excluding I and O."
        )
    northern = letter >= "N"
    epsg = (32600 if northern else 32700) + zone
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(easting_m, northing_m)
    return base_xyz_from_llh(lat, lon, h_m)


def parse_base_spec(spec: str) -> Optional[tuple[float, float, float]]:
    """Parse a free-form base position spec into Cartesian XYZ (X, Y, Z) metres.

    Supported forms (auto-detected):
      - "X,Y,Z" three signed metres (range checked: |val| > 100 km)
      - "lat,lon,h" lat/lon decimal degrees, height metres
      - "ecef:X,Y,Z"
      - "llh:lat,lon,h"
      - "rinex:/path/to/base.obs"

    Returns None if input doesn't parse cleanly.
    """
    import math as _math
    s = spec.strip()
    if not s:
        return None

    def _parse3(sub: str) -> Optional[tuple[float, float, float]]:
        try:
            parts = [float(p) for p in sub.split(",")]
        except ValueError:
            return None
        if len(parts) != 3:
            return None
        if not all(_math.isfinite(v) for v in parts):
            return None
        return (parts[0], parts[1], parts[2])

    if s.lower().startswith("rinex:"):
        return read_rinex_approx_xyz(Path(s.split(":", 1)[1]))
    if s.lower().startswith("ecef:"):
        return _parse3(s.split(":", 1)[1])
    if s.lower().startswith("llh:"):
        triple = _parse3(s.split(":", 1)[1])
        if triple is None:
            return None
        return base_xyz_from_llh(*triple)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if not all(_math.isfinite(v) for v in vals):
        return None
    # Cartesian XYZ magnitudes are on the order of 5-6 million metres; LLH lat/lon
    # are bounded to [-180, 180].
    if all(abs(v) > 1.0e5 for v in vals):
        return (vals[0], vals[1], vals[2])
    if abs(vals[0]) <= 90.0 and abs(vals[1]) <= 180.0:
        return base_xyz_from_llh(vals[0], vals[1], vals[2])
    return None
