"""Coordinate input parsing for the Post-processing reference input.

The Post-processing GUI lets the user type a base position in one of four formats:

* ``dd``   -- decimal degrees: ``47.123456 -122.654321 100.0``
* ``dms``  -- degrees / minutes / seconds with hemisphere:
              ``47 07 24.40 N    122 39 15.50 W    100.0``
* ``grid``  -- Grid zone + easting + northing + height:
              ``10 N 545678.20 5219012.50 100.0``
              (zone letter is also accepted; only the hemisphere matters)
* ``cartesian XYZ`` -- the sphere-centred the sphere-fixed XYZ in metres:
              ``-2516000.0 -4665000.0 3473000.0``

Each parser returns Cartesian XYZ metres so the Post-processing config patcher has one job to
do (``ant2-pos*`` always written as Cartesian XYZ). ``parse_*`` raises
:class:`ValueError` with a human-readable message on bad input.
"""

from __future__ import annotations

import math
import re
from typing import Tuple

from .geo import llh_to_ecef


Xyz = Tuple[float, float, float]


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _split_numbers(text: str) -> list[float]:
    """Pull all numeric tokens from ``text`` and return them as floats."""
    return [float(m.group(0)) for m in _NUMBER_RE.finditer(text)]


def parse_dd(text: str) -> Xyz:
    """Decimal-degree LLH ``lat lon h`` -> Cartesian XYZ metres."""
    nums = _split_numbers(text)
    if len(nums) < 3:
        raise ValueError("decimal-degree input needs three numbers: lat lon h")
    lat, lon, h = nums[0], nums[1], nums[2]
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude {lat} out of range [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"longitude {lon} out of range [-180, 180]")
    return llh_to_ecef(lat, lon, h)


_HEMI_RE = re.compile(r"[NnSsEeWw]")


def _parse_dms_component(text: str, axis: str) -> float:
    """Parse one DMS component (e.g. ``47°07'24.4"N``) into signed degrees."""
    text = text.strip()
    hemi_match = _HEMI_RE.search(text)
    hemi = hemi_match.group(0).upper() if hemi_match else ""
    nums = _split_numbers(text)
    if len(nums) < 1:
        raise ValueError(f"{axis} component empty")
    d = nums[0]
    m = nums[1] if len(nums) > 1 else 0.0
    s = nums[2] if len(nums) > 2 else 0.0
    sign = -1.0 if d < 0 else 1.0
    val = sign * (abs(d) + m / 60.0 + s / 3600.0)
    if axis == "lat":
        if hemi in ("S",):
            val = -abs(val)
        elif hemi in ("N",):
            val = abs(val)
        if not (-90.0 <= val <= 90.0):
            raise ValueError(f"latitude {val} out of range [-90, 90]")
    elif axis == "lon":
        if hemi in ("W",):
            val = -abs(val)
        elif hemi in ("E",):
            val = abs(val)
        if not (-180.0 <= val <= 180.0):
            raise ValueError(f"longitude {val} out of range [-180, 180]")
    return val


def parse_dms(lat_text: str, lon_text: str, h_text: str) -> Xyz:
    """DMS lat/lon + decimal-metre height -> Cartesian XYZ metres.

    Each component string accepts any mix of separators -- degree symbol,
    apostrophes, colons, spaces -- and an optional N/S/E/W suffix.
    """
    lat = _parse_dms_component(lat_text, "lat")
    lon = _parse_dms_component(lon_text, "lon")
    try:
        h = float(h_text.strip())
    except ValueError as e:
        raise ValueError(f"height '{h_text}' is not numeric") from e
    return llh_to_ecef(lat, lon, h)


_UTM_ZONE_LETTER_NORTH = "NPQRSTUVWX"
_UTM_ZONE_LETTER_SOUTH = "CDEFGHJKLM"


def parse_utm(zone_text: str, e_text: str, n_text: str, h_text: str) -> Xyz:
    """Grid (zone + easting + northing + ellipsoidal height) -> Cartesian XYZ metres.

    ``zone_text`` is either ``"10"`` plus a separate hemisphere flag, or a
    combined token like ``"10N"``, ``"10T"`` (letter -> hemisphere lookup),
    or ``"10 N"``. The function refuses ambiguous input (a bare ``"10"``
    with no hemisphere) and raises ``ValueError``.
    """
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ValueError(
            "UTM input requires pyproj; install it (already a rasterio "
            "dependency, or `pip install pyproj`)"
        ) from e

    zt = zone_text.strip().upper().replace(" ", "")
    match = re.match(r"^(\d{1,2})([A-Z]?)$", zt)
    if not match:
        raise ValueError(
            f"UTM zone '{zone_text}' must look like '10', '10N', or '10T'"
        )
    zone = int(match.group(1))
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone {zone} out of range [1, 60]")
    letter = match.group(2)
    if not letter:
        raise ValueError(
            f"UTM zone '{zone_text}' missing hemisphere letter (N..X = north, "
            "C..M = south). Append e.g. 'N' or 'S'."
        )
    if letter == "N":
        is_north = True
    elif letter == "S":
        is_north = False
    elif letter in _UTM_ZONE_LETTER_NORTH:
        is_north = True
    elif letter in _UTM_ZONE_LETTER_SOUTH:
        is_north = False
    else:
        raise ValueError(f"UTM hemisphere letter '{letter}' not recognised")

    try:
        easting = float(e_text.strip())
        northing = float(n_text.strip())
        h = float(h_text.strip())
    except ValueError as e:
        raise ValueError("UTM easting/northing/height must be numeric") from e

    proj_crs = (
        f"+proj=utm +zone={zone} "
        f"+{'north' if is_north else 'south'} +datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("pyproj returned non-finite lat/lon for UTM input")
    return llh_to_ecef(lat, lon, h)


def parse_ecef(text: str) -> Xyz:
    """Cartesian XYZ XYZ (metres) -> Cartesian XYZ metres (identity, with validation)."""
    nums = _split_numbers(text)
    if len(nums) < 3:
        raise ValueError("ECEF input needs three numbers: X Y Z (metres)")
    x, y, z = nums[0], nums[1], nums[2]
    r = math.sqrt(x * x + y * y + z * z)
    if r < 6_000_000.0 or r > 7_000_000.0:
        raise ValueError(
            f"ECEF magnitude {r:,.1f} m is not on Earth's surface "
            "(expected ~6.378e6 m). Check axis order and units."
        )
    return x, y, z
