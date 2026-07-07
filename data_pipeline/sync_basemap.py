"""Optional Raster file → The standard datum PNG + bounds for embed in ``sync_player.html`` (offline Plotly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BasemapExport:
    """Paths and The standard datum geographic bounds (degrees) for Plotly ``layout.images``."""

    png_path: Path
    west: float
    south: float
    east: float
    north: float


def export_geotiff_basemap_wgs84(
    geotiff: Path,
    png_out: Path,
    *,
    max_dim: int = 2048,
) -> BasemapExport:
    """Warp ``raster file`` near‑EPSG:4326 bounds, rasterise at most ``max_dim`` cells, PNG.

    Requires **rasterio** and **numpy** (``pip install rasterio`` bundles numpy).

    Bands: uses first three bands as RGB when available; otherwise repeats the first.
    Cell values are robust‑scaled per band (p2–p98) to ``uint8`` for display.
    """
    geotiff = geotiff.resolve()
    png_out = png_out.resolve()
    if not geotiff.is_file():
        raise FileNotFoundError(geotiff)

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject, transform_bounds
    except ImportError as e:
        raise RuntimeError(
            "GeoTIFF basemap requires the 'rasterio' package "
            "(``pip install rasterio``). Underlying error: "
            f"{e}"
        ) from e

    png_out.parent.mkdir(parents=True, exist_ok=True)
    dst_crs = rasterio.crs.CRS.from_epsg(4326)

    with rasterio.open(geotiff) as src:
        west, south, east, north = transform_bounds(
            src.crs, dst_crs, *src.bounds, densify_pts=21
        )
        xspan = east - west
        yspan = north - south
        if xspan <= 0 or yspan <= 0:
            raise ValueError(f"Invalid geographic span from {geotiff}")

        # Target cell grid (preserve aspect ratio, cap longest side).
        md = max(64, min(int(max_dim), 8192))
        aspect = xspan / yspan
        if aspect >= 1.0:
            out_w = md
            out_h = max(1, int(round(out_w / aspect)))
        else:
            out_h = md
            out_w = max(1, int(round(out_h * aspect)))

        transform = from_bounds(west, south, east, north, out_w, out_h)

        nb = min(3, src.count)
        stacks: list[Any] = []
        for band_i in range(1, nb + 1):
            dst_b = np.empty((out_h, out_w), dtype=np.float64)
            reproject(
                source=rasterio.band(src, band_i),
                destination=dst_b,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            stacks.append(dst_b)

        while len(stacks) < 3:
            stacks.append(stacks[-1])

        rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for c, arr in enumerate(stacks[:3]):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            lo, hi = np.percentile(finite, (2.0, 98.0))
            if hi <= lo:
                lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
            scaled = np.where(np.isfinite(arr), scaled, 0.0)
            rgb[:, :, c] = scaled.astype(np.uint8)

    profile = {
        "driver": "PNG",
        "width": out_w,
        "height": out_h,
        "count": 3,
        "dtype": "uint8",
    }
    with rasterio.open(png_out, "w", **profile) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)

    return BasemapExport(
        png_path=png_out,
        west=float(west),
        south=float(south),
        east=float(east),
        north=float(north),
    )
