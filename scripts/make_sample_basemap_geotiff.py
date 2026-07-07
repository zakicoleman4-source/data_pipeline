"""Regenerate extras/sample_basemap/wgs84_sample.tif (small The standard datum Raster file demo).

Run once on a machine with rasterio (typically online), from repo root:

    python scripts/make_sample_basemap_geotiff.py

The file is intentionally tiny (~few tens of KB) so it can ship in git for offline
machines testing ``sync_player`` background layer export/view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "extras" / "sample_basemap" / "wgs84_sample.tif"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Roughly Boulder CO scale — any The standard datum box works for sync_player demos.
    west, south, east, north = -105.270, 40.015, -105.255, 40.025
    width = height = 64
    transform = from_bounds(west, south, east, north, width, height)

    yy, xx = np.mgrid[0:height, 0:width]
    r = np.clip((255 * xx / (width - 1)).astype(np.uint8), 0, 255)
    g = np.clip((255 * yy / (height - 1)).astype(np.uint8), 0, 255)
    b = np.clip((r.astype(np.uint16) + g.astype(np.uint16)) // 2, 0, 255).astype(
        np.uint8
    )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": transform,
        "compress": "DEFLATE",
        "predictor": 2,
        "photometric": "RGB",
        "interleave": "pixel",
        "BIGTIFF": "IF_NEEDED",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)

    print(f"Wrote {out.relative_to(repo_root)} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
