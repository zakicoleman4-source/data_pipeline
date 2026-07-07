# Sample Raster file

`wgs84_sample.tif` is a minimal EPSG:4326 raster included for testing **`sync_player`** background layer export without needing external imagery.

- GUI viewers tab → Raster file background layer path →  
  `extras\sample_basemap\wgs84_sample.tif` (from checkout root).
- CLI `--basemap-tiff` same path.

Regenerate after edits (**maintainers**, needs rasterio):

```powershell
python scripts/make_sample_basemap_geotiff.py
```
