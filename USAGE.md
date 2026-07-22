# Accuracy Report — Usage

Compare one or more camera-model reconstructions against a survey-grade
ground-truth track, frame by frame. Produces a single self-contained HTML
report (opens in any browser, works offline).

## What you need

- Python 3 with `numpy` and `scipy`  →  `pip install numpy scipy`
- This repository (cloned)
- Your data files (see the table at the bottom)

## Step 1 — export camera positions from each model project

In your reconstruction project:

1. Enable only the photos you want to score (uncheck the rest).
2. `Tools → Run Script…` → choose `scripts/metashape_solve_export.py`.
3. It writes `ms_out/cameras_est_computed.csv` inside the project folder.

Repeat for every project you want to compare.

## Step 2 — build the report

### Easiest: the main app, "Accuracy" tab

Open the app and go to the **Accuracy** tab. Browse to the four shared inputs,
press **Add project…** once per project (pick its `cameras_est_computed.csv`
and give it a label), then **Build accuracy report** — it opens in your
browser. Add a second project the same way to compare two on one report.

### Standalone window (no main app)

```
python scripts/accuracy_report_gui.py
```

Same file-picker flow in its own window.

### Or the command line

```
python scripts/accuracy_report.py ^
  --gt      ground_truth.pos ^
  --track   device_track.pos ^
  --georef  georef.csv ^
  --ftimes  frame_times.csv ^
  --meta    project1=project1/ms_out/cameras_est_computed.csv ^
            project2=project2/ms_out/cameras_est_computed.csv ^
  --out     report.html
```

Open `report.html` in a browser.

## Comparing several projects at once

Add one `--meta label=path` for each project. Every project becomes its own
row in the tables and its own coloured line on the map and charts, with its own
coverage %. Works whether the projects cover different parts of the capture or
the same frames with different settings.

Device-GPS-vs-ground-truth only (no model): add `--no-meta`.

## Input files

| flag | file | contents |
|------|------|----------|
| `--gt`     | `.pos` | ground-truth trajectory (survey-grade reference) |
| `--track`  | `.pos` | device track (used only to align the frame times) |
| `--georef` | `.csv` | per-frame device coordinates (`Image,Latitude,Longitude,Altitude,…`) |
| `--ftimes` | `.csv` | per-frame times (`Image,t_video_s`) |
| `--meta`   | `.csv` | camera-model estimate (`Label,Longitude,Latitude,Altitude`) — one per project |

Frame labels in every file must share the same stem (e.g. `frame_000123`) so
the report can line them up.

## Coordinate system

The camera-model CSVs should be **WGS84 (EPSG:4326)** — set that in Metashape's
Export Reference and no extra step is needed. If a CSV is in a projected CRS
(UTM, national grid, …), give its EPSG code — the **Accuracy** tab has a
"Projected CRS EPSG" box; on the command line use `--epsg 32636`, or per
project `--meta label=path@32636`. It is reprojected to WGS84 automatically.

## Per-frame trajectory CSV (Frame export tab)

The main app's **Frame export** tab joins `extracted_frame_times.csv` +
the per-frame coordinates CSV into one CSV, one row per frame:

`Image, t_audio_s, utc_s, utc_iso, latitude, longitude, altitude_m,
utm_zone, utm_easting, utm_northing, vE_mps, vN_mps, vU_mps, speed_mps,
azimuth_deg`

- Position is written in **both** WGS84 (lat/lon/alt) and **UTM** (zone
  auto-detected for the capture).
- Velocity is the geographic **East/North/Up** components, so `azimuth_deg`
  (`atan2(vE, vN)`) is a true-north bearing regardless of the position CRS.
  Taken from the coordinate CSV's Doppler columns when present, else derived
  from consecutive positions.
- `utc_s` / `utc_iso` / `t_audio_s` need the **raw session folder** (for the
  boot→UTC + audio anchors). Without it the video-relative time is written and
  UTC is left blank.

## What the report shows

Position accuracy (1/2/3σ, RMS, CEP, DRMS, MRSE), absolute vs best-fit
(systematic offset vs shape), robust and distribution statistics, confidence
intervals, per-axis and along/cross-track error, speed and heading accuracy,
error percentiles and thresholds, a top-view map, and error-vs-time,
error-vs-speed, covariance-ellipse, histogram, and cumulative-distribution
charts.
