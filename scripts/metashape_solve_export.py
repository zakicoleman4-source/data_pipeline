"""Solve the ENABLED photos in the CURRENTLY OPEN Metashape chunk and export
estimated camera coordinates for the accuracy report.

Workflow (you do the first three, script does the rest):
  1. New chunk, import your photos.
  2. Import camera reference coordinates, CRS = WGS84 (EPSG:4326).
  3. Disable (uncheck) the photos you do NOT want; keep only the ones to solve.
  4. Tools -> Run Script... -> pick this file.  (or paste in the Console)

It constrains the self-calibration (the thing that blew up: f=250743, k3=2.7e13
on forward linear motion) by fitting only f/cx/cy/k1/k2 and freezing
k3/k4/p1/p2/b1/b2, seeds a sane focal length, aligns with position-reference
preselection, runs a constrained bundle adjust, and writes:

  <project_dir>/ms_out/cameras_est_computed.csv   (Label,Longitude,Latitude,Altitude)

Only ENABLED + successfully-aligned cameras are exported. Disabled photos are
ignored; frames that fail to align are simply left out (not an error).

Config below: flip REALIGN=False to skip match/align (keep your existing
alignment) and only re-optimize + export.
"""
import os, csv
import Metashape

# ---- config ----
REALIGN      = True     # False = keep current alignment, only optimize + export
DOWNSCALE    = 1        # match accuracy: 1=highest, 2=high, 4=medium
F_INIT_PX    = 1500.0   # sane focal seed (1920px-wide ~65 deg HFOV); ignored if already calibrated
KEYPOINT_LIM = 40000
TIEPOINT_LIM = 10000
OUT_NAME     = "cameras_est_computed.csv"

def log(m):
    print(m, flush=True)

doc = Metashape.app.document
chunk = doc.chunk
if chunk is None:
    raise RuntimeError("No active chunk. Open your project first.")

enabled = [c for c in chunk.cameras if c.enabled]
log(f"=== solve+export: chunk '{chunk.label}' — {len(enabled)}/{len(chunk.cameras)} photos enabled ===")
if len(enabled) < 3:
    raise RuntimeError(f"Only {len(enabled)} enabled cameras; need >=3.")

# Reference coordinates must exist on the enabled photos (workflow step 2) —
# otherwise alignment "succeeds" but the exported LLH is meaningless (arbitrary
# local coordinates reprojected through the CRS look plausible but are garbage).
have_ref = any(c.reference.location is not None for c in enabled)
if not have_ref:
    raise RuntimeError(
        "No reference coordinates (Longitude/Latitude/Altitude) found on any "
        "enabled photo. Import camera reference LLH before running this script — "
        "without it, exported coordinates would be meaningless."
    )

# CRS: assume WGS84 geographic if not already geographic.
if chunk.crs is None:
    chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
log(f"[crs] {chunk.crs}")

# Seed a sane focal length on any sensor that isn't calibrated yet. Only touch
# sensor.type when we are actually seeding a fresh (perspective) calibration —
# forcing it unconditionally would silently clobber an existing Fisheye/
# Spherical sensor's lens model.
for s in chunk.sensors:
    if not s.calibration or s.calibration.f <= 1.0 or s.calibration.f > 100000.0:
        s.type = Metashape.Sensor.Type.Frame
        cal = Metashape.Calibration()
        cal.width, cal.height = s.width, s.height
        cal.f = F_INIT_PX; cal.cx = 0.0; cal.cy = 0.0
        s.user_calib = cal
        log(f"[calib] seeded f={F_INIT_PX}px on sensor {s.label} ({s.width}x{s.height})")

if REALIGN:
    log("[match] matchPhotos (reference preselection, enabled photos only)...")
    chunk.matchPhotos(downscale=DOWNSCALE, generic_preselection=True,
                      reference_preselection=True,
                      reference_preselection_mode=Metashape.ReferencePreselectionSource,
                      keypoint_limit=KEYPOINT_LIM, tiepoint_limit=TIEPOINT_LIM)
    log("[align] alignCameras...")
    chunk.alignCameras()

aligned = [c for c in enabled if c.transform is not None]
log(f"[align] {len(aligned)}/{len(enabled)} enabled cameras aligned")
if len(aligned) < 3:
    raise RuntimeError("Fewer than 3 aligned cameras — cannot export.")

# Constrained bundle adjust: freeze the params that diverge on forward linear motion.
log("[opt] optimizeCameras (k3/k4/p1/p2/b1/b2 frozen)...")
chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_k1=True, fit_k2=True,
                      fit_k3=False, fit_k4=False, fit_p1=False, fit_p2=False,
                      fit_b1=False, fit_b2=False, adaptive_fitting=False)
# Log the calibration for every sensor actually used by an aligned camera
# (not just sensors[0] — wrong/misleading if the chunk has multiple sensors).
_logged_sensors = set()
for c in aligned:
    s = c.sensor
    if s is None or s.key in _logged_sensors:
        continue
    _logged_sensors.add(s.key)
    cal = s.calibration
    log(f"[opt] sensor '{s.label}': f={cal.f:.1f} cx={cal.cx:.1f} cy={cal.cy:.1f} "
        f"k1={cal.k1:.4f} k2={cal.k2:.4f}")

# Estimated camera LLH straight from the solved transform (version-proof).
if chunk.transform is None or chunk.crs is None:
    raise RuntimeError(
        "Chunk has no transform/CRS after alignment+optimization — "
        "georeferencing did not take. Check reference coordinates and CRS, "
        "then re-run before exporting."
    )
out_dir = os.path.join(os.path.dirname(doc.path) if doc.path else os.getcwd(), "ms_out")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, OUT_NAME)
T, crs = chunk.transform.matrix, chunk.crs
n = 0
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Label", "Longitude", "Latitude", "Altitude"])
    for c in aligned:
        lon, lat, alt = crs.project(T.mulp(c.center))
        w.writerow([c.label, f"{lon:.9f}", f"{lat:.9f}", f"{alt:.4f}"])
        n += 1
log(f"[export] {out_csv}  ({n} estimated cameras)")
log("=== DONE — point the accuracy report at this CSV ===")
