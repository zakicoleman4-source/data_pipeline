"""data_pipeline first-launch health check.

Run AFTER ``setup.bat`` to confirm every external dependency the
pipeline relies on is actually reachable on this machine. Prints a
matrix with pass/fail per tool plus the resolved path. Exits non-zero
when any required component is missing so a CI / install script can
gate on it.

Usage::

    python scripts/doctor.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


PY_PACKAGES = [
    ("numpy",          True,  "core math"),
    ("scipy",          True,  "Gaussian smoothing + Kalman utilities"),
    ("cv2",            True,  "frame extraction + adaptive selector"),
    ("tkinterdnd2",    False, "GUI drag-and-drop (optional)"),
    ("matplotlib",     False, "velocity plot (optional)"),
    ("rasterio",       False, "GeoTIFF basemap (optional)"),
]


def check_python_packages() -> int:
    print("[1] Python packages")
    fails = 0
    for name, required, purpose in PY_PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"    OK  {name:14s} {ver:<10s} — {purpose}")
        except Exception as e:
            tag = "FAIL" if required else "warn"
            print(f"    {tag} {name:14s} {'(missing)':<10s} — {purpose}: {e}")
            if required:
                fails += 1
    return fails


def check_ffmpeg() -> int:
    print("\n[2] FFmpeg")
    try:
        from data_pipeline.ffmpeg_paths import resolve_ffmpeg, resolve_ffprobe
        ffmpeg = resolve_ffmpeg()
        ffprobe = resolve_ffprobe()
        print(f"    OK   ffmpeg  : {ffmpeg}")
        print(f"    OK   ffprobe : {ffprobe}")
        return 0
    except Exception as e:
        print(f"    FAIL ffmpeg / ffprobe: {e}")
        return 1


def check_lab_tools() -> int:
    print("\n[3] External lab tools (PPK + T02 / GNSS-binary tabs)")
    from data_pipeline.lab_tools import report
    res = report()
    fails = 0
    # Post-processing tab needs at minimum the solver binary. T02 tab needs jps2rin OR
    # (runpkr00 + teqc). the converter binary is fallback. The pipeline degrades
    # cleanly when these are missing — but the corresponding GUI tabs
    # will throw on use, so warn loudly.
    required_for_ppk = ("rnx2rtkp",)
    required_for_t02_jps = ("jps2rin",)
    required_for_t02_trimble = ("runpkr00", "teqc")
    for name, path in sorted(res.items()):
        ok = path != "MISSING"
        tag = "OK  " if ok else "warn"
        print(f"    {tag} {name:10s}: {path}")

    if any(res[t] == "MISSING" for t in required_for_ppk):
        print("    !!! PPK tab will FAIL — install rnx2rtkp (RTKLIB).")
        fails += 1
    if all(res[t] == "MISSING" for t in required_for_t02_jps):
        if any(res[t] == "MISSING" for t in required_for_t02_trimble):
            print("    !!! T02/JPS tab will FAIL — install at least one of: "
                  "jps2rin (Javad), or runpkr00+teqc (Trimble).")
    return fails


def check_vendored() -> int:
    print("\n[4] Vendored data")
    fails = 0
    must_exist = [
        _REPO / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe",
        _REPO / "vendor" / "android_rinex" / "src",
        _REPO / "data_pipeline" / "assets" / "plotly.min.js",
        _REPO / "data_pipeline" / "assets" / "sync_player.html",
    ]
    for p in must_exist:
        if p.exists():
            print(f"    OK   {p.relative_to(_REPO)}")
        else:
            print(f"    FAIL {p.relative_to(_REPO)}  (missing)")
            fails += 1
    return fails


def check_writable_temp() -> int:
    print("\n[5] Filesystem")
    fails = 0
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=True) as f:
            print(f"    OK   temp dir writable ({Path(f.name).parent})")
    except Exception as e:
        print(f"    FAIL temp dir not writable: {e}")
        fails += 1
    return fails


def main() -> int:
    print("=" * 60)
    print(" data_pipeline — environment doctor")
    print("=" * 60)

    fails = 0
    fails += check_python_packages()
    fails += check_ffmpeg()
    fails += check_lab_tools()
    fails += check_vendored()
    fails += check_writable_temp()

    print()
    if fails:
        print(f"!!! {fails} hard problems found. See SETUP.md.")
    else:
        print("All required components reachable. Run:  python -m data_pipeline")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
