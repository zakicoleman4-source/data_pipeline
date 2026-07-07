"""data_to_frames installer — handles the gtsam-not-on-PyPI problem.

Run:
    python install.py              # required only (gtsam skipped by default)
    python install.py --with-gtsam # required + try gtsam
    python install.py --dev        # required + pytest + pyinstaller (no gtsam)

What it does
============
1. ``pip install -r requirements.txt``     (always — required core)
2. ``pip install -r requirements-dev.txt`` (if --dev)
3. Try to install gtsam (optional) via:
     a. ``conda install -c conda-forge gtsam -y``  if conda on PATH
     b. ``pip install --pre gtsam``                 fallback (works on some pythons)
   On both failing, prints a clear hint pointing the user at
   conda-forge instructions; the rest of the pipeline still works
   (FGO is the only feature that needs gtsam, and it imports it
   lazily with an actionable ImportError).
4. Smoke-import every required dep so the user knows installation
   succeeded BEFORE they hit a runtime ImportError.

No silent failures — every step prints what it tried, what it got.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
REQ = REPO / "requirements.txt"
REQ_DEV = REPO / "requirements-dev.txt"
VENDOR_WHEELS = REPO / "vendor" / "wheels"

REQUIRED_IMPORTS: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("tkinterdnd2", "tkinterdnd2"),
    ("pyproj", "pyproj"),
    ("rasterio", "rasterio"),
    ("PIL", "pillow"),
]

OPTIONAL_IMPORTS: list[tuple[str, str]] = [
    ("gtsam", "gtsam (conda-forge)"),
]


def _run(cmd: list[str], *, allow_fail: bool = False) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not allow_fail:
        print(f"\nFAILED: exit code {proc.returncode}")
        sys.exit(proc.returncode)
    return proc.returncode


def _pip_install(req_file: Path) -> None:
    if not req_file.is_file():
        print(f"WARN: {req_file} not found; skipping")
        return
    _run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def _try_gtsam() -> bool:
    """Return True if gtsam ends up importable.

    Install attempts in order:
      1. vendor/wheels/gtsam-*.whl    (offline bundle for clients without
         conda OR PyPI access — see bundle_wheels.py)
      2. pip install --pre gtsam      (works on Linux x86_64 + macOS for
         most pythons; no-op on Python 3.13 / Windows)
      3. conda install -c conda-forge gtsam   (most reliable on Windows)
      4. graceful skip with hint     (FGO is optional; rest of pipeline
         unaffected — fgo.py raises actionable ImportError on use)
    """
    try:
        import gtsam  # noqa: F401
        print("\ngtsam already installed — skipping")
        return True
    except ImportError:
        pass

    # 1. Local bundled wheel — works without conda OR PyPI access.
    if VENDOR_WHEELS.is_dir():
        bundled = sorted(VENDOR_WHEELS.glob("gtsam-*.whl"))
        if bundled:
            print(f"\n--- gtsam install via bundled wheel "
                  f"({bundled[0].name}) ---")
            rc = _run([sys.executable, "-m", "pip", "install",
                       "--no-index", "--find-links", str(VENDOR_WHEELS),
                       "gtsam"],
                      allow_fail=True)
            if rc == 0:
                try:
                    import gtsam  # noqa: F401
                    return True
                except ImportError as e:
                    print(f"bundled wheel installed but import failed: {e}")

    # 2. pip --pre (no conda needed; works when a PyPI wheel exists).
    print("\n--- gtsam install via pip --pre ---")
    rc = _run([sys.executable, "-m", "pip", "install", "--pre", "gtsam"],
              allow_fail=True)
    if rc == 0:
        try:
            import gtsam  # noqa: F401
            return True
        except ImportError:
            pass

    # 3. conda fallback (most reliable on Windows when present).
    conda = shutil.which("conda") or shutil.which("conda.exe")
    if conda:
        print("\n--- gtsam install via conda-forge ---")
        rc = _run([conda, "install", "-c", "conda-forge", "gtsam", "-y"],
                  allow_fail=True)
        if rc == 0:
            try:
                import gtsam  # noqa: F401
                return True
            except ImportError as e:
                print(f"conda install reported success but import failed: {e}")
    else:
        print("\nconda not on PATH; skipping conda fallback")

    print(
        "\ngtsam NOT installed. Pipeline still works — only the FGO\n"
        "smoother (CsvOptions.use_fgo_smoothing=True) needs it. The other\n"
        "smoothers (cv_rts_pv, ekf_smoothed, Gaussian) all run without it.\n\n"
        "To install later:\n"
        "  1) Download a wheel into vendor/wheels/ and re-run install.py,\n"
        "     OR  python bundle_wheels.py gtsam   (downloads to vendor/wheels)\n"
        "  2) pip install --pre gtsam              (when your Python has a wheel)\n"
        "  3) conda install -c conda-forge gtsam   (if conda is available)\n"
    )
    return False


def _verify_imports(items: list[tuple[str, str]], *, required: bool) -> int:
    failures = 0
    label = "REQUIRED" if required else "OPTIONAL"
    print(f"\n=== {label} import smoke ===")
    for mod_name, pkg_name in items:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "(no __version__)")
            print(f"  OK   {pkg_name:30s} {ver}")
        except ImportError as e:
            print(f"  MISS {pkg_name:30s} ImportError: {e}")
            failures += 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-gtsam", action="store_true",
                    help="Try to install gtsam (optional; device IMU not accurate "
                         "enough to benefit from FGO in practice).")
    ap.add_argument("--dev", action="store_true",
                    help="Also install requirements-dev.txt (pytest, pyinstaller).")
    args = ap.parse_args()

    print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Repo    : {REPO}")

    print("\n=== Installing required dependencies ===")
    _pip_install(REQ)

    if args.dev:
        print("\n=== Installing dev / build dependencies ===")
        _pip_install(REQ_DEV)

    if args.with_gtsam:
        _try_gtsam()

    req_fails = _verify_imports(REQUIRED_IMPORTS, required=True)
    _verify_imports(OPTIONAL_IMPORTS, required=False)

    print()
    if req_fails:
        print(f"INSTALL INCOMPLETE — {req_fails} required dep(s) missing. "
              "See messages above.")
        return 1
    print("INSTALL OK — all required deps importable. Run "
          "`python -m data_pipeline` to launch the GUI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
