"""Pre-download wheels into ``vendor/wheels/`` for offline / no-conda
client installs.

Use this BEFORE shipping a copy of the repo to a client whose machine
won't have internet / conda / PyPI access at install time.

Run:
    python bundle_wheels.py                # all requirements + gtsam
    python bundle_wheels.py gtsam          # just gtsam
    python bundle_wheels.py --platform win_amd64 --python-version 311
    python bundle_wheels.py --platform manylinux2014_x86_64 --python-version 312

The client then runs ``python install.py`` and the bundled wheels are
preferred over PyPI (``--find-links vendor/wheels``).

What this solves
================
gtsam has no PyPI wheel for Python 3.13. Clients without conda can't
install it via pip. By bundling a compatible wheel into vendor/wheels/
BEFORE shipping, the client install runs purely offline-friendly:
``install.py`` picks up the vendored wheel first (see ``_try_gtsam``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENDOR_WHEELS = REPO / "vendor" / "wheels"
REQ = REPO / "requirements.txt"


def _run(cmd: list[str], *, allow_fail: bool = False) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not allow_fail:
        sys.exit(proc.returncode)
    return proc.returncode


def _download(pkg: str, *, platform: str | None, python_version: str | None,
              allow_pre: bool) -> int:
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(VENDOR_WHEELS),
        "--no-deps",
    ]
    if platform:
        cmd += ["--platform", platform, "--only-binary=:all:"]
    if python_version:
        cmd += ["--python-version", python_version]
    if allow_pre:
        cmd += ["--pre"]
    cmd.append(pkg)
    return _run(cmd, allow_fail=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("packages", nargs="*",
                    help="Packages to download. Default: every line in "
                         "requirements.txt PLUS gtsam.")
    ap.add_argument("--platform", default=None,
                    help="Target platform tag (e.g. win_amd64, "
                         "manylinux2014_x86_64). Defaults to current host.")
    ap.add_argument("--python-version", default=None,
                    help="Target CPython minor (e.g. 311, 312). "
                         "Defaults to current host.")
    ap.add_argument("--no-gtsam", action="store_true",
                    help="Skip gtsam download.")
    args = ap.parse_args()

    VENDOR_WHEELS.mkdir(parents=True, exist_ok=True)

    if args.packages:
        pkgs = list(args.packages)
        include_gtsam = ("gtsam" in pkgs) and not args.no_gtsam
    else:
        # Whole requirements file in one shot.
        print(f"=== downloading every line of {REQ} into {VENDOR_WHEELS} ===")
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(VENDOR_WHEELS),
            "-r", str(REQ),
        ]
        if args.platform:
            cmd += ["--platform", args.platform, "--only-binary=:all:"]
        if args.python_version:
            cmd += ["--python-version", args.python_version]
        _run(cmd, allow_fail=True)
        pkgs = []
        include_gtsam = not args.no_gtsam

    if include_gtsam:
        print("\n=== downloading gtsam (pre-release allowed) ===")
        rc = _download("gtsam", platform=args.platform,
                       python_version=args.python_version, allow_pre=True)
        if rc != 0:
            print(
                "\ngtsam download FAILED — no public wheel for this "
                "platform/python combo. Options:\n"
                "  1. Build from source on a machine with C++ toolchain\n"
                "  2. Copy the wheel out of a working conda env:\n"
                "     conda install -c conda-forge gtsam\n"
                "     pip wheel gtsam -w vendor/wheels/\n"
                "  3. Ship without FGO (cv_rts_pv smoother beats FGO\n"
                "     2.416m vs ~2.85m on the reference session anyway)\n"
            )

    for pkg in pkgs:
        if pkg == "gtsam":
            continue  # already handled
        print(f"\n=== downloading {pkg} ===")
        _download(pkg, platform=args.platform,
                  python_version=args.python_version, allow_pre=False)

    print(f"\nBundle done. {VENDOR_WHEELS} contents:")
    for w in sorted(VENDOR_WHEELS.iterdir()):
        if w.is_file() and w.suffix in (".whl", ".tar.gz"):
            print(f"  {w.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
