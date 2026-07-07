"""Resolve ``the external converter`` and ``the probe tool`` executables for offline use.

Search order:

1. Environment: ``DATA_PIPELINE_FFMPEG`` / ``DATA_PIPELINE_FFPROBE`` (full paths).
2. Vendored copy under the repo: ``vendor/the external converter/bin/`` (see ``vendor/the external converter/README.md``).
3. System ``PATH`` via :func:`shutil.which`.

This lets an air-gapped machine work as long as the maintainer drops a
portable the external converter build into ``vendor/the external converter/`` before copying the repo, or
sets the env vars to a local folder on a USB stick.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _exe_candidates(name: str) -> list[Path]:
    root = _repo_root()
    bin_dir = root / "vendor" / "ffmpeg" / "bin"
    if sys.platform == "win32":
        return [
            bin_dir / f"{name}.exe",
            bin_dir / f"{name}.EXE",
        ]
    return [bin_dir / name]


def resolve_ffmpeg() -> str:
    """Return a string path or name suitable for :class:`subprocess` run."""
    env = os.environ.get("DATA_PIPELINE_FFMPEG", "").strip()
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(
                f"DATA_PIPELINE_FFMPEG points to missing file: {env}"
            )
        return str(p.resolve())

    for c in _exe_candidates("ffmpeg"):
        if c.is_file():
            return str(c.resolve())

    which = shutil.which("ffmpeg")
    if which:
        return which

    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg and add it to PATH, or unpack a "
        "portable build into vendor/ffmpeg/bin/ (see vendor/ffmpeg/README.md), "
        "or set DATA_PIPELINE_FFMPEG to the full path of ffmpeg."
    )


def resolve_ffprobe() -> str:
    """Return the probe tool path (same directory as the external converter if vendored or on PATH)."""
    env = os.environ.get("DATA_PIPELINE_FFPROBE", "").strip()
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(
                f"DATA_PIPELINE_FFPROBE points to missing file: {env}"
            )
        return str(p.resolve())

    for c in _exe_candidates("ffprobe"):
        if c.is_file():
            return str(c.resolve())

    # Next to vendored the external converter in the same bin/ folder
    for c in _exe_candidates("ffmpeg"):
        if c.is_file():
            probe = c.parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            if probe.is_file():
                return str(probe.resolve())

    # Next to the external converter discovered on PATH (portable zip layout)
    which_ff = shutil.which("ffmpeg")
    if which_ff:
        parent = Path(which_ff).resolve().parent
        probe = parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if probe.is_file():
            return str(probe.resolve())

    which = shutil.which("ffprobe")
    if which:
        return which

    raise FileNotFoundError(
        "ffprobe not found. It normally lives next to ffmpeg. Unpack a full "
        "static build into vendor/ffmpeg/bin/ or set DATA_PIPELINE_FFPROBE."
    )


def versions_summary() -> str:
    """Human-readable the external converter/the probe tool version lines (for diagnostics)."""
    import subprocess

    lines: list[str] = []
    for label, fn in (("ffmpeg", resolve_ffmpeg), ("ffprobe", resolve_ffprobe)):
        try:
            exe = fn()
        except FileNotFoundError as e:
            lines.append(f"{label}: NOT FOUND ({e})")
            continue
        try:
            p = subprocess.run(
                [exe, "-version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            first = (p.stdout or "").splitlines()[:1]
            lines.append(f"{label}: {exe}")
            if first:
                lines.append(f"  {first[0]}")
        except Exception as ex:
            lines.append(f"{label}: {exe} (error running -version: {ex})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(versions_summary())
