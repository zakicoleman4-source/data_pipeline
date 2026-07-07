"""Single source of truth for external lab-tool executable paths.

The pipeline wraps several non-Python tools that the user installs
once and forgets about:

* ``rnx2rtkp.exe``   — the external solver engine (Post-processing tab)
* ``convbin.exe``    — vendor binary → interchange-format converter (T02 tab fallback)
* ``jps2rin.exe``    — vendor JPS → interchange-format converter (T02 tab default for .jps)
* ``runpkr00.exe``   — vendor T02 unpacker (T02 tab default for .t02)
* ``teqc.exe``       — vendor TGD → interchange-format converter (T02 tab; pairs w/ runpkr00)

Resolution order for every tool, in priority:

1. Explicit ``override`` argument passed by a caller.
2. Per-tool environment variable (e.g. ``RNX2RTKP``).
3. ``data_to_frames.config.json`` next to the package root if present.
4. The lab developer's default install path (see constants below).
5. ``shutil.which`` of the binary name on the system PATH.

This module is consumed by ``data_pipeline.stages.ppk`` and
``data_pipeline.stages.t02``; both modules previously baked the
lab developer's paths directly into their source, which exploded on
the client's machine because those paths didn't exist there. The
config file + env-var path means the client edits one JSON file
(or sets a handful of env vars) and the pipeline finds everything.

See ``SETUP.md`` for the documented setup matrix.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


# Default install locations are NOT shipped with the project — the user
# must set env vars (RNX2RTKP, CONVBIN, JPS2RIN, RUNPKR00, TEQC) or
# place the binaries somewhere on PATH. The lookup chain still works
# without these defaults.
_LAB_DEFAULTS: dict[str, Path] = {}

_LAB_FALLBACKS: dict[str, tuple[Path, ...]] = {}

# Environment variable name per tool.
_ENV_VARS: dict[str, str] = {
    "rnx2rtkp": "RNX2RTKP",
    "convbin":  "CONVBIN",
    "jps2rin":  "JPS2RIN",
    "runpkr00": "RUNPKR00",
    "teqc":     "TEQC",
}


def _config_file() -> Path:
    """Return the optional ``data_to_frames.config.json`` location.

    The file lives next to the installed pipeline (one level above the
    ``data_pipeline`` package). It is OPTIONAL — clients that prefer
    env vars or that don't customise paths never need to create it.
    """
    return Path(__file__).resolve().parent.parent / "data_to_frames.config.json"


def _bundled_tool_path(name: str) -> Optional[Path]:
    """Resolve a tool bundled inside the PyInstaller dist or source tree.

    When the user runs the frozen ``data_to_frames.exe``,
    ``sys.frozen`` is set and ``sys._MEIPASS`` points at the
    extracted ``_internal`` directory. The bundled ``vendor/rtklib/`` +
    ``vendor/the external converter/`` live there. From a source checkout, the same
    folders live two levels above this module file.
    """
    rel_map = {
        "rnx2rtkp": ("vendor", "rtklib", "rnx2rtkp.exe"),
    }
    if name not in rel_map:
        return None
    # Frozen exe: _internal/ is the resource root.
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cand = Path(meipass).joinpath(*rel_map[name])
            if cand.is_file():
                return cand
    # Source checkout: <repo>/vendor/...
    src = Path(__file__).resolve().parent.parent.joinpath(*rel_map[name])
    if src.is_file():
        return src
    return None


def _read_config() -> dict[str, str]:
    p = _config_file()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def resolve_tool(name: str, override: Optional[Path] = None) -> Path:
    """Return the resolved path to ``name``.

    ``name`` is one of: rnx2rtkp, convbin, jps2rin, runpkr00, teqc.

    Raises :class:`FileNotFoundError` listing every probed location when
    no candidate is on disk — that error is what the client sees if
    setup is incomplete, so we make it actionable.
    """
    if name not in _ENV_VARS:
        raise ValueError(
            f"Unknown lab tool {name!r}. "
            f"Known: {', '.join(sorted(_ENV_VARS))}"
        )

    tried: list[str] = []

    # 1) caller override
    if override is not None:
        p = Path(override)
        tried.append(f"override={p}")
        if p.is_file():
            return p

    # 2) environment variable
    env_name = _ENV_VARS[name]
    env_val = os.environ.get(env_name)
    if env_val:
        p = Path(env_val)
        tried.append(f"${env_name}={p}")
        if p.is_file():
            return p

    # 3) config file
    cfg = _read_config().get(name)
    if cfg:
        p = Path(cfg)
        tried.append(f"config[{name}]={p}")
        if p.is_file():
            return p

    # 3.5) bundled binary (frozen exe ships the solver under vendor/rtklib)
    bundled = _bundled_tool_path(name)
    if bundled is not None:
        tried.append(f"bundled={bundled}")
        return bundled

    # 4) lab default
    default = _LAB_DEFAULTS.get(name)
    if default is not None:
        tried.append(f"default={default}")
        if default.is_file():
            return default

    # 4b) fallbacks
    for fb in _LAB_FALLBACKS.get(name, ()):
        tried.append(f"fallback={fb}")
        if fb.is_file():
            return fb

    # 5) PATH
    exe_name = f"{name}.exe" if os.name == "nt" else name
    found = shutil.which(exe_name) or shutil.which(name)
    if found:
        return Path(found)
    tried.append(f"PATH={exe_name}")

    raise FileNotFoundError(
        f"Could not locate {name!r}. Tried:\n  - "
        + "\n  - ".join(tried)
        + f"\n\nFixes (any one is enough):\n"
          f"  • set the {env_name} environment variable\n"
          f"  • drop a {_config_file().name} next to the pipeline with "
          f'\n        {{ "{name}": "C:/path/to/{exe_name}" }}\n'
          f"  • install the binary at the default path "
          f"({_LAB_DEFAULTS.get(name, '(none)')})\n"
          f"See SETUP.md for the external-tools matrix."
    )


def list_tools() -> list[str]:
    """Names of every external lab tool the pipeline knows about."""
    return sorted(_ENV_VARS)


def report() -> dict[str, str]:
    """One-line resolution status per tool — for setup smoke / GUI status.

    Each entry is either the resolved path or ``"MISSING"``.
    """
    out: dict[str, str] = {}
    for name in list_tools():
        try:
            out[name] = str(resolve_tool(name))
        except FileNotFoundError:
            out[name] = "MISSING"
    return out
