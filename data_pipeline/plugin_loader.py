"""Discover and load plugins into the :mod:`data_pipeline.plugins_api` registry.

Two discovery paths, both tolerant of failure:

1. **Drop-in directory** (primary): every ``*.py`` under
   ``data_pipeline/plugins/`` is imported at startup. Importing the module
   runs its ``@register_fusion`` / ``@register_calibration`` decorators, which
   register (and validate) the plugins. A broken plugin file is caught and
   reported — it never crashes the host.

2. **Entry-points** (thin extra): packages installed in the environment can
   advertise plugins under the setuptools entry-point groups
   ``client_pipeline.fusion`` and ``client_pipeline.calibration``. Each
   entry point is loaded; loading the referenced object is expected to trigger
   its registration decorator (or the object can be registered manually inside
   its module). Tolerates "none installed".

Call :func:`load_all_plugins` once at startup (e.g. from the GUI or pipeline
bootstrap). It returns a :class:`LoadReport` summarising what loaded and what
failed, so the caller can surface a banner without digging through logs.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .plugins_api import (
    PluginError,
    list_calibration_plugins,
    list_fusion_plugins,
)

_log = logging.getLogger(__name__)

# Entry-point groups third-party packages advertise plugins under.
FUSION_ENTRY_POINT_GROUP = "client_pipeline.fusion"
CALIBRATION_ENTRY_POINT_GROUP = "client_pipeline.calibration"

# Where drop-in plugin files live.
_PLUGINS_PKG = "data_pipeline.plugins"
_PLUGINS_DIR = Path(__file__).parent / "plugins"


@dataclass
class LoadReport:
    """Summary of a plugin discovery run."""

    loaded_modules: list[str] = field(default_factory=list)
    loaded_entry_points: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (source, msg)
    fusion_plugins: list[str] = field(default_factory=list)
    calibration_plugins: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"plugins: {len(self.fusion_plugins)} fusion, "
            f"{len(self.calibration_plugins)} calibration "
            f"({len(self.loaded_modules)} drop-in module(s), "
            f"{len(self.loaded_entry_points)} entry-point(s))"
        ]
        for src, msg in self.errors:
            lines.append(f"  ! {src}: {msg}")
        return "\n".join(lines)


def load_dropin_plugins(report: Optional[LoadReport] = None) -> LoadReport:
    """Import every ``*.py`` module under ``data_pipeline/plugins/``.

    Each module's import-time decorators do the registration. Per-file
    failures are logged in the report; they do not propagate.
    """
    report = report or LoadReport()
    if not _PLUGINS_DIR.is_dir():
        return report

    try:
        pkg = importlib.import_module(_PLUGINS_PKG)
    except Exception as e:  # noqa: BLE001
        report.errors.append((_PLUGINS_PKG, f"{type(e).__name__}: {e}"))
        return report

    for mod_info in pkgutil.iter_modules(pkg.__path__):
        mod_name = mod_info.name
        if mod_name.startswith("_"):
            continue
        full = f"{_PLUGINS_PKG}.{mod_name}"
        try:
            importlib.import_module(full)
            report.loaded_modules.append(full)
        except PluginError as e:
            report.errors.append((full, f"PluginError: {e}"))
            _log.warning("[plugin_loader] %s rejected: %s", full, e)
        except Exception as e:  # noqa: BLE001
            report.errors.append((full, f"{type(e).__name__}: {e}"))
            _log.warning("[plugin_loader] %s failed to import: %s", full, e)
    return report


def load_entry_point_plugins(report: Optional[LoadReport] = None) -> LoadReport:
    """Load plugins advertised via setuptools entry points.

    Looks under :data:`FUSION_ENTRY_POINT_GROUP` and
    :data:`CALIBRATION_ENTRY_POINT_GROUP`. Loading each entry point is
    expected to trigger its registration decorator. Tolerates no entry
    points installed and missing/old ``importlib.metadata`` APIs.
    """
    report = report or LoadReport()
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:  # pragma: no cover - py<3.8
        report.errors.append(("entry_points", "importlib.metadata unavailable"))
        return report

    for group in (FUSION_ENTRY_POINT_GROUP, CALIBRATION_ENTRY_POINT_GROUP):
        try:
            eps = _iter_entry_points(importlib_metadata, group)
        except Exception as e:  # noqa: BLE001
            report.errors.append((group, f"{type(e).__name__}: {e}"))
            continue
        for ep in eps:
            label = f"{group}:{getattr(ep, 'name', '?')}"
            try:
                ep.load()  # importing the target runs its decorators
                report.loaded_entry_points.append(label)
            except PluginError as e:
                report.errors.append((label, f"PluginError: {e}"))
                _log.warning("[plugin_loader] %s rejected: %s", label, e)
            except Exception as e:  # noqa: BLE001
                report.errors.append((label, f"{type(e).__name__}: {e}"))
                _log.warning("[plugin_loader] %s failed to load: %s", label, e)
    return report


def _iter_entry_points(importlib_metadata, group: str):
    """Yield entry points for ``group`` across importlib.metadata API versions.

    Python 3.10+ ``entry_points(group=...)`` vs the older dict-returning API.
    """
    entry_points = importlib_metadata.entry_points
    try:
        # Python 3.10+ selectable API.
        return list(entry_points(group=group))
    except TypeError:
        # Older API: entry_points() -> dict[str, list[EntryPoint]].
        all_eps = entry_points()
        if hasattr(all_eps, "get"):
            return list(all_eps.get(group, []))
        # SelectableGroups fallback.
        return [ep for ep in all_eps if getattr(ep, "group", None) == group]


def load_all_plugins(
    *,
    dropin: bool = True,
    entry_points: bool = True,
) -> LoadReport:
    """Run both discovery paths and return a combined :class:`LoadReport`.

    Idempotent in effect: re-importing already-imported modules is a no-op,
    and the registry overwrites duplicate names. Safe to call at startup.
    """
    report = LoadReport()
    if dropin:
        load_dropin_plugins(report)
    if entry_points:
        load_entry_point_plugins(report)
    report.fusion_plugins = list_fusion_plugins()
    report.calibration_plugins = list_calibration_plugins()
    if report.errors:
        _log.warning("[plugin_loader] %s", report.summary())
    else:
        _log.info("[plugin_loader] %s", report.summary())
    return report
