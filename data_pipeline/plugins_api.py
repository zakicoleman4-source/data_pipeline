"""Pluggable Motion sensor-Signal fusion + Motion sensor-calibration plugin contracts and registry.

Why this module
===============
``smoothers.py`` ships 15 built-in fusion smoothers and ``imu_calibration.py``
ships one Allan-derived calibration backend. External developers want to plug
in THEIR OWN fusion algorithm or calibration method without forking the repo.

This module defines the *contract* (two strict :class:`typing.Protocol`
classes) and a *self-contained registry* so a third-party plugin can be
registered with a decorator and later surfaced alongside the built-ins.

Design decisions
----------------
* **Protocol + in-repo registry/drop-in is the primary path.** Entry-points
  discovery (see :mod:`data_pipeline.plugin_loader`) is a thin extra.
* **STRICT TYPED contract with validation.** Plugins are validated at
  registration time against the protocol signature AND smoke-tested on a tiny
  synthetic input. Any mismatch raises :class:`PluginError` listing exactly
  what is wrong — we fail loudly, never silently accept a broken plugin.
* **The registry is self-contained.** It keeps its own dicts and does NOT
  import ``smoothers.py`` / ``imu_calibration.py``. That keeps this module
  import-cheap and avoids a circular dependency; the integration that surfaces
  registered plugins inside those two modules is a separate, later wiring pass.

Contracts
---------
:class:`FusionPlugin`
    ``run(pos_rows, imu_rows, calibration, options) -> list[PosRow]`` plus a
    ``name: str`` attribute. Same in/out row type as the built-in smoothers,
    so a registered fusion plugin is a drop-in alternative to a smoother.

:class:`CalibrationPlugin`
    ``compute(sensors_rows, options) -> dict`` plus a ``name: str`` attribute.
    The returned dict must match the :class:`imu_calibration.ImuCalibration`
    serialised schema (``device_label``, ``date``, ``sample_rate_hz``,
    ``duration_s``, ``source``, ``axes`` with per-axis ARW/VRW =
    ``random_walk``, ``bias_instability``, ``rate_random_walk`` =RRW).

Both protocols are ``@runtime_checkable`` so ``isinstance`` can confirm an
object exposes the right callable, but we do NOT rely on that alone — the
``validate_*`` helpers inspect the call signature and run a smoke test.
"""
from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from .parsers import ImuRow, PosRow

__all__ = [
    "PluginError",
    "FusionPlugin",
    "CalibrationPlugin",
    "register_fusion",
    "register_calibration",
    "get_fusion_plugin",
    "get_calibration_plugin",
    "list_fusion_plugins",
    "list_calibration_plugins",
    "unregister_fusion",
    "unregister_calibration",
    "clear_registry",
    "validate_fusion_plugin",
    "validate_calibration_plugin",
    "REQUIRED_CALIBRATION_KEYS",
    "REQUIRED_AXIS_KEYS",
]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class PluginError(Exception):
    """A plugin failed to conform to its contract.

    The message enumerates every problem found so a plugin author can fix
    them in one pass instead of one error at a time.
    """


# ---------------------------------------------------------------------------
# Protocol contracts (strict)
# ---------------------------------------------------------------------------

@runtime_checkable
class FusionPlugin(Protocol):
    """Contract for a third-party Motion sensor-Signal fusion algorithm.

    A conforming object is either an instance with a ``run`` method or a
    class whose instances expose ``run``. It MUST also carry a ``name``
    string used as the registry key.
    """

    name: str

    def run(
        self,
        pos_rows: list[PosRow],
        imu_rows: list,
        calibration: Optional[dict],
        options: dict,
    ) -> list[PosRow]:
        """Fuse ``pos_rows`` (+ optional ``imu_rows``/``calibration``) and
        return a new ``list[PosRow]``.

        * ``pos_rows`` — Signal/Post-processing positions (``data_pipeline.parsers.PosRow``).
        * ``imu_rows`` — Motion sensor samples (``ImuRow``); may be empty for Signal-only.
        * ``calibration`` — an :class:`ImuCalibration`-shaped dict or ``None``.
        * ``options`` — free-form ``dict`` of tuning knobs.

        Returns a list of ``PosRow`` (smoothed/fused positions).
        """
        ...


@runtime_checkable
class CalibrationPlugin(Protocol):
    """Contract for a third-party Motion sensor calibration method.

    A conforming object exposes ``compute`` and carries a ``name`` string.
    """

    name: str

    def compute(self, sensors_rows: list, options: dict) -> dict:
        """Compute an Motion sensor calibration from raw sensor rows.

        * ``sensors_rows`` — Motion sensor samples (``ImuRow``), typically static or
          mined ZUPT segments.
        * ``options`` — free-form ``dict`` (e.g. ``device_label``,
          ``sample_rate_hz`` override).

        Returns a calibration ``dict`` matching the
        :class:`imu_calibration.ImuCalibration` serialised schema
        (see :data:`REQUIRED_CALIBRATION_KEYS` / :data:`REQUIRED_AXIS_KEYS`).
        """
        ...


# Schema the CalibrationPlugin output is validated against. Mirrors
# imu_calibration.ImuCalibration.to_dict() (asdict of the dataclass).
REQUIRED_CALIBRATION_KEYS: tuple[str, ...] = (
    "device_label",
    "date",
    "sample_rate_hz",
    "duration_s",
    "source",
    "axes",
)
# Each axis entry (gx/gy/gz/ax/ay/az -> AxisParams) must carry these.
REQUIRED_AXIS_KEYS: tuple[str, ...] = (
    "random_walk",          # ARW (rate sensor) / VRW (linear sensor), units/sqrt(Hz)
    "bias_instability",     # B coefficient
    "rate_random_walk",     # RRW / K coefficient
)


# ---------------------------------------------------------------------------
# Registry (self-contained — no import of smoothers / imu_calibration)
# ---------------------------------------------------------------------------

_FUSION_PLUGINS: dict[str, FusionPlugin] = {}
_CALIBRATION_PLUGINS: dict[str, CalibrationPlugin] = {}

_T = TypeVar("_T")


def _coerce_instance(obj: Any) -> Any:
    """Accept a class OR an instance. If a class is passed, instantiate it
    with no args so the rest of the pipeline always deals with instances."""
    if inspect.isclass(obj):
        try:
            return obj()
        except Exception as e:  # noqa: BLE001
            raise PluginError(
                f"plugin class {obj.__name__!r} could not be instantiated "
                f"with no arguments: {type(e).__name__}: {e}"
            ) from e
    return obj


def _resolve_name(obj: Any, explicit: Optional[str]) -> str:
    name = explicit or getattr(obj, "name", None)
    if not name or not isinstance(name, str) or not name.strip():
        raise PluginError(
            "plugin must define a non-empty string `name` attribute "
            "(or pass an explicit name to the decorator); "
            f"got {getattr(obj, 'name', None)!r}"
        )
    return name.strip()


# ---- fusion ---------------------------------------------------------------

def register_fusion(name: Optional[str] = None) -> Callable[[_T], _T]:
    """Decorator: register a :class:`FusionPlugin` under ``name``.

    Usage::

        @register_fusion("my_fusion")
        class MyFusion:
            name = "my_fusion"
            def run(self, pos_rows, imu_rows, calibration, options): ...

    The plugin is validated at registration time; a non-conforming plugin
    raises :class:`PluginError` and is NOT registered.
    """
    def deco(obj: _T) -> _T:
        instance = _coerce_instance(obj)
        plugin_name = _resolve_name(instance, name)
        # Make the resolved name authoritative on the instance.
        try:
            instance.name = plugin_name
        except Exception:  # noqa: BLE001 - some objects are read-only
            pass
        validate_fusion_plugin(instance)
        _FUSION_PLUGINS[plugin_name] = instance
        return obj
    return deco


def register_calibration(name: Optional[str] = None) -> Callable[[_T], _T]:
    """Decorator: register a :class:`CalibrationPlugin` under ``name``.

    Validated at registration time; non-conforming plugins raise
    :class:`PluginError`.
    """
    def deco(obj: _T) -> _T:
        instance = _coerce_instance(obj)
        plugin_name = _resolve_name(instance, name)
        try:
            instance.name = plugin_name
        except Exception:  # noqa: BLE001
            pass
        validate_calibration_plugin(instance)
        _CALIBRATION_PLUGINS[plugin_name] = instance
        return obj
    return deco


def get_fusion_plugin(name: str) -> FusionPlugin:
    if name not in _FUSION_PLUGINS:
        raise KeyError(
            f"unknown fusion plugin {name!r}; "
            f"available: {list_fusion_plugins()}"
        )
    return _FUSION_PLUGINS[name]


def get_calibration_plugin(name: str) -> CalibrationPlugin:
    if name not in _CALIBRATION_PLUGINS:
        raise KeyError(
            f"unknown calibration plugin {name!r}; "
            f"available: {list_calibration_plugins()}"
        )
    return _CALIBRATION_PLUGINS[name]


def list_fusion_plugins() -> list[str]:
    """Registered fusion plugin names (registration order)."""
    return list(_FUSION_PLUGINS.keys())


def list_calibration_plugins() -> list[str]:
    """Registered calibration plugin names (registration order)."""
    return list(_CALIBRATION_PLUGINS.keys())


def unregister_fusion(name: str) -> None:
    """Remove a fusion plugin (no error if absent). Mainly for tests."""
    _FUSION_PLUGINS.pop(name, None)


def unregister_calibration(name: str) -> None:
    """Remove a calibration plugin (no error if absent). Mainly for tests."""
    _CALIBRATION_PLUGINS.pop(name, None)


def clear_registry() -> None:
    """Wipe both registries. Mainly for tests / hot reload."""
    _FUSION_PLUGINS.clear()
    _CALIBRATION_PLUGINS.clear()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_callable_params(
    fn: Any,
    required_params: Sequence[str],
    problems: list[str],
    *,
    label: str,
) -> None:
    """Confirm ``fn`` is callable and accepts every name in
    ``required_params`` (by position or keyword, or via ``**kwargs``)."""
    if not callable(fn):
        problems.append(f"{label} is not callable")
        return
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins / C funcs without an introspectable signature: accept.
        return
    params = sig.parameters
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    has_var_pos = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
    )
    if has_var_kw or has_var_pos:
        return  # *args/**kwargs swallow everything — accept.
    named = {
        n for n, p in params.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        ) and n != "self"
    }
    missing = [p for p in required_params if p not in named]
    if missing:
        problems.append(
            f"{label} signature is missing required parameter(s) "
            f"{missing}; expected ({', '.join(required_params)}). "
            f"Got: ({', '.join(params)})"
        )


def _synthetic_pos_rows(n: int = 3) -> list[PosRow]:
    base = 1_700_000_000.0
    rows = []
    for i in range(n):
        rows.append(PosRow(
            utc_s=base + i,
            lat_deg=37.0 + i * 1e-6,
            lon_deg=-122.0 + i * 1e-6,
            h_m=10.0 + i * 0.01,
            quality=1,
            vn=0.0, ve=0.0, vu=0.0,
            ns=10,
            sd_n=0.5, sd_e=0.5, sd_u=1.0,
            ratio=3.0, age_s=0.5,
        ))
    return rows


def _synthetic_imu_rows(n: int = 5) -> list[ImuRow]:
    base = 1_700_000_000.0
    return [
        ImuRow(utc_s=base + i * 0.005,
               ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0)
        for i in range(n)
    ]


def validate_fusion_plugin(plugin: Any) -> None:
    """Validate a fusion plugin against :class:`FusionPlugin`.

    Checks (all collected, then reported together):

    1. carries a non-empty ``name: str``;
    2. exposes a callable ``run`` whose signature accepts
       ``pos_rows, imu_rows, calibration, options``;
    3. on a tiny synthetic input, ``run`` returns a ``list`` whose elements
       are all :class:`PosRow`.

    Raises :class:`PluginError` listing every problem. Returns ``None`` on
    success.
    """
    problems: list[str] = []

    name = getattr(plugin, "name", None)
    if not isinstance(name, str) or not name.strip():
        problems.append(
            "missing non-empty string attribute `name`"
        )

    run = getattr(plugin, "run", None)
    if run is None:
        problems.append("missing required method `run`")
    else:
        _check_callable_params(
            run,
            ("pos_rows", "imu_rows", "calibration", "options"),
            problems,
            label="run",
        )

    # Smoke test only if structurally sound so far.
    if run is not None and callable(run) and not problems:
        try:
            out = run(
                _synthetic_pos_rows(),
                _synthetic_imu_rows(),
                None,
                {},
            )
        except Exception as e:  # noqa: BLE001
            problems.append(
                f"run() raised on synthetic input: {type(e).__name__}: {e}"
            )
        else:
            if not isinstance(out, list):
                problems.append(
                    f"run() must return list[PosRow]; got {type(out).__name__}"
                )
            else:
                bad = [
                    i for i, r in enumerate(out)
                    if not isinstance(r, PosRow)
                ]
                if bad:
                    sample = type(out[bad[0]]).__name__
                    problems.append(
                        f"run() returned non-PosRow element(s) at index "
                        f"{bad[:5]} (e.g. {sample}); every element must be a "
                        f"data_pipeline.parsers.PosRow"
                    )

    if problems:
        raise PluginError(
            f"FusionPlugin {name!r} failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def validate_calibration_plugin(plugin: Any) -> None:
    """Validate a calibration plugin against :class:`CalibrationPlugin`.

    Checks:

    1. carries a non-empty ``name: str``;
    2. exposes a callable ``compute`` whose signature accepts
       ``sensors_rows, options``;
    3. on a tiny synthetic input, ``compute`` returns a ``dict`` carrying
       every key in :data:`REQUIRED_CALIBRATION_KEYS`, with an ``axes`` map
       whose entries carry every key in :data:`REQUIRED_AXIS_KEYS`.

    Raises :class:`PluginError` listing every problem.
    """
    problems: list[str] = []

    name = getattr(plugin, "name", None)
    if not isinstance(name, str) or not name.strip():
        problems.append("missing non-empty string attribute `name`")

    compute = getattr(plugin, "compute", None)
    if compute is None:
        problems.append("missing required method `compute`")
    else:
        _check_callable_params(
            compute,
            ("sensors_rows", "options"),
            problems,
            label="compute",
        )

    if compute is not None and callable(compute) and not problems:
        try:
            out = compute(_synthetic_imu_rows(8), {"device_label": "test"})
        except Exception as e:  # noqa: BLE001
            problems.append(
                f"compute() raised on synthetic input: "
                f"{type(e).__name__}: {e}"
            )
        else:
            if not isinstance(out, dict):
                problems.append(
                    f"compute() must return a calibration dict; got "
                    f"{type(out).__name__}"
                )
            else:
                missing = [
                    k for k in REQUIRED_CALIBRATION_KEYS if k not in out
                ]
                if missing:
                    problems.append(
                        f"compute() output missing required key(s) {missing}; "
                        f"expected all of {list(REQUIRED_CALIBRATION_KEYS)}"
                    )
                axes = out.get("axes")
                if axes is None:
                    pass  # already covered by the missing-key check
                elif not isinstance(axes, dict):
                    problems.append(
                        f"compute() output 'axes' must be a dict; got "
                        f"{type(axes).__name__}"
                    )
                else:
                    for ax_name, ax_vals in axes.items():
                        av = (
                            ax_vals if isinstance(ax_vals, dict)
                            else getattr(ax_vals, "__dict__", None)
                        )
                        if not isinstance(av, dict):
                            problems.append(
                                f"axis {ax_name!r} entry is not a dict / "
                                f"dataclass; got {type(ax_vals).__name__}"
                            )
                            continue
                        miss_ax = [
                            k for k in REQUIRED_AXIS_KEYS if k not in av
                        ]
                        if miss_ax:
                            problems.append(
                                f"axis {ax_name!r} missing {miss_ax}; "
                                f"expected {list(REQUIRED_AXIS_KEYS)}"
                            )

    if problems:
        raise PluginError(
            f"CalibrationPlugin {name!r} failed validation:\n  - "
            + "\n  - ".join(problems)
        )
