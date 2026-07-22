"""GUI wiring tests for the export-source / final-velocity controls and the
camera-model accuracy report button.

1. Export-source + final-velocity controls: ``var_exp_source`` /
   ``var_emit_final_vel`` / ``var_vel_disagree`` exist with neutral defaults
   and feed ``_build_export_source_options()`` -> ``(source,
   emit_final_velocity, vel_disagree_threshold_mps)``. Neutral defaults keep
   the client export byte-identical (source=None, False, None).

2. Camera-model accuracy report button (Analysis tab) is present and bound
   to ``_run_camera_report``.

Headless-safe: skipped entirely when no Tk display is available (mirrors
tests/test_gui_export_wiring.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tkinter

_TK_AVAILABLE = False
for _attempt in range(3):  # Tk init is transiently flaky under load — retry
    try:
        _root = tkinter.Tk()
        _root.withdraw()
        _root.destroy()
        _TK_AVAILABLE = True
        break
    except Exception:
        import time as _time
        _time.sleep(0.5)
pytestmark = pytest.mark.skipif(not _TK_AVAILABLE, reason="No Tk display")


@pytest.fixture
def app(tmp_path: Path):
    import data_pipeline.gui as gui_mod
    orig_recent = gui_mod._RECENT_FILE
    gui_mod._RECENT_FILE = tmp_path / "recent.json"
    from data_pipeline.gui import App
    a = App()
    a.root.update()
    try:
        yield a
    finally:
        try:
            a.root.destroy()
        except Exception:
            pass
        gui_mod._RECENT_FILE = orig_recent


def _find_widgets(root, klass):
    got = []

    def walk(w):
        if isinstance(w, klass):
            got.append(w)
        for c in w.winfo_children():
            walk(c)

    walk(root)
    return got


# ---------------------------------------------------------------------------
# Export-source + final-velocity controls
# ---------------------------------------------------------------------------

def test_export_source_vars_exist_with_neutral_defaults(app):
    assert app.var_exp_source.get() == app.EXPORT_SOURCE_AS_RUN
    assert app.var_emit_final_vel.get() is False
    assert app.var_vel_disagree.get() == ""


def test_export_source_options_default_is_neutral(app):
    """Defaults must keep the client export byte-identical."""
    source, emit_fv, thr = app._build_export_source_options()
    assert source is None
    assert emit_fv is False
    assert thr is None


def test_export_source_options_reads_controls(app):
    app.var_exp_source.set("raw")
    app.var_emit_final_vel.set(True)
    app.var_vel_disagree.set("1.5")
    source, emit_fv, thr = app._build_export_source_options()
    assert source == "raw"
    assert emit_fv is True
    assert thr == pytest.approx(1.5)


def test_export_source_options_bad_threshold_disables_gate(app):
    app.var_vel_disagree.set("not-a-number")
    *_, thr = app._build_export_source_options()
    assert thr is None
    app.var_vel_disagree.set("-2")
    *_, thr = app._build_export_source_options()
    assert thr is None


def test_export_source_combobox_lists_raw_and_smoothers(app):
    import tkinter.ttk as ttk
    from data_pipeline.smoothers import list_smoothers
    combos = _find_widgets(app.root, ttk.Combobox)
    target = [c for c in combos
              if str(c.cget("textvariable")) == str(app.var_exp_source)]
    assert len(target) == 1
    values = list(target[0].cget("values"))
    assert values[0] == app.EXPORT_SOURCE_AS_RUN
    assert "raw" in values
    for name in list_smoothers():
        assert name in values


# ---------------------------------------------------------------------------
# Camera-model accuracy report button (Analysis tab)
# ---------------------------------------------------------------------------

def test_camera_report_button_present(app):
    import tkinter.ttk as ttk
    labels = [b.cget("text") for b in _find_widgets(app.root, ttk.Button)]
    assert "Camera-model accuracy report" in labels


def test_camera_report_handler_exists(app):
    assert callable(getattr(app, "_run_camera_report", None))
