"""End-to-end-ish GUI smoke tests for the seven tabs.

These tests instantiate the real ``App`` and exercise every interactive
helper (recent menu, drag-drop registration, T02 converter switching,
Post-processing auto-fill, adaptive-mode wiring, etc.) WITHOUT calling
``mainloop()`` and WITHOUT shelling out to the external converter / the solver binary. Heavy
external calls are monkey-patched out so the suite stays under a
second per test.

Goal: catch wiring regressions (missing variables, broken callbacks,
mis-typed attribute names) that a pure smoke instantiate would let
through.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import tkinter
import tkinter.ttk as ttk


# Skip the whole module if a display is not available (CI sometimes
# has no Tk; locally Windows always has).
_TK_AVAILABLE = True
try:
    _root = tkinter.Tk()
    _root.withdraw()
    _root.destroy()
except Exception:
    _TK_AVAILABLE = False
pytestmark = pytest.mark.skipif(not _TK_AVAILABLE, reason="No Tk display")


@pytest.fixture
def app(tmp_path: Path):
    """Construct the App, wait for it to build, yield, then tear down."""
    # Isolate the recent-projects file so the test doesn't leak into the
    # user's real ``~/.data_pipeline_recent.json``.
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


def _notebook(app) -> ttk.Notebook:
    def find(w):
        if isinstance(w, ttk.Notebook):
            return w
        for c in w.winfo_children():
            r = find(c)
            if r:
                return r
        return None
    return find(app.root)


# ---------------------------------------------------------------------------
# Tab inventory
# ---------------------------------------------------------------------------

def test_core_tabs_present(app):
    """Smoke: core tabs present regardless of complexity mode."""
    nb = _notebook(app)
    tabs = [nb.tab(i, "text") for i in range(nb.index("end"))]
    assert len(tabs) >= 4, f"expected at least 4 tabs, got {tabs}"
    flat = " | ".join(tabs).lower()
    for tag in ["inputs", "rinex", "frames + csv", "viewers"]:
        assert tag in flat, f"missing tab containing {tag!r}"


# ---------------------------------------------------------------------------
# Critical run methods exist + are callable on the instance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "_run_rinex", "_run_ppk", "_run_frames", "_run_csv",
    "_run_frames_and_csv", "_run_t02", "_run_video_only",
    "_run_traj_viewer", "_run_orient_panel", "_run_compare_viewer",
    "_run_sync_player", "_run_vel_viewer", "_run_geo_viewer",
    "_run_speed_vs_gt_viewer",
    "_compute_adaptive_indices_if_requested",
    "_on_fps_mode_changed",
    "_on_t02_input_changed", "_on_t02_converter_changed",
    "_ppk_autodetect_nav", "_ppk_pick_preset",
    "_open_folder", "_open_last_output",
    "_push_recent", "_rebuild_recent_menu",
    "_refresh_preview", "_update_rotation_preview",
    "_register_dnd_folder", "_register_dnd_video", "_register_dnd_nav",
])
def test_method_present(app, name):
    assert hasattr(app, name), f"App missing {name}"
    assert callable(getattr(app, name)), f"{name} is not callable"


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------

def test_initial_state(app):
    p = app.paths
    assert p.raw is None and p.out_dir is None
    assert p.pos_path is None
    # Adaptive mode defaults
    assert app.var_fps_mode.get() == "fixed"
    assert app.var_adapt_spacing_m.get() == 2.0
    assert app.var_adapt_turn_overlap.get() == 0.8
    # T02 defaults
    assert app.var_t02_conv.get() == "trimble"


def test_recent_projects_roundtrip(app, tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    app._push_recent(str(a))
    app._push_recent(str(b))
    app._push_recent(str(c))
    assert len(app._recent) == 3
    # Most-recent first
    assert Path(app._recent[0]) == Path(str(c))
    # Dedup
    app._push_recent(str(b))
    assert len(app._recent) == 3
    assert Path(app._recent[0]) == Path(str(b))
    # File persisted
    import data_pipeline.gui as gui_mod
    assert gui_mod._RECENT_FILE.is_file()
    saved = json.loads(gui_mod._RECENT_FILE.read_text())
    assert saved == app._recent


def test_t02_converter_switch_resets_tool_path(app):
    app.var_t02_conv.set("trimble")
    app._on_t02_converter_changed()
    trimble_default = app.var_t02_tool.get()
    assert "runpkr00" in trimble_default.lower()
    app.var_t02_conv.set("jps2rin")
    app._on_t02_converter_changed()
    assert "jps2rin" in app.var_t02_tool.get().lower()
    app.var_t02_conv.set("convbin")
    app._on_t02_converter_changed()
    assert "convbin" in app.var_t02_tool.get().lower()


def test_adaptive_compute_requires_pos(app):
    """Adaptive mode must surface a clear error when the .pos isn't set."""
    app.var_fps_mode.set("adaptive")
    # No RAW or .pos in paths -> should hit the messagebox guard and return None.
    with patch("tkinter.messagebox.showerror") as mb:
        result = app._compute_adaptive_indices_if_requested("adaptive")
    assert result is None
    assert mb.called


def test_open_last_output_no_path(app):
    """Should show messagebox.showinfo when out_dir is None, never crash."""
    app.paths.out_dir = None
    with patch("tkinter.messagebox.showinfo") as mb:
        app._open_last_output()
    assert mb.called


def test_doctor_status_attrs(app):
    """Critical status attributes exist so _set_busy doesn't AttributeError."""
    assert hasattr(app, "_status_lbl")
    assert hasattr(app, "_dot_canvas")
    assert hasattr(app, "_progress_bar")


def test_busy_transitions(app):
    """_set_busy / _show_progress_bar / _hide_progress_bar don't crash.

    Note: ``winfo_ismapped`` requires the geometry manager to have
    processed pending pack/grid requests, which means we need
    ``update_idletasks()`` (or a full ``update()`` after a brief delay).
    A single ``root.update()`` is not enough on Windows.
    """
    app._set_busy(True, "test stage")
    app.root.update_idletasks()
    app.root.update()
    assert "Running" in app._status_lbl.cget("text")
    app._show_progress_bar(indeterminate=True)
    app.root.update_idletasks()
    app.root.update()
    # Confirm the running flag tracks even if winfo_ismapped lags.
    assert app._progress_running, "progress bar should be marked running"
    app._hide_progress_bar()
    app.root.update_idletasks()
    app.root.update()
    assert not app._progress_running
    app._set_busy(False)
    app.root.update_idletasks()
    app.root.update()
    assert app._status_lbl.cget("text") == "Ready"


def test_ppk_show_command_missing_inputs(app):
    """The 'Show command' button must not crash when inputs are empty."""
    with patch("tkinter.messagebox.showerror") as mb:
        app._ppk_show_command()
    assert mb.called  # informed the user instead of crashing


def test_video_only_input_triggers_output_default(app, tmp_path):
    """Picking a media file populates the default output folder."""
    fake_vid = tmp_path / "clip.mp4"
    fake_vid.write_bytes(b"fake")
    app.var_vo_video.set(str(fake_vid))
    app._on_vo_video_changed()
    # Output dir should now be the parent dir or a derived subfolder.
    assert app.var_vo_out.get(), "expected output folder default"
    assert Path(app.var_vo_out.get()).parent == tmp_path or \
           Path(app.var_vo_out.get()) == tmp_path


# ---------------------------------------------------------------------------
# Style / palette wired
# ---------------------------------------------------------------------------

def test_style_palette(app):
    for attr in ("_bg", "_fg", "_accent", "_accent_hi",
                 "_good_green", "_warn_amber", "_err_red"):
        assert hasattr(app, attr), f"palette token {attr} missing"


def test_progress_bar_styled(app):
    style = ttk.Style(app.root)
    # The TProgressbar style should resolve a background colour. Empty
    # string would mean the style block never registered.
    assert style.lookup("TProgressbar", "background")
