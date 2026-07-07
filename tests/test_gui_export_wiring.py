"""GUI wiring tests for the two shipped export backends.

1. Time-basis controls: ``var_tb_*`` checkboxes feed
   ``_build_export_options()`` -> ``(coords, smooth_z, z_sigma, time_bases,
   audio_start_utc_s)`` in a stable order; default stays ``("reference time",)`` (byte
   -identical legacy export); 'stream' resolves the session anchor or is
   dropped with a warning.

2. Media+stream export button: ``_run_export_av`` discovers clips via
   ``combine_av.discover_videos``, plans + runs the mux on the worker, and
   opens the output.

Headless-safe: skipped entirely when no Tk display is available, and no
the external converter / real session data is ever touched (backends are monkey-patched).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Task A: time-basis controls
# ---------------------------------------------------------------------------

def test_time_basis_vars_exist_with_defaults(app):
    assert app.var_tb_gpst.get() is True
    assert app.var_tb_utc.get() is False
    assert app.var_tb_audio.get() is False
    assert app.var_tb_iso.get() is False


def test_export_options_default_is_gpst_only(app):
    """Defaults must keep the legacy export byte-identical."""
    coords, smooth_z, z_sigma, time_bases, audio_utc = (
        app._build_export_options())
    assert time_bases == ("gpst",)
    assert audio_utc is None
    # Pre-existing fields unchanged by the extension.
    assert coords is None
    assert smooth_z is True
    assert z_sigma == 3.0


def test_export_options_nothing_ticked_falls_back_to_gpst(app):
    app.var_tb_gpst.set(False)
    *_, time_bases, audio_utc = app._build_export_options()
    assert time_bases == ("gpst",)
    assert audio_utc is None


def test_export_options_stable_order(app):
    # Tick in "wrong" order — result must be the canonical order.
    app.var_tb_iso.set(True)
    app.var_tb_utc.set(True)
    *_, time_bases, audio_utc = app._build_export_options()
    assert time_bases == ("gpst", "utc", "iso")
    assert audio_utc is None


def test_export_options_audio_dropped_without_session(app):
    """No RAW session loaded -> 'stream' dropped with a warning, no crash."""
    app.var_tb_audio.set(True)
    assert app.paths.raw_folder is None
    *_, time_bases, audio_utc = app._build_export_options()
    assert "audio" not in time_bases
    assert time_bases == ("gpst",)
    assert audio_utc is None


def test_export_options_audio_resolved_from_anchors(app, tmp_path):
    """'stream' ticked + session loaded -> audio_start_utc_s from anchors."""

    class _BootAnchor:
        @staticmethod
        def boottime_to_utc_s(boot_ns):
            return boot_ns / 1e9 + 1_000_000.0

    class _Anchors:
        boot_anchor = _BootAnchor()
        boot_anchor_source = "recording-map"
        audio_start_boot_ns = 5_000_000_000.0  # 5 s after boot

    app.paths.raw_folder = tmp_path  # pretend a session is loaded
    app.var_tb_audio.set(True)
    with patch("data_pipeline.audio_frame_export.resolve_session_anchors",
               return_value=_Anchors()) as m:
        *_, time_bases, audio_utc = app._build_export_options()
    assert m.call_count == 1
    assert time_bases == ("gpst", "audio")
    assert audio_utc == pytest.approx(1_000_005.0)


def test_export_options_audio_dropped_on_anchor_failure(app, tmp_path):
    app.paths.raw_folder = tmp_path
    app.var_tb_audio.set(True)
    with patch("data_pipeline.audio_frame_export.resolve_session_anchors",
               side_effect=ValueError("no audio_anchor_*.txt")):
        *_, time_bases, audio_utc = app._build_export_options()
    assert time_bases == ("gpst",)
    assert audio_utc is None


def test_export_trajectory_accepts_gui_option_tuple(app, tmp_path):
    """The 5-tuple threads straight into export_trajectory (real backend)."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.user_export import export_trajectory

    coords, smooth_z, z_sigma, time_bases, audio_utc = (
        app._build_export_options())
    rows = [
        PosRow(utc_s=1_400_000_000.0 + i, lat_deg=52.0 + i * 1e-6,
               lon_deg=13.0, h_m=40.0, quality=1, ns=10)
        for i in range(5)
    ]
    out = tmp_path / "client.csv"
    res = export_trajectory(
        rows, out, source_tag="test",
        coord_systems=coords, smooth_z=smooth_z, z_sigma_s=z_sigma,
        time_bases=time_bases, audio_start_utc_s=audio_utc,
    )
    assert out.is_file()
    assert tuple(res.time_bases) == ("gpst",)


# ---------------------------------------------------------------------------
# Task B: media + stream export button
# ---------------------------------------------------------------------------

def _find_widgets(root, klass):
    got = []

    def walk(w):
        if isinstance(w, klass):
            got.append(w)
        for c in w.winfo_children():
            walk(c)

    walk(root)
    return got


def test_export_av_button_present(app):
    import tkinter.ttk as ttk
    labels = [b.cget("text") for b in _find_widgets(app.root, ttk.Button)]
    assert "Export video + audio (full or crop)" in labels


def test_run_export_av_single_clip_muxes_and_opens(app, tmp_path):
    """One discovered clip -> no chooser; plan+run on worker; open output."""
    from data_pipeline import combine_av

    class _Clip:
        kind = "full"
        label = "recording_1.mp4 (full)"
        mp4 = tmp_path / "recording_1.mp4"

    class _Plan:
        audio_seek_s = 1.25
        ppm = -3.2
        atempo = None
        out_path = tmp_path / "combined_recording_1.mp4"
        warnings = ["synthetic warning"]
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", "x"]

    ran, opened = [], []
    app.paths.raw_folder = tmp_path

    # Run the worker inline so the test is deterministic (no thread).
    def _sync_run_async(fn, stage):
        fn()

    with patch.object(combine_av, "discover_videos",
                      return_value=[_Clip()]) as m_disc, \
         patch.object(combine_av, "plan_mux",
                      return_value=_Plan()) as m_plan, \
         patch.object(combine_av, "run_mux",
                      side_effect=lambda p: (ran.append(p), p.out_path)[1]), \
         patch.object(app, "_run_async", side_effect=_sync_run_async), \
         patch.object(app, "_open_path_in_default",
                      side_effect=lambda p: opened.append(p)):
        app._run_export_av()
        app.root.update()  # flush the root.after(0, ...) open callback

    assert m_disc.call_args[0][0] == tmp_path
    assert m_plan.call_count == 1
    assert m_plan.call_args.kwargs["which"].label == "recording_1.mp4 (full)"
    assert len(ran) == 1
    assert opened == [_Plan.out_path]


def test_run_export_av_no_clips_shows_error_not_crash(app, tmp_path):
    from data_pipeline import combine_av
    app.paths.raw_folder = tmp_path
    with patch.object(combine_av, "discover_videos", return_value=[]), \
         patch("data_pipeline.gui.messagebox.showerror") as m_err:
        app._run_export_av()
    assert m_err.call_count == 1
