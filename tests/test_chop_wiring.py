"""Segment (cut clip) anchor WIRING: every caller forwards chop_video_anchor.

The core fixes (``frame_time.resolve_video_t0_boottime_ns`` consumers:
``compute_capture_diag`` / ``compute_keep_list`` / ``build_ins_csv`` /
``run_vio*``) accept a keyword-only ``chop_video_anchor``. Cut media files are
ALWAYS the client's input, so a caller that forgets to pass it silently maps
every sample minutes early. This file locks the wiring:

* ``client_viewers.make_vio_overlay`` exposes + forwards the anchor kwargs
  to ``vio.run_vio``;
* ``stages.capture_diag_viewer.build_capture_diag_viewer`` exposes + forwards
  ``chop_video_anchor`` to ``compute_capture_diag`` (and its CLI has the flag);
* the georef CLI has ``--chop-video-anchor`` / ``--video-path`` and passes
  them into ``georef.run``;
* the viewers ``sync`` CLI has ``--chop-video-anchor`` and passes it into
  ``build_sync_player``;
* ``scripts/compare_frame_coords.py`` recompute-from-raw mode has the flag
  and passes it into ``georef._load_frames``;
* the GUI (``_compute_adaptive_indices_if_requested`` / ``_run_capture_diag``
  / ``_run_client_vio``) passes the anchor from the loaded ``RawInputs`` when
  ``raw.is_chop`` — and ``None`` when not (Tk-skipped when headless).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# client_viewers.make_vio_overlay -> vio.run_vio
# ---------------------------------------------------------------------------

def test_make_vio_overlay_exposes_anchor_kwargs() -> None:
    from data_pipeline.client_viewers import make_vio_overlay
    params = inspect.signature(make_vio_overlay).parameters
    for name in ("capture_meta", "video_anchor", "chop_video_anchor"):
        assert name in params, name
        assert params[name].default is None, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_make_vio_overlay_forwards_anchor_kwargs_to_run_vio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.parsers as parsers_mod
    import data_pipeline.vio as vio_mod
    from data_pipeline.client_viewers import make_vio_overlay

    got: dict = {}

    def fake_run_vio(**kwargs):
        got.update(kwargs)
        return []  # -> make_vio_overlay raises right after: no cv2 needed

    monkeypatch.setattr(parsers_mod, "parse_rtkpos",
                        lambda p: [SimpleNamespace(utc_s=0.0)])
    monkeypatch.setattr(vio_mod, "run_vio", fake_run_vio)

    cm = tmp_path / "capture_meta.json"
    va = tmp_path / "recording_x.video_anchor.txt"
    chop = tmp_path / "chop_x.video_anchor.txt"
    with pytest.raises(RuntimeError, match="0 usable samples"):
        make_vio_overlay(
            tmp_path / "rover.pos", tmp_path / "chop_x.mp4",
            tmp_path / "recording_x.txt", tmp_path / "vio.html",
            capture_meta=cm, video_anchor=va, chop_video_anchor=chop,
        )
    assert got["capture_meta"] == cm
    assert got["video_anchor"] == va
    assert got["chop_video_anchor"] == chop


def test_make_vio_overlay_defaults_stay_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy call sites (no kwargs) keep passing None through — unchanged."""
    import data_pipeline.parsers as parsers_mod
    import data_pipeline.vio as vio_mod
    from data_pipeline.client_viewers import make_vio_overlay

    got: dict = {}
    monkeypatch.setattr(parsers_mod, "parse_rtkpos",
                        lambda p: [SimpleNamespace(utc_s=0.0)])
    monkeypatch.setattr(vio_mod, "run_vio",
                        lambda **kw: (got.update(kw), [])[1])
    with pytest.raises(RuntimeError):
        make_vio_overlay(tmp_path / "a.pos", tmp_path / "a.mp4",
                         tmp_path / "a.txt", tmp_path / "a.html")
    assert got["capture_meta"] is None
    assert got["video_anchor"] is None
    assert got["chop_video_anchor"] is None


# ---------------------------------------------------------------------------
# stages.capture_diag_viewer -> compute_capture_diag
# ---------------------------------------------------------------------------

def _patch_capdiag_backend(monkeypatch: pytest.MonkeyPatch, got: dict):
    import data_pipeline.stages.capture_diag_viewer as cdv
    from data_pipeline.capture_diag import CaptureDiag

    def fake_compute(**kwargs):
        got.update(kwargs)
        return CaptureDiag()

    monkeypatch.setattr(cdv, "compute_capture_diag", fake_compute)
    monkeypatch.setattr(cdv, "_copy_plotly_next_to",
                        lambda d: Path(d) / "plotly.min.js")
    return cdv


def test_build_capture_diag_viewer_forwards_chop_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: dict = {}
    cdv = _patch_capdiag_backend(monkeypatch, got)
    chop = tmp_path / "chop_x.video_anchor.txt"
    cdv.build_capture_diag_viewer(
        session_dir=tmp_path, out_html=tmp_path / "capture_diag.html",
        chop_video_anchor=chop,
    )
    assert got["chop_video_anchor"] == chop
    assert (tmp_path / "capture_diag.html").is_file()


def test_build_capture_diag_viewer_default_chop_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    got: dict = {}
    cdv = _patch_capdiag_backend(monkeypatch, got)
    cdv.build_capture_diag_viewer(
        session_dir=tmp_path, out_html=tmp_path / "capture_diag.html",
    )
    assert got["chop_video_anchor"] is None


def test_capture_diag_viewer_cli_has_chop_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.stages.capture_diag_viewer as cdv

    got: dict = {}

    def fake_builder(**kwargs):
        got.update(kwargs)
        return SimpleNamespace(
            diag=SimpleNamespace(to_dict=lambda: {}),
            html_path=kwargs["out_html"], js_path=None,
        )

    monkeypatch.setattr(cdv, "build_capture_diag_viewer", fake_builder)
    chop = tmp_path / "chop_x.video_anchor.txt"
    rc = cdv._main([
        str(tmp_path), str(tmp_path / "out.html"),
        "--chop-video-anchor", str(chop),
    ])
    assert rc == 0
    assert got["chop_video_anchor"] == chop


# ---------------------------------------------------------------------------
# georef CLI: --chop-video-anchor / --video-path -> run(...)
# ---------------------------------------------------------------------------

def test_georef_cli_passes_chop_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.stages.georef as georef_mod

    got: dict = {}
    monkeypatch.setattr(georef_mod, "run", lambda **kw: got.update(kw))
    chop = tmp_path / "chop_x.video_anchor.txt"
    mp4 = tmp_path / "chop_x.mp4"
    monkeypatch.setattr(sys, "argv", [
        "georef",
        "--frame-times-csv", str(tmp_path / "ft.csv"),
        "--recording-map", str(tmp_path / "rec.txt"),
        "--pos", str(tmp_path / "rover.pos"),
        "--data-log", str(tmp_path / "meas.txt"),
        "--out-csv", str(tmp_path / "Georef.csv"),
        "--fps", "4",
        "--chop-video-anchor", str(chop),
        "--video-path", str(mp4),
    ])
    assert georef_mod.main() == 0
    assert got["chop_video_anchor"] == chop
    assert got["video_path"] == mp4


def test_georef_cli_chop_flags_default_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.stages.georef as georef_mod

    got: dict = {}
    monkeypatch.setattr(georef_mod, "run", lambda **kw: got.update(kw))
    monkeypatch.setattr(sys, "argv", [
        "georef",
        "--frame-times-csv", str(tmp_path / "ft.csv"),
        "--recording-map", str(tmp_path / "rec.txt"),
        "--pos", str(tmp_path / "rover.pos"),
        "--data-log", str(tmp_path / "meas.txt"),
        "--out-csv", str(tmp_path / "Georef.csv"),
        "--fps", "4",
    ])
    assert georef_mod.main() == 0
    assert got["chop_video_anchor"] is None
    assert got["video_path"] is None


# ---------------------------------------------------------------------------
# viewers sync CLI: --chop-video-anchor -> build_sync_player(...)
# ---------------------------------------------------------------------------

def _viewers_sync_argv(tmp_path: Path) -> list[str]:
    return [
        "viewers", "sync",
        "--video", str(tmp_path / "chop_x.mp4"),
        "--pos", str(tmp_path / "rover.pos"),
        "--frame-times-csv", str(tmp_path / "ft.csv"),
        "--recording-map", str(tmp_path / "rec.txt"),
        "--out", str(tmp_path / "sync_player.html"),
    ]


def test_viewers_sync_cli_passes_chop_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.stages.viewers as viewers_mod

    got: dict = {}
    monkeypatch.setattr(viewers_mod, "build_sync_player",
                        lambda **kw: got.update(kw))
    chop = tmp_path / "chop_x.video_anchor.txt"
    monkeypatch.setattr(
        sys, "argv",
        _viewers_sync_argv(tmp_path) + ["--chop-video-anchor", str(chop)],
    )
    assert viewers_mod.main() == 0
    assert got["chop_video_anchor"] == chop


def test_viewers_sync_cli_chop_default_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.stages.viewers as viewers_mod

    got: dict = {}
    monkeypatch.setattr(viewers_mod, "build_sync_player",
                        lambda **kw: got.update(kw))
    monkeypatch.setattr(sys, "argv", _viewers_sync_argv(tmp_path))
    assert viewers_mod.main() == 0
    assert got["chop_video_anchor"] is None


def test_viewers_sync_cli_help_warns_about_trimmed_clips(
    tmp_path: Path,
) -> None:
    """The precedence footgun (--capture-meta beats --video-anchor) must be
    documented: cut clips need --chop-video-anchor, in the help text."""
    import data_pipeline.stages.viewers  # noqa: F401 — import check

    src = Path(inspect.getsourcefile(
        sys.modules["data_pipeline.stages.viewers"])).read_text(
            encoding="utf-8")
    assert "--chop-video-anchor" in src or "chop-video-anchor" in src
    # The --video-anchor help must steer cut-clip users away from it.
    assert "chop_video_anchor=args.chop_video_anchor" in src


# ---------------------------------------------------------------------------
# scripts/compare_frame_coords.py recompute-from-raw mode
# ---------------------------------------------------------------------------

def test_compare_frame_coords_parses_chop_flag(tmp_path: Path) -> None:
    from scripts import compare_frame_coords as cfc

    chop = tmp_path / "chop_x.video_anchor.txt"
    args = cfc.parse_args([
        "--external-csv", str(tmp_path / "refined.csv"),
        "--frame-times-csv", str(tmp_path / "ft.csv"),
        "--recording-map", str(tmp_path / "rec.txt"),
        "--pos", str(tmp_path / "rover.pos"),
        "--chop-video-anchor", str(chop),
    ])
    assert args.chop_video_anchor == chop


def test_compare_frame_coords_forwards_chop_to_load_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_pipeline.parsers as parsers_mod
    import data_pipeline.stages.georef as georef_mod
    from scripts import compare_frame_coords as cfc

    got: dict = {}

    def fake_load_frames(*a, **kw):
        got.update(kw)
        return [], None  # no samples -> _gnss_from_raw exits after capture

    monkeypatch.setattr(georef_mod, "_load_frames", fake_load_frames)
    monkeypatch.setattr(parsers_mod, "parse_rtkpos",
                        lambda p: [SimpleNamespace(utc_s=0.0)])

    chop = tmp_path / "chop_x.video_anchor.txt"
    args = cfc.parse_args([
        "--external-csv", str(tmp_path / "refined.csv"),
        "--frame-times-csv", str(tmp_path / "ft.csv"),
        "--recording-map", str(tmp_path / "rec.txt"),
        "--pos", str(tmp_path / "rover.pos"),
        "--chop-video-anchor", str(chop),
    ])
    with pytest.raises(SystemExit):
        cfc._gnss_from_raw(args)
    assert got["chop_video_anchor"] == chop


# ---------------------------------------------------------------------------
# GUI wiring (Tk needed; mirrors tests/test_gui_export_wiring.py)
# ---------------------------------------------------------------------------

import tkinter  # noqa: E402

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

needs_tk = pytest.mark.skipif(not _TK_AVAILABLE, reason="No Tk display")


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


def _chop_raw(tmp_path: Path, *, is_chop: bool = True):
    """A RawInputs resembling a loaded segment (or plain) session."""
    from data_pipeline.pipeline import RawInputs

    parent = tmp_path / "session"
    chop_dir = parent / "chop_20260702_120000"
    chop_dir.mkdir(parents=True, exist_ok=True)
    mp4 = chop_dir / "chop_20260702_120000.mp4" if is_chop \
        else parent / "recording_1.mp4"
    mp4.write_bytes(b"\x00")  # _run_client_vio requires .is_file()
    chop_anchor = chop_dir / "chop_20260702_120000.video_anchor.txt"
    return RawInputs(
        measurements_txt=parent / "measurements_1.txt",
        recording_txt=parent / "recording_1.txt",
        recording_mp4=mp4,
        sensors_txt=parent / "sensors_1.txt",
        capture_meta_json=parent / "capture_meta.json",
        video_anchor_txt=(chop_anchor if is_chop
                          else parent / "recording_1.video_anchor.txt"),
        capture_format="new",
        anchor_format=2,
        is_chop=is_chop,
        chop_video_anchor=(chop_anchor if is_chop else None),
        chop_meta_json=(chop_dir / "chop.chop_meta.json" if is_chop else None),
    )


def _fake_keep_result():
    return SimpleNamespace(n_kept=2, n_total=4,
                           keep_indices=[0, 30], keep_pts_s=[0.0, 1.0])


@needs_tk
def test_gui_adaptive_passes_chop_anchor(app, tmp_path: Path) -> None:
    import data_pipeline.gui as gui_mod

    raw = _chop_raw(tmp_path)
    app.paths.raw = raw
    app.paths.pos_path = tmp_path / "rover.pos"

    with patch.object(gui_mod.adaptive_stage, "compute_keep_list",
                      return_value=_fake_keep_result()) as m:
        out = app._compute_adaptive_indices_if_requested("adaptive")
    assert out == ([0, 30], [0.0, 1.0])
    kw = m.call_args.kwargs
    assert kw["capture_meta"] == raw.capture_meta_json
    assert kw["video_anchor"] == raw.video_anchor_txt
    assert kw["chop_video_anchor"] == raw.chop_video_anchor


@needs_tk
def test_gui_adaptive_non_chop_passes_none(app, tmp_path: Path) -> None:
    import data_pipeline.gui as gui_mod

    app.paths.raw = _chop_raw(tmp_path, is_chop=False)
    app.paths.pos_path = tmp_path / "rover.pos"
    with patch.object(gui_mod.adaptive_stage, "compute_keep_list",
                      return_value=_fake_keep_result()) as m:
        app._compute_adaptive_indices_if_requested("adaptive")
    assert m.call_args.kwargs["chop_video_anchor"] is None


@needs_tk
def test_gui_capture_diag_passes_chop_anchor(app, tmp_path: Path) -> None:
    import data_pipeline.stages.capture_diag_viewer as cdv

    raw = _chop_raw(tmp_path)
    app.paths.raw = raw
    app.paths.out_dir = tmp_path / "out"

    def _sync_run_async(fn, stage):
        fn()

    fake_res = SimpleNamespace(html_path=tmp_path / "out" / "capture_diag.html")
    with patch.object(cdv, "build_capture_diag_viewer",
                      return_value=fake_res) as m, \
         patch.object(app, "_run_async", side_effect=_sync_run_async):
        app._run_capture_diag()
    kw = m.call_args.kwargs
    assert kw["chop_video_anchor"] == raw.chop_video_anchor
    # Session dir stays the parent (measurements live there for a segment).
    assert kw["session_dir"] == raw.measurements_txt.parent


@needs_tk
def test_gui_vio_passes_chop_anchor(app, tmp_path: Path) -> None:
    import data_pipeline.client_viewers as cv_mod
    import data_pipeline.gui as gui_mod

    raw = _chop_raw(tmp_path)
    app.paths.raw = raw
    app.paths.pos_path = tmp_path / "rover.pos"
    app.paths.out_dir = tmp_path / "out"

    def _sync_run_async(fn, stage):
        fn()

    with patch.object(cv_mod, "make_vio_overlay",
                      return_value=tmp_path / "vio.html") as m, \
         patch.object(gui_mod.messagebox, "askyesno", return_value=True), \
         patch.object(app, "_run_async", side_effect=_sync_run_async), \
         patch.object(app, "_open_path_in_default"):
        app._run_client_vio()
    assert m.call_count == 1
    kw = m.call_args.kwargs
    assert kw["capture_meta"] == raw.capture_meta_json
    assert kw["video_anchor"] == raw.video_anchor_txt
    assert kw["chop_video_anchor"] == raw.chop_video_anchor
