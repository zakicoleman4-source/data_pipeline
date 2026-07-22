"""Regression tests for the post-processed log-flood GUI crash.

rnx2rtkp emits one ``\r``-terminated progress line to stderr per epoch
(unconditionally -- no isatty check), so a long / high-rate session captures
100k-500k progress "lines". The old code re-logged all of them into the GUI
log queue at once and retained the full streams on PpkResult, which blocked
the Tk mainloop for minutes and could kill the process via Tcl_Panic when
Tk's text B-tree allocation failed.

These tests are headless-safe: nothing here creates a Tk root.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from data_pipeline.stages import ppk


# ---------------------------------------------------------------------------
# _tail_lines: collapse the \r progress stream
# ---------------------------------------------------------------------------

def test_tail_lines_collapses_huge_cr_stream_fast() -> None:
    # 200 000 progress entries joined by bare \r, exactly like rnx2rtkp.
    entries = [f"processing : 2024/01/01 00:{i % 60:02d}  Q=1" for i in range(200_000)]
    blob = "\r".join(entries) + "\r"

    t0 = time.perf_counter()
    tail = ppk._tail_lines(blob, 50)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"_tail_lines too slow: {elapsed:.3f}s"
    lines = tail.splitlines()
    assert len(lines) <= 50
    # It must be the LAST entries, in order.
    assert lines == entries[-50:]


def test_tail_lines_default_n_is_50_and_drops_empties() -> None:
    blob = "\r\r\n\na\r\rb\nc\r\n\r\n"
    assert ppk._tail_lines(blob) == "a\nb\nc"
    # Default n
    many = "\n".join(str(i) for i in range(500))
    assert len(ppk._tail_lines(many).splitlines()) == 50


def test_tail_lines_empty_input() -> None:
    assert ppk._tail_lines("", 50) == ""


# ---------------------------------------------------------------------------
# PpkResult must not retain the full multi-MB streams
# ---------------------------------------------------------------------------

def _stub_resolve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "rnx2rtkp.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(ppk, "resolve_rnx2rtkp", lambda override=None: fake_exe)


def test_ppk_result_does_not_retain_full_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("# empty\n")
    out = tmp_path / "out.pos"

    # ~9 MB of \r progress noise on stderr, ~4 MB on stdout.
    huge_err = "\r".join(
        f"processing : epoch {i} Q=1" for i in range(300_000)
    )
    huge_out = "x" * 4_000_000

    logged: list[str] = []

    class FakeProc:
        stdout = huge_out
        stderr = huge_err
        returncode = 0

    def fake_run(cmd, **kw):
        out.write_text("% header\n2024/01/01 00:00:00 32.0 34.0 100 1 12\n")
        return FakeProc()

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    res = ppk.run(
        rover_obs=rover, base_obs=base, nav_files=[nav],
        config_file=cfg, output_pos=out, log=logged.append,
    )

    # Retained streams are bounded (tail only), not the multi-MB originals.
    assert len(res.stdout) <= ppk._RESULT_STREAM_MAX_CHARS
    assert len(res.stderr) <= ppk._RESULT_STREAM_MAX_CHARS
    # The retained stderr tail ends with the LAST progress entries.
    assert "epoch 299999" in res.stderr

    # The log did not get flooded: bounded number of messages, and the
    # summary line is present.
    assert len(logged) < 200, f"log flooded with {len(logged)} messages"
    assert any("rnx2rtkp finished rc=0" in m for m in logged)
    # The last progress line made it into the logged tail.
    assert any("epoch 299999" in m for m in logged)


# ---------------------------------------------------------------------------
# GUI drain caps (no Tk root needed -- constants + pure helper only)
# ---------------------------------------------------------------------------

def test_gui_drain_constants_exist() -> None:
    gui = pytest.importorskip("data_pipeline.gui")
    assert gui.App.MAX_PER_TICK == 200
    assert gui.App.MAX_LOG_LINES >= 1_000
    # POLL_MS still drives rescheduling.
    assert gui.App.POLL_MS > 0


def test_gui_log_tag_helper_is_tk_free() -> None:
    gui = pytest.importorskip("data_pipeline.gui")
    tag = gui.App._log_tag_for  # staticmethod, callable without an App/Tk root
    assert tag("=== PPK done ===") == "t_done"
    assert tag("=== Stage: PPK ===") == "t_stage"
    assert tag("!!! failed") == "t_error"
    assert tag("Traceback (most recent call last):") == "t_error"
    assert tag("[ppk] cmd = ...") == "t_step"
    assert tag("warning: low satellites") == "t_warn"
    assert tag("plain line") == "t_normal"
