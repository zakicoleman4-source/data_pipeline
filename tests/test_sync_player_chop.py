"""Cut-clip ("segment") t0 handling in the viewers (sync player + compare).

Regression: ``build_sync_player`` / ``build_comparison_viewer`` resolved the
media sample-0 boottime via ``_resolve_boottime_t0_ns(capture_meta, ...)``,
which takes capture_meta's ``video_t0_boottime_ns`` FIRST. For a segment clip the
callers pass the PARENT session's capture_meta (original full-session
sample-0 boot) while the segment container file's PTS are rebased to 0 — every sample mapped
seconds/minutes early and the Signal marker ran ahead of the media. Same bug as
the one fixed in ``stages.georef._load_frames`` via ``chop_video_anchor``;
this file locks the mirrored fix in ``stages.viewers``.

Covers:
- segment anchor min bootNs OVERRIDES the capture_meta t0;
- ``frame_to_utc(0)`` lands in the segment window, not ~session start;
- non-segment path (``chop_video_anchor=None``) byte-for-byte unchanged;
- unreadable/empty segment anchor: WARN + fall back (no crash, no silent lie);
- both public builders expose the ``chop_video_anchor`` parameter.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

import pytest

from data_pipeline.stages.viewers import (
    _make_frame_to_utc,
    _resolve_boottime_t0_ns,
    build_comparison_viewer,
    build_sync_player,
)
from data_pipeline.time_sync import fit_time_anchor


# Reference UTC second the synthetic anchor hangs off (2026-06-01T00:00:00Z).
BASE_UTC = 1780272000.0
# Full-session sample-0 boottime: 100 s since boot, in ns.
SESSION_T0_NS = 100_000_000_000
# The segment starts 300 s into the session.
CHOP_OFFSET_S = 300.0
CHOP_T0_NS = SESSION_T0_NS + int(CHOP_OFFSET_S * 1e9)
FRAME_DT_S = 1.0 / 30.0


def _iso(utc_s: float) -> str:
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def _write_full_recording(path: Path, span_s: int = 400) -> None:
    """Full-session recording_*.txt: absolute boottime_ns,ISO-UTC,interval."""
    lines = []
    for i in range(span_s + 1):
        boot_ns = SESSION_T0_NS + i * 1_000_000_000
        lines.append(f"{boot_ns},{_iso(BASE_UTC + i)},1000000000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chop_video_anchor(path: Path, n: int = 12) -> None:
    """Segment video_anchor.txt: min bootNs == CHOP_T0_NS (well inside session)."""
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(n):
        boot = CHOP_T0_NS + int(i * FRAME_DT_S * 1e9)
        lines.append(f"{9000 + i},{boot},{boot},REALTIME")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_session_video_anchor(path: Path, n: int = 4) -> None:
    """Parent-session video_anchor.txt: min bootNs == SESSION_T0_NS."""
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(n):
        boot = SESSION_T0_NS + int(i * FRAME_DT_S * 1e9)
        lines.append(f"{i},{boot},{boot},REALTIME")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_original_capture_meta(path: Path) -> None:
    """capture_meta.json carrying the ORIGINAL session-start media t0."""
    path.write_text(json.dumps({
        "anchor_format": 2,
        "video": {
            "mp4": "recording_x.mp4",
            "video_t0_boottime_ns": SESSION_T0_NS,
            "timestamp_source": "boottime",
        },
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# t0 resolution: segment anchor overrides capture_meta
# ---------------------------------------------------------------------------

def test_chop_anchor_overrides_capture_meta_t0(tmp_path: Path) -> None:
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)
    chop_anchor = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(chop_anchor)

    logs: list[str] = []
    t0 = _resolve_boottime_t0_ns(cm, None, logs.append,
                                 chop_video_anchor=chop_anchor)
    assert t0 == pytest.approx(float(CHOP_T0_NS))
    assert t0 != pytest.approx(float(SESSION_T0_NS))
    assert any("chop clip" in ln and "overridden" in ln for ln in logs)


def test_chop_anchor_overrides_parent_video_anchor_too(tmp_path: Path) -> None:
    """Even with BOTH parent capture_meta and parent video_anchor present,
    the segment anchor wins."""
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)
    parent_anchor = tmp_path / "recording_x.video_anchor.txt"
    _write_session_video_anchor(parent_anchor)
    chop_anchor = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(chop_anchor)

    t0 = _resolve_boottime_t0_ns(cm, parent_anchor, lambda *_: None,
                                 chop_video_anchor=chop_anchor)
    assert t0 == pytest.approx(float(CHOP_T0_NS))


def test_chop_frame0_maps_into_chop_window(tmp_path: Path) -> None:
    """frame_to_utc(0) lands at session start + 300 s, NOT ~session start."""
    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)
    chop_anchor = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(chop_anchor)

    anchor = fit_time_anchor(rec)
    t0 = _resolve_boottime_t0_ns(cm, None, lambda *_: None,
                                 chop_video_anchor=chop_anchor)
    f = _make_frame_to_utc(anchor, t0)
    assert f(0.0) == pytest.approx(BASE_UTC + CHOP_OFFSET_S, abs=5e-3)
    assert f(1.0) == pytest.approx(BASE_UTC + CHOP_OFFSET_S + 1.0, abs=5e-3)
    # NOT mapped to session start (the bug: parent t0 + rebased pts).
    assert abs(f(0.0) - BASE_UTC) > CHOP_OFFSET_S - 1.0


# ---------------------------------------------------------------------------
# Non-segment path: byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_non_chop_capture_meta_still_wins(tmp_path: Path) -> None:
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)

    logs: list[str] = []
    t0 = _resolve_boottime_t0_ns(cm, None, logs.append)
    assert t0 == pytest.approx(float(SESSION_T0_NS))
    assert not any("chop" in ln for ln in logs)

    # Explicit None behaves identically to the omitted default.
    t0b = _resolve_boottime_t0_ns(cm, None, lambda *_: None,
                                  chop_video_anchor=None)
    assert t0b == t0


def test_non_chop_video_anchor_fallback_unchanged(tmp_path: Path) -> None:
    parent_anchor = tmp_path / "recording_x.video_anchor.txt"
    _write_session_video_anchor(parent_anchor)
    t0 = _resolve_boottime_t0_ns(None, parent_anchor, lambda *_: None,
                                 chop_video_anchor=None)
    assert t0 == pytest.approx(float(SESSION_T0_NS))


def test_non_chop_legacy_returns_none() -> None:
    assert _resolve_boottime_t0_ns(None, None, lambda *_: None,
                                   chop_video_anchor=None) is None


# ---------------------------------------------------------------------------
# Unreadable/empty segment anchor: WARN + fall back, never crash
# ---------------------------------------------------------------------------

def test_empty_chop_anchor_warns_and_falls_back(tmp_path: Path) -> None:
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)
    empty_anchor = tmp_path / "chop_x.video_anchor.txt"
    empty_anchor.write_text(
        "# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource\n",
        encoding="utf-8",
    )

    logs: list[str] = []
    t0 = _resolve_boottime_t0_ns(cm, None, logs.append,
                                 chop_video_anchor=empty_anchor)
    # Falls back to the existing resolution (tolerant: viewer must not die),
    # but says so loudly.
    assert t0 == pytest.approx(float(SESSION_T0_NS))
    assert any("WARN" in ln and "chop" in ln for ln in logs)


def test_missing_chop_anchor_warns_and_falls_back(tmp_path: Path) -> None:
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)
    missing = tmp_path / "does_not_exist.video_anchor.txt"

    logs: list[str] = []
    t0 = _resolve_boottime_t0_ns(cm, None, logs.append,
                                 chop_video_anchor=missing)
    assert t0 == pytest.approx(float(SESSION_T0_NS))
    assert any("WARN" in ln for ln in logs)


# ---------------------------------------------------------------------------
# Both public builders expose the parameter (so the CLI/GUI wiring can't rot)
# ---------------------------------------------------------------------------

def test_builders_accept_chop_video_anchor_kwarg() -> None:
    for fn in (build_sync_player, build_comparison_viewer):
        params = inspect.signature(fn).parameters
        assert "chop_video_anchor" in params, fn.__name__
        assert params["chop_video_anchor"].default is None, fn.__name__
