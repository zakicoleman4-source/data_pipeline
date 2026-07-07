"""Zero-bias cut-clip ("segment") handling.

A segment is a cut container file produced by the source app in a ``chop_*`` subdir of
the parent session. Its PTS are rebased to zero, so the sample-0 boottime t0
must come from the segment's OWN ``video_anchor.txt`` (min bootNs) — never from
capture_meta's original-session ``video_t0_boottime_ns`` (which would map
every sample minutes early), and never with any mono_to_boot offset added.

Covers:
- ``georef._load_frames(chop_video_anchor=...)`` maps samples into the segment
  window of the FULL session anchor;
- the segment t0 wins over a capture_meta carrying the original session t0;
- no ``mono_to_boot_offset_ns`` is ever folded in;
- ``RawInputs.from_folder`` segment detection (parent-with-subdir and the segment
  dir itself) plus the unchanged normal-session path.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from data_pipeline.pipeline import RawInputs
from data_pipeline.stages.georef import _load_frames


# Reference UTC second the synthetic anchor hangs off (2026-06-01T00:00:00Z).
BASE_UTC = 1780272000.0
# Full-session sample-0 boottime: 100 s since boot, in ns.
SESSION_T0_NS = 100_000_000_000
# The segment starts 300 s into the session.
CHOP_OFFSET_S = 300.0
CHOP_T0_NS = SESSION_T0_NS + int(CHOP_OFFSET_S * 1e9)
# A deliberately large mono->boot offset: if it ever leaks into the mapping
# the samples jump by 5 s and the assertions below fail.
MONO_TO_BOOT_OFFSET_NS = 5_000_000_000
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


def _write_chop_frame_times(path: Path, n: int = 5, start_s: float = 0.0) -> None:
    """Rebased-to-zero sample times, as extracted from the segment container file.

    ``start_s`` shifts every t_video_s (0.0 = the normal no-copyts output;
    non-zero simulates unexpectedly preserved PTS / an edit list).
    """
    rows = ["Image,t_video_s"]
    for i in range(n):
        rows.append(f"frame_{i:06d}.png,{start_s + i * FRAME_DT_S:.6f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_original_capture_meta(path: Path) -> None:
    """capture_meta.json carrying the ORIGINAL session-start media t0."""
    path.write_text(json.dumps({
        "anchor_format": 2,
        "video": {
            "mp4": "recording_x.mp4",
            "video_t0_boottime_ns": SESSION_T0_NS,
            "timestamp_source": "boottime",
        },
        "clock": {"mono_to_boot_offset_ns": MONO_TO_BOOT_OFFSET_NS},
    }), encoding="utf-8")


def _write_chop_meta(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "chop_meta/1",
        "source_dir": "/storage/emulated/0/whatever/20241104_081922_038",
        "source_mp4": "recording_x.mp4",
        "source_capture_meta": "capture_meta.json",
        "source_audio_wav": "audio_x.wav",
        "source_audio_anchor": "audio_anchor_x.txt",
        "source_gnss": "recording_x.txt",
        "video_t0_boottime_ns": SESSION_T0_NS,
        "in_pts_us_original": int(CHOP_OFFSET_S * 1e6),
        "out_pts_us_original": int((CHOP_OFFSET_S + 60) * 1e6),
        "start_boottime_ns": CHOP_T0_NS - 3_000,  # us-rounded proxy, ~3us early
        "end_boottime_ns": CHOP_T0_NS + 60_000_000_000,
        "chopped_pts_rebased_to_zero": True,
        "frame_count": 5,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# _load_frames: segment t0 mapping
# ---------------------------------------------------------------------------

def test_chop_anchor_maps_frames_into_chop_window(tmp_path: Path) -> None:
    """With the segment anchor supplied, sample 0 lands at session start + 300 s."""
    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    anchor_txt = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(anchor_txt)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc)

    frames, _ = _load_frames(
        ftc, rec, lambda *_a: None, chop_video_anchor=anchor_txt,
    )
    assert len(frames) == 5
    assert frames[0].utc_s == pytest.approx(BASE_UTC + CHOP_OFFSET_S, abs=5e-3)
    assert frames[1].utc_s == pytest.approx(
        BASE_UTC + CHOP_OFFSET_S + FRAME_DT_S, abs=5e-3
    )
    # NOT mapped to session start (the bug: original t0 + rebased pts).
    assert abs(frames[0].utc_s - BASE_UTC) > CHOP_OFFSET_S - 1.0


def test_chop_t0_wins_over_capture_meta_original_t0(tmp_path: Path) -> None:
    """capture_meta's original video_t0 must be overridden by the segment t0."""
    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    anchor_txt = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(anchor_txt)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc)
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)

    logs: list[str] = []
    frames, _ = _load_frames(
        ftc, rec, logs.append,
        capture_meta=cm, chop_video_anchor=anchor_txt,
    )
    # Segment window, not 300 s early at session start.
    assert frames[0].utc_s == pytest.approx(BASE_UTC + CHOP_OFFSET_S, abs=5e-3)
    assert frames[0].utc_s != pytest.approx(BASE_UTC, abs=1.0)
    assert any("chop clip" in ln and "overridden" in ln for ln in logs)


def test_no_mono_to_boot_offset_added(tmp_path: Path) -> None:
    """bootNs are raw CLOCK_BOOTTIME: a huge mono_to_boot_offset_ns in
    capture_meta must not shift the samples (they would jump by 5 s)."""
    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    anchor_txt = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(anchor_txt)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc)
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)  # carries mono_to_boot_offset_ns = 5e9

    frames, _ = _load_frames(
        ftc, rec, lambda *_a: None,
        capture_meta=cm, chop_video_anchor=anchor_txt,
    )
    expected = BASE_UTC + CHOP_OFFSET_S
    assert frames[0].utc_s == pytest.approx(expected, abs=5e-3)
    # Explicitly: not shifted by the mono->boot offset in either direction.
    off_s = MONO_TO_BOOT_OFFSET_NS / 1e9
    assert abs(frames[0].utc_s - (expected + off_s)) > 1.0
    assert abs(frames[0].utc_s - (expected - off_s)) > 1.0
    # Inter-sample spacing stays the rebased pts spacing.
    assert frames[2].utc_s - frames[0].utc_s == pytest.approx(
        2 * FRAME_DT_S, abs=2e-3
    )


# ---------------------------------------------------------------------------
# _load_frames: pts guard is sanity-only, unreadable segment anchor raises
# ---------------------------------------------------------------------------

def test_chop_no_residual_added_even_with_nonzero_ffprobe_start_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction runs without -copyts, so t_video_s already starts at 0.

    Even when the probe tool reports a non-zero container start_time, NOTHING may be
    added to the sample times — adding it shifts every segment sample late. The probe tool guard is a sanity check only.
    """
    import data_pipeline.stages.georef as georef_mod

    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    anchor_txt = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(anchor_txt)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc)  # first t_video_s == 0.0
    mp4 = tmp_path / "chop_x.mp4"
    mp4.write_bytes(b"\x00")

    # the probe tool "reports" a 2.5 s container start_time.
    monkeypatch.setattr(
        georef_mod, "_probe_video_start_time_s", lambda *_a, **_k: 2.5,
    )

    logs: list[str] = []
    frames, _ = _load_frames(
        ftc, rec, logs.append,
        chop_video_anchor=anchor_txt, video_path=mp4,
    )
    # Correct segment-window mapping, with NO residual folded in.
    assert frames[0].utc_s == pytest.approx(BASE_UTC + CHOP_OFFSET_S, abs=5e-3)
    assert frames[0].t_video_s == pytest.approx(0.0, abs=1e-9)
    # Explicitly not shifted late by the probed start_time.
    assert abs(frames[0].utc_s - (BASE_UTC + CHOP_OFFSET_S + 2.5)) > 1.0
    assert not any("applying residual" in ln for ln in logs)
    # First t_video_s ~0 -> the sanity check stays silent.
    assert not any("expected ~0" in ln for ln in logs)


def test_chop_nonzero_first_pts_warns_and_is_not_corrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First t_video_s far from 0 (PTS preserved / edit list): the pipeline
    warns loudly but must NOT silently apply any correction."""
    import data_pipeline.stages.georef as georef_mod

    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    anchor_txt = tmp_path / "chop_x.video_anchor.txt"
    _write_chop_video_anchor(anchor_txt)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc, start_s=3.0)  # unexpectedly non-zero
    mp4 = tmp_path / "chop_x.mp4"
    mp4.write_bytes(b"\x00")

    monkeypatch.setattr(
        georef_mod, "_probe_video_start_time_s", lambda *_a, **_k: 3.0,
    )

    logs: list[str] = []
    frames, _ = _load_frames(
        ftc, rec, logs.append,
        chop_video_anchor=anchor_txt, video_path=mp4,
    )
    warns = [ln for ln in logs if "WARN" in ln and "expected ~0" in ln]
    assert warns, f"expected a first-t_video_s sanity WARN, got: {logs}"
    # Uncorrected pass-through: samples map 3.0 s into the segment window,
    # neither pulled back to the window start nor shifted by another 3 s.
    assert frames[0].utc_s == pytest.approx(
        BASE_UTC + CHOP_OFFSET_S + 3.0, abs=5e-3
    )
    assert frames[0].t_video_s == pytest.approx(3.0, abs=1e-9)


def test_unreadable_chop_anchor_raises(tmp_path: Path) -> None:
    """A known segment with an unreadable/empty anchor must RAISE — never fall
    back to the full-session t0 (which maps samples ~minutes early)."""
    rec = tmp_path / "recording_x.txt"
    _write_full_recording(rec)
    ftc = tmp_path / "extracted_frame_times.csv"
    _write_chop_frame_times(ftc)
    cm = tmp_path / "capture_meta.json"
    _write_original_capture_meta(cm)  # the stale parent t0 it must NOT use

    # Header-only (empty) segment anchor.
    empty_anchor = tmp_path / "chop_x.video_anchor.txt"
    empty_anchor.write_text(
        "# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as ei:
        _load_frames(
            ftc, rec, lambda *_a: None,
            capture_meta=cm, chop_video_anchor=empty_anchor,
        )
    assert "chop_x.video_anchor.txt" in str(ei.value)

    # Missing segment anchor file.
    missing_anchor = tmp_path / "chop_missing.video_anchor.txt"
    with pytest.raises(ValueError) as ei:
        _load_frames(
            ftc, rec, lambda *_a: None,
            capture_meta=cm, chop_video_anchor=missing_anchor,
        )
    assert "chop_missing.video_anchor.txt" in str(ei.value)


# ---------------------------------------------------------------------------
# RawInputs.from_folder: segment detection + resolution
# ---------------------------------------------------------------------------

def _make_parent_session(parent: Path) -> None:
    (parent / "measurements_x.txt").write_text("m\n", encoding="utf-8")
    _write_full_recording(parent / "recording_x.txt")
    (parent / "sensors_x.txt").write_text("s\n", encoding="utf-8")
    _write_original_capture_meta(parent / "capture_meta.json")
    (parent / "audio_x.wav").write_bytes(b"RIFF")
    (parent / "audio_anchor_x.txt").write_text("a\n", encoding="utf-8")
    (parent / "recording_x.mp4").write_bytes(b"\x00")
    (parent / "recording_x.video_anchor.txt").write_text(
        "# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource\n"
        f"0,{SESSION_T0_NS},{SESSION_T0_NS},REALTIME\n",
        encoding="utf-8",
    )


def _make_chop_dir(parent: Path, name: str = "chop_123") -> Path:
    chop = parent / name
    chop.mkdir()
    _write_chop_meta(chop / f"{name}.chop_meta.json")
    (chop / f"{name}.mp4").write_bytes(b"\x00")
    _write_chop_video_anchor(chop / f"{name}.video_anchor.txt")
    return chop


def test_from_folder_parent_with_chop_subdir(tmp_path: Path) -> None:
    _make_parent_session(tmp_path)
    chop = _make_chop_dir(tmp_path)

    ri = RawInputs.from_folder(tmp_path)
    assert ri.is_chop is True
    assert ri.chop_meta_json == chop / "chop_123.chop_meta.json"
    # container file + media anchor come from the segment...
    assert ri.recording_mp4 == chop / "chop_123.mp4"
    assert ri.video_anchor_txt == chop / "chop_123.video_anchor.txt"
    assert ri.chop_video_anchor == chop / "chop_123.video_anchor.txt"
    # ...stream/Signal/sensors/capture_meta from the PARENT session.
    assert ri.measurements_txt == tmp_path / "measurements_x.txt"
    assert ri.recording_txt == tmp_path / "recording_x.txt"
    assert ri.sensors_txt == tmp_path / "sensors_x.txt"
    assert ri.capture_meta_json == tmp_path / "capture_meta.json"
    assert ri.audio_wav == tmp_path / "audio_x.wav"
    assert ri.audio_anchor_txt == tmp_path / "audio_anchor_x.txt"
    assert ri.capture_format == "new"


def test_from_folder_on_chop_dir_directly(tmp_path: Path) -> None:
    _make_parent_session(tmp_path)
    chop = _make_chop_dir(tmp_path)

    ri = RawInputs.from_folder(chop)
    assert ri.is_chop is True
    assert ri.recording_mp4 == chop / "chop_123.mp4"
    assert ri.chop_video_anchor == chop / "chop_123.video_anchor.txt"
    assert ri.recording_txt == tmp_path / "recording_x.txt"
    assert ri.audio_wav == tmp_path / "audio_x.wav"


def test_from_folder_normal_session_unchanged(tmp_path: Path) -> None:
    """No segment anywhere: the existing resolution path is untouched."""
    _make_parent_session(tmp_path)

    ri = RawInputs.from_folder(tmp_path)
    assert ri.is_chop is False
    assert ri.chop_meta_json is None
    assert ri.chop_video_anchor is None
    assert ri.recording_mp4 == tmp_path / "recording_x.mp4"
    assert ri.video_anchor_txt == tmp_path / "recording_x.video_anchor.txt"
    assert ri.recording_txt == tmp_path / "recording_x.txt"
    assert ri.capture_format == "new"


def test_from_folder_multiple_chop_subdirs_falls_back(tmp_path: Path) -> None:
    """Ambiguous (2 segment subdirs): resolve the parent as a normal session."""
    _make_parent_session(tmp_path)
    _make_chop_dir(tmp_path, "chop_123")
    _make_chop_dir(tmp_path, "chop_456")

    ri = RawInputs.from_folder(tmp_path)
    assert ri.is_chop is False
    assert ri.recording_mp4 == tmp_path / "recording_x.mp4"


def test_from_folder_chop_files_dropped_in_session_dir_no_grandparent(
    tmp_path: Path,
) -> None:
    """Segment files dropped DIRECTLY into a full session dir (not a chop_*
    subdir) must NOT make that dir resolve as a segment dir: doing so sets
    parent = folder.parent and silently globs Signal/stream from the
    GRANDPARENT — another session's data. With measurements_*.txt present in
    the folder, normal full-session resolution must win."""
    # Grandparent holds a DECOY session: if grandparent resolution ever
    # triggers, these files would be picked up silently.
    (tmp_path / "measurements_decoy.txt").write_text("m\n", encoding="utf-8")
    _write_full_recording(tmp_path / "recording_decoy.txt")
    (tmp_path / "sensors_decoy.txt").write_text("s\n", encoding="utf-8")
    (tmp_path / "audio_decoy.wav").write_bytes(b"RIFF")

    session = tmp_path / "session"
    session.mkdir()
    _make_parent_session(session)
    # Segment files dropped straight into the session dir (no chop_* subdir).
    _write_chop_meta(session / "chop_999.chop_meta.json")
    (session / "chop_999.mp4").write_bytes(b"\x00")
    _write_chop_video_anchor(session / "chop_999.video_anchor.txt")

    ri = RawInputs.from_folder(session)
    assert ri.is_chop is False
    assert ri.chop_video_anchor is None
    # Everything resolves from the SESSION dir, never the grandparent decoy.
    assert ri.measurements_txt == session / "measurements_x.txt"
    assert ri.recording_txt == session / "recording_x.txt"
    assert ri.sensors_txt == session / "sensors_x.txt"
    assert ri.audio_wav == session / "audio_x.wav"
    assert ri.recording_mp4 == session / "recording_x.mp4"
    assert ri.video_anchor_txt == session / "recording_x.video_anchor.txt"


def test_from_folder_on_chop_dir_directly_still_works(tmp_path: Path) -> None:
    """The grandparent guard must not break the legit direct-segment-dir case:
    a real chop_* dir has no measurements_*.txt, so it still resolves as a
    segment with files from its parent session."""
    _make_parent_session(tmp_path)
    chop = _make_chop_dir(tmp_path, "chop_777")

    ri = RawInputs.from_folder(chop)
    assert ri.is_chop is True
    assert ri.recording_mp4 == chop / "chop_777.mp4"
    assert ri.measurements_txt == tmp_path / "measurements_x.txt"
