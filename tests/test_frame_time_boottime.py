"""Boottime/segment sample->UTC mapping regressions for ins + motion model.

Covers the shared helper ``data_pipeline.frame_time`` and its wiring into
the INS per-sample CSV exporter (``stages.ins.build_ins_csv``) and all three
``run_vio*`` variants.

The bug being pinned: those call sites used the raw
``anchor.video_pts_to_utc_s(pts)`` mapping, which for boottime-format
sessions (anchor x-domain = absolute CLOCK_BOOTTIME ns) puts every sample
~t0 seconds early (505.8 s on the day15 segment), so
``_sample_smoothed_at_frame_utc`` misses ``max_extrap_s`` and samples get no
coordinates. For a cut ("segment") clip the PTS are additionally rebased
to 0, so the segment's own video_anchor min bootNs must WIN over the parent
capture_meta t0.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_pipeline.frame_time import (
    first_boottime_ns_from_video_anchor,
    make_frame_to_utc,
    resolve_video_t0_boottime_ns,
)
from data_pipeline.parsers import PosRow
from data_pipeline.time_sync import fit_time_anchor
from data_pipeline.stages import ins as ins_mod


# Reference UTC second the synthetic anchors hang off (2026-06-01T00:00:00Z).
BASE_UTC = 1780272000.0
# media sample-0 boottime: 100 s since boot, in ns.
VIDEO_T0_NS = 100_000_000_000
# The segment starts 5 s into the original session.
CHOP_OFFSET_NS = 5_000_000_000


def _iso(utc_s: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(utc_s, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def _write_boottime_recording(path: Path, n: int = 60, hz: float = 5.0) -> None:
    """recording_*.txt: boottime_ns, utc_iso, interval_ns. 1:1 boottime->UTC."""
    dt_ns = int(1e9 / hz)
    lines = []
    for i in range(n):
        boot_ns = VIDEO_T0_NS + i * dt_ns
        utc_s = BASE_UTC + (boot_ns - VIDEO_T0_NS) / 1e9
        lines.append(f"{boot_ns},{_iso(utc_s)},{dt_ns}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_legacy_recording(path: Path, n: int = 60, hz: float = 5.0) -> None:
    """Legacy session map: video_ns starts at 0 and maps 1:1 onto UTC."""
    dt_ns = int(1e9 / hz)
    lines = []
    for i in range(n):
        vns = i * dt_ns
        lines.append(f"{vns},{_iso(BASE_UTC + vns / 1e9)},{dt_ns}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_capture_meta(path: Path, t0_ns: int = VIDEO_T0_NS) -> None:
    path.write_text(json.dumps({
        "anchor_format": "boottime",
        "video": {
            "mp4": "video_123.mp4",
            "video_t0_boottime_ns": t0_ns,
            "timestamp_source": "boottime",
        },
        "clock": {"mono_to_boot_offset_ns": 0},
    }), encoding="utf-8")


def _write_video_anchor(path: Path, t0_ns: int, n: int = 10,
                        fps: float = 30.0) -> None:
    """Per-sample video_anchor.txt: frameNumber,sensorTsNs,bootNs,source."""
    dt_ns = int(1e9 / fps)
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(n):
        boot = t0_ns + i * dt_ns
        lines.append(f"{i},{boot},{boot},boottime")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────
# Shared helper: t0 resolution precedence + the mapping itself
# ─────────────────────────────────────────────────────────────────────────

class TestResolveT0:
    def test_capture_meta_t0(self, tmp_path: Path) -> None:
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm)
        assert resolve_video_t0_boottime_ns(capture_meta=cm) == float(VIDEO_T0_NS)

    def test_chop_anchor_wins_over_capture_meta(self, tmp_path: Path) -> None:
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm, t0_ns=VIDEO_T0_NS)
        chop = tmp_path / "video_anchor.txt"
        _write_video_anchor(chop, VIDEO_T0_NS + CHOP_OFFSET_NS)
        t0 = resolve_video_t0_boottime_ns(capture_meta=cm, chop_video_anchor=chop)
        assert t0 == float(VIDEO_T0_NS + CHOP_OFFSET_NS)

    def test_video_anchor_fallback(self, tmp_path: Path) -> None:
        va = tmp_path / "video_anchor.txt"
        _write_video_anchor(va, VIDEO_T0_NS)
        assert resolve_video_t0_boottime_ns(video_anchor=va) == float(VIDEO_T0_NS)
        assert first_boottime_ns_from_video_anchor(va) == float(VIDEO_T0_NS)

    def test_legacy_returns_none(self) -> None:
        assert resolve_video_t0_boottime_ns() is None


class TestMakeFrameToUtc:
    def test_boottime_lifts_pts_into_bootns(self, tmp_path: Path) -> None:
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        anchor = fit_time_anchor(rec)
        f = make_frame_to_utc(anchor, float(VIDEO_T0_NS))
        for pts in (0.0, 1.0, 2.5):
            assert f(pts) == pytest.approx(
                anchor.boottime_to_utc_s(VIDEO_T0_NS + pts * 1e9), abs=1e-9
            )
            assert f(pts) == pytest.approx(BASE_UTC + pts, abs=2e-3)
            # NOT the ~t0-early legacy value.
            legacy = anchor.video_pts_to_utc_s(pts)
            assert abs(f(pts) - legacy) > 90.0

    def test_legacy_mapping_unchanged(self, tmp_path: Path) -> None:
        rec = tmp_path / "recording_x.txt"
        _write_legacy_recording(rec)
        anchor = fit_time_anchor(rec)
        f = make_frame_to_utc(anchor, None)
        for pts in (0.0, 1.0, 2.5):
            assert f(pts) == anchor.video_pts_to_utc_s(pts)

    def test_manual_shift(self, tmp_path: Path) -> None:
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        anchor = fit_time_anchor(rec)
        f = make_frame_to_utc(anchor, float(VIDEO_T0_NS), manual_shift_s=0.25)
        assert f(1.0) == pytest.approx(BASE_UTC + 1.25, abs=2e-3)


# ─────────────────────────────────────────────────────────────────────────
# INS per-sample CSV exporter wiring
# ─────────────────────────────────────────────────────────────────────────

def _fake_smoothed(span_s: float = 10.0, hz: float = 5.0):
    """Smoothed path whose latitude encodes time:
    lat = 10 + (utc - BASE_UTC) * 1e-3."""
    ts = [BASE_UTC + i / hz for i in range(int(span_s * hz) + 1)]
    fused = [
        PosRow(utc_s=t, lat_deg=10.0 + (t - BASE_UTC) * 1e-3,
               lon_deg=20.0, h_m=0.0, quality=1)
        for t in ts
    ]
    q = [np.array([1.0, 0.0, 0.0, 0.0])] * len(ts)
    return SimpleNamespace(fused=fused, q_att=q)


@pytest.fixture()
def ins_stubs(monkeypatch):
    """Bypass the heavy Motion sensor/EKF/RTS chain; keep the per-sample exporter real."""
    sm = _fake_smoothed()
    pos = [PosRow(utc_s=BASE_UTC + i, lat_deg=10.0, lon_deg=20.0, h_m=0.0,
                  quality=1) for i in range(11)]
    monkeypatch.setattr(ins_mod, "parse_imu", lambda p: [object()])
    monkeypatch.setattr(ins_mod, "parse_rtkpos", lambda p: pos)
    monkeypatch.setattr(
        ins_mod, "run_ekf",
        lambda imu, pos, **kw: SimpleNamespace(
            tape_t=[r.utc_s for r in sm.fused],
            n_pos_updates=0, n_vel_updates=0, n_pos_rejected=0,
            n_vel_rejected=0, n_zupt=0, n_nhc=0,
        ),
    )
    monkeypatch.setattr(ins_mod, "rts_smooth", lambda fwd, ref: sm)
    return sm


def _write_frame_times(path: Path) -> None:
    path.write_text(
        "Image,t_video_s\nframe_0.png,0.0\nframe_1.png,1.0\nframe_2.png,2.0\n",
        encoding="utf-8",
    )


def _run_ins(tmp_path: Path, rec: Path, out_name: str, **kw):
    out = tmp_path / out_name
    res = ins_mod.build_ins_csv(
        sensors_txt=tmp_path / "sensors_x.txt",
        pos_file=tmp_path / "x.pos",
        recording_map=rec,
        frame_times_csv=tmp_path / "extracted_frame_times.csv",
        out_csv=out,
        **kw,
    )
    return res, out


def _csv_lats(path: Path) -> list[float]:
    import csv as _csv
    with path.open("r", newline="", encoding="utf-8") as f:
        return [float(r["Latitude"]) for r in _csv.DictReader(f)]


class TestInsExporter:
    def test_boottime_session_frames_get_coords(self, tmp_path, ins_stubs):
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm)
        _write_frame_times(tmp_path / "extracted_frame_times.csv")

        res, out = _run_ins(tmp_path, rec, "georef_ins.csv", capture_meta=cm)
        assert res.n_frames == 3
        assert res.n_with_position == 3
        lats = _csv_lats(out)
        # sample utc = boottime_to_utc(t0 + pts*1e9) = BASE_UTC + pts,
        # and lat encodes (utc - BASE_UTC) * 1e-3.
        assert lats == pytest.approx(
            [10.0, 10.0 + 1e-3, 10.0 + 2e-3], abs=1e-5)

    def test_boottime_session_without_meta_reproduces_bug(self, tmp_path,
                                                          ins_stubs):
        """Without the t0, samples map ~t0 early and miss max_extrap_s —
        this is the pre-fix behaviour the new params exist to fix."""
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        _write_frame_times(tmp_path / "extracted_frame_times.csv")

        res, out = _run_ins(tmp_path, rec, "georef_ins.csv")
        assert res.n_frames == 3
        assert res.n_with_position == 0
        assert _csv_lats(out) == []

    def test_chop_anchor_wins_over_capture_meta(self, tmp_path, ins_stubs):
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm, t0_ns=VIDEO_T0_NS)  # parent full-session t0
        chop = tmp_path / "chop_video_anchor.txt"
        _write_video_anchor(chop, VIDEO_T0_NS + CHOP_OFFSET_NS)
        _write_frame_times(tmp_path / "extracted_frame_times.csv")

        res, out = _run_ins(
            tmp_path, rec, "georef_ins.csv",
            capture_meta=cm, chop_video_anchor=chop,
        )
        assert res.n_with_position == 3
        # sample utc = BASE_UTC + 5 + pts (segment t0), NOT BASE_UTC + pts.
        chop_s = CHOP_OFFSET_NS / 1e9
        assert _csv_lats(out) == pytest.approx(
            [10.0 + (chop_s + p) * 1e-3 for p in (0.0, 1.0, 2.0)], abs=1e-5)

    def test_legacy_session_byte_for_byte_unchanged(self, tmp_path, ins_stubs):
        rec = tmp_path / "recording_legacy.txt"
        _write_legacy_recording(rec)
        _write_frame_times(tmp_path / "extracted_frame_times.csv")

        res_a, out_a = _run_ins(tmp_path, rec, "a.csv")
        res_b, out_b = _run_ins(
            tmp_path, rec, "b.csv",
            capture_meta=None, video_anchor=None, chop_video_anchor=None,
        )
        assert res_a.n_with_position == 3
        assert out_a.read_bytes() == out_b.read_bytes()
        assert _csv_lats(out_a) == pytest.approx(
            [10.0, 10.0 + 1e-3, 10.0 + 2e-3], abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# Motion model wiring: run_vio / run_vio_multiframe(v1) / run_vio_multiframe_v2
# ─────────────────────────────────────────────────────────────────────────

N_FRAMES = 6
FPS = 5.0


def _fake_cv2() -> types.ModuleType:
    """Minimal cv2 stand-in: identity Sparse-feature, always-inlier relative-pose.

    Enough for the run_vio* loops to emit VioSamples whose ``utc_s`` we can
    check, without The feature library or a real media file.
    """
    m = types.ModuleType("cv2")
    m.CAP_PROP_FPS = 5
    m.CAP_PROP_FRAME_WIDTH = 3
    m.CAP_PROP_FRAME_HEIGHT = 4
    m.CAP_PROP_FRAME_COUNT = 7
    m.COLOR_BGR2GRAY = 6
    m.TERM_CRITERIA_EPS = 1
    m.TERM_CRITERIA_COUNT = 2
    m.RANSAC = 8
    m.OPTFLOW_USE_INITIAL_FLOW = 4
    m.error = type("error", (Exception,), {})

    class FakeCap:
        def __init__(self, _path):
            self._i = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                m.CAP_PROP_FPS: FPS,
                m.CAP_PROP_FRAME_WIDTH: 32.0,
                m.CAP_PROP_FRAME_HEIGHT: 24.0,
                m.CAP_PROP_FRAME_COUNT: float(N_FRAMES),
            }[prop]

        def read(self):
            if self._i >= N_FRAMES:
                return False, None
            self._i += 1
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            pass

    def _pts(n, img):
        h, w = img.shape[:2]
        p = np.zeros((n, 1, 2), dtype=np.float32)
        p[:, 0, 0] = np.linspace(1.0, max(2.0, w - 2.0), n)
        p[:, 0, 1] = np.linspace(1.0, max(2.0, h - 2.0), n)
        return p

    m.VideoCapture = FakeCap
    m.cvtColor = lambda frame, code: frame[:, :, 0]
    m.goodFeaturesToTrack = (
        lambda img, maxCorners=12, qualityLevel=0.01, minDistance=8:
        _pts(min(int(maxCorners), 12), img)
    )
    m.cornerSubPix = (
        lambda gray, pts, winSize=None, zeroZone=None, criteria=None: pts
    )

    def calcOpticalFlowPyrLK(prev, cur, pts, nxt, winSize=None, maxLevel=None,
                             criteria=None, flags=0):
        n = len(pts)
        return (np.asarray(pts, dtype=np.float32).copy(),
                np.ones((n, 1), dtype=np.uint8),
                np.zeros((n, 1), dtype=np.float32))

    m.calcOpticalFlowPyrLK = calcOpticalFlowPyrLK

    def findEssentialMat(p0, p1, cameraMatrix=None, method=None, prob=None,
                         threshold=None):
        return np.eye(3), np.ones((len(p0), 1), dtype=np.uint8)

    m.findEssentialMat = findEssentialMat

    def recoverPose(E, p0, p1, cameraMatrix=None, mask=None):
        return (len(p0), np.eye(3), np.array([[0.0], [0.0], [1.0]]),
                np.ones((len(p0), 1), dtype=np.uint8))

    m.recoverPose = recoverPose
    return m


@pytest.fixture()
def fake_cv2(monkeypatch):
    stub = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", stub)
    return stub


def _expected_utcs(t0_offset_s: float) -> list[float]:
    """One sample per kept sample after the first (keep_every=1)."""
    return [BASE_UTC + t0_offset_s + i / FPS for i in range(1, N_FRAMES)]


class TestVioFrameToUtc:
    def test_run_vio_boottime(self, tmp_path, fake_cv2):
        from data_pipeline.vio import run_vio
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm)

        samples = run_vio(
            video_path=tmp_path / "v.mp4", recording_map=rec,
            frame_decim_hz=FPS, capture_meta=cm,
        )
        assert [s.utc_s for s in samples] == pytest.approx(
            _expected_utcs(0.0), abs=5e-3)

    def test_run_vio_boottime_without_meta_is_t0_early(self, tmp_path,
                                                       fake_cv2):
        """Documents the pre-fix failure: no t0 -> samples land ~100 s
        early on a boottime session."""
        from data_pipeline.vio import run_vio
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)

        samples = run_vio(
            video_path=tmp_path / "v.mp4", recording_map=rec,
            frame_decim_hz=FPS,
        )
        t0_s = VIDEO_T0_NS / 1e9
        assert [s.utc_s for s in samples] == pytest.approx(
            [u - t0_s for u in _expected_utcs(0.0)], abs=5e-3)

    def test_run_vio_multiframe_v1_chop_wins(self, tmp_path, fake_cv2):
        from data_pipeline.vio import run_vio_multiframe
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm, t0_ns=VIDEO_T0_NS)
        chop = tmp_path / "chop_video_anchor.txt"
        _write_video_anchor(chop, VIDEO_T0_NS + CHOP_OFFSET_NS)

        samples = run_vio_multiframe(
            video_path=tmp_path / "v.mp4", recording_map=rec,
            frame_decim_hz=FPS, min_inliers=5, use_v2=False,
            capture_meta=cm, chop_video_anchor=chop,
        )
        assert len(samples) == N_FRAMES - 1
        assert [s.utc_s for s in samples] == pytest.approx(
            _expected_utcs(CHOP_OFFSET_NS / 1e9), abs=5e-3)

    def test_run_vio_multiframe_v2_boottime(self, tmp_path, fake_cv2):
        from data_pipeline.vio import run_vio_multiframe
        rec = tmp_path / "recording_x.txt"
        _write_boottime_recording(rec)
        cm = tmp_path / "capture_meta.json"
        _write_capture_meta(cm)

        samples = run_vio_multiframe(
            video_path=tmp_path / "v.mp4", recording_map=rec,
            frame_decim_hz=FPS, min_inliers=5, use_v2=True,
            capture_meta=cm,
        )
        assert len(samples) == N_FRAMES - 1
        assert [s.utc_s for s in samples] == pytest.approx(
            _expected_utcs(0.0), abs=5e-3)

    def test_run_vio_legacy_unchanged(self, tmp_path, fake_cv2):
        from data_pipeline.vio import run_vio
        rec = tmp_path / "recording_legacy.txt"
        _write_legacy_recording(rec)

        a = run_vio(video_path=tmp_path / "v.mp4", recording_map=rec,
                    frame_decim_hz=FPS)
        b = run_vio(video_path=tmp_path / "v.mp4", recording_map=rec,
                    frame_decim_hz=FPS, capture_meta=None,
                    video_anchor=None, chop_video_anchor=None)
        assert [s.utc_s for s in a] == [s.utc_s for s in b]
        assert [s.utc_s for s in a] == pytest.approx(
            _expected_utcs(0.0), abs=5e-3)
