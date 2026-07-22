"""Unit tests for the Rate-signal + CV adaptive sample selector."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.stages import adaptive_frames as af
from data_pipeline.parsers import PosRow


def _pos(t: float, ve: float, vn: float) -> PosRow:
    return PosRow(
        utc_s=t, lat_deg=0.0, lon_deg=0.0, h_m=0.0,
        quality=1, vn=vn, ve=ve, vu=0.0,
    )


# ---------------------------------------------------------------------------
# Heading / speed helpers
# ---------------------------------------------------------------------------

def test_speed_at_returns_zero_outside_window() -> None:
    rows = [_pos(100.0, 0.0, 0.0), _pos(200.0, 0.0, 0.0)]
    assert af._speed_at(rows, 50.0) == 0.0
    assert af._speed_at(rows, 250.0) == 0.0


def test_speed_at_linear_interp() -> None:
    rows = [_pos(0.0, 0.0, 0.0), _pos(10.0, 0.0, 10.0)]  # speed 0 → 10
    assert math.isclose(af._speed_at(rows, 5.0), 5.0, abs_tol=1e-6)


def test_heading_deg_north() -> None:
    rows = [_pos(0.0, 0.0, 5.0), _pos(10.0, 0.0, 5.0)]
    h = af._heading_deg_at(rows, 5.0, 0.4)
    assert h is not None and abs(h - 0.0) < 1e-6


def test_heading_deg_east() -> None:
    rows = [_pos(0.0, 5.0, 0.0), _pos(10.0, 5.0, 0.0)]
    h = af._heading_deg_at(rows, 5.0, 0.4)
    assert h is not None and abs(h - 90.0) < 1e-6


def test_heading_deg_static_returns_none() -> None:
    rows = [_pos(0.0, 0.01, 0.01), _pos(10.0, 0.01, 0.01)]
    assert af._heading_deg_at(rows, 5.0, static_min_speed := 0.4) is None  # noqa: F841


def test_heading_rate_steady_turn() -> None:
    """30° turn over 2 s = 15 deg/s, sampled with a 1 s centred window."""
    rows: list[PosRow] = []
    speed = 5.0
    # Headings at t = 0.0, 1.0, 2.0 → 0°, 15°, 30° → 15 deg/s.
    for i, hd in enumerate([0.0, 15.0, 30.0]):
        rad = math.radians(hd)
        rows.append(_pos(
            t=i * 1.0,
            ve=speed * math.sin(rad),
            vn=speed * math.cos(rad),
        ))
    yr = af._heading_rate_dps(rows, 1.0, window_s=1.0, static_speed_mps=0.4)
    assert 10.0 < yr < 20.0, f"expected ~15 dps, got {yr}"


def test_angle_wrap_around_zero() -> None:
    assert af._angle_wrap_deg(370.0) == 10.0
    assert af._angle_wrap_deg(-190.0) == 170.0
    assert af._angle_wrap_deg(180.0) == 180.0


# ---------------------------------------------------------------------------
# Keypoint overlap fraction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not af._HAS_CV2, reason="cv2 not installed")
def test_orb_overlap_identical_images_is_one() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    ov = af._orb_overlap_fraction(img, img.copy())
    assert ov is not None and ov >= 0.95


@pytest.mark.skipif(not af._HAS_CV2, reason="cv2 not installed")
def test_orb_overlap_shifted_image_drops() -> None:
    rng = np.random.default_rng(123)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    shifted = np.roll(img, 80, axis=1)  # 25% horizontal shift
    ov = af._orb_overlap_fraction(img, shifted)
    # Synthetic noise plus a translation: overlap should be measurably <1.0.
    assert ov is None or ov < 1.0


@pytest.mark.skipif(not af._HAS_CV2, reason="cv2 not installed")
def test_orb_overlap_unrelated_images_either_none_or_small() -> None:
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    a = rng1.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    b = rng2.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    ov = af._orb_overlap_fraction(a, b)
    # Either too-weak match (None) or quite small overlap is acceptable.
    assert ov is None or ov < 0.5


# ---------------------------------------------------------------------------
# Real-binary end-to-end (gated on reference session sample data)
# ---------------------------------------------------------------------------

_REF_SESSION = Path(os.environ.get("DTF_REF_SESSION_DIR", ""))
VIDEO = _REF_SESSION / "recording_20260505_152247_615.mp4"
REC   = _REF_SESSION / "recording_20260505_152247_472.txt"
POS   = _REF_SESSION / "measurements_20260505_152247_472_javad_base.pos"


@pytest.mark.skipif(
    not (VIDEO.is_file() and REC.is_file() and POS.is_file()),
    reason="reference session sample data not present",
)
def test_compute_keep_list_smoke() -> None:
    """Selector returns a sensible keep list on the reference session session."""
    opts = af.AdaptiveOptions(
        spacing_m=4.0,            # bigger spacing → faster + smaller list
        turn_overlap=0.80,
        yaw_rate_threshold_dps=8.0,
    )
    res = af.compute_keep_list(
        video=VIDEO, pos_file=POS, recording_map=REC, options=opts,
        log=lambda _m: None,
    )
    # ~35 min @ 30 fps ≈ 60k+ source samples; expect a non-trivial keep list.
    assert res.n_total > 10000
    assert res.n_kept > 100
    # First and last kept indices must be monotonic and inside the source range.
    assert res.keep_indices[0] == 0
    assert all(b > a for a, b in zip(res.keep_indices, res.keep_indices[1:]))
    assert res.keep_indices[-1] < res.n_total
    # CRITICAL TIMING INVARIANT: every kept PTS must match the true
    # showinfo PTS for that source sample index. Anything else means the
    # spacing math + downstream Post-processing time-anchor lookups are running on
    # the wrong instant.
    true_pts = dict(af.enumerate_source_frames(VIDEO))
    for n, kept_pts in zip(res.keep_indices, res.keep_pts_s):
        true = true_pts[n]
        assert abs(kept_pts - true) < 1e-6, (
            f"PTS mismatch at n={n}: kept={kept_pts!r} true={true!r}"
        )


@pytest.mark.skipif(
    not VIDEO.is_file(),
    reason="reference session sample video not present",
)
def test_frames_run_streaming_carries_true_pts(tmp_path: "Path") -> None:
    """samples.run with select_indices + pts_for_indices must label every
    output sample with the supplied PTS to <1µs.

    Regression for the reference session incident where the streaming-extract path
    silently used ``n / src_fps`` and mislabelled every filename + CSV row
    when the source media had variable PTS cadence.
    """
    from data_pipeline.stages import frames as fs
    true_pts = dict(af.enumerate_source_frames(VIDEO))
    sample = [0, 1, 4, 5, 6, 30, 100, 500, 1000, 5000, 30000,
              max(true_pts) - 1]
    pts_in = [true_pts[i] for i in sample]
    out = tmp_path / "frames_out"
    out.mkdir()
    fr = fs.run(
        video=VIDEO, out_dir=out, fps=1.0, fmt="png",
        pts_name_decimals=6, rotation=0,
        select_indices=sample, pts_for_indices=pts_in,
        log=lambda _m: None,
    )
    import csv as _csv
    with open(fr.frame_times_csv) as f:
        rows = list(_csv.reader(f))[1:]
    assert len(rows) == len(sample)
    # ``sample`` is already index-sorted, and PTS is monotonic with source
    # index for this CFR sample, so CSV row order == sequential sample order.
    for seq, (r, n_src, expected) in enumerate(zip(rows, sample, pts_in)):
        csv_pts = float(r[1])
        assert abs(csv_pts - expected) < 1e-6, (
            f"streaming wrote wrong PTS for n={n_src}: csv={csv_pts!r} "
            f"expected={expected!r}"
        )
        # Dot-free zero-padded sequential filename (the external tool-label safe).
        expected_name = f"frame_{seq:06d}.png"
        assert r[0] == expected_name, (
            f"streaming wrote wrong filename for n={n_src}: got {r[0]!r} "
            f"expected {expected_name!r}"
        )
        # No dot anywhere except the real extension.
        assert r[0].count(".") == 1 and r[0].endswith(".png"), r[0]


# ---------------------------------------------------------------------------
# Cross-source timing invariants (different devices / codecs / resolutions)
# ---------------------------------------------------------------------------

# Local sentinel media files that are too valuable to drop. Anything else picked
# up automatically from the timing fixtures directory below.
TIMING_FIXTURE_VIDEOS: list[Path] = [
    VIDEO,  # reference session: 640x480, no B-samples, rot=-90
]

_REPO = Path(__file__).resolve().parent.parent
TIMING_FIXTURES_DIR = Path(
    os.environ.get("DTF_TIMING_FIXTURES_DIR",
                   str(_REPO / "test_fixtures" / "videos"))
)


def _discover_fixture_videos() -> list[Path]:
    found: list[Path] = list(TIMING_FIXTURE_VIDEOS)
    if TIMING_FIXTURES_DIR.is_dir():
        for ext in ("*.mp4", "*.mov", "*.mkv", "*.MP4"):
            found.extend(TIMING_FIXTURES_DIR.glob(ext))
            found.extend(TIMING_FIXTURES_DIR.glob(f"synthetic/{ext}"))
    # De-dup, preserve order, skip zero-byte garbage.
    seen: dict[Path, None] = {}
    out: list[Path] = []
    for p in found:
        if p.is_file() and p.stat().st_size > 0:
            key = p.resolve()
            if key not in seen:
                seen[key] = None
                out.append(p)
    return out


def _existing_fixtures() -> list[Path]:
    return _discover_fixture_videos()


@pytest.mark.parametrize("video", _existing_fixtures(),
                         ids=lambda p: p.name)
def test_cv2_count_matches_showinfo(video: Path) -> None:
    """cv2.VideoCapture must enumerate the same number of samples as the external converter
    showinfo for every supported source. A divergence (as happened on the
    deprecated raw rgb24 pipe with reference session: 62280 vs 62468) silently
    mislabels PTS for every kept sample downstream.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("OpenCV not installed")
    show_pts = af.enumerate_source_frames(video)
    assert show_pts, f"showinfo returned zero frames for {video}"
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f"cv2 could not open {video}"
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    assert n == len(show_pts), (
        f"{video.name}: cv2 saw {n} frames, showinfo saw {len(show_pts)}"
    )


@pytest.mark.parametrize("video", _existing_fixtures(),
                         ids=lambda p: p.name)
def test_cv2_pixels_match_select_eq_n_within_decoder_tolerance(
    tmp_path: Path, video: Path,
) -> None:
    """Sequential cv2.VideoCapture reads must produce cells within decoder
    rounding tolerance of ``the external converter -vf select=eq(n,N) PNG`` for every
    supported source.

    Tolerance budget: ``abs-mean ≤ 3.0`` and ``max ≤ 32`` per channel.

    Why a tolerance and not byte-identical: cv2 and the external converter's select+PNG
    path both decode via The external converter but apply their own swscale conversion
    chains. For 8-bit BT.601 sources (which reference session, the dashcams, and
    every samplelib clip happen to be) the chains pick bit-identical
    coefficients — and the empirical result IS np.array_equal. For
    10-bit HEVC (yuv420p10le) and tagged-BT.709 sources the chains
    diverge by 1–3 LSB per channel because of slightly different
    rounding when collapsing 10-bit YUV → 8-bit RGB. The remaining
    cell error is invisible to feature matchers / human eyes and
    irrelevant to coordinate tagging.

    Sample INDEX alignment — the cv2 ``read()`` count matching the
    showinfo ``n:`` — is asserted strictly in
    ``test_cv2_count_matches_showinfo``. That's the load-bearing
    invariant the pipeline depends on; this test just rules out
    catastrophic colour or rotation drift (>3 LSB mean would already
    break feature matching).
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import subprocess
        import numpy as np
    except ImportError:
        pytest.skip("OpenCV / NumPy not installed")
    from data_pipeline.ffmpeg_paths import resolve_ffmpeg
    ffmpeg = resolve_ffmpeg()
    show_pts = af.enumerate_source_frames(video)
    total = len(show_pts)
    sample = sorted(
        {0, 1, min(total - 1, 10), min(total - 1, 100), min(total - 1, 500)}
    )
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    try:
        next_idx = 0
        for n_target in sample:
            while next_idx <= n_target:
                ok, frame = cap.read()
                assert ok, f"early stream end at n={next_idx}, want {n_target}"
                next_idx += 1
            assert frame is not None
            ref_path = tmp_path / f"ref_{n_target}.png"
            subprocess.run(
                [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(video),
                 "-vf", f"select=eq(n\\,{n_target})", "-vframes", "1",
                 str(ref_path)],
                check=True,
            )
            ref = cv2.imread(str(ref_path))
            assert ref is not None
            assert frame.shape == ref.shape, (
                f"{video.name} n={n_target}: shape cv2={frame.shape} "
                f"ref={ref.shape}"
            )
            diff = np.abs(frame.astype("int16") - ref.astype("int16"))
            abs_mean = float(diff.mean())
            abs_max = int(diff.max())
            assert abs_mean <= 3.0 and abs_max <= 32, (
                f"{video.name} n={n_target}: cv2 vs select PNG diverge "
                f"(abs-mean={abs_mean:.3f}, max={abs_max})"
            )
    finally:
        cap.release()


# Reference set known a priori to be byte-identical (8-bit BT.601 family).
# This is the strict guarantee callers of the streaming extractor rely on
# for the common device-media case.
_BYTE_IDENTICAL_NAMES = {
    "recording_20260505_152247_615.mp4",  # reference session
    "bbb_1080_10s_h264.mp4",
    "jellyfish_1080_10s_h264_high.mp4",
    "samplelib_5s_1080p_h264.mp4",
    "samplelib_5s_720p_h264.mp4",
    "samplelib_5s_360p_h264.mp4",
    "samplelib_30s_1080p_h264.mp4",
    "samplelib_30s_720p_h264.mp4",
    "samplelib_10s_2160p_h264.mp4",
    "samplelib_10s_720p_h265.mp4",
    "samplelib_10s_720p_vp9.mp4",
    "rotate_90.mp4", "rotate_180.mp4", "rotate_270.mp4",
    "vfr_paused.mp4", "bt601_pure.mp4",
}


def _strict_fixtures() -> list[Path]:
    return [p for p in _existing_fixtures() if p.name in _BYTE_IDENTICAL_NAMES]


@pytest.mark.parametrize("video", _strict_fixtures(),
                         ids=lambda p: p.name)
def test_cv2_byte_identical_to_select_eq_n_strict(
    tmp_path: Path, video: Path,
) -> None:
    """Strict guarantee on the 8-bit-BT.601 reference corpus: cv2 cells
    must equal ``the external converter select=eq(n,N) PNG`` exactly. Catches catastrophic
    regressions while the tolerant test absorbs known 1–3 LSB decoder
    rounding on 10-bit and pure-BT.709 sources.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import subprocess
        import numpy as np
    except ImportError:
        pytest.skip("OpenCV / NumPy not installed")
    from data_pipeline.ffmpeg_paths import resolve_ffmpeg
    ffmpeg = resolve_ffmpeg()
    show_pts = af.enumerate_source_frames(video)
    total = len(show_pts)
    sample = sorted(
        {0, 1, min(total - 1, 10), min(total - 1, 100), min(total - 1, 500)}
    )
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    try:
        next_idx = 0
        for n_target in sample:
            while next_idx <= n_target:
                ok, frame = cap.read()
                assert ok, f"early stream end at n={next_idx}, want {n_target}"
                next_idx += 1
            ref_path = tmp_path / f"ref_{n_target}.png"
            subprocess.run(
                [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(video),
                 "-vf", f"select=eq(n\\,{n_target})", "-vframes", "1",
                 str(ref_path)],
                check=True,
            )
            ref = cv2.imread(str(ref_path))
            assert frame.shape == ref.shape, (
                f"{video.name} n={n_target}: shape mismatch"
            )
            assert np.array_equal(frame, ref), (
                f"{video.name} n={n_target}: NOT byte-identical "
                f"(abs-mean={np.abs(frame.astype('int16')-ref.astype('int16')).mean():.4f})"
            )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Boottime / segment sample->UTC mapping in compute_keep_list (synthetic, no media)
# ---------------------------------------------------------------------------
#
# For boottime-format sessions the session-map anchor's x-domain is absolute
# CLOCK_BOOTTIME ns, so mapping a raw PTS directly (video_pts_to_utc_s) lands
# ~t0 seconds early (verified 505.8 s early on the day15 segment): every sample
# falls outside the .pos window, _speed_at/_heading_rate_dps return 0, and the
# selector runs blind (keeps only via the max_interval_s floor). These tests
# synthesize such a session (no real media/the external converter: decode + enumeration + Post-processing
# parsing are monkeypatched) and assert the t0-lifted mapping puts samples
# INSIDE the window so the Rate-signal spacing rule actually runs.

_BT_BOOT0 = 5_000_000_000_000        # anchor span starts 5000 s after boot
_BT_UTC0 = 1_700_000_000.0
_BT_FPS = 10.0
_BT_N_FRAMES = 201                   # 20 s of media
_BT_VIDEO_T0_OFFSET_S = 10.0         # media sample 0 = 10 s into the anchor span


def _bt_iso(utc_s: float) -> str:
    d = _dt.datetime.fromtimestamp(utc_s, tz=_dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _bt_write_video_anchor(path: Path, boots) -> None:
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i, b in enumerate(boots):
        lines.append(f"{i},{int(b)},{int(b)},REALTIME")
    path.write_text("\n".join(lines), encoding="utf-8")


def _bt_env(monkeypatch, tmp_path: Path, *, boottime_anchor: bool):
    """Fake the media-decode side of compute_keep_list on a synthetic session.

    Returns ``(media, recording_map, pos_file, video_t0_boot_ns)``. The Post-processing
    path is constant 5 m/s due-east and covers the samples' TRUE UTC
    window; with spacing_m=2.0 the selector should keep ~1 sample per 0.4 s.
    """
    pts_list = [(n, n / _BT_FPS) for n in range(_BT_N_FRAMES)]
    monkeypatch.setattr(af, "enumerate_source_frames", lambda _v: list(pts_list))

    class _FakeStream:
        def __init__(self, video, thumb_h):
            self._it = iter(pts_list)

        def read_next(self):
            try:
                n, _pts = next(self._it)
            except StopIteration:
                return None
            return n, np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(af, "_ThumbStream", _FakeStream)

    video_t0_boot = _BT_BOOT0 + int(_BT_VIDEO_T0_OFFSET_S * 1e9)

    # Session map: 61 anchor rows over 60 s (slope exactly 1 s/s).
    lines = []
    for i in range(61):
        x = (_BT_BOOT0 + i * 1_000_000_000) if boottime_anchor else i * 1_000_000_000
        lines.append(f"{x},{_bt_iso(_BT_UTC0 + i)}")
    rec = tmp_path / "recording_map.txt"
    rec.write_text("\n".join(lines), encoding="utf-8")

    # True sample UTCs: boottime session -> UTC0 + 10 + pts (pts 0..20);
    # legacy session -> UTC0 + pts. Cover both generously.
    rows = [_pos(float(t), 5.0, 0.0)
            for t in np.arange(_BT_UTC0 - 5.0, _BT_UTC0 + 45.0, 1.0)]
    monkeypatch.setattr(af, "parse_rtkpos", lambda _p: list(rows))

    video = tmp_path / "dummy.mp4"
    video.write_text("", encoding="utf-8")
    pos = tmp_path / "dummy.pos"
    pos.write_text("", encoding="utf-8")
    return video, rec, pos, video_t0_boot


def test_keep_list_boottime_session_uses_capture_meta_t0(
    monkeypatch, tmp_path: Path,
) -> None:
    video, rec, pos, t0 = _bt_env(monkeypatch, tmp_path, boottime_anchor=True)
    meta = tmp_path / "capture_meta.json"
    meta.write_text(json.dumps({
        "anchor_format": 2,
        "video": {"video_t0_boottime_ns": t0},
    }), encoding="utf-8")
    opts = af.AdaptiveOptions(spacing_m=2.0)

    # WITHOUT the t0 the raw-PTS mapping lands ~5000 s early -> outside the
    # .pos window -> selector is blind and keeps only sample 0 (20 s media <
    # 30 s max_interval floor). This documents the pre-fix failure mode.
    blind = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        log=lambda _m: None,
    )
    assert blind.n_kept == 1

    # WITH capture_meta the samples land in-window: 5 m/s @ spacing 2 m over
    # 20 s -> ~51 keeps via the Rate-signal distance rule (not the interval floor).
    res = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        capture_meta=meta, log=lambda _m: None,
    )
    assert res.n_total == _BT_N_FRAMES
    assert res.keep_indices[0] == 0
    assert res.n_kept >= 40, f"selector still blind: kept {res.n_kept}"
    assert res.n_straight_decisions > 0
    assert all(b > a for a, b in zip(res.keep_indices, res.keep_indices[1:]))


def test_keep_list_boottime_session_uses_video_anchor_min(
    monkeypatch, tmp_path: Path,
) -> None:
    """No capture_meta: the session video_anchor min(bootNs) resolves t0."""
    video, rec, pos, t0 = _bt_env(monkeypatch, tmp_path, boottime_anchor=True)
    va = tmp_path / "recording.video_anchor.txt"
    boots = [t0 + i * 100_000_000 for i in range(10)]
    boots[0], boots[1] = boots[1], boots[0]  # first row is NOT the min
    _bt_write_video_anchor(va, boots)

    res = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec,
        options=af.AdaptiveOptions(spacing_m=2.0),
        video_anchor=va, log=lambda _m: None,
    )
    assert res.n_kept >= 40


def test_keep_list_chop_anchor_wins_over_capture_meta(
    monkeypatch, tmp_path: Path,
) -> None:
    """Segment clip: the segment's own video_anchor t0 must WIN over capture_meta.

    capture_meta carries the PARENT full-session t0 (600 s earlier here);
    using it maps every segment sample outside the .pos window (blind selector).
    """
    video, rec, pos, t0 = _bt_env(monkeypatch, tmp_path, boottime_anchor=True)
    meta = tmp_path / "capture_meta.json"
    meta.write_text(json.dumps({
        "anchor_format": 2,
        "video": {"video_t0_boottime_ns": t0 - 600_000_000_000},
    }), encoding="utf-8")
    chop = tmp_path / "chop.video_anchor.txt"
    _bt_write_video_anchor(chop, [t0 + i * 100_000_000 for i in range(10)])
    opts = af.AdaptiveOptions(spacing_m=2.0)

    # Parent t0 only -> samples land 600 s early -> blind.
    parent = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        capture_meta=meta, log=lambda _m: None,
    )
    assert parent.n_kept == 1

    # Segment anchor overrides -> samples in-window -> Rate-signal rule runs.
    res = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        capture_meta=meta, chop_video_anchor=chop, log=lambda _m: None,
    )
    assert res.n_kept >= 40


def test_keep_list_legacy_session_unchanged(
    monkeypatch, tmp_path: Path,
) -> None:
    """Legacy video_ns sessions: no t0 resolves -> byte-for-byte old mapping."""
    video, rec, pos, _t0 = _bt_env(monkeypatch, tmp_path, boottime_anchor=False)
    opts = af.AdaptiveOptions(spacing_m=2.0)

    base = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        log=lambda _m: None,
    )
    # Legacy mapping puts samples in-window directly; the selector works.
    assert base.n_kept >= 40
    assert base.keep_indices[0] == 0

    # A capture_meta WITHOUT video_t0_boottime_ns must not change anything.
    meta = tmp_path / "capture_meta.json"
    meta.write_text(json.dumps({"video": {}}), encoding="utf-8")
    same = af.compute_keep_list(
        video=video, pos_file=pos, recording_map=rec, options=opts,
        capture_meta=meta, log=lambda _m: None,
    )
    assert same.keep_indices == base.keep_indices
    assert same.keep_pts_s == base.keep_pts_s


@pytest.mark.parametrize("video", _existing_fixtures(),
                         ids=lambda p: p.name)
def test_showinfo_pts_monotonic(video: Path) -> None:
    """Every fixture media must have strictly increasing showinfo PTS.
    Decoder reorder bugs would surface here.
    """
    pts = af.enumerate_source_frames(video)
    assert len(pts) >= 2, f"{video.name} has <2 frames"
    deltas = [b[1] - a[1] for a, b in zip(pts, pts[1:])]
    bad = [i for i, d in enumerate(deltas) if d <= 0]
    assert not bad, (
        f"{video.name}: non-monotonic PTS at frame indices {bad[:5]}"
    )
