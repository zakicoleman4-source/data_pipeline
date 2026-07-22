import datetime as dt
import json
from pathlib import Path

from data_pipeline.stages import viewers
from data_pipeline.stages.viewers import build_sync_player


def test_placeholder_replaced_null_safe():
    # The token helper fills __IMU_STRIP__ and leaves no literal placeholder
    # behind, even when there is no Motion sensor (null).
    out = viewers._render_imu_strip_token("<body>IMU=__IMU_STRIP__;</body>", None)
    assert "__IMU_STRIP__" not in out
    assert "IMU=null;" in out


def test_placeholder_replaced_with_payload():
    payload = {"t_video": [0.0, 1.0], "flags": {"gyro_dead": True}}
    out = viewers._render_imu_strip_token("x=__IMU_STRIP__", payload)
    assert "__IMU_STRIP__" not in out
    assert json.loads(out.split("x=", 1)[1])["flags"]["gyro_dead"] is True


def test_template_has_imu_panel_scaffold():
    html = (Path(__file__).resolve().parents[1]
            / "data_pipeline" / "assets" / "sync_player.html").read_text(encoding="utf-8")
    assert "__IMU_STRIP__" in html            # token present for the builder
    assert 'id="imuTrust"' in html            # panel scaffold present
    assert "imuTrustToggle" in html           # Local/Whole toggle present


def test_template_has_verdict_and_wedge():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1]
            / "data_pipeline" / "assets" / "sync_player.html").read_text(encoding="utf-8")
    assert "verdict" in html          # verdict badge wired
    assert "drift" in html            # drift readout wired
    assert "vel_source" in html       # heading-source note wired


# ---------------------------------------------------------------------------
# Intuitive labels / legend / time display (client-clarity pass)
# ---------------------------------------------------------------------------
# Build a real sync player from synthesized inputs (same pattern as
# tests/test_skyplot.py) and assert the new client-facing labels are emitted.

def _write_minimal_pos(path: Path, n_epochs: int = 5) -> None:
    """Minimal solver .pos: 1 Hz, base 2026/05/05 12:23:07 GPST (=UTC+18s)."""
    header = (
        "% GPST          latitude(deg) longitude(deg)  height(m)   Q  ns"
        "   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m)"
        " age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)"
        "   sdvn     sdve     sdvu    sdvne    sdveu    sdvun\n"
    )
    lines = [header]
    base = dt.datetime(2026, 5, 5, 12, 23, 7)
    for i in range(n_epochs):
        t = base + dt.timedelta(seconds=i)
        ts = t.strftime("%Y/%m/%d %H:%M:%S") + ".000"
        lines.append(
            f"{ts}   31.500000000   34.800000000    100.000  1  12"
            f"   0.010   0.010   0.020   0.000   0.000   0.000   0.5  99.9"
            f"   0.500  0.200  0.000   0.01   0.01   0.02   0.00   0.00   0.00\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_recording_txt(path: Path, n_frames: int = 5) -> None:
    """recording_*.txt anchor: video_ns -> ISO-UTC (UTC = GPST - 18 s)."""
    base_utc = dt.datetime(2026, 5, 5, 12, 22, 49, tzinfo=dt.timezone.utc)
    lines = []
    for i in range(n_frames):
        utc = base_utc + dt.timedelta(seconds=i)
        lines.append(f"{i * 1_000_000_000},{utc.isoformat()},unused\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_frame_times_csv(path: Path, n_frames: int = 5) -> None:
    lines = ["Image,t_video_s\n"]
    for i in range(n_frames):
        lines.append(f"frame_{i}.jpg,{float(i):.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _build_player(tmp_path: Path, **extra) -> str:
    (tmp_path / "video.mp4").write_bytes(b"\x00" * 64)
    _write_minimal_pos(tmp_path / "test.pos")
    _write_recording_txt(tmp_path / "recording.txt")
    _write_frame_times_csv(tmp_path / "frame_times.csv")
    out = tmp_path / "out" / "sync_player.html"
    build_sync_player(
        video=tmp_path / "video.mp4",
        pos_file=tmp_path / "test.pos",
        frame_times_csv=tmp_path / "frame_times.csv",
        recording_map=tmp_path / "recording.txt",
        out_html=out,
        **extra,
    )
    return out.read_text(encoding="utf-8")


def test_emitted_html_has_intuitive_labels(tmp_path: Path):
    html = _build_player(tmp_path)
    # 1. IMU/sensor strips: title + units + plain-language caption
    assert "Yaw rate" in html
    assert "how fast the car is turning" in html
    assert "braking (−) / accelerating (+)" in html
    assert "cornering" in html
    assert "deg/s" in html
    assert "m/s²" in html
    # 2. Shoebox legend: box vs needle, heading sources, North
    assert "travel direction" in html
    assert "GPS-PPK positions" in html
    assert "nose = forward" in html
    assert "deg from North" in html
    assert "N = North" in html
    # GPS-post-processed marker meaning stated plainly
    assert "post-processed GPS" in html
    # 3. Time display: which time is shown
    assert "Video time" in html
    assert "GPS position time" in html
    assert 'id="tr-utc"' in html
    # 4. Collapsible help box
    assert "What am I looking at" in html
    assert "<details" in html
    # HUD speed sources renamed to plain language
    assert "GPS VEL" in html
    assert "PHONE GPS" in html


def test_clip_note_null_for_full_session(tmp_path: Path):
    html = _build_player(tmp_path)
    assert "__CLIP_NOTE__" not in html          # token always replaced
    assert "const CLIP_NOTE  = null;" in html   # no trimmed-clip note
    assert "Trimmed clip" not in html


def test_clip_note_set_for_trimmed_clip(tmp_path: Path):
    # Segment ("chop") anchor whose min bootNs maps sample 0 into the window.
    chop_anchor = tmp_path / "chop.video_anchor.txt"
    lines = ["# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource"]
    for i in range(5):
        boot = 1_000_000 + i * 1_000_000_000
        lines.append(f"{9000 + i},{boot},{boot},REALTIME")
    chop_anchor.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html = _build_player(tmp_path, chop_video_anchor=chop_anchor)
    assert "__CLIP_NOTE__" not in html
    assert "Trimmed clip" in html
    # The note explains the video-0 vs recording-start offset in plain words.
    assert "START OF THIS CLIP" in html
