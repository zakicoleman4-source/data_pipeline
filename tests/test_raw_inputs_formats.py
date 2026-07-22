"""Tests for RawInputs old/new capture-format detection + altitude smoothing.

Covers:

JOB 1 (RawInputs.from_folder):
  * old format (4 core files, no capture_meta) detected as "old";
  * new format (capture_meta anchor_format=2 + per-sample video_anchor) detected
    as "new" and exposes video_anchor_txt;
  * extra files (audio_*.wav, video_anchor.txt) never cause a failure and do NOT
    collide with the recording_*.txt time-anchor glob;
  * a missing required file raises an error that lists what IS present;
  * graceful handling when optional files are absent.
  * an end-to-end georef.run() on BOTH an old-format and a new-format synthetic
    session produces a path CSV without error.

JOB 3 (altitude smoothing opt-in):
  * CsvOptions.smooth_altitude tri-state controls the Z sigma returned by
    sigmas() (None=legacy, False=off, True=on with its own sigma).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from data_pipeline.pipeline import RawInputs
from data_pipeline.stages.georef import CsvOptions


# --------------------------------------------------------------------------- #
# Synthetic session builders                                                  #
# --------------------------------------------------------------------------- #

_START_UTC = 1704110400.0  # 2024-01-01 12:00:00 UTC


def _iso(utc_s: float) -> str:
    """ISO-8601 UTC string as the recording_*.txt time column carries it."""
    return (dt.datetime.fromtimestamp(utc_s, dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z")


def _write_recording_video_ns(path: Path, n: int = 40) -> None:
    """OLD format: column 0 = relative video_ns (starts near 0)."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            t = i * 30.0 / (n - 1)
            video_ns = 1_000_000_000 + t * 1e9
            utc = _START_UTC + t
            f.write(f"{video_ns:.0f},{_iso(utc)}\n")


def _write_recording_boottime(path: Path, t0_boot_ns: int, n: int = 40) -> None:
    """NEW format: column 0 = absolute CLOCK_BOOTTIME ns (large values)."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            t = i * 30.0 / (n - 1)
            boot_ns = t0_boot_ns + t * 1e9
            utc = _START_UTC + t
            f.write(f"{boot_ns:.0f},{_iso(utc)}\n")


def _write_pos(path: Path, n: int = 60) -> None:
    lat_base, lon_base, h_base = 37.3382, -122.0324, 10.0
    with path.open("w", encoding="utf-8") as f:
        f.write("% Sample RTKLIB .pos file\n")
        for i in range(n):
            t_utc = _START_UTC + i
            o = dt.datetime.fromtimestamp(t_utc, dt.timezone.utc)
            date_str = o.strftime("%Y/%m/%d")
            time_str = o.strftime("%H:%M:%S.%f")[:12]
            lat = lat_base + (i - n // 2) * 1e-5
            lon = lon_base + (i - n // 2) * 1e-5
            h = h_base + (i % 5) * 0.2
            # The external solver layout: date time lat lon h Q ns sdn sde sdu sdne sdeu
            # sdun age ratio (ns is the integer source count at parts[6]).
            f.write(
                f"{date_str} {time_str} {lat:.8f} {lon:.8f} {h:.4f} 1 20 "
                f"0.0100 0.0100 0.0300 0.0 0.0 0.0 0.0 10.0\n"
            )


def _write_frame_times(path: Path, n: int = 120, fps: float = 6.0) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Image,t_video_s\n")
        for i in range(n):
            f.write(f"IMG_{i:06d}.jpg,{i / fps:.6f}\n")


def _write_measurements(path: Path) -> None:
    path.write_text("# measurements placeholder\n", encoding="utf-8")


def _write_sensors(path: Path) -> None:
    # Minimal sensors file; coordinate output tolerates an empty Motion sensor parse.
    path.write_text("# sensors placeholder\n", encoding="utf-8")


def _write_capture_meta(path: Path, video_name: str, t0_boot_ns: int,
                        anchor_format: int = 2) -> None:
    import json
    path.write_text(json.dumps({
        "anchor_format": anchor_format,
        "video": {
            "mp4": video_name,
            "video_t0_boottime_ns": t0_boot_ns,
            "timestamp_source": "boottime",
        },
        "audio": {"timebase": "boottime"},
        "clock": {"mono_to_boot_offset_ns": 0},
    }), encoding="utf-8")


def _write_video_anchor(path: Path, t0_boot_ns: int, n: int = 120,
                        fps: float = 6.0) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# frameNumber,sensorTimestampNs(raw),bootNs,timestampSource\n")
        for i in range(n):
            boot_ns = int(t0_boot_ns + (i / fps) * 1e9)
            f.write(f"{i},{boot_ns},{boot_ns},hal\n")


def _make_old_session(folder: Path) -> None:
    _write_measurements(folder / "measurements_111.txt")
    _write_recording_video_ns(folder / "recording_111.txt")
    _write_sensors(folder / "sensors_111.txt")
    (folder / "recording_111.mp4").write_bytes(b"\x00")


def _make_new_session(folder: Path, t0_boot_ns: int = 5_000_000_000_000) -> None:
    _write_measurements(folder / "measurements_222.txt")
    _write_recording_boottime(folder / "recording_222.txt", t0_boot_ns)
    _write_sensors(folder / "sensors_222.txt")
    (folder / "recording_222.mp4").write_bytes(b"\x00")
    # Extra new-format files:
    (folder / "audio_222.wav").write_bytes(b"\x00")
    (folder / "audio_anchor_222.txt").write_text("0,0\n", encoding="utf-8")
    _write_video_anchor(folder / "recording_222.video_anchor.txt", t0_boot_ns)
    _write_capture_meta(folder / "capture_meta.json", "recording_222.mp4",
                        t0_boot_ns)


# --------------------------------------------------------------------------- #
# JOB 1 — detection / tolerance                                               #
# --------------------------------------------------------------------------- #

def test_old_format_detected(tmp_path):
    _make_old_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)
    assert ri.capture_format == "old"
    assert ri.is_new_format is False
    assert ri.capture_meta_json is None
    assert ri.video_anchor_txt is None
    assert ri.has_per_frame_anchor is False
    assert ri.recording_txt.name == "recording_111.txt"
    assert ri.recording_mp4 is not None


def test_new_format_detected_with_extras(tmp_path):
    _make_new_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)
    assert ri.capture_format == "new"
    assert ri.is_new_format is True
    assert ri.anchor_format == 2
    assert ri.capture_meta_json is not None
    assert ri.audio_anchor_txt is not None
    # The per-sample anchor is detected and exposed:
    assert ri.video_anchor_txt is not None
    assert ri.video_anchor_txt.name == "recording_222.video_anchor.txt"
    assert ri.has_per_frame_anchor is True


def test_recording_glob_does_not_collide_with_video_anchor(tmp_path):
    """recording_*.txt must resolve to the timing file, NOT the video_anchor."""
    _make_new_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)
    # Both recording_222.txt and recording_222.video_anchor.txt are present;
    # the time-anchor file must be the plain one.
    assert ri.recording_txt.name == "recording_222.txt"


def test_extra_files_never_fail(tmp_path):
    """Presence of audio_*.wav / video_anchor.txt must not raise."""
    _make_old_session(tmp_path)
    # Drop in extra files that the old pipeline never saw:
    (tmp_path / "audio_111.wav").write_bytes(b"\x00")
    (tmp_path / "stray_notes.txt").write_text("hi\n", encoding="utf-8")
    ri = RawInputs.from_folder(tmp_path)  # must not raise
    assert ri.capture_format == "old"


def test_missing_required_lists_present_files(tmp_path):
    # Only measurements + sensors; recording_*.txt missing.
    _write_measurements(tmp_path / "measurements_333.txt")
    _write_sensors(tmp_path / "sensors_333.txt")
    (tmp_path / "audio_333.wav").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError) as ei:
        RawInputs.from_folder(tmp_path)
    msg = str(ei.value)
    assert "recording_*.txt" in msg
    # Error lists what IS present:
    assert "measurements_333.txt" in msg
    assert "sensors_333.txt" in msg


def test_missing_optional_graceful(tmp_path):
    """No container file, no capture_meta, no audio_anchor -> still resolves (old)."""
    _write_measurements(tmp_path / "measurements_444.txt")
    _write_recording_video_ns(tmp_path / "recording_444.txt")
    _write_sensors(tmp_path / "sensors_444.txt")
    ri = RawInputs.from_folder(tmp_path)
    assert ri.recording_mp4 is None
    assert ri.capture_meta_json is None
    assert ri.audio_anchor_txt is None
    assert ri.video_anchor_txt is None
    assert ri.capture_format == "old"


def test_new_format_without_capture_meta_still_detected(tmp_path):
    """A per-sample video_anchor with no manifest still flags the new layout."""
    _write_measurements(tmp_path / "measurements_555.txt")
    _write_recording_boottime(tmp_path / "recording_555.txt", 5_000_000_000_000)
    _write_sensors(tmp_path / "sensors_555.txt")
    _write_video_anchor(tmp_path / "recording_555.video_anchor.txt",
                        5_000_000_000_000)
    ri = RawInputs.from_folder(tmp_path)
    assert ri.capture_format == "new"
    assert ri.video_anchor_txt is not None


# --------------------------------------------------------------------------- #
# JOB 1 — end-to-end coordinate output on both formats                                    #
# --------------------------------------------------------------------------- #

def _run_georef(folder: Path, ri: RawInputs, out_csv: Path):
    from data_pipeline.stages import georef
    ft = folder / "frame_times.csv"
    _write_frame_times(ft)
    pos = folder / "session.pos"
    _write_pos(pos)
    return georef.run(
        frame_times_csv=ft,
        recording_map=ri.recording_txt,
        pos_file=pos,
        data_log=ri.measurements_txt,
        sensors_txt=ri.sensors_txt,
        out_csv=out_csv,
        fps=6.0,
        options=CsvOptions(smoothing="car", add_ypr=False),
        capture_meta=ri.capture_meta_json,
        video_anchor=ri.video_anchor_txt,
    )


def test_end_to_end_old_format(tmp_path):
    _make_old_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)
    out = tmp_path / "georef_old.csv"
    res = _run_georef(tmp_path, ri, out)
    assert out.exists()
    assert res.n_with_position > 0


def test_end_to_end_new_format(tmp_path):
    _make_new_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)
    out = tmp_path / "georef_new.csv"
    res = _run_georef(tmp_path, ri, out)
    assert out.exists()
    assert res.n_with_position > 0


def test_georef_image_equals_frame_filename_with_extension(tmp_path):
    """Synthetic coordinate output run: the georef.csv ``Image`` column must equal the
    EXACT dot-free sample filename WITH extension (the external tool source-label match),
    and every Image must correspond to a row in extracted_frame_times.csv.

    Contract: sample file ``frame_000001.png`` <-> CSV Image ``frame_000001.png``
    <-> coordinate output Image ``frame_000001.png`` — all identical, dot-free except the
    real extension.
    """
    import csv as _csv
    from data_pipeline.stages import georef

    _make_old_session(tmp_path)
    ri = RawInputs.from_folder(tmp_path)

    # extracted_frame_times.csv with DOT-FREE sequential sample names (.png).
    ft = tmp_path / "extracted_frame_times.csv"
    n, fps = 60, 6.0
    expected_names = [f"frame_{i:06d}.png" for i in range(n)]
    with ft.open("w", encoding="utf-8") as f:
        f.write("Image,t_video_s\n")
        for i, name in enumerate(expected_names):
            f.write(f"{name},{i / fps:.12f}\n")

    pos = tmp_path / "session.pos"
    _write_pos(pos)
    out = tmp_path / "Georef.csv"
    res = georef.run(
        frame_times_csv=ft,
        recording_map=ri.recording_txt,
        pos_file=pos,
        data_log=ri.measurements_txt,
        sensors_txt=ri.sensors_txt,
        out_csv=out,
        fps=fps,
        options=CsvOptions(smoothing="car", add_ypr=False),  # default keep-ext
    )
    assert res.n_with_position > 0
    assert out.exists()

    # Read the coordinate output Image column (skip the leading '#'-comment header line).
    with out.open("r", encoding="utf-8") as f:
        reader = _csv.reader(r for r in f if not r.startswith("#"))
        rows = list(reader)
    header, data = rows[0], rows[1:]
    assert header[0] == "Image"
    georef_images = [r[0] for r in data]
    assert georef_images, "georef produced no data rows"

    expected_set = set(expected_names)
    for img in georef_images:
        # Equals the exact on-disk sample filename, WITH extension.
        assert img in expected_set, f"georef Image {img!r} not a known frame file"
        # Dot-free except the real extension.
        assert img.count(".") == 1 and img.endswith(".png"), img
        # No bare-stem stripping happened.
        assert img != Path(img).stem


def test_end_to_end_new_format_t0_recovered_from_video_anchor(tmp_path):
    """No capture_meta -> boottime t0 is recovered from video_anchor.txt."""
    t0 = 5_000_000_000_000
    _write_measurements(tmp_path / "measurements_666.txt")
    _write_recording_boottime(tmp_path / "recording_666.txt", t0)
    _write_sensors(tmp_path / "sensors_666.txt")
    (tmp_path / "recording_666.mp4").write_bytes(b"\x00")
    _write_video_anchor(tmp_path / "recording_666.video_anchor.txt", t0)
    ri = RawInputs.from_folder(tmp_path)
    assert ri.capture_format == "new"
    assert ri.capture_meta_json is None
    assert ri.video_anchor_txt is not None
    out = tmp_path / "georef_recovered.csv"
    res = _run_georef(tmp_path, ri, out)
    assert out.exists()
    assert res.n_with_position > 0


# --------------------------------------------------------------------------- #
# JOB 3 — altitude smoothing opt-in                                           #
# --------------------------------------------------------------------------- #

def test_smooth_altitude_legacy_default():
    """Default (None) keeps the profile's Z sigma (legacy behaviour)."""
    opt = CsvOptions(smoothing="car")  # car profile z = 10s
    xy, z = opt.sigmas()
    assert z == pytest.approx(10.0)
    assert opt.smooth_altitude is None


def test_smooth_altitude_off_zeroes_z():
    """Explicit OFF passes Z through raw (sigma 0) regardless of profile."""
    opt = CsvOptions(smoothing="car", smooth_altitude=False)
    xy, z = opt.sigmas()
    assert z == 0.0
    # XY is untouched.
    assert xy == pytest.approx(2.0)


def test_smooth_altitude_on_uses_dedicated_sigma():
    opt = CsvOptions(smoothing="car", smooth_altitude=True,
                     altitude_smooth_sigma_s=25.0)
    xy, z = opt.sigmas()
    assert z == pytest.approx(25.0)


def test_smooth_altitude_on_falls_back_to_profile():
    """ON without a dedicated sigma keeps the resolved profile/override z."""
    opt = CsvOptions(smoothing="car", smooth_altitude=True)
    xy, z = opt.sigmas()
    assert z == pytest.approx(10.0)


def test_smooth_altitude_independent_of_xy():
    """Turning altitude smoothing off leaves horizontal smoothing intact."""
    opt = CsvOptions(smoothing="aggressive", smooth_altitude=False)
    xy, z = opt.sigmas()
    assert xy == pytest.approx(5.0)  # aggressive xy
    assert z == 0.0
