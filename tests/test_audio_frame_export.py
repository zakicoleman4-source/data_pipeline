"""Tests for the stream-zero sample export (pure math + naming + CSV shape).

No the external converter / no real session required: extraction is exercised elsewhere;
here we pin the timeline math, the sortable millisecond naming, and that the
emitted CSV keeps the Coordinate output-style column shape the repo's ``frame_compare``
loaders already understand.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_pipeline.audio_frame_export import (
    EXTERNAL_CSV_HEADER,
    FrameAudioRow,
    compute_frame_audio_times,
    frame_name_for_time,
    write_external_csv,
)


# ---------------------------------------------------------------------------
# frame_name_for_time
# ---------------------------------------------------------------------------


class TestFrameNameForTime:
    def test_zero(self):
        assert frame_name_for_time(0.0) == "0000.000"

    def test_millisecond_rounding(self):
        assert frame_name_for_time(12.3456) == "0012.346"

    def test_exact_hundred(self):
        assert frame_name_for_time(100.0) == "0100.000"

    def test_half_up_rounding(self):
        # 0.0005 rounds half-up to 0.001 (not banker's to 0.000).
        assert frame_name_for_time(0.0005) == "0000.001"

    def test_sub_millisecond_negative_zero_is_accepted(self):
        # -0.0004 is 0.000 at millisecond precision — must not raise.
        assert frame_name_for_time(-0.0004) == "0000.000"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            frame_name_for_time(-0.5)

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            frame_name_for_time(float("nan"))
        with pytest.raises(ValueError):
            frame_name_for_time(float("inf"))

    def test_lexicographic_order_matches_time_order(self):
        times = [0.0, 0.001, 0.167, 1.0, 2.5, 9.999, 10.0, 12.3456,
                 100.0, 999.75, 1000.0, 9999.999]
        names = [frame_name_for_time(t) for t in times]
        assert names == sorted(names)

    def test_decimals_parameter(self):
        assert frame_name_for_time(12.3456, decimals=4) == "0012.3456"

    def test_stem_has_single_dot(self):
        # One dot in the stem => Path(...).stem strips only the extension.
        name = frame_name_for_time(12.3456) + ".png"
        assert Path(name).stem == "0012.346"


# ---------------------------------------------------------------------------
# compute_frame_audio_times (pure path: anchors supplied, no disk access)
# ---------------------------------------------------------------------------


class _FakeBootAnchor:
    """Minimal TimeAnchor stand-in: utc_s = (boot_ns * 1e-9) + offset."""

    OFFSET_S = 1.7e9

    def boottime_to_utc_s(self, boot_ns: float) -> float:
        return boot_ns * 1e-9 + self.OFFSET_S


class TestComputeFrameAudioTimes:
    AUDIO_START_BOOT = 104_157_489_198_363.0   # ns
    VIDEO_T0_BOOT = 104_157_885_413_670.0      # ns (media starts after stream)

    def _rows(self, frame_times, boot_anchor=None):
        return compute_frame_audio_times(
            None,
            frame_times,
            audio_start_boot_ns=self.AUDIO_START_BOOT,
            video_t0_boot_ns=self.VIDEO_T0_BOOT,
            boot_anchor=boot_anchor,
        )

    def test_exact_formula(self):
        frame_times = [("frame_000000.png", 0.0),
                       ("frame_000001.png", 0.2001),
                       ("frame_000002.png", 12.345678)]
        rows = self._rows(frame_times)
        assert len(rows) == 3
        for (img, pts), row in zip(frame_times, rows):
            expected = (
                self.VIDEO_T0_BOOT + pts * 1e9 - self.AUDIO_START_BOOT
            ) / 1e9
            assert row.image == img
            assert row.t_video_s == pts
            assert row.t_audio_s == expected  # exact same float arithmetic

    def test_frame0_offset_is_video_minus_audio_start(self):
        rows = self._rows([("frame_000000.png", 0.0)])
        expected = (self.VIDEO_T0_BOOT - self.AUDIO_START_BOOT) / 1e9
        assert rows[0].t_audio_s == pytest.approx(expected, abs=1e-12)
        assert rows[0].t_audio_s > 0  # media started after stream here

    def test_pre_audio_frames_are_identifiable(self):
        # Stream starts AFTER the media: sample at PTS 0 is pre-stream.
        rows = compute_frame_audio_times(
            None,
            [("a.png", 0.0), ("b.png", 5.0)],
            audio_start_boot_ns=self.VIDEO_T0_BOOT + 2.0e9,  # stream 2 s later
            video_t0_boot_ns=self.VIDEO_T0_BOOT,
        )
        assert rows[0].pre_audio and rows[0].t_audio_s == pytest.approx(-2.0)
        assert not rows[1].pre_audio and rows[1].t_audio_s == pytest.approx(3.0)

    def test_utc_via_boot_anchor(self):
        anchor = _FakeBootAnchor()
        rows = self._rows([("a.png", 1.5)], boot_anchor=anchor)
        frame_boot = self.VIDEO_T0_BOOT + 1.5e9
        assert rows[0].utc_s == pytest.approx(
            frame_boot * 1e-9 + _FakeBootAnchor.OFFSET_S, abs=1e-9
        )

    def test_utc_none_without_anchor(self):
        rows = self._rows([("a.png", 0.0)])
        assert rows[0].utc_s is None

    def test_missing_anchors_and_session_raises(self):
        with pytest.raises(ValueError):
            compute_frame_audio_times(None, [("a.png", 0.0)])

    def test_sub_millisecond_precision_survives(self):
        # A 0.1 ms PTS step must produce exactly a 0.1 ms t_audio_s step.
        rows = self._rows([("a.png", 10.0000), ("b.png", 10.0001)])
        dt = rows[1].t_audio_s - rows[0].t_audio_s
        assert dt == pytest.approx(1e-4, abs=1e-9)


# ---------------------------------------------------------------------------
# CSV shape: Coordinate output-style header, readable by the frame_compare loaders
# ---------------------------------------------------------------------------


class TestExternalCsvShape:
    def _write(self, tmp_path: Path) -> Path:
        entries = [
            ("0000.000.png", (48.123456789, 11.987654321, 512.3456), 0.0),
            ("0012.346.png", (48.123500000, 11.987700000, 512.5000), 12.3456),
            ("0020.000.png", None, 20.0),  # no Signal -> blank coords, kept
        ]
        return write_external_csv(tmp_path / "frames_for_external.csv", entries)

    def test_header(self, tmp_path):
        p = self._write(tmp_path)
        header = p.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(EXTERNAL_CSV_HEADER)
        assert tuple(EXTERNAL_CSV_HEADER[:4]) == (
            "Image", "Latitude", "Longitude", "Altitude",
        )

    def test_loadable_by_external_loader(self, tmp_path):
        from data_pipeline.frame_compare import load_external_frame_coords

        coords = load_external_frame_coords(self._write(tmp_path))
        # Keys are extension-stripped stems; the blank-coord row is skipped.
        assert set(coords) == {"0000.000", "0012.346"}
        lat, lon, h = coords["0012.346"]
        assert lat == pytest.approx(48.1235, abs=1e-6)
        assert lon == pytest.approx(11.9877, abs=1e-6)
        assert h == pytest.approx(512.5, abs=1e-3)

    def test_loadable_by_georef_loader(self, tmp_path):
        from data_pipeline.frame_compare import (
            load_gnss_frame_coords_from_georef,
        )

        coords = load_gnss_frame_coords_from_georef(self._write(tmp_path))
        assert set(coords) == {"0000.000", "0012.346"}
        lat, lon, h = coords["0000.000"]
        assert lat == pytest.approx(48.123456789, abs=1e-9)
        assert lon == pytest.approx(11.987654321, abs=1e-9)
        assert h == pytest.approx(512.3456, abs=1e-4)

    def test_t_audio_column_round_trips(self, tmp_path):
        import csv as _csv

        p = self._write(tmp_path)
        with p.open("r", newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert [r["Image"] for r in rows] == [
            "0000.000.png", "0012.346.png", "0020.000.png",
        ]
        assert float(rows[1]["t_audio_s"]) == pytest.approx(12.3456, abs=1e-9)
        # Blank-coordinate row keeps its time mapping.
        assert rows[2]["Latitude"] == "" and rows[2]["Longitude"] == ""
        assert float(rows[2]["t_audio_s"]) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# End-to-end naming of computed rows (drop negatives, name, stay sorted)
# ---------------------------------------------------------------------------


class TestNamingPipeline:
    def test_drop_then_name_is_sorted_and_nonnegative(self):
        video_t0 = 1_000_000_000_000.0
        audio_start = video_t0 + 0.5e9  # stream starts 0.5 s after media
        pts = [i / 6.0 for i in range(12)]  # 6 fps, 2 s of media
        rows = compute_frame_audio_times(
            None,
            [(f"frame_{i:06d}.png", t) for i, t in enumerate(pts)],
            audio_start_boot_ns=audio_start,
            video_t0_boot_ns=video_t0,
        )
        kept = [r for r in rows if not r.pre_audio]
        dropped = len(rows) - len(kept)
        # PTS < 0.5 s are pre-stream: indices 0,1,2 (0.0, 0.1667, 0.3333).
        assert dropped == 3
        assert all(r.t_audio_s >= 0.0 for r in kept)
        names = [frame_name_for_time(r.t_audio_s) for r in kept]
        assert names == sorted(names)
        assert names[0] == "0000.000"  # 0.5 - 0.5 exactly at the origin
