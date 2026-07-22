"""Tests for data_pipeline.frame_time_table (per-frame every-clock table)."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipeline.frame_time_table import (
    FRAME_TIME_TABLE_HEADER,
    build_frame_time_table,
)

# --- Synthetic timeline ------------------------------------------------------

UTC0 = 1_760_000_000.0          # 2025-10-09, leap = 18 s
BOOT0_NS = 5_000_000_000_000.0  # boottime of the boot->UTC anchor origin
VIDEO_T0_NS = 5_000_000_000_000.0   # frame PTS 0 boottime (== BOOT0 for simplicity)
AUDIO_START_NS = VIDEO_T0_NS + 1.0e9  # audio sample 0 is 1 s after frame 0

PTS = [0.0, 0.5, 1.0, 1.5, 2.0]


class FakeBootAnchor:
    """Minimal TimeAnchor stand-in: exact linear boot->UTC map."""

    def boottime_to_utc_s(self, x_ns: float) -> float:
        return UTC0 + (x_ns - BOOT0_NS) / 1e9

    def video_pts_to_utc_s(self, pts_s: float) -> float:
        # legacy direct mapping (used only when video_t0_boot_ns is None)
        return UTC0 + pts_s


def make_anchors(*, video_t0=VIDEO_T0_NS, audio_start=AUDIO_START_NS):
    return SimpleNamespace(
        boot_anchor=FakeBootAnchor(),
        video_t0_boot_ns=video_t0,
        audio_start_boot_ns=audio_start,
    )


@pytest.fixture()
def frame_times_csv(tmp_path: Path) -> Path:
    p = tmp_path / "extracted_frame_times.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image", "t_video_s"])
        for i, pts in enumerate(PTS):
            w.writerow([f"frame_{i:06d}.png", f"{pts:.6f}"])
    return p


def read_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


# --- Core behaviour -----------------------------------------------------------


def test_header_exact(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    header, rows = read_rows(out)
    assert header == list(FRAME_TIME_TABLE_HEADER)
    assert header == [
        "Image", "video_pts_s", "boot_ns", "utc_s", "utc_iso",
        "gpst_s", "t_audio_s",
    ]
    assert len(rows) == len(PTS)


def test_boot_utc_gpst_columns(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    _, rows = read_rows(out)
    prev_utc = None
    for row, pts in zip(rows, PTS):
        image, pts_cell, boot_cell, utc_cell, iso_cell, gpst_cell, _ = row
        assert image.startswith("frame_")
        assert float(pts_cell) == pytest.approx(pts, abs=1e-9)
        # boot_ns == video_t0 + pts*1e9, integer formatted
        assert "." not in boot_cell
        assert float(boot_cell) == pytest.approx(VIDEO_T0_NS + pts * 1e9, abs=1.0)
        # utc via the linear anchor
        utc = float(utc_cell)
        assert utc == pytest.approx(UTC0 + pts, abs=2e-3)  # 3-decimal cell
        # gpst == utc + 18 (leap for 2025/2026 epochs)
        assert float(gpst_cell) == pytest.approx(utc + 18.0, abs=1e-9)
        # monotonic UTC with pts
        if prev_utc is not None:
            assert utc > prev_utc
        prev_utc = utc
        # ISO cell matches the same instant at ms precision
        parsed = dt.datetime.strptime(
            iso_cell, "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=dt.timezone.utc)
        assert parsed.timestamp() == pytest.approx(utc, abs=1e-3)


def test_t_audio_zero_at_audio_start_and_blank_before(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    _, rows = read_rows(out)
    audio_cells = [r[6] for r in rows]
    # pts 0.0 and 0.5 are pre-audio (boot_ns < audio_start) -> blank
    assert audio_cells[0] == ""
    assert audio_cells[1] == ""
    # pts 1.0: boot_ns == audio_start_boot_ns -> t_audio == 0
    assert float(audio_cells[2]) == pytest.approx(0.0, abs=1e-9)
    # increases by dt after
    assert float(audio_cells[3]) == pytest.approx(0.5, abs=1e-9)
    assert float(audio_cells[4]) == pytest.approx(1.0, abs=1e-9)


def test_no_audio_anchor_blank_column(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(audio_start=None),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    _, rows = read_rows(out)
    assert all(r[6] == "" for r in rows)
    # boot/utc/gpst untouched
    assert all(r[2] != "" and r[3] != "" and r[5] != "" for r in rows)


def test_legacy_no_video_t0_falls_back_to_direct_pts_map(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(video_t0=None, audio_start=None),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    _, rows = read_rows(out)
    for row, pts in zip(rows, PTS):
        assert row[2] == ""   # boot_ns blank
        assert row[6] == ""   # t_audio_s blank
        assert float(row[3]) == pytest.approx(UTC0 + pts, abs=2e-3)
        assert float(row[5]) == pytest.approx(UTC0 + pts + 18.0, abs=2e-3)


def test_html_written_and_self_contained(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(),
        out_csv=tmp_path / "table.csv", write_html=True,
    )
    html_path = out.with_suffix(".html")
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")
    # header + every image name present; no external resources
    for col in FRAME_TIME_TABLE_HEADER:
        assert col in text
    assert "frame_000000.png" in text
    assert "http://" not in text and "https://" not in text
    assert "<script>" in text  # filter/sort JS inlined


def test_no_html_flag(frame_times_csv, tmp_path):
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(),
        out_csv=tmp_path / "table.csv", write_html=False,
    )
    assert not out.with_suffix(".html").exists()


def test_empty_frame_times_raises(tmp_path):
    p = tmp_path / "extracted_frame_times.csv"
    p.write_text("Image,t_video_s\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No frame rows"):
        build_frame_time_table(
            p, anchors=make_anchors(), out_csv=tmp_path / "t.csv",
            write_html=False,
        )


def test_missing_boot_anchor_raises(frame_times_csv, tmp_path):
    anchors = SimpleNamespace(
        boot_anchor=None, video_t0_boot_ns=VIDEO_T0_NS,
        audio_start_boot_ns=None,
    )
    with pytest.raises(ValueError, match="anchor"):
        build_frame_time_table(
            frame_times_csv, anchors=anchors,
            out_csv=tmp_path / "t.csv", write_html=False,
        )


def test_needs_session_or_anchors(frame_times_csv, tmp_path):
    with pytest.raises(ValueError, match="session_dir or anchors"):
        build_frame_time_table(
            frame_times_csv, out_csv=tmp_path / "t.csv", write_html=False,
        )


def test_returns_csv_path(frame_times_csv, tmp_path):
    target = tmp_path / "sub" / "table.csv"  # parent auto-created
    out = build_frame_time_table(
        frame_times_csv, anchors=make_anchors(), out_csv=target,
        write_html=False,
    )
    assert out == target
    assert target.is_file()


# --- CLI ----------------------------------------------------------------------


def test_cli_with_explicit_anchors_not_supported_but_frame_times_flow(
    frame_times_csv, tmp_path, monkeypatch, capsys
):
    """CLI end-to-end with a monkeypatched anchor resolver (no real session)."""
    import scripts.export_frame_times as cli
    import data_pipeline.frame_time_table as ftt

    def fake_resolve(session_dir, log):
        return FakeBootAnchor(), VIDEO_T0_NS, AUDIO_START_NS

    monkeypatch.setattr(ftt, "_resolve_anchors", fake_resolve)
    out_csv = tmp_path / "cli_table.csv"
    rc = cli.main([
        "--session", str(tmp_path),
        "--frame-times", str(frame_times_csv),
        "--out", str(out_csv),
        "--no-html",
    ])
    assert rc == 0
    assert out_csv.is_file()
    captured = capsys.readouterr().out
    assert "Preview:" in captured
    assert "Image,video_pts_s,boot_ns" in captured


def test_cli_missing_frame_times_returns_2(tmp_path, capsys):
    import scripts.export_frame_times as cli

    rc = cli.main(["--session", str(tmp_path)])
    assert rc == 2
    assert "no extracted_frame_times.csv" in capsys.readouterr().out
