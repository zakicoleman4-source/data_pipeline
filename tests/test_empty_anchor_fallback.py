"""Regression tests for the empty-session-anchor fallback (DAY14).

7 of 17 DAY14 sessions shipped a 0-byte ``recording_*.txt`` (the boot<->UTC time
bridge the platform app normally writes was never flushed). The fallback recovers
the anchor from ``measurements_*.txt``: each ``Raw`` row carries the Signal clock
(TimeNanos/FullBiasNanos/BiasNanos -> Signal time -> UTC minus leap) AND the
ChipsetElapsedRealtimeNanos boottime (last column). When the boottime column is
NOT populated (e.g. some Cell captures), the session is correctly reported as
anchor-unrecoverable instead of guessing.
"""

import math
from pathlib import Path

import pytest

from data_pipeline.errors import PipelineError
from data_pipeline.time_sync import (
    GPS_UTC_LEAP_SECONDS_2026,
    boot_utc_pairs_from_measurements,
    fit_time_anchor_from_measurements,
    fit_time_anchor_with_fallback,
)

_GPS_EPOCH_UNIX_S = 315964800.0
# A fixed FullBiasNanos chosen so derived UTC lands in 2026 (post-leap table).
_FULL_BIAS = -1466696345743000064.0


def _raw_row(time_nanos: int, chipset_boot_ns: int) -> str:
    """Build a measurements_*.txt 'Raw' row with the columns the fallback reads.

    Only columns 2 (TimeNanos), 5 (FullBiasNanos), 6 (BiasNanos) and the LAST
    (ChipsetElapsedRealtimeNanos) matter to the parser; the rest are padding to
    match the canonical 37-column The logger app layout.
    """
    cols = ["Raw"] + ["0"] * 36
    cols[1] = "0"                       # utcTimeMillis (unused by fallback)
    cols[2] = str(time_nanos)           # TimeNanos
    cols[5] = repr(_FULL_BIAS)          # FullBiasNanos
    cols[6] = "0.0"                     # BiasNanos
    cols[-1] = str(chipset_boot_ns)     # ChipsetElapsedRealtimeNanos (boottime)
    return ",".join(cols)


def _expected_utc(time_nanos: float) -> float:
    gnss_ns = time_nanos - _FULL_BIAS
    utc_unadj = _GPS_EPOCH_UNIX_S + gnss_ns / 1e9
    return utc_unadj - GPS_UTC_LEAP_SECONDS_2026


def _write_measurements(path: Path, *, populated: bool, n: int = 200) -> Path:
    """Write a synthetic measurements file. ``populated`` toggles the boottime."""
    lines = [
        "# Header Description:",
        "# Raw,utcTimeMillis,TimeNanos,...,ChipsetElapsedRealtimeNanos",
    ]
    boot0 = 1_000_000_000_000          # 1000 s since boot
    step_ns = 100_000_000              # 10 Hz
    for i in range(n):
        time_nanos = 121_012_000_000 + i * step_ns
        boot = (boot0 + i * step_ns) if populated else 0
        lines.append(_raw_row(time_nanos, boot))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestMeasurementsFallback:
    def test_pairs_extracted_when_chipset_populated(self, tmp_path):
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=True)
        pairs = boot_utc_pairs_from_measurements(m)
        assert len(pairs) == 200
        boot0, utc0 = pairs[0]
        assert boot0 == 1_000_000_000_000
        assert abs(utc0 - _expected_utc(121_012_000_000)) < 1e-6

    def test_anchor_fit_recovers_utc(self, tmp_path):
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=True)
        anchor = fit_time_anchor_from_measurements(m)
        assert anchor.n == 200
        # boottime -> UTC round-trips to the Signal-derived UTC within sub-ms.
        utc = anchor.boottime_to_utc_s(1_000_000_000_000)
        assert abs(utc - _expected_utc(121_012_000_000)) < 1e-3

    def test_unrecoverable_when_chipset_zero(self, tmp_path):
        """All-zero ChipsetElapsedRealtimeNanos -> reported, not guessed."""
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=False)
        assert boot_utc_pairs_from_measurements(m) == []
        with pytest.raises(PipelineError) as ei:
            fit_time_anchor_from_measurements(m)
        assert ei.value.code == "E-PP-306"


class TestFitWithFallback:
    def test_empty_recording_uses_measurements(self, tmp_path):
        empty_rec = tmp_path / "recording_x.txt"
        empty_rec.write_text("", encoding="utf-8")           # 0 bytes
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=True)
        anchor, source = fit_time_anchor_with_fallback(empty_rec, m)
        assert source == "measurements-fallback"
        assert anchor.n == 200

    def test_present_recording_takes_precedence(self, tmp_path):
        rec = tmp_path / "recording_x.txt"
        rec.write_text(
            "\n".join(
                f"{1_000_000_000_000 + i*100_000_000},"
                f"2026-06-28T15:55:{50+i*0:02d}.000000000Z,200000000"
                for i in range(5)
            ),
            encoding="utf-8",
        )
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=True)
        _anchor, source = fit_time_anchor_with_fallback(rec, m)
        assert source == "recording.txt"

    def test_empty_recording_and_unrecoverable_measurements_raises(self, tmp_path):
        empty_rec = tmp_path / "recording_x.txt"
        empty_rec.write_text("", encoding="utf-8")
        m = _write_measurements(tmp_path / "measurements_x.txt", populated=False)
        with pytest.raises(PipelineError) as ei:
            fit_time_anchor_with_fallback(empty_rec, m)
        assert ei.value.code == "E-PP-306"
