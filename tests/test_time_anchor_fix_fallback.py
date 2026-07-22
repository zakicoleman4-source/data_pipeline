"""Regression test: Fix-row boot<->UTC fallback for the DAY14 dodge sessions.

Some DAY14 "dodge" Cell captures ship a 0-byte ``recording_*.txt`` AND
``Raw,`` rows whose ``ChipsetElapsedRealtimeNanos`` (last column) is 0/blank,
so ``boot_utc_pairs_from_measurements`` yields 0 pairs and
``fit_time_anchor_from_measurements`` used to raise E-PP-306 unconditionally.

But the same measurements_*.txt carries ``Fix,`` rows that pair
UnixTimeMillis (col 8) with elapsedRealtimeNanos (col 11) --
``data_pipeline.capture_diag.boot_utc_pairs_from_fix_rows`` already knows how
to extract those. This test proves ``fit_time_anchor_from_measurements`` now
falls back to that source instead of raising when the Raw-row bridge is dead.
"""

from pathlib import Path

import pytest

from data_pipeline.errors import PipelineError
from data_pipeline.time_sync import (
    boot_utc_pairs_from_measurements,
    fit_time_anchor_from_measurements,
)


def _raw_row_no_boottime() -> str:
    """A 'Raw,' row with a blank/zero ChipsetElapsedRealtimeNanos (last col)."""
    cols = ["Raw"] + ["0"] * 36
    cols[1] = "0"          # utcTimeMillis (unused by fallback)
    cols[2] = "121012000000"   # TimeNanos
    cols[5] = "-1466696345743000064.0"  # FullBiasNanos
    cols[6] = "0.0"        # BiasNanos
    cols[-1] = ""           # ChipsetElapsedRealtimeNanos -- BLANK (dodge bug)
    return ",".join(cols)


def _fix_row(unix_ms: int, boot_ns: int, provider: str = "fused") -> str:
    """Build a 'Fix,' row matching boot_utc_pairs_from_fix_rows' column layout.

    Header (canonical):
        Fix,Provider,Lat,Lon,Alt,Speed,Acc,Bearing,UnixTimeMillis,...,
        elapsedRealtimeNanos,...                          (col 8 + col 11)
    """
    cols = ["Fix"] + ["0"] * 20
    cols[1] = provider                # Provider
    cols[2] = "37.0"                  # Lat
    cols[3] = "-122.0"                # Lon
    cols[4] = "10.0"                  # Alt
    cols[5] = "0.0"                   # Speed
    cols[6] = "5.0"                   # Acc
    cols[7] = "0.0"                   # Bearing
    cols[8] = str(unix_ms)            # UnixTimeMillis
    cols[11] = str(boot_ns)           # elapsedRealtimeNanos
    return ",".join(cols)


def _write_measurements_dodge(path: Path, *, n_fix: int = 5) -> Path:
    """Synthetic dodge-session measurements: dead Raw bridge, live Fix bridge."""
    lines = [
        "# Header Description:",
        "# Raw,utcTimeMillis,TimeNanos,...,ChipsetElapsedRealtimeNanos",
        "# Fix,Provider,Lat,Lon,Alt,Speed,Acc,Bearing,UnixTimeMillis,...,elapsedRealtimeNanos,...",
    ]
    # Raw rows: boottime column blank/zero -> 0 usable pairs from Raw fallback.
    for _ in range(50):
        lines.append(_raw_row_no_boottime())

    # Fix rows: >=2 valid (boottime_ns, UTC_s) pairs.
    unix_ms0 = 1_782_000_000_000  # ~2026, arbitrary but > 0
    boot0 = 1_000_000_000_000
    step_ms = 1000
    step_ns = 1_000_000_000
    for i in range(n_fix):
        lines.append(_fix_row(unix_ms0 + i * step_ms, boot0 + i * step_ns))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestFixRowFallback:
    def test_raw_bridge_yields_no_pairs(self, tmp_path):
        m = _write_measurements_dodge(tmp_path / "measurements_dodge.txt")
        assert boot_utc_pairs_from_measurements(m) == []

    def test_fit_time_anchor_uses_fix_row_fallback(self, tmp_path):
        """E-PP-306 must NOT fire when Fix rows carry a usable boot<->UTC map."""
        m = _write_measurements_dodge(tmp_path / "measurements_dodge.txt")
        anchor = fit_time_anchor_from_measurements(m)
        assert anchor is not None
        assert anchor.n >= 2
        # boottime -> UTC round-trips close to the Fix-row UnixTimeMillis.
        utc = anchor.boottime_to_utc_s(1_000_000_000_000)
        assert abs(utc - 1_782_000_000.0) < 1.0

    def test_still_raises_when_no_source_usable(self, tmp_path):
        """Sanity: with zero Fix rows too, E-PP-306 must still fire."""
        lines = [
            "# Header Description:",
            "# Raw,utcTimeMillis,TimeNanos,...,ChipsetElapsedRealtimeNanos",
        ]
        for _ in range(10):
            lines.append(_raw_row_no_boottime())
        m = tmp_path / "measurements_unrecoverable.txt"
        m.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(PipelineError) as ei:
            fit_time_anchor_from_measurements(m)
        assert ei.value.code == "E-PP-306"

    def test_with_fallback_reports_fix_row_source(self, tmp_path):
        """The Fix-row bridge must be distinguishable from the Raw bridge.

        The Fix-row bridge is systematically ~0.1-0.15 s EARLY (fix delivery
        latency), so callers need to know WHICH fallback produced the anchor
        to warn the user. Locks the source tag "measurements-fix-fallback".
        """
        from data_pipeline.time_sync import fit_time_anchor_with_fallback

        empty_rec = tmp_path / "recording_dodge.txt"
        empty_rec.write_text("", encoding="utf-8")  # 0 bytes (dodge case)
        m = _write_measurements_dodge(tmp_path / "measurements_dodge.txt")

        anchor, source = fit_time_anchor_with_fallback(empty_rec, m)
        assert source == "measurements-fix-fallback"
        assert anchor.n >= 2

    def test_gps_provider_rows_preferred_over_other_providers(self, tmp_path):
        """network/fused UnixTimeMillis can be SYSTEM-clock time (seconds off).

        Only the reference provider's UnixTimeMillis is Signal-derived. When reference rows
        exist, non-reference rows must not contaminate the boot<->UTC bridge: here
        the network rows are shifted +5 s (a wrong system clock) and would
        drag a mixed fit ~2.5 s off.
        """
        from data_pipeline.capture_diag import boot_utc_pairs_from_fix_rows
        from data_pipeline.time_sync import fit_time_anchor_from_pairs

        unix_ms0 = 1_782_000_000_000
        boot0 = 1_000_000_000_000
        lines = []
        for i in range(10):
            # reference rows: exact 1:1 boot<->UTC line.
            lines.append(_fix_row(
                unix_ms0 + i * 1000, boot0 + i * 1_000_000_000, provider="gps",
            ))
            # network rows: same boottime axis, UTC shifted +5 s (bad clock).
            lines.append(_fix_row(
                unix_ms0 + i * 1000 + 5000, boot0 + i * 1_000_000_000,
                provider="network",
            ))
        m = tmp_path / "measurements_mixed.txt"
        m.write_text("\n".join(lines) + "\n", encoding="utf-8")

        pairs = boot_utc_pairs_from_fix_rows(m)
        assert len(pairs) == 10  # reference rows only
        anchor = fit_time_anchor_from_pairs(iter(pairs))
        utc = anchor.boottime_to_utc_s(boot0)
        assert abs(utc - unix_ms0 / 1e3) < 0.010  # on the reference line, < 10 ms

    def test_all_rows_used_when_no_gps_provider(self, tmp_path):
        """With no reference rows at all, other providers are still better than
        nothing (the historical behaviour is preserved)."""
        from data_pipeline.capture_diag import boot_utc_pairs_from_fix_rows

        m = _write_measurements_dodge(tmp_path / "measurements_dodge.txt")
        pairs = boot_utc_pairs_from_fix_rows(m)  # provider "fused" only
        assert len(pairs) >= 2
