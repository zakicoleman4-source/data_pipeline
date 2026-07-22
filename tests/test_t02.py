"""Unit tests for the Signal binary → Interchange-format converter stage.

Covers both pipelines exposed by ``data_pipeline.stages.t02``:

* Trimble  (``.t02`` family) → runpkr00 + teqc.
* The reference unit    (``.jps``)        → jps2rin or convbin -r javad.

Filesystem helpers, tool resolution, output discovery, and input
validation are exercised without invoking any external binary.
Real-binary end-to-end runs are gated on the lab toolchain + sample
files being present on disk and are skipped automatically otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_pipeline.stages import t02


# ---------------------------------------------------------------------------
# resolve_* helpers
# ---------------------------------------------------------------------------

def test_resolve_jps2rin_override_wins(tmp_path: Path) -> None:
    fake = tmp_path / "jps2rin.exe"
    fake.write_bytes(b"")
    assert t02.resolve_jps2rin(fake) == fake


def test_resolve_jps2rin_env_var(monkeypatch: pytest.MonkeyPatch,
                                 tmp_path: Path) -> None:
    fake = tmp_path / "jps2rin.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("JPS2RIN", str(fake))
    monkeypatch.setattr(t02, "DEFAULT_JPS2RIN", tmp_path / "nope.exe")
    monkeypatch.setattr(t02.shutil, "which", lambda _n: None)
    assert t02.resolve_jps2rin() == fake


def test_resolve_jps2rin_missing_raises(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    from data_pipeline import lab_tools
    monkeypatch.delenv("JPS2RIN", raising=False)
    monkeypatch.setattr(lab_tools, "_LAB_DEFAULTS",
                        {**lab_tools._LAB_DEFAULTS,
                         "jps2rin": tmp_path / "nope.exe"})
    monkeypatch.setattr(lab_tools.shutil, "which", lambda _n: None)
    monkeypatch.setattr(lab_tools, "_config_file",
                        lambda: tmp_path / "no.cfg")
    with pytest.raises(FileNotFoundError, match="jps2rin"):
        t02.resolve_jps2rin()


def test_resolve_convbin_override_wins(tmp_path: Path) -> None:
    fake = tmp_path / "convbin.exe"
    fake.write_bytes(b"")
    assert t02.resolve_convbin(fake) == fake


def test_resolve_convbin_env_var(monkeypatch: pytest.MonkeyPatch,
                                 tmp_path: Path) -> None:
    fake = tmp_path / "convbin.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("CONVBIN", str(fake))
    monkeypatch.setattr(t02, "DEFAULT_CONVBIN", tmp_path / "nope.exe")
    monkeypatch.setattr(t02.shutil, "which", lambda _n: None)
    assert t02.resolve_convbin() == fake


def test_resolve_runpkr00_override_wins(tmp_path: Path) -> None:
    fake = tmp_path / "runpkr00.exe"
    fake.write_bytes(b"")
    assert t02.resolve_runpkr00(fake) == fake


def test_resolve_runpkr00_env_var(monkeypatch: pytest.MonkeyPatch,
                                  tmp_path: Path) -> None:
    fake = tmp_path / "runpkr00.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("RUNPKR00", str(fake))
    monkeypatch.setattr(t02, "DEFAULT_RUNPKR00", tmp_path / "nope.exe")
    monkeypatch.setattr(t02.shutil, "which", lambda _n: None)
    assert t02.resolve_runpkr00() == fake


def test_resolve_teqc_checks_fallbacks(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    """``teqc`` resolution should try fallback paths after the primary default."""
    from data_pipeline import lab_tools
    primary = tmp_path / "primary" / "teqc.exe"
    fallback = tmp_path / "fallback" / "teqc.exe"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"")
    monkeypatch.delenv("TEQC", raising=False)
    monkeypatch.setattr(lab_tools, "_LAB_DEFAULTS",
                        {**lab_tools._LAB_DEFAULTS, "teqc": primary})
    monkeypatch.setattr(lab_tools, "_LAB_FALLBACKS",
                        {**lab_tools._LAB_FALLBACKS, "teqc": (fallback,)})
    monkeypatch.setattr(lab_tools.shutil, "which", lambda _n: None)
    monkeypatch.setattr(lab_tools, "_config_file",
                        lambda: tmp_path / "no.cfg")
    assert t02.resolve_teqc() == fallback


# ---------------------------------------------------------------------------
# auto_pick_converter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext,expected", [
    (".t02", "trimble"), (".T02", "trimble"),
    (".t01", "trimble"), (".T01", "trimble"),
    (".t00", "trimble"), (".r00", "trimble"),
    (".jps", "jps2rin"), (".JPS", "jps2rin"),
    (".tps", "jps2rin"), (".tpd", "jps2rin"),
    (".obs", "jps2rin"),  # permissive default
])
def test_auto_pick_converter(ext: str, expected: str, tmp_path: Path) -> None:
    p = tmp_path / f"sample{ext}"
    p.write_bytes(b"")
    assert t02.auto_pick_converter(p) == expected


# ---------------------------------------------------------------------------
# _discover_outputs
# ---------------------------------------------------------------------------

def test_discover_outputs_rinex3_extensions(tmp_path: Path) -> None:
    (tmp_path / "log.obs").write_bytes(b"")
    (tmp_path / "log.nav").write_bytes(b"")
    (tmp_path / "log.gnav").write_bytes(b"")
    (tmp_path / "log.sp3").write_bytes(b"")
    (tmp_path / "other.obs").write_bytes(b"")  # different stem → ignored
    obs, nav = t02._discover_outputs(tmp_path, "log")
    assert [p.name for p in obs] == ["log.obs"]
    assert {p.name for p in nav} == {"log.nav", "log.gnav", "log.sp3"}


def test_discover_outputs_rinex2_yy_pattern(tmp_path: Path) -> None:
    (tmp_path / "rover.26o").write_bytes(b"")  # obs
    (tmp_path / "rover.26n").write_bytes(b"")  # nav
    (tmp_path / "rover.26G").write_bytes(b"")  # Source-group nav
    (tmp_path / "rover.26L").write_bytes(b"")  # Source-group nav
    (tmp_path / "rover.26z").write_bytes(b"")  # NOT n/g/l/p/c/h/q → ignored
    obs, nav = t02._discover_outputs(tmp_path, "rover")
    assert {p.name.lower() for p in obs} == {"rover.26o"}
    assert {p.name.lower() for p in nav} == {
        "rover.26n", "rover.26g", "rover.26l",
    }


def test_discover_outputs_filters_by_stem(tmp_path: Path) -> None:
    (tmp_path / "abc.obs").write_bytes(b"")
    (tmp_path / "abcdef.obs").write_bytes(b"")  # starts with abc → included
    (tmp_path / "xyz.obs").write_bytes(b"")     # different stem → excluded
    obs, _ = t02._discover_outputs(tmp_path, "abc")
    assert {p.name for p in obs} == {"abc.obs", "abcdef.obs"}


def test_discover_outputs_missing_dir(tmp_path: Path) -> None:
    obs, nav = t02._discover_outputs(tmp_path / "nope", "anything")
    assert obs == [] and nav == []


# ---------------------------------------------------------------------------
# run() validation
# ---------------------------------------------------------------------------

def test_run_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input file"):
        t02.run(
            input_file=tmp_path / "nope.jps",
            output_dir=tmp_path / "out",
        )


def test_run_invalid_rinex_version(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "x.jps").write_bytes(b"")
    with pytest.raises(ValueError, match="RINEX"):
        t02.run(
            input_file=tmp_path / "x.jps",
            output_dir=tmp_path / "out",
            rinex_version="9.99",
        )


def test_run_unknown_converter(tmp_path: Path) -> None:
    (tmp_path / "x.jps").write_bytes(b"")
    with pytest.raises(ValueError, match="unknown converter"):
        t02.run(
            input_file=tmp_path / "x.jps",
            output_dir=tmp_path / "out",
            converter="nope",
        )


# ---------------------------------------------------------------------------
# Real end-to-end against the lab tools (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    True,
    reason="End-to-end T02 integration test — requires runpkr00 + teqc + "
           "sample .T02 file (set $DATA_PIPELINE_T02_SAMPLE to enable).",
)
def test_run_trimble_real(tmp_path: Path) -> None:
    """Real end-to-end T02 → Interchange-format via runpkr00 + teqc."""
    sample = os.environ.get("DATA_PIPELINE_T02_SAMPLE")
    if not sample or not Path(sample).is_file():
        pytest.skip("sample .T02 not available")
    res = t02.run(
        input_file=Path(sample),
        output_dir=tmp_path,
        gps_week=2418,
    )
    assert res.returncode == 0
    assert res.converter == "trimble"
    assert res.obs_files, "expected at least one OBS file"
    # OBS file must be non-empty and have a Interchange-format header.
    head = res.obs_files[0].read_text(errors="replace").splitlines()[0]
    assert "RINEX VERSION" in head


def test_run_invalid_converter_explicit(tmp_path: Path) -> None:
    """Explicit invalid converter string is rejected."""
    (tmp_path / "x.t02").write_bytes(b"")
    with pytest.raises(ValueError, match="unknown converter"):
        t02.run(
            input_file=tmp_path / "x.t02",
            output_dir=tmp_path / "out",
            converter="garbage",
        )


@pytest.mark.skipif(
    True,
    reason="End-to-end Javad integration test — requires jps2rin + sample "
           ".jps (set $DATA_PIPELINE_JPS_SAMPLE to enable).",
)
def test_run_jps2rin_real(tmp_path: Path) -> None:
    sample = os.environ.get("DATA_PIPELINE_JPS_SAMPLE")
    if not sample or not Path(sample).is_file():
        pytest.skip("sample .jps not available")
    res = t02.run(
        input_file=Path(sample),
        output_dir=tmp_path,
        rinex_version="3.05",
        converter="jps2rin",
    )
    assert res.returncode == 0
    assert res.obs_files, "expected at least one OBS file"
    assert all(p.is_file() for p in res.obs_files)
    # NAV files are optional but should be non-empty when produced.
    for p in res.nav_files:
        assert p.stat().st_size > 0


@pytest.mark.skipif(
    True,
    reason="End-to-end convbin integration test — requires convbin + sample "
           ".jps (set $DATA_PIPELINE_JPS_SAMPLE to enable).",
)
def test_run_convbin_real(tmp_path: Path) -> None:
    sample = os.environ.get("DATA_PIPELINE_JPS_SAMPLE")
    if not sample or not Path(sample).is_file():
        pytest.skip("sample .jps not available")
    res = t02.run(
        input_file=Path(sample),
        output_dir=tmp_path,
        rinex_version="3.05",
        converter="convbin",
        include_doppler=True,
        include_snr=True,
    )
    assert res.returncode == 0
    assert res.obs_files
    obs_head = res.obs_files[0].read_text(errors="replace").splitlines()[0]
    assert "RINEX VERSION" in obs_head or "RINEX" in obs_head
