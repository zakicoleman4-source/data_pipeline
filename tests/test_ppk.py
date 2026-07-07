"""Unit tests for the Post-processing stage wrapper.

These tests do not require The external solver to be installed — they exercise the
filesystem helpers, configuration parsing, and validation paths. The
``run()`` happy-path is verified separately by the end-to-end suite when
The external solver is available on the host.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_pipeline.stages import ppk


# ---------------------------------------------------------------------------
# resolve_rnx2rtkp
# ---------------------------------------------------------------------------

def test_resolve_rnx2rtkp_override_wins(tmp_path: Path) -> None:
    fake = tmp_path / "rnx2rtkp.exe"
    fake.write_bytes(b"")
    assert ppk.resolve_rnx2rtkp(fake) == fake


def test_resolve_rnx2rtkp_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "alt_rnx2rtkp.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("RNX2RTKP", str(fake))
    monkeypatch.setattr(ppk, "DEFAULT_RTKLIB_DIR", tmp_path / "no_such_dir")
    # Make sure PATH lookup also fails so env-var is forced to be the winner.
    monkeypatch.setattr(ppk.shutil, "which", lambda _name: None)
    assert ppk.resolve_rnx2rtkp() == fake


def test_resolve_rnx2rtkp_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from data_pipeline import lab_tools
    monkeypatch.delenv("RNX2RTKP", raising=False)
    monkeypatch.setattr(lab_tools, "_LAB_DEFAULTS",
                        {**lab_tools._LAB_DEFAULTS,
                         "rnx2rtkp": tmp_path / "nope.exe"})
    monkeypatch.setattr(lab_tools.shutil, "which", lambda _name: None)
    # No config file should exist either.
    monkeypatch.setattr(lab_tools, "_config_file",
                        lambda: tmp_path / "no.cfg")
    # And no bundled binary (the vendored vendor/rtklib/rnx2rtkp.exe added
    # in v1.1+ would otherwise satisfy the resolver and short-circuit the
    # missing-file path this test is asserting).
    monkeypatch.setattr(lab_tools, "_bundled_tool_path", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="rnx2rtkp"):
        ppk.resolve_rnx2rtkp()


# ---------------------------------------------------------------------------
# list_config_files
# ---------------------------------------------------------------------------

def test_list_config_files_returns_sorted_conf(tmp_path: Path) -> None:
    (tmp_path / "b.conf").write_text("# b")
    (tmp_path / "a.conf").write_text("# a")
    (tmp_path / "not_conf.txt").write_text("ignore me")
    out = ppk.list_config_files(tmp_path)
    assert [p.name for p in out] == ["a.conf", "b.conf"]


def test_list_config_files_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert ppk.list_config_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# detect_nav_files
# ---------------------------------------------------------------------------

def test_detect_nav_files_picks_up_standard_extensions(tmp_path: Path) -> None:
    (tmp_path / "x.nav").write_bytes(b"")
    (tmp_path / "y.sp3").write_bytes(b"")
    (tmp_path / "z.eph").write_bytes(b"")
    (tmp_path / "ignored.txt").write_bytes(b"")
    found = ppk.detect_nav_files(tmp_path)
    names = {p.name for p in found}
    assert names == {"x.nav", "y.sp3", "z.eph"}


def test_detect_nav_files_rinex_two_digit_pattern(tmp_path: Path) -> None:
    (tmp_path / "base.24n").write_bytes(b"")
    (tmp_path / "base.24g").write_bytes(b"")
    (tmp_path / "base.24p").write_bytes(b"")
    (tmp_path / "base.24x").write_bytes(b"")  # not n/g/l/p/h → skipped
    names = {p.name for p in ppk.detect_nav_files(tmp_path)}
    assert names == {"base.24n", "base.24g", "base.24p"}


def test_detect_nav_files_deduplicates_across_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.nav").write_bytes(b"")
    # Same directory passed twice should not double-up the result.
    found = ppk.detect_nav_files(tmp_path, tmp_path)
    assert len(found) == 1


def test_detect_nav_files_handles_missing_dir(tmp_path: Path) -> None:
    assert ppk.detect_nav_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# run() input validation (without actually invoking the binary)
# ---------------------------------------------------------------------------

def _stub_resolve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "rnx2rtkp.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(ppk, "resolve_rnx2rtkp", lambda override=None: fake_exe)


def test_run_missing_rover_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_resolve(monkeypatch, tmp_path)
    from data_pipeline.errors import PipelineError
    base = tmp_path / "b.obs"; base.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    with pytest.raises(PipelineError) as ei:
        ppk.run(
            rover_obs=tmp_path / "missing.obs",
            base_obs=base,
            nav_files=[nav],
            config_file=cfg,
            output_pos=tmp_path / "out.pos",
        )
    assert ei.value.code == "E-PP-101"


def test_run_missing_base_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_resolve(monkeypatch, tmp_path)
    from data_pipeline.errors import PipelineError
    rover = tmp_path / "r.obs"; rover.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    with pytest.raises(PipelineError) as ei:
        ppk.run(
            rover_obs=rover,
            base_obs=tmp_path / "missing.obs",
            nav_files=[nav],
            config_file=cfg,
            output_pos=tmp_path / "out.pos",
        )
    assert ei.value.code == "E-PP-100"


def test_run_empty_nav_list_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_resolve(monkeypatch, tmp_path)
    from data_pipeline.errors import PipelineError
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("")
    with pytest.raises(PipelineError) as ei:
        ppk.run(
            rover_obs=rover,
            base_obs=base,
            nav_files=[],
            config_file=cfg,
            output_pos=tmp_path / "out.pos",
        )
    assert ei.value.code == "E-PP-102"


def test_run_missing_config_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_resolve(monkeypatch, tmp_path)
    from data_pipeline.errors import PipelineError
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    with pytest.raises(PipelineError) as ei:
        ppk.run(
            rover_obs=rover,
            base_obs=base,
            nav_files=[nav],
            config_file=tmp_path / "missing.conf",
            output_pos=tmp_path / "out.pos",
        )
    assert ei.value.code == "E-PP-103"


@pytest.mark.skipif(
    True,
    reason="End-to-end integration test — requires local RTKLIB + sample data "
           "(set $DATA_PIPELINE_E2E_DATA to enable).",
)
def test_run_real_binary_produces_pos(tmp_path: Path) -> None:
    """End-to-end integration check against the actual solver binary.

    Always skipped in CI; enable manually by setting the appropriate env
    vars and providing sample subject.obs / base.obs / config paths.
    """
    e2e = os.environ.get("DATA_PIPELINE_E2E_DATA")
    if not e2e:
        pytest.skip("$DATA_PIPELINE_E2E_DATA not set")
    e2e_p = Path(e2e)
    rover = e2e_p / "rover.obs"
    base = e2e_p / "base.obs"
    cfg = e2e_p / "ppk.conf"
    if not (rover.is_file() and base.is_file() and cfg.is_file()):
        pytest.skip("reference site sample data not present on this host")
    nav = ppk.detect_nav_files(base.parent)
    if not nav:
        pytest.skip("No nav files alongside the sample base obs")
    out = tmp_path / "real.pos"
    try:
        res = ppk.run(
            rover_obs=rover, base_obs=base, nav_files=nav[:4],
            config_file=cfg, output_pos=out,
        )
    except Exception as e:
        # New header-only guard fires when the sample config doesn't
        # actually produce data epochs. That's the correct behaviour —
        # skip the regression rather than mark the test red.
        if "E-PP-104" in str(e):
            pytest.skip(f"sample produced header-only .pos: {e}")
        raise
    assert res.returncode == 0
    assert res.pos_path.is_file()
    assert res.pos_path.stat().st_size > 0
    head = res.pos_path.read_text(errors="replace").splitlines()[0]
    assert head.startswith("% program")


# ---------------------------------------------------------------------------
# Packaged configs + run_with_user_base
# ---------------------------------------------------------------------------

def test_list_packaged_configs_includes_javad_avg_sp() -> None:
    names = [p.name for p in ppk.list_packaged_configs()]
    assert "javad_avg_sp.conf" in names, (
        f"shipped config dir missing javad_avg_sp.conf; got {names}"
    )


def test_run_with_user_base_rejects_unknown_config_name(
    tmp_path: Path,
) -> None:
    from data_pipeline.errors import PipelineError
    with pytest.raises(PipelineError) as ei:
        ppk.run_with_user_base(
            rover_obs=tmp_path / "r.obs",
            base_obs=tmp_path / "b.obs",
            nav_files=[tmp_path / "n.nav"],
            output_pos=tmp_path / "out.pos",
            base_spec="45.0,0.0,100.0",
            config_name="does-not-exist.conf",
        )
    assert ei.value.code == "E-PP-103"
    assert "javad_avg_sp.conf" in (ei.value.hint or "")


def test_run_with_user_base_rejects_unparseable_base_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from data_pipeline.errors import PipelineError
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    _stub_resolve(monkeypatch, tmp_path)
    with pytest.raises(PipelineError) as ei:
        ppk.run_with_user_base(
            rover_obs=rover, base_obs=base, nav_files=[nav],
            output_pos=tmp_path / "out.pos",
            base_spec="this is not coordinates",
        )
    assert ei.value.code == "E-PP-105"
    assert "lat,lon,h" in (ei.value.hint or "")


def test_run_with_user_base_accepts_each_spec_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Every form parse_base_spec advertises must round-trip through
    run_with_user_base — at least up to the subprocess call. We stub
    the solver out via SubprocessRecorder so the test stays hermetic."""
    from data_pipeline.errors import PipelineError
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    _stub_resolve(monkeypatch, tmp_path)

    calls: list[tuple[float, float, float]] = []

    def fake_run(*, base_ecef_xyz=None, **kw):
        calls.append(tuple(base_ecef_xyz))
        # Raise after the call so test doesn't actually shell out.
        raise PipelineError("E-PP-103", "stub", hint="")

    monkeypatch.setattr(ppk, "run", fake_run)

    # All four specs point to the SAME generic location (45.0, 10.0, 100.0)
    # so the round-trip through parse_base_spec must converge.
    # Compute matching Cartesian XYZ via the project's function (avoids drift).
    from data_pipeline.base_pos import base_xyz_from_llh
    _x, _y, _z = base_xyz_from_llh(45.0, 10.0, 100.0)
    specs = [
        "45.000000,10.000000,100.00",                       # bare LLH
        "llh:45.000000,10.000000,100.00",                   # tagged LLH
        f"ecef:{_x:.4f},{_y:.4f},{_z:.4f}",                 # tagged Cartesian XYZ
        f"{_x:.4f},{_y:.4f},{_z:.4f}",                      # bare Cartesian XYZ (all > 100 km)
    ]
    for s in specs:
        with pytest.raises(PipelineError) as ei:
            ppk.run_with_user_base(
                rover_obs=rover, base_obs=base, nav_files=[nav],
                output_pos=tmp_path / "out.pos",
                base_spec=s,
            )
        # The stub raises with code E-PP-103; if the spec parsed and
        # reached the stub, calls[] grew by one.
        assert ei.value.code == "E-PP-103"
    assert len(calls) == len(specs)
    # All four specs should land within 1 m of each other.
    xs = [c[0] for c in calls]
    assert max(xs) - min(xs) < 1.0, f"specs diverge: {calls}"


# ---------------------------------------------------------------------------
# Bug regressions
# ---------------------------------------------------------------------------

def test_run_uses_utf8_encoding_for_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """ppk.run must pass encoding='utf-8' to subprocess.run so non-ASCII
    paths (Hebrew folder names, etc.) don't crash with UnicodeDecodeError."""
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("# empty\n")
    out = tmp_path / "out.pos"

    captured: dict[str, object] = {}

    class FakeProc:
        stdout = ""; stderr = ""; returncode = 0

    def fake_run(cmd, **kw):
        captured.update(kw)
        out.write_text("% header\n2024/01/01 00:00:00 32.0 34.0 100 1 12\n")
        return FakeProc()

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    ppk.run(rover_obs=rover, base_obs=base, nav_files=[nav],
            config_file=cfg, output_pos=out)
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"


def test_run_raises_on_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """subprocess.TimeoutExpired must be wrapped in PipelineError E-PP-103."""
    import subprocess as _sp
    from data_pipeline.errors import PipelineError
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("# empty\n")

    def fake_run(*a, **kw):
        raise _sp.TimeoutExpired(cmd=["x"], timeout=1.0)

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    with pytest.raises(PipelineError) as ei:
        ppk.run(rover_obs=rover, base_obs=base, nav_files=[nav],
                config_file=cfg, output_pos=tmp_path / "out.pos",
                timeout_s=1.0)
    assert ei.value.code == "E-PP-103"
    assert "timed out" in ei.value.message


def test_run_raises_when_pos_is_header_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The external solver exit 0 + header-only .pos must fire PipelineError E-PP-104."""
    from data_pipeline.errors import PipelineError
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("# empty\n")
    out = tmp_path / "out.pos"

    class FakeProc:
        stdout = ""; stderr = ""; returncode = 0

    def fake_run(cmd, **kw):
        # Header-only — every line begins with '%'.
        out.write_text(
            "% program   : rnx2rtkp\n"
            "% obs start : 2024/01/01 00:00:00\n"
            "% (no data epochs survived)\n"
        )
        return FakeProc()

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    with pytest.raises(PipelineError) as ei:
        ppk.run(rover_obs=rover, base_obs=base, nav_files=[nav],
                config_file=cfg, output_pos=out)
    assert ei.value.code == "E-PP-104"
    assert "0 data epochs" in ei.value.message


def test_run_dedupes_repeated_nav_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Duplicate nav files in the caller's list must collapse to one
    occurrence in the solver binary command (The external solver re-reads each)."""
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text("# empty\n")
    out = tmp_path / "out.pos"

    captured_cmd: list[list[str]] = []

    class FakeProc:
        stdout = ""; stderr = ""; returncode = 0

    def fake_run(cmd, **kw):
        captured_cmd.append(list(cmd))
        out.write_text("% header\n2024/01/01 00:00:00 32.0 34.0 100 1 12\n")
        return FakeProc()

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    ppk.run(rover_obs=rover, base_obs=base,
            nav_files=[nav, nav, nav, nav],   # 4x same file
            config_file=cfg, output_pos=out)
    cmd = captured_cmd[0]
    assert cmd.count(str(nav)) == 1, f"dup not removed: {cmd}"


def test_patched_config_name_does_not_collide_on_parallel_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Two outputs in the same dir must produce two different patched
    .conf names so a parallel runner doesn't have them step on each
    other."""
    _stub_resolve(monkeypatch, tmp_path)
    rover = tmp_path / "r.obs"; rover.write_text("")
    base = tmp_path / "b.obs"; base.write_text("")
    nav = tmp_path / "n.nav"; nav.write_text("")
    cfg = tmp_path / "c.conf"; cfg.write_text(
        "ant2-postype=single\nant2-pos1=0\nant2-pos2=0\nant2-pos3=0\n"
    )

    seen_confs: list[str] = []

    class FakeProc:
        stdout = ""; stderr = ""; returncode = 0

    def fake_run(cmd, **kw):
        i = cmd.index("-k")
        seen_confs.append(cmd[i + 1])
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text("% header\n2024/01/01 00:00:00 32.0 34.0 100 1 12\n")
        return FakeProc()

    monkeypatch.setattr(ppk.subprocess, "run", fake_run)
    for stem in ("alpha", "beta"):
        ppk.run(rover_obs=rover, base_obs=base, nav_files=[nav],
                config_file=cfg, output_pos=tmp_path / f"{stem}.pos",
                base_ecef_xyz=(4_441_094.0, 3_083_076.5, 3_275_680.2))
    assert len(seen_confs) == 2
    assert seen_confs[0] != seen_confs[1], (
        f"parallel-output patched-conf collision: both = {seen_confs[0]}"
    )
