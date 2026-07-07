"""Tests for the central error-code registry + report writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.errors import (
    ERROR_CODES,
    PipelineError,
    report_user_message,
    save_error_report,
)


def test_error_codes_unique_and_well_formed():
    """Every code must follow E-PP-NNN and resolve to a description."""
    for code, desc in ERROR_CODES.items():
        assert code.startswith("E-PP-"), f"bad prefix: {code}"
        assert len(code) == 8, f"bad length: {code}"
        suffix = code[5:]
        assert suffix.isdigit(), f"suffix not numeric: {code}"
        assert desc.strip(), f"empty description: {code}"


def test_pipeline_error_format_includes_code_and_hint():
    err = PipelineError(
        "E-PP-100",
        "Base .obs not found at /no/such/path.obs",
        hint="set --base in CLI or pick a Base file in GUI Inputs",
        context={"path": "/no/such/path.obs"},
    )
    s = err.format()
    assert "E-PP-100" in s
    assert "Base .obs not found" in s
    assert "Fix:" in s


def test_pipeline_error_to_dict_round_trip():
    err = PipelineError("E-PP-104", "PPK produced 0 epochs",
                        hint="lower elevation mask", context={"n": 0})
    d = err.to_dict()
    assert d["code"] == "E-PP-104"
    assert d["message"] == "PPK produced 0 epochs"
    assert d["hint"] == "lower elevation mask"
    assert d["context"] == {"n": 0}
    assert d["description"] == ERROR_CODES["E-PP-104"]


def test_unregistered_code_warns_but_does_not_raise(caplog):
    with caplog.at_level("WARNING"):
        err = PipelineError("E-PP-999999", "test")   # not in registry
    assert "unregistered code" in caplog.text
    # Still constructible and formattable.
    assert "E-PP-999999" in err.format()


def test_save_error_report_writes_json(tmp_path: Path):
    out = tmp_path / "report.json"
    err = PipelineError("E-PP-300", "video not found",
                        hint="check path", context={"video": "/x.mp4"})
    saved = save_error_report(err, path=out, stage="frames")
    assert saved == out and out.is_file()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["code"] == "E-PP-300"
    assert d["stage"] == "frames"
    assert "env" in d and "python" in d["env"]
    assert "traceback" in d


def test_save_error_report_handles_plain_exception(tmp_path: Path):
    out = tmp_path / "report.json"
    try:
        raise ValueError("not a PipelineError")
    except ValueError as e:
        save_error_report(e, path=out, stage="ppk")
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["code"] == "E-PP-900"      # generic "this is a bug" code
    assert "ValueError" in d["message"]


def test_report_user_message_includes_followup_instructions():
    err = PipelineError("E-PP-100", "Base .obs missing")
    msg = report_user_message(err)
    assert "E-PP-100" in msg
    assert "support" in msg.lower()
    assert ".data_to_frames_last_error.json" in msg
