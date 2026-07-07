import json
from pathlib import Path

from data_pipeline.stages import viewers


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
