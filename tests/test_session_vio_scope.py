"""Smoke + validation tests for session-motion model scope.

Covers the new modules shipped this session that previously had no test
coverage:
  - data_pipeline.ns_sigma
  - data_pipeline.cv_rts
  - data_pipeline.smoothing (new functions only)
  - data_pipeline.pos_metadata
  - data_pipeline.epoch_weight
  - data_pipeline.base_pos
  - data_pipeline.fgo  (import only — the factor library optional)
  - data_pipeline.parsers.PosRow (new fields backwards compat)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


# ----- ns_sigma -----

def test_sigma_h_from_ns_monotone():
    from data_pipeline.ns_sigma import sigma_h_from_ns
    # Higher ns -> lower sigma (monotonic).
    sigs = sigma_h_from_ns(np.array([2.0, 4.0, 10.0, 16.0, 22.0]))
    assert sigs[0] > sigs[1] > sigs[2] > sigs[3] > sigs[4]
    assert all(s > 0 for s in sigs)


def test_sigma_samples_from_ns_axis_validation():
    from data_pipeline.ns_sigma import sigma_samples_from_ns
    with pytest.raises(ValueError, match="axis"):
        sigma_samples_from_ns(np.array([10.0]), 1.0, axis="z")


def test_weights_from_ns_axis_validation():
    from data_pipeline.ns_sigma import weights_from_ns
    with pytest.raises(ValueError, match="axis"):
        weights_from_ns(np.array([10.0]), axis="bogus")


def test_ns_is_informative():
    from data_pipeline.ns_sigma import ns_is_informative
    assert ns_is_informative(np.array([10, 12, 15, 8, 14])) is True
    assert ns_is_informative(np.array([0, 0, 0, 0])) is False
    assert ns_is_informative(np.array([])) is False


# ----- cv_rts -----

def test_cv_rts_validates_dt():
    from data_pipeline.cv_rts import cv_rts
    with pytest.raises(ValueError, match="dt"):
        cv_rts(np.array([1.0, 2.0]), dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        cv_rts(np.array([1.0, 2.0]), dt=-1.0)


def test_cv_rts_handles_nan_samples():
    from data_pipeline.cv_rts import cv_rts
    z = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = cv_rts(z, dt=1.0)
    assert np.all(np.isfinite(out)), "NaN propagated through cv_rts"


def test_cv_rts_handles_all_nan():
    from data_pipeline.cv_rts import cv_rts
    out = cv_rts(np.full(5, np.nan), dt=1.0)
    assert np.all(np.isnan(out))


def test_cv_rts_handles_empty():
    from data_pipeline.cv_rts import cv_rts
    out = cv_rts(np.array([]), dt=1.0)
    assert out.size == 0


def test_doppler_gate_nan_safe():
    from data_pipeline.cv_rts import doppler_gate
    E = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    N = np.array([0.0, 1.0, 2.0, np.nan, 4.0])
    ve = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    vn = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    bad = doppler_gate(E, N, ve, vn, ts, K=3.0)
    assert bad.dtype == bool
    assert len(bad) == 5


def test_gate_then_cv_one_sample_raises():
    from data_pipeline.cv_rts import gate_then_cv
    single = np.array([1.0])
    with pytest.raises(ValueError, match=">= 2"):
        gate_then_cv(single, single, single, single, single)


# ----- smoothing (new functions) -----

def test_gaussian_smooth_weighted_length_validation():
    from data_pipeline.smoothing import gaussian_smooth_weighted
    with pytest.raises(ValueError, match="length"):
        gaussian_smooth_weighted([1, 2, 3], 1.0, [1, 2])


def test_gaussian_smooth_weighted_uniform_matches_unweighted():
    from data_pipeline.smoothing import gaussian_smooth, gaussian_smooth_weighted
    arr = list(range(50))
    a = gaussian_smooth(arr, 3.0)
    b = gaussian_smooth_weighted(arr, 3.0, [1.0] * 50)
    assert all(abs(x - y) < 1e-10 for x, y in zip(a, b))


def test_gaussian_smooth_adaptive_bw_length_validation():
    from data_pipeline.smoothing import gaussian_smooth_adaptive_bw
    with pytest.raises(ValueError, match="length"):
        gaussian_smooth_adaptive_bw([1, 2, 3], [0.5, 0.5])


# ----- base_pos -----

def test_base_xyz_from_llh_magnitude_correct():
    from data_pipeline.base_pos import base_xyz_from_llh
    # Generic mid-latitude coords -> Cartesian XYZ should be ~6.4M m magnitude.
    x, y, z = base_xyz_from_llh(45.0, 0.0, 100.0)
    mag = math.sqrt(x * x + y * y + z * z)
    assert 6.3e6 < mag < 6.5e6


def test_parse_dms_hemispheres():
    from data_pipeline.base_pos import parse_dms
    assert abs(parse_dms("47 07 24.4 N") - 47.1234) < 1e-3
    assert abs(parse_dms("122 39 15.5 W") - -122.6543) < 1e-3


def test_parse_dms_empty_raises():
    from data_pipeline.base_pos import parse_dms
    with pytest.raises(ValueError):
        parse_dms("")


def test_parse_base_spec_rejects_garbage():
    from data_pipeline.base_pos import parse_base_spec
    for bad in ["", "foo", "ecef:", "ecef:abc,def,ghi",
                "llh:1,2", "1,2,3,4", "inf,inf,inf",
                "ecef:nan,nan,nan", "llh:nan,1,2"]:
        assert parse_base_spec(bad) is None, f"should reject {bad!r}"


def test_parse_base_spec_accepts_ecef():
    from data_pipeline.base_pos import parse_base_spec
    out = parse_base_spec("4500000,3100000,3300000")
    assert out is not None
    assert all(abs(v) > 1e5 for v in out)


def test_parse_base_spec_accepts_llh():
    from data_pipeline.base_pos import parse_base_spec
    out = parse_base_spec("45.0,10.0,100.0")
    assert out is not None
    # Should auto-convert LLH to Cartesian XYZ (magnitudes > 1e5).
    assert all(abs(v) > 1e5 for v in out)


def test_utm_zone_bounds():
    from data_pipeline.base_pos import base_xyz_from_utm
    with pytest.raises(ValueError, match="zone must be 1..60"):
        base_xyz_from_utm(500000, 4649776, 100, "99T")


def test_utm_band_letter_validated():
    from data_pipeline.base_pos import base_xyz_from_utm
    for bad in ["I", "O", "A", "Y"]:
        with pytest.raises(ValueError, match="band letter"):
            base_xyz_from_utm(500000, 4649776, 100, f"10{bad}")


def test_read_rinex_approx_xyz_missing_file():
    from data_pipeline.base_pos import read_rinex_approx_xyz
    assert read_rinex_approx_xyz(Path("/nonexistent/file.obs")) is None


# ----- anchor_pick -----

def test_arbitrate_anchors_validates_sigma():
    from data_pipeline.anchor_pick import arbitrate_anchors
    with pytest.raises(ValueError, match="sigma_ref_m"):
        arbitrate_anchors([], [], [], [], [], sigma_ref_m=0)


def test_arbitrate_anchors_validates_lengths():
    from data_pipeline.anchor_pick import arbitrate_anchors
    from data_pipeline.parsers import PosRow
    r = PosRow(1.0, 32.0, 34.0, 60.0, 2)
    with pytest.raises(ValueError, match="length mismatch"):
        arbitrate_anchors([r], [32.0, 32.1], [34.0], [60.0], [])


# ----- motion model -----

def test_calibrate_R_body_from_cam_validates_pairs():
    from data_pipeline.vio import calibrate_R_body_from_cam
    with pytest.raises(ValueError, match="PPK-heading"):
        calibrate_R_body_from_cam([], [])


# ----- pos_metadata -----

def test_quality_score_handles_nan_sd():
    from data_pipeline.parsers import PosRow
    from data_pipeline.pos_metadata import quality_score
    rows = [PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8)]  # NaN sd_n / sd_e
    qs = quality_score(rows, inflation=1.0)
    assert np.all(np.isfinite(qs)), "NaN propagated through quality_score"


def test_calibrate_sigma_inflation_handles_empty():
    from data_pipeline.pos_metadata import calibrate_sigma_inflation
    assert calibrate_sigma_inflation([]) == 1.0


# ----- epoch_weight -----

def test_inverse_variance_weights_handles_all_nan_sd():
    from data_pipeline.epoch_weight import EpochFeatures, inverse_variance_weights
    feats = [
        EpochFeatures(0, float("nan"), float("nan"), float("nan"), 0, 2),
        EpochFeatures(1, float("nan"), float("nan"), float("nan"), 0, 2),
    ]
    w = inverse_variance_weights(feats)
    assert np.all(np.isfinite(w))
    assert w[0] > 0


def test_aggregate_p_resid_missing_file_returns_empty():
    from data_pipeline.epoch_weight import aggregate_p_resid_per_epoch
    assert aggregate_p_resid_per_epoch(Path("/nonexistent.stat")) == {}


def test_cv_rts_pv_weighted_validates_dt():
    from data_pipeline.epoch_weight import cv_rts_pv_weighted
    z = np.array([1.0, 2.0])
    v = np.array([0.5, 0.5])
    sp = np.array([1.0, 1.0])
    sv = np.array([0.3, 0.3])
    use_v = np.array([True, True])
    with pytest.raises(ValueError, match="dt"):
        cv_rts_pv_weighted(z, v, use_v, dt=0.0, sigma_p_arr=sp,
                           sigma_v_arr=sv, sigma_a=0.2)


def test_cv_rts_pv_weighted_nan_safe():
    from data_pipeline.epoch_weight import cv_rts_pv_weighted
    z = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    v = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    use_v = np.array([True] * 5)
    sp = np.array([1.0] * 5)
    sv = np.array([0.3] * 5)
    out = cv_rts_pv_weighted(z, v, use_v, dt=1.0, sigma_p_arr=sp,
                             sigma_v_arr=sv, sigma_a=0.2)
    assert np.all(np.isfinite(out)), "NaN propagated"


# ----- PosRow -----

def test_posrow_positional_backwards_compat():
    """Old-style positional construction must still work."""
    from data_pipeline.parsers import PosRow
    r = PosRow(1.0, 32.0, 34.0, 60.0, 2, 0.5, 0.3, 0.1, 8)
    assert r.utc_s == 1.0
    assert r.quality == 2
    assert r.ns == 8
    assert math.isnan(r.ratio)  # default NaN


def test_posrow_new_fields_default_nan():
    """New sd_* / ratio / age fields default to NaN (not 0)."""
    from data_pipeline.parsers import PosRow
    r = PosRow(1.0, 32.0, 34.0, 60.0, 2)
    for field in ["sd_n", "sd_e", "sd_u", "sd_ne", "sd_eu", "sd_un",
                  "age_s", "ratio", "sd_vn", "sd_ve", "sd_vu",
                  "sd_vne", "sd_veu", "sd_vun"]:
        assert math.isnan(getattr(r, field)), f"{field} should default NaN"


# ----- fgo (optional) -----

def test_fgo_imports_and_raises_actionable():
    """FGO module imports; the factor library absence raises actionable ImportError."""
    import data_pipeline.fgo  # must import
    try:
        from data_pipeline.fgo import _import_gtsam
        _import_gtsam()  # the factor library IS installed (conda-forge) — should succeed
    except ImportError as e:
        assert "conda install -c conda-forge gtsam" in str(e)


def test_fgo_run_empty_pos_raises():
    from data_pipeline.fgo import run_fgo
    with pytest.raises(ValueError, match="pos_rows is empty"):
        run_fgo([], [{"utc_s": 0.0}])


def test_fgo_run_empty_imu_raises():
    from data_pipeline.fgo import run_fgo
    from data_pipeline.parsers import PosRow
    r = PosRow(1.0, 32.0, 34.0, 60.0, 2)
    with pytest.raises(ValueError, match="imu_rows is empty"):
        run_fgo([r], [])


# ----- user_export -----

def test_user_export_writes_csv(tmp_path):
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.user_export import export_trajectory
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8, sd_n=0.5, sd_e=0.4, sd_u=1.0,
               vn=0.1, ve=0.2, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1),
        PosRow(2.0, 32.51, 34.51, 60.1, 2, ns=8, sd_n=0.5, sd_e=0.4, sd_u=1.0,
               vn=0.1, ve=0.2, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1),
    ]
    out = tmp_path / "traj.csv"
    # These two synthetic points are ~1.1 km apart 1 s apart (a physically
    # impossible 1100 m/s jump), so the default robust_filter would reject one.
    # This smoke test exercises the CSV writer mechanics, not the filter, so it
    # is disabled here; filter behaviour is covered by test_export_robust_filter.
    res = export_trajectory(rows, out, source_tag="unit_test",
                            robust_filter_enabled=False)
    assert res.n_rows == 2
    assert out.is_file()
    content = out.read_text(encoding="utf-8").splitlines()
    assert "gpstime" in content[0]
    assert "std_xy_m" in content[0]
    assert "unit_test" in content[1]


def test_user_export_empty_raises():
    from data_pipeline.stages.user_export import export_trajectory
    with pytest.raises(ValueError, match="empty rows"):
        export_trajectory([], Path("/tmp/x.csv"))


def test_user_export_kml_writes(tmp_path):
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.user_export import export_kml
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8),
        PosRow(2.0, 32.51, 34.51, 60.0, 2, ns=8),
        PosRow(3.0, 32.52, 34.52, 60.0, 2, ns=8),
    ]
    out = tmp_path / "trk.kml"
    export_kml(rows, out, color_by_trust=True, trust_arr=[1.0, 0.5, 0.0])
    body = out.read_text(encoding="utf-8")
    assert "<kml" in body
    assert "LineString" in body


# ----- smoothed_trust_viewer -----

def test_smoothed_trust_viewer_length_mismatch_raises(tmp_path):
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.viewers import build_smoothed_trust_viewer
    raw = [PosRow(1.0, 32.5, 34.5, 60.0, 2)]
    sm = [PosRow(1.0, 32.5, 34.5, 60.0, 2), PosRow(2.0, 32.5, 34.5, 60.0, 2)]
    with pytest.raises(ValueError, match="length mismatch"):
        build_smoothed_trust_viewer(
            raw_pos_rows=raw, smoothed_pos_rows=sm,
            out_html=tmp_path / "x.html",
        )


def test_trajectory_compare_viewer_renders(tmp_path):
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.viewers import build_trajectory_compare_viewer
    rows_a = [PosRow(1.0 + i, 32.5 + i * 1e-5, 34.5 + i * 1e-5, 60.0, 2, ns=8)
              for i in range(20)]
    rows_b = [PosRow(1.0 + i, 32.5 + i * 1e-5 + 5e-7, 34.5 + i * 1e-5, 60.1,
                     2, ns=8)
              for i in range(20)]
    out = tmp_path / "tc.html"
    res = build_trajectory_compare_viewer(
        routes={"raw": rows_a, "v1": rows_b}, out_html=out,
    )
    assert res.n_routes == 2
    assert res.n_epochs == 20
    assert "raw__VS__v1" in res.pairwise_stats or "v1__VS__raw" in res.pairwise_stats
    assert out.is_file()
    assert (tmp_path / "tc.data.js").is_file()


def test_trajectory_compare_viewer_length_mismatch_raises(tmp_path):
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.viewers import build_trajectory_compare_viewer
    rows_a = [PosRow(1.0, 32.5, 34.5, 60.0, 2)]
    rows_b = [PosRow(1.0, 32.5, 34.5, 60.0, 2), PosRow(2.0, 32.5, 34.5, 60.0, 2)]
    with pytest.raises(ValueError, match="same length"):
        build_trajectory_compare_viewer(
            routes={"a": rows_a, "b": rows_b}, out_html=tmp_path / "x.html",
        )


def test_trajectory_compare_viewer_empty_raises(tmp_path):
    from data_pipeline.stages.viewers import build_trajectory_compare_viewer
    with pytest.raises(ValueError, match="empty routes"):
        build_trajectory_compare_viewer(routes={}, out_html=tmp_path / "x.html")


def test_routes_viewer_renders(tmp_path):
    """Multi-route viewer accepts dict of {label: PosRow list}."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.viewers import build_routes_viewer
    rows_a = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8),
        PosRow(2.0, 32.5001, 34.5001, 60.0, 2, ns=8),
    ]
    rows_b = [
        PosRow(1.0, 32.5, 34.5, 60.1, 2, ns=8),
        PosRow(2.0, 32.5002, 34.5002, 60.1, 2, ns=8),
    ]
    out = tmp_path / "routes.html"
    res = build_routes_viewer(routes={"a": rows_a, "b": rows_b}, out_html=out)
    assert res.n_routes == 2
    assert res.n_total_points == 4
    assert out.is_file()
    assert (tmp_path / "routes.data.js").is_file()


def test_routes_viewer_empty_dict_raises(tmp_path):
    from data_pipeline.stages.viewers import build_routes_viewer
    with pytest.raises(ValueError, match="empty routes"):
        build_routes_viewer(routes={}, out_html=tmp_path / "x.html")


def test_accuracy_predictor_empty_raises():
    from data_pipeline.accuracy_predictor import smart_session_std
    with pytest.raises(ValueError, match="empty"):
        smart_session_std([])


def test_accuracy_predictor_returns_profile():
    from data_pipeline.accuracy_predictor import smart_session_std
    from data_pipeline.parsers import PosRow
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7),
        PosRow(2.0, 32.5001, 34.5001, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7),
        PosRow(3.0, 32.5002, 34.5002, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7),
        PosRow(4.0, 32.5003, 34.5003, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7),
        PosRow(5.0, 32.5004, 34.5004, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7),
    ]
    prof = smart_session_std(rows)
    assert prof.smart_std_m >= 0.5  # floor
    assert prof.trust_class in {"trustworthy", "tight", "spike_risk"}
    assert 0.0 <= prof.q2_frac <= 1.0


def test_predicted_epoch_std_length():
    from data_pipeline.accuracy_predictor import predicted_epoch_std
    from data_pipeline.parsers import PosRow
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3),
        PosRow(2.0, 32.5, 34.5, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3),
    ]
    arr = predicted_epoch_std(rows)
    assert arr.shape == (2,)
    assert (arr >= 0.5).all()


def test_predicted_epoch_std_empty_returns_empty():
    from data_pipeline.accuracy_predictor import predicted_epoch_std
    assert predicted_epoch_std([]).size == 0


def test_epoch_weighted_v2_empty():
    from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2
    res = smooth_epoch_weighted_v2([])
    assert res.E_smooth.size == 0
    assert res.n_nhc_updates == 0
    assert res.n_zupt_updates == 0


def test_epoch_weighted_v2_runs_without_imu():
    from data_pipeline.epoch_weight_v2 import (
        EpochWeightV2Options,
        smooth_epoch_weighted_v2,
    )
    from data_pipeline.parsers import PosRow
    rows = [
        PosRow(1.0 + i, 32.5 + i * 1e-5, 34.5 + i * 1e-5, 60.0, 2, ns=8,
               sd_n=0.4, sd_e=0.3, sd_u=0.7,
               vn=0.1, ve=0.2, vu=0.0,
               sd_vn=0.05, sd_ve=0.05, sd_vu=0.1)
        for i in range(20)
    ]
    opts = EpochWeightV2Options(nhc_enabled=False, zupt_enabled=False)
    res = smooth_epoch_weighted_v2(rows, imu_rows=None, options=opts)
    assert res.E_smooth.shape == (20,)
    assert res.N_smooth.shape == (20,)
    assert res.U_smooth.shape == (20,)


def test_epoch_weighted_v2_zupt_triggers_on_low_speed():
    """All-stationary segment should fire ZUPT updates."""
    from data_pipeline.epoch_weight_v2 import (
        EpochWeightV2Options,
        smooth_epoch_weighted_v2,
    )
    from data_pipeline.parsers import PosRow
    rows = [
        PosRow(1.0 + i, 32.5, 34.5, 60.0, 2, ns=8,
               sd_n=0.4, sd_e=0.3, sd_u=0.7,
               vn=0.0, ve=0.0, vu=0.0,
               sd_vn=0.05, sd_ve=0.05, sd_vu=0.1)
        for i in range(20)
    ]
    opts = EpochWeightV2Options(zupt_enabled=True, zupt_min_duration_s=2.0,
                                 nhc_enabled=False)
    res = smooth_epoch_weighted_v2(rows, imu_rows=None, options=opts)
    assert res.n_zupt_updates > 0


def test_accuracy_dashboard_empty_raises(tmp_path):
    from data_pipeline.stages.accuracy_dashboard import build_accuracy_dashboard
    with pytest.raises(ValueError, match="empty raw"):
        build_accuracy_dashboard(
            raw_pos_rows=[], filter_outputs={}, out_html=tmp_path / "x.html",
        )


def test_accuracy_dashboard_renders(tmp_path):
    """Dashboard writes HTML + companion JS with all required structure."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.accuracy_dashboard import build_accuracy_dashboard
    rows = [
        PosRow(1.0 + i, 32.5 + i * 1e-5, 34.5 + i * 1e-5, 60.0, 2,
               ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7,
               vn=0.1, ve=0.2, vu=0.0)
        for i in range(30)
    ]
    res = build_accuracy_dashboard(
        raw_pos_rows=rows, filter_outputs={"test_smooth": rows},
        out_html=tmp_path / "dash.html",
    )
    assert res.n_epochs == 30
    assert res.n_filters == 1
    assert res.trust_class in {"trustworthy", "tight", "spike_risk"}
    assert (tmp_path / "dash.html").is_file()
    assert (tmp_path / "dash.data.js").is_file()
    body = (tmp_path / "dash.html").read_text(encoding="utf-8")
    assert "accuracy_dashboard" in body.lower() or "trust_class" in body


def test_user_export_has_smart_std_column(tmp_path):
    """The new std_xy_smart_m + trust_class columns must be present."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.user_export import export_trajectory
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7,
               vn=0.1, ve=0.2, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1),
        PosRow(2.0, 32.5001, 34.5001, 60.0, 2, ns=8, sd_n=0.4, sd_e=0.3, sd_u=0.7,
               vn=0.1, ve=0.2, vu=0.0, sd_vn=0.05, sd_ve=0.05, sd_vu=0.1),
    ]
    out = tmp_path / "traj.csv"
    res = export_trajectory(rows, out, source_tag="test")
    content = out.read_text(encoding="utf-8").splitlines()
    assert "std_xy_smart_m" in content[0]
    assert "trust_class" in content[0]
    assert res.smart_std_m >= 0.5
    assert res.trust_class in {"trustworthy", "tight", "spike_risk"}


def test_routes_viewer_empty_first_route_raises(tmp_path):
    from data_pipeline.stages.viewers import build_routes_viewer
    with pytest.raises(ValueError, match="is empty"):
        build_routes_viewer(routes={"a": []}, out_html=tmp_path / "x.html")


def test_smoothed_trust_viewer_perfect_agreement(tmp_path):
    """When smoothed == raw, every epoch should have trust ≈ 1."""
    from data_pipeline.parsers import PosRow
    from data_pipeline.stages.viewers import build_smoothed_trust_viewer
    rows = [
        PosRow(1.0, 32.5, 34.5, 60.0, 2),
        PosRow(2.0, 32.5001, 34.5001, 60.0, 2),
        PosRow(3.0, 32.5002, 34.5002, 60.0, 2),
    ]
    res = build_smoothed_trust_viewer(
        raw_pos_rows=rows, smoothed_pos_rows=rows,
        out_html=tmp_path / "t.html",
    )
    assert res.trust_median >= 0.99  # smoothed == raw -> trust ≈ 1
    assert (tmp_path / "t.html").is_file()
    assert (tmp_path / "t.data.js").is_file()
