"""Tests for data_pipeline.photo_compare.

Covers: reconstruction pose parsing (text + binary; camera center =
-R^T t), the similarity (Umeyama) fit incl. outlier rejection, the
georeferenced-vs-raw frame auto-detection, the 1/2/3-sigma aggregation,
frame-time recovery by track projection, and an end-to-end build_report on
synthesized camera / GPS / ground-truth data with a known expected verdict.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import struct
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.geo import enu_to_llh
from data_pipeline.parsers import PosRow
from data_pipeline.photo_compare import (
    FrameRecord,
    Result,
    _motion_fields,
    build_report,
    camera_center_from_pose,
    detect_reconstruction_frame,
    fit_similarity_robust,
    frame_times_from_colmap_names,
    load_frame_times,
    parse_colmap_images,
    parse_metashape_cameras,
    recover_frame_times_from_pos,
    sigma_bands,
    umeyama_similarity,
)

# A plausible mid-latitude reference track origin.
LAT0, LON0, H0 = 47.397_5, 8.545_2, 432.7

BASE_UTC = dt.datetime(2026, 7, 7, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Camera-center math + pose parsing
# ---------------------------------------------------------------------------


def test_camera_center_identity_quat() -> None:
    # Identity rotation: C = -t.
    c = camera_center_from_pose((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert c == pytest.approx((-1.0, -2.0, -3.0))


def test_camera_center_90deg_z() -> None:
    # 90 deg about z: R = [[0,-1,0],[1,0,0],[0,0,1]], C = -R^T t = (-ty, tx, -tz).
    s = math.sqrt(0.5)
    c = camera_center_from_pose((s, 0.0, 0.0, s), (1.0, 2.0, 3.0))
    assert c == pytest.approx((-2.0, 1.0, -3.0), abs=1e-12)


def test_camera_center_unnormalized_quat() -> None:
    # The quaternion must be normalised before use: scaling it must not
    # change the center.
    s = math.sqrt(0.5)
    a = camera_center_from_pose((s, 0.0, 0.0, s), (1.0, 2.0, 3.0))
    b = camera_center_from_pose((2 * s, 0.0, 0.0, 2 * s), (1.0, 2.0, 3.0))
    assert a == pytest.approx(b, abs=1e-12)


_POSES = [
    # (image_id, qvec wxyz, tvec, camera_id, name)
    (1, (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 1, "frame_000001.png"),
    (2, (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (1.0, 2.0, 3.0), 1,
     "frame_000002.jpg"),
    (3, (0.9, 0.1, -0.2, 0.3), (-4.0, 0.5, 7.25), 2, "frame_000003.png"),
]

_EXPECTED_CENTERS = {
    "frame_000001": camera_center_from_pose(_POSES[0][1], _POSES[0][2]),
    "frame_000002": camera_center_from_pose(_POSES[1][1], _POSES[1][2]),
    "frame_000003": camera_center_from_pose(_POSES[2][1], _POSES[2][2]),
}


def _write_images_txt(path: Path) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    points_lines = [
        "10.0 20.0 5 30.5 40.5 -1",  # two 2D points
        "",                          # image with no 2D points -> empty line
        "1.0 2.0 -1",
    ]
    for (img_id, q, t, cam_id, name), pts in zip(_POSES, points_lines):
        lines.append(
            f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} "
            f"{cam_id} {name}"
        )
        lines.append(pts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images_bin(path: Path) -> None:
    n_points = [2, 0, 1]
    buf = bytearray()
    buf += struct.pack("<Q", len(_POSES))
    for (img_id, q, t, cam_id, name), npts in zip(_POSES, n_points):
        buf += struct.pack("<I", img_id)
        buf += struct.pack("<4d", *q)
        buf += struct.pack("<3d", *t)
        buf += struct.pack("<I", cam_id)
        buf += name.encode("utf-8") + b"\x00"
        buf += struct.pack("<Q", npts)
        for k in range(npts):
            buf += struct.pack("<ddQ", 1.0 * k, 2.0 * k, 0xFFFFFFFFFFFFFFFF)
    path.write_bytes(bytes(buf))


def test_parse_images_txt(tmp_path: Path) -> None:
    p = tmp_path / "images.txt"
    _write_images_txt(p)
    got = parse_colmap_images(p)
    assert set(got) == set(_EXPECTED_CENTERS)
    for stem, want in _EXPECTED_CENTERS.items():
        assert got[stem] == pytest.approx(want, abs=1e-9), stem


def test_parse_images_bin_matches_txt(tmp_path: Path) -> None:
    pt = tmp_path / "images.txt"
    pb = tmp_path / "images.bin"
    _write_images_txt(pt)
    _write_images_bin(pb)
    from_txt = parse_colmap_images(pt)
    from_bin = parse_colmap_images(pb)
    assert set(from_bin) == set(from_txt)
    for stem in from_txt:
        assert from_bin[stem] == pytest.approx(from_txt[stem], abs=1e-9)


def test_parse_images_dir_resolution(tmp_path: Path) -> None:
    sub = tmp_path / "sparse" / "0"
    sub.mkdir(parents=True)
    _write_images_bin(sub / "images.bin")
    got = parse_colmap_images(tmp_path)
    assert set(got) == set(_EXPECTED_CENTERS)


# ---------------------------------------------------------------------------
# Similarity (Umeyama) fit
# ---------------------------------------------------------------------------


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_umeyama_recovers_known_similarity() -> None:
    rng = random.Random(7)
    src = np.array([[rng.uniform(-40, 40) for _ in range(3)] for _ in range(20)])
    true_s = 3.7
    true_r = _rot_z(33.0)
    true_t = np.array([120.0, -45.0, 8.5])
    dst = true_s * (src @ true_r.T) + true_t

    scale, r, t = umeyama_similarity(src, dst)
    assert scale == pytest.approx(true_s, rel=1e-9)
    assert np.allclose(r, true_r, atol=1e-9)
    assert np.allclose(t, true_t, atol=1e-6)
    resid = np.linalg.norm(dst - (scale * (src @ r.T) + t), axis=1)
    assert float(resid.max()) < 1e-6


def test_fit_similarity_robust_rejects_outlier() -> None:
    rng = random.Random(3)
    names = [f"frame_{i:06d}" for i in range(15)]
    src = np.array([[rng.uniform(-30, 30) for _ in range(3)] for _ in range(15)])
    true_s = 0.27
    true_r = _rot_z(-58.0)
    true_t = np.array([-12.0, 300.0, 1.5])
    dst = true_s * (src @ true_r.T) + true_t
    dst[4] += np.array([100.0, -50.0, 20.0])  # gross outlier

    fit = fit_similarity_robust(names, src, dst)
    assert fit.scale == pytest.approx(true_s, rel=1e-6)
    assert fit.rms_m < 1e-6
    assert fit.n_used == 14
    assert "frame_000004" in fit.rejected


def test_umeyama_needs_three_points() -> None:
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Georeference auto-detection
# ---------------------------------------------------------------------------


def _gps_track_enu(n: int = 20) -> dict[str, tuple[float, float, float]]:
    return {
        f"frame_{i:06d}": (2.0 * i, 15.0 * math.sin(i / 8.0), 0.1 * i)
        for i in range(n)
    }


def test_detect_georeferenced_llh() -> None:
    origin = (LAT0, LON0, H0)
    gps_enu = _gps_track_enu()
    cams: dict[str, tuple[float, float, float]] = {}
    for stem, (e, n, u) in gps_enu.items():
        lat, lon, h = enu_to_llh(e + 0.4, n - 0.2, u, origin)
        cams[stem] = (lon, lat, h)  # x=lon, y=lat, z=h
    assert detect_reconstruction_frame(cams, gps_enu, origin) == "llh"


def test_detect_georeferenced_local_metric() -> None:
    origin = (LAT0, LON0, H0)
    gps_enu = _gps_track_enu()
    cams = {
        stem: (e + 0.3, n + 0.1, u - 0.05)
        for stem, (e, n, u) in gps_enu.items()
    }
    assert detect_reconstruction_frame(cams, gps_enu, origin) == "local"


def test_detect_raw_frame() -> None:
    origin = (LAT0, LON0, H0)
    gps_enu = _gps_track_enu()
    r = _rot_z(75.0)
    cams = {}
    for stem, p in gps_enu.items():
        q = 0.05 * (r @ np.asarray(p)) + np.array([170.0, 80.0, 3.0])
        cams[stem] = (float(q[0]), float(q[1]), float(q[2]))
    # Fits inside |x|<=180, |y|<=90, but is thousands of km from the track
    # as lon/lat and far outside the ENU extent as metric coords -> raw.
    assert detect_reconstruction_frame(cams, gps_enu, origin) == "raw"


# ---------------------------------------------------------------------------
# Sigma-band aggregation
# ---------------------------------------------------------------------------


def test_sigma_bands_known_distribution() -> None:
    b = sigma_bands([float(v) for v in range(1, 101)])
    assert b["n"] == 100
    assert b["mean"] == pytest.approx(50.5)
    assert b["std"] == pytest.approx(math.sqrt((100.0 ** 2 - 1.0) / 12.0), rel=1e-9)
    # percentiles of |e| over 1..100 with linear interpolation on (n-1)
    assert b["sigma1"] == pytest.approx(1.0 + 0.6827 * 99.0, rel=1e-9)
    assert b["sigma2"] == pytest.approx(1.0 + 0.9545 * 99.0, rel=1e-9)
    assert b["sigma3"] == pytest.approx(1.0 + 0.9973 * 99.0, rel=1e-9)
    assert b["max_abs"] == pytest.approx(100.0)


def test_sigma_bands_signed_and_empty() -> None:
    b = sigma_bands([-2.0, 2.0])
    assert b["mean"] == pytest.approx(0.0)
    assert b["mean_abs"] == pytest.approx(2.0)
    assert b["sigma2"] == pytest.approx(2.0)
    empty = sigma_bands([float("nan")])
    assert empty["n"] == 0
    assert math.isnan(empty["sigma2"])


# ---------------------------------------------------------------------------
# Frame-time recovery by track projection
# ---------------------------------------------------------------------------


def test_recover_frame_times_from_pos() -> None:
    origin = (LAT0, LON0, H0)
    rows = []
    for i in range(40):
        lat, lon, h = enu_to_llh(2.0 * i, 0.0, 0.0, origin)
        rows.append(PosRow(BASE_UTC + i, lat, lon, h, 1))
    gps_llh = {}
    truth = {}
    for i in range(2, 38, 3):
        t = BASE_UTC + i + 0.5
        lat, lon, h = enu_to_llh(2.0 * (i + 0.5), 0.0, 0.0, origin)
        stem = f"frame_{i:06d}"
        gps_llh[stem] = (lat, lon, h)
        truth[stem] = t
    got, stats = recover_frame_times_from_pos(gps_llh, rows)
    assert stats["n_matched"] == len(truth)
    assert stats["median_proj_dist_m"] < 0.01
    for stem, want in truth.items():
        assert got[stem] == pytest.approx(want, abs=0.01)


# ---------------------------------------------------------------------------
# End-to-end build_report
# ---------------------------------------------------------------------------

N_FRAMES = 60


def _gt_enu_at(t_rel: float) -> tuple[float, float, float]:
    """Synthetic ground-truth track: ~2 m/s east with gentle weaving."""
    return (
        2.0 * t_rel,
        15.0 * math.sin(t_rel / 8.0),
        2.0 * math.sin(t_rel / 15.0),
    )


def _write_pos(path: Path, times_rel: list[float]) -> None:
    origin = (LAT0, LON0, H0)
    lines = [
        "% program : synthetic test writer",
        "%  UTC          latitude(deg) longitude(deg)  height(m)   Q  ns",
    ]
    for tr in times_rel:
        lat, lon, h = enu_to_llh(*_gt_enu_at(tr), origin)
        stamp = dt.datetime.fromtimestamp(
            BASE_UTC + tr, tz=dt.timezone.utc
        ).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        lines.append(f"{stamp}   {lat:.12f}   {lon:.12f}   {h:.4f}   1  12")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_stems() -> list[str]:
    return [f"frame_{i:06d}" for i in range(N_FRAMES)]


def _frame_time_rel(i: int) -> float:
    return 2.0 + i  # inside the .pos coverage


def _write_frame_times(path: Path) -> None:
    lines = ["Image,utc_s"]
    for i, stem in enumerate(_frame_stems()):
        lines.append(f"{stem},{BASE_UTC + _frame_time_rel(i):.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_georef(path: Path, noise_amp: float, seed: int) -> None:
    rng = random.Random(seed)
    origin = (LAT0, LON0, H0)
    lines = ["Image,Latitude,Longitude,Altitude,Trust"]
    for i, stem in enumerate(_frame_stems()):
        e, n, u = _gt_enu_at(_frame_time_rel(i))
        e += rng.uniform(-noise_amp, noise_amp)
        n += rng.uniform(-noise_amp, noise_amp)
        u += rng.uniform(-noise_amp, noise_amp)
        lat, lon, h = enu_to_llh(e, n, u, origin)
        lines.append(f"{stem},{lat:.12f},{lon:.12f},{h:.4f},1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_camera_images_txt(
    path: Path,
    noise_amp: float,
    seed: int,
    *,
    georeferenced: bool,
) -> None:
    """Camera model with the given accuracy vs truth.

    ``georeferenced=True`` writes centers as (lon, lat, h); otherwise
    centers are pushed through a known similarity into an arbitrary raw
    local frame (which build_report must undo via the fit).
    """
    rng = random.Random(seed)
    origin = (LAT0, LON0, H0)
    raw_s = 3.7
    raw_r = _rot_z(33.0)
    raw_t = np.array([500.0, -200.0, 12.0])
    lines = ["# synthetic reconstruction"]
    for i, stem in enumerate(_frame_stems()):
        e, n, u = _gt_enu_at(_frame_time_rel(i))
        e += rng.uniform(-noise_amp, noise_amp)
        n += rng.uniform(-noise_amp, noise_amp)
        u += rng.uniform(-noise_amp, noise_amp)
        if georeferenced:
            lat, lon, h = enu_to_llh(e, n, u, origin)
            cx, cy, cz = lon, lat, h
        else:
            q = raw_s * (raw_r @ np.array([e, n, u])) + raw_t
            cx, cy, cz = float(q[0]), float(q[1]), float(q[2])
        # Identity rotation -> center C = -t, so store t = -C.
        lines.append(
            f"{i + 1} 1 0 0 0 {-cx:.9f} {-cy:.9f} {-cz:.9f} 1 {stem}.png"
        )
        lines.append("")  # empty 2D-points line
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case(
    tmp_path: Path, *, cam_noise: float, gps_noise: float, georeferenced: bool,
) -> Result:
    gt_pos = tmp_path / "truth.pos"
    _write_pos(gt_pos, [float(i) for i in range(N_FRAMES + 5)])
    georef = tmp_path / "Georef.csv"
    _write_georef(georef, gps_noise, seed=11)
    images = tmp_path / "images.txt"
    _write_camera_images_txt(images, cam_noise, seed=22,
                             georeferenced=georeferenced)
    ftimes = tmp_path / "frame_times_utc.csv"
    _write_frame_times(ftimes)
    return build_report(
        images, gt_pos, tmp_path / "out",
        georef_csv=georef, frame_times=ftimes,
    )


def test_build_report_camera_model_wins(tmp_path: Path) -> None:
    # Camera model: 2 cm scatter in a RAW frame; GPS: 1 m scatter.
    res = _run_case(tmp_path, cam_noise=0.02, gps_noise=1.0,
                    georeferenced=False)
    assert res.mode == "raw"
    assert res.fit is not None
    # The fit maps the raw frame back to metric: scale ~= 1/3.7.
    assert res.fit.scale == pytest.approx(1.0 / 3.7, rel=0.02)
    assert res.summary["n_matched"] == N_FRAMES

    v = res.summary["verdict"]
    assert v["horizontal_2sigma_m"]["winner"] == "camera model"
    assert v["speed_2sigma_mps"]["winner"] == "camera model"
    assert v["azimuth_2sigma_deg"]["winner"] == "camera model"
    assert "camera model is closer" in res.verdict

    agg = res.summary["aggregates"]
    assert agg["camera"]["horiz_m"]["sigma2"] < agg["gps"]["horiz_m"]["sigma2"]
    assert agg["camera"]["azimuth_deg"]["n"] > 0
    assert agg["camera"]["speed_mps"]["n"] > 0

    assert res.csv_path is not None and res.csv_path.is_file()
    assert res.html_path is not None and res.html_path.is_file()
    html = res.html_path.read_text(encoding="utf-8")
    assert "camera model is closer" in html
    assert "fitted scale" in html
    csv_text = res.csv_path.read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("Image,utc_s,gt_lat")
    assert len(csv_text.splitlines()) == 1 + N_FRAMES


def test_build_report_gps_wins(tmp_path: Path) -> None:
    # Camera model: 1.2 m scatter, georeferenced (lon/lat); GPS: 5 cm.
    res = _run_case(tmp_path, cam_noise=1.2, gps_noise=0.05,
                    georeferenced=True)
    assert res.mode == "llh"
    assert res.fit is None

    v = res.summary["verdict"]
    assert v["horizontal_2sigma_m"]["winner"] == "GPS"
    assert v["speed_2sigma_mps"]["winner"] == "GPS"
    assert v["azimuth_2sigma_deg"]["winner"] == "GPS"
    assert "GPS track is closer" in res.verdict

    agg = res.summary["aggregates"]
    assert agg["gps"]["horiz_m"]["sigma2"] < agg["camera"]["horiz_m"]["sigma2"]
    # GPS scatter is 5 cm; its 2-sigma vs truth must be small.
    assert agg["gps"]["horiz_m"]["sigma2"] < 0.15


def test_load_frame_times_utc_column(tmp_path: Path) -> None:
    p = tmp_path / "ft.csv"
    _write_frame_times(p)
    got = load_frame_times(p)
    assert got is not None
    assert got["frame_000000"] == pytest.approx(BASE_UTC + 2.0, abs=1e-3)


def test_load_frame_times_video_relative_rejected(tmp_path: Path) -> None:
    p = tmp_path / "extracted_frame_times.csv"
    p.write_text("Image,t_video_s\nframe_000000,0.033\nframe_000001,1.033\n",
                 encoding="utf-8")
    # No UTC column -> None (needs the session anchor instead).
    assert load_frame_times(p) is None


def test_records_have_expected_error_shape(tmp_path: Path) -> None:
    res = _run_case(tmp_path, cam_noise=0.02, gps_noise=1.0,
                    georeferenced=False)
    r: FrameRecord = res.records[5]
    assert r.cam_horiz_err_m == pytest.approx(
        math.hypot(r.cam_de, r.cam_dn), abs=1e-9)
    assert r.gps_horiz_err_m == pytest.approx(
        math.hypot(r.gps_de, r.gps_dn), abs=1e-9)
    # First frame has no motion-derived fields.
    first = res.records[0]
    assert math.isnan(first.cam_speed_mps)
    assert math.isnan(first.cam_az_err_deg)


# ---------------------------------------------------------------------------
# camera-model estimated-coordinates CSV parsing
# ---------------------------------------------------------------------------


def test_parse_metashape_hash_header_x_lon_y_lat(tmp_path: Path) -> None:
    # Real-file shape: leading comment lines, '#'-prefixed header row and
    # the camera-model geographic column order X=Longitude, Y=Latitude, Z=Alt.
    p = tmp_path / "cameras.csv"
    p.write_text(
        "# Cameras (synthetic Metashape export)\n"
        "#Label,X,Y,Z,Yaw,Pitch,Roll\n"
        "frame_000001.jpg,8.545200,47.397500,432.700,10.0,0.5,-0.5\n"
        "frame_000002.jpg,8.545300,47.397600,433.100,11.0,0.4,-0.4\n",
        encoding="utf-8",
    )
    got = parse_metashape_cameras(p)
    assert set(got) == {"frame_000001", "frame_000002"}
    lat, lon, h = got["frame_000001"]
    assert lat == pytest.approx(47.3975, abs=1e-9)   # Y -> lat
    assert lon == pytest.approx(8.5452, abs=1e-9)    # X -> lon
    assert h == pytest.approx(432.7, abs=1e-6)       # Z -> alt


def test_parse_metashape_explicit_headers_semicolon(tmp_path: Path) -> None:
    # Explicit Longitude/Latitude/Altitude headers win over X/Y/Z order;
    # semicolon delimiter is sniffed; header not '#'-prefixed here.
    p = tmp_path / "cameras.csv"
    p.write_text(
        "Label;Longitude;Latitude;Altitude\n"
        "a.jpg;8.500000;47.400000;432.000\n"
        "b.jpg;8.500100;47.400100;432.500\n",
        encoding="utf-8",
    )
    got = parse_metashape_cameras(p)
    assert got["a"] == pytest.approx((47.4, 8.5, 432.0), abs=1e-9)
    assert got["b"] == pytest.approx((47.4001, 8.5001, 432.5), abs=1e-9)


def test_parse_metashape_no_coordinate_columns(tmp_path: Path) -> None:
    p = tmp_path / "cameras.csv"
    p.write_text("Label,Foo,Bar\na.jpg,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate columns"):
        parse_metashape_cameras(p)


def test_parse_metashape_no_name_column(tmp_path: Path) -> None:
    p = tmp_path / "cameras.csv"
    p.write_text("#Foo,X,Y,Z\n1,8.5,47.4,432.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="name column"):
        parse_metashape_cameras(p)


# ---------------------------------------------------------------------------
# build_report with camera_source="metashape"
# ---------------------------------------------------------------------------


def _write_metashape_csv(path: Path, noise_amp: float, seed: int) -> None:
    rng = random.Random(seed)
    origin = (LAT0, LON0, H0)
    lines = [
        "# Cameras (synthetic Metashape export)",
        "#Label,X,Y,Z,Yaw,Pitch,Roll",
    ]
    for i, stem in enumerate(_frame_stems()):
        e, n, u = _gt_enu_at(_frame_time_rel(i))
        e += rng.uniform(-noise_amp, noise_amp)
        n += rng.uniform(-noise_amp, noise_amp)
        u += rng.uniform(-noise_amp, noise_amp)
        lat, lon, h = enu_to_llh(e, n, u, origin)
        lines.append(f"{stem}.jpg,{lon:.12f},{lat:.12f},{h:.4f},0,0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_report_metashape_source(tmp_path: Path) -> None:
    gt_pos = tmp_path / "truth.pos"
    _write_pos(gt_pos, [float(i) for i in range(N_FRAMES + 5)])
    georef = tmp_path / "Georef.csv"
    _write_georef(georef, 0.05, seed=11)
    ms = tmp_path / "cameras.csv"
    _write_metashape_csv(ms, 0.04, seed=33)
    ftimes = tmp_path / "frame_times_utc.csv"
    _write_frame_times(ftimes)

    res = build_report(
        ms, gt_pos, tmp_path / "out",
        georef_csv=georef, frame_times=ftimes,
        camera_source="metashape",
    )
    # camera-model is WGS84 -> forced llh path, no similarity fit.
    assert res.mode == "llh"
    assert res.fit is None
    assert res.summary["camera_source"] == "metashape"
    assert res.summary["n_matched"] == N_FRAMES

    agg = res.summary["aggregates"]
    for key in ("horiz_m", "speed_mps", "vel3d_mps", "azimuth_deg"):
        band = agg["camera"][key]
        assert band["n"] > 0, key
        assert math.isfinite(band["sigma1"]), key
        assert math.isfinite(band["sigma2"]), key
        assert math.isfinite(band["sigma3"]), key
        assert math.isfinite(band["max_abs"]), key
    # Camera scatter is 4 cm -> its horizontal 2-sigma must be small.
    assert agg["camera"]["horiz_m"]["sigma2"] < 0.15
    assert res.csv_path is not None and res.csv_path.is_file()

    # The metashape= keyword alone also selects the source.
    res2 = build_report(
        ms, gt_pos, tmp_path / "out2",
        georef_csv=georef, frame_times=ftimes,
        metashape=ms, write_html=False,
    )
    assert res2.mode == "llh"
    assert res2.summary["camera_source"] == "metashape"


def test_build_report_colmap_unchanged_by_default(tmp_path: Path) -> None:
    # Sanity: default camera_source keeps the COLMAP raw-frame path intact.
    res = _run_case(tmp_path, cam_noise=0.02, gps_noise=1.0,
                    georeferenced=False)
    assert res.summary["camera_source"] == "colmap"
    assert res.mode == "raw"
    assert res.fit is not None


def _write_metashape_local_csv(path: Path, seed: int) -> None:
    """A camera-model export in an arbitrary LOCAL chunk frame (metric, not
    georeferenced) -- like an indoor / un-georeferenced reconstruction. The
    coords are the GT ENU track pushed through a rotation+scale+offset, so
    treating X/Y as lon/lat would be nonsense; the report must fit it to GPS.
    """
    rng = random.Random(seed)
    th = math.radians(40.0)
    c, s = math.cos(th), math.sin(th)
    # Small metric scale so coords stay IN the lat/lon numeric range (like a
    # real Polycam/Metashape local export, e.g. -0.04, 0.02) -- the parser
    # accepts them; the report must still detect they are not geographic.
    scale = 0.01
    lines = ["# Cameras (synthetic local Metashape export)", "#Label,X,Y,Z"]
    for i, stem in enumerate(_frame_stems()):
        e, n, u = _gt_enu_at(_frame_time_rel(i))
        e += rng.uniform(-0.02, 0.02)
        n += rng.uniform(-0.02, 0.02)
        x = scale * (c * e - s * n) + 0.10
        y = scale * (s * e + c * n) - 0.05
        z = scale * u + 0.02
        lines.append(f"{stem}.jpg,{x:.6f},{y:.6f},{z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_report_metashape_local_frame_fits_to_gps(tmp_path: Path) -> None:
    # A local / un-georeferenced camera-model export must NOT be treated as WGS84;
    # it auto-detects as local/raw and is fit to the GPS track.
    gt_pos = tmp_path / "truth.pos"
    _write_pos(gt_pos, [float(i) for i in range(N_FRAMES + 5)])
    georef = tmp_path / "Georef.csv"
    _write_georef(georef, 0.05, seed=11)
    ms = tmp_path / "cameras_local.csv"
    _write_metashape_local_csv(ms, seed=7)
    ftimes = tmp_path / "frame_times_utc.csv"
    _write_frame_times(ftimes)

    res = build_report(
        ms, gt_pos, tmp_path / "out_local",
        georef_csv=georef, frame_times=ftimes,
        camera_source="metashape", write_html=False,
    )
    assert res.summary["camera_source"] == "metashape"
    assert res.mode in ("raw", "local")   # NOT llh
    if res.mode == "raw":
        assert res.fit is not None
    band = res.summary["aggregates"]["camera"]["horiz_m"]
    assert band["n"] > 0
    assert math.isfinite(band["sigma2"])


# ---------------------------------------------------------------------------
# cam-vs-gps heading + Doppler-grade velocity
# ---------------------------------------------------------------------------


def test_cam_vs_gps_azimuth_error(tmp_path: Path) -> None:
    res = _run_case(tmp_path, cam_noise=1.2, gps_noise=0.05,
                    georeferenced=True)
    finite = [r.cam_vs_gps_az_err_deg for r in res.records
              if math.isfinite(r.cam_vs_gps_az_err_deg)]
    # Track moves at ~2 m/s -> nearly every non-first frame qualifies.
    assert len(finite) > len(res.records) // 2
    assert all(-180.0 < v <= 180.0 for v in finite)
    assert res.summary["aggregates"]["cam_vs_gps"]["azimuth_deg"]["n"] == len(finite)
    # First frame has no bearings -> no cam-vs-gps error either.
    assert math.isnan(res.records[0].cam_vs_gps_az_err_deg)

    csv_text = res.csv_path.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert header.endswith("cam_vs_gps_az_err_deg")
    html = res.html_path.read_text(encoding="utf-8")
    assert "camera model vs GPS" in html


def _motion_recs(n: int = 5) -> list[FrameRecord]:
    """Frames moving due east at exactly 2 m/s in all three sources."""
    recs = []
    for i in range(n):
        e = 2.0 * i
        recs.append(FrameRecord(
            name=f"frame_{i:06d}", utc_s=BASE_UTC + i,
            gt_lat=LAT0, gt_lon=LON0, gt_h=H0,
            cam_e=e, cam_n=0.0, cam_u=0.0,
            gps_e=e, gps_n=0.0, gps_u=0.0,
            gt_e=e, gt_n=0.0, gt_u=0.0,
        ))
    return recs


def test_motion_fields_doppler_velocity_for_gps() -> None:
    # .pos rows carry a native velocity of 5 m/s east -- deliberately NOT
    # the 2 m/s finite-difference speed -- so the Doppler path is provable.
    recs = _motion_recs()
    rows = [
        PosRow(BASE_UTC + i, LAT0, LON0, H0, 1, vn=0.0, ve=5.0, vu=0.0)
        for i in range(len(recs))
    ]
    _motion_fields(recs, speed_floor_mps=0.5, max_step_s=5.0,
                   gps_pos_rows=rows, gt_pos_rows=None, max_gap_s=2.0)
    r = recs[2]
    assert r.gps_speed_mps == pytest.approx(5.0, abs=1e-9)   # Doppler
    assert r.gt_speed_mps == pytest.approx(2.0, abs=1e-9)    # finite-diff
    assert r.cam_speed_mps == pytest.approx(2.0, abs=1e-9)   # finite-diff
    assert r.gps_az_deg == pytest.approx(90.0, abs=1e-9)
    assert r.gps_speed_err_mps == pytest.approx(3.0, abs=1e-9)
    # vel3d error = |(5,0,0) - (2,0,0)| = 3 m/s.
    assert r.gps_vel3d_err_mps == pytest.approx(3.0, abs=1e-9)
    assert r.gps_az_err_deg == pytest.approx(0.0, abs=1e-9)
    assert r.cam_vs_gps_az_err_deg == pytest.approx(0.0, abs=1e-9)
    # Doppler velocity needs no previous frame: defined on the first frame.
    assert recs[0].gps_speed_mps == pytest.approx(5.0, abs=1e-9)
    assert math.isnan(recs[0].cam_speed_mps)


def test_frame_times_from_colmap_names() -> None:
    names = [
        "frame_1474.071944_0",
        "frame_1474.239111_1",
        "chopA_clip_99.5_12",        # any prefix before the float is fine
        "frame_000001",              # plain counter: no embedded time
        "IMG_20260217_180720",       # int_int (no decimal point): skipped
        "frame_1474.071944",         # no _<idx> suffix: skipped
        "notatime",
    ]
    got = frame_times_from_colmap_names(names)
    assert got == {
        "frame_1474.071944_0": pytest.approx(1474.071944),
        "frame_1474.239111_1": pytest.approx(1474.239111),
        "chopA_clip_99.5_12": pytest.approx(99.5),
    }
    assert frame_times_from_colmap_names(["a", "frame_000002"]) == {}


# ---------------------------------------------------------------------------
# build_report with times embedded in the COLMAP image names
# ---------------------------------------------------------------------------


def _name_timed_stems() -> list[str]:
    """Stems in the camera-model->COLMAP export style: frame_<utc>_<idx>."""
    return [
        f"frame_{BASE_UTC + _frame_time_rel(i):.6f}_{i}"
        for i in range(N_FRAMES)
    ]


def _write_name_timed_images_txt(path: Path, noise_amp: float, seed: int) -> None:
    """Georeferenced (lon/lat/h) reconstruction whose image NAMES carry the
    per-frame UTC (name_time_base='utc' in the test for simplicity)."""
    rng = random.Random(seed)
    origin = (LAT0, LON0, H0)
    lines = ["# synthetic self-timing reconstruction"]
    for i, stem in enumerate(_name_timed_stems()):
        e, n, u = _gt_enu_at(_frame_time_rel(i))
        e += rng.uniform(-noise_amp, noise_amp)
        n += rng.uniform(-noise_amp, noise_amp)
        u += rng.uniform(-noise_amp, noise_amp)
        lat, lon, h = enu_to_llh(e, n, u, origin)
        # Identity rotation -> center C = -t, so store t = -C.
        lines.append(
            f"{i + 1} 1 0 0 0 {-lon:.12f} {-lat:.12f} {-h:.9f} 1 {stem}.png"
        )
        lines.append("")  # empty 2D-points line
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_report_uses_colmap_name_times(tmp_path: Path) -> None:
    # No frame-times CSV, no Georef.csv: the only time source is the frame
    # names themselves (frame_<utc>_<idx>); GPS comes from --pos interpolation.
    gt_pos = tmp_path / "truth.pos"
    _write_pos(gt_pos, [float(i) for i in range(N_FRAMES + 5)])
    gps_pos = tmp_path / "rover.pos"
    _write_pos(gps_pos, [float(i) for i in range(N_FRAMES + 5)])
    images = tmp_path / "images.txt"
    _write_name_timed_images_txt(images, 0.05, seed=44)

    res = build_report(
        images, gt_pos, tmp_path / "out",
        pos=gps_pos, name_time_base="utc",
    )
    assert res.mode == "llh"
    assert res.summary["n_matched"] == N_FRAMES
    assert "embedded frame-name times" in res.summary["time_source"]
    assert "utc base" in res.summary["time_source"]

    r = res.records[3]
    assert r.utc_s == pytest.approx(BASE_UTC + _frame_time_rel(3), abs=1e-3)

    agg = res.summary["aggregates"]
    for key in ("horiz_m", "speed_mps", "vel3d_mps", "azimuth_deg"):
        band = agg["camera"][key]
        assert band["n"] > 0, key
        assert math.isfinite(band["sigma1"]), key
        assert math.isfinite(band["sigma2"]), key
        assert math.isfinite(band["sigma3"]), key
        assert math.isfinite(band["max_abs"]), key
    assert res.csv_path is not None and res.csv_path.is_file()


def test_build_report_name_times_video_base_needs_session(tmp_path: Path) -> None:
    # Video-relative embedded times without a session anchor cannot be
    # converted -- build_report must refuse rather than misuse them as UTC.
    gt_pos = tmp_path / "truth.pos"
    _write_pos(gt_pos, [float(i) for i in range(N_FRAMES + 5)])
    gps_pos = tmp_path / "rover.pos"
    _write_pos(gps_pos, [float(i) for i in range(N_FRAMES + 5)])
    images = tmp_path / "images.txt"
    _write_name_timed_images_txt(images, 0.05, seed=44)

    with pytest.raises(ValueError, match="frame_<t>_<idx>"):
        build_report(images, gt_pos, tmp_path / "out",
                     pos=gps_pos, name_time_base="video")


def test_build_report_rejects_bad_name_time_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name_time_base"):
        build_report(tmp_path / "images.txt", tmp_path / "truth.pos",
                     tmp_path / "out", name_time_base="gps")


# ---------------------------------------------------------------------------
# Percentage speed error
# ---------------------------------------------------------------------------


def test_speed_pct_err_values() -> None:
    # GT 10 m/s, camera 11 m/s, GPS 12 m/s (all east, finite differences):
    # cam pct = 100*|11-10|/10 = 10 %, gps pct = 20 %.
    recs = []
    for i in range(5):
        recs.append(FrameRecord(
            name=f"frame_{i:06d}", utc_s=BASE_UTC + i,
            gt_lat=LAT0, gt_lon=LON0, gt_h=H0,
            cam_e=11.0 * i, cam_n=0.0, cam_u=0.0,
            gps_e=12.0 * i, gps_n=0.0, gps_u=0.0,
            gt_e=10.0 * i, gt_n=0.0, gt_u=0.0,
        ))
    _motion_fields(recs, speed_floor_mps=0.5, max_step_s=5.0)
    r = recs[2]
    assert r.gt_speed_mps == pytest.approx(10.0, abs=1e-9)
    assert r.cam_speed_mps == pytest.approx(11.0, abs=1e-9)
    assert r.cam_speed_pct_err == pytest.approx(10.0, abs=1e-9)
    assert r.gps_speed_pct_err == pytest.approx(20.0, abs=1e-9)
    # First frame has no motion fields at all.
    assert math.isnan(recs[0].cam_speed_pct_err)
    assert math.isnan(recs[0].gps_speed_pct_err)


def test_speed_pct_err_floor_guards_stationary_truth() -> None:
    # Stationary truth: the divide uses max(gt_speed, floor), never /0.
    recs = []
    for i in range(4):
        recs.append(FrameRecord(
            name=f"frame_{i:06d}", utc_s=BASE_UTC + i,
            gt_lat=LAT0, gt_lon=LON0, gt_h=H0,
            cam_e=1.0 * i, cam_n=0.0, cam_u=0.0,
            gps_e=0.0, gps_n=0.0, gps_u=0.0,
            gt_e=0.0, gt_n=0.0, gt_u=0.0,
        ))
    _motion_fields(recs, speed_floor_mps=0.5, max_step_s=5.0)
    r = recs[2]
    assert r.gt_speed_mps == pytest.approx(0.0, abs=1e-9)
    # cam speed 1 m/s vs GT 0 -> 100*1/max(0, 0.5) = 200 %.
    assert r.cam_speed_pct_err == pytest.approx(200.0, abs=1e-6)
    assert math.isfinite(r.gps_speed_pct_err)


def test_speed_pct_in_aggregate_csv_and_html(tmp_path: Path) -> None:
    res = _run_case(tmp_path, cam_noise=0.02, gps_noise=1.0,
                    georeferenced=False)
    agg = res.summary["aggregates"]
    for src in ("camera", "gps"):
        band = agg[src]["speed_pct"]
        assert band["n"] > 0, src
        assert math.isfinite(band["sigma2"]), src

    # Per-frame value matches the definition on a real record.
    checked = 0
    for r in res.records:
        if math.isfinite(r.cam_speed_pct_err):
            want = (100.0 * abs(r.cam_speed_mps - r.gt_speed_mps)
                    / max(r.gt_speed_mps, 0.5))
            assert r.cam_speed_pct_err == pytest.approx(want, abs=1e-9)
            checked += 1
    assert checked > 0

    header = res.csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert ",cam_speed_pct_err,gps_speed_pct_err," in header
    html = res.html_path.read_text(encoding="utf-8")
    assert "Speed error (% of GT speed)" in html


def test_motion_fields_falls_back_without_velocity_columns() -> None:
    # Same rows but with the default NaN velocities (a .pos without
    # velocity columns) -> finite-difference speeds as before.
    recs = _motion_recs()
    rows = [PosRow(BASE_UTC + i, LAT0, LON0, H0, 1) for i in range(len(recs))]
    _motion_fields(recs, speed_floor_mps=0.5, max_step_s=5.0,
                   gps_pos_rows=rows, gt_pos_rows=rows, max_gap_s=2.0)
    r = recs[2]
    assert r.gps_speed_mps == pytest.approx(2.0, abs=1e-9)
    assert r.gt_speed_mps == pytest.approx(2.0, abs=1e-9)
    assert r.gps_vel3d_err_mps == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(recs[0].gps_speed_mps)
