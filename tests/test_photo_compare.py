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
    build_report,
    camera_center_from_pose,
    detect_reconstruction_frame,
    fit_similarity_robust,
    load_frame_times,
    parse_colmap_images,
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
