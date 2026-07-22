import csv
from data_pipeline.traj_score import score_trajectories


def _write_traj(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gpstime", "lat_deg", "lon_deg"])
        for t, la, lo in rows:
            w.writerow([f"{t:.3f}", f"{la:.9f}", f"{lo:.9f}"])


def test_identical_trajectories_score_zero(tmp_path):
    rows = [(1000.0 + i, 32.06 + i * 1e-5, 34.79 + i * 1e-5) for i in range(50)]
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_traj(a, rows); _write_traj(b, rows)
    s = score_trajectories(a, b)
    assert s["n"] == 50
    assert s["max_m"] < 1e-6
    assert s["two_sigma_m"] < 1e-6
    assert s["le1m_pct"] == 100.0


def test_constant_offset_removed(tmp_path):
    rows_a = [(1000.0 + i, 32.06, 34.79) for i in range(30)]
    rows_b = [(1000.0 + i, 32.06 + 1e-5, 34.79) for i in range(30)]
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_traj(a, rows_a); _write_traj(b, rows_b)
    s = score_trajectories(a, b)
    assert 0.9 < s["median_offset_m"] < 1.3
    assert s["two_sigma_m"] < 0.05


def test_single_spike_shows_in_max(tmp_path):
    rows_a = [(1000.0 + i, 32.06, 34.79) for i in range(30)]
    rows_b = [(1000.0 + i, 32.06, 34.79) for i in range(30)]
    rows_b[15] = (1015.0, 32.06 + 2e-5, 34.79)
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_traj(a, rows_a); _write_traj(b, rows_b)
    s = score_trajectories(a, b)
    assert s["max_m"] > 1.5
    assert s["le1m_pct"] < 100.0


def test_score_reports_le1m_and_max_together(tmp_path):
    rows_a = [(1000.0 + i, 32.06, 34.79) for i in range(100)]
    rows_b = list(rows_a)
    for i in range(0, 100, 10):
        rows_b[i] = (1000.0 + i, 32.06 + 2e-5, 34.79)
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_traj(a, rows_a); _write_traj(b, rows_b)
    s = score_trajectories(a, b)
    assert 88.0 <= s["le1m_pct"] <= 92.0
    assert s["max_m"] > 1.5
