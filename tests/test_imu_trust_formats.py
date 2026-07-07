# tests/test_imu_trust_formats.py
from data_pipeline.parsers import parse_imu


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_imu_legacy_7col(tmp_path):
    # GPS_s, gx,gy,gz, ax,ay,az  (day12)
    f = _write(tmp_path, "sensors_legacy.txt",
               "1462018987.003,-0.11,-0.39,0.10,-1.59,7.77,5.87\n"
               "1462018987.008,-0.12,-0.40,0.10,-1.60,7.78,5.86\n")
    rows = parse_imu(f)
    assert len(rows) == 2
    r = rows[0]
    assert abs(r.gx + 0.11) < 1e-6           # rate sensor from cols 1-3
    assert 8.0 < (r.ax**2 + r.ay**2 + r.az**2) ** 0.5 < 12.0  # linear sensor ~ g


def test_parse_imu_new_13col(tmp_path):
    # GPS_s, gx,gy,gz, lin(3), ax,ay,az, mag(3)  (day14)
    f = _write(tmp_path, "sensors_new.txt",
               "1466698156.005,-0.006,0.014,0.001,-0.007,0.012,0.005,9.61,0.16,2.09,-0.08,0.1,0.2\n"
               "1466698156.010,-0.006,0.014,0.001,-0.007,0.012,0.005,9.60,0.17,2.10,-0.08,0.1,0.2\n")
    rows = parse_imu(f)
    assert len(rows) == 2
    r = rows[0]
    assert abs(r.gx + 0.006) < 1e-6          # rate sensor from cols 1-3
    assert 8.0 < (r.ax**2 + r.ay**2 + r.az**2) ** 0.5 < 12.0  # linear sensor WITH gravity (cols 7-9)
