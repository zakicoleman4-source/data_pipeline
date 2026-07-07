import numpy as np
from data_pipeline.parsers import PosRow
from data_pipeline.epoch_weight_v2 import smooth_epoch_weighted_v2, EpochWeightV2Options


def _straight_track(n=60, spike_at=30, spike_m=3.0):
    rows = []
    lat0, lon0 = 32.06, 34.79
    mlat = 111320.0
    mlon = mlat * np.cos(np.radians(lat0))
    for i in range(n):
        north_m = 5.0 * i
        east_m = spike_m if i == spike_at else 0.0
        rows.append(PosRow(
            utc_s=1000.0 + i,
            lat_deg=lat0 + north_m / mlat,
            lon_deg=lon0 + east_m / mlon,
            h_m=50.0, quality=2, ns=8,
            sd_n=0.3, sd_e=0.3, sd_u=0.6,
            vn=5.0, ve=0.0, vu=0.0,
        ))
    return rows


def test_innov_gate_suppresses_position_spike():
    rows = _straight_track()
    off = smooth_epoch_weighted_v2(rows, imu_rows=None,
        options=EpochWeightV2Options(innov_gate_enabled=False))
    on = smooth_epoch_weighted_v2(rows, imu_rows=None,
        options=EpochWeightV2Options(innov_gate_enabled=True, innov_gate_thresh=2.5,
                                     innov_gate_r_mult=10.0))
    # E_smooth is Local-frame east (m) about the first row; the spike is purely east.
    spike_off = abs(float(off.E_smooth[30]))
    spike_on = abs(float(on.E_smooth[30]))
    assert spike_on < spike_off          # gate pulled the spike toward the track
    assert spike_on < 1.0
