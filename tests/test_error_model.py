from data_pipeline.error_model import ErrorModel, fit_error_model
from data_pipeline.parsers import PosRow


def _row(q=1, ns=10, sd=0.3):
    return PosRow(utc_s=0.0, lat_deg=32.06, lon_deg=34.79, h_m=50.0,
                  quality=q, ns=ns, sd_n=sd, sd_e=sd, sd_u=sd * 2)


def test_float_bin_gets_larger_sigma_than_fix():
    # samples: (PosRow, horizontal_error_m) — Q=2 float rows carry 4x the error
    samples = ([( _row(q=1), 0.2) for _ in range(50)]
               + [(_row(q=2, sd=1.2), 0.8) for _ in range(50)])
    m = fit_error_model(samples)
    assert m.sigma_h(_row(q=2, sd=1.2)) > 2.0 * m.sigma_h(_row(q=1))


def test_unseen_bin_falls_back_to_global():
    samples = [(_row(q=1), 0.3) for _ in range(40)]
    m = fit_error_model(samples)
    s = m.sigma_h(_row(q=2, ns=3, sd=5.0))   # bin never seen
    assert s > 0 and s < 100                  # finite global fallback, no crash
