import numpy as np
from data_pipeline.consistency import nees


def test_nees_consistent_covariance():
    rng = np.random.default_rng(0)
    C = np.array([[0.25, 0.0], [0.0, 0.25]])
    L = np.linalg.cholesky(C)
    errs = (L @ rng.standard_normal((2, 2000))).T
    covs = np.broadcast_to(C, (2000, 2, 2))
    r = nees(errs, covs)
    assert 1.7 < r["mean_nees"] < 2.3
    assert r["consistent"] is True


def test_nees_flags_overconfident_covariance():
    rng = np.random.default_rng(1)
    errs = rng.standard_normal((2000, 2)) * 1.0
    covs = np.broadcast_to(np.eye(2) * 0.01, (2000, 2, 2))
    r = nees(errs, covs)
    assert r["mean_nees"] > 10
    assert r["consistent"] is False
