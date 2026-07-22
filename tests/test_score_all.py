from scripts.score_all import PAIRS, mean_metric


def test_pairs_defined():
    assert len(PAIRS) == 3
    assert [p[0] for p in PAIRS] == ["pair1_190336", "pair2_202751", "pair3_205044"]


def test_mean_metric_aggregates():
    rows = [{"max_m": 4.0, "two_sigma_m": 1.0}, {"max_m": 6.0, "two_sigma_m": 2.0}]
    assert mean_metric(rows, "max_m") == 5.0
    assert mean_metric(rows, "two_sigma_m") == 1.5
