import pytest

from anomaly_inspector.stats import linear_quantile


def test_linear_quantile_matches_numpy_definition():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    # sorted: 1 1 2 3 4 5 6 9; index (8-1)*0.95 = 6.65 -> 6 + .65*(9-6)
    assert linear_quantile(values, 0.95) == pytest.approx(7.95)
    assert linear_quantile(values, 0.0) == 1.0
    assert linear_quantile(values, 1.0) == 9.0
    assert linear_quantile([7.0], 0.95) == 7.0
