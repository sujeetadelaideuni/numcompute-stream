import numpy as np
import pytest
from numcompute.rank import rank, percentile


def test_rank_average():
    result = rank(np.array([3, 1, 2, 3]), method='average')
    assert np.array_equal(result, [3.5, 1.0, 2.0, 3.5])


def test_rank_dense():
    result = rank(np.array([3, 1, 2, 3]), method='dense')
    assert np.array_equal(result, [3.0, 1.0, 2.0, 3.0])


def test_rank_ordinal():
    result = rank(np.array([3, 1, 2, 3]), method='ordinal')
    assert np.array_equal(result, [3.0, 1.0, 2.0, 4.0])


def test_rank_nan():
    result = rank(np.array([3.0, np.nan, 1.0]))
    assert np.isnan(result[1])
    assert result[0] == 2.0
    assert result[2] == 1.0


def test_rank_invalid_method():
    with pytest.raises(ValueError):
        rank(np.array([1, 2, 3]), method='wrong')


def test_percentile_median():
    result = percentile(np.array([1, 2, 3, 4, 5]), 50)
    assert result == 3.0


def test_percentile_min_max():
    data = np.array([1, 2, 3, 4, 5])
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 5.0


def test_percentile_nan_ignored():
    result = percentile(np.array([1.0, np.nan, 3.0, 5.0]), 50)
    assert result == 3.0
