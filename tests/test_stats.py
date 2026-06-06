import numpy as np
import pytest
from numcompute.stats import (
    mean, median, std, var, minimum, maximum,
    summary, welford_update, welford_finalize,
    histogram, quantile,
)


def test_mean_basic():
    assert np.isclose(mean(np.array([1.0, 2.0, 3.0, 4.0])), 2.5)

def test_mean_nan_ignored():
    assert np.isclose(mean(np.array([1.0, 2.0, np.nan, 4.0])), 7.0 / 3.0)

def test_mean_all_nan():
    assert np.isnan(mean(np.array([np.nan, np.nan])))

def test_median_odd_and_even():
    assert np.isclose(median(np.array([3.0, 1.0, 2.0])), 2.0)
    assert np.isclose(median(np.array([1.0, 2.0, 3.0, 4.0])), 2.5)

def test_median_nan_ignored():
    assert np.isclose(median(np.array([1.0, np.nan, 3.0])), 2.0)

def test_std_known():
    assert np.isclose(std(np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)), 2.0)

def test_std_constant():
    assert std(np.array([5.0, 5.0, 5.0])) == 0.0

def test_var_equals_std_squared():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(var(arr), std(arr) ** 2)

def test_minimum_and_maximum():
    arr = np.array([3.0, 1.0, 4.0])
    assert np.isclose(minimum(arr), 1.0)
    assert np.isclose(maximum(arr), 4.0)

def test_minimum_maximum_nan_ignored():
    assert np.isclose(minimum(np.array([3.0, np.nan, 1.0])), 1.0)
    assert np.isclose(maximum(np.array([np.nan, 3.0, 1.0])), 3.0)

def test_summary_keys():
    assert set(summary(np.array([1.0, 2.0, 3.0])).keys()) == {"mean", "median", "std", "min", "max", "count_nan"}

def test_summary_nan_count():
    assert summary(np.array([1.0, np.nan, 3.0, np.nan]))["count_nan"] == 2

def test_welford_mean_and_variance():
    data = [1.0, 2.0, 3.0, 4.0]
    state = (0, 0.0, 0.0)
    for v in data:
        state = welford_update(state, v)
    m, wvar = welford_finalize(*state, ddof=0)
    assert np.isclose(m, 2.5)
    assert np.isclose(wvar, np.var(data))

def test_welford_empty():
    m, v = welford_finalize(0, 0.0, 0.0)
    assert np.isnan(m) and np.isnan(v)

def test_histogram_shape_and_total():
    counts, edges = histogram(np.array([1, 2, 3, 4, 5], dtype=float), bins=5)
    assert len(counts) == 5 and len(edges) == 6
    assert counts.sum() == 5

def test_histogram_nan_excluded():
    c1, _ = histogram(np.array([1.0, 2.0, 3.0]), bins=3)
    c2, _ = histogram(np.array([1.0, 2.0, 3.0, np.nan]), bins=3)
    assert np.array_equal(c1, c2)

def test_histogram_zero_bins_raises():
    with pytest.raises(ValueError):
        histogram(np.array([1.0]), bins=0)

def test_quantile_basic():
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    assert np.isclose(quantile(arr, 0.0), 1.0)
    assert np.isclose(quantile(arr, 0.5), 3.0)
    assert np.isclose(quantile(arr, 1.0), 5.0)

def test_quantile_out_of_range_raises():
    with pytest.raises(ValueError):
        quantile(np.array([1.0, 2.0]), 1.5)
    with pytest.raises(ValueError):
        quantile(np.array([1.0, 2.0]), -0.1)