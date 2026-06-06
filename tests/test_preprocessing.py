import numpy as np
import pytest
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

def test_standard_scaler_mean_and_std() -> None:
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    out = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(np.mean(out, axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(np.std(out, axis=0), np.ones(2), atol=1e-12)


def test_standard_scaler_constant_column_no_nan() -> None:
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    out = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, 0], np.zeros(3))


def test_minmax_scaler_range_and_bounds() -> None:
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    out = MinMaxScaler().fit_transform(X)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    np.testing.assert_allclose(np.min(out, axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(np.max(out, axis=0), np.ones(2), atol=1e-12)


def test_onehot_output_shape_and_row_sums() -> None:
    X = np.array([[0, 1], [1, 1], [0, 2], [2, 2]])
    enc = OneHotEncoder()
    out = enc.fit_transform(X)
    total_unique = sum(len(c) for c in enc.categories_)
    assert out.shape == (X.shape[0], total_unique)
    np.testing.assert_allclose(np.sum(out, axis=1), np.full(X.shape[0], X.shape[1]))


def test_transform_before_fit_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((3, 2)))
