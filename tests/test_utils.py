import numpy as np
import pytest
from numcompute import utils

def test_euclidean_pairwise_shape() -> None:
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    d = utils.euclidean_distance(a, b)
    assert d.shape == (2, 3)
    np.testing.assert_allclose(d[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(d[1, 2], 0.0, atol=1e-12)


def test_cosine_zero_vector_returns_zero() -> None:
    assert utils.cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


def test_softmax_sums_to_one() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    s = utils.softmax(x, axis=-1)
    np.testing.assert_allclose(s.sum(axis=-1), np.ones(2))


def test_sigmoid_finite_for_large_magnitude() -> None:
    out = utils.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))


def test_logsumexp_matches_naive_stable() -> None:
    x = np.array([1000.0, 1001.0, 1002.0])
    direct = float(np.log(np.sum(np.exp(x - x.max()))) + x.max())
    assert np.isclose(utils.logsumexp(x), direct)


def test_batch_iter_sizes_and_shuffle() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    batches = list(utils.batch_iter(X, batch_size=3, shuffle=False))
    assert [b.shape[0] for b in batches] == [3, 3, 3, 1]
    rng = np.random.default_rng(0)
    shuffled = list(utils.batch_iter(X, batch_size=4, shuffle=True, rng=rng))
    assert sum(b.shape[0] for b in shuffled) == 10
