from __future__ import annotations
from collections.abc import Generator
import numpy as np

def euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> np.floating | np.ndarray:
    """Euclidean (L2) distance with broadcasting for pairwise rows.

    Parameters
    ----------
    a
        Array of shape ``(d,)`` or ``(n, d)``.
    b
        Array of shape ``(d,)`` or ``(m, d)``. When both are 2-D, trailing dimension ``d`` must match.

    Returns
    -------
    float or np.ndarray
        Scalar distance for two vectors, or an ``(n, m)`` matrix of pairwise distances.

    Raises
    ------
    ValueError
        If trailing dimensions do not match or shapes are invalid.

    Notes
    -----
    Uses ``sqrt(sum((a_i - b_j)^2))`` without Python loops over elements (vectorised broadcasting).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1 and b.ndim == 1:
        if a.shape != b.shape:
            raise ValueError("1-D inputs must have the same shape.")
        return float(np.sqrt(np.sum((a - b) ** 2)))
    if a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[1]:
            raise ValueError("Incompatible feature dimensions for pairwise distances.")
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))
    raise ValueError("Inputs must be both 1-D or both 2-D (n, d) and (m, d).")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Parameters
    ----------
    a, b
        One-dimensional arrays of identical shape ``(d,)``.

    Returns
    -------
    float
        ``dot(a, b) / (||a|| * ||b||)``. Returns ``0.0`` if either vector has zero norm.

    Raises
    ------
    ValueError
        If inputs are not 1-D or shapes differ.

    Notes
    -----
    Time complexity ``O(d)``.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("cosine_similarity expects two 1-D arrays of equal length.")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis`` using the max-shift trick.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which probabilities sum to ``1``.

    Returns
    -------
    np.ndarray
        Same shape as ``x``; non-negative entries summing to ``1`` along ``axis``.

    Notes
    -----
    Time and space complexity ``O(N)`` for ``N`` elements of ``x``.
    """
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Elementwise logistic sigmoid with clipping to reduce overflow.

    Parameters
    ----------
    x
        Any shape; values are clipped to ``[-60, 60]`` before ``exp``.

    Returns
    -------
    np.ndarray
        ``1 / (1 + exp(-x))`` with the same shape as ``x``.

    Notes
    -----
    Time complexity ``O(N)``.
    """
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-xc))


def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable ``log(sum(exp(x)))`` along ``axis`` (max-shift trick).

    Parameters
    ----------
    x
        Input array.
    axis
        Axis or ``None`` to reduce the entire array to a scalar.

    Returns
    -------
    np.ndarray
        Log-sum-exp values; shape follows standard NumPy reduction rules.

    Notes
    -----
    Time complexity ``O(N)``. Equivalent to ``scipy.special.logsumexp`` for comparable ``axis``.
    """
    x = np.asarray(x, dtype=float)
    if axis is None:
        m = np.max(x)
        return np.asarray(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def batch_iter(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield contiguous row batches from a 2-D array.

    Parameters
    ----------
    X
        Array of shape ``(n_samples, n_features)``.
    batch_size
        Maximum rows per batch (last batch may be smaller).
    shuffle
        If ``True``, shuffle row order before batching (uses ``rng`` or a new default generator).
    rng
        Optional :class:`numpy.random.Generator` for reproducible shuffling.

    Yields
    ------
    np.ndarray
        Slices of shape ``(<= batch_size, n_features)``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D or ``batch_size`` < ``1``.

    Notes
    -----
    Time complexity ``O(n_samples * n_features)`` for a full pass; extra ``O(n_samples log n_samples)``
    when ``shuffle`` is ``True``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        gen = rng if rng is not None else np.random.default_rng()
        gen.shuffle(idx)
    for start in range(0, n, batch_size):
        yield np.asarray(X[idx[start : start + batch_size]])
