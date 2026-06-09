from __future__ import annotations

import numpy as np


def mean(arr, axis=None):
    """Arithmetic mean, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis along which the mean is computed. None flattens first.

    Returns
    -------
    np.ndarray or float
        Mean value(s). Shape is ``arr.shape`` with ``axis`` removed.

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=axis)


def median(arr, axis=None):
    """Median, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis along which the median is computed.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmedian(arr, axis=axis)


def std(arr, axis=None, ddof=0):
    """Standard deviation, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Reduction axis.
    ddof : int
        Delta degrees of freedom. 0 for population, 1 for sample.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=ddof)


def var(arr, axis=None, ddof=0):
    """Variance, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Reduction axis.
    ddof : int
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanvar(arr, axis=axis, ddof=ddof)


def minimum(arr, axis=None):
    """Minimum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmin(arr, axis=axis)


def maximum(arr, axis=None):
    """Maximum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmax(arr, axis=axis)


def summary(arr, axis=None):
    """Dictionary of descriptive statistics for quick exploration.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Axis along which stats are computed. None reduces the whole array.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``count_nan``.

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return {
        "mean":      np.nanmean(arr, axis=axis),
        "median":    np.nanmedian(arr, axis=axis),
        "std":       np.nanstd(arr, axis=axis),
        "min":       np.nanmin(arr, axis=axis),
        "max":       np.nanmax(arr, axis=axis),
        "count_nan": int(np.sum(np.isnan(arr))),
    }


def welford_update(existing_aggregate, new_value):
    """One-step Welford online variance update.

    Takes ``(count, mean, M2)`` and a new scalar, returns updated
    ``(count, mean, M2)``. Allows computing variance in a single
    streaming pass without storing all values.

    Parameters
    ----------
    existing_aggregate : tuple of (int, float, float)
        Current state ``(count, mean, M2)``.
    new_value : float
        Next observation.

    Returns
    -------
    tuple of (int, float, float)
        Updated ``(count, mean, M2)``.

    Time complexity: O(1)
    """
    count, mean_val, m2 = existing_aggregate
    count += 1
    delta = new_value - mean_val
    mean_val += delta / count
    delta2 = new_value - mean_val
    m2 += delta * delta2
    return count, mean_val, m2


def welford_finalize(count, mean_val, m2, ddof=0):
    """Finalise Welford accumulator into ``(mean, variance)``.

    Parameters
    ----------
    count : int
        Total observations processed.
    mean_val : float
        Running mean.
    m2 : float
        Sum of squared deviations.
    ddof : int
        0 for population, 1 for sample variance.

    Returns
    -------
    tuple of (float, float)
        ``(mean, variance)``. Returns ``(nan, nan)`` when ``count <= ddof``.

    Time complexity: O(1)
    """
    if count == 0 or count <= ddof:
        return float("nan"), float("nan")
    variance = m2 / (count - ddof)
    return mean_val, variance


def histogram(arr, bins=10, range=None):
    """Compute a histogram, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data (flattened internally).
    bins : int
        Number of equal-width bins. Must be >= 1.
    range : tuple of (float, float) or None
        Bin range. None uses ``(nanmin, nanmax)``.

    Returns
    -------
    counts : np.ndarray, shape ``(bins,)``
    bin_edges : np.ndarray, shape ``(bins + 1,)``

    Raises
    ------
    ValueError
        If ``bins < 1``.

    Time complexity: O(n)
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    arr = np.asarray(arr, dtype=float)
    clean = arr[~np.isnan(arr)].ravel()
    return np.histogram(clean, bins=bins, range=range)


def quantile(arr, q, axis=None):
    """Compute quantile(s) along a given axis, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    q : float or array-like
        Quantile(s) in ``[0, 1]``.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Raises
    ------
    ValueError
        If any value in ``q`` is outside ``[0, 1]``.

    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("All quantile values must be in [0, 1].")
    return np.nanpercentile(arr, q * 100, axis=axis)
