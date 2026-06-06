from __future__ import annotations
import numpy as np


def mean(arr, axis=None):
    """Arithmetic mean, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
    axis : int or None

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=axis)


def median(arr, axis=None):
    """Median, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
    axis : int or None

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmedian(arr, axis=axis)


def std(arr, axis=None, ddof=0):
    """Standard deviation, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
    axis : int or None
    ddof : int

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=ddof)


def var(arr, axis=None, ddof=0):
    """Variance, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
    axis : int or None
    ddof : int

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanvar(arr, axis=axis, ddof=ddof)


def minimum(arr, axis=None):
    """Minimum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
    axis : int or None

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmin(arr, axis=axis)


def maximum(arr, axis=None):
    """Maximum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
    axis : int or None

    Returns
    -------
    np.ndarray or float
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmax(arr, axis=axis)


def summary(arr, axis=None):
    """Dictionary of descriptive statistics.

    Parameters
    ----------
    arr : array-like
    axis : int or None

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``count_nan``.
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

    Parameters
    ----------
    existing_aggregate : tuple of (int, float, float)
        Current state ``(count, mean, M2)``.
    new_value : float

    Returns
    -------
    tuple of (int, float, float)
        Updated ``(count, mean, M2)``.
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
    mean_val : float
    m2 : float
    ddof : int

    Returns
    -------
    tuple of (float, float)
        ``(mean, variance)``. Returns ``(nan, nan)`` when ``count <= ddof``.
    """
    if count == 0 or count <= ddof:
        return float("nan"), float("nan")
    return mean_val, m2 / (count - ddof)


def histogram(arr, bins=10, range=None):
    """Compute a histogram, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
    bins : int
    range : tuple or None

    Returns
    -------
    counts : np.ndarray, shape ``(bins,)``
    bin_edges : np.ndarray, shape ``(bins + 1,)``

    Raises
    ------
    ValueError
        If ``bins < 1``.
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
    q : float or array-like, values in ``[0, 1]``
    axis : int or None

    Returns
    -------
    np.ndarray or float

    Raises
    ------
    ValueError
        If any value in ``q`` is outside ``[0, 1]``.
    """
    arr = np.asarray(arr, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("All quantile values must be in [0, 1].")
    return np.nanpercentile(arr, q * 100, axis=axis)


class StreamingStats:
    """Per-feature streaming statistics using Welford's online algorithm.

    Parameters
    ----------
    window_size : int or None
        If set, only the last ``window_size`` chunks are kept for
        histogram and quantile estimates.

    Examples
    --------
    >>> ss = StreamingStats()
    >>> for chunk in chunks:
    ...     ss.update(chunk)
    >>> ss.get_mean()
    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None
        self._global_min: np.ndarray | None = None
        self._global_max: np.ndarray | None = None
        self._window_buffer: list[np.ndarray] = []
        self._window_total_rows: int = 0
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0

    def update(self, X_chunk: np.ndarray) -> "StreamingStats":
        """Incorporate a new chunk into the running statistics.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If feature count is inconsistent with previous chunks.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(1, -1)
        if X_chunk.ndim != 2:
            raise ValueError("X_chunk must be 1-D or 2-D.")

        n_samples, n_features = X_chunk.shape

        if self.n_features_ is None:
            self.n_features_ = n_features
            self._mean = np.zeros(n_features, dtype=float)
            self._M2 = np.zeros(n_features, dtype=float)
            self._global_min = np.full(n_features, np.inf)
            self._global_max = np.full(n_features, -np.inf)
        elif n_features != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {n_features}."
            )

        if n_samples == 0:
            return self

        for i in range(n_samples):
            row = X_chunk[i]
            valid = ~np.isnan(row)
            if not np.any(valid):
                continue
            self._count += 1
            delta = np.where(valid, row - self._mean, 0.0)
            self._mean += delta / self._count
            delta2 = np.where(valid, row - self._mean, 0.0)
            self._M2 += delta * delta2

        self._global_min = np.minimum(self._global_min, np.nanmin(X_chunk, axis=0))
        self._global_max = np.maximum(self._global_max, np.nanmax(X_chunk, axis=0))
        self.n_samples_seen_ += n_samples

        if self.window_size is not None:
            self._window_buffer.append(X_chunk.copy())
            self._window_total_rows += n_samples
            while (self._window_total_rows - self._window_buffer[0].shape[0]
                   >= self.window_size):
                removed = self._window_buffer.pop(0)
                self._window_total_rows -= removed.shape[0]

        return self

    def get_mean(self) -> np.ndarray:
        """Per-feature running mean.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        return self._mean.copy()

    def get_var(self, ddof: int = 0) -> np.ndarray:
        """Per-feature running variance.

        Parameters
        ----------
        ddof : int

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        if self._count <= ddof:
            return np.full(self.n_features_, np.nan)
        return self._M2 / (self._count - ddof)

    def get_std(self, ddof: int = 0) -> np.ndarray:
        """Per-feature running standard deviation.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        return np.sqrt(self.get_var(ddof=ddof))

    def get_min(self) -> np.ndarray:
        """Per-feature global minimum seen so far.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        return self._global_min.copy()

    def get_max(self) -> np.ndarray:
        """Per-feature global maximum seen so far.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        return self._global_max.copy()

    def get_histogram(self, feature_idx: int = 0, bins: int = 10):
        """Histogram for one feature from the window buffer.

        Parameters
        ----------
        feature_idx : int
        bins : int

        Returns
        -------
        counts : np.ndarray, shape (bins,)
        bin_edges : np.ndarray, shape (bins + 1,)

        Raises
        ------
        RuntimeError
            If window_size is not set or buffer is empty.
        """
        self._check_fitted()
        if self.window_size is None:
            raise RuntimeError(
                "Histogram requires window_size to be set."
            )
        if not self._window_buffer:
            raise RuntimeError("No data in window buffer yet.")
        data = np.concatenate(self._window_buffer, axis=0)
        return histogram(data[:, feature_idx], bins=bins)

    def get_quantile(self, q, feature_idx: int = 0):
        """Quantile for one feature from the window buffer.

        Parameters
        ----------
        q : float or array-like, values in [0, 1]
        feature_idx : int

        Returns
        -------
        float or np.ndarray

        Raises
        ------
        RuntimeError
            If window_size is not set or buffer is empty.
        """
        self._check_fitted()
        if self.window_size is None or not self._window_buffer:
            raise RuntimeError(
                "Quantile requires window_size to be set and data to be present."
            )
        data = np.concatenate(self._window_buffer, axis=0)
        return quantile(data[:, feature_idx], q)

    def summary_dict(self) -> dict:
        """Summary of all running statistics.

        Returns
        -------
        dict with keys: n_samples_seen, mean, std, min, max
        """
        self._check_fitted()
        return {
            "n_samples_seen": self.n_samples_seen_,
            "mean": self.get_mean(),
            "std":  self.get_std(),
            "min":  self.get_min(),
            "max":  self.get_max(),
        }

    def reset(self) -> "StreamingStats":
        """Reset all accumulated state.

        Returns
        -------
        self
        """
        self._count = 0
        self._mean = None
        self._M2 = None
        self._global_min = None
        self._global_max = None
        self._window_buffer = []
        self._window_total_rows = 0
        self.n_features_ = None
        self.n_samples_seen_ = 0
        return self

    def _check_fitted(self) -> None:
        if self.n_features_ is None:
            raise RuntimeError("StreamingStats has not received any data yet. Call .update() first.")
