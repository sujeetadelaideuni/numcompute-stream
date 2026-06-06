"""
preprocessing.py — Data preprocessing with streaming support.

Extends the original NumCompute preprocessing module. All scalers and
transformers now support .partial_fit() for incremental chunk-wise updates
using Welford's online algorithm for running mean/variance.

Author: [Your Name]
Module: numcompute_stream.preprocessing
"""

from __future__ import annotations
import warnings
import numpy as np


class StandardScaler:
    """Per-feature standardisation: (X - mean) / scale.

    Supports both batch .fit() and streaming .partial_fit() using
    Welford's online algorithm for numerically stable mean/variance.

    Parameters
    ----------
    with_mean : bool
        Subtract mean if True (default).
    with_std : bool
        Divide by std if True (default).

    Examples
    --------
    Batch:
        >>> scaler = StandardScaler()
        >>> X_scaled = scaler.fit_transform(X_train)

    Streaming:
        >>> scaler = StandardScaler()
        >>> for chunk in chunks:
        ...     scaler.partial_fit(chunk)
        >>> X_scaled = scaler.transform(X_new)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples_seen_: int = 0
        self.n_features_: int | None = None

        # Welford state
        self._welford_count: int = 0
        self._welford_mean: np.ndarray | None = None
        self._welford_M2: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit scaler on the full dataset at once.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        self._reset()
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling to X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, same shape as X

        Raises
        ------
        RuntimeError if not fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        result = X.copy()
        if self.with_mean:
            result -= self.mean_
        if self.with_std:
            result /= self.scale_
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit then transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, same shape
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
        """Incrementally update scaler statistics with a new data chunk.

        Uses Welford's online algorithm for numerically stable updates.
        Can be called multiple times before calling .transform().

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New data chunk. NaN values are ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If feature count is inconsistent with previous calls.
        """
        X = self._validate(X)
        n_samples, n_features = X.shape

        if self._welford_mean is None:
            self.n_features_ = n_features
            self._welford_mean = np.zeros(n_features, dtype=float)
            self._welford_M2 = np.zeros(n_features, dtype=float)
            self._welford_count = 0
        elif n_features != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {n_features}."
            )

        # Welford update — vectorised over features
        for i in range(n_samples):
            row = X[i]
            valid = ~np.isnan(row)
            if not np.any(valid):
                continue
            self._welford_count += 1
            delta = np.where(valid, row - self._welford_mean, 0.0)
            self._welford_mean += delta / self._welford_count
            delta2 = np.where(valid, row - self._welford_mean, 0.0)
            self._welford_M2 += delta * delta2

        self.n_samples_seen_ = self._welford_count

        # Recompute exposed attributes
        self.mean_ = self._welford_mean.copy()
        if self._welford_count > 1:
            var = self._welford_M2 / self._welford_count
        else:
            var = np.zeros(n_features, dtype=float)

        std_arr = np.sqrt(var)
        zero_std = std_arr == 0.0
        if np.any(zero_std):
            warnings.warn(
                "Zero standard deviation detected; scale set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.scale_ = np.where(zero_std, 1.0, std_arr)

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        return X

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler is not fitted yet. Call .fit() or .partial_fit() first."
            )

    def _reset(self) -> None:
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        self.n_features_ = None
        self._welford_count = 0
        self._welford_mean = None
        self._welford_M2 = None


class MinMaxScaler:
    """Per-feature min-max scaling to a target range [min, max].

    Supports both batch .fit() and streaming .partial_fit() by tracking
    running global min/max across chunks.

    Parameters
    ----------
    feature_range : tuple of (float, float)
        Target output range. Default (0.0, 1.0).
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Fit on the full dataset."""
        X = self._validate(X)
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Incrementally update min/max with a new data chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        n_samples, n_features = X.shape

        if self.data_min_ is None:
            self.n_features_ = n_features
            self.data_min_ = np.full(n_features, np.inf)
            self.data_max_ = np.full(n_features, -np.inf)
        elif n_features != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {n_features}."
            )

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        self.data_min_ = np.minimum(self.data_min_, chunk_min)
        self.data_max_ = np.maximum(self.data_max_, chunk_max)

        span = self.data_max_ - self.data_min_
        self.scale_ = np.where(span == 0.0, 1.0, span)
        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to the target feature range."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - self.data_min_) / self.scale_
        X_scaled = X_std * (hi - lo) + lo
        constant = self.data_max_ == self.data_min_
        if np.any(constant):
            X_scaled[:, constant] = 0.0
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the min-max scaling."""
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        lo, hi = self.feature_range
        X_std = (X - lo) / (hi - lo)
        return X_std * self.scale_ + self.data_min_

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        return X

    def _check_fitted(self):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted yet.")

    def _reset(self):
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0


class SimpleImputer:
    """Fill NaNs with per-feature statistics or a constant.

    Supports streaming via .partial_fit() which maintains running estimates
    of mean/median using Welford (mean) or a reservoir for median.

    Parameters
    ----------
    strategy : str
        'mean', 'median', or 'constant'.
    fill_value : float
        Used when strategy='constant'.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.statistics_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0

        # Welford state for streaming mean
        self._welford_count: int = 0
        self._welford_mean: np.ndarray | None = None
        self._welford_M2: np.ndarray | None = None

        # For median: store all valid values per feature (bounded by window_size)
        self._reservoir: list[list] | None = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """Fit imputer on full dataset."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        """Incrementally update fill statistics with a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        n_samples, n_features = X.shape

        if self.n_features_ is None:
            self.n_features_ = n_features
            if self.strategy == "mean":
                self._welford_mean = np.zeros(n_features, dtype=float)
                self._welford_M2 = np.zeros(n_features, dtype=float)
                self._welford_count = 0
            elif self.strategy == "median":
                self._reservoir = [[] for _ in range(n_features)]
            self.statistics_ = np.zeros(n_features, dtype=float)
        elif n_features != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {n_features}."
            )

        if self.strategy == "constant":
            self.statistics_ = np.full(n_features, self.fill_value, dtype=float)

        elif self.strategy == "mean":
            for i in range(n_samples):
                row = X[i]
                for j in range(n_features):
                    v = row[j]
                    if not np.isnan(v):
                        self._welford_count += 1
                        delta = v - self._welford_mean[j]
                        self._welford_mean[j] += delta / self._welford_count
            self.statistics_ = np.where(
                np.isnan(self._welford_mean), 0.0, self._welford_mean
            )

        elif self.strategy == "median":
            for j in range(n_features):
                col = X[:, j]
                valid = col[~np.isnan(col)]
                self._reservoir[j].extend(valid.tolist())
            for j in range(n_features):
                if self._reservoir[j]:
                    self.statistics_[j] = float(np.median(self._reservoir[j]))
                else:
                    self.statistics_[j] = 0.0

        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN values using the fitted statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, same shape, no NaNs
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet.")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _reset(self):
        self.statistics_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._welford_count = 0
        self._welford_mean = None
        self._welford_M2 = None
        self._reservoir = None


class OneHotEncoder:
    """One-hot expansion for discrete columns.

    Supports streaming via .partial_fit() which incrementally discovers
    new categories in each chunk.

    Examples
    --------
    >>> enc = OneHotEncoder()
    >>> enc.partial_fit(chunk1)
    >>> enc.partial_fit(chunk2)   # may discover new categories
    >>> X_ohe = enc.transform(X)
    """

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Fit on full dataset."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        self.categories_ = None
        self.n_features_ = None
        return self.partial_fit(X)

    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Incrementally discover new categories from a chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        n_samples, n_features = X.shape

        if self.categories_ is None:
            self.n_features_ = n_features
            self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]
        else:
            if n_features != self.n_features_:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features_}, got {n_features}."
                )
            # Merge any new categories
            for j in range(n_features):
                new_cats = np.unique(X[:, j])
                self.categories_[j] = np.union1d(self.categories_[j], new_cats)

        self.n_samples_seen_ += n_samples
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode X using fitted categories.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, sum of category counts)
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")
        X = np.asarray(X)
        blocks: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            blocks.append((col[:, None] == cats[None, :]).astype(float))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
