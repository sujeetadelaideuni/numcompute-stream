from __future__ import annotations
import warnings
import numpy as np


class StandardScaler:
    """Per-feature standardisation: ``(X - mean) / scale`` with safe division."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        if np.any(std == 0):
            warnings.warn(
                "Zero standard deviation detected; scale set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    """One-hot expansion for discrete columns (vectorised column blocks)."""

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> OneHotEncoder:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        X = np.asarray(X)
        blocks: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            blocks.append((col[:, None] == cats[None, :]).astype(float))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Per-feature min-max scaling to a target range."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        span = self.data_max_ - self.data_min_
        self.scale_ = np.where(span == 0.0, 1.0, span)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None or self.scale_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform.")
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


class SimpleImputer:
    """Fill NaNs with per-feature statistics or constants."""

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.statistics_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if self.strategy == "mean":
            stats = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            stats = np.nanmedian(X, axis=0)
        else:
            stats = np.full(X.shape[1], self.fill_value, dtype=float)
        self.statistics_ = np.where(np.isnan(stats), 0.0, stats)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform.")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
