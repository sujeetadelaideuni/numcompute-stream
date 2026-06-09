"""
ensemble.py — Ensemble tree-based classifiers with streaming support.

Implements Bagging and Random Forest ensemble methods built from multiple
DecisionTreeClassifiers. Supports .partial_fit() for incremental adaptation
and majority-vote prediction across all estimators.

Author: Sujeet Ghosh
Module: numcompute_stream.ensemble
"""
from __future__ import annotations
import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Ensemble of decision trees supporting Bagging and Random Forest.

    Parameters
    ----------
    n_estimators : int
    method : str
        ``'bagging'`` or ``'random_forest'``.
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int, float, str, or None
        Overrides default feature subsampling. If ``None`` and
        ``method='random_forest'``, defaults to ``'sqrt'``.
    criterion : str
        ``'gini'`` or ``'entropy'``.
    bootstrap_fraction : float
        Fraction of samples drawn per tree per chunk. Default ``1.0``.
    random_state : int or None

    Examples
    --------
    >>> clf = EnsembleClassifier(n_estimators=10, method='random_forest')
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)

    >>> clf = EnsembleClassifier(n_estimators=10, method='random_forest')
    >>> for chunk_X, chunk_y in stream:
    ...     clf.partial_fit(chunk_X, chunk_y)
    """

    def __init__(
        self,
        n_estimators: int = 10,
        method: str = "random_forest",
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        criterion: str = "gini",
        bootstrap_fraction: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if method not in ("bagging", "random_forest"):
            raise ValueError("method must be 'bagging' or 'random_forest'.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if not (0.0 < bootstrap_fraction <= 1.0):
            raise ValueError("bootstrap_fraction must be in (0, 1].")

        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = random_state
        self.max_features = max_features if max_features is not None else ("sqrt" if method == "random_forest" else None)
        self._rng = np.random.default_rng(random_state)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0
        self.n_chunks_seen_: int = 0
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """Train the ensemble on the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate(X, y, reset=True)
        self._setup_classes(y)
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = self._make_tree(seed=i)
            X_boot, y_boot = self._bootstrap_sample(X, y, seed_offset=i)
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
        self.n_samples_seen_ = len(y)
        self.n_chunks_seen_ = 1
        self._fitted = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "EnsembleClassifier":
        """Incrementally update each tree with a new data chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X, y = self._validate(X, y, reset=not self._fitted)

        if classes is not None:
            self.classes_ = np.union1d(
                self.classes_ if self.classes_ is not None else np.array([]),
                np.asarray(classes)
            )
        self._setup_classes(y)

        if not self._fitted:
            self.estimators_ = [self._make_tree(seed=i) for i in range(self.n_estimators)]

        for i, tree in enumerate(self.estimators_):
            X_boot, y_boot = self._bootstrap_sample(X, y, seed_offset=i)
            if len(y_boot) == 0:
                continue
            tree.partial_fit(X_boot, y_boot, classes=self.classes_)

        self.n_samples_seen_ += len(y)
        self.n_chunks_seen_ += 1
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via majority vote.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If not fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        n_samples = X.shape[0]
        voted = np.empty(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            col = all_preds[:, i]
            classes, counts = np.unique(col, return_counts=True)
            voted[i] = classes[np.argmax(counts)]
        return voted

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities as average across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)
        for tree in self.estimators_:
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                proba_sum[i, np.searchsorted(self.classes_, p)] += 1.0
        return proba_sum / self.n_estimators

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _make_tree(self, seed: int = 0) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=int(self._rng.integers(0, 2**31)) + seed,
        )

    def _bootstrap_sample(self, X, y, seed_offset=0):
        n = len(y)
        k = max(1, int(n * self.bootstrap_fraction))
        rng = np.random.default_rng(int(self._rng.integers(0, 2**31)) + seed_offset)
        indices = rng.choice(n, size=k, replace=True)
        return X[indices], y[indices]

    def _validate(self, X, y, reset=False):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if y.ndim != 1:
            raise ValueError("y must be 1-D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if reset or self.n_features_ is None:
            self.n_features_ = X.shape[1]
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {X.shape[1]}."
            )
        return X, y

    def _setup_classes(self, y: np.ndarray) -> None:
        new_classes = np.unique(y)
        self.classes_ = new_classes if self.classes_ is None else np.union1d(self.classes_, new_classes)

    def _check_fitted(self) -> None:
        if not self._fitted or len(self.estimators_) == 0:
            raise RuntimeError(
                "EnsembleClassifier is not fitted yet. Call .fit() or .partial_fit() first."
            )
