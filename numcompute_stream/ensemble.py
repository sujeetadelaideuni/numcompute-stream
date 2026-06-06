"""
ensemble.py — Ensemble tree-based classifiers with streaming support.

Implements Bagging and Random Forest ensemble methods built from multiple
DecisionTreeClassifiers. Supports .partial_fit() for streaming adaptation
and .predict() via majority vote.

Author: [Your Name]
Module: numcompute_stream.ensemble
"""

from __future__ import annotations
import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Ensemble of decision trees supporting Bagging and Random Forest.

    Bagging (Bootstrap Aggregating):
        Each tree trains on a random bootstrap sample (sampling with
        replacement) of the data. All features are available to each tree.

    Random Forest:
        Same as Bagging, but each tree also uses a random feature subset
        at every split (max_features='sqrt' by default).

    The two methods differ only in how many features each tree considers
    at split time — set method='bagging' for full features, 'random_forest'
    for sqrt subsampling.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the ensemble.
    method : str
        'bagging' or 'random_forest'.
    max_depth : int or None
        Maximum depth for each tree.
    min_samples_split : int
        Minimum samples to split a node.
    min_samples_leaf : int
        Minimum samples in a leaf.
    max_features : int, float, str, or None
        Feature subsampling per split. If None and method='random_forest',
        defaults to 'sqrt'. Ignored by 'bagging' (uses all features).
    criterion : str
        'gini' or 'entropy'.
    bootstrap_fraction : float
        Fraction of samples to draw (with replacement) per tree per chunk.
        Default 1.0 (full bootstrap sample).
    random_state : int or None
        Seed for reproducibility.

    Examples
    --------
    Batch:
        >>> clf = EnsembleClassifier(n_estimators=10, method='random_forest')
        >>> clf.fit(X_train, y_train)
        >>> preds = clf.predict(X_test)

    Streaming:
        >>> clf = EnsembleClassifier(n_estimators=10, method='random_forest')
        >>> for chunk_X, chunk_y in stream:
        ...     clf.partial_fit(chunk_X, chunk_y)
        >>> preds = clf.predict(X_test)
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

        # Resolve max_features
        if max_features is not None:
            self.max_features = max_features
        elif method == "random_forest":
            self.max_features = "sqrt"
        else:
            self.max_features = None  # bagging uses all features

        self._rng = np.random.default_rng(random_state)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0
        self.n_chunks_seen_: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            X_boot, y_boot = self._bootstrap_sample(X, y)
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

        Each estimator receives a bootstrap sample from the current chunk.
        This allows the ensemble to adapt to new data without discarding
        previously learned structure.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None
            All possible class labels. Useful when early chunks are incomplete.

        Returns
        -------
        self
        """
        reset = not self._fitted
        X, y = self._validate(X, y, reset=reset)

        if classes is not None:
            all_cls = np.union1d(
                self.classes_ if self.classes_ is not None else np.array([]),
                np.asarray(classes)
            )
            self.classes_ = all_cls
        self._setup_classes(y)

        # First call: initialise trees
        if not self._fitted:
            self.estimators_ = []
            for i in range(self.n_estimators):
                self.estimators_.append(self._make_tree(seed=i))

        # Update each tree with a bootstrap sample from this chunk
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
        """Predict class labels via majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError if not fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])

        # Majority vote per sample
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
                cls_idx = np.searchsorted(self.classes_, p)
                proba_sum[i, cls_idx] += 1.0

        return proba_sum / self.n_estimators

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy of the ensemble on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_tree(self, seed: int = 0) -> DecisionTreeClassifier:
        """Create a new DecisionTreeClassifier with this ensemble's settings."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=int(self._rng.integers(0, 2**31)) + seed,
        )

    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a bootstrap sample from (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        seed_offset : int
            Added to rng state to get different samples per tree.

        Returns
        -------
        X_boot, y_boot : np.ndarray
        """
        n = len(y)
        k = max(1, int(n * self.bootstrap_fraction))
        rng = np.random.default_rng(
            int(self._rng.integers(0, 2**31)) + seed_offset
        )
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
                f"Feature count mismatch: expected {self.n_features_}, "
                f"got {X.shape[1]}."
            )
        return X, y

    def _setup_classes(self, y: np.ndarray) -> None:
        new_classes = np.unique(y)
        if self.classes_ is None:
            self.classes_ = new_classes
        else:
            self.classes_ = np.union1d(self.classes_, new_classes)

    def _check_fitted(self) -> None:
        if not self._fitted or len(self.estimators_) == 0:
            raise RuntimeError(
                "EnsembleClassifier is not fitted yet. "
                "Call .fit() or .partial_fit() first."
            )
