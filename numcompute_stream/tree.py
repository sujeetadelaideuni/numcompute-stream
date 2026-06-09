"""
tree.py — Decision tree classifier with streaming support.

Implements a depth-limited decision tree from scratch using only NumPy.
Supports Gini impurity and entropy splitting criteria, NaN-safe prediction,
and incremental growth via .partial_fit() for streaming data scenarios.

Author: [Your Name]
Module: numcompute_stream.tree
"""
from __future__ import annotations
import numpy as np


class _Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        depth=0,
        n_samples=0,
        impurity=0.0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth
        self.n_samples = n_samples
        self.impurity = impurity

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """Depth-limited decision tree for multi-class classification.

    Parameters
    ----------
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int, float, str, or None
        ``int`` exact count, ``float`` fraction, ``'sqrt'``, ``'log2'``, or ``None`` for all.
    criterion : str
        ``'gini'`` or ``'entropy'``.
    random_state : int or None

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=5)
    >>> tree.fit(X_train, y_train)
    >>> preds = tree.predict(X_test)

    >>> tree = DecisionTreeClassifier(max_depth=5)
    >>> for chunk_X, chunk_y in stream:
    ...     tree.partial_fit(chunk_X, chunk_y)
    """

    def __init__(
        self,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_classes_: int = 0
        self.n_samples_seen_: int = 0
        self._X_accum: np.ndarray | None = None
        self._y_accum: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Train the tree on the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate(X, y)
        n_features = X.shape[1]
        self._reset()
        self.n_features_ = n_features
        self._setup_classes(y)
        self._X_accum = X.copy()
        self._y_accum = y.copy()
        self.n_samples_seen_ = len(y)
        self.root_ = self._grow_tree(X, y, depth=0)
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "DecisionTreeClassifier":
        """Incrementally grow the tree with a new data chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X, y = self._validate(X, y)

        if self._X_accum is None:
            self._X_accum = X.copy()
            self._y_accum = y.copy()
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features_}, got {X.shape[1]}."
                )
            self._X_accum = np.vstack([self._X_accum, X])
            self._y_accum = np.concatenate([self._y_accum, y])

        if classes is not None:
            self.classes_ = np.union1d(
                self.classes_ if self.classes_ is not None else np.array([]),
                np.asarray(classes)
            )
        self._setup_classes(self._y_accum)
        self.n_samples_seen_ += len(y)
        self.root_ = self._grow_tree(self._X_accum, self._y_accum, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

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
        ValueError
            If feature count does not match.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, got {X.shape[1]}."
            )
        return np.array([self._predict_single(x, self.root_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

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
        return np.array([self._predict_proba_single(x, self.root_) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)
        impurity = self._compute_impurity(y)
        node = _Node(depth=depth, n_samples=n_samples, impurity=impurity)

        # Stop growing if any stopping criterion is met
        should_stop = (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1  # pure node — all same class
            or impurity == 0.0
        )

        if should_stop:
            node.value = self._leaf_value(y)
            return node

        # Find the best feature and threshold to split on
        feature, threshold = self._best_split(X, y)

        if feature is None:
            # No valid split found — make this a leaf
            node.value = self._leaf_value(y)
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Enforce min_samples_leaf on both children
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = self._leaf_value(y)
            return node

        # Recurse on left and right partitions
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        parent_impurity = self._compute_impurity(y)

        # Only consider a random subset of features at each split (max_features)
        for feat in self._get_feature_indices(n_features):
            col = X[:, feat]
            # Skip features that are NaN for all or all-but-one sample
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 2:
                continue
            col_valid = col[valid_mask]
            y_valid = y[valid_mask]
            unique_vals = np.unique(col_valid)
            if len(unique_vals) < 2:
                continue
            # Candidate thresholds are midpoints between consecutive sorted unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            gain = self._best_threshold_gain(col_valid, y_valid, thresholds, parent_impurity)
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = self._find_best_threshold(col_valid, y_valid, thresholds)

        return best_feature, best_threshold

    def _best_threshold_gain(self, col, y, thresholds, parent_impurity):
        n = len(y)
        best = -np.inf
        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left == 0 or n_right == 0:
                continue
            gain = parent_impurity - (
                n_left * self._compute_impurity(y[left_mask]) +
                n_right * self._compute_impurity(y[right_mask])
            ) / n
            if gain > best:
                best = gain
        return best

    def _find_best_threshold(self, col, y, thresholds):
        n = len(y)
        parent_impurity = self._compute_impurity(y)
        best_gain = -np.inf
        best_t = thresholds[0]
        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left == 0 or n_right == 0:
                continue
            gain = parent_impurity - (
                n_left * self._compute_impurity(y[left_mask]) +
                n_right * self._compute_impurity(y[right_mask])
            ) / n
            if gain > best_gain:
                best_gain = gain
                best_t = t
        return float(best_t)

    def _compute_impurity(self, y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        # Compute class probabilities over all known classes
        probs = np.array([np.sum(y == c) for c in self.classes_]) / n
        if self.criterion == "gini":
            # Gini: 1 - sum(p_i^2), ranges from 0 (pure) to 1 - 1/C (maximally mixed)
            return float(1.0 - np.sum(probs ** 2))
        # Entropy: -sum(p_i * log2(p_i)), filter zero probs to avoid log(0)
        safe_probs = probs[probs > 0]
        return float(-np.sum(safe_probs * np.log2(safe_probs)))

    def _leaf_value(self, y: np.ndarray):
        counts = np.array([np.sum(y == c) for c in self.classes_])
        return self.classes_[np.argmax(counts)]

    def _predict_single(self, x: np.ndarray, node: _Node):
        if node.is_leaf:
            return node.value
        if np.isnan(x[node.feature]):
            return self._predict_single(x, node.left)
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def _predict_proba_single(self, x: np.ndarray, node: _Node) -> np.ndarray:
        if node.is_leaf:
            proba = np.zeros(self.n_classes_)
            proba[np.searchsorted(self.classes_, node.value)] = 1.0
            return proba
        if np.isnan(x[node.feature]):
            return self._predict_proba_single(x, node.left)
        if x[node.feature] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        return self._predict_proba_single(x, node.right)

    def _get_feature_indices(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError("max_features must be int, float, 'sqrt', 'log2', or None.")
        return self._rng.choice(n_features, size=k, replace=False)

    def _validate(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be 1-D with shape (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}."
            )
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        return X, y

    def _setup_classes(self, y: np.ndarray) -> None:
        new_classes = np.unique(y)
        if self.classes_ is None:
            self.classes_ = new_classes
        else:
            self.classes_ = np.union1d(self.classes_, new_classes)
        self.n_classes_ = len(self.classes_)

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier is not fitted yet. Call .fit() or .partial_fit() first."
            )

    def _reset(self) -> None:
        self.root_ = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._X_accum = None
        self._y_accum = None
