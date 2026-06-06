"""
tree.py — Decision tree classifier with streaming support.

Implements a depth-limited decision tree from scratch using only NumPy.
Supports both batch training via .fit() and incremental growth via
.partial_fit() for streaming data scenarios.

Splitting criteria: Gini impurity or entropy (information gain).

Author: [Your Name]
Module: numcompute_stream.tree
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Internal node representation
# ---------------------------------------------------------------------------

class _Node:
    """A single node in the decision tree.

    Parameters
    ----------
    feature : int or None
        Feature index used for splitting. None if leaf.
    threshold : float or None
        Split threshold: go left if X[feature] <= threshold.
    left : _Node or None
    right : _Node or None
    value : int or None
        Predicted class label (only set for leaf nodes).
    depth : int
        Depth of this node in the tree.
    n_samples : int
        Number of training samples that reached this node.
    impurity : float
        Gini or entropy at this node before splitting.
    """

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


# ---------------------------------------------------------------------------
# Decision tree classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """Depth-limited decision tree for multi-class classification.

    Built entirely from scratch using NumPy. Supports Gini impurity and
    entropy as splitting criteria. Incremental growth is supported via
    .partial_fit() which appends new data and re-grows the tree from scratch
    on the accumulated dataset (a practical approach for moderate stream sizes).

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None = unlimited (not recommended
        for streaming — use a small value like 5).
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples in any leaf node.
    max_features : int, float, str, or None
        Number of features to consider at each split.
        int   → exact count
        float → fraction of total features
        'sqrt' → int(sqrt(n_features))
        'log2' → int(log2(n_features))
        None  → all features
    criterion : str
        'gini' or 'entropy'.
    random_state : int or None
        Seed for feature subsampling reproducibility.

    Examples
    --------
    Batch:
        >>> tree = DecisionTreeClassifier(max_depth=5)
        >>> tree.fit(X_train, y_train)
        >>> preds = tree.predict(X_test)

    Streaming:
        >>> tree = DecisionTreeClassifier(max_depth=5)
        >>> for chunk_X, chunk_y in stream:
        ...     tree.partial_fit(chunk_X, chunk_y)
        >>> preds = tree.predict(X_test)
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

        # Accumulated data for partial_fit re-growth
        self._X_accum: np.ndarray | None = None
        self._y_accum: np.ndarray | None = None

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        Accumulates the chunk, then re-grows the tree on all seen data.
        This ensures the tree structure reflects all history.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        classes : array-like or None
            All possible class labels. Useful when early chunks don't
            contain all classes.

        Returns
        -------
        self
        """
        X, y = self._validate(X, y)

        # Accumulate data
        if self._X_accum is None:
            self._X_accum = X.copy()
            self._y_accum = y.copy()
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"Feature count mismatch: expected {self.n_features_}, "
                    f"got {X.shape[1]}."
                )
            self._X_accum = np.vstack([self._X_accum, X])
            self._y_accum = np.concatenate([self._y_accum, y])

        # Update known classes
        if classes is not None:
            known = np.union1d(
                self.classes_ if self.classes_ is not None else np.array([]),
                np.asarray(classes)
            )
            self.classes_ = known
        self._setup_classes(self._y_accum)

        self.n_samples_seen_ += len(y)

        # Re-grow tree on all accumulated data
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
        RuntimeError if not fitted.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_}, "
                f"got {X.shape[1]}."
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
            Each row sums to 1.0.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_proba_single(x, self.root_) for x in X])

    # ------------------------------------------------------------------
    # Tree building internals
    # ------------------------------------------------------------------

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively build the decision tree.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n,)
        depth : int

        Returns
        -------
        _Node
        """
        n_samples = len(y)
        impurity = self._compute_impurity(y)
        node = _Node(depth=depth, n_samples=n_samples, impurity=impurity)

        # Stopping criteria
        should_stop = (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
            or impurity == 0.0
        )

        if should_stop:
            node.value = self._leaf_value(y)
            return node

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            # No valid split found
            node.value = self._leaf_value(y)
            return node

        # Partition data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Check min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = self._leaf_value(y)
            return node

        node.feature = feature
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the feature and threshold that minimise weighted impurity.

        Parameters
        ----------
        X : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n,)

        Returns
        -------
        best_feature : int or None
        best_threshold : float or None
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = self._compute_impurity(y)

        # Feature subsampling
        feature_indices = self._get_feature_indices(n_features)

        for feat in feature_indices:
            col = X[:, feat]
            # Handle NaN values — skip NaN rows for this feature
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 2:
                continue

            col_valid = col[valid_mask]
            y_valid = y[valid_mask]

            # Candidate thresholds: midpoints of sorted unique values
            unique_vals = np.unique(col_valid)
            if len(unique_vals) < 2:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            # Vectorised gain calculation over all thresholds
            gain = self._best_threshold_gain(
                col_valid, y_valid, thresholds, parent_impurity
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                # Find which threshold gave this gain
                best_threshold = self._find_best_threshold(
                    col_valid, y_valid, thresholds
                )

        return best_feature, best_threshold

    def _best_threshold_gain(
        self,
        col: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
        parent_impurity: float,
    ) -> float:
        """Return the maximum information gain over all thresholds for one feature."""
        n = len(y)
        best = -np.inf

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()

            if n_left == 0 or n_right == 0:
                continue

            imp_left = self._compute_impurity(y[left_mask])
            imp_right = self._compute_impurity(y[right_mask])
            weighted = (n_left * imp_left + n_right * imp_right) / n
            gain = parent_impurity - weighted

            if gain > best:
                best = gain

        return best

    def _find_best_threshold(
        self, col: np.ndarray, y: np.ndarray, thresholds: np.ndarray
    ) -> float:
        """Find the threshold with maximum gain."""
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
            imp_left = self._compute_impurity(y[left_mask])
            imp_right = self._compute_impurity(y[right_mask])
            gain = parent_impurity - (n_left * imp_left + n_right * imp_right) / n
            if gain > best_gain:
                best_gain = gain
                best_t = t

        return float(best_t)

    def _compute_impurity(self, y: np.ndarray) -> float:
        """Compute Gini impurity or entropy for label array y.

        Parameters
        ----------
        y : np.ndarray, shape (n,)

        Returns
        -------
        float
        """
        n = len(y)
        if n == 0:
            return 0.0
        # Count occurrences of each known class
        counts = np.array([np.sum(y == c) for c in self.classes_])
        probs = counts / n

        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:  # entropy
            # Avoid log(0)
            safe_probs = probs[probs > 0]
            return float(-np.sum(safe_probs * np.log2(safe_probs)))

    def _leaf_value(self, y: np.ndarray) -> int:
        """Return the majority class label for a leaf node.

        Ties broken by smallest class index.
        """
        counts = np.array([np.sum(y == c) for c in self.classes_])
        return self.classes_[np.argmax(counts)]

    def _leaf_proba(self, y: np.ndarray) -> np.ndarray:
        """Return class probability vector for a leaf node."""
        n = len(y)
        if n == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        return counts / n

    def _predict_single(self, x: np.ndarray, node: _Node):
        """Traverse the tree for one sample and return the predicted label."""
        if node.is_leaf:
            return node.value
        if np.isnan(x[node.feature]):
            # NaN handling: go left by default
            return self._predict_single(x, node.left)
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def _predict_proba_single(self, x: np.ndarray, node: _Node) -> np.ndarray:
        """Traverse and return probability vector for one sample."""
        if node.is_leaf:
            # Reconstruct probabilities from the leaf's accumulated training data
            # We store value (majority class); for proba we use the counts stored
            # in the node. For simplicity, return one-hot of predicted class.
            proba = np.zeros(self.n_classes_)
            class_idx = np.searchsorted(self.classes_, node.value)
            proba[class_idx] = 1.0
            return proba
        if np.isnan(x[node.feature]):
            return self._predict_proba_single(x, node.left)
        if x[node.feature] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        return self._predict_proba_single(x, node.right)

    # ------------------------------------------------------------------
    # Feature subsampling
    # ------------------------------------------------------------------

    def _get_feature_indices(self, n_features: int) -> np.ndarray:
        """Return the subset of feature indices to consider at each split."""
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
            raise ValueError(
                "max_features must be int, float, 'sqrt', 'log2', or None."
            )
        return self._rng.choice(n_features, size=k, replace=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be 1-D with shape (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}."
            )
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        return X, y

    def _setup_classes(self, y: np.ndarray) -> None:
        """Update known classes from label array."""
        new_classes = np.unique(y)
        if self.classes_ is None:
            self.classes_ = new_classes
        else:
            self.classes_ = np.union1d(self.classes_, new_classes)
        self.n_classes_ = len(self.classes_)

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier is not fitted yet. "
                "Call .fit() or .partial_fit() first."
            )

    def _reset(self) -> None:
        self.root_ = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._X_accum = None
        self._y_accum = None
