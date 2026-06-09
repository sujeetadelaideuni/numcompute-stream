"""
metrics.py — Classification metrics with streaming support.

Provides batch metric functions (accuracy, precision, recall, f1, AUC)
and a StreamingMetrics class that accumulates results chunk by chunk
via .update(), with rolling window and cumulative tracking support.

Author: Sujeet Ghosh
Module: numcompute_stream.metrics
"""
from __future__ import annotations
import numpy as np


def _validate_inputs(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D. "
            f"Got shapes {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}."
        )
    return y_true, y_pred


def accuracy(y_true, y_pred):
    """Proportion of correct predictions.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Build a (C x C) confusion matrix.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    np.ndarray, shape (C, C)
        ``cm[i, j]`` counts samples with true label ``i`` predicted as ``j``.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    true_idx = np.array([label_to_idx[l] for l in y_true])
    pred_idx = np.array([label_to_idx[l] for l in y_pred])
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def precision(y_true, y_pred, average="macro"):
    """Precision: TP / (TP + FP), per class then averaged.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str
        ``'macro'`` or ``'binary'``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are invalid or ``average='binary'`` with more than 2 classes.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fp = float(np.sum((y_pred == pos) & (y_true != pos)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(np.mean(per_class))


def recall(y_true, y_pred, average="macro"):
    """Recall: TP / (TP + FN), per class then averaged.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str
        ``'macro'`` or ``'binary'``.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are invalid or ``average='binary'`` with more than 2 classes.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fn = float(np.sum((y_pred != pos) & (y_true == pos)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(np.mean(per_class))


def f1(y_true, y_pred, average="macro"):
    """F1 score: harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str

    Returns
    -------
    float
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    """Mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true_binary, y_scores):
    """ROC curve for a binary classifier.

    Parameters
    ----------
    y_true_binary : array-like, shape (n,)
        Binary labels (0 or 1).
    y_scores : array-like, shape (n,)
        Scores for the positive class.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray

    Raises
    ------
    ValueError
        If inputs are not 1-D, differ in length, or are not binary.
    """
    y_true_binary, y_scores = _validate_inputs(y_true_binary, y_scores)
    unique_labels = np.unique(y_true_binary)
    if len(unique_labels) != 2:
        raise ValueError(
            f"roc_curve requires exactly 2 classes, found {len(unique_labels)}."
        )
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true_binary[desc_idx]
    thresholds = y_scores[desc_idx]
    total_pos = float(np.sum(y_true_binary == 1))
    total_neg = float(np.sum(y_true_binary == 0))
    tp_cumsum = np.cumsum(y_sorted == 1).astype(float)
    fp_cumsum = np.cumsum(y_sorted == 0).astype(float)
    tpr = tp_cumsum / total_pos if total_pos > 0 else np.zeros_like(tp_cumsum)
    fpr = fp_cumsum / total_neg if total_neg > 0 else np.zeros_like(fp_cumsum)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule.

    Parameters
    ----------
    fpr : array-like, shape (n,)
    tpr : array-like, shape (n,)

    Returns
    -------
    float
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.trapezoid(tpr, fpr))


class StreamingMetrics:
    """Accumulate classification metrics incrementally over streaming chunks.

    Parameters
    ----------
    n_classes : int or None
        Number of classes. Inferred from first ``.update()`` call if None.
    window_size : int or None
        If set, only the last ``window_size`` samples are used for rolling metrics.

    Examples
    --------
    >>> sm = StreamingMetrics()
    >>> for chunk_true, chunk_pred in stream:
    ...     sm.update(chunk_true, chunk_pred)
    >>> sm.result()
    """

    def __init__(
        self,
        n_classes: int | None = None,
        window_size: int | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.window_size = window_size
        self._cm: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._history: list[dict] = []
        self._window_true: list[np.ndarray] = []
        self._window_pred: list[np.ndarray] = []
        self._window_n: int = 0
        self.n_samples_seen_: int = 0
        self.n_chunks_seen_: int = 0

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingMetrics":
        """Incorporate a new chunk of predictions.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n,)
        y_pred_chunk : array-like, shape (n,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If inputs have mismatched lengths.
        """
        y_true = np.asarray(y_true_chunk).ravel()
        y_pred = np.asarray(y_pred_chunk).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}."
            )

        n = len(y_true)
        if n == 0:
            return self

        chunk_classes = np.unique(np.concatenate([y_true, y_pred]))
        if self._classes is None:
            self._classes = chunk_classes
            nc = len(self._classes)
            self._cm = np.zeros((nc, nc), dtype=np.int64)
        else:
            merged = np.union1d(self._classes, chunk_classes)
            if len(merged) > len(self._classes):
                new_nc = len(merged)
                new_cm = np.zeros((new_nc, new_nc), dtype=np.int64)
                old_idx = np.searchsorted(merged, self._classes)
                new_cm[np.ix_(old_idx, old_idx)] = self._cm
                self._cm = new_cm
                self._classes = merged

        label_to_idx = {label: idx for idx, label in enumerate(self._classes)}
        true_idx = np.array([label_to_idx[l] for l in y_true])
        pred_idx = np.array([label_to_idx[l] for l in y_pred])
        np.add.at(self._cm, (true_idx, pred_idx), 1)

        self._history.append({
            "chunk":    self.n_chunks_seen_,
            "n":        n,
            "accuracy": float(np.mean(y_true == y_pred)),
        })

        if self.window_size is not None:
            self._window_true.append(y_true.copy())
            self._window_pred.append(y_pred.copy())
            self._window_n += n
            while self._window_n - len(self._window_true[0]) >= self.window_size:
                removed = self._window_true.pop(0)
                self._window_pred.pop(0)
                self._window_n -= len(removed)

        self.n_samples_seen_ += n
        self.n_chunks_seen_ += 1
        return self

    def reset(self) -> "StreamingMetrics":
        """Clear all accumulated state.

        Returns
        -------
        self
        """
        self._cm = None
        self._classes = None
        self._history = []
        self._window_true = []
        self._window_pred = []
        self._window_n = 0
        self.n_samples_seen_ = 0
        self.n_chunks_seen_ = 0
        return self

    def result(self) -> dict:
        """Current cumulative metrics.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, confusion_matrix,
        n_samples_seen, n_chunks_seen
        """
        self._check_fitted()
        y_true, y_pred = self._reconstruct_from_cm()
        return {
            "accuracy":          self._cm_accuracy(),
            "precision":         precision(y_true, y_pred, average="macro"),
            "recall":            recall(y_true, y_pred, average="macro"),
            "f1":                f1(y_true, y_pred, average="macro"),
            "confusion_matrix":  self._cm.copy(),
            "n_samples_seen":    self.n_samples_seen_,
            "n_chunks_seen":     self.n_chunks_seen_,
        }

    def get_accuracy(self) -> float:
        """Cumulative accuracy across all seen chunks.

        Returns
        -------
        float
        """
        self._check_fitted()
        return self._cm_accuracy()

    def get_confusion_matrix(self) -> np.ndarray:
        """Cumulative confusion matrix.

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        return self._cm.copy()

    def get_accuracy_history(self) -> list[float]:
        """Per-chunk accuracy values in order of arrival.

        Returns
        -------
        list of float
        """
        return [h["accuracy"] for h in self._history]

    def get_rolling_accuracy(self) -> float:
        """Accuracy over the rolling window only.

        Returns
        -------
        float. Returns nan if window_size is None or window is empty.
        """
        if self.window_size is None or self._window_n == 0:
            return float("nan")
        y_true = np.concatenate(self._window_true)
        y_pred = np.concatenate(self._window_pred)
        return float(np.mean(y_true == y_pred))

    def get_auc(self, pos_label=None) -> float:
        """AUC for binary problems from the cumulative confusion matrix.

        Parameters
        ----------
        pos_label : scalar or None

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If not a binary problem.
        """
        self._check_fitted()
        if len(self._classes) != 2:
            raise ValueError("AUC from confusion matrix requires exactly 2 classes.")
        pos = pos_label if pos_label is not None else self._classes[-1]
        pos_idx = int(np.searchsorted(self._classes, pos))
        tp = float(self._cm[pos_idx, pos_idx])
        fn = float(np.sum(self._cm[pos_idx, :]) - tp)
        fp = float(np.sum(self._cm[:, pos_idx]) - tp)
        tn = float(np.sum(self._cm) - tp - fn - fp)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(0.5 * (tpr * (1 - fpr) + (1 + tpr) * fpr))

    def _check_fitted(self) -> None:
        if self._cm is None:
            raise RuntimeError(
                "StreamingMetrics has not received any data yet. Call .update() first."
            )

    def _cm_accuracy(self) -> float:
        total = float(self._cm.sum())
        if total == 0:
            return 0.0
        return float(np.trace(self._cm)) / total

    def _reconstruct_from_cm(self):
        rows, cols = np.nonzero(self._cm)
        counts = self._cm[rows, cols]
        y_true = np.repeat(self._classes[rows], counts)
        y_pred = np.repeat(self._classes[cols], counts)
        return y_true, y_pred
