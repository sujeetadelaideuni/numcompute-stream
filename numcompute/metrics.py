from __future__ import annotations

import numpy as np


def _validate_inputs(y_true, y_pred):
    """Check that y_true and y_pred are 1-D arrays of equal length."""
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
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Build a (C x C) confusion matrix.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.

    Returns
    -------
    np.ndarray, shape (C, C)
        Integer matrix where ``cm[i, j]`` counts samples with true
        label *i* predicted as *j*.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n + C^2)
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
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        ``'macro'`` for unweighted mean of per-class precision.
        ``'binary'`` for positive-class precision (requires exactly 2 classes).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs invalid or ``average='binary'`` with more than 2 classes.

    Time complexity: O(n * C)
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
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        ``'macro'`` or ``'binary'``. Same semantics as :func:`precision`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs invalid or ``average='binary'`` with more than 2 classes.

    Time complexity: O(n * C)
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

    ``2 * P * R / (P + R)``. Returns 0.0 when both are zero.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        ``'macro'`` or ``'binary'``.

    Returns
    -------
    float

    Time complexity: O(n * C)
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    """Mean squared error for regression.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth values.
    y_pred : array-like, shape (n,)
        Predicted values.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true_binary, y_scores):
    """ROC curve for a binary classifier.

    Parameters
    ----------
    y_true_binary : array-like, shape (n,)
        Binary ground-truth labels (0 or 1).
    y_scores : array-like, shape (n,)
        Continuous scores for the positive class.

    Returns
    -------
    fpr : np.ndarray
        False positive rates, starting at 0.
    tpr : np.ndarray
        True positive rates, starting at 0.
    thresholds : np.ndarray
        Decreasing threshold values.

    Raises
    ------
    ValueError
        If inputs are not 1-D, different lengths, or not binary.

    Time complexity: O(n log n)
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
        False positive rates (x-axis).
    tpr : array-like, shape (n,)
        True positive rates (y-axis).

    Returns
    -------
    float
        AUC value. 1.0 = perfect, ~0.5 = random.

    Time complexity: O(n)
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.trapezoid(tpr, fpr))
