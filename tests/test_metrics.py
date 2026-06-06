import numpy as np
import pytest
from numcompute.metrics import (
    accuracy, confusion_matrix, precision, recall, f1, mse,
    roc_curve, auc,
)


def test_accuracy_perfect():
    y = np.array([0, 1, 2, 1])
    assert accuracy(y, y) == 1.0

def test_accuracy_all_wrong():
    assert accuracy(np.array([0, 1, 2]), np.array([1, 2, 0])) == 0.0

def test_accuracy_partial():
    assert np.isclose(accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])), 0.75)

def test_accuracy_shape_mismatch():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 2]))

def test_accuracy_2d_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([[0, 1]]), np.array([[0, 1]]))


def test_confusion_matrix_binary_perfect():
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert cm[0, 0] == 2 and cm[1, 1] == 2

def test_confusion_matrix_all_wrong():
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    assert cm[0, 0] == 0 and cm[1, 1] == 0

def test_confusion_matrix_multiclass_shape():
    cm = confusion_matrix(np.array([0, 1, 2, 0, 1, 2]), np.array([0, 2, 1, 0, 0, 2]))
    assert cm.shape == (3, 3) and cm.sum() == 6

def test_confusion_matrix_diagonal_sum():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 0, 2, 2])
    cm = confusion_matrix(y_true, y_pred)
    assert np.trace(cm) == np.sum(y_true == y_pred)


def test_precision_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert np.isclose(precision(y, y), 1.0)

def test_precision_macro_manual():
    # Class 0 pred at idx 0,1,3,4: TP=2, FP=2 -> P=0.5
    # Class 1 pred at idx 2,5:     TP=1, FP=1 -> P=0.5
    # Class 2 pred at none:         TP=0, FP=0 -> P=0.0
    # macro = (0.5 + 0.5 + 0.0) / 3 = 0.333...
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    assert np.isclose(precision(y_true, y_pred, average="macro"), 1 / 3)

def test_precision_binary():
    assert np.isclose(
        precision(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), average="binary"),
        2 / 3,
    )

def test_precision_binary_multiclass_raises():
    with pytest.raises(ValueError):
        precision(np.array([0, 1, 2]), np.array([0, 1, 2]), average="binary")


def test_recall_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert np.isclose(recall(y, y), 1.0)

def test_recall_binary():
    assert np.isclose(
        recall(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), average="binary"),
        0.5,
    )

def test_recall_all_wrong():
    assert np.isclose(recall(np.array([0, 0]), np.array([1, 1])), 0.0)


def test_f1_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert np.isclose(f1(y, y), 1.0)

def test_f1_zero():
    assert f1(np.array([0, 0]), np.array([1, 1])) == 0.0

def test_f1_binary_known():
    # P=2/3, R=1.0 -> F1=0.8
    assert np.isclose(
        f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), average="binary"),
        0.8,
    )

def test_f1_bounded():
    rng = np.random.default_rng(42)
    result = f1(rng.integers(0, 3, 500), rng.integers(0, 3, 500))
    assert 0.0 <= result <= 1.0


def test_mse_perfect():
    assert mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0

def test_mse_known():
    assert np.isclose(mse(np.array([1, 2, 3], dtype=float), np.array([2, 3, 4], dtype=float)), 1.0)

def test_mse_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.random(50), rng.random(50)
    assert np.isclose(mse(a, b), mse(b, a))

def test_mse_shape_mismatch():
    with pytest.raises(ValueError):
        mse(np.array([1.0, 2.0]), np.array([1.0]))

def test_mse_2d_raises():
    with pytest.raises(ValueError):
        mse(np.array([[1.0]]), np.array([[1.0]]))


def test_roc_curve_perfect():
    fpr, tpr, _ = roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert np.isclose(auc(fpr, tpr), 1.0)

def test_roc_curve_origin():
    fpr, tpr, _ = roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert fpr[0] == 0.0 and tpr[0] == 0.0

def test_roc_curve_non_binary_raises():
    with pytest.raises(ValueError):
        roc_curve(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

def test_roc_curve_worst_classifier():
    fpr, tpr, _ = roc_curve(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert auc(fpr, tpr) < 0.1

def test_roc_curve_random():
    rng = np.random.default_rng(0)
    fpr, tpr, _ = roc_curve(rng.integers(0, 2, 1000), rng.random(1000))
    assert 0.4 < auc(fpr, tpr) < 0.6


def test_auc_rectangle():
    assert np.isclose(auc(np.array([0.0, 1.0]), np.array([1.0, 1.0])), 1.0)
 
def test_auc_triangle():
    assert np.isclose(auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])), 0.5)

def test_auc_zero():
    assert np.isclose(auc(np.array([0.0, 1.0]), np.array([0.0, 0.0])), 0.0)

def test_recall_imbalanced():
    y_true = np.array([0] * 98 + [1] * 2)
    y_pred = np.array([0] * 100)
    assert np.isclose(recall(y_true, y_pred, average="macro"), 0.5)

def test_mse_constant_offset():
    y = np.arange(10.0)
    assert np.isclose(mse(y, y + 3.0), 9.0)