"""
tests/test_numcompute_stream.py

35 unit tests covering all modules in numcompute_stream.
Includes standard functionality and edge cases (NaN, empty chunks,
single samples, zero variance, streaming scenarios).

Run with:  python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from numcompute_stream.stats import StreamingStats
from numcompute_stream.metrics import StreamingMetrics, accuracy, confusion_matrix
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


# ===========================================================================
# Helpers
# ===========================================================================

def make_classification_data(n=100, n_features=4, n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, n_classes, n)
    return X, y


def split_chunks(X, y, n_chunks=5):
    size = len(y) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * size
        end = (i + 1) * size if i < n_chunks - 1 else len(y)
        chunks.append((X[start:end], y[start:end]))
    return chunks


# ===========================================================================
# StreamingStats tests (7 tests)
# ===========================================================================

class TestStreamingStats:

    def test_mean_matches_numpy(self):
        """Running mean should match np.mean after all data seen."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        ss = StreamingStats()
        for chunk in np.array_split(X, 10):
            ss.update(chunk)
        np.testing.assert_allclose(ss.get_mean(), np.mean(X, axis=0), atol=1e-10)

    def test_std_matches_numpy(self):
        """Running std should match np.std after all chunks."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 4))
        ss = StreamingStats()
        for chunk in np.array_split(X, 8):
            ss.update(chunk)
        np.testing.assert_allclose(ss.get_std(), np.std(X, axis=0), atol=1e-8)

    def test_empty_chunk_ignored(self):
        """Empty chunk should not crash or change state."""
        X = np.random.standard_normal((50, 3))
        ss = StreamingStats()
        ss.update(X)
        mean_before = ss.get_mean().copy()
        ss.update(np.empty((0, 3)))
        np.testing.assert_array_equal(ss.get_mean(), mean_before)

    def test_nan_values_ignored(self):
        """NaN values in chunks should be ignored, not cause errors."""
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
        ss = StreamingStats()
        ss.update(X)
        m = ss.get_mean()
        assert not np.any(np.isnan(m))

    def test_min_max_tracking(self):
        """Global min/max should be tracked correctly across chunks."""
        X = np.array([[1.0, 10.0], [5.0, 2.0], [3.0, 8.0]])
        ss = StreamingStats()
        ss.update(X[:2])
        ss.update(X[2:])
        np.testing.assert_array_equal(ss.get_min(), np.nanmin(X, axis=0))
        np.testing.assert_array_equal(ss.get_max(), np.nanmax(X, axis=0))

    def test_feature_mismatch_raises(self):
        """Inconsistent feature counts across chunks should raise ValueError."""
        ss = StreamingStats()
        ss.update(np.ones((5, 3)))
        with pytest.raises(ValueError, match="Feature count mismatch"):
            ss.update(np.ones((5, 4)))

    def test_reset_clears_state(self):
        """After reset, stats should behave as if no data was seen."""
        ss = StreamingStats()
        ss.update(np.random.standard_normal((20, 2)))
        ss.reset()
        assert ss.n_features_ is None
        assert ss.n_samples_seen_ == 0


# ===========================================================================
# StreamingMetrics tests (6 tests)
# ===========================================================================

class TestStreamingMetrics:

    def test_cumulative_accuracy(self):
        """Cumulative accuracy should equal batch accuracy on same data."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 0, 2])
        sm = StreamingMetrics()
        sm.update(y_true[:3], y_pred[:3])
        sm.update(y_true[3:], y_pred[3:])
        expected = accuracy(y_true, y_pred)
        assert abs(sm.get_accuracy() - expected) < 1e-10

    def test_update_then_reset(self):
        """After reset, the state should be completely clear."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 2]), np.array([0, 1, 0]))
        sm.reset()
        assert sm.n_samples_seen_ == 0
        assert sm._cm is None

    def test_accuracy_history_length(self):
        """History list length should equal number of update calls."""
        sm = StreamingMetrics()
        for _ in range(5):
            sm.update(np.array([0, 1, 0]), np.array([0, 0, 0]))
        assert len(sm.get_accuracy_history()) == 5

    def test_empty_chunk_skipped(self):
        """Empty chunks should not increment chunk count."""
        sm = StreamingMetrics()
        sm.update(np.array([]), np.array([]))
        assert sm.n_chunks_seen_ == 0

    def test_result_contains_expected_keys(self):
        """result() dict should have required metric keys."""
        sm = StreamingMetrics()
        sm.update(np.array([0, 1, 2, 0]), np.array([0, 1, 0, 0]))
        r = sm.result()
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in r

    def test_rolling_window(self):
        """Rolling accuracy should reflect only the recent window."""
        sm = StreamingMetrics(window_size=4)
        sm.update(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))  # 0% acc
        sm.update(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))  # 100% acc
        rolling = sm.get_rolling_accuracy()
        # Rolling window covers latest 4 (all correct), so ~1.0
        assert rolling > 0.5


# ===========================================================================
# Preprocessing tests (7 tests)
# ===========================================================================

class TestPreprocessing:

    def test_standard_scaler_partial_fit(self):
        """partial_fit StandardScaler should produce ~zero mean after transform."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 3))
        scaler = StandardScaler()
        for chunk in np.array_split(X, 5):
            scaler.partial_fit(chunk)
        X_scaled = scaler.transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=0.1)

    def test_standard_scaler_zero_variance(self):
        """Constant features should not cause division by zero."""
        X = np.ones((10, 3))
        scaler = StandardScaler()
        scaler.fit(X)
        result = scaler.transform(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_minmax_scaler_partial_fit_range(self):
        """MinMaxScaler output should be within [0, 1] after partial_fit."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 3))
        scaler = MinMaxScaler()
        for chunk in np.array_split(X, 10):
            scaler.partial_fit(chunk)
        X_scaled = scaler.transform(X)
        assert X_scaled.min() >= -0.01
        assert X_scaled.max() <= 1.01

    def test_imputer_partial_fit_nan(self):
        """SimpleImputer partial_fit should fill NaNs with running mean."""
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[np.nan, 1.0], [5.0, np.nan]])
        imp = SimpleImputer(strategy="mean")
        imp.partial_fit(X1)
        imp.partial_fit(X2)
        X_filled = imp.transform(X2)
        assert not np.any(np.isnan(X_filled))

    def test_imputer_constant_strategy(self):
        """Constant imputer should fill NaNs with the specified value."""
        X = np.array([[np.nan, 1.0], [2.0, np.nan]])
        imp = SimpleImputer(strategy="constant", fill_value=-99.0)
        imp.fit(X)
        X_filled = imp.transform(X)
        assert X_filled[0, 0] == -99.0
        assert X_filled[1, 1] == -99.0

    def test_onehot_partial_fit_new_categories(self):
        """OneHotEncoder should expand when new categories arrive."""
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[0], [1]]))
        enc.partial_fit(np.array([[2]]))   # new category
        result = enc.transform(np.array([[0], [1], [2]]))
        assert result.shape[1] == 3  # 3 categories total

    def test_scaler_feature_mismatch_raises(self):
        """StandardScaler should raise on inconsistent feature count."""
        scaler = StandardScaler()
        scaler.partial_fit(np.random.randn(10, 3))
        with pytest.raises(ValueError):
            scaler.partial_fit(np.random.randn(10, 5))


# ===========================================================================
# DecisionTreeClassifier tests (6 tests)
# ===========================================================================

class TestDecisionTree:

    def test_fit_predict_basic(self):
        """Tree should achieve >50% accuracy on simple separable data."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 4))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert accuracy(y, preds) > 0.85

    def test_partial_fit_accumulates(self):
        """partial_fit should accumulate data and improve over chunks."""
        X, y = make_classification_data(n=120, n_classes=2)
        tree = DecisionTreeClassifier(max_depth=4)
        chunks = split_chunks(X, y, n_chunks=6)
        for cx, cy in chunks:
            tree.partial_fit(cx, cy)
        preds = tree.predict(X)
        assert accuracy(y, preds) > 0.4

    def test_single_sample_chunk(self):
        """partial_fit should handle a chunk with just one sample."""
        tree = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification_data(n=30, n_classes=2)
        tree.partial_fit(X[:1], y[:1])
        tree.partial_fit(X[1:10], y[1:10])
        preds = tree.predict(X[:5])
        assert preds.shape == (5,)

    def test_predict_with_nan(self):
        """predict should not crash when input contains NaN values."""
        X, y = make_classification_data(n=50, n_classes=2)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        X_test = X[:5].copy()
        X_test[0, 0] = np.nan
        preds = tree.predict(X_test)
        assert preds.shape == (5,)

    def test_max_depth_respected(self):
        """Tree depth should not exceed max_depth."""
        X, y = make_classification_data(n=100, n_classes=3)
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)

        def max_depth(node, d=0):
            if node is None or node.is_leaf:
                return d
            return max(max_depth(node.left, d + 1), max_depth(node.right, d + 1))

        assert max_depth(tree.root_) <= 2

    def test_not_fitted_predict_raises(self):
        """predict before fit should raise RuntimeError."""
        tree = DecisionTreeClassifier()
        with pytest.raises(RuntimeError):
            tree.predict(np.random.randn(5, 3))


# ===========================================================================
# EnsembleClassifier tests (5 tests)
# ===========================================================================

class TestEnsemble:

    def test_random_forest_better_than_single_tree(self):
        """Random Forest should generally match or beat a single tree."""
        X, y = make_classification_data(n=200, n_classes=3, seed=99)
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        tree = DecisionTreeClassifier(max_depth=4)
        tree.fit(X_train, y_train)

        rf = EnsembleClassifier(n_estimators=10, method="random_forest", max_depth=4)
        rf.fit(X_train, y_train)

        tree_acc = accuracy(y_test, tree.predict(X_test))
        rf_acc = accuracy(y_test, rf.predict(X_test))
        # RF should be at least as good (within 10%)
        assert rf_acc >= tree_acc - 0.1

    def test_partial_fit_streaming(self):
        """EnsembleClassifier should work incrementally over chunks."""
        X, y = make_classification_data(n=150, n_classes=2)
        clf = EnsembleClassifier(n_estimators=5, method="bagging", max_depth=3)
        chunks = split_chunks(X, y, n_chunks=5)
        for cx, cy in chunks:
            clf.partial_fit(cx, cy)
        preds = clf.predict(X)
        assert preds.shape == (150,)
        assert accuracy(y, preds) > 0.4

    def test_predict_proba_sums_to_one(self):
        """predict_proba rows should sum to 1."""
        X, y = make_classification_data(n=60, n_classes=3)
        clf = EnsembleClassifier(n_estimators=5, method="random_forest")
        clf.fit(X, y)
        proba = clf.predict_proba(X[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            EnsembleClassifier(method="boosting")

    def test_not_fitted_predict_raises(self):
        """predict before any fit should raise RuntimeError."""
        clf = EnsembleClassifier(n_estimators=3)
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(5, 3))


# ===========================================================================
# Pipeline tests (4 tests)
# ===========================================================================

class TestPipeline:

    def test_batch_fit_predict(self):
        """Batch pipeline should fit and predict without errors."""
        X, y = make_classification_data(n=80, n_classes=2)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   DecisionTreeClassifier(max_depth=3)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (80,)

    def test_partial_fit_streaming(self):
        """Pipeline partial_fit should chain correctly over chunks."""
        X, y = make_classification_data(n=100, n_classes=2)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(n_estimators=5)),
        ])
        for cx, cy in split_chunks(X, y, n_chunks=5):
            pipe.partial_fit(cx, cy)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_empty_steps_raises(self):
        """Pipeline with no steps should raise ValueError."""
        with pytest.raises(ValueError):
            Pipeline([])

    def test_predict_before_fit_raises(self):
        """predict before fit should raise RuntimeError."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier()),
        ])
        with pytest.raises(RuntimeError):
            pipe.predict(np.random.randn(5, 3))


# ===========================================================================
# StreamTrainer tests (4 tests)
# ===========================================================================

class TestStreamTrainer:

    def test_fit_score_chunk_logs_accuracy(self):
        """After fit and score, logs should contain chunk_accuracy."""
        X, y = make_classification_data(n=60, n_classes=2)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(max_depth=3)),
        ])
        trainer = StreamTrainer(pipeline=pipe, verbose=False)
        trainer.fit_chunk(X, y)
        trainer.score_chunk(X, y)
        logs = trainer.get_logs()
        assert "chunk_accuracy" in logs[0]

    def test_accuracy_history_grows_per_chunk(self):
        """Accuracy history should grow by one per scored chunk."""
        X, y = make_classification_data(n=100, n_classes=2)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(max_depth=3)),
        ])
        trainer = StreamTrainer(pipeline=pipe, verbose=False)
        chunks = split_chunks(X, y, n_chunks=4)
        for cx, cy in chunks:
            trainer.fit_score_chunk(cx, cy)
        assert len(trainer.get_accuracy_history()) == 4

    def test_score_before_fit_raises(self):
        """score_chunk before fit_chunk should raise RuntimeError."""
        X, y = make_classification_data(n=20, n_classes=2)
        pipe = Pipeline([("model", DecisionTreeClassifier())])
        trainer = StreamTrainer(pipeline=pipe, verbose=False)
        with pytest.raises(RuntimeError):
            trainer.score_chunk(X, y)

    def test_reset_clears_logs(self):
        """reset() should clear all logs and counters."""
        X, y = make_classification_data(n=40, n_classes=2)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier()),
        ])
        trainer = StreamTrainer(pipeline=pipe, verbose=False)
        trainer.fit_chunk(X, y)
        trainer.reset()
        assert trainer.n_chunks_trained_ == 0
        assert len(trainer.get_logs()) == 0


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
