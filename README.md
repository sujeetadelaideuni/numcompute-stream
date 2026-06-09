# NumCompute-Stream

A streaming, decision tree–based machine learning framework extending the NumCompute package. Implements incremental learning, ensemble methods, and real-time visualisation using only NumPy and matplotlib.

**Assignment 2.2 — Individual Project**

---

## Overview

NumCompute-Stream extends the original NumCompute package (Assignment 2.1) with a full streaming ML framework. All new components support chunk-wise updates via `.partial_fit()`, simulating a real-world online learning scenario where data arrives incrementally.

The framework supports:
- Single decision trees and ensemble methods (Bagging, Random Forest)
- Incremental preprocessing with running statistics via Welford's algorithm
- Streaming metrics and statistics with rolling window support
- A fully chained pipeline with streaming compatibility
- Built-in visualisation for monitoring model performance over time

---

## Project Structure

```
numcompute-stream-solo/
│
├── numcompute/                    # Original NumCompute package (Assignment 2.1)
│   ├── __init__.py
│   ├── io.py                      # CSV loading and saving
│   ├── utils.py                   # Distance, similarity, activation functions
│   ├── stats.py                   # Batch statistical functions
│   ├── metrics.py                 # Batch classification metrics
│   ├── preprocessing.py           # Batch scalers and encoders
│   ├── pipeline.py                # Batch pipeline and FeatureUnion
│   ├── benchmarking.py            # Timing and benchmarking utilities
│   ├── rank.py                    # Ranking functions
│   ├── sort_search.py             # Sorting and searching
│   └── optim.py                   # Gradient and Jacobian estimation
│
├── numcompute_stream/             # Streaming ML framework (Assignment 2.2)
│   ├── __init__.py
│   ├── stats.py                   # StreamingStats with Welford's algorithm
│   ├── metrics.py                 # StreamingMetrics with rolling window
│   ├── preprocessing.py           # Scalers and imputers with partial_fit
│   ├── tree.py                    # DecisionTreeClassifier from scratch
│   ├── ensemble.py                # EnsembleClassifier (Bagging, Random Forest)
│   ├── pipeline.py                # Streaming Pipeline and FeatureUnion
│   ├── stream.py                  # StreamTrainer for chunk-wise training
│   └── visualise.py               # Plotting utilities for streaming metrics
│
├── tests/
│   ├── test_numcompute_stream.py  # 39 unit tests for streaming modules
│   ├── test_benchmarking.py       # Tests for benchmarking utilities
│   ├── test_io.py                 # Tests for I/O functions
│   ├── test_metrics.py            # Tests for batch metrics
│   ├── test_optim.py              # Tests for optimisation functions
│   ├── test_pipeline.py           # Tests for batch pipeline
│   ├── test_preprocessing.py      # Tests for batch preprocessing
│   ├── test_rank.py               # Tests for ranking functions
│   ├── test_sort_search.py        # Tests for sort and search
│   ├── test_stats.py              # Tests for batch statistics
│   ├── test_utils.py              # Tests for utility functions
│   └── __init__.py
│
├── benchmark/
│   └── benchmark_runner.py        # Batch vs vectorised benchmarks (Assignment 2.1)
│
├── benchmark_stream/              # Streaming benchmarks (Assignment 2.2)
│   └── stream_benchmark.py        # Batch fit vs streaming partial_fit timing
│
├── demo/
│   └── quickstart.ipynb           # Original quickstart notebook (Assignment 2.1)
│
├── demo_stream/                   # Streaming demo (Assignment 2.2)
│   ├── stream_demo.ipynb          # Full streaming demo notebook
│   ├── iris_stream.csv            # Demo dataset (300 samples, 3 classes)
│   ├── stream_benchmark.py        # Batch vs streaming timing benchmark
│   ├── tree_accuracy.png          # Decision tree accuracy over chunks
│   ├── model_comparison.png       # Decision tree vs random forest comparison
│   ├── predictions_vs_truth.png   # Predictions vs ground truth plot
│   ├── confusion_matrix.png       # Cumulative confusion matrix
│   ├── fit_times.png              # Training time per chunk
│   └── cumulative_accuracy.png    # Cumulative accuracy over chunks
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python 3.11+
- NumPy
- matplotlib

No other external libraries required. scikit-learn, pandas, and PyTorch are not used.

```bash
pip install numpy matplotlib
```

---

## Quick Start

### Streaming Pipeline

```python
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler()),
    ('model',   EnsembleClassifier(n_estimators=10, method='random_forest')),
])

trainer = StreamTrainer(pipeline=pipe, verbose=True)

for X_chunk, y_chunk in data_stream:
    trainer.fit_chunk(X_chunk, y_chunk)
    trainer.score_chunk(X_chunk, y_chunk)

print(trainer.summary())
```

### Load CSV and Stream Chunks

```python
from numcompute.io import load_csv_chunked

for chunk in load_csv_chunked('data.csv', chunksize=30):
    X_chunk = chunk[:, :-1]
    y_chunk = chunk[:, -1].astype(int)
    trainer.fit_chunk(X_chunk, y_chunk)
    trainer.score_chunk(X_chunk, y_chunk)
```

### Streaming Statistics

```python
from numcompute_stream.stats import StreamingStats

ss = StreamingStats(window_size=100)
for chunk in chunks:
    ss.update(chunk)

print(ss.get_mean())
print(ss.get_std())
```

### Streaming Metrics

```python
from numcompute_stream.metrics import StreamingMetrics

sm = StreamingMetrics()
for y_true_chunk, y_pred_chunk in stream:
    sm.update(y_true_chunk, y_pred_chunk)

print(sm.result())
print(sm.get_accuracy_history())
```

### Visualisation

```python
from numcompute_stream import visualise

visualise.plot_metric_over_time(
    trainer.get_accuracy_history(),
    title='Accuracy over Chunks',
    ylabel='Accuracy',
)

visualise.compare_models(
    tree_accs, rf_accs,
    labels=('Decision Tree', 'Random Forest'),
)

visualise.plot_predictions_vs_ground_truth(y_true, y_pred)
```

---

## Module Reference

### `numcompute_stream.stats`

| Class / Function | Description |
|---|---|
| `StreamingStats` | Per-feature running mean, variance, min, max using Welford's algorithm |
| `.update(X_chunk)` | Incorporate a new chunk into running statistics |
| `.get_mean()` | Per-feature running mean |
| `.get_std()` | Per-feature running standard deviation |
| `.get_histogram()` | Histogram from sliding window buffer |
| `.reset()` | Clear all accumulated state |

### `numcompute_stream.metrics`

| Class / Function | Description |
|---|---|
| `StreamingMetrics` | Accumulates accuracy, precision, recall, F1, confusion matrix over chunks |
| `.update(y_true, y_pred)` | Update with a new chunk |
| `.reset()` | Clear all accumulated state |
| `.result()` | Return full metrics dict |
| `.get_accuracy_history()` | Per-chunk accuracy list |
| `.get_rolling_accuracy()` | Accuracy over the rolling window only |

### `numcompute_stream.preprocessing`

| Class | Description |
|---|---|
| `StandardScaler` | Z-score normalisation with `partial_fit` using Welford's algorithm |
| `MinMaxScaler` | Min-max scaling with `partial_fit` tracking running min/max |
| `SimpleImputer` | Fill NaNs with running mean, median, or constant |
| `OneHotEncoder` | One-hot encoding with incremental category discovery |

### `numcompute_stream.tree`

| Class | Description |
|---|---|
| `DecisionTreeClassifier` | Depth-limited tree with Gini or entropy splitting, built from scratch |
| `.fit(X, y)` | Batch training |
| `.partial_fit(X_chunk, y_chunk)` | Incremental training — accumulates data and re-grows |
| `.predict(X)` | Predict class labels (NaN-safe) |
| `.predict_proba(X)` | Predict class probabilities |

Supports: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`

### `numcompute_stream.ensemble`

| Class | Description |
|---|---|
| `EnsembleClassifier` | N decision trees with Bagging or Random Forest strategy |
| `.fit(X, y)` | Batch training with bootstrap sampling |
| `.partial_fit(X_chunk, y_chunk)` | Each tree gets a bootstrap sample from the chunk |
| `.predict(X)` | Majority vote across all trees |
| `.predict_proba(X)` | Average probability across all trees |
| `.score(X, y)` | Accuracy on (X, y) |

### `numcompute_stream.pipeline`

| Class | Description |
|---|---|
| `Pipeline` | Chain of transformers + final estimator |
| `.fit(X, y)` | Batch training through all steps |
| `.partial_fit(X, y)` | Update each step incrementally |
| `.predict(X)` | Transform then predict |
| `.score(X, y)` | Accuracy on (X, y) |
| `FeatureUnion` | Parallel transformers concatenated column-wise |

### `numcompute_stream.stream`

| Class | Description |
|---|---|
| `StreamTrainer` | Orchestrates fit/score loop with logging |
| `.fit_chunk(X, y)` | Train on one chunk, log timing and memory |
| `.score_chunk(X, y)` | Score on one chunk, update metrics |
| `.fit_score_chunk(X, y)` | Train then score in one call |
| `.prequential_chunk(X, y)` | Test-then-train evaluation protocol |
| `.get_logs()` | Full per-chunk log |
| `.get_accuracy_history()` | Per-chunk accuracy list |
| `.summary()` | Training summary dict |

### `numcompute_stream.visualise`

| Function | Description |
|---|---|
| `plot_metric_over_time()` | Line chart of a metric across chunks with rolling mean |
| `compare_models()` | Side-by-side model comparison with difference plot |
| `plot_predictions_vs_ground_truth()` | Strip chart of predictions vs actuals |
| `plot_confusion_matrix()` | Heatmap of cumulative confusion matrix |
| `plot_fit_times()` | Bar chart of training time per chunk |

---

## Running Tests

```bash
python -m pytest tests/test_numcompute_stream.py -v
```

39 tests covering all streaming modules including edge cases such as NaN inputs, empty chunks, single-sample chunks, zero-variance features, and feature count mismatches.

---

## Running the Demo

```bash
cd demo_stream
jupyter notebook stream_demo.ipynb
```

The demo notebook:
1. Loads `iris_stream.csv` using `numcompute.io.load_csv_chunked`
2. Splits into 10 chunks of 30 rows to simulate streaming
3. Trains a Decision Tree and Random Forest pipeline incrementally
4. Logs and visualises accuracy, confusion matrix, fit times, and model comparison

---

## Running the Benchmark

```bash
python benchmark_stream/stream_benchmark.py --quick
```

Compares batch `.fit()` vs streaming `.partial_fit()` timing for Decision Tree, Random Forest, and Pipeline. Use `--quick` for a fast run (5 iterations) or omit for full 20-run benchmark.

---

## Design Decisions

**Welford's Algorithm** — Used for numerically stable running mean and variance in `StandardScaler`, `SimpleImputer`, and `StreamingStats`. Avoids storing all data while maintaining accuracy across chunks.

**Tree Re-growth Strategy** — `partial_fit` accumulates all seen data and re-grows the tree from scratch on each call. This ensures the tree structure always reflects the full history, which is more accurate than incremental node splitting for moderate dataset sizes.

**Bootstrap Sampling** — Each ensemble tree receives a different random bootstrap sample per chunk, ensuring diversity across estimators even with small chunk sizes.

**NaN Safety** — All modules handle NaN values gracefully. The tree skips NaN features during splitting and routes NaN samples left by default during prediction. Scalers and imputers use `np.nanmean` and ignore NaN rows in Welford updates.