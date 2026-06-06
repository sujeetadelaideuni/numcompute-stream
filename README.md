# NumCompute

A modular, production-grade scientific computing toolkit built with **plain Python + NumPy only**.

NumCompute replicates the core functionality of ML libraries like scikit-learn from scratch — with an emphasis on deep algorithmic understanding, numerical stability, fully vectorised computation, and clean software engineering. No pandas, no scikit-learn, no external ML/DL libraries permitted.

> **Course:** COMP-5004 — Programming for AI  
> **University:** University of Adelaide

---

## Team

| Person | Name | Modules Owned |
|--------|------|---------------|
| Person 1 | Shaun D'souza | `io.py`, `preprocessing.py` |
| Person 2 | Manini Sikri | `sort_search.py`, `rank.py` |
| Person 3 | Sujeet Ghosh | `stats.py`, `metrics.py` |
| Person 4 | Sushmeet Matharu | `optim.py`, `pipeline.py`, `benchmarking.py`, demo |

---

## Project Structure

```
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py              # CSV loading (streaming/chunking), dtype handling
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── sort_search.py     # Stable sort, multi-key sort, top-k, quickselect, binary search
│   ├── rank.py            # Ranking with tie handling; percentiles
│   ├── stats.py           # Descriptive statistics, histogram, quantiles (Welford streaming)
│   ├── metrics.py         # Accuracy, Precision, Recall, F1, MSE, Confusion Matrix, ROC/AUC
│   ├── optim.py           # Finite-difference gradients, Jacobian
│   ├── pipeline.py        # Transformer/Estimator protocol, Pipeline chaining
│   ├── utils.py           # Distances, activations, logsumexp, top-k helpers, batching
│   └── benchmarking.py    # Micro-benchmark harness, vectorised vs loop comparisons
├── tests/
│   ├── __init__.py
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   └── test_pipeline.py
├── demo/
│   └── quickstart.ipynb   # End-to-end demo notebook
├── benchmark/
│   └── benchmark_runner.py
├── README.md
└── pyproject.toml
```

---

## Installation

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/SD-adelaideuni/NumCompute.git
cd NumCompute
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24

---

## Quick Start

```python
import numpy as np
from numcompute.io import load_csv, save_csv
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from numcompute.pipeline import Pipeline

# Load data
X = load_csv("data.csv")

# Build and run a pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
])
X_transformed = pipe.fit_transform(X)

# Save results
save_csv(X_transformed, "output.csv")
```

---

## Module API Reference

### `io.py` — Data I/O (Shaun D'souza)

| Function | Description |
|----------|-------------|
| `load_csv(filepath, delimiter, dtype, missing_values, filling_values)` | Load CSV → 2-D NumPy array. Missing cells filled with `np.nan` by default. |
| `load_csv_chunked(filepath, chunksize, **kwargs)` | Generator yielding row-chunks for large files. |
| `save_csv(array, filepath, delimiter, header)` | Write 1-D or 2-D array to CSV via `numpy.savetxt`. |

```python
from numcompute.io import load_csv, load_csv_chunked, save_csv

# Full load
X = load_csv("data.csv", delimiter=",")

# Streaming load for large files
for chunk in load_csv_chunked("big_data.csv", chunksize=1000):
    process(chunk)

# Save
save_csv(X, "output.csv", header="col1,col2,col3")
```

---

### `preprocessing.py` — Data Preprocessing (Shaun D'souza)

All classes implement the `fit(X) → self`, `transform(X) → X_out`, `fit_transform(X) → X_out` API.

| Class | Description |
|-------|-------------|
| `StandardScaler` | Z-score standardisation: subtract mean, divide by std |
| `MinMaxScaler` | Scale features to a given range (default [0, 1]) |
| `OneHotEncoder` | Encode categorical integer columns as binary columns |
| `SimpleImputer` | Replace NaN values with a constant or column statistic |

```python
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

---

### `sort_search.py` — Sorting & Searching (Manini Sikri)

| Function | Description |
|----------|-------------|
| `stable_sort(arr, axis)` | Wrapper around `np.sort(kind='stable')` |
| `multi_key_sort(arr, keys)` | Sort 2-D array by multiple column indices |
| `topk(values, k, largest, return_indices)` | Top-k via `np.argpartition` |
| `quickselect(arr, k)` | Educational quickselect implementation |
| `binary_search(sorted_array, x)` | Returns `(insertion_index, exists_bool)` |

---

### `rank.py` — Ranking (Manini Sikri)

| Function | Description |
|----------|-------------|
| `rank(data, method)` | Rank with tie handling: `'average'`, `'dense'`, `'ordinal'` |
| `percentile(data, q, interpolation)` | Percentile with `'linear'`, `'lower'`, `'higher'`, `'midpoint'` |

---

### `stats.py` — Descriptive Statistics (Sujeet Ghosh)

| Function | Description |
|----------|-------------|
| `mean(X, axis)` | Axis-wise mean with NaN handling |
| `median(X, axis)` | Axis-wise median |
| `std(X, axis)` | Standard deviation (Welford streaming) |
| `histogram(X, bins)` | Histogram counts and bin edges |
| `quantile(X, q, axis)` | Quantiles with NaN handling |

---

### `metrics.py` — Evaluation Metrics (Sujeet Ghosh)

| Function | Description |
|----------|-------------|
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `precision(y_true, y_pred, average)` | Precision score |
| `recall(y_true, y_pred, average)` | Recall score |
| `f1(y_true, y_pred, average)` | F1 score |
| `confusion_matrix(y_true, y_pred)` | Confusion matrix as 2-D array |
| `mse(y_true, y_pred)` | Mean squared error |
| `roc_curve(y_true, y_score)` | *(Bonus)* ROC curve FPR/TPR arrays |
| `auc(fpr, tpr)` | *(Bonus)* Area under the curve |

---

### `optim.py` — Gradient Estimation (Sushmeet Matharu)

| Function | Description |
|----------|-------------|
| `grad(f, x, h, method)` | Finite-difference gradient: `'central'` or `'forward'` |
| `jacobian(F, x, h, method)` | Jacobian matrix for vector-valued functions |

---

### `pipeline.py` — Pipeline (Sushmeet Matharu)

| Class/Function | Description |
|----------------|-------------|
| `Pipeline(steps)` | Chain transformers; supports `fit`, `transform`, `fit_transform` |

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, OneHotEncoder

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('encode', OneHotEncoder()),
])
X_out = pipe.fit_transform(X)
```

---

## API Conventions

These conventions are consistent across all modules:

- **Array shape:** rows = samples, columns = features — shape `(n_samples, n_features)`
- **Axis semantics:** `axis=0` operates over rows (per-column stats); `axis=1` operates over columns (per-row stats)
- **Missing values:** represented as `np.nan` throughout; all functions handle NaN safely
- **Scaler API:** all preprocessing classes implement `fit(X)`, `transform(X)`, `fit_transform(X)`
- **Metrics API:** all metric functions take `(y_true, y_pred)` as first two positional arguments
- **Error messages:** `ValueError` for shape/type mismatches; `FileNotFoundError` for missing files

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=numcompute --cov-report=term-missing

# Run a specific module's tests
pytest tests/test_io.py -v
```

The test suite contains ≥ 20 unit tests covering edge cases including empty arrays, all-equal values, duplicates and ties, extreme `k` values, NaNs, and non-contiguous strides.

---

## Benchmarks

Run the benchmark suite to compare vectorised implementations against Python loop equivalents:

```bash
python benchmark/benchmark_runner.py
```

Sample performance table (Apple M-series, NumPy 1.26):

| Operation | Loop (ms) | Vectorised (ms) | Speedup |
|-----------|-----------|-----------------|---------|
| StandardScaler (n=100k) | TBD | TBD | TBD |
| Top-k (n=100k, k=10) | TBD | TBD | TBD |
| MSE (n=100k) | TBD | TBD | TBD |

*(Table will be populated after benchmarking is complete)*

---

## Design Principles

- **No Python loops** in core computations — all operations are fully vectorised using NumPy
- **Numerical stability** — handles overflow/underflow, NaNs, ties, and uses stable forms (e.g. max-shifted softmax, `logsumexp`)
- **Consistent API** — clear `axis` semantics, documented input/output shapes, informative error messages
- **Comprehensive docstrings** — every function documents parameters, return values, shapes, exceptions, and time/space complexity


## License

MIT License — see [LICENSE](LICENSE) for details.
