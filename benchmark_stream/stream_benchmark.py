"""
benchmark/stream_benchmark.py

Compares streaming (partial_fit) vs batch (fit) training performance
for both DecisionTreeClassifier and EnsembleClassifier.

Run with:  python benchmark/stream_benchmark.py
"""

from __future__ import annotations
import sys
import os
import time
import platform
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.pipeline import Pipeline
from numcompute.benchmarking import timeit_func, print_benchmark_table


def make_data(n=500, n_features=8, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, n_classes, n)
    return X, y


def split_chunks(X, y, n_chunks=10):
    size = len(y) // n_chunks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size]) for i in range(n_chunks)]


def batch_tree_fit(X, y):
    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X, y)
    return tree


def streaming_tree_fit(chunks):
    tree = DecisionTreeClassifier(max_depth=4)
    for X_c, y_c in chunks:
        tree.partial_fit(X_c, y_c)
    return tree


def batch_rf_fit(X, y):
    rf = EnsembleClassifier(n_estimators=10, method='random_forest', max_depth=4)
    rf.fit(X, y)
    return rf


def streaming_rf_fit(chunks):
    rf = EnsembleClassifier(n_estimators=10, method='random_forest', max_depth=4)
    for X_c, y_c in chunks:
        rf.partial_fit(X_c, y_c)
    return rf


def batch_pipeline_fit(X, y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  DecisionTreeClassifier(max_depth=4)),
    ])
    pipe.fit(X, y)
    return pipe


def streaming_pipeline_fit(chunks):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  DecisionTreeClassifier(max_depth=4)),
    ])
    for X_c, y_c in chunks:
        pipe.partial_fit(X_c, y_c)
    return pipe


def run_benchmarks(n=500, n_chunks=10, n_runs=20):
    X, y = make_data(n=n)
    chunks = split_chunks(X, y, n_chunks=n_chunks)

    results = []

    tree_batch   = timeit_func(batch_tree_fit, X, y, n_runs=n_runs)
    tree_stream  = timeit_func(streaming_tree_fit, chunks, n_runs=n_runs)
    results.append({
        'operation':  'DecisionTree fit',
        'batch':      tree_batch,
        'stream':     tree_stream,
        'speedup':    tree_batch['mean_ms'] / tree_stream['mean_ms'],
    })

    rf_batch   = timeit_func(batch_rf_fit, X, y, n_runs=n_runs)
    rf_stream  = timeit_func(streaming_rf_fit, chunks, n_runs=n_runs)
    results.append({
        'operation':  'RandomForest fit (10 trees)',
        'batch':      rf_batch,
        'stream':     rf_stream,
        'speedup':    rf_batch['mean_ms'] / rf_stream['mean_ms'],
    })

    pipe_batch  = timeit_func(batch_pipeline_fit, X, y, n_runs=n_runs)
    pipe_stream = timeit_func(streaming_pipeline_fit, chunks, n_runs=n_runs)
    results.append({
        'operation':  'Pipeline fit (scaler + tree)',
        'batch':      pipe_batch,
        'stream':     pipe_stream,
        'speedup':    pipe_batch['mean_ms'] / pipe_stream['mean_ms'],
    })

    return results


def run_size_comparison():
    print('\n=== Accuracy: Batch vs Streaming (same data) ===\n')
    X, y = make_data(n=300)
    chunks = split_chunks(X, y, n_chunks=10)

    tree_batch = DecisionTreeClassifier(max_depth=4)
    tree_batch.fit(X, y)
    batch_acc = float(np.mean(tree_batch.predict(X) == y))

    tree_stream = DecisionTreeClassifier(max_depth=4)
    for X_c, y_c in chunks:
        tree_stream.partial_fit(X_c, y_c)
    stream_acc = float(np.mean(tree_stream.predict(X) == y))

    rf_batch = EnsembleClassifier(n_estimators=10, max_depth=4)
    rf_batch.fit(X, y)
    rf_batch_acc = float(np.mean(rf_batch.predict(X) == y))

    rf_stream = EnsembleClassifier(n_estimators=10, max_depth=4)
    for X_c, y_c in chunks:
        rf_stream.partial_fit(X_c, y_c)
    rf_stream_acc = float(np.mean(rf_stream.predict(X) == y))

    print(f'{"Model":<30} {"Batch Acc":>12} {"Stream Acc":>12} {"Diff":>8}')
    print('-' * 65)
    print(f'{"DecisionTree (max_depth=4)":<30} {batch_acc:>12.4f} {stream_acc:>12.4f} {stream_acc-batch_acc:>+8.4f}')
    print(f'{"RandomForest (10 trees)":<30} {rf_batch_acc:>12.4f} {rf_stream_acc:>12.4f} {rf_stream_acc-rf_batch_acc:>+8.4f}')


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv

    env = (
        f"CPU: {platform.processor() or 'unknown'} | "
        f"Python {__import__('sys').version.split()[0]} | "
        f"NumPy {np.__version__}"
    )
    print(env)
    print()
    print('=== Timing: Batch fit vs Streaming partial_fit ===')
    print('(Batch = single .fit() call | Stream = 10 x .partial_fit() calls)')
    print()

    n_runs = 5 if quick else 20
    n = 200 if quick else 500
    results = run_benchmarks(n=n, n_chunks=10, n_runs=n_runs)

    headers = ("Operation", "Batch fit (ms)", "Stream partial_fit (ms)", "Speedup")
    rows = []
    for r in results:
        rows.append((
            r['operation'],
            f"{r['batch']['mean_ms']:.4f}",
            f"{r['stream']['mean_ms']:.4f}",
            f"{r['speedup']:.2f}x",
        ))

    widths = tuple(max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers))
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    print(line)
    print(fmt_row(headers))
    print(line)
    for r in rows:
        print(fmt_row(r))
    print(line)

    run_size_comparison()
