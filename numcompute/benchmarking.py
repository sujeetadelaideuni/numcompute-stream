from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def timeit_func(
    func: Callable[..., Any],
    *args: Any,
    n_runs: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    """Measure wall-clock latency statistics for ``func(*args)``.

    Parameters
    ----------
    func
        Callable to benchmark.
    *args
        Positional arguments forwarded to ``func``.
    n_runs
        Number of timed executions after warm-up.
    warmup
        Number of discarded warm-up calls.

    Returns
    -------
    dict
        Keys ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.

    Notes
    -----
    Uses ``time.perf_counter``; complexity ``O(n_runs * C_func)``.
    """
    for _ in range(warmup):
        func(*args)
    samples: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def compare(
    label_a: str,
    func_a: Callable[..., Any],
    label_b: str,
    func_b: Callable[..., Any],
    *args: Any,
    n_runs: int = 100,
) -> dict[str, Any]:
    """Benchmark two callables on the same ``*args`` and summarise relative speed.

    Returns
    -------
    dict
        Nested timing dicts under ``label_a`` / ``label_b`` plus ``speedup`` = mean_a / mean_b.
    """
    stats_a = timeit_func(func_a, *args, n_runs=n_runs)
    stats_b = timeit_func(func_b, *args, n_runs=n_runs)
    speedup = stats_a["mean_ms"] / stats_b["mean_ms"] if stats_b["mean_ms"] > 0 else float("inf")
    return {label_a: stats_a, label_b: stats_b, "speedup": float(speedup)}


def print_benchmark_table(results: list[dict[str, Any]]) -> None:
    """Print a fixed-width ASCII table for README / notebook capture.

    Each ``results`` item should contain keys: ``operation``, ``vectorised``, ``loop``,
    ``speedup`` (the latter three may be nested dicts with ``mean_ms`` or plain floats).
    """
    rows: list[tuple[str, str, str, str]] = []
    for item in results:
        op = str(item.get("operation", ""))

        def fmt_ms(val: Any) -> str:
            if isinstance(val, dict):
                return f"{val['mean_ms']:.4f}"
            return f"{float(val):.4f}"

        v = fmt_ms(item.get("vectorised", {}))
        l = fmt_ms(item.get("loop", {}))
        sp = item.get("speedup", 0.0)
        if isinstance(sp, dict):
            sp_val = sp.get("speedup", float("nan"))
        else:
            sp_val = float(sp)
        rows.append((op, v, l, f"{sp_val:.2f}x"))

    headers = ("Operation", "Vectorised (ms)", "Loop (ms)", "Speedup")
    widths = (
        max(len(headers[0]), max((len(r[0]) for r in rows), default=0)),
        max(len(headers[1]), max((len(r[1]) for r in rows), default=0)),
        max(len(headers[2]), max((len(r[2]) for r in rows), default=0)),
        max(len(headers[3]), max((len(r[3]) for r in rows), default=0)),
    )
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    def fmt_row(cells: tuple[str, str, str, str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    print(line)
    print(fmt_row(headers))
    print(line)
    for r in rows:
        print(fmt_row(r))
    print(line)


def _loop_mean(arr: np.ndarray) -> float:
    s = 0.0
    for v in arr.ravel():
        s += float(v)
    return s / arr.size


def _loop_sort_copy(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(float).ravel().tolist()
    x.sort()
    return np.asarray(x, dtype=float)


def run_default_suite(
    n_elems: int = 500_000,
    n_runs: int = 50,
) -> tuple[list[dict[str, Any]], str]:
    """Execute the two required paired benchmarks and return table rows plus an env note."""
    rng = np.random.default_rng(0)
    big = rng.standard_normal(n_elems)

    def vec_mean() -> float:
        return float(np.mean(big))

    mean_cmp = compare(
        "loop_mean",
        lambda: _loop_mean(big),
        "np.mean",
        vec_mean,
        n_runs=n_runs,
    )

    def vec_sort() -> np.ndarray:
        return np.sort(big)

    sort_cmp = compare(
        "loop_sort",
        lambda: _loop_sort_copy(big),
        "np.sort",
        vec_sort,
        n_runs=max(5, n_runs // 5),
    )

    rows = [
        {
            "operation": "Mean of 1-D array",
            "vectorised": mean_cmp["np.mean"],
            "loop": mean_cmp["loop_mean"],
            "speedup": mean_cmp,
        },
        {
            "operation": "Sort 1-D array",
            "vectorised": sort_cmp["np.sort"],
            "loop": sort_cmp["loop_sort"],
            "speedup": sort_cmp,
        },
    ]
    env = (
        f"CPU: {platform.processor() or 'unknown'} | "
        f"Python {sys.version.split()[0]} | NumPy {np.__version__}"
    )
    return rows, env


if __name__ == "__main__":
    table_rows, env_note = run_default_suite()
    print(env_note)
    print()
    print_benchmark_table(table_rows)
