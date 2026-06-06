from numcompute import benchmarking

def test_timeit_func_keys_and_positive_stats() -> None:
    import numpy as np

    x = np.ones(1000)

    def f() -> None:
        _ = float(np.sum(x**2))

    stats = benchmarking.timeit_func(f, n_runs=10, warmup=2)
    for key in ("mean_ms", "std_ms", "min_ms", "max_ms"):
        assert key in stats
        assert stats[key] >= 0.0


def test_compare_reports_speedup() -> None:
    import numpy as np

    arr = np.arange(10_000, dtype=float)

    def loop_sum() -> float:
        s = 0.0
        for v in arr:
            s += float(v)
        return s

    def vec_sum() -> float:
        return float(np.sum(arr))

    out = benchmarking.compare("loop", loop_sum, "vec", vec_sum, n_runs=20)
    assert "speedup" in out
    assert out["speedup"] > 0.0


def test_default_suite_returns_required_rows() -> None:
    rows, env = benchmarking.run_default_suite(n_elems=10_000, n_runs=5)
    assert isinstance(env, str)
    assert len(rows) == 2
    for row in rows:
        assert "operation" in row
        assert "vectorised" in row
        assert "loop" in row
        assert "speedup" in row
