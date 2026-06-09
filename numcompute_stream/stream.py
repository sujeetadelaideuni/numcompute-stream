"""
stream.py — StreamTrainer for orchestrating incremental learning.

Manages a pipeline through streaming data chunks, logging per-chunk
metrics, timing, and memory usage. Supports fit-then-score and
prequential (test-then-train) evaluation protocols.

Author: Sujeet Ghosh
Module: numcompute_stream.stream
"""
from __future__ import annotations
import time
import tracemalloc
import numpy as np

from .metrics import StreamingMetrics


class StreamTrainer:
    """Orchestrate streaming learning over a pipeline.

    Parameters
    ----------
    pipeline : object
        Any pipeline or estimator with ``.partial_fit(X, y)`` and ``.predict(X)``.
    metrics : StreamingMetrics or None
        Metrics accumulator. A new one is created if None.
    verbose : bool

    Examples
    --------
    >>> trainer = StreamTrainer(pipeline=pipe, verbose=True)
    >>> for chunk_X, chunk_y in stream:
    ...     trainer.fit_chunk(chunk_X, chunk_y)
    ...     trainer.score_chunk(chunk_X, chunk_y)
    >>> trainer.get_logs()
    """

    def __init__(
        self,
        pipeline,
        metrics: StreamingMetrics | None = None,
        verbose: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics if metrics is not None else StreamingMetrics()
        self.verbose = verbose
        self._logs: list[dict] = []
        self.n_chunks_trained_: int = 0
        self.n_samples_trained_: int = 0
        self._total_fit_time_ms: float = 0.0

    def fit_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "StreamTrainer":
        """Train the pipeline on one data chunk.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
        y_chunk : np.ndarray, shape (n_samples,)
        classes : array-like or None

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        n = len(y_chunk)

        if n == 0:
            return self

        tracemalloc.start()
        t0 = time.perf_counter()

        if classes is not None:
            try:
                self.pipeline.partial_fit(X_chunk, y_chunk, classes=classes)
            except TypeError:
                self.pipeline.partial_fit(X_chunk, y_chunk)
        else:
            self.pipeline.partial_fit(X_chunk, y_chunk)

        fit_time_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self._total_fit_time_ms += fit_time_ms
        self.n_chunks_trained_ += 1
        self.n_samples_trained_ += n

        self._logs.append({
            "chunk":          self.n_chunks_trained_,
            "n_samples":      n,
            "fit_time_ms":    round(fit_time_ms, 3),
            "peak_memory_kb": round(peak_mem / 1024, 2),
            "cumulative_n":   self.n_samples_trained_,
        })

        if self.verbose:
            print(
                f"[Chunk {self.n_chunks_trained_:>3}] "
                f"n={n:<5} | "
                f"fit={fit_time_ms:.1f}ms | "
                f"mem={peak_mem/1024:.1f}KB | "
                f"total_seen={self.n_samples_trained_}"
            )

        return self

    def score_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
    ) -> float:
        """Score the pipeline on one chunk and update metrics.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
        y_chunk : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
            Accuracy on this chunk.

        Raises
        ------
        RuntimeError
            If pipeline has not been trained yet.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if self.n_chunks_trained_ == 0:
            raise RuntimeError(
                "StreamTrainer has not been trained yet. Call .fit_chunk() before .score_chunk()."
            )

        y_pred = self.pipeline.predict(X_chunk)
        chunk_acc = float(np.mean(y_pred == y_chunk))
        self.metrics.update(y_chunk, y_pred)

        if self._logs:
            self._logs[-1]["chunk_accuracy"] = round(chunk_acc, 4)
            self._logs[-1]["cumulative_accuracy"] = round(self.metrics.get_accuracy(), 4)

        if self.verbose:
            print(
                f"           "
                f"chunk_acc={chunk_acc:.4f} | "
                f"cumul_acc={self.metrics.get_accuracy():.4f}"
            )

        return chunk_acc

    def fit_score_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> float:
        """Train then score on the same chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
        y_chunk : np.ndarray
        classes : array-like or None

        Returns
        -------
        float
        """
        self.fit_chunk(X_chunk, y_chunk, classes=classes)
        return self.score_chunk(X_chunk, y_chunk)

    def prequential_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> float:
        """Test-then-train: score before fitting on this chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
        y_chunk : np.ndarray
        classes : array-like or None

        Returns
        -------
        float
            Accuracy before training. Returns nan on the first chunk.
        """
        if self.n_chunks_trained_ == 0:
            self.fit_chunk(X_chunk, y_chunk, classes=classes)
            return float("nan")
        chunk_acc = self.score_chunk(X_chunk, y_chunk)
        self.fit_chunk(X_chunk, y_chunk, classes=classes)
        return chunk_acc

    def get_logs(self) -> list[dict]:
        """Full per-chunk training log.

        Returns
        -------
        list of dict
        """
        return list(self._logs)

    def get_accuracy_history(self) -> list[float]:
        """Per-chunk accuracy values in order.

        Returns
        -------
        list of float
        """
        return [e["chunk_accuracy"] for e in self._logs if "chunk_accuracy" in e]

    def get_cumulative_accuracy_history(self) -> list[float]:
        """Cumulative accuracy after each chunk.

        Returns
        -------
        list of float
        """
        return [e["cumulative_accuracy"] for e in self._logs if "cumulative_accuracy" in e]

    def get_fit_times(self) -> list[float]:
        """Fit time in ms per chunk.

        Returns
        -------
        list of float
        """
        return [e["fit_time_ms"] for e in self._logs]

    def get_memory_usage(self) -> list[float]:
        """Peak memory in KB per chunk.

        Returns
        -------
        list of float
        """
        return [e["peak_memory_kb"] for e in self._logs]

    def summary(self) -> dict:
        """Summary of training statistics.

        Returns
        -------
        dict with keys: n_chunks, n_samples, total_fit_time_ms,
        mean_chunk_accuracy, final_cumulative_accuracy
        """
        acc_hist = self.get_accuracy_history()
        cumul_hist = self.get_cumulative_accuracy_history()
        return {
            "n_chunks":                  self.n_chunks_trained_,
            "n_samples":                 self.n_samples_trained_,
            "total_fit_time_ms":         round(self._total_fit_time_ms, 2),
            "mean_chunk_accuracy":       round(float(np.mean(acc_hist)), 4) if acc_hist else None,
            "final_cumulative_accuracy": round(cumul_hist[-1], 4) if cumul_hist else None,
        }

    def reset(self) -> "StreamTrainer":
        """Reset all logs and metrics. Pipeline state is NOT reset.

        Returns
        -------
        self
        """
        self._logs = []
        self.n_chunks_trained_ = 0
        self.n_samples_trained_ = 0
        self._total_fit_time_ms = 0.0
        self.metrics.reset()
        return self
