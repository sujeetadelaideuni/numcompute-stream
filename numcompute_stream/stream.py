"""
stream.py — StreamTrainer for orchestrating incremental learning.

Manages a pipeline + metrics + logging across streaming data chunks.
Supports .fit_chunk() and .score_chunk() methods and tracks per-chunk
performance, memory footprint, and cumulative accuracy over time.

Author: [Your Name]
Module: numcompute_stream.stream
"""

from __future__ import annotations
import time
import tracemalloc
import numpy as np

from .metrics import StreamingMetrics


class StreamTrainer:
    """Orchestrate streaming learning over a pipeline.

    Handles the training loop chunk by chunk, logging metrics and
    memory usage at each step. Works with any pipeline that supports
    .partial_fit(X, y).

    Parameters
    ----------
    pipeline : object
        Any pipeline or estimator with .partial_fit(X, y) and .predict(X).
    metrics : StreamingMetrics or None
        Metrics accumulator. If None, a new one is created automatically.
    verbose : bool
        If True, print progress after each chunk.

    Examples
    --------
    >>> trainer = StreamTrainer(pipeline=pipe, verbose=True)
    >>> for chunk_X, chunk_y in stream:
    ...     trainer.fit_chunk(chunk_X, chunk_y)
    ...     trainer.score_chunk(chunk_X, chunk_y)
    >>> logs = trainer.get_logs()
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

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "StreamTrainer":
        """Train the pipeline on one data chunk.

        Calls pipeline.partial_fit(X_chunk, y_chunk) and logs timing
        and memory usage.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
        y_chunk : np.ndarray, shape (n_samples,)
        classes : array-like or None
            All possible class labels (useful for first chunk).

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        n = len(y_chunk)

        if n == 0:
            return self

        # Track memory
        tracemalloc.start()
        t0 = time.perf_counter()

        if classes is not None and hasattr(self.pipeline, "partial_fit"):
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

        log_entry = {
            "chunk":           self.n_chunks_trained_,
            "n_samples":       n,
            "fit_time_ms":     round(fit_time_ms, 3),
            "peak_memory_kb":  round(peak_mem / 1024, 2),
            "cumulative_n":    self.n_samples_trained_,
        }
        self._logs.append(log_entry)

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
        RuntimeError if pipeline has not been trained yet.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if self.n_chunks_trained_ == 0:
            raise RuntimeError(
                "StreamTrainer has not been trained yet. "
                "Call .fit_chunk() before .score_chunk()."
            )

        y_pred = self.pipeline.predict(X_chunk)
        chunk_acc = float(np.mean(y_pred == y_chunk))

        # Update streaming metrics
        self.metrics.update(y_chunk, y_pred)

        # Add score to the latest log entry
        if self._logs:
            self._logs[-1]["chunk_accuracy"] = round(chunk_acc, 4)
            self._logs[-1]["cumulative_accuracy"] = round(
                self.metrics.get_accuracy(), 4
            )

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
        """Convenience method: fit then score on the same chunk.

        This is the standard prequential (test-then-train) evaluation:
        score first, then train. This version trains first, then scores.

        Parameters
        ----------
        X_chunk : np.ndarray
        y_chunk : np.ndarray
        classes : array-like or None

        Returns
        -------
        float — chunk accuracy after training
        """
        self.fit_chunk(X_chunk, y_chunk, classes=classes)
        return self.score_chunk(X_chunk, y_chunk)

    def prequential_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> float:
        """Test-then-train evaluation: score before fitting.

        This is the proper streaming evaluation protocol. Score the model
        on the new chunk BEFORE training on it (since that's the realistic
        scenario — the model hasn't seen this data yet).

        Parameters
        ----------
        X_chunk : np.ndarray
        y_chunk : np.ndarray
        classes : array-like or None

        Returns
        -------
        float — chunk accuracy before training on this chunk.
                Returns nan on the first chunk (no model yet).
        """
        if self.n_chunks_trained_ == 0:
            # No model yet — just train
            self.fit_chunk(X_chunk, y_chunk, classes=classes)
            return float("nan")

        # Score first, then train
        chunk_acc = self.score_chunk(X_chunk, y_chunk)
        self.fit_chunk(X_chunk, y_chunk, classes=classes)
        return chunk_acc

    # ------------------------------------------------------------------
    # Logging and results
    # ------------------------------------------------------------------

    def get_logs(self) -> list[dict]:
        """Return the full per-chunk training log.

        Returns
        -------
        list of dicts with keys: chunk, n_samples, fit_time_ms,
        peak_memory_kb, cumulative_n, chunk_accuracy (if scored),
        cumulative_accuracy (if scored).
        """
        return list(self._logs)

    def get_accuracy_history(self) -> list[float]:
        """Return per-chunk accuracy values in order.

        Returns
        -------
        list of float (only chunks that were scored)
        """
        return [
            entry["chunk_accuracy"]
            for entry in self._logs
            if "chunk_accuracy" in entry
        ]

    def get_cumulative_accuracy_history(self) -> list[float]:
        """Return cumulative accuracy after each chunk.

        Returns
        -------
        list of float
        """
        return [
            entry["cumulative_accuracy"]
            for entry in self._logs
            if "cumulative_accuracy" in entry
        ]

    def get_fit_times(self) -> list[float]:
        """Return fit time in ms per chunk.

        Returns
        -------
        list of float
        """
        return [entry["fit_time_ms"] for entry in self._logs]

    def get_memory_usage(self) -> list[float]:
        """Return peak memory in KB per chunk.

        Returns
        -------
        list of float
        """
        return [entry["peak_memory_kb"] for entry in self._logs]

    def summary(self) -> dict:
        """Return a summary of training statistics.

        Returns
        -------
        dict with keys: n_chunks, n_samples, total_fit_time_ms,
        mean_chunk_accuracy, final_cumulative_accuracy
        """
        acc_hist = self.get_accuracy_history()
        cumul_hist = self.get_cumulative_accuracy_history()
        return {
            "n_chunks":                 self.n_chunks_trained_,
            "n_samples":                self.n_samples_trained_,
            "total_fit_time_ms":        round(self._total_fit_time_ms, 2),
            "mean_chunk_accuracy":      round(float(np.mean(acc_hist)), 4) if acc_hist else None,
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
