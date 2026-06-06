"""
pipeline.py — ML pipeline with streaming support.

Extends the original NumCompute pipeline to support .partial_fit() for
incremental training. Transformers are updated first, then the final
estimator. Supports both classification and general transformer chains.

Author: [Your Name]
Module: numcompute_stream.pipeline
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Transformer(Protocol):
    """Structural protocol for pipeline-compatible transformers."""

    def fit(self, X: np.ndarray, /) -> Any: ...
    def transform(self, X: np.ndarray, /) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray, /) -> np.ndarray: ...


def _validate_step(name: str, obj: object) -> None:
    """Check that a pipeline step has the required methods."""
    for attr in ("fit", "transform", "fit_transform"):
        if not callable(getattr(obj, attr, None)):
            raise ValueError(
                f"Step '{name}' is missing required method '{attr}'. "
                f"All pipeline steps must implement fit, transform, and fit_transform."
            )


class Pipeline:
    """Chain of transformers followed by a final estimator.

    Supports both batch training via .fit() and streaming via
    .partial_fit(). In streaming mode, each transformer's .partial_fit()
    is called to update running statistics, then the model's .partial_fit()
    is called on the transformed chunk.

    Parameters
    ----------
    steps : list of (name, estimator) tuples
        Sequence of (name, transform/model) pairs. All but the last must
        implement fit/transform/fit_transform. The last step must at minimum
        implement fit and predict.

    Examples
    --------
    Batch:
        >>> pipe = Pipeline([
        ...     ('imputer', SimpleImputer()),
        ...     ('scaler', StandardScaler()),
        ...     ('model', DecisionTreeClassifier()),
        ... ])
        >>> pipe.fit(X_train, y_train)
        >>> preds = pipe.predict(X_test)

    Streaming:
        >>> pipe = Pipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('model', EnsembleClassifier()),
        ... ])
        >>> for chunk_X, chunk_y in stream:
        ...     pipe.partial_fit(chunk_X, chunk_y)
        >>> preds = pipe.predict(X_test)
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [s[0] for s in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        # Validate transformer steps (all except last)
        for name, est in steps[:-1]:
            _validate_step(name, est)
        self.steps = steps
        self._fitted = False

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Pipeline":
        """Fit all steps on the full dataset.

        Transformers use fit_transform; the final estimator uses fit.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray or None, shape (n_samples,)

        Returns
        -------
        self
        """
        Xt = np.asarray(X, dtype=float)
        for name, est in self.steps[:-1]:
            Xt = est.fit_transform(Xt)
        final_name, final_est = self.steps[-1]
        if y is not None:
            final_est.fit(Xt, y)
        else:
            final_est.fit(Xt)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformer steps (not the final estimator).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted yet. Call .fit() or .partial_fit() first.")
        Xt = np.asarray(X, dtype=float)
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit then transform through all steps."""
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> "Pipeline":
        """Incrementally update all steps with a new data chunk.

        Each transformer step calls .partial_fit() to update its running
        statistics, then transforms the chunk. The final estimator calls
        .partial_fit() on the transformed chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray or None, shape (n_samples,)
        classes : array-like or None
            Passed to the final estimator's .partial_fit() if supported.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If a transformer step doesn't support .partial_fit().
        """
        Xt = np.asarray(X, dtype=float)

        # Update and transform through intermediate steps
        for name, est in self.steps[:-1]:
            if hasattr(est, "partial_fit"):
                est.partial_fit(Xt)
            else:
                # Fall back to fit if partial_fit not available
                est.fit(Xt)
            Xt = est.transform(Xt)

        # Update final estimator
        final_name, final_est = self.steps[-1]
        if hasattr(final_est, "partial_fit"):
            if y is not None:
                if classes is not None and "classes" in final_est.partial_fit.__code__.co_varnames:
                    final_est.partial_fit(Xt, y, classes=classes)
                else:
                    final_est.partial_fit(Xt, y)
            else:
                final_est.partial_fit(Xt)
        else:
            # Fall back to fit
            if y is not None:
                final_est.fit(Xt, y)
            else:
                final_est.fit(Xt)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X through all steps then call predict on the final estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted yet.")
        Xt = self.transform(X)
        final_est = self.steps[-1][1]
        if not hasattr(final_est, "predict"):
            raise AttributeError(
                f"Final step '{self.steps[-1][0]}' does not implement .predict()."
            )
        return final_est.predict(Xt)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Transform X then call predict_proba on the final estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted yet.")
        Xt = self.transform(X)
        final_est = self.steps[-1][1]
        if not hasattr(final_est, "predict_proba"):
            raise AttributeError(
                f"Final step '{self.steps[-1][0]}' does not implement .predict_proba()."
            )
        return final_est.predict_proba(Xt)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float in [0, 1]
        """
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_step(self, name: str) -> Any:
        """Return a step by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        The estimator object.

        Raises
        ------
        KeyError if name not found.
        """
        for n, est in self.steps:
            if n == name:
                return est
        raise KeyError(f"No step named '{name}' in this pipeline.")

    def __repr__(self) -> str:
        step_str = " -> ".join(f"{n}({type(e).__name__})" for n, e in self.steps)
        return f"Pipeline([{step_str}])"


class FeatureUnion:
    """Fit several transformers on the same X and concatenate outputs column-wise.

    Supports streaming via .partial_fit().

    Parameters
    ----------
    transformers : list of (name, transformer) tuples
    """

    def __init__(self, transformers: list[tuple[str, Any]]) -> None:
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = [t[0] for t in transformers]
        if len(set(names)) != len(names):
            raise ValueError("FeatureUnion names must be unique.")
        for name, est in transformers:
            _validate_step(name, est)
        self.transformers = transformers
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureUnion":
        X = np.asarray(X)
        for _, est in self.transformers:
            est.fit(X)
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray) -> "FeatureUnion":
        """Update all transformers with a new chunk."""
        X = np.asarray(X)
        for _, est in self.transformers:
            if hasattr(est, "partial_fit"):
                est.partial_fit(X)
            else:
                est.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureUnion is not fitted yet.")
        X = np.asarray(X)
        parts = [est.transform(X) for _, est in self.transformers]
        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        parts = [est.fit_transform(X) for _, est in self.transformers]
        self._fitted = True
        return np.hstack(parts)
