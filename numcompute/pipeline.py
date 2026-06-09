from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Transformer(Protocol):
    """Structural protocol for objects usable inside :class:`Pipeline` / :class:`FeatureUnion`."""

    def fit(self, X: np.ndarray, /) -> Any: ...

    def transform(self, X: np.ndarray, /) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray, /) -> np.ndarray: ...


def _validate_transformer(name: str, obj: object) -> None:
    for attr in ("fit", "transform", "fit_transform"):
        if not callable(getattr(obj, attr, None)):
            raise ValueError(f"Step '{name}' does not implement required method '{attr}'.")


class Pipeline:
    """Sequentially apply transformers; all but the last use ``fit_transform`` during ``fit``."""

    def __init__(self, steps: list[tuple[str, Transformer]]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [s[0] for s in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        for name, est in steps:
            _validate_transformer(name, est)
        self.steps = steps
        self._fitted = False

    def fit(self, X: np.ndarray) -> Pipeline:
        Xt = np.asarray(X)
        for name, est in self.steps[:-1]:
            Xt = est.fit_transform(Xt)
        self.steps[-1][1].fit(Xt)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("This Pipeline instance is not fitted yet; call fit first.")
        Xt = np.asarray(X)
        for _, est in self.steps:
            Xt = est.transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class FeatureUnion:
    """Fit several transformers on the same ``X`` and concatenate outputs column-wise."""

    def __init__(self, transformers: list[tuple[str, Transformer]]) -> None:
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = [t[0] for t in transformers]
        if len(set(names)) != len(names):
            raise ValueError("FeatureUnion names must be unique.")
        for name, est in transformers:
            _validate_transformer(name, est)
        self.transformers = transformers
        self._fitted = False

    def fit(self, X: np.ndarray) -> FeatureUnion:
        X = np.asarray(X)
        for _, est in self.transformers:
            est.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("This FeatureUnion instance is not fitted yet; call fit first.")
        X = np.asarray(X)
        parts = [est.transform(X) for _, est in self.transformers]
        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        parts = [est.fit_transform(X) for _, est in self.transformers]
        self._fitted = True
        return np.hstack(parts)
