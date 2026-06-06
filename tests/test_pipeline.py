import numpy as np
import pytest
from numcompute.pipeline import FeatureUnion, Pipeline
from numcompute.preprocessing import OneHotEncoder, StandardScaler

def test_pipeline_end_to_end_toy() -> None:
    rng = np.random.default_rng(42)
    X = np.column_stack([rng.normal(size=(20, 2)), rng.integers(0, 3, size=(20, 1))]).astype(float)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("encoder", OneHotEncoder()),
        ]
    )
    out = pipe.fit_transform(X)
    assert out.shape[0] == X.shape[0]
    assert out.shape[1] > 0
    assert np.all(np.isfinite(out))


def test_pipeline_transform_before_fit() -> None:
    pipe = Pipeline([("scaler", StandardScaler())])
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.transform(np.ones((3, 2)))


def test_pipeline_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Pipeline(
            [
                ("same", StandardScaler()),
                ("same", StandardScaler()),
            ]
        )


def test_pipeline_invalid_step() -> None:
    class Bad:
        def fit(self, X):
            return self

    with pytest.raises(ValueError, match="required method"):
        Pipeline([("bad", Bad())])


def test_feature_union_column_counts() -> None:
    class SliceTransform:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        def fit(self, X):
            return self

        def transform(self, X):
            X = np.asarray(X)
            return X[:, self.idx : self.idx + 1]

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    X = np.arange(12, dtype=float).reshape(4, 3)
    fu = FeatureUnion(
        [
            ("c0", SliceTransform(0)),
            ("c12", SliceTransform(1)),
        ]
    )
    out = fu.fit_transform(X)
    assert out.shape == (4, 2)


def test_feature_union_transform_before_fit() -> None:
    fu = FeatureUnion([("a", StandardScaler())])
    with pytest.raises(RuntimeError, match="not fitted"):
        fu.transform(np.ones((2, 3)))
