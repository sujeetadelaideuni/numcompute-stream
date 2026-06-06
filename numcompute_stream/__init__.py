"""
numcompute_stream — Streaming ML framework built on NumCompute.

A decision tree–based machine learning framework supporting streaming
(incremental) learning and ensemble methods. Built using only NumPy
and matplotlib.

Modules
-------
stats          : Descriptive + streaming statistics (StreamingStats)
metrics        : Classification metrics + streaming accumulator (StreamingMetrics)
preprocessing  : Scalers, imputers with .partial_fit() support
tree           : DecisionTreeClassifier with streaming support
ensemble       : EnsembleClassifier (Bagging, Random Forest) with streaming support
pipeline       : Pipeline and FeatureUnion with .partial_fit()
stream         : StreamTrainer for orchestrating streaming learning
visualise      : Plotting utilities for streaming metrics and predictions
io             : CSV loading/saving including chunked streaming reads

Usage
-----
    from numcompute_stream.pipeline import Pipeline
    from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
    from numcompute_stream.ensemble import EnsembleClassifier
    from numcompute_stream.stream import StreamTrainer
    from numcompute_stream import visualise

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   EnsembleClassifier(n_estimators=10, method='random_forest')),
    ])

    trainer = StreamTrainer(pipeline=pipe, verbose=True)

    for X_chunk, y_chunk in data_stream:
        trainer.fit_chunk(X_chunk, y_chunk)
        trainer.score_chunk(X_chunk, y_chunk)

    visualise.plot_metric_over_time(
        trainer.get_accuracy_history(),
        title='Accuracy over chunks',
        ylabel='Accuracy',
    )
"""

from . import stats, metrics, preprocessing, tree, ensemble, pipeline, stream, visualise

__version__ = "2.0.0"
__all__ = [
    "stats",
    "metrics",
    "preprocessing",
    "tree",
    "ensemble",
    "pipeline",
    "stream",
    "visualise",
]
