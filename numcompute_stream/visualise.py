"""
visualise.py — Visualisation utilities for streaming ML pipelines.

Provides reusable matplotlib-based plotting functions for monitoring
model performance over streaming chunks. Supports saving to file or
inline display in notebooks.

Author: Sujeet Ghosh
Module: numcompute_stream.visualise
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_metric_over_time(
    metric_values: list[float] | np.ndarray,
    title: str = "Metric over time",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    color: str = "#4C72B0",
    show_rolling: bool = True,
    rolling_window: int = 5,
    save_path: str | None = None,
    figsize: tuple = (9, 4),
) -> plt.Figure:
    """Plot a metric across streaming chunks.

    Parameters
    ----------
    metric_values : list of float or np.ndarray
    title : str
    ylabel : str
    xlabel : str
    color : str
    show_rolling : bool
    rolling_window : int
    save_path : str or None
        If provided, saves the figure instead of displaying it.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_metric_over_time(trainer.get_accuracy_history(), ylabel='Accuracy')
    """
    values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(chunks, values, color=color, linewidth=1.5,
            marker="o", markersize=4, alpha=0.7, label=ylabel)

    if show_rolling and len(values) >= rolling_window:
        kernel = np.ones(rolling_window) / rolling_window
        rolling = np.convolve(values, kernel, mode="valid")
        ax.plot(chunks[rolling_window - 1:], rolling, color=color, linewidth=2.5,
                linestyle="--", alpha=1.0, label=f"Rolling mean (w={rolling_window})")

    if len(values) > 0:
        ax.axhline(np.nanmax(values), color="grey", linewidth=0.8,
                   linestyle=":", alpha=0.6, label=f"Max: {np.nanmax(values):.3f}")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig


def compare_models(
    metric1: list[float] | np.ndarray,
    metric2: list[float] | np.ndarray,
    labels: tuple[str, str] = ("Model 1", "Model 2"),
    title: str = "Model comparison over time",
    ylabel: str = "Accuracy",
    xlabel: str = "Chunk",
    colors: tuple[str, str] = ("#4C72B0", "#DD8452"),
    save_path: str | None = None,
    figsize: tuple = (9, 4),
) -> plt.Figure:
    """Compare two models on streaming metrics over time.

    Parameters
    ----------
    metric1 : list or np.ndarray
    metric2 : list or np.ndarray
    labels : tuple of str
    title : str
    ylabel : str
    xlabel : str
    colors : tuple of str
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> compare_models(tree_accs, rf_accs, labels=('Single Tree', 'Random Forest'))
    """
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    chunks1 = np.arange(1, len(m1) + 1)
    chunks2 = np.arange(1, len(m2) + 1)

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1, figsize=(figsize[0], figsize[1] * 1.4),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )

    ax_main.plot(chunks1, m1, color=colors[0], linewidth=2,
                 marker="o", markersize=4, label=labels[0])
    ax_main.plot(chunks2, m2, color=colors[1], linewidth=2,
                 marker="s", markersize=4, label=labels[1])

    min_len = min(len(m1), len(m2))
    if min_len > 0:
        ax_main.fill_between(
            np.arange(1, min_len + 1), m1[:min_len], m2[:min_len],
            alpha=0.12, color="grey"
        )

    ax_main.set_title(title, fontsize=13, pad=10)
    ax_main.set_ylabel(ylabel, fontsize=11)
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3, linestyle="--")
    ax_main.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if min_len > 0:
        diff = m1[:min_len] - m2[:min_len]
        x_shared = np.arange(1, min_len + 1)
        ax_diff.bar(x_shared, diff,
                    color=["#4C72B0" if d >= 0 else "#DD8452" for d in diff],
                    alpha=0.7, width=0.7)
        ax_diff.axhline(0, color="black", linewidth=0.8)
        ax_diff.set_ylabel(f"{labels[0]} − {labels[1]}", fontsize=9)
        ax_diff.set_xlabel(xlabel, fontsize=11)
        ax_diff.grid(True, alpha=0.25, linestyle="--")
        ax_diff.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig


def plot_predictions_vs_ground_truth(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    title: str = "Predictions vs ground truth (latest chunk)",
    max_samples: int = 100,
    save_path: str | None = None,
    figsize: tuple = (10, 3),
) -> plt.Figure:
    """Visualise predictions vs actual labels on the latest chunk.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    title : str
    max_samples : int
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_predictions_vs_ground_truth(y_true_chunk, y_pred_chunk)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) > max_samples:
        y_true = y_true[-max_samples:]
        y_pred = y_pred[-max_samples:]

    n = len(y_true)
    correct = y_true == y_pred

    fig, (ax_strip, ax_bar) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=12)

    colors_strip = ["#2ca02c" if c else "#d62728" for c in correct]
    ax_strip.scatter(np.arange(n), y_true, c=colors_strip, marker="|", s=60,
                     linewidths=1.5, label="True label")
    ax_strip.scatter(np.arange(n), y_pred, c=colors_strip, marker=".", s=20,
                     alpha=0.5, label="Predicted")

    for i in range(n):
        if not correct[i]:
            ax_strip.plot([i, i], [y_true[i], y_pred[i]],
                          color="#d62728", linewidth=0.5, alpha=0.4)

    acc = float(np.mean(correct))
    ax_strip.set_xlabel("Sample index", fontsize=10)
    ax_strip.set_ylabel("Class label", fontsize=10)
    ax_strip.set_title(
        f"n={n} | acc={acc:.3f} | correct={correct.sum()} | wrong={n - correct.sum()}",
        fontsize=9
    )
    ax_strip.grid(True, alpha=0.2, linestyle="--")

    classes = np.unique(y_true)
    class_accs = [
        float(np.mean(y_pred[y_true == c] == c)) if (y_true == c).sum() > 0 else 0.0
        for c in classes
    ]
    ax_bar.barh(
        [str(c) for c in classes], class_accs,
        color=["#4C72B0" if a >= 0.7 else "#DD8452" if a >= 0.4 else "#d62728" for a in class_accs],
        edgecolor="white", height=0.6
    )
    ax_bar.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax_bar.set_xlim(0, 1.1)
    ax_bar.set_xlabel("Per-class accuracy", fontsize=10)
    ax_bar.set_title("By class", fontsize=9)
    ax_bar.grid(True, alpha=0.25, axis="x", linestyle="--")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion matrix",
    cmap: str = "Blues",
    save_path: str | None = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
    class_names : list of str or None
    title : str
    cmap : str
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig


def plot_fit_times(
    fit_times_ms: list[float],
    title: str = "Training time per chunk",
    save_path: str | None = None,
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Plot fit time in milliseconds per chunk.

    Parameters
    ----------
    fit_times_ms : list of float
    title : str
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = np.asarray(fit_times_ms)
    chunks = np.arange(1, len(times) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(chunks, times, color="#5fa8d3", edgecolor="white", width=0.7)
    ax.axhline(np.mean(times), color="red", linewidth=1.5,
               linestyle="--", label=f"Mean: {np.mean(times):.1f}ms")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Chunk", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig
