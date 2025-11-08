"""
Utility metrics for the Adult Income dendritic example.

Provides lightweight wrappers around common binary classification metrics
with fallbacks that keep the training script resilient to degenerate splits.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_numpy(array_like: Iterable) -> np.ndarray:
    """Convert an array-like object to a NumPy array."""
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(list(array_like))


def _lazy_metrics():
    from sklearn import metrics as sk_metrics  # type: ignore

    return sk_metrics


def auc_score(y_true: Iterable, y_prob: Iterable) -> float:
    """Compute ROC AUC, returning NaN if the score is undefined."""
    y_true_np = _to_numpy(y_true)
    y_prob_np = _to_numpy(y_prob)
    sk_metrics = _lazy_metrics()
    try:
        return float(sk_metrics.roc_auc_score(y_true_np, y_prob_np))
    except ValueError:
        return float("nan")


def accuracy_score(y_true: Iterable, y_prob: Iterable, threshold: float = 0.5) -> float:
    """Binary accuracy computed from probabilities and a decision threshold."""
    y_true_np = _to_numpy(y_true)
    y_prob_np = _to_numpy(y_prob)
    y_pred = (y_prob_np >= threshold).astype(np.int32)
    if y_true_np.size == 0:
        return float("nan")
    return float((y_pred == y_true_np).mean())


def f1_score_binary(
    y_true: Iterable, y_prob: Iterable, threshold: float = 0.5
) -> float:
    """Binary F1 score computed from probabilities."""
    y_true_np = _to_numpy(y_true)
    y_prob_np = _to_numpy(y_prob)
    y_pred = (y_prob_np >= threshold).astype(np.int32)
    sk_metrics = _lazy_metrics()
    try:
        return float(sk_metrics.f1_score(y_true_np, y_pred))
    except ValueError:
        return float("nan")
