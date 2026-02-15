"""Pure evaluation functions (no W&B dependency)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """Per-class precision, recall, F1 as a dict."""
    return classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )


def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> np.ndarray:
    """Confusion matrix as a 2-D numpy array."""
    return confusion_matrix(y_true, y_pred)


def compute_topk_accuracy(
    y_true: list[int],
    y_probs: np.ndarray,
    k: int = 3,
) -> float:
    """Top-k accuracy from probability matrix.

    Parameters
    ----------
    y_true : list[int]
        True label indices.
    y_probs : np.ndarray
        Shape (n_samples, n_classes) probability matrix.
    k : int
        Number of top predictions to consider.
    """
    labels = list(range(y_probs.shape[1]))
    return float(top_k_accuracy_score(y_true, y_probs, k=k, labels=labels))
