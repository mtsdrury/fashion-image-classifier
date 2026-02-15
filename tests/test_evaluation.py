"""Tests for src/evaluation -- all use known values, no model or data needed."""

import numpy as np

from src.evaluation import (
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix,
    compute_topk_accuracy,
)


# ── compute_accuracy ───────────────────────────────────────────────────


def test_accuracy_perfect():
    assert compute_accuracy([0, 1, 2, 0], [0, 1, 2, 0]) == 1.0


def test_accuracy_half():
    assert compute_accuracy([0, 1, 0, 1], [0, 0, 0, 0]) == 0.5


def test_accuracy_zero():
    assert compute_accuracy([0, 0, 0], [1, 1, 1]) == 0.0


# ── compute_classification_report ─────────────────────────────────────


def test_report_returns_dict():
    report = compute_classification_report([0, 1, 0, 1], [0, 1, 1, 0])
    assert isinstance(report, dict)
    assert "accuracy" in report


def test_report_with_label_names():
    report = compute_classification_report(
        [0, 1, 0, 1], [0, 1, 0, 1], label_names=["cat", "dog"]
    )
    assert "cat" in report
    assert "dog" in report


# ── compute_confusion_matrix ──────────────────────────────────────────


def test_confusion_matrix_shape():
    cm = compute_confusion_matrix([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 2, 2])
    assert cm.shape == (3, 3)


def test_confusion_matrix_perfect():
    cm = compute_confusion_matrix([0, 1, 2], [0, 1, 2])
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(cm, expected)


# ── compute_topk_accuracy ─────────────────────────────────────────────


def test_topk_perfect_top1():
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    assert compute_topk_accuracy([0, 1, 2], probs, k=1) == 1.0


def test_topk_top3_catches_all():
    # True labels are always in top 3 when there are only 3 classes
    probs = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]])
    assert compute_topk_accuracy([0, 1, 2], probs, k=3) == 1.0
