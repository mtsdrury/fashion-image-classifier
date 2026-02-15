"""Tests for src/wandb_utils -- W&B is mocked throughout."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.wandb_utils import _denormalize


# ── _denormalize ───────────────────────────────────────────────────────


def test_denormalize_output_shape():
    tensor = torch.randn(3, 224, 224)
    result = _denormalize(tensor)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_denormalize_output_range():
    tensor = torch.zeros(3, 32, 32)
    result = _denormalize(tensor)
    assert result.min() >= 0
    assert result.max() <= 255


# ── init_wandb_run ─────────────────────────────────────────────────────


def test_init_wandb_run():
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = MagicMock()
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        from src.wandb_utils import init_wandb_run

        init_wandb_run(config={"lr": 0.01}, project="test-project")
        mock_wandb.init.assert_called_once_with(
            project="test-project", entity=None, config={"lr": 0.01}
        )


# ── log_confusion_matrix ──────────────────────────────────────────────


def test_log_confusion_matrix():
    mock_wandb = MagicMock()
    mock_wandb.plot.confusion_matrix.return_value = "cm_plot"
    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        from src.wandb_utils import log_confusion_matrix

        log_confusion_matrix(
            y_true=[0, 1, 2],
            y_pred=[0, 1, 1],
            class_names=["a", "b", "c"],
        )
        mock_wandb.log.assert_called_once()
