"""W&B helper functions for logging images, confusion matrices, and predictions."""

import numpy as np
import torch

from src.config import IMAGENET_MEAN, IMAGENET_STD


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to uint8 HWC numpy array for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def init_wandb_run(config: dict, project: str, entity: str | None = None, **kwargs):
    """Initialize a W&B run. Returns the run object."""
    import wandb

    return wandb.init(project=project, entity=entity, config=config, **kwargs)


def log_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str = "Confusion Matrix",
):
    """Log a confusion matrix to the current W&B run."""
    import wandb

    wandb.log(
        {
            title: wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=class_names
            )
        }
    )


def log_sample_predictions(
    images: torch.Tensor,
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    class_names: list[str],
    max_images: int = 16,
):
    """Log a grid of sample predictions (correct and incorrect) to W&B."""
    import wandb

    table = wandb.Table(columns=["image", "true", "predicted", "confidence", "correct"])
    n = min(len(y_true), max_images)
    for i in range(n):
        img_array = _denormalize(images[i])
        table.add_data(
            wandb.Image(img_array),
            class_names[y_true[i]],
            class_names[y_pred[i]],
            f"{confidences[i]:.3f}",
            y_true[i] == y_pred[i],
        )
    wandb.log({"sample_predictions": table})
