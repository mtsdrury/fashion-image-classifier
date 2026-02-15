"""Model definitions: SimpleCNN, ResNet-50, EfficientNet-B0."""

import torch
import torch.nn as nn
from torchvision import models

from src.config import DEFAULT_DROPOUT


# ── SimpleCNN (from scratch) ──────────────────────────────────────────


class SimpleCNN(nn.Module):
    """3-block CNN trained from scratch."""

    def __init__(self, num_classes: int, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ── ResNet-50 (pretrained) ────────────────────────────────────────────


class ResNet50Classifier(nn.Module):
    """ResNet-50 with pretrained ImageNet backbone and replaced FC head."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = DEFAULT_DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── EfficientNet-B0 (pretrained) ──────────────────────────────────────


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 with pretrained ImageNet backbone and replaced classifier."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = DEFAULT_DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Model registry + factory ──────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "resnet50": ResNet50Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
}


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a model by registry name.

    Parameters
    ----------
    name : str
        One of the keys in MODEL_REGISTRY.
    num_classes : int
        Number of output classes.
    **kwargs
        Forwarded to the model constructor (e.g. dropout, pretrained).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
