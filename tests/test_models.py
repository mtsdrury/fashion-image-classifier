"""Tests for src/models -- SimpleCNN uses real forward pass; ResNet/EfficientNet mock the backbone."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.models import (
    EfficientNetB0Classifier,
    ResNet50Classifier,
    SimpleCNN,
    build_model,
)

NUM_CLASSES = 10
BATCH_SIZE = 2


# ── SimpleCNN ──────────────────────────────────────────────────────────


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


def test_simple_cnn_different_num_classes():
    for nc in [5, 15, 20]:
        model = SimpleCNN(num_classes=nc)
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, nc)


def test_simple_cnn_custom_dropout():
    model = SimpleCNN(num_classes=NUM_CLASSES, dropout=0.5)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


# ── ResNet-50 (mocked backbone) ───────────────────────────────────────


def _make_mock_resnet(*args, **kwargs):
    """Return a tiny model that mimics ResNet-50 interface (has .fc attribute)."""
    mock_model = nn.Module()
    mock_model.fc = nn.Linear(2048, 1000)

    # Make forward work by using a simple pipeline
    features = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 2048),
    )
    mock_model.features = features

    def forward(x):
        x = features(x)
        return mock_model.fc(x)

    mock_model.forward = forward
    return mock_model


@patch("src.models.models.resnet50", _make_mock_resnet)
def test_resnet50_output_shape():
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


@patch("src.models.models.resnet50", _make_mock_resnet)
def test_resnet50_replaces_fc_head():
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)
    # The final layer should output num_classes, not 1000
    fc = model.backbone.fc
    assert isinstance(fc, nn.Sequential)
    linear = fc[-1]
    assert isinstance(linear, nn.Linear)
    assert linear.out_features == NUM_CLASSES


# ── EfficientNet-B0 (mocked backbone) ─────────────────────────────────


def _make_mock_efficientnet(*args, **kwargs):
    """Return a tiny model that mimics EfficientNet-B0 interface."""
    mock_model = nn.Module()
    mock_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1000),
    )

    features = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 1280),
    )
    mock_model.features = features

    def forward(x):
        x = features(x)
        return mock_model.classifier(x)

    mock_model.forward = forward
    return mock_model


@patch("src.models.models.efficientnet_b0", _make_mock_efficientnet)
def test_efficientnet_b0_output_shape():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


@patch("src.models.models.efficientnet_b0", _make_mock_efficientnet)
def test_efficientnet_b0_replaces_classifier():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    classifier = model.backbone.classifier
    assert isinstance(classifier, nn.Sequential)
    linear = classifier[-1]
    assert isinstance(linear, nn.Linear)
    assert linear.out_features == NUM_CLASSES


# ── build_model factory ───────────────────────────────────────────────


def test_build_model_simple_cnn():
    model = build_model("simple_cnn", num_classes=NUM_CLASSES)
    assert isinstance(model, SimpleCNN)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent", num_classes=NUM_CLASSES)
