"""Tests for src/training -- uses tiny model and synthetic data, no GPU needed."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import SimpleCNN
from src.training import evaluate, train_model, train_one_epoch


def _make_tiny_loader(n: int = 16, num_classes: int = 3, batch_size: int = 4):
    """Create a DataLoader with random images and labels."""
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


NUM_CLASSES = 3


def test_train_one_epoch_returns_loss_and_accuracy():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    loader = _make_tiny_loader(num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_returns_predictions():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    loader = _make_tiny_loader(n=8, num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss, acc, y_true, y_pred = evaluate(model, loader, criterion, device)
    assert isinstance(loss, float)
    assert len(y_true) == 8
    assert len(y_pred) == 8


def test_train_model_without_wandb():
    model = SimpleCNN(num_classes=NUM_CLASSES)
    train_loader = _make_tiny_loader(num_classes=NUM_CLASSES)
    val_loader = _make_tiny_loader(n=8, num_classes=NUM_CLASSES)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        lr=0.01,
        device=torch.device("cpu"),
        use_wandb=False,
    )
    assert "train_loss" in results
    assert "val_accuracy" in results
    assert "best_val_accuracy" in results
