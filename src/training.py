"""Training and evaluation loops with optional W&B logging."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEFAULT_EPOCHS, DEFAULT_LR
from src.evaluation import compute_accuracy
from src.models import build_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = compute_accuracy(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model on a data loader. Returns (avg_loss, accuracy, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = compute_accuracy(all_labels, all_preds)
    return avg_loss, accuracy, all_labels, all_preds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: torch.device | None = None,
    use_wandb: bool = False,
    class_names: list[str] | None = None,
) -> dict:
    """Full training pipeline with optional W&B logging.

    Returns a dict with final train/val loss and accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_true, val_pred = evaluate(
            model, val_loader, criterion, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                }
            )

    # Log final confusion matrix
    if use_wandb and class_names:
        from src.wandb_utils import log_confusion_matrix

        log_confusion_matrix(val_true, val_pred, class_names, title="val/confusion_matrix")

    return {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "best_val_accuracy": best_val_acc,
    }


# ── Sweep entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import wandb

    from src.config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_DROPOUT,
        DEFAULT_EPOCHS,
        DEFAULT_LR,
        IMAGES_DIR,
        STYLES_CSV,
    )
    from src.data_loader import (
        build_dataloaders,
        encode_labels,
        filter_top_categories,
        load_metadata,
        split_data,
        verify_images_exist,
    )

    wandb.init()
    config = wandb.config

    # Load and prepare data
    df = load_metadata(STYLES_CSV)
    df = filter_top_categories(df)
    df = verify_images_exist(df, IMAGES_DIR)
    df, label_to_idx, idx_to_label = encode_labels(df)
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    train_df, val_df, test_df = split_data(df)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        images_dir=IMAGES_DIR,
        batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
        augment_flip=config.get("augment_flip", True),
        augment_rotation=config.get("augment_rotation", 15.0),
        augment_color_jitter=config.get("augment_color_jitter", True),
    )

    model = build_model(
        name=config.get("model_name", "simple_cnn"),
        num_classes=len(class_names),
        dropout=config.get("dropout", DEFAULT_DROPOUT),
        pretrained=config.get("model_name", "simple_cnn") != "simple_cnn",
    )

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get("epochs", DEFAULT_EPOCHS),
        lr=config.get("learning_rate", DEFAULT_LR),
        use_wandb=True,
        class_names=class_names,
    )
