"""Parse styles.csv, filter categories, build DataLoaders."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    IMAGES_DIR,
    MIN_CATEGORY_COUNT,
    SEED,
    STYLES_CSV,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)
from src.transforms import get_eval_transforms, get_train_transforms


# ── Metadata helpers ───────────────────────────────────────────────────


def load_metadata(csv_path: Path = STYLES_CSV) -> pd.DataFrame:
    """Load styles.csv and return a DataFrame with id and articleType."""
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df[["id", "articleType"]].dropna()
    df["id"] = df["id"].astype(int)
    return df


def filter_top_categories(
    df: pd.DataFrame,
    min_count: int = MIN_CATEGORY_COUNT,
) -> pd.DataFrame:
    """Keep only categories with at least *min_count* images."""
    counts = df["articleType"].value_counts()
    keep = counts[counts >= min_count].index
    return df[df["articleType"].isin(keep)].reset_index(drop=True)


def verify_images_exist(
    df: pd.DataFrame,
    images_dir: Path = IMAGES_DIR,
) -> pd.DataFrame:
    """Drop rows whose image file is missing on disk."""
    exists_mask = df["id"].apply(
        lambda pid: (images_dir / f"{pid}.jpg").is_file()
    )
    return df[exists_mask].reset_index(drop=True)


def encode_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """Add an integer 'label' column; return label-to-index and index-to-label maps."""
    categories = sorted(df["articleType"].unique())
    label_to_idx = {cat: i for i, cat in enumerate(categories)}
    idx_to_label = {i: cat for cat, i in label_to_idx.items()}
    df = df.copy()
    df["label"] = df["articleType"].map(label_to_idx)
    return df, label_to_idx, idx_to_label


# ── Train / val / test split ──────────────────────────────────────────


def split_data(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train, validation, and test sets."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label"], random_state=seed
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── PyTorch Dataset ───────────────────────────────────────────────────


class FashionDataset(Dataset):
    """Map-style dataset that loads product images and returns (image, label)."""

    def __init__(self, df: pd.DataFrame, images_dir: Path = IMAGES_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ── DataLoader builder ────────────────────────────────────────────────


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: Path = IMAGES_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    augment_flip: bool = True,
    augment_rotation: float = 15.0,
    augment_color_jitter: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    train_transform = get_train_transforms(
        flip=augment_flip,
        rotation=augment_rotation,
        color_jitter=augment_color_jitter,
    )
    eval_transform = get_eval_transforms()

    train_ds = FashionDataset(train_df, images_dir, transform=train_transform)
    val_ds = FashionDataset(val_df, images_dir, transform=eval_transform)
    test_ds = FashionDataset(test_df, images_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
