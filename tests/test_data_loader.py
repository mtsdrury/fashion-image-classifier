"""Tests for src/data_loader -- uses synthetic CSV and tiny dummy images."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data_loader import (
    FashionDataset,
    build_dataloaders,
    encode_labels,
    filter_top_categories,
    load_metadata,
    split_data,
    verify_images_exist,
)
from src.transforms import get_eval_transforms


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Create a minimal styles.csv and matching 8x8 dummy images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    # 6 categories: 3 with >=5 images, 3 with <5
    categories = {
        "Tshirts": 8,
        "Shirts": 6,
        "Shoes": 5,
        "Ties": 2,
        "Belts": 1,
        "Socks": 3,
    }
    pid = 1000
    for cat, count in categories.items():
        for _ in range(count):
            rows.append({"id": pid, "articleType": cat})
            img = Image.new("RGB", (8, 8), color="red")
            img.save(images_dir / f"{pid}.jpg")
            pid += 1

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "styles.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, images_dir, df


# ── load_metadata ──────────────────────────────────────────────────────


def test_load_metadata_returns_expected_columns(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    assert list(df.columns) == ["id", "articleType"]
    assert len(df) == 25  # 8+6+5+2+1+3


def test_load_metadata_drops_na(tmp_path):
    csv_path = tmp_path / "styles.csv"
    pd.DataFrame(
        {"id": [1, 2, 3], "articleType": ["Shirts", None, "Shoes"]}
    ).to_csv(csv_path, index=False)
    df = load_metadata(csv_path)
    assert len(df) == 2


# ── filter_top_categories ─────────────────────────────────────────────


def test_filter_keeps_categories_above_threshold(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    filtered = filter_top_categories(df, min_count=5)
    remaining = set(filtered["articleType"].unique())
    assert remaining == {"Tshirts", "Shirts", "Shoes"}


def test_filter_removes_small_categories(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    filtered = filter_top_categories(df, min_count=5)
    assert "Ties" not in filtered["articleType"].values
    assert "Belts" not in filtered["articleType"].values
    assert "Socks" not in filtered["articleType"].values


# ── verify_images_exist ───────────────────────────────────────────────


def test_verify_drops_missing_images(tmp_dataset):
    csv_path, images_dir, _ = tmp_dataset
    df = load_metadata(csv_path)
    # Remove one image file
    first_id = df["id"].iloc[0]
    (images_dir / f"{first_id}.jpg").unlink()

    verified = verify_images_exist(df, images_dir)
    assert len(verified) == len(df) - 1
    assert first_id not in verified["id"].values


def test_verify_keeps_all_when_none_missing(tmp_dataset):
    csv_path, images_dir, _ = tmp_dataset
    df = load_metadata(csv_path)
    verified = verify_images_exist(df, images_dir)
    assert len(verified) == len(df)


# ── encode_labels ──────────────────────────────────────────────────────


def test_encode_labels_creates_mapping(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    df = filter_top_categories(df, min_count=5)
    df, label_to_idx, idx_to_label = encode_labels(df)

    assert "label" in df.columns
    assert len(label_to_idx) == 3  # Tshirts, Shirts, Shoes
    assert len(idx_to_label) == 3
    # Round-trip
    for cat, idx in label_to_idx.items():
        assert idx_to_label[idx] == cat


# ── split_data ─────────────────────────────────────────────────────────


def test_split_preserves_total_rows(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    df = filter_top_categories(df, min_count=5)
    df, _, _ = encode_labels(df)
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_no_overlap(tmp_dataset):
    csv_path, _, _ = tmp_dataset
    df = load_metadata(csv_path)
    df = filter_top_categories(df, min_count=5)
    df, _, _ = encode_labels(df)
    train, val, test = split_data(df)

    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


# ── FashionDataset ─────────────────────────────────────────────────────


def test_dataset_len_and_getitem(tmp_dataset):
    csv_path, images_dir, _ = tmp_dataset
    df = load_metadata(csv_path)
    df = filter_top_categories(df, min_count=5)
    df, _, _ = encode_labels(df)

    transform = get_eval_transforms()
    ds = FashionDataset(df, images_dir, transform=transform)

    assert len(ds) == len(df)
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)


# ── build_dataloaders ─────────────────────────────────────────────────


def test_build_dataloaders_returns_three_loaders(tmp_dataset):
    csv_path, images_dir, _ = tmp_dataset
    df = load_metadata(csv_path)
    df = filter_top_categories(df, min_count=5)
    df, _, _ = encode_labels(df)
    train_df, val_df, test_df = split_data(df)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        images_dir=images_dir,
        batch_size=4,
        num_workers=0,
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
