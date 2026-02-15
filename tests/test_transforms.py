"""Tests for src/transforms."""

import torch
from PIL import Image

from src.transforms import get_eval_transforms, get_train_transforms


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color="blue")


def test_eval_transforms_output_shape():
    transform = get_eval_transforms()
    tensor = transform(_dummy_image())
    assert tensor.shape == (3, 224, 224)


def test_train_transforms_output_shape():
    transform = get_train_transforms()
    tensor = transform(_dummy_image())
    assert tensor.shape == (3, 224, 224)


def test_train_transforms_no_augmentation():
    transform = get_train_transforms(flip=False, rotation=0, color_jitter=False)
    tensor = transform(_dummy_image())
    assert tensor.shape == (3, 224, 224)


def test_transforms_output_is_float_tensor():
    transform = get_eval_transforms()
    tensor = transform(_dummy_image())
    assert tensor.dtype == torch.float32
