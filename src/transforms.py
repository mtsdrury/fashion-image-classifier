"""Augmentation and normalization pipelines."""

from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(
    flip: bool = True,
    rotation: float = 15.0,
    color_jitter: bool = True,
) -> transforms.Compose:
    """Build training transform pipeline with configurable augmentation.

    Parameters
    ----------
    flip : bool
        Apply random horizontal flip.
    rotation : float
        Max rotation angle in degrees (0 disables).
    color_jitter : bool
        Apply random brightness/contrast/saturation shifts.
    """
    steps: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if flip:
        steps.append(transforms.RandomHorizontalFlip())
    if rotation > 0:
        steps.append(transforms.RandomRotation(rotation))
    if color_jitter:
        steps.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        )

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def get_eval_transforms() -> transforms.Compose:
    """Resize and normalize only (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
