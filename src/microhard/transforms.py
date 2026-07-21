"""Shared albumentations pipelines.

Everything pads and crops — never resizes — so the physical scale (µm/px)
encoded in the micrographs is preserved.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .backbone import IMAGENET_NORM, PAD_MULTIPLE


def _pad_divisor() -> A.PadIfNeeded:
    return A.PadIfNeeded(
        min_height=None,
        min_width=None,
        pad_height_divisor=PAD_MULTIPLE,
        pad_width_divisor=PAD_MULTIPLE,
    )


def _augment() -> list:
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ]


def seg_crop_size(image_size: int) -> int:
    """Largest /32 square that fits the configured image size (min 32)."""
    return max(PAD_MULTIPLE, (image_size // PAD_MULTIPLE) * PAD_MULTIPLE)


def seg_train_transforms(image_size: int) -> A.Compose:
    """Square random crops: RandomRotate90 on a non-square image would change
    its orientation mid-batch and break collation (real UHCS frames are
    645x484); square crops keep every batch tensor the same shape."""
    crop = seg_crop_size(image_size)
    return A.Compose(
        [
            A.PadIfNeeded(min_height=crop, min_width=crop),
            A.RandomCrop(crop, crop),
            *_augment(),
            A.Normalize(**IMAGENET_NORM),
            ToTensorV2(),
        ]
    )


def seg_eval_transforms() -> A.Compose:
    return A.Compose([_pad_divisor(), A.Normalize(**IMAGENET_NORM), ToTensorV2()])


def clf_train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=image_size, min_width=image_size),
            A.RandomCrop(image_size, image_size),
            *_augment(),
            A.Normalize(**IMAGENET_NORM),
            ToTensorV2(),
        ]
    )


def clf_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=image_size, min_width=image_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(**IMAGENET_NORM),
            ToTensorV2(),
        ]
    )
