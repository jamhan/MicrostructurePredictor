"""Canonical records: the common format between dataset adapters and tasks.

Every adapter emits ``CanonicalRecord`` instances and every task (router,
classifier, segmenter, feature extraction) consumes them. Image, scale, and
modality are the core fields; labels, masks, and properties are optional so
partially annotated datasets still work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CanonicalRecord:
    """One micrograph in canonical form.

    Required (must be passed explicitly, though scale may be None when the
    source metadata genuinely lacks it):
      record_id, image_path, scale_um_per_px, modality
    Optional annotations:
      group_id         split unit (physical sample); defaults to record_id
      taxonomy_labels  taxonomy node ids describing the image
      mask_path        pixel-level label mask, if this record is in a benchmark
      mask_class_nodes taxonomy node id for each integer mask class (index = class)
      properties       measured sample properties, e.g. {"hardness_hv": 310.0}
    """

    record_id: str
    image_path: Path
    scale_um_per_px: float | None
    modality: str
    group_id: str = ""
    taxonomy_labels: tuple[str, ...] | None = None
    mask_path: Path | None = None
    mask_class_nodes: tuple[str, ...] | None = None
    properties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id:
            object.__setattr__(self, "group_id", self.record_id)


def split_records_by_group(
    records: Sequence[CanonicalRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[CanonicalRecord]]:
    """Deterministic train/val/test split assigned per group_id, so records of
    the same physical sample never straddle splits (no leakage)."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(sorted({r.group_id for r in records}))
    n = len(groups)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if n - n_val - n_test < 1:
        raise ValueError(f"split fractions leave no training groups (n_groups={n})")
    assignment = {g: "test" for g in groups[:n_test]}
    assignment |= {g: "val" for g in groups[n_test : n_test + n_val]}
    assignment |= {g: "train" for g in groups[n_test + n_val :]}
    out: dict[str, list[CanonicalRecord]] = {s: [] for s in SPLITS}
    for record in records:
        out[assignment[record.group_id]].append(record)
    return out


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0


def load_image(record: CanonicalRecord) -> np.ndarray:
    return np.asarray(Image.open(record.image_path).convert("RGB"))


class RecordClassificationDataset(Dataset):
    """(image, class index) pairs over records, for any classification task.

    ``label_fn`` maps a record to a hashable label (e.g. its sorted
    taxonomy_labels tuple, or its family id); ``classes`` fixes the label
    order. Records whose label is None or unknown are dropped.
    """

    def __init__(
        self,
        records: Sequence[CanonicalRecord],
        classes: Sequence[Hashable],
        label_fn: Callable[[CanonicalRecord], Hashable | None],
        transform=None,
    ) -> None:
        self.classes = list(classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.items = [
            (r, class_to_idx[label])
            for r in records
            if (label := label_fn(r)) is not None and label in class_to_idx
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record, label = self.items[idx]
        image = load_image(record)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = _to_tensor(image)
        return image, label


class RecordSegmentationDataset(Dataset):
    """(image, mask) pairs over records that carry a mask_path."""

    def __init__(self, records: Sequence[CanonicalRecord], transform=None) -> None:
        self.records = [r for r in records if r.mask_path is not None]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        image = load_image(record)
        mask = np.asarray(Image.open(record.mask_path)).astype(np.int64)
        if mask.ndim == 3:  # palette/RGB-encoded masks: class id in channel 0
            mask = mask[..., 0]
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image = _to_tensor(image)
        return image, torch.as_tensor(np.asarray(mask), dtype=torch.long)
