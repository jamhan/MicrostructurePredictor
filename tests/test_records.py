from pathlib import Path

import pytest
import torch

from microhard.records import (
    CanonicalRecord,
    RecordClassificationDataset,
    RecordSegmentationDataset,
    split_records_by_group,
)
from microhard.adapters.base import BaseAdapter
from tests.conftest import write_image


def _record(i: int, group: str, tmp_path: Path, label: str | None = "ferrous/pearlite"):
    image = tmp_path / f"r{i}.png"
    if not image.exists():
        write_image(image, seed=i)
    return CanonicalRecord(
        record_id=f"r{i}",
        image_path=image,
        scale_um_per_px=0.1,
        modality="SEM",
        group_id=group,
        taxonomy_labels=(label,) if label else None,
    )


def test_group_id_defaults_to_record_id(tmp_path: Path) -> None:
    record = CanonicalRecord(
        record_id="x", image_path=tmp_path / "x.png", scale_um_per_px=None, modality="SEM"
    )
    assert record.group_id == "x"


def test_property_weight_defaults_distinguish_direct_and_distant(tmp_path: Path) -> None:
    direct = CanonicalRecord(
        record_id="direct",
        image_path=tmp_path / "direct.png",
        scale_um_per_px=None,
        modality="SEM",
        properties={"hardness_hv": 300},
    )
    distant = CanonicalRecord(
        record_id="distant",
        image_path=tmp_path / "distant.png",
        scale_um_per_px=None,
        modality="SEM",
        properties={"hardness_hv": 300},
        property_sources={"hardness_hv": "distant"},
    )
    assert direct.property_weight("hardness_hv") == 1.0
    assert distant.property_weight("hardness_hv") == 0.5


def test_invalid_property_weight_is_rejected(tmp_path: Path) -> None:
    record = CanonicalRecord(
        record_id="bad",
        image_path=tmp_path / "bad.png",
        scale_um_per_px=None,
        modality="SEM",
        properties={"hardness_hv": 300},
        property_weights={"hardness_hv": 1.5},
    )
    with pytest.raises(ValueError, match="finite in"):
        BaseAdapter._check_property_weights(record)


def test_split_by_group_no_leakage(tmp_path: Path) -> None:
    records = [_record(i, f"g{i // 3}", tmp_path) for i in range(15)]  # 5 groups x 3
    splits = split_records_by_group(records, 0.2, 0.2, seed=0)
    seen: dict[str, str] = {}
    for split, split_records in splits.items():
        for record in split_records:
            assert seen.setdefault(record.group_id, split) == split
    assert splits["train"]


def test_split_deterministic(tmp_path: Path) -> None:
    records = [_record(i, f"g{i}", tmp_path) for i in range(10)]
    a = split_records_by_group(records, seed=7)
    b = split_records_by_group(records, seed=7)
    assert {s: [r.record_id for r in v] for s, v in a.items()} == {
        s: [r.record_id for r in v] for s, v in b.items()
    }


def test_split_rejects_degenerate_fractions(tmp_path: Path) -> None:
    records = [_record(0, "g0", tmp_path)]
    with pytest.raises(ValueError):
        split_records_by_group(records, 0.6, 0.5)


def test_classification_dataset(tmp_path: Path) -> None:
    records = [
        _record(0, "g0", tmp_path, "ferrous/pearlite"),
        _record(1, "g0", tmp_path, "ferrous/martensite"),
        _record(2, "g1", tmp_path, None),  # unlabeled -> dropped
        _record(3, "g1", tmp_path, "ferrous/unregistered-not-in-classes"),  # dropped
    ]
    classes = [("ferrous/martensite",), ("ferrous/pearlite",)]
    ds = RecordClassificationDataset(
        records, classes, lambda r: r.taxonomy_labels
    )
    assert len(ds) == 2
    image, label = ds[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert classes[label] == ("ferrous/pearlite",)


def test_segmentation_dataset_filters_maskless(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    with_mask = _record(0, "g0", tmp_path)
    mask_path = tmp_path / "m0.png"
    Image.fromarray(np.ones((64, 64), dtype=np.uint8)).save(mask_path)
    object.__setattr__(with_mask, "mask_path", mask_path)  # frozen dataclass, test-only
    without_mask = _record(1, "g1", tmp_path)

    ds = RecordSegmentationDataset([with_mask, without_mask])
    assert len(ds) == 1
    image, mask = ds[0]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.long
