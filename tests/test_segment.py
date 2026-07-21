import numpy as np
import pytest
import torch

from microhard.adapters.uhcs import SEG_CLASS_NODES
from microhard.segment import (
    build_segmentation_model,
    load_segmenter,
    mask_records,
    segment_image,
    train_segmentation,
)
from microhard.taxonomy import Taxonomy


def test_seg_train_transforms_square_output_from_nonsquare_input() -> None:
    """Real UHCS frames are 645x484; RandomRotate90 must not produce
    mixed-orientation tensors within a batch (regression: collate crash)."""
    import numpy as np

    from microhard.transforms import seg_crop_size, seg_train_transforms

    assert seg_crop_size(484) == 480
    transforms = seg_train_transforms(484)
    image = np.random.default_rng(0).integers(0, 255, (484, 645, 3), dtype=np.uint8)
    mask = np.zeros((484, 645), dtype=np.int64)
    for _ in range(8):  # augmentation is stochastic — try several draws
        out = transforms(image=image, mask=mask)
        assert out["image"].shape == (3, 480, 480)
        assert out["mask"].shape == (480, 480)


def test_segment_image_pads_and_crops(cfg) -> None:
    model = build_segmentation_model(cfg, num_classes=4)
    image = np.random.default_rng(0).integers(0, 255, (70, 50, 3), dtype=np.uint8)
    mask = segment_image(model, image, torch.device("cpu"))
    assert mask.shape == (70, 50)  # non-multiple-of-32 input comes back unpadded
    assert set(np.unique(mask)) <= set(range(4))


def test_mask_records_requires_benchmark(synthetic_db) -> None:
    with pytest.raises(FileNotFoundError, match="11256/964"):
        mask_records(synthetic_db, Taxonomy.load(None))


def test_mask_records_finds_benchmark(seg_benchmark) -> None:
    records, class_nodes = mask_records(seg_benchmark, Taxonomy.load(None))
    assert len(records) == 4
    assert class_nodes == SEG_CLASS_NODES


def test_train_reload_and_segment(seg_benchmark) -> None:
    checkpoint = train_segmentation(seg_benchmark)
    assert checkpoint.exists()

    model, class_nodes = load_segmenter(seg_benchmark)
    assert class_nodes == SEG_CLASS_NODES  # taxonomy ids survive the checkpoint
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = segment_image(model, image, torch.device("cpu"))
    assert mask.shape == (64, 64)
