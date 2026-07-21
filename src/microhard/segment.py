"""Segmentation head: U-Net decoder over the frozen shared backbone.

Trains on canonical records that carry pixel masks (for UHCS: the 11256/964
benchmark). Mask integer classes map to taxonomy node ids via each record's
``mask_class_nodes``; the checkpoint stores that node list, so downstream
features are keyed by taxonomy id, never by bare class index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from .adapters import enabled_adapters
from .backbone import build_encoder, freeze, trainable_parameters
from .config import Config
from .records import CanonicalRecord, RecordSegmentationDataset, split_records_by_group
from .taxonomy import Taxonomy
from .transforms import seg_eval_transforms, seg_train_transforms


def build_segmentation_model(cfg: Config, num_classes: int) -> torch.nn.Module:
    """U-Net whose encoder is the shared backbone, frozen; decoder trains."""
    model = smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=None,  # weights come from the shared backbone below
        in_channels=3,
        classes=num_classes,
    )
    model.encoder.load_state_dict(build_encoder(cfg).state_dict())
    freeze(model.encoder)
    return model


def mask_records(cfg: Config, taxonomy: Taxonomy) -> tuple[list[CanonicalRecord], tuple[str, ...]]:
    """All enabled-adapter records with masks, plus their (single, consistent)
    mask class-node tuple."""
    records = [
        r
        for adapter in enabled_adapters(cfg, taxonomy)
        for r in adapter.validated_records()
        if r.mask_path is not None
    ]
    if not records:
        raise FileNotFoundError(
            "No records with masks found. For UHCS, run `bash download.sh` to fetch "
            "the segmentation benchmark (https://hdl.handle.net/11256/964)."
        )
    node_sets = {r.mask_class_nodes for r in records}
    if len(node_sets) > 1:
        raise ValueError(
            f"Records mix different mask class-node tuples ({node_sets}); "
            "train one segmenter per mask taxonomy."
        )
    return records, node_sets.pop()


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> float:
    """Mean IoU over classes present in ground truth or predictions."""
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, masks in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        t, p = masks.reshape(-1), preds.reshape(-1)
        valid = (t >= 0) & (t < num_classes)
        idx = t[valid] * num_classes + p[valid]
        conf += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    intersection = conf.diag().float()
    union = (conf.sum(dim=0) + conf.sum(dim=1)).float() - intersection
    present = union > 0
    if not present.any():
        return 0.0
    return float((intersection[present] / union[present]).mean())


def train_segmentation(cfg: Config) -> Path:
    """Train the decoder head on all masked records; save best-IoU checkpoint."""
    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    records, class_nodes = mask_records(cfg, taxonomy)
    splits = split_records_by_group(records, cfg.val_frac, 0.0, cfg.seed)
    train_records, val_records = splits["train"], splits["val"]
    if not val_records:  # tiny benchmark: validate on train rather than nothing
        print("[microhard] warning: no val groups — validating on the training records")
        val_records = train_records
    print(f"[microhard] {len(train_records)} train / {len(val_records)} val masked records")

    train_loader = DataLoader(
        RecordSegmentationDataset(train_records, seg_train_transforms(cfg.image_size)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        RecordSegmentationDataset(val_records, seg_eval_transforms()),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    device = cfg.resolve_device()
    model = build_segmentation_model(cfg, num_classes=len(class_nodes)).to(device)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        model.encoder.eval()  # frozen backbone: BN statistics must not drift
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = dice(logits, masks) + ce(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        mean_iou = evaluate_segmentation(model, val_loader, device, len(class_nodes))
        marker = ""
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": cfg.encoder,
                    "class_nodes": list(class_nodes),
                },
                cfg.seg_checkpoint,
            )
            marker = "  (saved)"
        n_train = max(len(train_records), 1)
        print(
            f"epoch {epoch + 1:>3}/{cfg.epochs}  "
            f"loss {total_loss / n_train:.3f}  val mIoU {mean_iou:.3f}{marker}"
        )
    print(f"[microhard] best val mIoU {best_iou:.3f} -> {cfg.seg_checkpoint}")
    return cfg.seg_checkpoint


def load_segmenter(cfg: Config) -> tuple[torch.nn.Module, tuple[str, ...]]:
    """(model, mask class-node ids) from the checkpoint."""
    if not cfg.seg_checkpoint.exists():
        raise FileNotFoundError(
            f"No segmentation checkpoint at {cfg.seg_checkpoint}. Run `microhard train-seg` first."
        )
    ckpt = torch.load(cfg.seg_checkpoint, map_location="cpu", weights_only=True)
    class_nodes = tuple(ckpt["class_nodes"])
    model = smp.Unet(
        encoder_name=ckpt.get("encoder", cfg.encoder),
        encoder_weights=None,
        in_channels=3,
        classes=len(class_nodes),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_nodes


@torch.no_grad()
def segment_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Per-pixel class mask (H, W) for an RGB uint8 image of any size.

    Pads to a multiple of 32 for the forward pass, then crops the prediction
    back to the input size (PadIfNeeded pads centered).
    """
    if device is None:
        device = next(model.parameters()).device
    h, w = image.shape[:2]
    batch = seg_eval_transforms()(image=image)["image"].unsqueeze(0).to(device)
    model.eval()
    pred = model(batch).argmax(dim=1).squeeze(0).cpu().numpy()
    top = (pred.shape[0] - h) // 2
    left = (pred.shape[1] - w) // 2
    return pred[top : top + h, left : left + w]
