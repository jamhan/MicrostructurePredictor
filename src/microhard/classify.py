"""Constituent classifier head: frozen backbone + linear head.

Classes are the distinct *sets* of taxonomy node ids observed in the enabled
adapters' records (UHCS combined labels like pearlite+spheroidite are one
class whose output is two node ids). All outputs are taxonomy node ids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapters import enabled_adapters
from .backbone import ClassifierHead, build_encoder, trainable_parameters
from .config import Config
from .records import CanonicalRecord, RecordClassificationDataset, split_records_by_group
from .taxonomy import Taxonomy
from .transforms import clf_eval_transforms, clf_train_transforms


def _label(record: CanonicalRecord) -> tuple[str, ...] | None:
    return tuple(sorted(record.taxonomy_labels)) if record.taxonomy_labels else None


def labeled_records(cfg: Config, taxonomy: Taxonomy) -> list[CanonicalRecord]:
    return [
        r
        for adapter in enabled_adapters(cfg, taxonomy)
        for r in adapter.validated_records()
        if r.taxonomy_labels and r.image_path.exists()
    ]


def build_classifier(cfg: Config, num_classes: int) -> ClassifierHead:
    return ClassifierHead(build_encoder(cfg), num_classes)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((preds == labels).sum())
        total += len(labels)
    return correct / total if total else 0.0


def train_classifier(cfg: Config) -> Path:
    """Train the linear head on the sample-aware split; save best accuracy."""
    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    records = labeled_records(cfg, taxonomy)
    if not records:
        raise FileNotFoundError(
            "No labeled records with images on disk; run `bash download.sh` first."
        )
    classes = sorted({label for r in records if (label := _label(r)) is not None})
    splits = split_records_by_group(records, cfg.val_frac, cfg.test_frac, cfg.seed)

    train_ds = RecordClassificationDataset(
        splits["train"], classes, _label, clf_train_transforms(cfg.image_size)
    )
    val_ds = RecordClassificationDataset(
        splits["val"], classes, _label, clf_eval_transforms(cfg.image_size)
    )
    print(f"[microhard] {len(train_ds)} train / {len(val_ds)} val labeled records, "
          f"{len(classes)} classes")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    device = cfg.resolve_device()
    model = build_classifier(cfg, len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        acc = evaluate_classifier(model, val_loader, device)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "head_state": model.head.state_dict(),
                    "classes": [list(c) for c in classes],
                    "encoder": cfg.encoder,
                },
                cfg.clf_checkpoint,
            )
            marker = "  (saved)"
        print(
            f"epoch {epoch + 1:>3}/{cfg.epochs}  "
            f"loss {total_loss / max(len(train_ds), 1):.3f}  val acc {acc:.3f}{marker}"
        )
    print(f"[microhard] best val accuracy {best_acc:.3f} -> {cfg.clf_checkpoint}")
    return cfg.clf_checkpoint


def load_classifier(cfg: Config) -> tuple[ClassifierHead, list[tuple[str, ...]]]:
    """(model, classes) — each class is a tuple of taxonomy node ids."""
    if not cfg.clf_checkpoint.exists():
        raise FileNotFoundError(
            f"No classifier checkpoint at {cfg.clf_checkpoint}. Run `microhard train-clf` first."
        )
    ckpt = torch.load(cfg.clf_checkpoint, map_location="cpu", weights_only=True)
    classes = [tuple(c) for c in ckpt["classes"]]
    model = build_classifier(cfg, len(classes))
    model.head.load_state_dict(ckpt["head_state"])
    model.eval()
    return model, classes


@torch.no_grad()
def classify_image(
    model: ClassifierHead,
    classes: list[tuple[str, ...]],
    image: np.ndarray,
    image_size: int,
    device: torch.device | None = None,
) -> tuple[str, ...]:
    """Taxonomy node ids for one RGB uint8 image."""
    if device is None:
        device = next(model.parameters()).device
    batch = clf_eval_transforms(image_size)(image=image)["image"].unsqueeze(0).to(device)
    model.eval()
    return classes[int(model(batch).argmax(dim=1))]
