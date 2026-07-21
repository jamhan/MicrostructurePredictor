"""Family router: taxonomy level-1 classifier with conformal abstention.

A linear head on the frozen backbone predicts the material family; a
split-conformal wrapper turns softmax scores into prediction *sets* and the
router abstains ("unknown family") unless the set is a single family.

Initially this is steel-vs-other: UHCS supplies the ferrous class and the
second adapter (micronet_al stub) supplies non-steel records. Swapping the
stub for real MicroNet non-steel classes needs no code changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
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

UNKNOWN_FAMILY = "unknown"


@dataclass
class RouterOutput:
    family: str | None  # None = abstained ("unknown family")
    probabilities: dict[str, float]
    prediction_set: tuple[str, ...]  # conformal set; family is set iff singleton


class ConformalAbstainer:
    """Split-conformal prediction sets for classification.

    Calibrate on held-out (probabilities, true index) pairs with
    nonconformity score ``1 - p_true``; at prediction time the set is every
    class whose score is within the calibrated quantile. Singleton set ->
    confident answer; empty or multi-class set -> abstain.
    """

    def __init__(self, alpha: float = 0.1, qhat: float | None = None) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.qhat = qhat

    def calibrate(self, probabilities: np.ndarray, true_indices: np.ndarray) -> float:
        n = len(true_indices)
        if n == 0:
            raise ValueError("cannot calibrate on an empty calibration set")
        scores = 1.0 - probabilities[np.arange(n), true_indices]
        # finite-sample-corrected quantile level, clipped for tiny n
        level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self.qhat

    def prediction_set(self, probabilities: np.ndarray) -> np.ndarray:
        if self.qhat is None:
            raise RuntimeError("abstainer is not calibrated; call calibrate() first")
        return np.flatnonzero(1.0 - probabilities <= self.qhat)


def _family_records(cfg: Config, taxonomy: Taxonomy) -> list[tuple[CanonicalRecord, str]]:
    out = []
    for adapter in enabled_adapters(cfg, taxonomy):
        for record in adapter.validated_records():
            if record.image_path.exists():
                out.append((record, adapter.family))
    return out


def train_router(cfg: Config) -> Path:
    """Train the family head, calibrate abstention on held-out groups, save."""
    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    pairs = _family_records(cfg, taxonomy)
    if not pairs:
        raise FileNotFoundError("No records with images on disk for any enabled adapter.")
    family_by_record = {r.record_id: fam for r, fam in pairs}
    records = [r for r, _ in pairs]
    families = sorted({fam for _, fam in pairs})
    if len(families) < 2:
        print(
            f"[microhard] warning: only one family ({families[0]}) in the enabled "
            "adapters, so the router will trivially predict it; add a second adapter "
            "for a meaningful steel-vs-other router."
        )

    def label(record: CanonicalRecord) -> str:
        return family_by_record[record.record_id]

    # calibration groups are held out from head training (split conformal)
    splits = split_records_by_group(records, cfg.router_calib_frac, 0.0, cfg.seed)
    train_records, calib_records = splits["train"], splits["val"]
    if not calib_records:
        raise ValueError("router_calib_frac left no calibration groups; increase it")
    print(
        f"[microhard] router: {len(train_records)} train / {len(calib_records)} "
        f"calibration records over families {families}"
    )

    train_loader = DataLoader(
        RecordClassificationDataset(
            train_records, families, label, clf_train_transforms(cfg.image_size)
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    calib_loader = DataLoader(
        RecordClassificationDataset(
            calib_records, families, label, clf_eval_transforms(cfg.image_size)
        ),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    device = cfg.resolve_device()
    model = ClassifierHead(build_encoder(cfg), len(families)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=cfg.lr)

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
        print(
            f"epoch {epoch + 1:>3}/{cfg.epochs}  "
            f"loss {total_loss / max(len(train_loader.dataset), 1):.3f}"
        )

    probabilities, true_indices = _collect_probabilities(model, calib_loader, device)
    abstainer = ConformalAbstainer(cfg.router_alpha)
    qhat = abstainer.calibrate(probabilities, true_indices)
    accuracy = float((probabilities.argmax(axis=1) == true_indices).mean())
    print(f"[microhard] router calibration accuracy {accuracy:.3f}, qhat {qhat:.3f}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state": model.head.state_dict(),
            "families": families,
            "encoder": cfg.encoder,
            "alpha": cfg.router_alpha,
            "qhat": qhat,
        },
        cfg.router_checkpoint,
    )
    print(f"[microhard] router saved -> {cfg.router_checkpoint}")
    return cfg.router_checkpoint


@torch.no_grad()
def _collect_probabilities(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, trues = [], []
    for images, labels in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
        trues.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(trues)


def load_router(cfg: Config) -> tuple[ClassifierHead, list[str], ConformalAbstainer]:
    if not cfg.router_checkpoint.exists():
        raise FileNotFoundError(
            f"No router checkpoint at {cfg.router_checkpoint}. Run `microhard train-router` first."
        )
    ckpt = torch.load(cfg.router_checkpoint, map_location="cpu", weights_only=True)
    families = list(ckpt["families"])
    model = ClassifierHead(build_encoder(cfg), len(families))
    model.head.load_state_dict(ckpt["head_state"])
    model.eval()
    return model, families, ConformalAbstainer(ckpt["alpha"], qhat=ckpt["qhat"])


@torch.no_grad()
def route_image(
    model: ClassifierHead,
    families: list[str],
    abstainer: ConformalAbstainer,
    image: np.ndarray,
    image_size: int,
    device: torch.device | None = None,
) -> RouterOutput:
    """Family for one RGB uint8 image, or abstention when not clearly one family."""
    if device is None:
        device = next(model.parameters()).device
    batch = clf_eval_transforms(image_size)(image=image)["image"].unsqueeze(0).to(device)
    model.eval()
    probabilities = torch.softmax(model(batch), dim=1).squeeze(0).cpu().numpy()
    indices = abstainer.prediction_set(probabilities)
    prediction_set = tuple(families[i] for i in indices)
    return RouterOutput(
        family=prediction_set[0] if len(prediction_set) == 1 else None,
        probabilities={f: float(p) for f, p in zip(families, probabilities)},
        prediction_set=prediction_set,
    )
