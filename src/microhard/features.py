"""FeatureVector: what a segmentation mask becomes for the property models.

Feature names are built from taxonomy node ids (``frac:ferrous/network``,
``n_regions:ferrous/spheroidite``), so features from different material
families share one namespace. Property heads consume these vectors and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

# scipy is a hard runtime dependency of scikit-learn, so it is always present
# even though it is not a direct dependency in pyproject.toml.
from scipy import ndimage

from .adapters import enabled_adapters
from .config import Config
from .taxonomy import Taxonomy

META_COLUMNS = ("record_id", "group_id", "family", "adapter")


@dataclass
class FeatureVector:
    """Named features for one image (or one aggregated sample)."""

    family: str  # taxonomy level-1 node id
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)

    def merged_with(self, extra: dict[str, float]) -> "FeatureVector":
        return FeatureVector(self.family, {**self.values, **extra})


def constituent_fractions(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Area fraction per mask class index; sums to 1."""
    counts = np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
    return counts / mask.size


def region_stats(mask: np.ndarray, class_idx: int) -> dict[str, float]:
    """Connected-component stats for one class (areas in px²; µm² conversion
    via CanonicalRecord.scale_um_per_px is a natural next step)."""
    labeled, n_regions = ndimage.label(mask == class_idx)
    if n_regions == 0:
        return {"n_regions": 0.0, "mean_region_area": 0.0, "max_region_area": 0.0}
    areas = np.bincount(labeled.reshape(-1))[1:]  # skip background label 0
    return {
        "n_regions": float(n_regions),
        "mean_region_area": float(areas.mean()),
        "max_region_area": float(areas.max()),
    }


def image_feature_vector(
    mask: np.ndarray, class_nodes: tuple[str, ...], family: str
) -> FeatureVector:
    """FeatureVector for one predicted mask, keyed by taxonomy node id."""
    fractions = constituent_fractions(mask, len(class_nodes))
    values = {f"frac:{node}": float(fractions[i]) for i, node in enumerate(class_nodes)}
    for i, node in enumerate(class_nodes):
        for stat, value in region_stats(mask, i).items():
            values[f"{stat}:{node}"] = value
    return FeatureVector(family=family, values=values)


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLUMNS]


def aggregate_by_group(features: pd.DataFrame) -> pd.DataFrame:
    """Sample-level features: mean of per-image features per group_id, keeping
    family/adapter metadata (weak-label aggregation)."""
    meta = features.groupby("group_id")[["family", "adapter"]].first()
    numeric = features.drop(columns=["record_id", "family", "adapter"], errors="ignore")
    aggregated = numeric.groupby("group_id").mean(numeric_only=True)
    return meta.join(aggregated).reset_index()


def extract_features(cfg: Config) -> pd.DataFrame:
    """Segment every on-disk record of the segmenter's family; write CSV."""
    from .segment import load_segmenter, segment_image  # heavy import kept local

    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    model, class_nodes = load_segmenter(cfg)
    seg_family = taxonomy.family_of(class_nodes[0])
    device = cfg.resolve_device()
    model.to(device)

    rows: list[dict] = []
    skipped_family = skipped_missing = 0
    for adapter in enabled_adapters(cfg, taxonomy):
        if adapter.family != seg_family:
            skipped_family += len(adapter.records())
            continue
        for record in adapter.validated_records():
            if not record.image_path.exists():
                skipped_missing += 1
                continue
            image = np.asarray(Image.open(record.image_path).convert("RGB"))
            mask = segment_image(model, image, device)
            fv = image_feature_vector(mask, class_nodes, adapter.family)
            rows.append(
                {
                    "record_id": record.record_id,
                    "group_id": record.group_id,
                    "family": fv.family,
                    "adapter": adapter.name,
                    **fv.values,
                }
            )
    if skipped_family:
        print(
            f"[microhard] skipped {skipped_family} records from other families "
            f"(segmenter is calibrated for '{seg_family}')"
        )
    if skipped_missing:
        print(f"[microhard] skipped {skipped_missing} records with no image on disk")

    features = pd.DataFrame(rows)
    cfg.features_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cfg.features_csv, index=False)
    print(f"[microhard] wrote {len(features)} rows -> {cfg.features_csv}")
    return features
