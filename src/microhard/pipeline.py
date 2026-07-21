"""End-to-end orchestration: image to family to features to property heads.

``predict_image`` is the routing entry point. When the family is unknown, the
segmenter does not cover the family, or no fitted property head exists, it
records the reason in ``abstentions`` and moves on rather than raising.

``fit_property_head`` trains any registered (scope, property) head. Property
values come from the adapters' canonical records, so heads read no
dataset-specific files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from . import heads
from .adapters import enabled_adapters
from .config import Config
from .features import FeatureVector, aggregate_by_group, feature_names, image_feature_vector
from .taxonomy import Taxonomy


@dataclass
class PredictionResult:
    image: Path
    family: str | None  # None = unknown family (router abstained or unavailable)
    family_probabilities: dict[str, float] = field(default_factory=dict)
    fractions: dict[str, float] = field(default_factory=dict)  # taxonomy id -> area frac
    features: FeatureVector | None = None
    properties: dict[str, float] = field(default_factory=dict)  # property -> estimate
    abstentions: dict[str, str] = field(default_factory=dict)  # stage/property -> reason


def predict_image(cfg: Config, image_path: Path, family: str | None = None) -> PredictionResult:
    """Route one micrograph through the full pipeline.

    ``family`` overrides the router (useful before a router is trained, and
    for tests); otherwise the conformal router decides, and may abstain.
    """
    from .router import load_router, route_image
    from .segment import load_segmenter, segment_image

    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    result = PredictionResult(image=Path(image_path), family=None)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    device = cfg.resolve_device()

    # --- stage 1: family -----------------------------------------------------
    if family is not None:
        taxonomy.node(family)  # reject an unknown family id before doing any work
        result.family = family
    else:
        try:
            model, families, abstainer = load_router(cfg)
        except FileNotFoundError as exc:
            result.abstentions["family"] = f"{exc} (or pass --family)"
            return result
        model.to(device)
        routed = route_image(model, families, abstainer, image, cfg.image_size, device)
        result.family_probabilities = routed.probabilities
        if routed.family is None:
            result.abstentions["family"] = (
                f"unknown family: conformal prediction set {routed.prediction_set or '()'} "
                "is not a single family"
            )
            return result
        result.family = routed.family

    # --- stage 2: segmentation -> features ----------------------------------
    try:
        segmenter, class_nodes = load_segmenter(cfg)
    except FileNotFoundError as exc:
        result.abstentions["features"] = str(exc)
        return result
    seg_family = taxonomy.family_of(class_nodes[0])
    if seg_family != result.family:
        result.abstentions["features"] = (
            f"segmenter is calibrated for family '{seg_family}', not '{result.family}'"
        )
        return result
    segmenter.to(device)
    mask = segment_image(segmenter, image, device)
    fv = image_feature_vector(mask, class_nodes, result.family)
    result.features = fv
    result.fractions = {node: fv.get(f"frac:{node}") for node in class_nodes}

    # --- stage 3: property heads --------------------------------------------
    registered = heads.heads_for_family(result.family)
    if not registered:
        result.abstentions["properties"] = f"no property heads registered for '{result.family}'"
        return result
    for scope, property_name in registered:
        head = heads.load_fitted(cfg, scope, property_name)
        if head is None:
            result.abstentions[property_name] = (
                f"no calibrated model for ({scope}, {property_name}): insufficient "
                "calibration data or fit not run"
            )
        else:
            result.properties[property_name] = head.predict(fv)
    return result


def fit_property_head(cfg: Config, scope: str, property_name: str) -> dict[str, Any] | None:
    """Fit + persist the registered head for (scope, property).

    Sample-level features come from features.csv (extract-features); property
    values come from adapter records. Returns metrics, or None when there is
    not enough labeled data (message printed, no error raised).
    """
    head_cls = heads.head_class(scope, property_name)
    if head_cls is None:
        raise KeyError(
            f"No head registered for ({scope}, {property_name}); known: {heads.registered()}"
        )
    if not cfg.features_csv.exists():
        raise FileNotFoundError(
            f"{cfg.features_csv} not found; run `microhard extract-features` first."
        )

    taxonomy = Taxonomy.load(cfg.taxonomy_path)
    frame = aggregate_by_group(pd.read_csv(cfg.features_csv))
    frame = _filter_scope(frame, scope)

    values_by_group = _property_by_group(cfg, taxonomy, property_name)
    frame = frame[frame["group_id"].isin(values_by_group)]
    if frame.empty:
        print(
            f"[microhard] insufficient calibration data: no {property_name} values "
            f"attached to any '{scope}' records."
        )
        return None
    y = frame["group_id"].map(values_by_group).to_numpy(dtype=float)
    X = frame[feature_names(frame)]

    head = head_cls()
    try:
        metrics = head.fit(X, y)
    except ValueError as exc:
        print(f"[microhard] skipping fit: {exc}")
        return None
    path = heads.fitted_path(cfg, scope, property_name)
    head.save(path)
    print(f"[microhard] saved {property_name} head -> {path}")
    return metrics


def _filter_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    parts = scope.split("/")
    frame = frame[frame["family"] == parts[0]]
    if len(parts) > 1:
        frame = frame[frame["adapter"] == parts[1]]
    return frame


def _property_by_group(cfg: Config, taxonomy: Taxonomy, property_name: str) -> dict[str, float]:
    """group_id -> property value, first non-null across adapter records."""
    out: dict[str, float] = {}
    for adapter in enabled_adapters(cfg, taxonomy):
        for record in adapter.records():
            value = record.properties.get(property_name)
            if value is not None and record.group_id not in out:
                out[record.group_id] = float(value)
    return out
