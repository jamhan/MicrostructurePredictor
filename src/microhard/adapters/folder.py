"""Generic image-folder adapter + the second-adapter stub (non-steel family).

Layout expected under the adapter's root directory::

    root/
      labels.csv     # columns: path, taxonomy_labels, [group_id], [scale_um_per_px]
      <images referenced by the path column>

``taxonomy_labels`` holds one or more node ids separated by ``|``.

``MicroNetAlAdapter`` is the stub required by the architecture: an aluminum
family dataset with classification labels only (no masks, no properties), so
the pipeline provably runs end-to-end on a non-steel family and abstains on
properties. Point its root at a real MicroNet aluminum subset (or EM3M) once
downloaded — the CSV contract is all that matters.
"""

from __future__ import annotations

import pandas as pd

from ..records import CanonicalRecord
from . import register_adapter
from .base import BaseAdapter


class ImageFolderAdapter(BaseAdapter):
    """Subclass and set ``name``, ``family``, ``modality``; records come from
    ``<data_dir>/<name>/labels.csv``."""

    modality = "SEM"

    @property
    def root(self):
        return self.cfg.data_dir / self.name

    def records(self) -> list[CanonicalRecord]:
        csv_path = self.root / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. The '{self.name}' adapter expects "
                "labels.csv with columns: path, taxonomy_labels[, group_id, scale_um_per_px]"
            )
        df = pd.read_csv(csv_path)
        for column in ("path", "taxonomy_labels"):
            if column not in df.columns:
                raise ValueError(f"{csv_path} is missing required column {column!r}")

        out: list[CanonicalRecord] = []
        for i, row in df.iterrows():
            labels = tuple(sorted(str(row["taxonomy_labels"]).split("|")))
            scale = row.get("scale_um_per_px")
            group = row.get("group_id")
            out.append(
                CanonicalRecord(
                    record_id=f"{self.name}-{i}",
                    image_path=self.root / str(row["path"]),
                    scale_um_per_px=float(scale) if pd.notna(scale) else None,
                    modality=self.modality,
                    group_id=f"{self.name}-{group}" if pd.notna(group) else "",
                    taxonomy_labels=labels,
                )
            )
        return out


@register_adapter
class MicroNetAlAdapter(ImageFolderAdapter):
    """Aluminum-family stub (classification labels only)."""

    name = "micronet_al"
    family = "aluminum"
