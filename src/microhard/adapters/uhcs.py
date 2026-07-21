"""UHCSDB adapter — implementation #1 of the adapter contract.

Reads microstructures.sqlite (schema: sample + micrograph tables, no hardness
columns), joins the hand-transcribed hardness CSV, computes physical scale
from the micron-bar fields, maps ``primary_microconstituent`` strings to
taxonomy node ids, and attaches pixel masks from the 11256/964 benchmark
where filename stems match.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

from ..normalize import normalize_grade, normalize_structured_condition
from ..records import MEASURED, CanonicalRecord
from . import register_adapter
from .base import BaseAdapter

HARDNESS_COLUMNS = ["sample_label", "hardness_hv", "source_note"]

# UHCS primary_microconstituent -> taxonomy node ids. Combined labels map to
# multiple nodes; classifier classes are the distinct label *sets*.
PRIMARY_TO_NODES: dict[str, tuple[str, ...]] = {
    "martensite": ("ferrous/martensite",),
    "network": ("ferrous/network",),
    "pearlite": ("ferrous/pearlite",),
    "pearlite+spheroidite": ("ferrous/pearlite", "ferrous/spheroidite"),
    "pearlite+widmanstatten": ("ferrous/pearlite", "ferrous/widmanstatten"),
    "spheroidite": ("ferrous/spheroidite",),
    "spheroidite+widmanstatten": ("ferrous/spheroidite", "ferrous/widmanstatten"),
}

# Integer class i in a benchmark mask means SEG_CLASS_NODES[i].
SEG_CLASS_NODES: tuple[str, ...] = (
    "ferrous/matrix",
    "ferrous/network",
    "ferrous/spheroidite",
    "ferrous/widmanstatten",
)

_IMAGE_DIR_NAMES = ("images", "micrographs", "src", "img")
_MASK_DIR_NAMES = ("labels", "annotations", "masks", "ground_truth", "gt")
_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_THERMAL_STEP = re.compile(
    r"(?<![-\w])(\d+(?:\.\d+)?)\s*C\s+(\d+(?:\.\d+)?)\s*([MH])\b",
    re.IGNORECASE,
)
_TEMPERATURE_TOKEN = re.compile(r"(?<![-\w])\d+(?:\.\d+)?\s*C\b", re.IGNORECASE)


def load_metadata(sqlite_path: Path) -> pd.DataFrame:
    """One row per micrograph, joined with its parent sample; the sample
    table's ``label`` column is renamed ``sample_label``."""
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"{sqlite_path} not found. Run `bash download.sh`, or download manually "
            "from https://hdl.handle.net/11256/940"
        )
    with sqlite3.connect(sqlite_path) as con:
        micrographs = pd.read_sql_query("SELECT * FROM micrograph", con)
        samples = pd.read_sql_query("SELECT * FROM sample", con)
    samples = samples.rename(columns={"label": "sample_label"})
    return micrographs.merge(samples, how="left", left_on="sample_key", right_on="sample_id")


def load_hardness_labels(csv_path: Path) -> pd.DataFrame:
    """Hand-transcribed hardness values; empty frame (with columns) when none
    exist yet — everything downstream degrades gracefully on empty."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=HARDNESS_COLUMNS)
    df = pd.read_csv(csv_path)
    missing = set(HARDNESS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; expected {HARDNESS_COLUMNS}"
        )
    df["hardness_hv"] = pd.to_numeric(df["hardness_hv"], errors="coerce")
    return df.dropna(subset=["hardness_hv"]).reset_index(drop=True)


def resolve_image_path(images_dir: Path, raw: str) -> Path:
    """Stored relative path first, then basename (download.sh flattens)."""
    images_dir = Path(images_dir)
    candidate = images_dir / raw
    return candidate if candidate.exists() else images_dir / Path(raw).name


def find_benchmark_masks(root: Path) -> dict[str, Path]:
    """stem -> mask path for the extracted 11256/964 benchmark, {} if absent."""
    root = Path(root)
    if not root.exists():
        return {}
    mask_dir = next((root / n for n in _MASK_DIR_NAMES if (root / n).is_dir()), None)
    if mask_dir is None:
        raise FileNotFoundError(
            f"{root} exists but has no mask subfolder; expected one of {_MASK_DIR_NAMES}. "
            "Check the extracted layout or extend the name lists in adapters/uhcs.py."
        )
    return {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}


@register_adapter
class UHCSAdapter(BaseAdapter):
    name = "uhcs"
    family = "ferrous"

    def records(self) -> list[CanonicalRecord]:
        cfg = self.cfg
        meta = load_metadata(cfg.sqlite_path)
        group_ids = self._group_ids(meta)
        hardness = load_hardness_labels(cfg.hardness_csv)
        hv_by_sample = dict(zip(hardness["sample_label"], hardness["hardness_hv"]))
        masks = find_benchmark_masks(cfg.segmentation_dir / "uhcs")

        out: list[CanonicalRecord] = []
        for _, row in meta.iterrows():
            image_path = resolve_image_path(cfg.micrographs_dir, str(row["path"]))
            scale = self._scale(row)
            labels = PRIMARY_TO_NODES.get(row.get("primary_microconstituent"))
            sample_label = row.get("sample_label")
            properties, property_sources = {}, {}
            if pd.notna(sample_label) and sample_label in hv_by_sample:
                properties["hardness_hv"] = float(hv_by_sample[sample_label])
                property_sources["hardness_hv"] = MEASURED
            grade, condition = self._join_key(row)
            mask_path = masks.get(image_path.stem)
            out.append(
                CanonicalRecord(
                    record_id=f"uhcs-{row['micrograph_id']}",
                    image_path=image_path,
                    scale_um_per_px=scale,
                    modality="SEM",
                    group_id=self._group_id(row, group_ids),
                    taxonomy_labels=labels,
                    mask_path=mask_path,
                    mask_class_nodes=SEG_CLASS_NODES if mask_path is not None else None,
                    properties=properties,
                    property_sources=property_sources,
                    alloy_grade=grade,
                    condition=condition,
                )
            )
        return out

    @staticmethod
    def _group_ids(meta: pd.DataFrame) -> dict[int, str]:
        """Sample id -> conservative physical split group.

        UHCSDB contains two sample rows with the same treatment label. They
        may be separate specimens, but treating them as independent would put
        near-identical material on both sides of a split. Duplicate labels are
        therefore grouped together until their independence is verified.
        """
        sample_rows = meta[["sample_key", "sample_label"]].dropna(subset=["sample_key"])
        sample_rows = sample_rows.drop_duplicates(subset=["sample_key"])
        ids_by_label: dict[str, list[int]] = {}
        for _, sample in sample_rows.iterrows():
            label = str(sample["sample_label"]) if pd.notna(sample["sample_label"]) else ""
            ids_by_label.setdefault(label, []).append(int(sample["sample_key"]))
        out: dict[int, str] = {}
        for label, sample_ids in ids_by_label.items():
            ids = sorted(sample_ids)
            group = (
                f"uhcs-samples-{'-'.join(map(str, ids))}"
                if label and len(ids) > 1
                else f"uhcs-sample-{ids[0]}"
            )
            out.update({sample_id: group for sample_id in ids})
        return out

    @staticmethod
    def _group_id(row: pd.Series, group_ids: dict[int, str]) -> str:
        sample_key = row.get("sample_key")
        if pd.notna(sample_key):
            return group_ids[int(sample_key)]
        return f"uhcs-orphan-{row['micrograph_id']}"

    @staticmethod
    def _join_key(row: pd.Series) -> tuple[str | None, str | None]:
        """Exact (alloy_grade, condition) ids from structured sample metadata.

        Only simple water-quench rows whose label agrees with the structured
        temperature and hold columns receive a condition. Multi-step routes,
        special cooling codes, and coarse Q/AR/FC routes stay unjoined until
        their full processing details are verified.
        """
        label = row.get("sample_label")
        raw_label = str(label) if pd.notna(label) else None
        return normalize_grade(raw_label), UHCSAdapter._exact_condition(row, raw_label)

    @staticmethod
    def _exact_condition(row: pd.Series, label: str | None) -> str | None:
        cool = row.get("cool_method")
        if not pd.notna(cool) or str(cool).strip().upper() != "WQ" or label is None:
            return None
        steps = _THERMAL_STEP.findall(label)
        if len(steps) != 1 or len(_TEMPERATURE_TOKEN.findall(label)) != 1:
            return None
        label_temperature, label_hold, label_unit = steps[0]
        temperature = row.get("anneal_temperature")
        hold = row.get("anneal_time")
        hold_unit = row.get("anneal_time_unit")
        if not all(pd.notna(value) for value in (temperature, hold, hold_unit)):
            return None
        row_minutes = float(hold) * (60.0 if str(hold_unit).strip().upper() == "H" else 1.0)
        label_minutes = float(label_hold) * (60.0 if label_unit.upper() == "H" else 1.0)
        if not math.isclose(float(temperature), float(label_temperature)) or not math.isclose(
            row_minutes, label_minutes
        ):
            return None
        return normalize_structured_condition(
            str(cool),
            temperature,
            row.get("anneal_temp_unit"),
            hold,
            hold_unit,
        )

    @staticmethod
    def _scale(row: pd.Series) -> float | None:
        """µm per pixel from the micron-bar annotation, when usable."""
        bar, bar_px = row.get("micron_bar"), row.get("micron_bar_px")
        if pd.notna(bar) and pd.notna(bar_px) and bar_px:
            return float(bar) / float(bar_px)
        return None
