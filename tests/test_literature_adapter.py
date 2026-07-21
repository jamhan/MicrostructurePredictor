from pathlib import Path

import pandas as pd
import pytest

from microhard.adapters.literature import (
    LiteratureSteelAdapter,
    load_literature_manifest,
)
from microhard.config import Config
from microhard.records import DISTANT
from microhard.taxonomy import Taxonomy

REPO_DATA = Path(__file__).parents[1] / "data"


def test_tracked_literature_manifest_is_complete() -> None:
    cfg = Config(data_dir=REPO_DATA)
    records = LiteratureSteelAdapter(cfg, Taxonomy.load(None)).validated_records()

    assert len(records) == 19
    assert len({record.group_id for record in records}) == 10
    assert all(record.image_path.is_file() for record in records)
    assert all(record.taxonomy_labels is None for record in records)
    assert all(record.property_sources == {"hardness_hv": DISTANT} for record in records)
    assert {record.properties["hardness_hv"] for record in records} == {
        183.77,
        201.25,
        210.17,
        230.88,
        242.26,
        246.14,
        253.70,
        262.82,
        270.32,
        532.1,
    }


def test_two_magnifications_share_one_split_group() -> None:
    cfg = Config(data_dir=REPO_DATA)
    records = LiteratureSteelAdapter(cfg, Taxonomy.load(None)).validated_records()
    upper_dq = [record for record in records if record.record_id in {"guan2026-fig3-a", "guan2026-fig3-b"}]

    assert len(upper_dq) == 2
    assert len({record.group_id for record in upper_dq}) == 1
    assert {record.properties["hardness_hv"] for record in upper_dq} == {201.25}


def test_missing_manifest_is_an_empty_optional_adapter(cfg) -> None:
    assert LiteratureSteelAdapter(cfg, Taxonomy.load(None)).records() == []


def test_unreported_hardness_count_and_scatter_kind_are_preserved() -> None:
    frame = load_literature_manifest(REPO_DATA / "literature_steel" / "manifest.csv")
    ren = frame.loc[frame["record_id"] == "ren2023-fig2-b"].iloc[0]

    assert ren["property_value"] == "532.1"
    assert ren["property_scatter"] == "7.2"
    assert ren["scatter_kind"] == "reported_plus_minus_unspecified"
    assert ren["n_measurements"] == ""
    assert ren["hardness_load_kgf"] == "0.5"
    assert ren["hardness_dwell_s"] == "10"


def test_manifest_rejects_changed_image_hash(tmp_path: Path) -> None:
    frame = pd.read_csv(REPO_DATA / "literature_steel" / "manifest.csv", dtype=str)
    row = frame.iloc[[0]].copy()
    source = REPO_DATA / "literature_steel" / row.iloc[0]["image_path"]
    root = tmp_path / "literature_steel"
    root.mkdir()
    target = root / source.name
    target.write_bytes(source.read_bytes())
    row.loc[:, "image_path"] = source.name
    row.loc[:, "image_sha256"] = "0" * 64
    manifest = root / "manifest.csv"
    row.to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_literature_manifest(manifest)


def test_manifest_rejects_disagreement_within_specimen(tmp_path: Path) -> None:
    frame = pd.read_csv(REPO_DATA / "literature_steel" / "manifest.csv", dtype=str)
    rows = frame.iloc[:2].copy()
    root = tmp_path / "literature_steel"
    root.mkdir()
    for index, row in rows.iterrows():
        source = REPO_DATA / "literature_steel" / row["image_path"]
        target = root / source.name
        target.write_bytes(source.read_bytes())
        rows.loc[index, "image_path"] = source.name
    rows.loc[rows.index[1], "property_value"] = "999"
    manifest = root / "manifest.csv"
    rows.to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="disagree on their property match"):
        load_literature_manifest(manifest)
