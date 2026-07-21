"""Synthetic fixtures — no real UHCS/MicroNet data required anywhere."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from microhard.adapters.uhcs import PRIMARY_TO_NODES, SEG_CLASS_NODES
from microhard.config import Config

N_SAMPLES = 5
MICROGRAPHS_PER_SAMPLE = 3
IMAGE_SIZE = 64

UHCS_PRIMARY_LABELS = sorted(PRIMARY_TO_NODES)


def write_image(path: Path, size: int = IMAGE_SIZE, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Config pointing at an empty per-test data directory, tuned for CPU tests."""
    config = Config(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        encoder_weights="none",  # never download pretrained weights in tests
        image_size=IMAGE_SIZE,
        batch_size=2,
        epochs=1,
        num_workers=0,
        device="cpu",
    )
    config.data_dir.mkdir(parents=True)
    return config


@pytest.fixture()
def synthetic_db(cfg: Config) -> Config:
    """Sqlite matching the UHCS schema + images on disk + empty hardness CSV."""
    cfg.micrographs_dir.mkdir()
    con = sqlite3.connect(cfg.sqlite_path)
    con.execute(
        "CREATE TABLE sample (sample_id INTEGER PRIMARY KEY, label TEXT,"
        " anneal_time REAL, anneal_time_unit TEXT, anneal_temperature REAL,"
        " anneal_temp_unit TEXT, cool_method TEXT)"
    )
    con.execute(
        "CREATE TABLE micrograph (micrograph_id INTEGER PRIMARY KEY, path TEXT,"
        " micron_bar REAL, micron_bar_units TEXT, micron_bar_px INTEGER,"
        " magnification REAL, detector TEXT, sample_key INTEGER,"
        " primary_microconstituent TEXT)"
    )
    micrograph_id = 0
    for sample_id in range(1, N_SAMPLES + 1):
        con.execute(
            "INSERT INTO sample VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sample_id, f"S{sample_id}", 60.0, "M", 800.0, "C", "AR"),
        )
        for _ in range(MICROGRAPHS_PER_SAMPLE):
            micrograph_id += 1
            filename = f"micrograph{micrograph_id}.png"
            write_image(cfg.micrographs_dir / filename, seed=micrograph_id)
            constituent = UHCS_PRIMARY_LABELS[micrograph_id % len(UHCS_PRIMARY_LABELS)]
            con.execute(
                "INSERT INTO micrograph VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (micrograph_id, filename, 10.0, "um", 100, 500.0, "SE", sample_id, constituent),
            )
    con.commit()
    con.close()
    cfg.hardness_csv.write_text("sample_label,hardness_hv,source_note\n")
    return cfg


@pytest.fixture()
def seg_benchmark(synthetic_db: Config) -> Config:
    """Benchmark masks under segmentation/uhcs whose stems match micrographs
    1..4, so UHCSAdapter attaches them to real records."""
    cfg = synthetic_db
    root = cfg.segmentation_dir / "uhcs"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 5):
        write_image(root / "images" / f"micrograph{i}.png", seed=i)
        mask = rng.integers(0, len(SEG_CLASS_NODES), (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        Image.fromarray(mask).save(root / "labels" / f"micrograph{i}.png")
    return cfg


@pytest.fixture()
def aluminum_stub(cfg: Config) -> Config:
    """Second-adapter stub data: aluminum family, classification labels only."""
    root = cfg.data_dir / "micronet_al"
    root.mkdir(parents=True)
    lines = ["path,taxonomy_labels,group_id"]
    for i in range(6):
        filename = f"al{i}.png"
        write_image(root / filename, seed=100 + i)
        label = "aluminum/matrix" if i % 2 else "aluminum/matrix|aluminum/precipitate"
        lines.append(f"{filename},{label},g{i // 2}")
    (root / "labels.csv").write_text("\n".join(lines) + "\n")
    return cfg
