from pathlib import Path

import pytest

from microhard.config import Config


def test_defaults() -> None:
    cfg = Config()
    assert cfg.encoder == "resnet50"
    assert cfg.image_size == 484
    assert cfg.adapters == ["uhcs", "literature_steel"]
    assert cfg.taxonomy_path is None  # bundled seed taxonomy
    assert cfg.sqlite_path == Path("data/microstructures.sqlite")
    assert cfg.seg_checkpoint == Path("checkpoints/segmenter.pt")
    assert cfg.backbone_checkpoint == Path("checkpoints/backbone.pt")
    assert cfg.router_checkpoint == Path("checkpoints/router.pt")
    assert cfg.heads_dir == Path("checkpoints/heads")
    assert cfg.literature_manifest_csv == Path("data/literature_steel/manifest.csv")


def test_toml_override(tmp_path: Path) -> None:
    toml = tmp_path / "run.toml"
    toml.write_text(
        'data_dir = "elsewhere"\nbatch_size = 4\nencoder_weights = "imagenet"\n'
        'adapters = ["uhcs", "micronet_al"]\n'
    )
    cfg = Config.load(toml)
    assert cfg.data_dir == Path("elsewhere")
    assert cfg.sqlite_path == Path("elsewhere/microstructures.sqlite")
    assert cfg.batch_size == 4
    assert cfg.encoder_weights == "imagenet"
    assert cfg.adapters == ["uhcs", "micronet_al"]
    assert cfg.lr == Config().lr  # untouched defaults survive


def test_unknown_key_rejected(tmp_path: Path) -> None:
    toml = tmp_path / "run.toml"
    toml.write_text("batchsize = 4\n")
    with pytest.raises(ValueError, match="batchsize"):
        Config.load(toml)


def test_bad_encoder_weights_rejected() -> None:
    with pytest.raises(ValueError, match="encoder_weights"):
        Config(encoder_weights="imagnet")
