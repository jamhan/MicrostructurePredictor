from pathlib import Path

from typer.testing import CliRunner

from microhard.cli import app
from tests.conftest import write_image

runner = CliRunner()


def _config_toml(cfg, tmp_path: Path) -> Path:
    toml = tmp_path / "run.toml"
    toml.write_text(
        f'data_dir = "{cfg.data_dir}"\n'
        f'checkpoint_dir = "{cfg.checkpoint_dir}"\n'
        'encoder_weights = "none"\ndevice = "cpu"\nimage_size = 64\n'
    )
    return toml


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "download",
        "taxonomy",
        "train-router",
        "train-seg",
        "train-clf",
        "extract-features",
        "fit-hardness",
        "audit-public-links",
        "validate-campaign",
        "plan-measurements",
        "predict",
    ):
        assert command in result.output


def test_taxonomy_command_prints_tree() -> None:
    result = runner.invoke(app, ["taxonomy"])
    assert result.exit_code == 0
    assert "ferrous" in result.output
    assert "ferrous/pearlite/lamellar" in result.output


def test_predict_without_router_reports_family_abstention(cfg, tmp_path: Path) -> None:
    image = tmp_path / "micrograph.png"
    write_image(image)
    result = runner.invoke(app, ["predict", str(image), "-c", str(_config_toml(cfg, tmp_path))])
    assert result.exit_code == 2
    assert "UNKNOWN" in result.output
    assert "train-router" in result.output


def test_predict_with_family_but_no_segmenter(cfg, tmp_path: Path) -> None:
    image = tmp_path / "micrograph.png"
    write_image(image)
    result = runner.invoke(
        app,
        ["predict", str(image), "--family", "ferrous", "-c", str(_config_toml(cfg, tmp_path))],
    )
    assert result.exit_code == 0
    assert "family: ferrous" in result.output
    assert "train-seg" in result.output  # tells the user what to run next


def test_download_verifies_missing_data(cfg, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no download.sh here: verify-only path
    result = runner.invoke(app, ["download", "-c", str(_config_toml(cfg, tmp_path))])
    assert result.exit_code == 1
    assert "MISSING" in result.output
    assert "11256/940" in result.output
