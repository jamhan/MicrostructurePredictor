"""Typer CLI for the microhard pipeline.

Heavy imports (torch, smp) live inside the commands so that
``microhard --help`` stays fast.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="SEM micrograph -> family -> microconstituent fractions -> properties.",
)

_CONFIG_OPTION = typer.Option(None, "--config", "-c", help="TOML file overriding Config defaults.")


def _cfg(config: Optional[Path]):
    from .config import Config

    return Config.load(config)


@app.command()
def download(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Run download.sh (if present in the working directory) and verify data files."""
    cfg = _cfg(config)
    script = Path("download.sh")
    if script.exists():
        subprocess.run(["bash", str(script)], check=False)
    else:
        typer.echo("download.sh not found in the current directory; verifying only.")

    checks = [
        (cfg.sqlite_path, "https://hdl.handle.net/11256/940 (microstructures.sqlite)"),
        (cfg.micrographs_dir, "https://hdl.handle.net/11256/940 (micrographs.zip)"),
        (cfg.segmentation_dir / "uhcs", "https://hdl.handle.net/11256/964"),
        (cfg.segmentation_dir / "particles", "https://hdl.handle.net/11256/964"),
        (cfg.hardness_csv, "transcribe HV values from the Hecht papers (starts empty)"),
    ]
    all_ok = True
    typer.echo("\nData check:")
    for path, hint in checks:
        ok = path.exists()
        all_ok &= ok
        typer.echo(f"  [{'ok' if ok else 'MISSING'}] {path}   <- {hint}")
    if not all_ok:
        typer.echo("\nSome data is missing; download manually from the URLs above.")
        raise typer.Exit(1)


@app.command()
def taxonomy(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Print the taxonomy tree (families, constituents, morphologies)."""
    from .taxonomy import Taxonomy

    tax = Taxonomy.load(_cfg(config).taxonomy_path)
    for node_id in tax.ids():
        node = tax.node(node_id)
        typer.echo(f"{'  ' * (node.level - 1)}{node.id}  ({node.name})")


@app.command("train-router")
def train_router_cmd(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Train the family router head + calibrate conformal abstention."""
    from .router import train_router

    train_router(_cfg(config))


@app.command("train-seg")
def train_seg(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Train the U-Net decoder head on records that carry pixel masks."""
    from .segment import train_segmentation

    train_segmentation(_cfg(config))


@app.command("train-clf")
def train_clf(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Train the constituent classifier head (outputs taxonomy node ids)."""
    from .classify import train_classifier

    train_classifier(_cfg(config))


@app.command("extract-features")
def extract_features_cmd(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Segment all on-disk records and write FeatureVectors to data/features.csv."""
    from .features import extract_features

    extract_features(_cfg(config))


@app.command("fit-hardness")
def fit_hardness_cmd(config: Optional[Path] = _CONFIG_OPTION) -> None:
    """Fit the ferrous/uhcs hardness head with leave-one-out CV."""
    from .heads.hardness import HARDNESS_HINT
    from .pipeline import fit_property_head

    if fit_property_head(_cfg(config), "ferrous/uhcs", "hardness_hv") is None:
        typer.echo(f"[microhard] {HARDNESS_HINT}")


@app.command()
def predict(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    family: Optional[str] = typer.Option(
        None, "--family", help="Skip the router and force this taxonomy family id."
    ),
    config: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Route one micrograph: family -> constituent fractions -> properties."""
    from .pipeline import predict_image

    result = predict_image(_cfg(config), image, family=family)

    typer.echo(f"\n{image.name}")
    if result.family is None:
        typer.echo(f"  family: UNKNOWN. {result.abstentions.get('family', 'abstained')}")
        if result.family_probabilities:
            for fam, p in sorted(result.family_probabilities.items(), key=lambda kv: -kv[1]):
                typer.echo(f"    p({fam}) = {p:.2f}")
        raise typer.Exit(2)
    typer.echo(f"  family: {result.family}")

    if result.fractions:
        typer.echo("\n  constituent                       area fraction")
        for node, fraction in result.fractions.items():
            typer.echo(f"  {node:<32s} {fraction:>12.1%}")
    elif "features" in result.abstentions:
        typer.echo(f"\n  features: unavailable — {result.abstentions['features']}")

    if result.properties:
        typer.echo("")
        for name, value in result.properties.items():
            typer.echo(f"  {name}: {value:.0f}")
        typer.echo("  (single-image estimate from a sample-level calibration; treat as rough)")
    for key, reason in result.abstentions.items():
        if key not in {"family", "features"}:
            typer.echo(f"\n  {key}: unavailable — {reason}")


if __name__ == "__main__":
    app()
