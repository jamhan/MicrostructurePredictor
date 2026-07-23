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
    """Print the taxonomy tree, one section per axis."""
    from .taxonomy import AXIS_MAX_LEVEL, Taxonomy

    tax = Taxonomy.load(_cfg(config).taxonomy_path)
    for axis in sorted(AXIS_MAX_LEVEL):
        nodes = tax.in_axis(axis)
        if not nodes:
            continue
        typer.echo(f"\n[{axis}]")
        for node in nodes:
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


@app.command("audit-public-links")
def audit_public_links_cmd(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional CSV path for every retained image/property candidate.",
    ),
    config: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Audit fuzzy image/property links in the public IN718 dataset."""
    import pandas as pd

    from .adapters.godec_in718 import GodecIN718Adapter, audit_godec_links
    from .taxonomy import Taxonomy

    cfg = _cfg(config)
    records = GodecIN718Adapter(cfg, Taxonomy.load(cfg.taxonomy_path)).validated_records()
    if not records:
        typer.echo(
            "No public IN718 images found. Run `python scripts/fetch_zenodo_in718.py`."
        )
        raise typer.Exit(1)

    audit = audit_godec_links(cfg.public_in718_dir)
    attached = audit[audit["auto_attach"]]
    candidates = audit[~audit["auto_attach"]]
    typer.echo(f"Public IN718 images:              {audit['record_id'].nunique()}")
    typer.echo(f"Independent condition groups:    {len({r.group_id for r in records})}")
    typer.echo(f"Auto-attached hardness labels:   {len(attached)}")
    typer.echo(
        "Orientation-sensitive candidates: "
        f"{len(candidates[candidates['property_name'] != 'hardness_hv'])}"
    )
    typer.echo(
        f"Validation-eligible links:        {int(audit['validation_eligible'].sum())}"
    )

    group_rows = []
    for group_id in sorted({record.group_id for record in records}):
        group = [record for record in records if record.group_id == group_id]
        group_rows.append(
            {
                "group_id": group_id,
                "images": len(group),
                "hardness_hv": group[0].properties.get("hardness_hv"),
            }
        )
    typer.echo("\nCondition-level hardness links:")
    typer.echo(pd.DataFrame(group_rows).to_string(index=False))
    typer.echo(
        "\nTensile values were not auto-attached: H/V orientation is absent "
        "from the image filenames."
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output, index=False)
        typer.echo(f"\nWrote {len(audit)} retained candidates to {output}")


@app.command("validate-campaign")
def validate_campaign_cmd(
    root: Optional[Path] = typer.Option(
        None,
        "--root",
        help="Campaign directory; defaults to data/experimental_campaign.",
    ),
    config: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Validate specimen, process, SEM, and mechanical-test linkage."""
    from .experimental_campaign import load_campaign

    cfg = _cfg(config)
    tables = load_campaign(root or cfg.experimental_campaign_dir)
    typer.echo("Campaign is valid:")
    for name, count in tables.summary().items():
        typer.echo(f"  {name.replace('_', ' '):<32s} {count}")


@app.command("plan-measurements")
def plan_measurements_cmd(
    limit: int = typer.Option(12, "--limit", "-n", min=1),
    grade: str = typer.Option(
        "grade/ferrous/uhcs_ac1",
        "--grade",
        help="Alloy-grade taxonomy id to keep the first campaign composition-specific.",
    ),
    include_unverified: bool = typer.Option(
        False,
        "--include-unverified",
        help="Also rank samples whose process route needs metadata verification.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional CSV path for the complete plan."
    ),
    config: Optional[Path] = _CONFIG_OPTION,
) -> None:
    """Rank already-imaged UHCS specimens for direct property measurement."""
    from .measurement_plan import plan_uhcs_measurements

    plan = plan_uhcs_measurements(
        _cfg(config),
        limit=limit,
        include_unverified=include_unverified,
        grade=grade,
    )
    if plan.empty:
        typer.echo("No unlabeled, imaged UHCS specimen groups are available.")
        return
    display_columns = [
        "rank",
        "sample_ids",
        "sample_label",
        "image_count",
        "priority_score",
        "rationale",
    ]
    typer.echo(plan[display_columns].to_string(index=False))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output, index=False)
        typer.echo(f"\nWrote {len(plan)} ranked groups to {output}")


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
