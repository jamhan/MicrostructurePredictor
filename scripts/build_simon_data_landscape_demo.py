"""Build the presentation-ready local-data demonstration notebook.

The notebook deliberately reads verified archives in place. It does not extract
the multi-gigabyte Zenodo downloads or create a second copy of the raw data.
Run it from the repository root:

    uv run --group notebook python scripts/build_simon_data_landscape_demo.py
    uv run --group notebook jupyter nbconvert \
      --to notebook --execute --inplace notebooks/simon_data_landscape_demo.ipynb \
      --ExecutePreprocessor.timeout=900
"""

# Long lines inside notebook cell strings are formatted for readable generated
# cells and are intentionally exempt from the builder's source-line limit.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "simon_data_landscape_demo.ipynb"


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    # Keep source visible: this notebook is both a presentation and a runnable
    # technical demonstration for discussion with a materials scientist.
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        r"""
        # From micrographs to a materials data system

        ## A live tour of the `microhard` data landscape

        **Prepared for a discussion with Simon · 24 July 2026**

        The project began as a small UHCS hardness proof of concept. The local
        corpus now spans three alloy families and contains microscopy, process
        records, composition, EBSD, measured microstructural descriptors,
        replicated mechanical tests, hardness, CALPHAD outputs, and explicit
        provenance.

        > **The data bottleneck has changed.** We have substantial raw material.
        > The next challenge is defensible entity resolution: deciding which
        > images, process states, specimens, and property measurements may be
        > joined without manufacturing false labels.

        This notebook reads the downloaded archives directly from their verified
        ZIP files. It does not extract or duplicate the multi-gigabyte source
        data.
        """
    ),
    code(
        """
        from __future__ import annotations

        import csv
        import io
        import re
        import sqlite3
        from collections import Counter
        from pathlib import Path
        from zipfile import ZipFile

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from IPython.display import Markdown, display
        from PIL import Image
        from sklearn.decomposition import PCA
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from microhard.entity_graph import (
            detect_salient_regions,
            normalise_grayscale,
            region_graph_from_mask,
        )

        ROOT = Path.cwd()
        if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
            ROOT = ROOT.parent
        DATA = ROOT / "data"

        NAVY = "#17324D"
        BLUE = "#3A6EA5"
        TEAL = "#2A9D8F"
        GOLD = "#E9C46A"
        ORANGE = "#F4A261"
        RED = "#E76F51"
        SLATE = "#607080"
        PALE = "#EEF4F7"
        MATERIAL_COLOURS = {"A": NAVY, "B": TEAL, "C": ORANGE, "D": RED}

        plt.rcParams.update(
            {
                "figure.dpi": 130,
                "savefig.dpi": 160,
                "font.size": 10.5,
                "axes.titlesize": 12,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "axes.edgecolor": "#9AA7B2",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.18,
                "grid.linewidth": 0.8,
                "legend.frameon": False,
            }
        )

        def archive_files(path: Path):
            with ZipFile(path) as archive:
                return [
                    item
                    for item in archive.infolist()
                    if not item.is_dir() and not item.filename.startswith("__MACOSX/")
                ]

        def image_from_zip(path: Path, member: str) -> Image.Image:
            with ZipFile(path) as archive:
                image = Image.open(io.BytesIO(archive.read(member)))
                image.load()
            return image

        def display_array(image: Image.Image) -> np.ndarray:
            if image.mode in {"RGB", "RGBA"}:
                return np.asarray(image.convert("RGB"))
            array = np.asarray(image.convert("I"))
            finite = array[np.isfinite(array)]
            low, high = np.percentile(finite, [1, 99])
            if high <= low:
                low, high = float(finite.min()), float(finite.max() or 1)
            return np.clip((array - low) / (high - low), 0, 1)

        def read_zip_text(path: Path, member_suffix: str) -> str:
            with ZipFile(path) as archive:
                matches = [
                    name
                    for name in archive.namelist()
                    if name.endswith(member_suffix) and not name.startswith("__MACOSX/")
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {member_suffix!r} in {path}, got {matches}")
                return archive.read(matches[0]).decode("utf-8-sig", errors="replace")

        def clean_axis(axis):
            axis.grid(axis="y", alpha=0.18)
            axis.spines["left"].set_color("#A9B4BC")
            axis.spines["bottom"].set_color("#A9B4BC")

        print(f"Repository: {ROOT}")
        """
    ),
    markdown(
        """
        ## 1 · What is actually on disk?

        Counts below distinguish **files and fields of view** from independent
        material states. They are an inventory, not a claim about statistical
        sample size.
        """
    ),
    code(
        """
        carbide_root = DATA / "public_in718_carbides_2025" / "raw"
        steel316_root = DATA / "public_316l_composition_2026" / "raw"

        carbide_archives = {
            "Powders": carbide_root / "00_Powders.zip",
            "Chemistry": carbide_root / "01_Chemical_composition.zip",
            "PBF-LB process": carbide_root / "02_PBF-LB_process.zip",
            "CALPHAD": carbide_root / "03_CALPHAD.zip",
            "Microscopy": carbide_root / "04_Microscopy.zip",
            "Mechanical": carbide_root / "05_Mechanical_properties.zip",
            "EBSD": carbide_root / "06_EBSD.zip",
            "Fractures": carbide_root / "07_Fractures.zip",
            "Property map": carbide_root / "08_Tensile property map.zip",
        }
        steel316_archive = steel316_root / "Data-Impact_of_chemical_composition.zip"

        archive_rows = []
        for module, path in carbide_archives.items():
            files = archive_files(path)
            extensions = Counter(Path(item.filename).suffix.lower() for item in files)
            archive_rows.append(
                {
                    "module": module,
                    "files": len(files),
                    "expanded GiB": sum(item.file_size for item in files) / 2**30,
                    "TIFF": extensions[".tif"],
                    "CSV": extensions[".csv"],
                    "Excel": extensions[".xlsx"] + extensions[".xls"],
                }
            )
        archive_inventory = pd.DataFrame(archive_rows)

        uhcs_images = len(
            [path for path in (DATA / "micrographs").iterdir() if path.suffix.lower() in {".tif", ".png"}]
        )
        literature_manifest = pd.read_csv(DATA / "literature_steel" / "manifest.csv")
        godec_audit = pd.read_csv(DATA / "public_in718_godec_2024" / "link_audit.csv")
        godec_hardness = pd.read_csv(DATA / "public_in718_godec_2024" / "hardness.csv")
        godec_tensile = pd.read_csv(DATA / "public_in718_godec_2024" / "tensile.csv")

        microscopy_files = archive_files(carbide_archives["Microscopy"])
        ebsd_files = archive_files(carbide_archives["EBSD"])
        process_files = archive_files(carbide_archives["PBF-LB process"])
        fracture_files = archive_files(carbide_archives["Fractures"])
        powder_files = archive_files(carbide_archives["Powders"])
        steel316_files = archive_files(steel316_archive)

        landscape = pd.DataFrame(
            [
                {
                    "source": "UHCSDB",
                    "material": "2C–4Cr ultrahigh-carbon steel",
                    "microscopy assets": f"{uhcs_images:,} SEM",
                    "property evidence": "7 directly measured hardness groups",
                    "status": "integrated",
                },
                {
                    "source": "Cited steel panels",
                    "material": "Low-alloy and 35CrMo steels",
                    "microscopy assets": f"{len(literature_manifest):,} SEM panels",
                    "property evidence": (
                        f"{literature_manifest['specimen_id'].nunique()} condition/location HV groups"
                    ),
                    "status": "integrated",
                },
                {
                    "source": "Zenodo 14163786",
                    "material": "LPBF IN718; Gaussian/ring beam",
                    "microscopy assets": f"{godec_audit['record_id'].nunique()} BSE-SEM",
                    "property evidence": (
                        f"{len(godec_hardness)} HV states; {len(godec_tensile)} tensile rows"
                    ),
                    "status": "integrated HV",
                },
                {
                    "source": "Zenodo 16603134",
                    "material": "IN718 + NbC/TiC/B₄C",
                    "microscopy assets": (
                        f"{sum(Path(i.filename).suffix.lower() == '.tif' for i in microscopy_files)} "
                        "microscopy TIFF; "
                        f"{sum(Path(i.filename).suffix.lower() == '.tif' for i in ebsd_files)} EBSD TIFF"
                    ),
                    "property evidence": "12 tensile curves; HV1; carbide measurements",
                    "status": "downloaded + verified",
                },
                {
                    "source": "Zenodo 18800251",
                    "material": "LPBF 316L; three powders",
                    "microscopy assets": (
                        f"{sum(Path(i.filename).suffix.lower() == '.tif' for i in steel316_files)} SEM; "
                        "3 EBSD map sets"
                    ),
                    "property evidence": "HV1 replicates; grain-size distributions",
                    "status": "downloaded + verified",
                },
            ]
        )

        display(
            landscape.style
            .hide(axis="index")
            .set_properties(**{"text-align": "left", "border-color": "#DDE5EA"})
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", NAVY),
                            ("color", "white"),
                            ("text-align", "left"),
                        ],
                    }
                ]
            )
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

        asset_counts = pd.Series(
            {
                "Microscopy TIFF": sum(
                    Path(item.filename).suffix.lower() == ".tif" for item in microscopy_files
                ),
                "Fracture SEM": sum(
                    Path(item.filename).suffix.lower() == ".tif" for item in fracture_files
                ),
                "Porosity TIFF": sum(
                    Path(item.filename).suffix.lower() == ".tif" for item in process_files
                ),
                "Powder SEM": sum(
                    Path(item.filename).suffix.lower() == ".tif" for item in powder_files
                ),
                "EBSD TIFF": sum(
                    Path(item.filename).suffix.lower() == ".tif" for item in ebsd_files
                ),
            }
        ).sort_values()
        axes[0].barh(asset_counts.index, asset_counts.values, color=[BLUE, TEAL, GOLD, ORANGE, RED])
        for y, value in enumerate(asset_counts.values):
            axes[0].text(value + 7, y, f"{value:,}", va="center", color=NAVY, fontweight="bold")
        axes[0].set_title("IN718 carbide archive: imaging assets")
        axes[0].set_xlabel("files in verified archives")
        axes[0].set_xlim(0, asset_counts.max() * 1.18)
        axes[0].grid(axis="x", alpha=0.18)

        module_plot = archive_inventory.sort_values("expanded GiB")
        axes[1].barh(module_plot["module"], module_plot["expanded GiB"], color=TEAL)
        for y, value in enumerate(module_plot["expanded GiB"]):
            label = f"{value:.2f}" if value >= 0.01 else "<0.01"
            axes[1].text(value + 0.08, y, label, va="center", color=NAVY, fontsize=9)
        axes[1].set_title("Expanded data represented by each archive")
        axes[1].set_xlabel("GiB (archives remain compressed locally)")
        axes[1].set_xlim(0, module_plot["expanded GiB"].max() * 1.12)
        axes[1].grid(axis="x", alpha=0.18)

        fig.suptitle(
            "The new IN718 source is a multi-modal experimental record—not just an image folder",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

        print(
            f"Verified local source archives represent "
            f"{archive_inventory['expanded GiB'].sum() + sum(i.file_size for i in steel316_files)/2**30:.1f} "
            "GiB when expanded."
        )
        """
    ),
    markdown(
        """
        ### Aerospace sources logged for later—not downloaded

        The source registry now also records the aluminium, titanium, and
        single-crystal superalloy datasets identified during the data scout.
        `queued-metadata` means that only the public record and its intended role
        are logged. No source files have been fetched.
        """
    ),
    code(
        """
        public_sources = pd.read_csv(DATA / "public_sources.csv")
        aerospace_queue = public_sources[
            public_sources["status"].eq("queued-metadata")
        ][
            [
                "source_id",
                "material",
                "images_or_specimens",
                "imaging",
                "properties",
                "status",
            ]
        ]
        display(
            aerospace_queue.style
            .hide(axis="index")
            .set_properties(**{"text-align": "left", "border-color": "#DDE5EA"})
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", NAVY),
                            ("color", "white"),
                            ("text-align", "left"),
                        ],
                    }
                ]
            )
        )
        print(
            f"{len(aerospace_queue)} aerospace candidates are logged as metadata only; "
            "zero new source archives were downloaded."
        )
        """
    ),
    markdown(
        """
        ### What those counts contain

        The carbide-additive IN718 record includes four material variants:

        - **A** — reference IN718;
        - **B** — IN718 + 0.6 wt.% NbC;
        - **C** — IN718 + 0.6 wt.% TiC;
        - **D** — IN718 + 0.2 wt.% micron-scale B₄C.

        Its microscopy archive includes optical fields, SEM in as-built and
        heat-treated states, carbide-focused SEM, binary masks, particle tables,
        and EDS line scans. The process archive includes 80 large porosity TIFFs
        and build parameter workbooks. Fracture images are retained as
        **post-test outcomes** and must not be used to predict the tensile result
        that created them.
        """
    ),
    markdown(
        """
        ## 2 · A visual tour across the corpus

        These are raw or provenance-preserving source images. Different scales,
        detectors, preparation routes, annotations, and bit depths are visible
        immediately—which is why acquisition metadata must be modelled rather
        than silently pooled.
        """
    ),
    code(
        """
        carbide_microscopy = carbide_archives["Microscopy"]
        carbide_ebsd = carbide_archives["EBSD"]

        gallery = [
            (
                Image.open(DATA / "micrographs" / "micrograph1022.png"),
                "UHCS · 970°C / 90 min / quench",
                "Direct label: 611 HV",
            ),
            (
                Image.open(
                    DATA
                    / "literature_steel"
                    / "guan_2026_metals_16_243"
                    / "figure3_e.jpg"
                ),
                "Published low-alloy steel panel",
                "Condition/location link: 183.8 HV1",
            ),
            (
                Image.open(DATA / "public_in718_godec_2024" / "raw" / "BEI AB Gauss 1.tif"),
                "LPBF IN718 · Gaussian beam",
                "Same-archive state: 302.5 ± 4.5 HV1",
            ),
            (
                image_from_zip(
                    carbide_microscopy,
                    "04_Microscopy/SEM AB and HT-treated/PlainImages/B-HT-XZ-100.tif",
                ),
                "IN718 + 0.6 wt.% NbC",
                "Heat-treated XZ-section SEM",
            ),
            (
                image_from_zip(
                    carbide_ebsd,
                    "06_EBSD/AB/B_XZ_AB_x100_IPF_map.tif",
                ),
                "IN718 + NbC · EBSD IPF",
                "As-built XZ section · 1 mm scale",
            ),
            (
                image_from_zip(
                    steel316_archive,
                    "Data-Impact_of_chemical_composition/SEM/"
                    "316L_EOS_80Cel_67rot_DEF_23_1500x.tif",
                ),
                "LPBF 316L · EOS powder",
                "Raw 16-bit SEM · 1500×",
            ),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9.3))
        for axis, (image, title, subtitle) in zip(axes.flat, gallery, strict=True):
            array = display_array(image)
            axis.imshow(array, cmap="gray" if array.ndim == 2 else None)
            axis.set_title(title, color=NAVY, pad=8)
            axis.text(
                0.5,
                -0.055,
                subtitle,
                transform=axis.transAxes,
                ha="center",
                va="top",
                color=SLATE,
                fontsize=9,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#D7E0E5")
        fig.suptitle(
            "The corpus now spans morphology, phase contrast, orientation maps, and literature evidence",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4.8, w_pad=1.0)
        plt.show()
        """
    ),
    markdown(
        """
        ## 3 · What is already linked and auditable?

        The Godec IN718 archive is the first complete runtime demonstration.
        Image filenames encode build strategy, state, and heat-treatment
        temperature. That supports high-confidence links to ten hardness states.

        Tensile measurements are richer but orientation-sensitive. The images do
        not identify H/V orientation, so the notebook shows the data while the
        runtime correctly refuses to auto-attach it.
        """
    ),
    code(
        """
        hardness = godec_hardness.copy()
        hardness["condition"] = hardness["temperature_c"].apply(
            lambda value: "As built" if pd.isna(value) else f"{int(value)}°C"
        )
        condition_order = ["As built", "954°C", "984°C", "1034°C", "1154°C"]

        tensile = godec_tensile.dropna(subset=["yield_strength_mpa"]).copy()
        tensile["condition"] = tensile["temperature_c"].apply(
            lambda value: "As built" if pd.isna(value) else f"{int(value)}°C"
        )
        yield_pairs = tensile.pivot_table(
            index=["build_strategy", "condition"],
            columns="orientation",
            values="yield_strength_mpa",
        ).dropna(subset=["H", "V"])
        yield_pairs = yield_pairs.reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

        for strategy, colour, marker in [("Gauss", BLUE, "o"), ("Ring", ORANGE, "s")]:
            subset = hardness[hardness["build_strategy"] == strategy].set_index("condition")
            subset = subset.reindex(condition_order).dropna(subset=["hardness_hv"])
            positions = [condition_order.index(item) for item in subset.index]
            axes[0].errorbar(
                positions,
                subset["hardness_hv"],
                yerr=subset["hardness_sd_hv"],
                marker=marker,
                markersize=7,
                linewidth=2,
                capsize=3,
                color=colour,
                label=strategy,
            )
        axes[0].set_xticks(range(len(condition_order)), condition_order, rotation=25, ha="right")
        axes[0].set_ylabel("Vickers hardness (HV1)")
        axes[0].set_title("Ten condition-level hardness states")
        axes[0].legend(title="Beam strategy")
        clean_axis(axes[0])

        labels = []
        for index, row in yield_pairs.iterrows():
            colour = BLUE if row["build_strategy"] == "Gauss" else ORANGE
            axes[1].plot([index, index], [row["V"], row["H"]], color=colour, alpha=0.55, linewidth=2)
            axes[1].scatter(index, row["H"], marker="^", s=60, color=colour, zorder=3)
            axes[1].scatter(index, row["V"], marker="v", s=60, facecolor="white", edgecolor=colour, zorder=3)
            labels.append(f"{row['build_strategy'][0]} · {row['condition']}")
        axes[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axes[1].set_ylabel("Yield strength (MPa)")
        axes[1].set_title("Orientation is a real linkage variable")
        axes[1].scatter([], [], marker="^", color=SLATE, label="H orientation")
        axes[1].scatter([], [], marker="v", facecolor="white", edgecolor=SLATE, label="V orientation")
        axes[1].legend()
        clean_axis(axes[1])

        fig.suptitle(
            "A useful matcher attaches what is defensible—and exposes what is not",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        auto_attached = godec_audit[godec_audit["auto_attach"]]
        print(
            f"{auto_attached['record_id'].nunique()} images have auto-attached hardness; "
            f"{godec_audit.loc[~godec_audit['auto_attach'], 'property_name'].count()} "
            "orientation-sensitive candidates remain review-only."
        )
        """
    ),
    markdown(
        """
        ## 4 · The new IN718 archive contains paired structure and response

        The most immediately useful addition is not the image count. It is the
        combination of:

        - four controlled material variants;
        - repeated SEM and EBSD observations;
        - existing carbide masks and particle tables;
        - three independent tensile curves per variant;
        - hardness and a literature property map.

        That supports a family-specific process–structure–property pilot once
        specimen and state keys have been audited.
        """
    ),
    code(
        """
        summary_rows = []
        with ZipFile(carbide_microscopy) as archive:
            for member in archive.namelist():
                if not member.lower().endswith("_summary.csv"):
                    continue
                row = pd.read_csv(io.BytesIO(archive.read(member))).iloc[0]
                summary_rows.append(
                    {
                        "material": Path(member).name[0],
                        "field": row["Slice"],
                        "particle_count": float(row["Count"]),
                        "area_pct": float(row["%Area"]),
                        "mean_particle_area_um2": float(row["Average Size"]),
                    }
                )
        carbide_summary = pd.DataFrame(summary_rows)

        mechanical_archive = carbide_archives["Mechanical"]
        tensile_curves = []
        with ZipFile(mechanical_archive) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "Specimen_RawData" in name
            ]
            for member in members:
                frame = pd.read_csv(
                    io.BytesIO(archive.read(member)),
                    sep=";",
                    header=0,
                    names=["time_s", "extension_mm", "force_n", "blank", "stress_mpa", "strain"],
                )
                material = Path(member).parts[-2].split("_")[0]
                replicate = int(Path(member).stem.rsplit("_", 1)[-1])
                frame = frame[frame["strain"].between(0, 0.35)].copy()
                frame["material"] = material
                frame["replicate"] = replicate
                tensile_curves.append(frame)
        tensile_curves = pd.concat(tensile_curves, ignore_index=True)

        material_names = {
            "A": "A · reference",
            "B": "B · +0.6% NbC",
            "C": "C · +0.6% TiC",
            "D": "D · +0.2% B₄C",
        }

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

        for material, group in carbide_summary.groupby("material"):
            x = np.full(len(group), list("ABCD").index(material), dtype=float)
            x += np.linspace(-0.07, 0.07, len(group))
            axes[0].scatter(
                x,
                group["area_pct"],
                s=52,
                color=MATERIAL_COLOURS[material],
                alpha=0.82,
                label=material_names[material],
            )
            axes[0].plot(
                [list("ABCD").index(material) - 0.16, list("ABCD").index(material) + 0.16],
                [group["area_pct"].mean()] * 2,
                color=NAVY,
                linewidth=2.2,
            )
        axes[0].set_xticks(range(4), [material_names[key] for key in "ABCD"], rotation=20, ha="right")
        axes[0].set_ylabel("Measured carbide area (%)")
        axes[0].set_title("Existing masks already quantify a composition trend")
        clean_axis(axes[0])

        for (material, replicate), curve in tensile_curves.groupby(["material", "replicate"]):
            plot_curve = curve.copy()
            # One supplied TiC trace has a discontinuous terminal sensor point.
            # Break the drawn line there rather than connecting it across the chart.
            discontinuity = plot_curve["strain"].diff().abs().gt(0.02)
            plot_curve.loc[discontinuity, ["strain", "stress_mpa"]] = np.nan
            axes[1].plot(
                plot_curve["strain"] * 100,
                plot_curve["stress_mpa"],
                color=MATERIAL_COLOURS[material],
                alpha=0.72,
                linewidth=1.35,
                label=material_names[material] if replicate == 1 else None,
            )
        axes[1].set_xlim(0, 30)
        axes[1].set_ylim(0, 1450)
        axes[1].set_xlabel("Engineering strain (%)")
        axes[1].set_ylabel("Engineering stress (MPa)")
        axes[1].set_title("Twelve raw tensile curves: three replicates per variant")
        axes[1].legend(fontsize=8)
        clean_axis(axes[1])

        fig.suptitle(
            "The additive series contains both measurable structure and replicated response",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        tensile_summary = (
            tensile_curves.groupby(["material", "replicate"])
            .agg(UTS_MPa=("stress_mpa", "max"), terminal_strain_pct=("strain", lambda x: 100 * x.max()))
            .reset_index()
            .groupby("material")
            .agg(
                replicates=("replicate", "nunique"),
                UTS_mean_MPa=("UTS_MPa", "mean"),
                UTS_sd_MPa=("UTS_MPa", "std"),
                terminal_strain_mean_pct=("terminal_strain_pct", "mean"),
            )
            .round(1)
        )
        tensile_summary.index = [material_names[index] for index in tensile_summary.index]
        display(tensile_summary)
        print(
            "Plotting note: a discontinuous terminal point in TiC replicate 3 is shown as "
            "a line break; the source record itself is unchanged."
        )
        """
    ),
    markdown(
        """
        ### The structure signal is not cosmetic

        The supplied masks report increasing carbide area from the reference
        material through NbC, TiC, and B₄C variants. The three tensile repeats
        also show that the variants are not interchangeable. This is the kind
        of controlled series in which composition, processing, microstructure,
        and response can be modelled together—provided the linkage audit confirms
        the state and specimen relationships.
        """
    ),
    markdown(
        """
        ## 5 · From an SEM field to a physical entity graph

        This is the first runnable version of the proposed representation.
        Every connected carbide region becomes a node. Node attributes encode
        physical size, shape, orientation, and local intensity. Symmetric
        nearest-neighbour edges encode separation, direction, size ratio, and
        contrast.

        The example uses a supplied binary carbide mask, so the graph is based
        on measured entities rather than an unvalidated threshold. The graph is
        correctly described as a **carbide-region graph**, not a grain graph.
        """
    ),
    code(
        """
        def carbide_mask_records(archive_path: Path) -> list[dict]:
            records = []
            with ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                binary_members = [
                    name
                    for name in names
                    if name.lower().endswith("_binary.tif")
                    and "SEM carbides binarized" in name
                ]
                for mask_member in sorted(binary_members):
                    filename = Path(mask_member).name.removeprefix("Drawing of ")
                    image_filename = filename.replace("_binary.tif", ".tif")
                    image_member = str(Path(mask_member).parent / image_filename)
                    if image_member not in names:
                        continue
                    match = re.search(r"_(\\d+)_binary\\.tif$", filename)
                    if match is None:
                        continue
                    material = filename[0]
                    field = match.group(1)
                    summary_member = str(
                        Path(mask_member).parent / f"{material}_{field}_summary.csv"
                    )
                    if summary_member not in names:
                        continue
                    records.append(
                        {
                            "material": material,
                            "field": field,
                            "image_member": image_member,
                            "mask_member": mask_member,
                            "summary_member": summary_member,
                        }
                    )
            return records


        mask_records = carbide_mask_records(carbide_microscopy)
        demo_record = next(
            row
            for row in mask_records
            if row["material"] == "C" and row["field"] == "000"
        )

        with ZipFile(carbide_microscopy) as archive:
            demo_image_pil = Image.open(io.BytesIO(archive.read(demo_record["image_member"])))
            demo_image_pil.load()
            demo_mask_pil = Image.open(io.BytesIO(archive.read(demo_record["mask_member"])))
            demo_mask_pil.load()

        demo_image = np.asarray(demo_image_pil.convert("L"))
        # The supplied ImageJ masks use black carbide regions on a white
        # background, so foreground is the low-valued palette entry.
        demo_mask = np.asarray(demo_mask_pil.convert("L")) < 128
        mask_resolution = demo_mask_pil.info.get("resolution", (18.0, 18.0))
        demo_pixel_size_um = 1.0 / float(mask_resolution[0])

        full_graph = region_graph_from_mask(
            demo_mask,
            demo_image,
            pixel_size_um=demo_pixel_size_um,
            min_area_px=1,
            max_area_fraction=1.0,
            k_neighbours=3,
            max_nodes=5000,
        )
        display_graph = region_graph_from_mask(
            demo_mask,
            demo_image,
            pixel_size_um=demo_pixel_size_um,
            min_area_px=1,
            max_area_fraction=1.0,
            k_neighbours=3,
            max_nodes=220,
        )

        height = min(demo_image.shape[0], demo_mask.shape[0])
        width = min(demo_image.shape[1], demo_mask.shape[1])
        image_crop = demo_image[:height, :width]
        mask_crop = demo_mask[:height, :width]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        axes[0].imshow(display_array(Image.fromarray(image_crop)), cmap="gray")
        axes[0].set_title("Raw carbide-focused SEM")
        axes[1].imshow(mask_crop, cmap="gray")
        axes[1].set_title("Supplied binary carbide mask")
        axes[2].imshow(display_array(Image.fromarray(image_crop)), cmap="gray")

        for edge in display_graph.edges.itertuples(index=False):
            source = display_graph.nodes.iloc[int(edge.source)]
            target = display_graph.nodes.iloc[int(edge.target)]
            axes[2].plot(
                [source.centroid_x_px, target.centroid_x_px],
                [source.centroid_y_px, target.centroid_y_px],
                color=GOLD,
                linewidth=0.45,
                alpha=0.35,
                zorder=2,
            )
        node_areas = display_graph.nodes["area_px"].to_numpy()
        axes[2].scatter(
            display_graph.nodes["centroid_x_px"],
            display_graph.nodes["centroid_y_px"],
            s=8 + 34 * np.sqrt(node_areas / node_areas.max()),
            facecolor=TEAL,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
        axes[2].set_title("Largest 220 nodes + spatial edges")

        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#D7E0E5")
        fig.suptitle(
            "The image becomes a relational, physically scaled microstructure representation",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

        graph_example_summary = pd.Series(full_graph.summary(), name="value").to_frame()
        graph_example_summary.loc["mean_diameter_um", "value"] = (
            full_graph.nodes["equivalent_diameter_um"].mean()
        )
        graph_example_summary.loc["mean_neighbour_distance_um", "value"] = (
            full_graph.edges["distance_um"].mean()
        )
        display(graph_example_summary.round(4))
        print(
            f"{len(full_graph.nodes):,} carbide entities and "
            f"{len(full_graph.edges):,} unique neighbour relations extracted; "
            f"calibration = {demo_pixel_size_um:.4f} µm/px."
        )
        """
    ),
    markdown(
        """
        ## 6A · Self-supervised pretraining across the SEM bank

        Only 14 carbide fields have supplied masks, but representation learning
        does not require a mask. Here a compact encoder first learns from every
        suitable local **pre-test bulk SEM image** by reconstructing randomly
        hidden image blocks.

        The audit is intentionally conservative:

        - included: UHCS, IN718 bulk/cross-section SEM, and 316L SEM;
        - excluded: fracture surfaces, EBSD renderings, optical microscopy,
          powder images, masks, and other derived images;
        - excluded from pretraining: the four carbide fields reserved for the
          final whole-field segmentation test.

        Two random crops are taken from every eligible image, while the footer
        is avoided to reduce microscope-label shortcuts. New hidden blocks,
        rotations, and flips are generated every epoch. This is a small masked
        autoencoder experiment—not a foundation model—but it demonstrates how
        the unlabelled image bank can contribute before property labels exist.
        """
    ),
    code(
        """
        validation_keys = {("A", "005"), ("B", "005"), ("C", "004"), ("D", "035")}
        validation_image_members = {
            record["image_member"]
            for record in mask_records
            if (record["material"], record["field"]) in validation_keys
        }

        image_suffixes = {".tif", ".tiff", ".png"}
        ssl_records = []

        for path in sorted((DATA / "micrographs").iterdir()):
            if path.suffix.lower() in image_suffixes:
                ssl_records.append(
                    {
                        "source": "UHCS",
                        "storage": "file",
                        "path": path,
                        "member": None,
                    }
                )

        for path in sorted((DATA / "public_in718_godec_2024" / "raw").glob("*.tif")):
            ssl_records.append(
                {
                    "source": "IN718 · beam",
                    "storage": "file",
                    "path": path,
                    "member": None,
                }
            )

        with ZipFile(carbide_microscopy) as archive:
            microscopy_names = archive.namelist()
        plain_in718_members = sorted(
            name
            for name in microscopy_names
            if "SEM AB and HT-treated/PlainImages/" in name
            and Path(name).suffix.lower() in image_suffixes
        )
        unlabelled_carbide_members = sorted(
            name
            for name in microscopy_names
            if "SEM carbides binarized HT-treated/" in name
            and Path(name).suffix.lower() in image_suffixes
            and "_binary" not in Path(name).stem.lower()
            and name not in validation_image_members
        )
        for member in plain_in718_members:
            ssl_records.append(
                {
                    "source": "IN718 · bulk",
                    "storage": "zip",
                    "path": carbide_microscopy,
                    "member": member,
                }
            )
        for member in unlabelled_carbide_members:
            ssl_records.append(
                {
                    "source": "IN718 · carbide",
                    "storage": "zip",
                    "path": carbide_microscopy,
                    "member": member,
                }
            )

        with ZipFile(steel316_archive) as archive:
            steel316_sem_members = sorted(
                name
                for name in archive.namelist()
                if "/SEM/" in name
                and Path(name).suffix.lower() in image_suffixes
            )
        for member in steel316_sem_members:
            ssl_records.append(
                {
                    "source": "316L",
                    "storage": "zip",
                    "path": steel316_archive,
                    "member": member,
                }
            )

        ssl_source_counts = (
            pd.Series([record["source"] for record in ssl_records])
            .value_counts()
            .rename_axis("source")
            .rename("images")
            .to_frame()
        )


        def load_ssl_array(record: dict, archives: dict[Path, ZipFile]) -> np.ndarray:
            if record["storage"] == "file":
                with Image.open(record["path"]) as image:
                    array = np.asarray(image.convert("L"))
            else:
                image = Image.open(
                    io.BytesIO(archives[record["path"]].read(record["member"]))
                )
                image.load()
                array = np.asarray(image.convert("L"))
            return normalise_grayscale(array).astype(np.float32)


        ssl_crop_size = 128
        ssl_crops_per_image = 2
        ssl_rng = np.random.default_rng(20260724)
        ssl_patch_bank = []
        ssl_patch_sources = []
        open_ssl_archives = {
            path: ZipFile(path)
            for path in sorted(
                {
                    record["path"]
                    for record in ssl_records
                    if record["storage"] == "zip"
                }
            )
        }
        try:
            for record in ssl_records:
                array = load_ssl_array(record, open_ssl_archives)
                height, width = array.shape
                # The lower part of many SEM files contains scale bars and
                # acquisition labels. Do not let the encoder use them as a
                # shortcut for material or source identity.
                content_height = max(ssl_crop_size, int(0.82 * height))
                for _ in range(ssl_crops_per_image):
                    top = int(
                        ssl_rng.integers(
                            0, max(1, content_height - ssl_crop_size + 1)
                        )
                    )
                    left = int(
                        ssl_rng.integers(0, max(1, width - ssl_crop_size + 1))
                    )
                    patch = array[
                        top : top + ssl_crop_size,
                        left : left + ssl_crop_size,
                    ]
                    if patch.shape != (ssl_crop_size, ssl_crop_size):
                        continue
                    ssl_patch_bank.append(np.round(255 * patch).astype(np.uint8))
                    ssl_patch_sources.append(record["source"])
        finally:
            for archive in open_ssl_archives.values():
                archive.close()

        ssl_patch_bank = np.stack(ssl_patch_bank)
        print(
            f"Audited {len(ssl_records):,} eligible pre-test SEM images and built "
            f"{len(ssl_patch_bank):,} crops ({ssl_patch_bank.nbytes / 2**20:.1f} MiB). "
            f"Four held-out carbide fields remain unseen."
        )
        display(ssl_source_counts)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
        source_order = ["UHCS", "IN718 · beam", "IN718 · bulk", "IN718 · carbide", "316L"]
        count_plot = ssl_source_counts.reindex(source_order).fillna(0)
        axes[0].barh(
            count_plot.index,
            count_plot["images"],
            color=[NAVY, BLUE, TEAL, ORANGE, RED],
        )
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Eligible pre-test SEM images")
        axes[0].set_title("Audited self-supervised corpus")
        for index, value in enumerate(count_plot["images"]):
            axes[0].text(value + 8, index, f"{int(value):,}", va="center", color=NAVY)
        clean_axis(axes[0])

        example_indices = [
            ssl_patch_sources.index(source)
            for source in ("UHCS", "IN718 · beam", "IN718 · carbide", "316L")
        ]
        montage = np.concatenate(
            [ssl_patch_bank[index] for index in example_indices], axis=1
        )
        axes[1].imshow(montage, cmap="gray", vmin=0, vmax=255)
        axes[1].set_xticks(
            np.arange(4) * ssl_crop_size + ssl_crop_size / 2,
            ["UHCS", "IN718 beam", "IN718 carbide", "316L"],
            fontsize=8,
        )
        axes[1].set_yticks([])
        axes[1].set_title("One representation task across alloy families")
        for boundary in range(1, 4):
            axes[1].axvline(
                boundary * ssl_crop_size - 0.5,
                color="white",
                linewidth=2,
            )
        fig.suptitle(
            "Every eligible pre-test SEM image contributes without a manual label",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()
        """
    ),
    code(
        """
        class MaskedSEMPatchDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                patches: np.ndarray,
                *,
                block_size: int = 16,
                mask_fraction: float = 0.55,
                seed: int = 37,
            ):
                self.patches = patches
                self.block_size = block_size
                self.mask_fraction = mask_fraction
                self.seed = seed
                self.epoch = 0

            def __len__(self) -> int:
                return len(self.patches)

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

            def __getitem__(
                self, index: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                rng = np.random.default_rng(
                    self.seed + self.epoch * len(self.patches) + index
                )
                target = self.patches[index].astype(np.float32) / 255.0
                rotations = int(rng.integers(4))
                target = np.rot90(target, rotations)
                if rng.random() < 0.5:
                    target = np.fliplr(target)
                if rng.random() < 0.5:
                    target = np.flipud(target)
                target = np.ascontiguousarray(target)

                block_rows = target.shape[0] // self.block_size
                block_columns = target.shape[1] // self.block_size
                block_mask = rng.random((block_rows, block_columns)) < self.mask_fraction
                pixel_mask = np.repeat(
                    np.repeat(block_mask, self.block_size, axis=0),
                    self.block_size,
                    axis=1,
                )
                corrupted = target.copy()
                corrupted[pixel_mask] = rng.uniform(
                    0.35, 0.65, size=np.count_nonzero(pixel_mask)
                )
                return (
                    torch.from_numpy(corrupted[None]).float(),
                    torch.from_numpy(target[None]).float(),
                    torch.from_numpy(pixel_mask[None]).float(),
                )


        class ConvBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.layers(inputs)


        class MaskedSEMAutoencoder(nn.Module):
            def __init__(self, base_channels: int = 8):
                super().__init__()
                self.encoder_1 = ConvBlock(1, base_channels)
                self.encoder_2 = ConvBlock(base_channels, 2 * base_channels)
                self.bottleneck = ConvBlock(2 * base_channels, 4 * base_channels)
                self.up_2 = nn.ConvTranspose2d(
                    4 * base_channels, 2 * base_channels, 2, stride=2
                )
                self.decoder_2 = ConvBlock(2 * base_channels, 2 * base_channels)
                self.up_1 = nn.ConvTranspose2d(
                    2 * base_channels, base_channels, 2, stride=2
                )
                self.decoder_1 = ConvBlock(base_channels, base_channels)
                self.reconstructor = nn.Conv2d(base_channels, 1, 1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                encoded_1 = self.encoder_1(inputs)
                encoded_2 = self.encoder_2(F.max_pool2d(encoded_1, 2))
                encoded = self.bottleneck(F.max_pool2d(encoded_2, 2))
                decoded_2 = self.decoder_2(self.up_2(encoded))
                decoded_1 = self.decoder_1(self.up_1(decoded_2))
                return self.reconstructor(decoded_1)


        torch.manual_seed(37)
        cnn_device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        ssl_dataset = MaskedSEMPatchDataset(ssl_patch_bank)
        ssl_loader = torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(37),
        )
        ssl_model = MaskedSEMAutoencoder().to(cnn_device)
        ssl_epochs = 12
        ssl_optimiser = torch.optim.AdamW(
            ssl_model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            ssl_optimiser,
            T_max=ssl_epochs,
            eta_min=1e-4,
        )

        ssl_loss_history = []
        for epoch in range(ssl_epochs):
            ssl_dataset.set_epoch(epoch)
            ssl_model.train()
            epoch_losses = []
            for corrupted, target, pixel_mask in ssl_loader:
                corrupted = corrupted.to(cnn_device)
                target = target.to(cnn_device)
                pixel_mask = pixel_mask.to(cnn_device)
                ssl_optimiser.zero_grad(set_to_none=True)
                reconstruction = torch.sigmoid(ssl_model(corrupted))
                absolute_error = torch.abs(reconstruction - target)
                masked_error = (absolute_error * pixel_mask).sum() / pixel_mask.sum()
                full_error = absolute_error.mean()
                loss = 0.9 * masked_error + 0.1 * full_error
                loss.backward()
                ssl_optimiser.step()
                epoch_losses.append(float(loss.detach().cpu()))
            ssl_loss_history.append(float(np.mean(epoch_losses)))
            ssl_scheduler.step()

        ssl_model.eval()
        ssl_dataset.set_epoch(ssl_epochs)
        reconstruction_examples = []
        for example_index in example_indices:
            corrupted, target, pixel_mask = ssl_dataset[example_index]
            with torch.inference_mode():
                reconstruction = torch.sigmoid(
                    ssl_model(corrupted[None].to(cnn_device))
                )[0, 0].cpu().numpy()
            reconstruction_examples.append(
                (
                    corrupted[0].numpy(),
                    target[0].numpy(),
                    pixel_mask[0].numpy().astype(bool),
                    reconstruction,
                )
            )

        fig = plt.figure(figsize=(14, 7.4))
        grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.25])
        for row, (corrupted, target, pixel_mask, reconstruction) in enumerate(
            reconstruction_examples[:2]
        ):
            error = np.abs(reconstruction - target)
            panels = [
                (target, "Original"),
                (corrupted, "55% hidden"),
                (reconstruction, "Reconstruction"),
                (error, "Absolute error"),
            ]
            for column, (panel, title) in enumerate(panels):
                axis = fig.add_subplot(grid[row, column])
                axis.imshow(
                    panel,
                    cmap="magma" if title == "Absolute error" else "gray",
                    vmin=0,
                    vmax=1,
                )
                if row == 0:
                    axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#D7E0E5")
        loss_axis = fig.add_subplot(grid[:, 4])
        loss_axis.plot(
            range(1, ssl_epochs + 1),
            ssl_loss_history,
            marker="o",
            color=TEAL,
            linewidth=2.2,
        )
        loss_axis.set_xlabel("Pretraining epoch")
        loss_axis.set_ylabel("Masked reconstruction L1")
        loss_axis.set_title("The encoder learns SEM structure")
        clean_axis(loss_axis)
        fig.suptitle(
            "Masked-image pretraining: learn texture and morphology before labels",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

        print(
            f"Masked reconstruction loss: {ssl_loss_history[0]:.4f} → "
            f"{ssl_loss_history[-1]:.4f} across {ssl_epochs} epochs on "
            f"{len(ssl_patch_bank):,} multi-alloy crops."
        )
        """
    ),
    markdown(
        """
        ## 6B · Transfer the encoder to carbide identification

        This section trains a genuine convolutional segmentation network—not a
        thresholding rule. The pretrained encoder is transferred to a compact
        U-Net and fine-tuned on ten complete labelled SEM fields. A second U-Net
        starts from random weights under the exact same crop stream and optimiser
        schedule, providing a like-for-like control.

        Four complete fields, one from each material variant, remain held out
        from **both pretraining and supervised training**. Patches from a held-out
        field never enter either training stage.

        Positive-centred patch sampling addresses the severe class imbalance:
        carbides occupy roughly 0.2–2.5% of these images. Random rotations and
        flips are valid here because carbide identity is invariant to the image
        reference frame.

        The archive contains 79 carbide-focused TIFFs but only 14 supplied
        binary masks. Unmasked fields contribute to self-supervision but are not
        silently treated as ground truth. Every supervised epoch still draws
        fresh crops from the ten labelled training fields.
        """
    ),
    code(
        """
        def load_carbide_pair(archive: ZipFile, record: dict) -> dict:
            image_pil = Image.open(io.BytesIO(archive.read(record["image_member"])))
            image_pil.load()
            mask_pil = Image.open(io.BytesIO(archive.read(record["mask_member"])))
            mask_pil.load()
            image = np.asarray(image_pil.convert("L"))
            mask = np.asarray(mask_pil.convert("L")) < 128
            height = min(image.shape[0], mask.shape[0])
            width = min(image.shape[1], mask.shape[1])
            resolution = mask_pil.info.get("resolution", (18.0, 18.0))
            return {
                **record,
                "image": normalise_grayscale(image[:height, :width]).astype(np.float32),
                "mask": mask[:height, :width],
                "pixel_size_um": 1.0 / float(resolution[0]),
            }


        validation_keys = {("A", "005"), ("B", "005"), ("C", "004"), ("D", "035")}
        with ZipFile(carbide_microscopy) as archive:
            carbide_pairs = [load_carbide_pair(archive, record) for record in mask_records]
        cnn_train_pairs = [
            pair
            for pair in carbide_pairs
            if (pair["material"], pair["field"]) not in validation_keys
        ]
        cnn_validation_pairs = [
            pair
            for pair in carbide_pairs
            if (pair["material"], pair["field"]) in validation_keys
        ]


        class CarbidePatchDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                pairs: list[dict],
                *,
                crop_size: int = 128,
                samples_per_epoch: int = 192,
                seed: int = 19,
            ):
                self.pairs = pairs
                self.crop_size = crop_size
                self.samples_per_epoch = samples_per_epoch
                self.seed = seed
                self.epoch = 0
                self.positive_pixels = [np.argwhere(pair["mask"]) for pair in pairs]

            def __len__(self) -> int:
                return self.samples_per_epoch

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

            def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
                # A new deterministic augmentation stream every epoch. The
                # earlier demonstration repeated the same 192 crops ten times.
                rng = np.random.default_rng(
                    self.seed + self.epoch * self.samples_per_epoch + index
                )
                pair_index = int(rng.integers(len(self.pairs)))
                pair = self.pairs[pair_index]
                image, mask = pair["image"], pair["mask"]
                crop = self.crop_size
                height, width = image.shape

                # Half the samples are centred near a known carbide pixel; the
                # other half represent the full field distribution.
                if index % 2 == 0 and len(self.positive_pixels[pair_index]):
                    centre_y, centre_x = self.positive_pixels[pair_index][
                        int(rng.integers(len(self.positive_pixels[pair_index])))
                    ]
                    centre_y += int(rng.integers(-crop // 4, crop // 4 + 1))
                    centre_x += int(rng.integers(-crop // 4, crop // 4 + 1))
                else:
                    centre_y = int(rng.integers(crop // 2, height - crop // 2))
                    centre_x = int(rng.integers(crop // 2, width - crop // 2))

                top = int(np.clip(centre_y - crop // 2, 0, height - crop))
                left = int(np.clip(centre_x - crop // 2, 0, width - crop))
                image_patch = image[top : top + crop, left : left + crop]
                mask_patch = mask[top : top + crop, left : left + crop]

                rotations = int(rng.integers(4))
                image_patch = np.rot90(image_patch, rotations)
                mask_patch = np.rot90(mask_patch, rotations)
                if rng.random() < 0.5:
                    image_patch = np.fliplr(image_patch)
                    mask_patch = np.fliplr(mask_patch)
                if rng.random() < 0.5:
                    image_patch = np.flipud(image_patch)
                    mask_patch = np.flipud(mask_patch)

                image_tensor = torch.from_numpy(
                    np.ascontiguousarray(image_patch[None])
                ).float()
                mask_tensor = torch.from_numpy(
                    np.ascontiguousarray(mask_patch[None])
                ).float()
                return image_tensor, mask_tensor


        class TinyUNet(nn.Module):
            def __init__(self, base_channels: int = 8):
                super().__init__()
                self.encoder_1 = ConvBlock(1, base_channels)
                self.encoder_2 = ConvBlock(base_channels, 2 * base_channels)
                self.bottleneck = ConvBlock(2 * base_channels, 4 * base_channels)
                self.up_2 = nn.ConvTranspose2d(
                    4 * base_channels, 2 * base_channels, 2, stride=2
                )
                self.decoder_2 = ConvBlock(4 * base_channels, 2 * base_channels)
                self.up_1 = nn.ConvTranspose2d(
                    2 * base_channels, base_channels, 2, stride=2
                )
                self.decoder_1 = ConvBlock(2 * base_channels, base_channels)
                self.classifier = nn.Conv2d(base_channels, 1, 1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                level_1 = self.encoder_1(inputs)
                level_2 = self.encoder_2(F.max_pool2d(level_1, 2))
                encoded = self.bottleneck(F.max_pool2d(level_2, 2))
                decoded_2 = self.decoder_2(
                    torch.cat((self.up_2(encoded), level_2), dim=1)
                )
                decoded_1 = self.decoder_1(
                    torch.cat((self.up_1(decoded_2), level_1), dim=1)
                )
                return self.classifier(decoded_1)


        def carbide_segmentation_loss(
            logits: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            positive_weight = torch.tensor([8.0], device=logits.device)
            binary_cross_entropy = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weight
            )
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * targets).sum(dim=(1, 2, 3))
            denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(
                dim=(1, 2, 3)
            )
            dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            return 0.45 * binary_cross_entropy + 0.55 * dice_loss


        cnn_epochs = 30


        def train_segmenter(
            model: nn.Module,
            *,
            seed: int = 19,
        ) -> tuple[nn.Module, list[float], list[float], CarbidePatchDataset]:
            patch_dataset = CarbidePatchDataset(
                cnn_train_pairs,
                samples_per_epoch=384,
                seed=seed,
            )
            patch_loader = torch.utils.data.DataLoader(
                patch_dataset,
                batch_size=12,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed),
            )
            model = model.to(cnn_device)
            optimiser = torch.optim.AdamW(
                model.parameters(), lr=2e-3, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=cnn_epochs,
                eta_min=2e-4,
            )
            loss_history = []
            learning_rates = []
            for epoch in range(cnn_epochs):
                patch_dataset.set_epoch(epoch)
                model.train()
                epoch_losses = []
                learning_rates.append(optimiser.param_groups[0]["lr"])
                for image_batch, mask_batch in patch_loader:
                    image_batch = image_batch.to(cnn_device)
                    mask_batch = mask_batch.to(cnn_device)
                    optimiser.zero_grad(set_to_none=True)
                    logits = model(image_batch)
                    loss = carbide_segmentation_loss(logits, mask_batch)
                    loss.backward()
                    optimiser.step()
                    epoch_losses.append(float(loss.detach().cpu()))
                loss_history.append(float(np.mean(epoch_losses)))
                scheduler.step()
            return model, loss_history, learning_rates, patch_dataset


        # The same seed gives both segmenters identical initial decoder weights.
        # The transfer model then replaces only its three encoder stages.
        torch.manual_seed(19)
        scratch_cnn = TinyUNet()
        torch.manual_seed(19)
        pretrained_cnn = TinyUNet()
        for layer_name in ("encoder_1", "encoder_2", "bottleneck"):
            getattr(pretrained_cnn, layer_name).load_state_dict(
                getattr(ssl_model, layer_name).state_dict()
            )

        scratch_cnn, scratch_loss_history, _, scratch_patch_dataset = (
            train_segmenter(scratch_cnn)
        )
        pretrained_cnn, pretrained_loss_history, cnn_learning_rates, patch_dataset = (
            train_segmenter(pretrained_cnn)
        )

        # Downstream cells use the transferred network; the scratch network is
        # retained for an honest whole-field comparison.
        carbide_cnn = pretrained_cnn
        cnn_loss_history = pretrained_loss_history

        parameter_count = sum(
            parameter.numel() for parameter in carbide_cnn.parameters()
        )
        previous_demo_final_loss = 0.4717
        loss_reduction = (
            1.0 - cnn_loss_history[-1] / previous_demo_final_loss
        )
        print(
            f"Fine-tuned two {parameter_count:,}-parameter CNNs on "
            f"{len(cnn_train_pairs)} complete fields using the same "
            f"{cnn_epochs * len(patch_dataset):,} augmented patches each; held out "
            f"{len(cnn_validation_pairs)} fields from both stages. Device: {cnn_device}. "
            f"Scratch loss: {scratch_loss_history[0]:.4f} → "
            f"{scratch_loss_history[-1]:.4f}; pretrained loss: "
            f"{cnn_loss_history[0]:.4f} → {cnn_loss_history[-1]:.4f}. "
            f"The transferred final loss is {loss_reduction:.1%} below the earlier "
            "0.4717 run."
        )
        """
    ),
    markdown(
        """
        ## 7 · What the CNN identifies on unseen fields

        Full-resolution predictions are assembled from overlapping tiles. The
        probability map shows the model's confidence before thresholding. The
        error overlay separates correct detections from false alarms and missed
        carbides, while the final panel converts the CNN prediction directly
        into the same entity graph used above.
        """
    ),
    code(
        """
        @torch.inference_mode()
        def predict_tiled(
            model: nn.Module,
            image: np.ndarray,
            *,
            device: torch.device,
            tile_size: int = 256,
            stride: int = 192,
            batch_size: int = 6,
        ) -> np.ndarray:
            model.eval()
            height, width = image.shape
            y_starts = list(range(0, max(1, height - tile_size + 1), stride))
            x_starts = list(range(0, max(1, width - tile_size + 1), stride))
            if not y_starts or y_starts[-1] != height - tile_size:
                y_starts.append(max(0, height - tile_size))
            if not x_starts or x_starts[-1] != width - tile_size:
                x_starts.append(max(0, width - tile_size))

            window_1d = 0.15 + 0.85 * np.hanning(tile_size)
            window = np.outer(window_1d, window_1d).astype(np.float32)
            probability_sum = np.zeros((height, width), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)
            locations = [(top, left) for top in y_starts for left in x_starts]

            for start in range(0, len(locations), batch_size):
                batch_locations = locations[start : start + batch_size]
                patches = np.stack(
                    [
                        image[top : top + tile_size, left : left + tile_size]
                        for top, left in batch_locations
                    ]
                )
                tensor = torch.from_numpy(patches[:, None]).float().to(device)
                probabilities = torch.sigmoid(model(tensor)).cpu().numpy()[:, 0]
                for probability, (top, left) in zip(
                    probabilities, batch_locations, strict=True
                ):
                    probability_sum[
                        top : top + tile_size, left : left + tile_size
                    ] += probability * window
                    weight_sum[top : top + tile_size, left : left + tile_size] += window
            return probability_sum / np.maximum(weight_sum, 1e-6)


        def segmentation_pixel_metrics(
            probability: np.ndarray,
            truth: np.ndarray,
            *,
            threshold: float = 0.5,
        ) -> dict[str, float | np.ndarray]:
            prediction = probability >= threshold
            true_positive = np.count_nonzero(prediction & truth)
            false_positive = np.count_nonzero(prediction & ~truth)
            false_negative = np.count_nonzero(~prediction & truth)
            return {
                "prediction": prediction,
                "precision": true_positive / max(1, true_positive + false_positive),
                "recall": true_positive / max(1, true_positive + false_negative),
                "iou": true_positive
                / max(1, true_positive + false_positive + false_negative),
            }


        scratch_validation_rows = []
        for pair in cnn_validation_pairs:
            probability = predict_tiled(
                scratch_cnn, pair["image"], device=cnn_device
            )
            metrics = segmentation_pixel_metrics(probability, pair["mask"])
            scratch_validation_rows.append(
                {
                    "material": pair["material"],
                    "field": pair["field"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "iou": metrics["iou"],
                }
            )
        scratch_validation = pd.DataFrame(scratch_validation_rows)

        validation_results = []
        for pair in cnn_validation_pairs:
            probability = predict_tiled(
                carbide_cnn, pair["image"], device=cnn_device
            )
            truth = pair["mask"]
            metrics = segmentation_pixel_metrics(probability, truth)
            prediction = metrics["prediction"]
            predicted_graph = region_graph_from_mask(
                prediction,
                pair["image"],
                pixel_size_um=pair["pixel_size_um"],
                min_area_px=3,
                max_area_fraction=0.05,
                k_neighbours=3,
                max_nodes=1500,
            )
            validation_results.append(
                {
                    **pair,
                    "probability": probability,
                    "prediction": prediction,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "iou": metrics["iou"],
                    "measured_area_pct": 100 * truth.mean(),
                    "predicted_area_pct": 100 * prediction.mean(),
                    "predicted_graph": predicted_graph,
                }
            )

        cnn_validation = pd.DataFrame(
            [
                {
                    key: result[key]
                    for key in (
                        "material",
                        "field",
                        "precision",
                        "recall",
                        "iou",
                        "measured_area_pct",
                        "predicted_area_pct",
                    )
                }
                for result in validation_results
            ]
        )
        cnn_model_comparison = pd.DataFrame(
            [
                {
                    "initialisation": "Random / scratch",
                    "final training loss": scratch_loss_history[-1],
                    "mean held-out IoU": scratch_validation["iou"].mean(),
                    "mean precision": scratch_validation["precision"].mean(),
                    "mean recall": scratch_validation["recall"].mean(),
                },
                {
                    "initialisation": "Masked-SEM pretrained",
                    "final training loss": cnn_loss_history[-1],
                    "mean held-out IoU": cnn_validation["iou"].mean(),
                    "mean precision": cnn_validation["precision"].mean(),
                    "mean recall": cnn_validation["recall"].mean(),
                },
            ]
        ).set_index("initialisation")

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
        axes[0].plot(
            range(1, len(scratch_loss_history) + 1),
            scratch_loss_history,
            marker="o",
            markersize=3.5,
            color=SLATE,
            linewidth=1.7,
            label="random initialisation",
        )
        axes[0].plot(
            range(1, len(pretrained_loss_history) + 1),
            pretrained_loss_history,
            marker="o",
            markersize=3.5,
            color=TEAL,
            linewidth=2,
            label="masked-SEM pretrained",
        )
        axes[0].axhline(
            previous_demo_final_loss,
            color=RED,
            linestyle="--",
            linewidth=1.3,
            label="earlier final loss · 0.4717",
        )
        axes[0].set_xlabel("Training epoch")
        axes[0].set_ylabel("BCE + Dice loss")
        axes[0].set_title("Same labels and crop stream; different start")
        axes[0].set_xticks(
            range(1, len(cnn_loss_history) + 1, 3)
        )
        axes[0].legend(fontsize=8)
        clean_axis(axes[0])

        maximum_area = 1.15 * max(
            cnn_validation["measured_area_pct"].max(),
            cnn_validation["predicted_area_pct"].max(),
        )
        axes[1].plot(
            [0, maximum_area], [0, maximum_area], color=SLATE, linewidth=1.2
        )
        for row in cnn_validation.itertuples(index=False):
            axes[1].scatter(
                row.measured_area_pct,
                row.predicted_area_pct,
                s=72,
                color=MATERIAL_COLOURS[row.material],
            )
            axes[1].annotate(
                f"{row.material}{row.field}",
                (row.measured_area_pct, row.predicted_area_pct),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
        axes[1].set_xlim(0, maximum_area)
        axes[1].set_ylim(0, maximum_area)
        axes[1].set_xlabel("Measured carbide area (%)")
        axes[1].set_ylabel("CNN-predicted carbide area (%)")
        axes[1].set_title("Whole-field holdouts—not random pixels")
        clean_axis(axes[1])

        fig.suptitle(
            "Does unlabelled SEM pretraining improve whole-field identification?",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

        representative = sorted(validation_results, key=lambda item: item["iou"])[
            len(validation_results) // 2
        ]
        image = representative["image"]
        truth = representative["mask"]
        prediction = representative["prediction"]
        probability = representative["probability"]
        predicted_graph = representative["predicted_graph"]

        base_rgb = np.repeat(image[..., None], 3, axis=2)
        error_overlay = base_rgb.copy()
        overlay_alpha = 0.78
        error_classes = [
            (prediction & truth, np.array([42, 157, 143]) / 255, "correct detection"),
            (prediction & ~truth, np.array([244, 162, 97]) / 255, "false positive"),
            (~prediction & truth, np.array([231, 111, 81]) / 255, "missed carbide"),
        ]
        for pixels, colour, _ in error_classes:
            error_overlay[pixels] = (
                (1 - overlay_alpha) * error_overlay[pixels]
                + overlay_alpha * colour
            )

        graph_for_display = region_graph_from_mask(
            prediction,
            image,
            pixel_size_um=representative["pixel_size_um"],
            min_area_px=3,
            max_area_fraction=0.05,
            k_neighbours=3,
            max_nodes=220,
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Held-out raw SEM field")

        probability_plot = axes[0, 1].imshow(
            probability, cmap="magma", vmin=0, vmax=1
        )
        axes[0, 1].contour(
            truth, levels=[0.5], colors=[TEAL], linewidths=0.65
        )
        axes[0, 1].set_title("CNN probability · teal contour is manual mask")
        colourbar = fig.colorbar(probability_plot, ax=axes[0, 1], fraction=0.035)
        colourbar.set_label("carbide probability")

        axes[1, 0].imshow(error_overlay)
        for _, colour, label in error_classes:
            axes[1, 0].scatter([], [], s=45, color=colour, label=label)
        error_legend = axes[1, 0].legend(
            loc="lower right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            framealpha=0.92,
        )
        for text in error_legend.get_texts():
            text.set_color(NAVY)
        axes[1, 0].set_title(
            f"Identification audit · IoU {representative['iou']:.3f} · "
            f"precision {representative['precision']:.3f} · "
            f"recall {representative['recall']:.3f}"
        )

        axes[1, 1].imshow(image, cmap="gray")
        for edge in graph_for_display.edges.itertuples(index=False):
            source = graph_for_display.nodes.iloc[int(edge.source)]
            target = graph_for_display.nodes.iloc[int(edge.target)]
            axes[1, 1].plot(
                [source.centroid_x_px, target.centroid_x_px],
                [source.centroid_y_px, target.centroid_y_px],
                color=GOLD,
                linewidth=0.45,
                alpha=0.35,
            )
        if len(graph_for_display.nodes):
            areas = graph_for_display.nodes["area_px"].to_numpy()
            axes[1, 1].scatter(
                graph_for_display.nodes["centroid_x_px"],
                graph_for_display.nodes["centroid_y_px"],
                s=8 + 34 * np.sqrt(areas / areas.max()),
                color=TEAL,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )
        axes[1, 1].set_title(
            f"CNN prediction → {len(predicted_graph.nodes):,} entities "
            "(largest 220 shown)"
        )

        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#D7E0E5")
        fig.suptitle(
            f"CNN identification on unseen field "
            f"{representative['material']}{representative['field']}",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        display(
            cnn_validation.assign(
                precision=lambda frame: frame["precision"].round(3),
                recall=lambda frame: frame["recall"].round(3),
                iou=lambda frame: frame["iou"].round(3),
                measured_area_pct=lambda frame: frame["measured_area_pct"].round(3),
                predicted_area_pct=lambda frame: frame["predicted_area_pct"].round(3),
            )
        )
        display(cnn_model_comparison.round(3))
        print(
            f"Transferred mean held-out IoU: {cnn_validation['iou'].mean():.3f}; "
            f"mean precision: {cnn_validation['precision'].mean():.3f}; "
            f"mean recall: {cnn_validation['recall'].mean():.3f}. "
            f"Scratch mean IoU under the same supervised schedule: "
            f"{scratch_validation['iou'].mean():.3f}."
        )
        """
    ),
    markdown(
        """
        ## 8 · Predicting quantitative microstructural characteristics

        Pixel overlap is not the final scientific target. A useful model should
        recover quantities that a metallurgist can inspect and compare:

        - carbide area fraction;
        - particle density;
        - mean equivalent diameter;
        - mean neighbour spacing.

        The decision threshold is selected using only the ten training fields
        and then locked before the four whole-field holdouts are measured. Three
        approaches are compared on those unseen fields:

        1. the training-set mean;
        2. ridge regression on global raw-image texture descriptors;
        3. the CNN segmentation converted into physical entities and measured.

        This is still a small same-study experiment. It tests whether the
        workflow can recover useful characteristics; it does not establish
        cross-microscope or cross-alloy generalisation.
        """
    ),
    code(
        """
        def pair_key(pair: dict) -> str:
            return f"{pair['material']}-{pair['field']}"


        def mask_characteristics(pair: dict, mask: np.ndarray) -> dict[str, float]:
            graph = region_graph_from_mask(
                mask,
                pair["image"],
                pixel_size_um=pair["pixel_size_um"],
                min_area_px=3,
                max_area_fraction=0.05,
                k_neighbours=3,
                max_nodes=5000,
            )
            mean_diameter_um = (
                float(graph.nodes["equivalent_diameter_um"].mean())
                if len(graph.nodes)
                else np.nan
            )
            mean_spacing_um = (
                float(graph.edges["distance_um"].mean())
                if len(graph.edges)
                else np.nan
            )
            return {
                "area_pct": float(100 * mask.mean()),
                "particle_density_per_mpx": float(
                    len(graph.nodes) / (mask.size / 1_000_000)
                ),
                "mean_diameter_um": mean_diameter_um,
                "mean_neighbour_distance_um": mean_spacing_um,
            }


        def texture_descriptor(image: np.ndarray) -> dict[str, float]:
            image = normalise_grayscale(image).astype(np.float32)
            gradient_y, gradient_x = np.gradient(image)
            gradient = np.hypot(gradient_x, gradient_y)
            laplacian = (
                -4 * image
                + np.roll(image, 1, axis=0)
                + np.roll(image, -1, axis=0)
                + np.roll(image, 1, axis=1)
                + np.roll(image, -1, axis=1)
            )

            tensor = torch.from_numpy(image[None, None])
            residual_std = {}
            for kernel in (5, 17, 65):
                local_mean = F.avg_pool2d(
                    tensor,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )[0, 0].numpy()
                residual_std[f"residual_std_{kernel}px"] = float(
                    np.std(image - local_mean)
                )

            downsampled = image[::4, ::4]
            power = np.abs(
                np.fft.fftshift(
                    np.fft.fft2(downsampled - downsampled.mean())
                )
            ) ** 2
            fy = np.fft.fftshift(np.fft.fftfreq(downsampled.shape[0]))
            fx = np.fft.fftshift(np.fft.fftfreq(downsampled.shape[1]))
            radius = np.hypot(fy[:, None], fx[None, :])
            total_power = float(power[radius > 0].sum()) + 1e-12
            spectral = {}
            for name, low, high in (
                ("spectral_low", 0.00, 0.08),
                ("spectral_mid", 0.08, 0.22),
                ("spectral_high", 0.22, 0.50),
            ):
                band = (radius > low) & (radius <= high)
                spectral[name] = float(power[band].sum() / total_power)

            q10, q50, q90 = np.quantile(image, [0.10, 0.50, 0.90])
            return {
                "intensity_mean": float(image.mean()),
                "intensity_std": float(image.std()),
                "intensity_q10": float(q10),
                "intensity_median": float(q50),
                "intensity_q90": float(q90),
                "gradient_mean": float(gradient.mean()),
                "gradient_std": float(gradient.std()),
                "laplacian_std": float(laplacian.std()),
                **residual_std,
                **spectral,
            }


        # Calibrate one operating threshold on training fields, then lock it.
        training_probability = {
            pair_key(pair): predict_tiled(
                carbide_cnn,
                pair["image"],
                device=cnn_device,
            )
            for pair in cnn_train_pairs
        }
        threshold_rows = []
        threshold_grid = np.unique(
            np.concatenate(
                (
                    np.linspace(0.35, 0.95, 25),
                    np.linspace(0.96, 0.995, 8),
                )
            )
        )
        for threshold in threshold_grid:
            field_iou = []
            field_area_error = []
            for pair in cnn_train_pairs:
                prediction = training_probability[pair_key(pair)] >= threshold
                truth = pair["mask"]
                intersection = np.count_nonzero(prediction & truth)
                union = np.count_nonzero(prediction | truth)
                field_iou.append(intersection / max(1, union))
                field_area_error.append(
                    abs(100 * prediction.mean() - 100 * truth.mean())
                )
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "mean_training_iou": np.mean(field_iou),
                    "training_area_mae_pct_points": np.mean(field_area_error),
                }
            )
        threshold_sweep = pd.DataFrame(threshold_rows)
        locked_threshold = float(
            threshold_sweep.loc[
                threshold_sweep["mean_training_iou"].idxmax(), "threshold"
            ]
        )

        truth_rows = []
        for split, pairs in (
            ("train", cnn_train_pairs),
            ("held out", cnn_validation_pairs),
        ):
            for pair in pairs:
                truth_rows.append(
                    {
                        "key": pair_key(pair),
                        "split": split,
                        "material": pair["material"],
                        "field": pair["field"],
                        **mask_characteristics(pair, pair["mask"]),
                        **texture_descriptor(pair["image"]),
                    }
                )
        characteristic_truth = pd.DataFrame(truth_rows).set_index("key")
        train_characteristics = characteristic_truth.query("split == 'train'")
        heldout_characteristics = characteristic_truth.query(
            "split == 'held out'"
        )

        validation_by_key = {
            pair_key(result): result for result in validation_results
        }
        segmented_characteristics = {}
        locked_segmentation_metrics = []
        for pair in cnn_validation_pairs:
            key = pair_key(pair)
            probability = validation_by_key[key]["probability"]
            prediction = probability >= locked_threshold
            segmented_characteristics[key] = mask_characteristics(
                pair, prediction
            )
            truth = pair["mask"]
            true_positive = np.count_nonzero(prediction & truth)
            false_positive = np.count_nonzero(prediction & ~truth)
            false_negative = np.count_nonzero(~prediction & truth)
            locked_segmentation_metrics.append(
                {
                    "key": key,
                    "precision": true_positive
                    / max(1, true_positive + false_positive),
                    "recall": true_positive
                    / max(1, true_positive + false_negative),
                    "iou": true_positive
                    / max(
                        1,
                        true_positive + false_positive + false_negative,
                    ),
                }
            )
        locked_segmentation_metrics = pd.DataFrame(
            locked_segmentation_metrics
        ).set_index("key")

        texture_columns = [
            column
            for column in characteristic_truth.columns
            if column.startswith(
                (
                    "intensity_",
                    "gradient_",
                    "laplacian_",
                    "residual_",
                    "spectral_",
                )
            )
        ]
        characteristic_targets = {
            "area_pct": ("Carbide area", "%"),
            "particle_density_per_mpx": ("Particle density", "regions / Mpx"),
            "mean_diameter_um": ("Mean diameter", "µm"),
            "mean_neighbour_distance_um": ("Mean neighbour spacing", "µm"),
        }

        prediction_methods = {
            "Mean baseline": {},
            "Raw texture ridge": {},
            "CNN → entities": {},
        }
        characteristic_scores = []
        for target, (label, unit) in characteristic_targets.items():
            observed = heldout_characteristics[target].to_numpy()
            baseline = np.full(
                len(heldout_characteristics),
                train_characteristics[target].mean(),
            )
            texture_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0),
            )
            texture_model.fit(
                train_characteristics[texture_columns],
                train_characteristics[target],
            )
            texture_prediction = texture_model.predict(
                heldout_characteristics[texture_columns]
            )
            entity_prediction = np.asarray(
                [
                    segmented_characteristics[key][target]
                    for key in heldout_characteristics.index
                ]
            )
            prediction_methods["Mean baseline"][target] = baseline
            prediction_methods["Raw texture ridge"][target] = (
                texture_prediction
            )
            prediction_methods["CNN → entities"][target] = entity_prediction

            all_observed = characteristic_truth[target].to_numpy()
            observed_span = max(
                1e-9,
                float(np.nanmax(all_observed) - np.nanmin(all_observed)),
            )
            for method, prediction in (
                ("Mean baseline", baseline),
                ("Raw texture ridge", texture_prediction),
                ("CNN → entities", entity_prediction),
            ):
                mae = mean_absolute_error(observed, prediction)
                characteristic_scores.append(
                    {
                        "characteristic": label,
                        "unit": unit,
                        "method": method,
                        "held-out MAE": mae,
                        "range-normalised MAE": mae / observed_span,
                    }
                )
        characteristic_scores = pd.DataFrame(characteristic_scores)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.5, 4.8),
            gridspec_kw={"width_ratios": [1.0, 1.25]},
        )
        axes[0].plot(
            threshold_sweep["threshold"],
            threshold_sweep["mean_training_iou"],
            color=BLUE,
            linewidth=2,
            marker="o",
            markersize=3,
            label="mean training-field IoU",
        )
        axes[0].axvline(
            locked_threshold,
            color=RED,
            linestyle="--",
            linewidth=1.5,
            label=f"locked threshold = {locked_threshold:.2f}",
        )
        axes[0].set_xlabel("CNN probability threshold")
        axes[0].set_ylabel("Mean field IoU")
        axes[0].set_title("Operating point selected before test")
        axes[0].legend(fontsize=8)
        clean_axis(axes[0])

        short_labels = ["Area", "Density", "Diameter", "Spacing"]
        positions = np.arange(len(short_labels))
        width = 0.24
        method_style = [
            ("Mean baseline", SLATE),
            ("Raw texture ridge", ORANGE),
            ("CNN → entities", TEAL),
        ]
        for offset, (method, colour) in enumerate(method_style):
            method_scores = (
                characteristic_scores[
                    characteristic_scores["method"].eq(method)
                ]
                .set_index("characteristic")
                .loc[
                    [
                        characteristic_targets[target][0]
                        for target in characteristic_targets
                    ]
                ]
            )
            axes[1].bar(
                positions + (offset - 1) * width,
                method_scores["range-normalised MAE"],
                width=width,
                color=colour,
                label=method,
            )
        axes[1].set_xticks(positions, short_labels)
        axes[1].set_ylabel("Range-normalised held-out MAE")
        axes[1].set_title("Prediction of physical characteristics")
        axes[1].legend(fontsize=8)
        clean_axis(axes[1])

        fig.suptitle(
            "Segmentation converts raw SEM pixels into testable metallographic quantities",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
        for axis, (target, (label, unit)) in zip(
            axes.flat,
            characteristic_targets.items(),
            strict=True,
        ):
            observed = heldout_characteristics[target].to_numpy()
            texture_prediction = prediction_methods["Raw texture ridge"][
                target
            ]
            entity_prediction = prediction_methods["CNN → entities"][target]
            limits = np.concatenate(
                (observed, texture_prediction, entity_prediction)
            )
            margin = max(1e-6, 0.08 * (limits.max() - limits.min()))
            low, high = limits.min() - margin, limits.max() + margin
            axis.plot([low, high], [low, high], color=SLATE, linewidth=1.1)
            axis.scatter(
                observed,
                texture_prediction,
                s=58,
                marker="s",
                color=ORANGE,
                label="raw texture ridge",
            )
            axis.scatter(
                observed,
                entity_prediction,
                s=65,
                marker="o",
                color=TEAL,
                label="CNN → entities",
            )
            for index, key in enumerate(heldout_characteristics.index):
                axis.annotate(
                    key.replace("-", ""),
                    (observed[index], entity_prediction[index]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=NAVY,
                )
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_xlabel(f"Measured {label.lower()} ({unit})")
            axis.set_ylabel(f"Predicted {label.lower()} ({unit})")
            axis.set_title(label)
            clean_axis(axis)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "Whole-field predictions on four SEM images excluded from CNN training",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        display(
            characteristic_scores.pivot(
                index=["characteristic", "unit"],
                columns="method",
                values="held-out MAE",
            ).round(3)
        )
        print(
            f"Locked threshold {locked_threshold:.2f}: held-out mean IoU "
            f"{locked_segmentation_metrics['iou'].mean():.3f}, precision "
            f"{locked_segmentation_metrics['precision'].mean():.3f}, recall "
            f"{locked_segmentation_metrics['recall'].mean():.3f}. "
            "The four test fields were not used to select this operating point."
        )
        winners = (
            characteristic_scores.loc[
                characteristic_scores.groupby("characteristic")[
                    "held-out MAE"
                ].idxmin(),
                ["characteristic", "method"],
            ]
            .sort_values("characteristic")
            .reset_index(drop=True)
        )
        winner_text = "; ".join(
            f"{row.characteristic}: {row.method}"
            for row in winners.itertuples(index=False)
        )
        display(
            Markdown(
                "### What this experiment says\\n\\n"
                f"- **Held-out segmentation:** IoU "
                f"{locked_segmentation_metrics['iou'].mean():.3f}, precision "
                f"{locked_segmentation_metrics['precision'].mean():.3f}, recall "
                f"{locked_segmentation_metrics['recall'].mean():.3f}.\\n"
                f"- **Lowest error by characteristic:** {winner_text}.\\n"
                "- The entity route is inspectable and physically scaled; the "
                "texture route is a useful same-domain benchmark, not yet an "
                "explanation. Both should be tested on additional studies."
            )
        )
        """
    ),
    markdown(
        """
        ## 9 · Batch graph extraction across every supplied carbide mask

        A useful representation must preserve ordinary quantitative
        metallography before a neural network is added. The next cell extracts
        graphs from all paired masks, checks graph-derived carbide area against
        the source ImageJ summaries, and projects the remaining morphology and
        connectivity descriptors into two dimensions.

        The projection is exploratory. Four material variants and fourteen
        fields are not fourteen independent tensile specimens.
        """
    ),
    code(
        """
        graph_rows = []
        with ZipFile(carbide_microscopy) as archive:
            for record in mask_records:
                image_pil = Image.open(io.BytesIO(archive.read(record["image_member"])))
                image_pil.load()
                mask_pil = Image.open(io.BytesIO(archive.read(record["mask_member"])))
                mask_pil.load()
                image_array = np.asarray(image_pil.convert("L"))
                mask_array = np.asarray(mask_pil.convert("L")) < 128
                resolution = mask_pil.info.get("resolution", (18.0, 18.0))
                pixel_size_um = 1.0 / float(resolution[0])
                graph = region_graph_from_mask(
                    mask_array,
                    image_array,
                    pixel_size_um=pixel_size_um,
                    min_area_px=1,
                    max_area_fraction=1.0,
                    k_neighbours=3,
                    max_nodes=5000,
                )
                source_summary = pd.read_csv(
                    io.BytesIO(archive.read(record["summary_member"]))
                ).iloc[0]
                row = {
                    "material": record["material"],
                    "field": record["field"],
                    "reported_area_pct": float(source_summary["%Area"]),
                    "reported_particle_count": float(source_summary["Count"]),
                    **graph.summary(),
                    "mean_diameter_um": graph.nodes["equivalent_diameter_um"].mean(),
                    "mean_neighbour_distance_um": graph.edges["distance_um"].mean(),
                }
                row["graph_area_pct"] = 100 * row["area_fraction"]
                graph_rows.append(row)

        carbide_graph_fields = pd.DataFrame(graph_rows).sort_values(
            ["material", "field"]
        )
        graph_feature_columns = [
            "node_density_per_mpx",
            "area_fraction",
            "mean_diameter_px",
            "diameter_cv",
            "mean_eccentricity",
            "interface_density_px_per_px2",
            "mean_neighbour_distance_px",
            "clustering_coefficient",
        ]
        graph_matrix = StandardScaler().fit_transform(
            carbide_graph_fields[graph_feature_columns].fillna(0.0)
        )
        graph_projection = PCA(n_components=2).fit_transform(graph_matrix)
        carbide_graph_fields["PC1"] = graph_projection[:, 0]
        carbide_graph_fields["PC2"] = graph_projection[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
        for material, group in carbide_graph_fields.groupby("material"):
            axes[0].scatter(
                group["PC1"],
                group["PC2"],
                s=68,
                color=MATERIAL_COLOURS[material],
                label=material_names[material],
                alpha=0.85,
            )
            for row in group.itertuples(index=False):
                axes[0].annotate(
                    f"{row.material}{row.field}",
                    (row.PC1, row.PC2),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=NAVY,
                )
        axes[0].axhline(0, color="#CDD6DC", linewidth=0.8)
        axes[0].axvline(0, color="#CDD6DC", linewidth=0.8)
        axes[0].set_xlabel("Graph morphology PC1")
        axes[0].set_ylabel("Graph morphology PC2")
        axes[0].set_title("Fields embedded by morphology + connectivity")
        axes[0].legend(fontsize=8)
        clean_axis(axes[0])

        maximum = 1.06 * max(
            carbide_graph_fields["reported_area_pct"].max(),
            carbide_graph_fields["graph_area_pct"].max(),
        )
        axes[1].plot([0, maximum], [0, maximum], color=SLATE, linewidth=1.2)
        for material, group in carbide_graph_fields.groupby("material"):
            axes[1].scatter(
                group["reported_area_pct"],
                group["graph_area_pct"],
                s=68,
                color=MATERIAL_COLOURS[material],
                alpha=0.85,
            )
        axes[1].set_xlim(0, maximum)
        axes[1].set_ylim(0, maximum)
        axes[1].set_xlabel("Source ImageJ carbide area (%)")
        axes[1].set_ylabel("Graph nodes: recovered area (%)")
        axes[1].set_title("Implementation check against supplied measurements")
        clean_axis(axes[1])

        fig.suptitle(
            "The graph retains measured phase fraction and adds relational descriptors",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        area_mae = mean_absolute_error(
            carbide_graph_fields["reported_area_pct"],
            carbide_graph_fields["graph_area_pct"],
        )
        variant_graph_summary = (
            carbide_graph_fields.groupby("material")
            .agg(
                fields=("field", "count"),
                carbide_area_pct=("graph_area_pct", "mean"),
                particles=("n_nodes", "mean"),
                mean_diameter_um=("mean_diameter_um", "mean"),
                neighbour_distance_um=("mean_neighbour_distance_um", "mean"),
                clustering=("clustering_coefficient", "mean"),
            )
            .round(3)
        )
        variant_graph_summary.index = [
            material_names[index] for index in variant_graph_summary.index
        ]
        display(variant_graph_summary)
        print(
            f"Recovered carbide-area MAE versus the supplied ImageJ summaries: "
            f"{area_mae:.4f} percentage points across {len(carbide_graph_fields)} fields."
        )
        """
    ),
    markdown(
        """
        ## 10 · A guarded first property test: ten IN718 hardness states

        The Godec series provides ten condition-level hardness targets and two
        or three SEM fields per state. Here a deliberately simple, unsupervised
        detector finds locally bright regions after removing slow background
        variation. Those regions are converted to graphs and **averaged within
        each material state before validation**.

        Leave-one-state-out predictions compare a mean baseline, process
        metadata, graph features, and the hybrid. This is a pipeline test—not a
        claim that every bright region is a particular phase or that ten states
        establish a production property model.
        """
    ),
    code(
        """
        godec_graph_rows = []
        for image_path in sorted((DATA / "public_in718_godec_2024" / "raw").glob("BEI *.tif")):
            tokens = image_path.stem.split()
            if tokens[1] == "AB":
                state_key = "AB"
                temperature_c = np.nan
                strategy = tokens[2]
                replicate = int(tokens[3])
            else:
                state_key = tokens[2]
                temperature_c = float(tokens[2])
                strategy = tokens[3]
                replicate = int(tokens[4])

            image_array = np.asarray(Image.open(image_path).convert("L"))
            detected = detect_salient_regions(
                image_array,
                polarity="bright",
                background_sigma_px=10.0,
                z_threshold=3.2,
                min_area_px=6,
                roi_bottom_fraction=0.88,
            )
            graph = region_graph_from_mask(
                detected,
                image_array,
                min_area_px=6,
                max_area_fraction=0.03,
                k_neighbours=3,
                max_nodes=1200,
            )
            godec_graph_rows.append(
                {
                    "build_strategy": strategy,
                    "state_key": state_key,
                    "temperature_c": temperature_c,
                    "replicate": replicate,
                    **graph.summary(),
                }
            )

        godec_image_graphs = pd.DataFrame(godec_graph_rows)
        godec_state_graphs = (
            godec_image_graphs.drop(columns="replicate")
            .groupby(["build_strategy", "state_key"], as_index=False)
            .mean(numeric_only=True)
        )
        hardness_targets = godec_hardness.copy()
        hardness_targets["state_key"] = hardness_targets.apply(
            lambda row: "AB"
            if row["state"] == "as-built"
            else str(int(row["temperature_c"])),
            axis=1,
        )
        property_frame = hardness_targets.merge(
            godec_state_graphs,
            on=["build_strategy", "state_key"],
            how="inner",
            suffixes=("", "_graph"),
            validate="one_to_one",
        )
        property_frame["heat_treated"] = (property_frame["state"] != "as-built").astype(float)
        property_frame["temperature_model"] = property_frame["temperature_c"].fillna(0.0)
        property_frame["ring_strategy"] = (
            property_frame["build_strategy"] == "Ring"
        ).astype(float)

        metadata_columns = ["heat_treated", "temperature_model", "ring_strategy"]
        property_graph_columns = [
            "node_density_per_mpx",
            "area_fraction",
            "mean_diameter_px",
            "diameter_cv",
            "mean_eccentricity",
            "interface_density_px_per_px2",
            "mean_neighbour_distance_px",
            "clustering_coefficient",
        ]
        y_hardness = property_frame["hardness_hv"].to_numpy()
        loo = LeaveOneOut()
        model_specs = {
            "Mean baseline": (
                DummyRegressor(strategy="mean"),
                np.ones((len(property_frame), 1)),
            ),
            "Process metadata": (
                make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
                property_frame[metadata_columns].to_numpy(),
            ),
            "Graph only": (
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                property_frame[property_graph_columns].fillna(0.0).to_numpy(),
            ),
            "Hybrid": (
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                property_frame[
                    metadata_columns + property_graph_columns
                ].fillna(0.0).to_numpy(),
            ),
        }
        hardness_predictions = {}
        model_mae = {}
        for name, (model, matrix) in model_specs.items():
            prediction = cross_val_predict(model, matrix, y_hardness, cv=loo)
            hardness_predictions[name] = prediction
            model_mae[name] = mean_absolute_error(y_hardness, prediction)
            property_frame[f"predicted_{name.lower().replace(' ', '_')}"] = prediction

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
        names = list(model_mae)
        values = [model_mae[name] for name in names]
        colours = [SLATE, BLUE, ORANGE, TEAL]
        axes[0].bar(range(len(names)), values, color=colours)
        axes[0].set_xticks(range(len(names)), names, rotation=22, ha="right")
        axes[0].set_ylabel("Leave-one-state-out MAE (HV1)")
        axes[0].set_title("Does microstructure add to process metadata?")
        for index, value in enumerate(values):
            axes[0].text(
                index,
                value + max(values) * 0.025,
                f"{value:.1f}",
                ha="center",
                color=NAVY,
                fontweight="bold",
            )
        axes[0].set_ylim(0, max(values) * 1.18)
        clean_axis(axes[0])

        low = min(y_hardness.min(), min(value.min() for value in hardness_predictions.values())) - 15
        high = max(y_hardness.max(), max(value.max() for value in hardness_predictions.values())) + 15
        axes[1].plot([low, high], [low, high], color=SLATE, linewidth=1.2)
        for name, colour, marker in [
            ("Process metadata", BLUE, "o"),
            ("Graph only", ORANGE, "s"),
            ("Hybrid", TEAL, "^"),
        ]:
            axes[1].scatter(
                y_hardness,
                hardness_predictions[name],
                s=58,
                color=colour,
                marker=marker,
                alpha=0.82,
                label=name,
            )
        axes[1].set_xlim(low, high)
        axes[1].set_ylim(low, high)
        axes[1].set_xlabel("Measured hardness (HV1)")
        axes[1].set_ylabel("Held-out prediction (HV1)")
        axes[1].set_title("Every point was excluded from its fitted model")
        axes[1].legend(fontsize=8)
        clean_axis(axes[1])

        fig.suptitle(
            "A leakage-controlled feasibility test, not a final property model",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        display(
            pd.Series(model_mae, name="leave-one-state-out MAE (HV1)")
            .to_frame()
            .round(1)
        )
        result_columns = [
            "build_strategy",
            "state_key",
            "hardness_hv",
            "predicted_process_metadata",
            "predicted_graph_only",
            "predicted_hybrid",
        ]
        display(property_frame[result_columns].round(1))
        graph_delta = model_mae["Process metadata"] - model_mae["Hybrid"]
        direction = "lower" if graph_delta > 0 else "higher"
        print(
            f"Hybrid MAE is {abs(graph_delta):.1f} HV {direction} than process-only MAE. "
            "With ten states this decides what to investigate next; it does not establish "
            "generalisation across studies or alloy families."
        )
        """
    ),
    markdown(
        """
        ## 11 · A compact 316L experiment adds a second family

        The 316L archive is smaller but unusually coherent: three powder sources,
        five raw SEM fields, three EBSD map sets, full EBSD backing data, HV1
        replicates, and area-weighted grain-size distributions.

        It is useful both as a second-family test and as a warning against
        one-variable explanations.
        """
    ),
    code(
        """
        hardness_text = read_zip_text(steel316_archive, "Hardness Praxair, EOS, SLM.csv")
        hardness_rows = list(csv.reader(io.StringIO(hardness_text), delimiter=";"))
        replicate_columns = {"EOS": 11, "SLM": 12, "Praxair": 13}
        hardness_replicates = {}
        for material, column in replicate_columns.items():
            values = []
            for row in hardness_rows[17:]:
                if column < len(row) and row[column].strip().isdigit():
                    values.append(float(row[column]))
            hardness_replicates[material] = np.asarray(values)

        def decimal_comma(value: str) -> float:
            return float(value.replace(",", "."))

        reported_hardness = {
            "EOS": {
                "mean": decimal_comma(hardness_rows[23][3]),
                "sd": decimal_comma(hardness_rows[24][3]),
            },
            "SLM": {
                "mean": decimal_comma(hardness_rows[12][10]),
                "sd": decimal_comma(hardness_rows[13][10]),
            },
            "Praxair": {
                "mean": decimal_comma(hardness_rows[15][5]),
                "sd": decimal_comma(hardness_rows[16][5]),
            },
        }

        feret_text = read_zip_text(steel316_archive, "Max Feret diameter.csv")
        feret_rows = list(csv.reader(io.StringIO(feret_text), delimiter=";"))

        feret = {}
        for material, start in {"SLM": 0, "Praxair": 6, "EOS": 12}.items():
            points = []
            for row in feret_rows[2:28]:
                low = decimal_comma(row[start])
                high = decimal_comma(row[start + 1])
                weight = decimal_comma(row[start + 2])
                points.append(((low + high) / 2, weight))
            feret[material] = pd.DataFrame(points, columns=["diameter_um", "area_fraction"])

        powder_colours = {"EOS": BLUE, "SLM": TEAL, "Praxair": ORANGE}
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

        positions = np.arange(3)
        order = ["EOS", "SLM", "Praxair"]
        means = [reported_hardness[name]["mean"] for name in order]
        stds = [reported_hardness[name]["sd"] for name in order]
        axes[0].bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=[powder_colours[name] for name in order],
            alpha=0.9,
        )
        for index, name in enumerate(order):
            values = hardness_replicates[name]
            jitter = np.linspace(-0.10, 0.10, len(values))
            axes[0].scatter(
                index + jitter,
                values,
                s=22,
                facecolor="white",
                edgecolor=NAVY,
                linewidth=0.6,
                alpha=0.8,
                zorder=3,
            )
            axes[0].text(index, means[index] + stds[index] + 4, f"{means[index]:.1f}", ha="center", color=NAVY)
        axes[0].set_xticks(positions, order)
        axes[0].set_ylabel("Vickers hardness (HV1)")
        axes[0].set_ylim(180, 245)
        axes[0].set_title("Measured hardness replicates")
        clean_axis(axes[0])

        weighted_means = {}
        for material, frame in feret.items():
            axes[1].plot(
                frame["diameter_um"],
                frame["area_fraction"],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=powder_colours[material],
                label=material,
            )
            weighted_means[material] = np.average(
                frame["diameter_um"], weights=frame["area_fraction"]
            )
        axes[1].set_xlabel("Maximum Feret diameter (µm)")
        axes[1].set_ylabel("Area-weighted fraction")
        axes[1].set_title("EBSD-derived grain-size distributions")
        axes[1].legend(title="Powder source")
        clean_axis(axes[1])

        fig.suptitle(
            "316L: chemistry and solidification history matter alongside grain size",
            x=0.02,
            ha="left",
            color=NAVY,
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        plt.show()

        comparison = pd.DataFrame(
            {
                "reported mean HV1": {name: reported_hardness[name]["mean"] for name in order},
                "reported HV1 SD": {name: reported_hardness[name]["sd"] for name in order},
                "area-weighted mean Feret diameter (µm)": weighted_means,
            }
        ).round(1)
        display(comparison)

        print(
            "EOS has a much larger area-weighted Feret diameter than SLM or Praxair, "
            "yet it is not the softest material. Grain size alone cannot explain this series."
        )
        """
    ),
    markdown(
        """
        ## 12 · Readiness: volume is no longer the only constraint

        The matrix below separates “data exists” from “the runtime can use it
        defensibly.” **Partial** means that relevant metadata exists but the
        exact specimen/state relationship still needs audit.
        """
    ),
    code(
        """
        readiness_labels = ["Not yet", "Partial", "Ready"]
        readiness = pd.DataFrame(
            {
                "SEM / optical": [2, 2, 2, 2, 2],
                "EBSD / structure": [0, 0, 0, 2, 2],
                "Process metadata": [2, 2, 2, 2, 2],
                "Property repeats": [1, 1, 2, 2, 2],
                "Exact specimen key": [2, 0, 0, 1, 1],
                "Runtime adapter": [2, 2, 2, 0, 0],
            },
            index=[
                "UHCSDB",
                "Cited steel panels",
                "IN718 beam strategy",
                "IN718 carbide additives",
                "316L powder comparison",
            ],
        )

        fig, axis = plt.subplots(figsize=(12.5, 4.4))
        cmap = plt.matplotlib.colors.ListedColormap(["#E9EEF1", GOLD, TEAL])
        axis.imshow(readiness.to_numpy(), cmap=cmap, vmin=0, vmax=2, aspect="auto")
        axis.set_xticks(range(readiness.shape[1]), readiness.columns, rotation=25, ha="right")
        axis.set_yticks(range(readiness.shape[0]), readiness.index)
        for row in range(readiness.shape[0]):
            for column in range(readiness.shape[1]):
                value = int(readiness.iloc[row, column])
                axis.text(
                    column,
                    row,
                    readiness_labels[value],
                    ha="center",
                    va="center",
                    color="white" if value == 2 else NAVY,
                    fontweight="bold" if value == 2 else "normal",
                    fontsize=9,
                )
        axis.set_title(
            "Data readiness by source: the new work is linkage and validation",
            loc="left",
            color=NAVY,
            fontsize=15,
            pad=14,
        )
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## What this now enables

        ### Immediate, defensible work

        1. **Promote the entity graph into the runtime.** Persist graph nodes,
           edges, physical calibration, and per-field summary features with
           tests against the supplied ImageJ measurements.
        2. **Build the carbide-IN718 adapter.** Parse the A–D material code,
           state, section, process, and replicate identities while keeping
           fracture surfaces in a separate post-mortem task.
        3. **Audit exact specimen linkage.** Determine whether the tensile,
           hardness, SEM, EBSD, and carbide measurements share physical
           specimens or only material/process batches.
        4. **Run a family-specific representation benchmark.** Compare image
           features, carbide descriptors, EBSD grain statistics, and metadata
           under held-out material/state splits.
        5. **Use 316L as an external-family test.** It can reveal whether a
           representation transfers or merely memorises IN718 acquisition and
           study effects.

        ### Claims the data does not yet support

        - Hundreds of fields do not mean hundreds of independent specimens.
        - Same-archive condition links are useful training evidence, not
          exact-specimen validation.
        - Fracture surfaces cannot be used as pre-test inputs for the tensile
          property that produced them.
        - A generative model is not inverse design until a forward
          process–structure–property model generalises with calibrated
          uncertainty.

        > **Proposed meeting outcome:** agree the minimum linkage key, select one
        > alloy/property benchmark, and define the exact-specimen validation
        > standard before modelling.

        **Local provenance:**

        [Zenodo 14163786](https://doi.org/10.5281/zenodo.14163786) ·
        [Zenodo 16603134](https://doi.org/10.5281/zenodo.16603134) ·
        [Zenodo 18800251](https://doi.org/10.5281/zenodo.18800251)
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (microhard)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "title": "microhard data landscape demonstration",
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")
