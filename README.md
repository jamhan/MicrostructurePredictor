# microhard

Predicts material properties from SEM micrographs. Proof of concept, currently
trained on the UHCS dataset: 961 micrographs of a 2C-4Cr ultrahigh carbon steel
under systematically varied heat treatments (DeCost, Francis & Holm 2017).

The pipeline runs in three stages. A router assigns an incoming image to a
material family, and answers "unknown" when it isn't sure. A U-Net segments the
image into microconstituents; for the steel branch these are ferritic matrix,
proeutectoid cementite network, spheroidite, and Widmanstätten cementite. The
segmentation is then reduced to a feature vector of area fractions and
morphology statistics, and small regression models map those features to
properties, starting with Vickers hardness. Any stage that lacks a trained
model or calibration data reports why and produces nothing, so a prediction is
never silently made up.

```
 download.sh                          hardness_labels.csv (from Hecht thesis)
     |                                        |
     v                                        v
 UHCSDB sqlite + micrographs         (sample_label, hardness_hv)
     |                                        |
     |  adapters/: canonical records          |
     v                                        |
        one frozen resnet50 backbone          |
        (checkpoints/backbone.pt)             |
     |          |               |             |
     v          v               v             |
 router.py   classify.py    segment.py        |
 family +    constituent    U-Net decoder     |
 abstention  classifier     head              |
     |                          |             |
     v                          v             v
 "unknown" -> stop      features.py      heads/: property
                        area fractions,  regressors keyed by
                        morphology       (family, property)
                            |                 |
                            +--------+--------+
                                     v
                microhard predict <image>
                family -> fractions -> property estimate
```

## Results so far

These are feasibility numbers on small data, not benchmarks.

Segmentation: mean IoU 0.495 on validation samples, training only the U-Net
decoder (9M parameters) over a frozen MicroNet encoder, on the 24 pixel-labeled
images of the DeCost benchmark. Full fine-tuning in the original paper reaches
roughly 0.7+, so head-only training leaves accuracy on the table in exchange
for a shared backbone.

Hardness: leave-one-sample-out MAE of about 123 HV across 7 labeled samples
spanning 410 to 876 HV (gradient boosting; a linear baseline diverges at this
sample size). The error has a physical explanation: hardness in this steel is
controlled by the matrix state (martensite vs. pearlite vs. bainite, set by
cooling rate), and the four-class segmentation lumps all of those into one
"matrix" class. Two treatments with 200 HV between them can look identical to
the segmenter. The likely fix is feeding the image-level constituent classifier
(which does distinguish martensite) into the feature vector.

## Setup

Python is pinned to 3.12 (`.python-version`) for wheel availability across the
torch stack; `uv` fetches it automatically.

```bash
uv sync                 # core dependencies
uv sync --extra topo    # optional persistent-homology features (Cubical Ripser)
bash download.sh        # fetch the UHCS data
uv run microhard download   # verify what's on disk
```

The canonical NIST host for the UHCS data has been down since at least July
2026. `download.sh` tries it first, then falls back to
`scripts/fetch_uhcs_mirror.py`, which pulls the metadata and micrographs from
the Materials Data Facility mirror and the segmentation labels from DeCost's
uhcs-segment repository, and applies the standard preprocessing (label files
renamed to sqlite keys, the 38 px instrument banner cropped from images and
masks, int64 label TIFFs converted to uint8 PNGs). Pass `--all` to fetch all
961 micrographs instead of the demo subset.

## Usage

```bash
uv run microhard taxonomy           # print the label tree
uv run microhard train-seg          # U-Net decoder on the benchmark masks
uv run microhard train-clf          # constituent classifier head
uv run microhard train-router       # family router + conformal calibration
uv run microhard extract-features   # segment everything -> data/features.csv
uv run microhard fit-hardness       # leave-one-out CV over labeled samples
uv run microhard predict path/to/image.tif
uv run microhard predict path/to/image.tif --family ferrous   # skip the router
```

Configuration is a single dataclass (`src/microhard/config.py`) with TOML
overrides:

```toml
# myrun.toml
adapters = ["uhcs", "micronet_al"]
encoder_weights = "imagenet"        # micronet | imagenet | none
batch_size = 4
device = "cpu"                      # auto | cpu | cuda | mps
router_alpha = 0.1
```

There is a worked walkthrough of the network itself, from raw pixels to the
hardness fit, in
[notebooks/how_the_net_works.ipynb](notebooks/how_the_net_works.ipynb). It is
committed with outputs, so it reads on GitHub without running anything; to run
it live use `uv run --with jupyter jupyter lab`.

## Design notes

Every label in the system is a node id from `src/microhard/taxonomy.yaml`.
The file holds three axes: what is in the image (family / constituent /
morphology, e.g. `ferrous/pearlite/lamellar`), what the material is
(`grade/ferrous/aisi_1045`), and how it was processed
(`condition/austenitize/water_quench`). Datasets enter through adapters that
emit canonical records against that vocabulary, so adding a material means
writing one adapter and possibly extending the taxonomy, not touching task
code.

Train/validation/test splits are grouped by physical sample. Micrographs of the
same sample are near-duplicates, and splitting them naively would inflate every
validation number in the project.

Images are padded rather than resized. The micron-per-pixel scale is physical
information, and resizing would corrupt it.

The resnet50 encoder is built once, written to `checkpoints/backbone.pt`, and
reused frozen by every task head. This keeps the heads mutually consistent and
means a failed weight download can never silently change the features under a
trained head. If head-only training proves limiting, the intended next step is
LoRA-style adapters, not full fine-tuning.

Property regressors see only feature vectors, never images. That keeps them
cheap to retrain, easy to inspect, and portable across imaging conditions.

## Where this is going

[docs/ROADMAP.md](docs/ROADMAP.md) records the longer arc: the original aim
(topological features of pore and grain structure to predict fatigue, and
eventually generative inverse design of microstructure for a target property),
an honest account of the data constraints that order the work, and the
near-term improvements worth pursuing (foundation-model features,
self-supervised pretraining, assisted annotation, physics-informed and
calibrated property models, and wiring in the persistent-homology featurizer
that `topo.py` already implements).

## Hardness labels

The UHCS sqlite contains no mechanical properties. `data/hardness_labels.csv`
holds values transcribed from Appendix Tables A.1 and A.2 of Matthew Hecht's
PhD thesis (CMU, doi:10.1184/R1/6716156.v1), which report Rockwell C for the
90-minute heat treatments; HV values are interpolated from ASTM E140-07
Table 1. Each row's `source_note` records the table, the original HRC value
with its uncertainty, and any assumption made in matching thesis conditions to
UHCSDB sample labels. Rows marked ASSUMED await verification. The thesis has no
hardness data for the other hold times, so growing past n=7 requires another
source or new measurements.

[docs/DATASET_PLAN.md](docs/DATASET_PLAN.md) is the plan for growing them by
distant supervision: attaching published hardness values to micrographs by
joining on (alloy grade, processing condition). It covers the join key, the
`data/property_lookup.csv` schema and the citation protocol for filling it, the
held-out-condition benchmark that tests whether a model learned physics or just
grade recognition, and why the same technique must not be used for
defect-controlled properties like fatigue. The table ships empty; no property
value in this repo is uncited.

## Data and citations

UHCS micrographs and metadata: <https://hdl.handle.net/11256/940>, distributed
by NIST under a Creative Commons license. Segmentation benchmark:
<https://hdl.handle.net/11256/964>. Micrographs collected by Matthew Hecht
(CMU).

If you use this data, cite:

- DeCost, Francis, Holm, "UHCSDB: UltraHigh Carbon Steel Micrograph DataBase,"
  *Integrating Materials and Manufacturing Innovation* 6 (2017).
- DeCost, Lei, Francis, Holm, "High throughput quantitative metallography for
  complex microstructures using deep learning," *Microscopy and Microanalysis*
  25 (2019).
- Stuckner, Harder, Smith, "Microstructure segmentation with deep learning
  encoders pre-trained on a large microscopy dataset," *npj Computational
  Materials* 8, 200 (2022). MicroNet weights:
  <https://github.com/nasa/pretrained-microscopy-models>.
- Hecht, "Effects of Heat Treatments and Compositional Modification on Carbide
  Network and Matrix Microstructure in Ultrahigh Carbon Steels," PhD thesis,
  Carnegie Mellon University (2017).

## Testing

```bash
uv run pytest
```

The suite runs on synthetic fixtures (a generated sqlite, tiny images, a stub
aluminum dataset) and does not require any real data. One end-to-end test
pushes a non-steel image through the pipeline and checks that property
prediction declines rather than extrapolating.

## Known limitations

Hardness is measured per sample, not per micrograph, so image features are
averaged across each sample's micrographs and a single-image prediction
inherits that approximation. Seven labeled samples support leave-one-out CV
and nothing stronger; error bars are wide. Everything is one alloy family, and
2D area fractions are stereological proxies for volume fractions. Morphology
statistics are currently in pixels squared; converting through each record's
micron-per-pixel scale is planned but not done. The segmentation ground truth
is 24 images, and segmentation error propagates into every downstream number.
