# microhard

Predict material properties from SEM micrographs — multi-material by design,
seeded with the UHCS (ultrahigh carbon steel) dataset:

1. **Route** — a family-level classifier (taxonomy level 1) with conformal
   abstention decides steel vs. other, or answers "unknown family".
2. **Segment/classify microconstituents** — U-Net decoder + linear classifier
   heads over one frozen shared backbone (MicroNet/ImageNet resnet50).
   All labels are taxonomy node ids, never bare strings.
3. **Regress properties** — a named, material-agnostic `FeatureVector`
   (constituent fractions + morphology stats keyed by taxonomy id) feeds
   registered property heads (hardness first: LOO CV, gradient boosting +
   linear baseline). Missing head or unknown family → explicit abstention.

MVP scaffold: working end-to-end beats sophisticated. Runs on a single
consumer GPU, Apple Silicon (MPS), or CPU.

## Architecture

```
                       taxonomy.yaml (family -> constituent -> morphology)
                                        |
 datasets --> adapters/ ----------------+----------------------------------
   UHCS sqlite+images   UHCSAdapter     |    every label = taxonomy node id
   MicroNet-Al (stub)   MicroNetAlAdapter
        |                               |
        v  CanonicalRecord (image, scale_um_per_px, modality,
        |                   group_id, labels?, mask?, properties?)
        |
        |          +------ frozen shared backbone (backbone.pt) ------+
        v          |               |                  |               |
   split BY GROUP  |  router.py    |  classify.py     |  segment.py   |
   (never leak a   |  family head  |  constituent     |  U-Net        |
   sample across   |  + conformal  |  head            |  decoder head |
   splits)         |  abstention   |                  |               |
                   +---------------+------------------+---------------+
                        |                                  |
                        v                                  v
                "unknown family"                 features.py: FeatureVector
                 -> abstain                      frac:ferrous/network, ...
                                                 mean per group (sample)
                                                     |
                                                     v
                                      heads/ registry: (scope, property)
                                      register("ferrous/uhcs",
                                                "hardness_hv", HardnessHead)
                                                     |
                                                     v
              microhard predict <image>
              -> family (or abstain) -> fractions -> HV estimate (or abstain)
```

Design rules enforced by tests:

- **No full-backbone fine-tuning.** One encoder is materialized once to
  `checkpoints/backbone.pt`; router/classifier/segmenter train heads only.
  (If head-only underperforms, add LoRA/adapters — not full fine-tuning.)
- **Splits by group.** `group_id` = physical sample; micrographs of one
  sample never straddle train/val/test.
- **Pad, never resize** — preserves the physical µm/px scale.
- **Heads consume FeatureVector only** — never raw images.
- **Abstain, don't fabricate**: unknown family, family without a segmenter,
  or property without a calibrated head all return recorded abstentions.

## Setup

Python is pinned to 3.12 (`.python-version`) for full wheel availability
across the torch/opencv stack; `uv` fetches it automatically.

```bash
uv sync                 # core deps (torch, smp, albumentations, ...)
uv sync --extra topo    # + optional persistent-homology features (Cubical Ripser)
bash download.sh        # fetch UHCS data (or prints manual URLs)
uv run microhard download   # verify what's on disk
```

## Usage

An executed walkthrough of how the network works — frozen MicroNet encoder,
feature pyramid, U-Net decoder head, live training loop on MPS — is in
[notebooks/how_the_net_works.ipynb](notebooks/how_the_net_works.ipynb)
(re-run with `uv run --with jupyter jupyter lab`).

```bash
uv run microhard taxonomy           # print the label tree
uv run microhard train-seg          # U-Net decoder on 11256/964 masks
uv run microhard train-clf          # constituent classifier head
uv run microhard train-router       # family router + conformal calibration
uv run microhard extract-features   # segment all micrographs -> data/features.csv
uv run microhard fit-hardness       # LOO CV; skips cleanly with no labels
uv run microhard predict path/to/image.tif            # routed
uv run microhard predict path/to/image.tif --family ferrous   # router bypass
```

Configuration is one dataclass ([src/microhard/config.py](src/microhard/config.py))
with TOML overrides:

```toml
# myrun.toml
adapters = ["uhcs", "micronet_al"]  # enabled dataset adapters
encoder_weights = "imagenet"        # micronet | imagenet | none
batch_size = 4
device = "cpu"                      # auto | cpu | cuda | mps
router_alpha = 0.1                  # lower = router abstains more often
```

```bash
uv run microhard train-router -c myrun.toml
```

## Extending (repo skills)

Claude Code skills in `.claude/skills/` walk through the four growth paths:

- **add-adapter** — integrate a new dataset/material family via
  `BaseAdapter`/`ImageFolderAdapter` (the `micronet_al` aluminum stub is the
  template; point it at a real MicroNet/EM3M subset).
- **add-property-head** — register a new `(scope, property)` regressor.
- **edit-taxonomy** — grow `taxonomy.yaml` without orphaning checkpoints.
- **transcribe-hardness** — add HV rows from the Hecht papers and recalibrate.

## Hardness labels

The UHCS sqlite has **no hardness columns**. `data/hardness_labels.csv`
(columns: `sample_label, hardness_hv, source_note`) starts empty; transcribe
HV values from the Hecht papers by hand. Every stage degrades gracefully
("insufficient calibration data") while the file is empty. `sample_label`
must match the `label` column of the sqlite `sample` table.

## Data, licenses, citations

- **UHCSDB — Ultrahigh Carbon Steel micrographs** (961 SEM images + sqlite
  metadata): <https://hdl.handle.net/11256/940>, distributed by NIST under a
  Creative Commons license (see the handle page). Micrographs collected by
  Matt Hecht (CMU); see the Hecht et al. papers on UHCS spheroidite
  coarsening for provenance and hardness measurements.
- **UHCS segmentation benchmark** (pixel-level microconstituent masks):
  <https://hdl.handle.net/11256/964>.
- Cite:
  - DeCost, Francis, Holm, "UHCSDB: UltraHigh Carbon Steel Micrograph
    DataBase," *Integrating Materials and Manufacturing Innovation* 6 (2017).
  - DeCost, Lei, Francis, Holm, "High throughput quantitative metallography
    for complex microstructures using deep learning: a case study in
    ultrahigh carbon steel," *Microscopy and Microanalysis* / IMMI (2019).
  - Stuckner, Harder, Smith, "Microstructure segmentation with deep learning
    encoders pre-trained on a large microscopy dataset," *npj Computational
    Materials* 8, 200 (2022) — MicroNet pretrained encoders,
    <https://github.com/nasa/pretrained-microscopy-models>.

## Testing

```bash
uv run pytest
```

Tests run entirely on synthetic fixtures (tiny generated sqlite + images +
an aluminum stub dataset) — the real datasets are **not** required. The
end-to-end test (tests/test_pipeline.py) proves a non-steel family flows
through the pipeline and abstains on properties instead of fabricating them.

## Known limitations

- **Weak sample-level labels**: hardness is measured per sample, not per
  micrograph; features are averaged across a sample's micrographs, and a
  single-image `predict` inherits that approximation.
- **Tiny calibration set**: hardness labels number in the tens at best, so
  only leave-one-out CV is meaningful and error bars are wide.
- **Router realism**: the non-steel class is a stub until real MicroNet/EM3M
  images are dropped into `data/micronet_al/`; conformal abstention is only
  as good as its calibration distribution.
- **One segmenter per mask taxonomy**: currently ferrous-only; other
  families abstain at the feature stage until they get masks and a segmenter.
- **2D micrographs**: area fractions are stereological proxies for volume
  fractions; morphology stats are in px² (µm conversion via
  `scale_um_per_px` is a TODO).
- **Segmentation ground truth is small** (~tens of images), so the segmenter
  drives most downstream uncertainty.
