"""Single source of configuration for the whole pipeline.

One dataclass, sensible defaults for a single consumer GPU (or CPU/MPS),
optional overrides from a TOML file (stdlib ``tomllib``, so no extra
dependency)::

    cfg = Config.load("myrun.toml")   # or Config() for pure defaults
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# NASA MicroNet resnet50 encoder (github.com/nasa/pretrained-microscopy-models,
# mirrored at huggingface.co/jstuckner). If the URL is unreachable, loading
# falls back to the ImageNet weights.
DEFAULT_MICRONET_URL = (
    "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
    "resnet50_pretrained_microscopynet_v1.1.pth.tar"
)

_PATH_FIELDS = {"data_dir", "checkpoint_dir", "taxonomy_path"}
_ENCODER_WEIGHT_CHOICES = {"micronet", "imagenet", "none"}


@dataclass
class Config:
    # --- locations ---
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    taxonomy_path: Path | None = None  # None -> bundled src/microhard/taxonomy.yaml

    # --- datasets (adapter registry names, see adapters/) ---
    adapters: list[str] = field(default_factory=lambda: ["uhcs"])

    # --- model ---
    encoder: str = "resnet50"
    encoder_weights: str = "micronet"  # micronet | imagenet | none
    micronet_url: str = DEFAULT_MICRONET_URL

    # --- training ---
    image_size: int = 484  # native UHCS micrograph height (after banner strip)
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 20
    num_workers: int = 2
    device: str = "auto"  # auto | cpu | cuda | mps

    # --- splits ---
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # --- family router (conformal abstention) ---
    router_alpha: float = 0.1  # target miscoverage; lower = abstains more often
    router_calib_frac: float = 0.25  # fraction of groups held out for calibration

    def __post_init__(self) -> None:
        if self.encoder_weights not in _ENCODER_WEIGHT_CHOICES:
            raise ValueError(
                f"encoder_weights={self.encoder_weights!r} not in {sorted(_ENCODER_WEIGHT_CHOICES)}"
            )

    # --- derived paths (fixed layout under data_dir / checkpoint_dir) ---
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "microstructures.sqlite"

    @property
    def micrographs_dir(self) -> Path:
        return self.data_dir / "micrographs"

    @property
    def segmentation_dir(self) -> Path:
        """Holds the 11256/964 benchmark: ``uhcs/`` and ``particles/``."""
        return self.data_dir / "segmentation"

    @property
    def hardness_csv(self) -> Path:
        return self.data_dir / "hardness_labels.csv"

    @property
    def features_csv(self) -> Path:
        return self.data_dir / "features.csv"

    @property
    def backbone_checkpoint(self) -> Path:
        """The frozen shared backbone, written once and reused by all heads."""
        return self.checkpoint_dir / "backbone.pt"

    @property
    def seg_checkpoint(self) -> Path:
        return self.checkpoint_dir / "segmenter.pt"

    @property
    def clf_checkpoint(self) -> Path:
        return self.checkpoint_dir / "classifier.pt"

    @property
    def router_checkpoint(self) -> Path:
        return self.checkpoint_dir / "router.pt"

    @property
    def heads_dir(self) -> Path:
        """Fitted property heads, one pickle per (scope, property)."""
        return self.checkpoint_dir / "heads"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Defaults, optionally overridden by a flat TOML file."""
        if path is None:
            return cls()
        raw = tomllib.loads(Path(path).read_text())
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(
                f"Unknown config keys {sorted(unknown)} in {path}; valid keys: {sorted(valid)}"
            )
        kwargs = {k: Path(v) if k in _PATH_FIELDS else v for k, v in raw.items()}
        return cls(**kwargs)

    def resolve_device(self) -> "torch.device":
        import torch  # deferred: keep Config importable without touching torch

        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
