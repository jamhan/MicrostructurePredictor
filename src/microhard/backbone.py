"""Shared frozen backbone; every task is a separate trainable head.

One encoder (resnet50 with MicroNet or ImageNet weights) is shared by the
router, the constituent classifier, and the segmenter. The backbone is always
frozen — only heads train. If head-only training underperforms, the next step
is LoRA/adapter layers here, not full fine-tuning.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from .config import Config

IMAGENET_NORM = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
PAD_MULTIPLE = 32  # encoder downsamples 5x; H and W must be divisible by 32


def build_encoder(cfg: Config) -> nn.Module:
    """The shared backbone, NOT yet frozen.

    Materialized once to checkpoints/backbone.pt and reloaded from there ever
    after, so every head (router, classifier, segmenter) — whenever it is
    trained or loaded — sees the *identical* frozen weights, even if the
    MicroNet URL is unreachable later.
    """
    encoder = smp.encoders.get_encoder(cfg.encoder, in_channels=3, depth=5, weights=None)
    cache = cfg.backbone_checkpoint
    if cache.exists():
        encoder.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
        return encoder
    if cfg.encoder_weights in {"imagenet", "micronet"}:
        encoder = smp.encoders.get_encoder(
            cfg.encoder, in_channels=3, depth=5, weights="imagenet"
        )
    if cfg.encoder_weights == "micronet":
        load_micronet_weights(encoder, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), cache)
    print(f"[microhard] shared backbone materialized -> {cache}")
    return encoder


def load_micronet_weights(encoder: nn.Module, cfg: Config) -> bool:
    """Best-effort load of NASA MicroNet weights; never blocks.

    On any failure the encoder keeps the ImageNet weights it was built with.
    """
    try:
        state = torch.hub.load_state_dict_from_url(cfg.micronet_url, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        _, unexpected = encoder.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if loaded == 0:
            raise RuntimeError("no keys matched the encoder state dict")
        print(f"[microhard] loaded MicroNet encoder weights ({loaded} tensors)")
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "fall back"
        print(
            f"[microhard] MicroNet weights unavailable ({exc}); using ImageNet weights.\n"
            "            See https://github.com/nasa/pretrained-microscopy-models "
            "(mirror: https://huggingface.co/jstuckner)"
        )
        return False


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def trainable_parameters(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


class ClassifierHead(nn.Module):
    """Frozen backbone + global-average-pool + linear head.

    Used by both the family router and the constituent classifier; only the
    linear layer trains. ``train()`` keeps the frozen encoder in eval mode so
    its BatchNorm statistics never drift.
    """

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = freeze(encoder)
        self.head = nn.Linear(encoder.out_channels[-1], num_classes)

    def train(self, mode: bool = True) -> "ClassifierHead":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)[-1]  # deepest feature map (B, C, H', W')
        return self.head(features.mean(dim=(2, 3)))
