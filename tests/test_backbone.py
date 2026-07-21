import torch

from microhard.backbone import ClassifierHead, build_encoder, trainable_parameters
from microhard.segment import build_segmentation_model


def test_classifier_head_freezes_backbone(cfg) -> None:
    model = ClassifierHead(build_encoder(cfg), num_classes=3)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    # trainable set == the linear head only
    assert len(trainable_parameters(model)) == len(list(model.head.parameters()))
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 3)


def test_classifier_train_mode_keeps_encoder_eval(cfg) -> None:
    model = ClassifierHead(build_encoder(cfg), num_classes=3)
    model.train()
    assert model.training
    assert not model.encoder.training  # frozen BN must not drift


def test_segmentation_model_freezes_encoder_only(cfg) -> None:
    model = build_segmentation_model(cfg, num_classes=4)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.decoder.parameters())
    out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 4, 64, 64)


def test_backbone_materialized_once_and_shared(cfg) -> None:
    first = build_encoder(cfg)
    assert cfg.backbone_checkpoint.exists()
    second = build_encoder(cfg)  # must come from the cache -> identical weights
    for a, b in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(a, b)
