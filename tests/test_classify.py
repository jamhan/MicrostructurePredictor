import torch

from microhard.classify import build_classifier, load_classifier, train_classifier
from microhard.transforms import clf_eval_transforms, clf_train_transforms


def test_build_classifier_forward(cfg) -> None:
    model = build_classifier(cfg, num_classes=7)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 7)


def test_transforms_produce_fixed_size() -> None:
    import numpy as np

    image = np.random.default_rng(0).integers(0, 255, (50, 90, 3), dtype=np.uint8)
    for transforms in (clf_train_transforms(64), clf_eval_transforms(64)):
        out = transforms(image=image)["image"]
        assert out.shape == (3, 64, 64)  # pad + crop, never resize
        assert out.dtype == torch.float32


def test_train_classifier_smoke_outputs_taxonomy_ids(synthetic_db) -> None:
    checkpoint = train_classifier(synthetic_db)
    assert checkpoint.exists()

    model, classes = load_classifier(synthetic_db)
    assert all(isinstance(c, tuple) for c in classes)
    assert all(node.startswith("ferrous/") for c in classes for node in c)

    from microhard.classify import classify_image
    import numpy as np

    labels = classify_image(
        model, classes, np.zeros((64, 64, 3), dtype=np.uint8), image_size=64
    )
    assert labels in classes  # output is a tuple of taxonomy node ids
