import numpy as np
import pytest

from microhard.entity_graph import (
    detect_salient_regions,
    normalise_grayscale,
    region_graph_from_mask,
)


def test_normalise_grayscale_handles_constant_and_rgb_images() -> None:
    constant = np.full((5, 7), 12, dtype=np.uint16)
    np.testing.assert_array_equal(normalise_grayscale(constant), 0.0)

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, 3:, :] = 255
    gray = normalise_grayscale(rgb)
    assert gray.shape == (4, 6)
    assert gray.min() == 0.0
    assert gray.max() == 1.0


def test_region_graph_extracts_nodes_edges_shapes_and_scale() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[4:8, 5:10] = True
    mask[18:24, 28:34] = True
    image = np.linspace(0, 1, mask.size).reshape(mask.shape)

    graph = region_graph_from_mask(
        mask,
        image,
        pixel_size_um=0.5,
        min_area_px=4,
        k_neighbours=1,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    np.testing.assert_allclose(graph.nodes["area_px"], [20.0, 36.0])
    np.testing.assert_allclose(graph.nodes["area_um2"], [5.0, 9.0])
    assert graph.edges.iloc[0]["distance_um"] == pytest.approx(
        graph.edges.iloc[0]["distance_px"] * 0.5
    )
    assert graph.nodes["eccentricity"].between(0, 1).all()

    summary = graph.summary()
    assert summary["n_nodes"] == 2.0
    assert summary["mean_degree"] == 1.0
    assert summary["largest_component_fraction"] == 1.0
    assert summary["area_fraction"] == pytest.approx(56 / mask.size)


def test_region_graph_filters_small_and_implausibly_large_regions() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 0] = True
    mask[2:18, 2:18] = True
    mask[5:9, 19:20] = True

    graph = region_graph_from_mask(
        mask,
        min_area_px=4,
        max_area_fraction=0.25,
    )
    assert len(graph.nodes) == 1
    assert graph.nodes.iloc[0]["area_px"] == 4


def test_detect_salient_regions_finds_bright_features_and_ignores_footer() -> None:
    y, x = np.mgrid[:120, :160]
    image = (0.2 + 0.001 * x + 0.0005 * y).astype(float)
    image[25:32, 30:38] += 0.8
    image[65:74, 100:110] += 0.8
    image[112:119, 5:150] = 1.0

    mask = detect_salient_regions(
        image,
        background_sigma_px=7,
        z_threshold=2.5,
        min_area_px=8,
        roi_bottom_fraction=0.9,
    )
    graph = region_graph_from_mask(mask, image, min_area_px=8)

    assert mask[25:32, 30:38].mean() > 0.5
    assert mask[65:74, 100:110].mean() > 0.5
    assert not mask[112:, :].any()
    assert len(graph.nodes) == 2


def test_invalid_graph_and_detector_arguments_are_rejected() -> None:
    image = np.zeros((8, 8), dtype=float)
    with pytest.raises(ValueError, match="polarity"):
        detect_salient_regions(image, polarity="neither")
    with pytest.raises(ValueError, match="pixel_size_um"):
        region_graph_from_mask(image, pixel_size_um=0)
