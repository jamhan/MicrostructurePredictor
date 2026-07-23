"""Turn segmented microstructural entities into a physical region graph.

The graph is intentionally material-agnostic.  A node can represent a carbide,
pore, inclusion, connected phase region, or (when the segmentation supports
it) a grain.  Edges encode spatial neighbourhood rather than pretending that
all SEM regions are crystallographic grains.

This module contains a lightweight bright-feature detector for demonstrations
and baselines.  It is not a replacement for a validated semantic segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

NODE_COLUMNS = (
    "node_id",
    "area_px",
    "area_um2",
    "centroid_x_px",
    "centroid_y_px",
    "centroid_x_um",
    "centroid_y_um",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "perimeter_px",
    "eccentricity",
    "orientation_deg",
    "mean_intensity",
)

EDGE_COLUMNS = (
    "source",
    "target",
    "distance_px",
    "distance_um",
    "angle_deg",
    "size_ratio",
    "intensity_contrast",
)


@dataclass(frozen=True)
class RegionGraph:
    """A region-adjacency graph extracted from one field of view."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    image_shape: tuple[int, int]
    pixel_size_um: float | None = None

    def summary(self) -> dict[str, float]:
        """Return fixed-width graph features suitable for tabular models."""

        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        image_area = float(np.prod(self.image_shape))
        if n_nodes == 0:
            return {
                "n_nodes": 0.0,
                "node_density_per_mpx": 0.0,
                "area_fraction": 0.0,
                "mean_diameter_px": 0.0,
                "diameter_cv": 0.0,
                "mean_eccentricity": 0.0,
                "interface_density_px_per_px2": 0.0,
                "mean_neighbour_distance_px": 0.0,
                "mean_degree": 0.0,
                "clustering_coefficient": 0.0,
                "largest_component_fraction": 0.0,
            }

        diameters = self.nodes["equivalent_diameter_px"].to_numpy(dtype=float)
        mean_diameter = float(diameters.mean())
        degrees, adjacency = _adjacency(n_nodes, self.edges)
        clustering = _mean_clustering(adjacency)
        largest_fraction = _largest_component_fraction(adjacency)
        return {
            "n_nodes": float(n_nodes),
            "node_density_per_mpx": float(n_nodes / image_area * 1_000_000),
            "area_fraction": float(self.nodes["area_px"].sum() / image_area),
            "mean_diameter_px": mean_diameter,
            "diameter_cv": float(diameters.std(ddof=0) / mean_diameter)
            if mean_diameter
            else 0.0,
            "mean_eccentricity": float(self.nodes["eccentricity"].mean()),
            "interface_density_px_per_px2": float(
                self.nodes["perimeter_px"].sum() / image_area
            ),
            "mean_neighbour_distance_px": float(self.edges["distance_px"].mean())
            if n_edges
            else 0.0,
            "mean_degree": float(degrees.mean()),
            "clustering_coefficient": clustering,
            "largest_component_fraction": largest_fraction,
        }


def normalise_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to robustly normalised float grayscale in [0, 1]."""

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] < 3:
            array = array[..., 0]
        else:
            rgb = array[..., :3].astype(np.float64)
            array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D or RGB image, got shape {array.shape}")

    gray = array.astype(np.float64)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(gray, dtype=np.float64)
    gray = np.nan_to_num(gray, nan=low, posinf=high, neginf=low)
    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def detect_salient_regions(
    image: np.ndarray,
    *,
    polarity: str = "bright",
    background_sigma_px: float = 12.0,
    z_threshold: float = 3.0,
    min_area_px: int = 8,
    roi_bottom_fraction: float = 0.92,
) -> np.ndarray:
    """Detect locally bright or dark regions for an unsupervised baseline.

    A Gaussian background estimate removes slow illumination changes.  The
    residual is thresholded using a robust median absolute deviation.  The
    bottom strip is excluded by default because SEM scale bars and annotations
    otherwise become very confident but physically meaningless detections.
    """

    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be 'bright' or 'dark'")
    if background_sigma_px <= 0:
        raise ValueError("background_sigma_px must be positive")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    if min_area_px < 1:
        raise ValueError("min_area_px must be at least 1")
    if not 0 < roi_bottom_fraction <= 1:
        raise ValueError("roi_bottom_fraction must be in (0, 1]")

    gray = normalise_grayscale(image)
    cutoff_row = max(1, int(round(gray.shape[0] * roi_bottom_fraction)))
    roi = gray[:cutoff_row]
    background = ndimage.gaussian_filter(roi, sigma=background_sigma_px)
    valid = roi - background
    if polarity == "dark":
        valid = -valid

    centre = float(np.median(valid))
    mad = float(np.median(np.abs(valid - centre)))
    robust_sigma = max(1.4826 * mad, np.finfo(float).eps)
    mask = np.zeros_like(gray, dtype=bool)
    mask[:cutoff_row] = valid > centre + z_threshold * robust_sigma

    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area_px
    keep[0] = False
    # Border-intersecting objects have truncated morphology and are commonly
    # scale bars, frames, or residual illumination edges.  Excluding them makes
    # this a conservative morphometry baseline.
    border_labels = np.unique(
        np.concatenate(
            (
                labels[0, :],
                labels[cutoff_row - 1, :],
                labels[:cutoff_row, 0],
                labels[:cutoff_row, -1],
            )
        )
    )
    keep[border_labels] = False
    return keep[labels]


def region_graph_from_mask(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    *,
    pixel_size_um: float | None = None,
    min_area_px: int = 8,
    max_area_fraction: float = 0.20,
    k_neighbours: int = 3,
    max_nodes: int = 1500,
) -> RegionGraph:
    """Build a symmetric k-nearest-neighbour graph from a binary mask."""

    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[..., 0]
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")
    binary = binary.astype(bool)
    if pixel_size_um is not None and pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive when supplied")
    if min_area_px < 1:
        raise ValueError("min_area_px must be at least 1")
    if not 0 < max_area_fraction <= 1:
        raise ValueError("max_area_fraction must be in (0, 1]")
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be at least 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")

    gray = None
    if image is not None:
        gray = normalise_grayscale(image)
        height = min(gray.shape[0], binary.shape[0])
        width = min(gray.shape[1], binary.shape[1])
        gray = gray[:height, :width]
        binary = binary[:height, :width]

    labels, n_labels = ndimage.label(
        binary, structure=np.ones((3, 3), dtype=bool)
    )
    if n_labels == 0:
        return RegionGraph(
            nodes=pd.DataFrame(columns=NODE_COLUMNS),
            edges=pd.DataFrame(columns=EDGE_COLUMNS),
            image_shape=binary.shape,
            pixel_size_um=pixel_size_um,
        )

    areas = np.bincount(labels.ravel())
    max_area_px = max_area_fraction * binary.size
    component_ids = np.flatnonzero(
        (areas >= min_area_px) & (areas <= max_area_px)
    )
    component_ids = component_ids[component_ids != 0]
    if len(component_ids) > max_nodes:
        order = np.argsort(areas[component_ids])[::-1][:max_nodes]
        component_ids = component_ids[order]
    component_ids = np.sort(component_ids)

    object_slices = ndimage.find_objects(labels)
    node_rows: list[dict[str, float]] = []
    for node_id, component_id in enumerate(component_ids):
        region_slice = object_slices[int(component_id) - 1]
        if region_slice is None:
            continue
        local = labels[region_slice] == component_id
        local_y, local_x = np.nonzero(local)
        y0 = region_slice[0].start or 0
        x0 = region_slice[1].start or 0
        y = local_y.astype(float) + y0
        x = local_x.astype(float) + x0
        area = float(len(x))
        centroid_x = float(x.mean())
        centroid_y = float(y.mean())
        perimeter = float(
            np.count_nonzero(local & ~ndimage.binary_erosion(local))
        )
        eccentricity, orientation = _shape_from_coordinates(x, y)
        mean_intensity = (
            float(gray[region_slice][local].mean()) if gray is not None else np.nan
        )
        equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))
        node_rows.append(
            {
                "node_id": float(node_id),
                "area_px": area,
                "area_um2": area * pixel_size_um**2
                if pixel_size_um is not None
                else np.nan,
                "centroid_x_px": centroid_x,
                "centroid_y_px": centroid_y,
                "centroid_x_um": centroid_x * pixel_size_um
                if pixel_size_um is not None
                else np.nan,
                "centroid_y_um": centroid_y * pixel_size_um
                if pixel_size_um is not None
                else np.nan,
                "equivalent_diameter_px": equivalent_diameter,
                "equivalent_diameter_um": equivalent_diameter * pixel_size_um
                if pixel_size_um is not None
                else np.nan,
                "perimeter_px": perimeter,
                "eccentricity": eccentricity,
                "orientation_deg": orientation,
                "mean_intensity": mean_intensity,
            }
        )

    nodes = pd.DataFrame(node_rows, columns=NODE_COLUMNS)
    edges = _knn_edges(nodes, k_neighbours, pixel_size_um)
    return RegionGraph(
        nodes=nodes,
        edges=edges,
        image_shape=binary.shape,
        pixel_size_um=pixel_size_um,
    )


def _shape_from_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    coordinates = np.column_stack((x - x.mean(), y - y.mean()))
    covariance = coordinates.T @ coordinates / len(coordinates)
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = float(max(eigenvalues[-1], 0.0))
    minor = float(max(eigenvalues[0], 0.0))
    eccentricity = np.sqrt(max(0.0, 1.0 - minor / major)) if major else 0.0
    orientation = 0.5 * np.degrees(
        np.arctan2(
            2.0 * covariance[0, 1],
            covariance[0, 0] - covariance[1, 1],
        )
    )
    return float(eccentricity), float(orientation)


def _knn_edges(
    nodes: pd.DataFrame,
    k_neighbours: int,
    pixel_size_um: float | None,
) -> pd.DataFrame:
    if len(nodes) < 2:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    points = nodes[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
    tree = cKDTree(points)
    k = min(k_neighbours + 1, len(nodes))
    _, neighbours = tree.query(points, k=k)
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]

    pairs: set[tuple[int, int]] = set()
    for source, candidates in enumerate(neighbours):
        for target in np.atleast_1d(candidates):
            target = int(target)
            if source != target:
                pairs.add(tuple(sorted((source, target))))

    rows = []
    for source, target in sorted(pairs):
        delta = points[target] - points[source]
        distance = float(np.linalg.norm(delta))
        area_a = float(nodes.iloc[source]["area_px"])
        area_b = float(nodes.iloc[target]["area_px"])
        intensity_a = float(nodes.iloc[source]["mean_intensity"])
        intensity_b = float(nodes.iloc[target]["mean_intensity"])
        rows.append(
            {
                "source": float(source),
                "target": float(target),
                "distance_px": distance,
                "distance_um": distance * pixel_size_um
                if pixel_size_um is not None
                else np.nan,
                "angle_deg": float(np.degrees(np.arctan2(delta[1], delta[0]))),
                "size_ratio": min(area_a, area_b) / max(area_a, area_b),
                "intensity_contrast": abs(intensity_a - intensity_b)
                if np.isfinite(intensity_a) and np.isfinite(intensity_b)
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _adjacency(
    n_nodes: int, edges: pd.DataFrame
) -> tuple[np.ndarray, list[set[int]]]:
    adjacency = [set() for _ in range(n_nodes)]
    for edge in edges.itertuples(index=False):
        source, target = int(edge.source), int(edge.target)
        adjacency[source].add(target)
        adjacency[target].add(source)
    return np.asarray([len(items) for items in adjacency], dtype=float), adjacency


def _mean_clustering(adjacency: list[set[int]]) -> float:
    coefficients = []
    for neighbours in adjacency:
        degree = len(neighbours)
        if degree < 2:
            coefficients.append(0.0)
            continue
        links = sum(
            target in adjacency[source]
            for source in neighbours
            for target in neighbours
            if source < target
        )
        coefficients.append(2.0 * links / (degree * (degree - 1)))
    return float(np.mean(coefficients)) if coefficients else 0.0


def _largest_component_fraction(adjacency: list[set[int]]) -> float:
    if not adjacency:
        return 0.0
    unseen = set(range(len(adjacency)))
    largest = 0
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 1
        while stack:
            source = stack.pop()
            new = adjacency[source] & unseen
            unseen -= new
            stack.extend(new)
            size += len(new)
        largest = max(largest, size)
    return float(largest / len(adjacency))
