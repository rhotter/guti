"""MIDA scalp-surface geometry helpers.

The downloaded MIDA model stores tissue surfaces as binary STL meshes in a
native anatomical frame. This module samples the outer skin surface and maps it
into the existing GUTI millimeter coordinate convention:

    raw MIDA x -> GUTI x
    raw MIDA z -> GUTI y
    raw MIDA y -> GUTI z

The y-axis in the MIDA surface files is the superior/inferior direction for the
head. Mapping it to GUTI z keeps the old "upward" convention while replacing
the spherical cap with a real scalp mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

import numpy as np

MIDA_ROOT = Path(__file__).resolve().parent / "mida"
MIDA_SURFACE_DIRNAME = "MIDA_v1_surfaces"
MIDA_VOXEL_DIRNAME = "MIDA_v1_voxels"
MIDA_NIFTI_FILE = "MIDA_v1.nii"
MIDA_LABEL_FILE = "MIDA_v1.txt"
MIDA_TISSUE_PROPERTIES_FILE = "tissue_properties.json"
MIDA_SCALP_SURFACE = "Epidermis_Dermis.stl"
MIDA_SKULL_SURFACE = "Skull Outer Table .stl"
GUTI_HEAD_CENTER_MM = np.array([80.0, 80.0, 0.0], dtype=np.float64)

MIDA_BRAIN_REFERENCE_SURFACES = (
    "Brain Gray Matter.stl",
    "Brain White Matter.stl",
    "Brainstem Midbrain.stl",
    "Brainstem Pons.stl",
    "Brainstem Medulla.stl",
    "Cerebellum Gray Matter.stl",
    "Cerebellum White Matter.stl",
)

ScalpRegion = Literal["superior", "cranial", "full"]
ScalpSamplingMethod = Literal["area", "projected_fibonacci", "max_distance"]
CoordinateFrame = Literal["raw", "guti"]

MIDA_BRAIN_SOURCE_TISSUES = frozenset(
    {
        "Amygdala",
        "Brain Gray Matter",
        "Brain White Matter",
        "Brainstem Medulla",
        "Brainstem Midbrain",
        "Brainstem Pons",
        "Caudate Nucleus",
        "Cerebellum Gray Matter",
        "Cerebellum White Matter",
        "Cerebral Peduncles",
        "Commissura (Anterior)",
        "Commissura (Posterior)",
        "Globus Pallidus",
        "Hippocampus",
        "Hypothalamus",
        "Mammillary Body",
        "Nucleus Accumbens",
        "Optic Chiasm",
        "Optic Tract",
        "Pineal Body",
        "Putamen",
        "Substantia Nigra",
        "Thalamus",
    }
)


@dataclass(frozen=True)
class MidaBemLayer:
    """One nested MIDA tissue layer usable by an OpenMEEG BEM head model."""

    domain_name: str
    interface_name: str
    surface_name: str
    tissue_name: str


@dataclass(frozen=True)
class MidaScalpAperture:
    """A named circular scalp aperture in GUTI millimeter coordinates."""

    name: str
    center_mm: tuple[float, float, float]
    radius_mm: float
    description: str


MIDA_EEG_BEM_LAYERS = (
    MidaBemLayer("Brain", "Brain", "Brain Gray Matter.stl", "Brain Gray Matter"),
    MidaBemLayer("CSF", "CSF", "CSF General.stl", "CSF General"),
    MidaBemLayer("Dura", "Dura", "Dura.stl", "Dura"),
    MidaBemLayer(
        "SkullInnerTable",
        "SkullInnerTable",
        "Skull Inner Table.stl",
        "Skull Inner Table",
    ),
    MidaBemLayer("SkullDiploe", "SkullDiploe", "Skull Diploe.stl", "Skull Diploe"),
    MidaBemLayer(
        "SkullOuterTable",
        "SkullOuterTable",
        "Skull Outer Table .stl",
        "Skull Outer Table",
    ),
    MidaBemLayer(
        "SubcutaneousAdipose",
        "SubcutaneousAdipose",
        "Subcutaneous Adipose Tissue.stl",
        "Subcutaneous Adipose Tissue",
    ),
    MidaBemLayer("Scalp", "Scalp", "Epidermis_Dermis.stl", "Epidermis_Dermis"),
)

MIDA_US_TEMPORAL_APERTURE_RADIUS_MM = 28.0
MIDA_US_OCCIPITAL_APERTURE_RADIUS_MM = 24.0


def mida_eeg_bem_layers() -> tuple[MidaBemLayer, ...]:
    """Return the nested MIDA layers used for the EEG OpenMEEG head model."""
    return MIDA_EEG_BEM_LAYERS


def mida_eeg_bem_layer_properties(
    *,
    mida_root: str | Path | None = None,
) -> dict[str, dict]:
    """Return MIDA physical-property records keyed by EEG BEM domain name."""
    return {
        layer.domain_name: mida_tissue_properties(layer.tissue_name, mida_root=mida_root)
        for layer in MIDA_EEG_BEM_LAYERS
    }


def _mida_side_center(
    surface_name: str,
    *,
    side: Literal["left", "right"],
    mida_root: str | Path | None = None,
) -> np.ndarray:
    points = load_mida_surface_triangles(
        surface_name,
        mida_root=mida_root,
        coordinate_frame="guti",
    ).reshape(-1, 3)
    if side == "left":
        side_points = points[points[:, 0] < GUTI_HEAD_CENTER_MM[0]]
    elif side == "right":
        side_points = points[points[:, 0] >= GUTI_HEAD_CENTER_MM[0]]
    else:
        raise ValueError("side must be 'left' or 'right'")
    if side_points.size == 0:
        raise ValueError(f"No {side} points found in MIDA surface {surface_name!r}")
    return side_points.mean(axis=0)


def mida_us_acoustic_windows(
    *,
    mida_root: str | Path | None = None,
) -> tuple[MidaScalpAperture, ...]:
    """Return the four ultrasound receiver apertures on the MIDA scalp.

    The two temporal windows are placed superior/anterior to the ear canal,
    matching the usual transtemporal TCD window above the zygomatic arch and in
    front of the tragus. The two occipital windows are placed on the lower
    posterior scalp, split left/right around the occipital belly to approximate
    paired suboccipital/transforaminal access.
    """
    windows: list[MidaScalpAperture] = []
    for side in ("left", "right"):
        ear_center = _mida_side_center(
            "Ear Auditory Canal.stl",
            side=side,
            mida_root=mida_root,
        )
        temporal_center = (
            float(ear_center[0]),
            float(ear_center[1] - 12.0),
            float(ear_center[2] + 55.0),
        )
        windows.append(
            MidaScalpAperture(
                name=f"{side}_temporal",
                center_mm=temporal_center,
                radius_mm=MIDA_US_TEMPORAL_APERTURE_RADIUS_MM,
                description="Transtemporal window above the zygomatic arch and anterior/superior to the ear canal.",
            )
        )

    for side in ("left", "right"):
        occipital_center = _mida_side_center(
            "Muscle - Occipitiofrontalis - Occipital Belly.stl",
            side=side,
            mida_root=mida_root,
        )
        lower_occipital_center = (
            float(occipital_center[0]),
            float(occipital_center[1] - 2.0),
            float(occipital_center[2] - 12.0),
        )
        windows.append(
            MidaScalpAperture(
                name=f"{side}_occipital",
                center_mm=lower_occipital_center,
                radius_mm=MIDA_US_OCCIPITAL_APERTURE_RADIUS_MM,
                description="Lower posterior occipital/suboccipital window approximating access toward the foramen magnum.",
            )
        )
    return tuple(windows)


def mida_model_available(mida_root: str | Path | None = None) -> bool:
    """Return whether the local MIDA scalp STL is available."""
    return _surface_path(MIDA_SCALP_SURFACE, mida_root).exists()


def _surface_path(surface_name: str, mida_root: str | Path | None = None) -> Path:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return root / MIDA_SURFACE_DIRNAME / surface_name


def _voxel_path(filename: str, mida_root: str | Path | None = None) -> Path:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return root / MIDA_VOXEL_DIRNAME / filename


def _properties_path(mida_root: str | Path | None = None) -> Path:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return root / MIDA_TISSUE_PROPERTIES_FILE


@lru_cache(maxsize=16)
def _read_binary_stl_triangles_cached(path_str: str) -> np.ndarray:
    """Read a binary STL and return triangles with shape ``(n, 3, 3)``."""
    path = Path(path_str)
    with path.open("rb") as f:
        f.seek(80)
        n_triangles_arr = np.fromfile(f, dtype="<u4", count=1)
        if n_triangles_arr.size != 1:
            raise ValueError(f"{path} is not a valid binary STL")
        n_triangles = int(n_triangles_arr[0])
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        records = np.fromfile(f, dtype=dtype, count=n_triangles)

    if records.shape[0] != n_triangles:
        raise ValueError(
            f"{path} ended early: expected {n_triangles} STL triangles, "
            f"read {records.shape[0]}"
        )
    triangles = records["vertices"].astype(np.float64, copy=True)
    triangles.setflags(write=False)
    return triangles


def load_mida_surface_triangles(
    surface_name: str = MIDA_SCALP_SURFACE,
    *,
    mida_root: str | Path | None = None,
    coordinate_frame: CoordinateFrame = "guti",
) -> np.ndarray:
    """Load a MIDA tissue surface as triangles.

    Parameters
    ----------
    surface_name:
        Filename under ``MIDA_v1_surfaces``.
    mida_root:
        Optional root directory containing the downloaded MIDA model.
    coordinate_frame:
        ``"raw"`` returns MIDA's native STL coordinates. ``"guti"`` maps the
        points into GUTI's millimeter coordinate convention.
    """
    path = _surface_path(surface_name, mida_root)
    if not path.exists():
        raise FileNotFoundError(f"MIDA surface file not found: {path}")

    raw = _read_binary_stl_triangles_cached(str(path))
    if coordinate_frame == "raw":
        return raw.copy()
    if coordinate_frame != "guti":
        raise ValueError("coordinate_frame must be 'raw' or 'guti'")
    return transform_mida_to_guti(raw, mida_root=mida_root)


@lru_cache(maxsize=4)
def _mida_reference_center_raw_cached(mida_root_str: str) -> np.ndarray:
    root = Path(mida_root_str)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []

    for surface_name in MIDA_BRAIN_REFERENCE_SURFACES:
        path = _surface_path(surface_name, root)
        if not path.exists():
            continue
        triangles = _read_binary_stl_triangles_cached(str(path))
        points = triangles.reshape(-1, 3)
        mins.append(points.min(axis=0))
        maxs.append(points.max(axis=0))

    if not mins:
        raise FileNotFoundError(
            f"No MIDA brain reference surfaces found under {root / MIDA_SURFACE_DIRNAME}"
        )

    min_corner = np.min(np.stack(mins), axis=0)
    max_corner = np.max(np.stack(maxs), axis=0)
    center = 0.5 * (min_corner + max_corner)
    center.setflags(write=False)
    return center


def mida_reference_center_raw(mida_root: str | Path | None = None) -> np.ndarray:
    """Return the raw MIDA brain-centered reference point used for alignment."""
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return _mida_reference_center_raw_cached(str(root)).copy()


def transform_mida_to_guti(
    points_raw_mm: np.ndarray,
    *,
    mida_root: str | Path | None = None,
) -> np.ndarray:
    """Map raw MIDA STL coordinates into GUTI millimeter coordinates."""
    points = np.asarray(points_raw_mm, dtype=np.float64)
    raw_center = mida_reference_center_raw(mida_root)
    delta = points - raw_center

    transformed = np.empty_like(delta, dtype=np.float64)
    transformed[..., 0] = GUTI_HEAD_CENTER_MM[0] + delta[..., 0]
    transformed[..., 1] = GUTI_HEAD_CENTER_MM[1] + delta[..., 2]
    transformed[..., 2] = GUTI_HEAD_CENTER_MM[2] + delta[..., 1]
    return transformed


def _region_threshold_raw_y(
    region: ScalpRegion,
    *,
    mida_root: str | Path | None = None,
) -> float | None:
    if region == "full":
        return None
    if region == "superior":
        return float(mida_reference_center_raw(mida_root)[1])
    if region == "cranial":
        skull_path = _surface_path(MIDA_SKULL_SURFACE, mida_root)
        if not skull_path.exists():
            return float(mida_reference_center_raw(mida_root)[1])
        skull = _read_binary_stl_triangles_cached(str(skull_path))
        return float(skull.reshape(-1, 3)[:, 1].min())
    raise ValueError("region must be 'superior', 'cranial', or 'full'")


def mida_scalp_triangles(
    *,
    mida_root: str | Path | None = None,
    region: ScalpRegion = "superior",
    coordinate_frame: CoordinateFrame = "guti",
) -> np.ndarray:
    """Return MIDA outer-skin triangles filtered to a scalp/head region."""
    raw = load_mida_surface_triangles(
        MIDA_SCALP_SURFACE, mida_root=mida_root, coordinate_frame="raw"
    )
    threshold = _region_threshold_raw_y(region, mida_root=mida_root)
    if threshold is None:
        filtered = raw
    else:
        centroids = raw.mean(axis=1)
        filtered = raw[centroids[:, 1] >= threshold]

    if filtered.size == 0:
        raise ValueError(f"No MIDA scalp triangles remain for region={region!r}")
    if coordinate_frame == "raw":
        return filtered.copy()
    if coordinate_frame != "guti":
        raise ValueError("coordinate_frame must be 'raw' or 'guti'")
    return transform_mida_to_guti(filtered, mida_root=mida_root)


def _triangle_areas(triangles: np.ndarray) -> np.ndarray:
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)


def _valid_surface_triangles(triangles: np.ndarray) -> np.ndarray:
    areas = _triangle_areas(triangles)
    valid = areas > 0.0
    triangles = triangles[valid]
    if triangles.size == 0:
        raise ValueError("MIDA scalp mesh has no nonzero-area triangles")
    return triangles


def _area_weighted_surface_points(triangles: np.ndarray, n_points: int) -> np.ndarray:
    triangles = _valid_surface_triangles(triangles)
    areas = _triangle_areas(triangles)

    cumulative = np.cumsum(areas)
    # Use a low-discrepancy sequence over cumulative triangle area instead of
    # walking STL order at fixed intervals. This keeps the sample deterministic
    # while avoiding visible bands when triangles are stored in spatial blocks.
    k = np.arange(n_points, dtype=np.float64) + 0.5
    targets = np.mod(k * 0.6180339887498949, 1.0) * cumulative[-1]
    triangle_indices = np.searchsorted(cumulative, targets, side="left")
    triangle_indices = np.clip(triangle_indices, 0, len(triangles) - 1)
    selected = triangles[triangle_indices]

    # Deterministic low-discrepancy barycentric samples inside each chosen
    # triangle. The constants are irrational fractional steps.
    k = np.arange(1, n_points + 1, dtype=np.float64)
    u = np.mod(k * 0.7548776662466927, 1.0)
    v = np.mod(k * 0.5698402909980532, 1.0)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    return (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    )


def _hemisphere_fibonacci_directions(n_points: int) -> np.ndarray:
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n_points, dtype=np.float64)
    z = (indices + 0.5) / n_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * indices
    return np.column_stack((radial * np.cos(phi), radial * np.sin(phi), z))


def _projected_fibonacci_positions(
    candidates: np.ndarray,
    n_sensors: int,
) -> np.ndarray:
    candidate_vectors = candidates - GUTI_HEAD_CENTER_MM
    norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > 0.0
    candidates = candidates[valid]
    candidate_dirs = candidate_vectors[valid] / norms[valid]
    directions = _hemisphere_fibonacci_directions(n_sensors)

    selected = np.empty((n_sensors, 3), dtype=np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    for index, direction in enumerate(directions):
        scores = candidate_dirs @ direction
        scores[used] = -np.inf
        candidate_index = int(np.argmax(scores))
        selected[index] = candidates[candidate_index]
        used[candidate_index] = True
    return selected


def _max_distance_positions(
    candidates: np.ndarray,
    n_sensors: int,
) -> np.ndarray:
    selected = np.empty((n_sensors, 3), dtype=np.float64)
    selected_indices = np.empty(n_sensors, dtype=np.int64)

    centroid = candidates.mean(axis=0)
    first_index = int(np.argmin(np.sum((candidates - centroid) ** 2, axis=1)))
    min_distance_sq = np.full(len(candidates), np.inf, dtype=np.float64)

    for index in range(n_sensors):
        if index == 0:
            candidate_index = first_index
        else:
            candidate_index = int(np.argmax(min_distance_sq))

        selected[index] = candidates[candidate_index]
        selected_indices[index] = candidate_index
        distances_sq = np.sum((candidates - selected[index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distances_sq)
        min_distance_sq[selected_indices[: index + 1]] = -np.inf

    return selected


def _offset_positions_outward(positions: np.ndarray, offset: float) -> np.ndarray:
    if offset == 0.0:
        return positions

    radial = positions - GUTI_HEAD_CENTER_MM
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot offset MIDA sensor at the head center")
    return positions + float(offset) * radial / norms


def sample_mida_scalp_positions(
    n_sensors: int,
    *,
    offset: float = 0.0,
    start_n: int = 0,
    end_n: int | None = None,
    mida_root: str | Path | None = None,
    region: ScalpRegion = "superior",
    method: ScalpSamplingMethod = "projected_fibonacci",
    candidate_count: int | None = None,
) -> np.ndarray:
    """Sample deterministic sensor positions on the MIDA scalp.

    The default ``"superior"`` region keeps the upper cranial scalp in GUTI's
    positive-z half-space, preserving compatibility with the existing source
    grids while replacing the artificial spherical cap with the MIDA skin mesh.
    ``method="projected_fibonacci"`` projects a spherical Fibonacci cap onto
    a deterministic area-weighted candidate set on the MIDA scalp.
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    if start_n < 0:
        raise ValueError("start_n must be non-negative")
    if end_n is not None and end_n < start_n:
        raise ValueError("end_n must be greater than or equal to start_n")

    triangles = mida_scalp_triangles(
        mida_root=mida_root,
        region=region,
        coordinate_frame="guti",
    )
    triangles = _valid_surface_triangles(triangles)

    if method == "area":
        positions = _area_weighted_surface_points(triangles, n_sensors)
    elif method in {"projected_fibonacci", "max_distance"}:
        if candidate_count is None:
            candidate_count = max(4096, n_sensors * 96)
        if candidate_count < n_sensors:
            raise ValueError("candidate_count must be at least n_sensors")
        candidates = _area_weighted_surface_points(triangles, candidate_count)
        if method == "projected_fibonacci":
            positions = _projected_fibonacci_positions(candidates, n_sensors)
        else:
            positions = _max_distance_positions(candidates, n_sensors)
    else:
        raise ValueError("method must be 'area', 'projected_fibonacci', or 'max_distance'")

    return _offset_positions_outward(positions, offset)[start_n:end_n]


def _split_counts_evenly(total: int, n_groups: int) -> tuple[int, ...]:
    base = total // n_groups
    remainder = total % n_groups
    return tuple(base + (1 if index < remainder else 0) for index in range(n_groups))


def sample_mida_us_acoustic_window_positions_by_window(
    n_sensors: int,
    *,
    offset: float = 0.0,
    mida_root: str | Path | None = None,
    candidate_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample ultrasound receivers only on temporal/occipital acoustic windows.

    Receivers are split evenly across left/right temporal and left/right
    occipital apertures. Within each aperture, a greedy max-distance sampler
    gives an even local distribution over deterministic MIDA scalp candidates.
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")

    windows = mida_us_acoustic_windows(mida_root=mida_root)
    counts = _split_counts_evenly(n_sensors, len(windows))
    if candidate_count is None:
        candidate_count = max(16384, n_sensors * 256)
    if candidate_count < n_sensors:
        raise ValueError("candidate_count must be at least n_sensors")

    triangles = mida_scalp_triangles(
        mida_root=mida_root,
        region="full",
        coordinate_frame="guti",
    )
    candidates = _area_weighted_surface_points(triangles, candidate_count)
    centers = np.array([window.center_mm for window in windows], dtype=np.float64)
    distances = np.linalg.norm(candidates[:, None, :] - centers[None, :, :], axis=2)
    nearest_window = np.argmin(distances, axis=1)

    positions_by_window: dict[str, np.ndarray] = {}
    for window_index, (window, count) in enumerate(zip(windows, counts)):
        if count == 0:
            positions_by_window[window.name] = np.empty((0, 3), dtype=np.float64)
            continue

        mask = (
            (distances[:, window_index] <= window.radius_mm)
            & (nearest_window == window_index)
        )
        aperture_candidates = candidates[mask]
        if len(aperture_candidates) < count:
            fallback_mask = distances[:, window_index] <= window.radius_mm
            aperture_candidates = candidates[fallback_mask]
        if len(aperture_candidates) < count:
            raise ValueError(
                f"Only {len(aperture_candidates)} MIDA scalp candidates found for "
                f"{window.name}; need {count}. Increase candidate_count or aperture radius."
            )

        selected = _max_distance_positions(aperture_candidates, count)
        positions_by_window[window.name] = _offset_positions_outward(selected, offset)

    return positions_by_window


def sample_mida_us_acoustic_window_positions(
    n_sensors: int,
    *,
    offset: float = 0.0,
    mida_root: str | Path | None = None,
    candidate_count: int | None = None,
) -> np.ndarray:
    """Sample ultrasound receivers on the four MIDA acoustic windows."""
    positions_by_window = sample_mida_us_acoustic_window_positions_by_window(
        n_sensors,
        offset=offset,
        mida_root=mida_root,
        candidate_count=candidate_count,
    )
    return np.vstack(list(positions_by_window.values()))


@lru_cache(maxsize=4)
def _mida_label_table_cached(mida_root_str: str) -> dict[int, str]:
    path = _voxel_path(MIDA_LABEL_FILE, mida_root_str)
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5 and parts[0].isdigit():
                labels[int(parts[0])] = parts[4].replace("/", "_")
    return labels


def mida_label_table(mida_root: str | Path | None = None) -> dict[int, str]:
    """Return ``{label_id: tissue_name}`` from the MIDA voxel label table."""
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return dict(_mida_label_table_cached(str(root)))


@lru_cache(maxsize=4)
def _mida_tissue_properties_cached(mida_root_str: str) -> dict:
    path = _properties_path(mida_root_str)
    return json.loads(path.read_text(encoding="utf-8"))


def _property_key_candidates(tissue_name: str) -> tuple[str, ...]:
    normalized = tissue_name.strip()
    return (
        normalized,
        normalized.replace("/", "_"),
        normalized.replace("/", " "),
        normalized.replace("_", " "),
    )


def mida_tissue_properties(
    tissue_name: str,
    *,
    mida_root: str | Path | None = None,
) -> dict:
    """Return the MIDA property record for a tissue or surface label."""
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    data = _mida_tissue_properties_cached(str(root))
    for key in _property_key_candidates(tissue_name):
        if key in data:
            return data[key]

    wanted = {key.lower() for key in _property_key_candidates(tissue_name)}
    for record in data.values():
        names = [record.get("name", "")]
        names.extend(record.get("alternativeNames", []))
        if any(str(name).lower() in wanted for name in names):
            return record
    raise KeyError(f"No MIDA tissue-property record found for {tissue_name!r}")


def mida_lf_conductivity_s_per_m(
    tissue_name: str,
    *,
    mida_root: str | Path | None = None,
) -> float:
    """Low-frequency conductivity from the MIDA/IT'IS properties database."""
    props = mida_tissue_properties(tissue_name, mida_root=mida_root)["properties"]
    value = props.get("dielectric", {}).get("lfConductivity")
    if value is None:
        raise KeyError(f"No low-frequency conductivity for {tissue_name!r}")
    return float(value)


def mida_acoustic_speed_m_s(
    tissue_name: str,
    *,
    mida_root: str | Path | None = None,
) -> float:
    props = mida_tissue_properties(tissue_name, mida_root=mida_root)["properties"]
    value = props.get("acoustic", {}).get("speedOfSound")
    if value is None:
        raise KeyError(f"No acoustic speed for {tissue_name!r}")
    return float(value)


def mida_acoustic_attenuation(
    tissue_name: str,
    *,
    mida_root: str | Path | None = None,
) -> dict:
    props = mida_tissue_properties(tissue_name, mida_root=mida_root)["properties"]
    value = props.get("acoustic", {}).get("attenuation")
    if value is None:
        raise KeyError(f"No acoustic attenuation for {tissue_name!r}")
    return dict(value)


def _weighted_average_property(
    tissue_names: tuple[str, ...],
    getter,
    *,
    mida_root: str | Path | None = None,
) -> float:
    values = [getter(name, mida_root=mida_root) for name in tissue_names]
    return float(np.mean(values))


def mida_brain_lf_conductivity_s_per_m(
    *,
    mida_root: str | Path | None = None,
) -> float:
    return _weighted_average_property(
        ("Brain Gray Matter", "Brain White Matter"),
        mida_lf_conductivity_s_per_m,
        mida_root=mida_root,
    )


def mida_brain_acoustic_speed_m_s(
    *,
    mida_root: str | Path | None = None,
) -> float:
    return _weighted_average_property(
        ("Brain Gray Matter", "Brain White Matter"),
        mida_acoustic_speed_m_s,
        mida_root=mida_root,
    )


@lru_cache(maxsize=4)
def _mida_nifti_header_cached(mida_root_str: str) -> tuple[tuple[int, int, int], int, np.ndarray]:
    path = _voxel_path(MIDA_NIFTI_FILE, mida_root_str)
    with path.open("rb") as f:
        header = f.read(352)
    sizeof_hdr = int(np.frombuffer(header[0:4], dtype="<i4", count=1)[0])
    if sizeof_hdr != 348:
        raise ValueError(f"{path} does not look like a NIfTI-1 file")

    dims = tuple(int(x) for x in np.frombuffer(header[40:56], dtype="<i2", count=8)[1:4])
    datatype = int(np.frombuffer(header[70:72], dtype="<i2", count=1)[0])
    bitpix = int(np.frombuffer(header[72:74], dtype="<i2", count=1)[0])
    if datatype != 512 or bitpix != 16:
        raise ValueError(f"Expected uint16 MIDA NIfTI labels, got datatype={datatype}, bitpix={bitpix}")
    offset = int(float(np.frombuffer(header[108:112], dtype="<f4", count=1)[0]))
    affine = np.array(
        [
            np.frombuffer(header[280:296], dtype="<f4", count=4),
            np.frombuffer(header[296:312], dtype="<f4", count=4),
            np.frombuffer(header[312:328], dtype="<f4", count=4),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    return dims, offset, affine


def _mida_nifti_header(
    mida_root: str | Path | None = None,
) -> tuple[tuple[int, int, int], int, np.ndarray]:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    return _mida_nifti_header_cached(str(root))


def _mida_nifti_labels(mida_root: str | Path | None = None) -> np.memmap:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    dims, offset, _ = _mida_nifti_header(root)
    return np.memmap(
        _voxel_path(MIDA_NIFTI_FILE, root),
        dtype=np.uint16,
        mode="r",
        offset=offset,
        shape=dims,
        order="F",
    )


def mida_brain_source_label_ids(
    tissue_names: set[str] | None = None,
    *,
    mida_root: str | Path | None = None,
) -> tuple[int, ...]:
    """Return voxel label ids used as brain source tissue."""
    if tissue_names is None:
        tissue_names = set(MIDA_BRAIN_SOURCE_TISSUES)
    labels = mida_label_table(mida_root)
    return tuple(sorted(label for label, name in labels.items() if name in tissue_names))


def _voxel_indices_to_raw_world(
    indices: np.ndarray,
    *,
    mida_root: str | Path | None = None,
) -> np.ndarray:
    _, _, affine = _mida_nifti_header(mida_root)
    homogeneous = np.column_stack([indices, np.ones(len(indices), dtype=np.float64)])
    return (homogeneous @ affine.T)[:, :3]


@lru_cache(maxsize=16)
def _mida_brain_center_raw_from_voxels_cached(
    mida_root_str: str,
    label_ids: tuple[int, ...],
) -> np.ndarray:
    labels = _mida_nifti_labels(mida_root_str)
    coords_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    coords_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    mask_values = np.array(label_ids, dtype=labels.dtype)

    # Scan in z slabs to avoid materializing every brain voxel coordinate.
    for z0 in range(0, labels.shape[2], 16):
        slab = labels[:, :, z0 : z0 + 16]
        mask = np.isin(slab, mask_values)
        if not np.any(mask):
            continue
        local_idx = np.argwhere(mask)
        local_idx[:, 2] += z0
        points = _voxel_indices_to_raw_world(local_idx, mida_root=mida_root_str)
        coords_min = np.minimum(coords_min, points.min(axis=0))
        coords_max = np.maximum(coords_max, points.max(axis=0))

    if not np.all(np.isfinite(coords_min)):
        raise ValueError("No MIDA brain-source voxels found")
    center = 0.5 * (coords_min + coords_max)
    center.setflags(write=False)
    return center


def mida_brain_center_raw_from_voxels(
    label_ids: tuple[int, ...] | None = None,
    *,
    mida_root: str | Path | None = None,
) -> np.ndarray:
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    if label_ids is None:
        label_ids = mida_brain_source_label_ids(mida_root=root)
    return _mida_brain_center_raw_from_voxels_cached(str(root), tuple(label_ids)).copy()


def transform_mida_voxel_world_to_guti(
    points_raw_mm: np.ndarray,
    *,
    center_raw_mm: np.ndarray | None = None,
    mida_root: str | Path | None = None,
) -> np.ndarray:
    """Map NIfTI world coordinates into GUTI coordinates."""
    points = np.asarray(points_raw_mm, dtype=np.float64)
    if center_raw_mm is None:
        center_raw_mm = mida_brain_center_raw_from_voxels(mida_root=mida_root)
    delta = points - np.asarray(center_raw_mm, dtype=np.float64)
    transformed = np.empty_like(delta, dtype=np.float64)
    transformed[..., 0] = GUTI_HEAD_CENTER_MM[0] + delta[..., 0]
    transformed[..., 1] = GUTI_HEAD_CENTER_MM[1] + delta[..., 2]
    transformed[..., 2] = GUTI_HEAD_CENTER_MM[2] + delta[..., 1]
    return transformed


def get_mida_grid_positions(
    grid_spacing_mm: float = 5.0,
    *,
    label_ids: tuple[int, ...] | None = None,
    mida_root: str | Path | None = None,
) -> np.ndarray:
    """Generate a source grid from MIDA brain tissue labels."""
    if grid_spacing_mm <= 0:
        raise ValueError("grid_spacing_mm must be positive")
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    labels = _mida_nifti_labels(root)
    if label_ids is None:
        label_ids = mida_brain_source_label_ids(mida_root=root)

    _, _, affine = _mida_nifti_header(root)
    voxel_spacing = float(np.linalg.norm(affine[:3, 0]))
    stride = max(1, int(round(grid_spacing_mm / voxel_spacing)))
    sampled = labels[::stride, ::stride, ::stride]
    mask = np.isin(sampled, np.array(label_ids, dtype=labels.dtype))
    indices = np.argwhere(mask) * stride
    if indices.size == 0:
        raise ValueError("No MIDA grid points selected; try a smaller grid spacing")

    raw_points = _voxel_indices_to_raw_world(indices, mida_root=root)
    center = mida_brain_center_raw_from_voxels(label_ids=label_ids, mida_root=root)
    return transform_mida_voxel_world_to_guti(raw_points, center_raw_mm=center)


@lru_cache(maxsize=16)
def _mida_label_voxel_count_cached(mida_root_str: str, label_ids: tuple[int, ...]) -> int:
    labels = _mida_nifti_labels(mida_root_str)
    mask_values = np.array(label_ids, dtype=labels.dtype)
    count = 0
    for z0 in range(0, labels.shape[2], 16):
        count += int(np.count_nonzero(np.isin(labels[:, :, z0 : z0 + 16], mask_values)))
    return count


def mida_brain_volume_mm3(
    label_ids: tuple[int, ...] | None = None,
    *,
    mida_root: str | Path | None = None,
) -> float:
    """Approximate MIDA brain-source volume represented by selected labels."""
    root = Path(mida_root) if mida_root is not None else MIDA_ROOT
    if label_ids is None:
        label_ids = mida_brain_source_label_ids(mida_root=root)
    _, _, affine = _mida_nifti_header(root)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    return voxel_volume * _mida_label_voxel_count_cached(str(root), tuple(label_ids))
