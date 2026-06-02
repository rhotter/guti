"""MEG Johnson-noise covariance from reciprocal field overlaps.

This module implements low-frequency reciprocal simulators for body/tissue
magnetic Johnson noise.  Both use the fluctuation-dissipation structure

    S_ij = 4 k_B T integral sigma(r) q_i(r) . q_j(r) dV

where q_i = E_i / (-i omega) is the reciprocal electric field per unit angular
frequency and per unit magnetic moment of detector channel i.

The more rigorous ``finite_volume`` path approximates

    q_i = A_i - grad psi_i

and solves the scalar-potential correction that enforces charge conservation in
the conductive head.  Detector channels are still point magnetic dipoles /
infinitesimal loops; finite SQUID pickup loops and gradiometers need explicit
loop geometry on top of this.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from guti.core import BRAIN_RADIUS
from guti.modalities.eeg.scalp_resistance import Layer, default_layers
from guti.noise_models import BODY_TEMP_K, K_B


MU0 = 4.0 * math.pi * 1e-7
HEAD_CENTER_MM = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0], dtype=float)

SensorComponents = Literal["xyz", "radial"]
JohnsonSolver = Literal["finite_volume", "vector_potential"]


@dataclass(frozen=True)
class MEGJohnsonMetadata:
    """Configuration summary for a computed MEG Johnson matrix."""

    sensor_components: str
    n_sensors: int
    n_channels: int
    n_voxels: int
    voxel_resolution_mm: float
    voxel_volume_m3: float
    temperature_k: float
    solver: str
    approximation: str
    layers: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class VoxelGrid:
    """Structured voxel-domain data for finite-volume scalar correction."""

    points_m: np.ndarray
    sigma_s_per_m: np.ndarray
    indices_ijk: np.ndarray
    voxel_resolution_m: float
    voxel_volume_m3: float


def voxelized_layered_hemisphere(
    *,
    voxel_resolution_mm: float = 8.0,
    layers: tuple[Layer, ...] | None = None,
    center_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return voxel centers, conductivities, and voxel volume for the GUTI head.

    Voxel centers are returned in meters and conductivities in S/m.  The head is
    the upper hemisphere used throughout GUTI: z >= 0, centered at
    ``(BRAIN_RADIUS, BRAIN_RADIUS, 0)`` in millimeters.
    """

    resolution = float(voxel_resolution_mm)
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError("voxel_resolution_mm must be positive and finite")

    head_layers = layers if layers is not None else default_layers()
    if len(head_layers) == 0:
        raise ValueError("layers must not be empty")

    center = (
        np.asarray(center_mm, dtype=float)
        if center_mm is not None
        else HEAD_CENTER_MM
    )
    if center.shape != (3,):
        raise ValueError("center_mm must have shape (3,)")

    outer_radius_mm = head_layers[-1].outer_radius_mm
    half_step = 0.5 * resolution
    xy = np.arange(
        -outer_radius_mm + half_step,
        outer_radius_mm,
        resolution,
        dtype=float,
    )
    z = np.arange(half_step, outer_radius_mm, resolution, dtype=float)
    x_grid, y_grid, z_grid = np.meshgrid(xy, xy, z, indexing="ij")
    rel_mm = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    )
    radii_mm = np.linalg.norm(rel_mm, axis=1)
    inside = radii_mm <= outer_radius_mm
    rel_mm = rel_mm[inside]
    radii_mm = radii_mm[inside]

    sigma = np.full(radii_mm.shape, np.nan, dtype=float)
    unassigned = np.ones(radii_mm.shape, dtype=bool)
    for layer in head_layers:
        in_layer = unassigned & (radii_mm <= layer.outer_radius_mm + 1e-12)
        sigma[in_layer] = layer.conductivity_s_per_m
        unassigned[in_layer] = False

    if np.any(~np.isfinite(sigma)):
        raise RuntimeError("failed to assign conductivity to every head voxel")

    points_mm = rel_mm + center[None, :]
    voxel_volume_m3 = (resolution * 1e-3) ** 3
    return points_mm * 1e-3, sigma, voxel_volume_m3


def voxelized_layered_hemisphere_grid(
    *,
    voxel_resolution_mm: float = 8.0,
    layers: tuple[Layer, ...] | None = None,
    center_mm: np.ndarray | None = None,
) -> VoxelGrid:
    """Return structured voxel grid data for finite-volume solves."""

    resolution = float(voxel_resolution_mm)
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError("voxel_resolution_mm must be positive and finite")

    head_layers = layers if layers is not None else default_layers()
    center = (
        np.asarray(center_mm, dtype=float)
        if center_mm is not None
        else HEAD_CENTER_MM
    )
    if center.shape != (3,):
        raise ValueError("center_mm must have shape (3,)")

    outer_radius_mm = head_layers[-1].outer_radius_mm
    half_step = 0.5 * resolution
    xy = np.arange(
        -outer_radius_mm + half_step,
        outer_radius_mm,
        resolution,
        dtype=float,
    )
    z = np.arange(half_step, outer_radius_mm, resolution, dtype=float)
    x_grid, y_grid, z_grid = np.meshgrid(xy, xy, z, indexing="ij")
    rel_mm = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    )
    indices = np.column_stack(
        [
            np.indices(x_grid.shape, dtype=np.int32)[axis].ravel()
            for axis in range(3)
        ]
    )
    radii_mm = np.linalg.norm(rel_mm, axis=1)
    inside = radii_mm <= outer_radius_mm
    rel_mm = rel_mm[inside]
    indices = indices[inside]
    radii_mm = radii_mm[inside]

    sigma = np.full(radii_mm.shape, np.nan, dtype=float)
    unassigned = np.ones(radii_mm.shape, dtype=bool)
    for layer in head_layers:
        in_layer = unassigned & (radii_mm <= layer.outer_radius_mm + 1e-12)
        sigma[in_layer] = layer.conductivity_s_per_m
        unassigned[in_layer] = False

    if np.any(~np.isfinite(sigma)):
        raise RuntimeError("failed to assign conductivity to every head voxel")

    points_mm = rel_mm + center[None, :]
    voxel_resolution_m = resolution * 1e-3
    return VoxelGrid(
        points_m=points_mm * 1e-3,
        sigma_s_per_m=sigma,
        indices_ijk=indices,
        voxel_resolution_m=voxel_resolution_m,
        voxel_volume_m3=voxel_resolution_m**3,
    )


def sensor_channel_orientations(
    sensor_positions_mm: np.ndarray,
    *,
    sensor_components: SensorComponents = "xyz",
    center_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel sensor indices and unit orientations.

    ``xyz`` produces rows grouped like the MEG forward model:
    ``sensor0-x, sensor0-y, sensor0-z, sensor1-x, ...``.  ``radial`` produces
    one outward radial component per sensor.
    """

    positions = np.asarray(sensor_positions_mm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("sensor_positions_mm must have shape (n_sensors, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("sensor_positions_mm must contain finite entries")

    n_sensors = positions.shape[0]
    if sensor_components == "xyz":
        sensor_indices = np.repeat(np.arange(n_sensors), 3)
        orientations = np.tile(np.eye(3, dtype=float), (n_sensors, 1))
        return sensor_indices, orientations

    if sensor_components == "radial":
        center = (
            np.asarray(center_mm, dtype=float)
            if center_mm is not None
            else HEAD_CENTER_MM
        )
        if center.shape != (3,):
            raise ValueError("center_mm must have shape (3,)")
        radial = positions - center[None, :]
        norms = np.linalg.norm(radial, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("sensor positions must not coincide with head center")
        return np.arange(n_sensors), radial / norms[:, None]

    raise ValueError(f"Unsupported sensor_components {sensor_components!r}")


def _reciprocal_feature_block(
    points_m: np.ndarray,
    sensor_positions_m: np.ndarray,
    channel_sensor_indices: np.ndarray,
    channel_orientations: np.ndarray,
    sqrt_sigma_dv: np.ndarray,
    *,
    start: int,
    stop: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Return weighted reciprocal features for a channel block."""

    sensors = sensor_positions_m[channel_sensor_indices[start:stop]]
    orientations = channel_orientations[start:stop]
    displacement = points_m[None, :, :] - sensors[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance <= 0.0):
        raise ValueError("a sensor coincides with a head voxel")

    kernel = (
        MU0
        / (4.0 * math.pi)
        * np.cross(orientations[:, None, :], displacement)
        / (distance[:, :, None] ** 3)
    )
    kernel *= sqrt_sigma_dv[None, :, None]
    return kernel.reshape(stop - start, -1).astype(dtype, copy=False)


def _magnetic_dipole_vector_potential(
    evaluation_points_m: np.ndarray,
    sensor_positions_m: np.ndarray,
    channel_sensor_indices: np.ndarray,
    channel_orientations: np.ndarray,
) -> np.ndarray:
    """Return A at evaluation points for all point-dipole detector channels."""

    sensors = sensor_positions_m[channel_sensor_indices]
    orientations = channel_orientations
    displacement = evaluation_points_m[None, :, :] - sensors[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance <= 0.0):
        raise ValueError("a sensor coincides with an evaluation point")
    return (
        MU0
        / (4.0 * math.pi)
        * np.cross(orientations[:, None, :], displacement)
        / (distance[:, :, None] ** 3)
    )


def _finite_volume_edges(grid: VoxelGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent voxel edge pairs and their coordinate axes."""

    index_to_row = {
        tuple(index): row for row, index in enumerate(grid.indices_ijk.tolist())
    }
    starts: list[int] = []
    stops: list[int] = []
    axes: list[int] = []
    for row, index in enumerate(grid.indices_ijk):
        for axis in range(3):
            neighbor = index.copy()
            neighbor[axis] += 1
            neighbor_row = index_to_row.get(tuple(neighbor))
            if neighbor_row is None:
                continue
            starts.append(row)
            stops.append(neighbor_row)
            axes.append(axis)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(stops, dtype=np.int64),
        np.asarray(axes, dtype=np.int64),
    )


def _assemble_finite_volume_laplacian(
    grid: VoxelGrid,
    edge_start: np.ndarray,
    edge_stop: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return graph conductivity Laplacian and per-edge conductance."""

    sigma_i = grid.sigma_s_per_m[edge_start]
    sigma_j = grid.sigma_s_per_m[edge_stop]
    sigma_face = 2.0 * sigma_i * sigma_j / (sigma_i + sigma_j)
    conductance = sigma_face * grid.voxel_resolution_m
    n = grid.points_m.shape[0]

    rows = np.concatenate([edge_start, edge_stop, edge_start, edge_stop])
    cols = np.concatenate([edge_start, edge_stop, edge_stop, edge_start])
    data = np.concatenate(
        [conductance, conductance, -conductance, -conductance]
    )
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return matrix, conductance


def _finite_volume_corrected_features(
    grid: VoxelGrid,
    sensor_positions_m: np.ndarray,
    channel_sensor_indices: np.ndarray,
    channel_orientations: np.ndarray,
) -> np.ndarray:
    """Return weighted q=A-grad(psi) edge features for all channels."""

    edge_start, edge_stop, edge_axis = _finite_volume_edges(grid)
    if edge_start.size == 0:
        raise ValueError("finite-volume grid has no interior edges")

    laplacian, conductance = _assemble_finite_volume_laplacian(
        grid,
        edge_start,
        edge_stop,
    )
    h = grid.voxel_resolution_m
    face_points = 0.5 * (grid.points_m[edge_start] + grid.points_m[edge_stop])
    vector_potential = _magnetic_dipole_vector_potential(
        face_points,
        sensor_positions_m,
        channel_sensor_indices,
        channel_orientations,
    )
    edge_a = vector_potential[:, np.arange(edge_axis.size), edge_axis]

    rhs = np.zeros((grid.points_m.shape[0], channel_orientations.shape[0]), dtype=float)
    edge_rhs = (conductance * h)[None, :] * edge_a
    np.add.at(rhs, edge_start, -edge_rhs.T)
    np.add.at(rhs, edge_stop, edge_rhs.T)

    # Pin one arbitrary voxel to remove the Neumann constant nullspace.  Only
    # potential differences are used below, so the chosen gauge is immaterial.
    reduced_laplacian = laplacian[1:, 1:].tocsc()
    solver = sparse_linalg.factorized(reduced_laplacian)
    psi = np.zeros_like(rhs)
    psi[1:] = solver(rhs[1:])

    edge_gradient = (psi[edge_stop] - psi[edge_start]).T / h
    corrected_edge_q = edge_a - edge_gradient
    sigma_i = grid.sigma_s_per_m[edge_start]
    sigma_j = grid.sigma_s_per_m[edge_stop]
    sigma_face = 2.0 * sigma_i * sigma_j / (sigma_i + sigma_j)
    edge_weight = np.sqrt(sigma_face * grid.voxel_volume_m3)
    return corrected_edge_q * edge_weight[None, :]


def compute_meg_johnson_noise_spectral_density(
    sensor_positions_mm: np.ndarray,
    *,
    sensor_components: SensorComponents = "xyz",
    voxel_resolution_mm: float = 8.0,
    temperature_k: float = BODY_TEMP_K,
    layers: tuple[Layer, ...] | None = None,
    center_mm: np.ndarray | None = None,
    solver: JohnsonSolver = "finite_volume",
    batch_size: int = 64,
    dtype: np.dtype | type = np.float64,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, MEGJohnsonMetadata]:
    """Return one-sided MEG body-Johnson spectral density in channel units.

    The returned matrix has units of approximately ``T^2 / Hz`` for point field
    channels.  Use ``sensor_components='xyz'`` to match GUTI's three-component
    MEG forward rows, or ``sensor_components='radial'`` for one scalar radial
    channel per sensor.

    ``solver='finite_volume'`` solves the scalar-potential correction on a
    voxel graph and is the default.  ``solver='vector_potential'`` reproduces
    the older shortcut that uses q=A directly.
    """

    if temperature_k <= 0.0 or not math.isfinite(float(temperature_k)):
        raise ValueError("temperature_k must be positive and finite")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    positions_mm = np.asarray(sensor_positions_mm, dtype=float)
    channel_sensor_indices, channel_orientations = sensor_channel_orientations(
        positions_mm,
        sensor_components=sensor_components,
        center_mm=center_mm,
    )
    out_dtype = np.dtype(dtype)
    n_channels = channel_orientations.shape[0]
    sensor_positions_m = (positions_mm * 1e-3).astype(out_dtype)
    channel_orientations = channel_orientations.astype(out_dtype)

    if solver == "vector_potential":
        points_m, sigma, voxel_volume_m3 = voxelized_layered_hemisphere(
            voxel_resolution_mm=voxel_resolution_mm,
            layers=layers,
            center_mm=center_mm,
        )
        sqrt_sigma_dv = np.sqrt(sigma * voxel_volume_m3).astype(out_dtype)
        points_m = points_m.astype(out_dtype)
        n_features = points_m.shape[0] * 3
        features = np.empty((n_channels, n_features), dtype=out_dtype)
        for start in range(0, n_channels, batch_size):
            stop = min(start + batch_size, n_channels)
            features[start:stop] = _reciprocal_feature_block(
                points_m,
                sensor_positions_m,
                channel_sensor_indices,
                channel_orientations,
                sqrt_sigma_dv,
                start=start,
                stop=stop,
                dtype=out_dtype,
            )
        n_voxels = int(points_m.shape[0])
        voxel_volume = float(voxel_volume_m3)
        approximation = (
            "point-dipole reciprocal vector-potential overlap; no scalar "
            "boundary-potential correction"
        )
    elif solver == "finite_volume":
        grid = voxelized_layered_hemisphere_grid(
            voxel_resolution_mm=voxel_resolution_mm,
            layers=layers,
            center_mm=center_mm,
        )
        features = _finite_volume_corrected_features(
            grid,
            sensor_positions_m,
            channel_sensor_indices,
            channel_orientations,
        ).astype(out_dtype, copy=False)
        n_voxels = int(grid.points_m.shape[0])
        voxel_volume = float(grid.voxel_volume_m3)
        approximation = (
            "point-dipole reciprocal vector-potential with finite-volume "
            "scalar-potential correction enforcing div(sigma E)=0"
        )
    else:
        raise ValueError(f"Unsupported solver {solver!r}")

    spectral_density = 4.0 * K_B * float(temperature_k) * (features @ features.T)
    spectral_density = 0.5 * (spectral_density + spectral_density.T)

    if not return_metadata:
        return spectral_density

    head_layers = layers if layers is not None else default_layers()
    metadata = MEGJohnsonMetadata(
        sensor_components=sensor_components,
        n_sensors=int(positions_mm.shape[0]),
        n_channels=int(n_channels),
        n_voxels=n_voxels,
        voxel_resolution_mm=float(voxel_resolution_mm),
        voxel_volume_m3=voxel_volume,
        temperature_k=float(temperature_k),
        solver=solver,
        approximation=approximation,
        layers=tuple(asdict(layer) for layer in head_layers),
    )
    return spectral_density, metadata


def regularize_covariance_diagonal(
    covariance: np.ndarray,
    *,
    relative_jitter: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Return a numerically positive-definite covariance and added jitter.

    The reciprocal/FDT construction is positive semidefinite analytically, but
    coarse voxel grids and dense correlation matrices can leave tiny negative
    eigenvalues after symmetrization.  The returned jitter is the diagonal term
    added to make whitening stable.
    """

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if relative_jitter < 0.0 or not math.isfinite(relative_jitter):
        raise ValueError("relative_jitter must be finite and nonnegative")
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        raise ValueError("covariance must have positive finite diagonal entries")

    scale = float(np.mean(diag))
    min_eigenvalue = float(np.linalg.eigvalsh(cov)[0])
    jitter = max(relative_jitter * scale, -min_eigenvalue + relative_jitter * scale)
    if jitter <= 0.0:
        return cov, 0.0
    return cov + jitter * np.eye(cov.shape[0], dtype=cov.dtype), float(jitter)


def compute_meg_johnson_noise_covariance(
    sensor_positions_mm: np.ndarray,
    *,
    noise_std: float | np.ndarray | None = None,
    sensor_components: SensorComponents = "radial",
    voxel_resolution_mm: float = 4.0,
    bandwidth_hz: float = 100.0,
    temperature_k: float = BODY_TEMP_K,
    layers: tuple[Layer, ...] | None = None,
    center_mm: np.ndarray | None = None,
    solver: JohnsonSolver = "finite_volume",
    batch_size: int = 64,
    dtype: np.dtype | type = np.float64,
    relative_jitter: float = 1e-10,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, MEGJohnsonMetadata, dict[str, object]]:
    """Return MEG Johnson output covariance in detector units.

    If ``noise_std`` is provided, the reciprocal Johnson calculation supplies
    the correlation structure and the covariance diagonal is matched to the
    supplied detector-noise standard deviation.  If ``noise_std`` is ``None``,
    the returned covariance uses the absolute body-Johnson scale integrated
    over ``bandwidth_hz``.
    """

    if bandwidth_hz <= 0.0 or not math.isfinite(float(bandwidth_hz)):
        raise ValueError("bandwidth_hz must be positive and finite")

    spectral_density, metadata = compute_meg_johnson_noise_spectral_density(
        sensor_positions_mm,
        sensor_components=sensor_components,
        voxel_resolution_mm=voxel_resolution_mm,
        temperature_k=temperature_k,
        layers=layers,
        center_mm=center_mm,
        solver=solver,
        batch_size=batch_size,
        dtype=dtype,
        return_metadata=True,
    )
    if noise_std is None:
        covariance = spectral_density * float(bandwidth_hz)
        scale_mode = "absolute_body_johnson"
    else:
        covariance = match_covariance_diagonal(spectral_density, noise_std)
        scale_mode = "matched_detector_diagonal"

    covariance, jitter = regularize_covariance_diagonal(
        covariance,
        relative_jitter=relative_jitter,
    )
    diag_spectral = np.diag(spectral_density)
    scale_info = {
        "bandwidth_hz": float(bandwidth_hz),
        "raw_body_noise_median_T_per_sqrtHz": float(np.median(np.sqrt(diag_spectral))),
        "raw_body_noise_mean_T_per_sqrtHz": float(np.mean(np.sqrt(diag_spectral))),
        "regularization_jitter_T2": float(jitter),
        "scale_mode": scale_mode,
    }
    if not return_metadata:
        return covariance
    return covariance, metadata, scale_info


def match_covariance_diagonal(
    covariance: np.ndarray,
    noise_std: float | np.ndarray,
) -> np.ndarray:
    """Return covariance with the same correlation but a chosen diagonal std."""

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    diag = np.diag(cov)
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        raise ValueError("covariance must have positive finite diagonal entries")

    std = np.asarray(noise_std, dtype=float)
    if std.ndim == 0:
        std = np.full(cov.shape[0], float(std))
    elif std.shape != (cov.shape[0],):
        raise ValueError("noise_std must be scalar or match covariance dimension")
    if np.any(std <= 0.0) or not np.all(np.isfinite(std)):
        raise ValueError("noise_std values must be positive and finite")

    corr = cov / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(corr, 1.0)
    matched = corr * np.outer(std, std)
    return 0.5 * (matched + matched.T)
