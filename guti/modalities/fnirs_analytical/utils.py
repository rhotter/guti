import numpy as np
import torch
from typing import Tuple
from tqdm.auto import tqdm


def compute_perpendicular_distance(
    pos: torch.Tensor, source_pos: torch.Tensor, detector_pos: torch.Tensor
) -> torch.Tensor:
    """
    Compute the perpendicular distance from points to the source-detector line using PyTorch.

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate perpendicular distance for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).

    Returns
    -------
    z : torch.Tensor
        Perpendicular distances. Shape (n_pairs, n_points).
    """
    # Distance from source to detector for each pair
    d_source_detector = torch.norm(detector_pos - source_pos, dim=1)  # (n_pairs,)

    # Transform coordinates: translate so source is at origin
    # pos: (n_points, 3), source_pos: (n_pairs, 3) -> pos_rel: (n_pairs, n_points, 3)
    pos_rel = pos[None, :, :] - source_pos[:, None, :]

    # Unit vector from source to detector for each pair
    dir_source_detector = (detector_pos - source_pos) / d_source_detector[
        :, None
    ]  # (n_pairs, 3)

    # Project each relative position onto the source-detector line
    # pos_rel: (n_pairs, n_points, 3), dir_source_detector: (n_pairs, 3) -> d_proj: (n_pairs, n_points)
    d_proj = torch.sum(pos_rel * dir_source_detector[:, None, :], dim=2)

    # Vector from each point to its projection on the line
    vec_perp = pos_rel - d_proj[:, :, None] * dir_source_detector[:, None, :]

    # z is the magnitude of the perpendicular vector
    z = torch.norm(vec_perp, dim=2)  # (n_pairs, n_points)

    return z


def cw_sensitivity(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    detector_pos: torch.Tensor,
    mu_eff: float,
) -> torch.Tensor:
    """
    Calculate the continuous wave sensitivity function using PyTorch on GPU.
    The sensitivity function is with respect to mu_a at pos.

    From eq (14.8) in the textbook Quantitative Biomedical Optics by Bigio and Fantini (p. 427).

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate sensitivity for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
    mu_eff : float
        The effective attenuation coefficient in mm^-1.

    Returns
    -------
    sensitivity : torch.Tensor
        The sensitivity function. Shape (n_pairs, n_points).
    """
    # Calculate distances
    # pos: (n_points, 3), source_pos: (n_pairs, 3) -> d_source_pos: (n_pairs, n_points)
    d_source_pos = torch.norm(source_pos[:, None, :] - pos[None, :, :], dim=2)
    d_detector_pos = torch.norm(detector_pos[:, None, :] - pos[None, :, :], dim=2)

    # Compute perpendicular distance z
    z = compute_perpendicular_distance(pos, source_pos, detector_pos)

    # Calculate sensitivity
    sensitivity = (
        z**2
        * (mu_eff + 1 / d_source_pos)
        * (mu_eff + 1 / d_detector_pos)
        * torch.exp(-mu_eff * (d_source_pos + d_detector_pos))
        / (d_source_pos**2 * d_detector_pos**2)
    )

    return sensitivity


def cw_sensitivity_batched(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    detector_pos: torch.Tensor,
    mu_eff: float,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Memory-efficient batched version of cw_sensitivity that processes pairs in chunks.

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate sensitivity for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
    mu_eff : float
        The effective attenuation coefficient in mm^-1.
    batch_size : int
        Number of pairs to process at once. Reduce this if you still get OOM errors.

    Returns
    -------
    sensitivity : torch.Tensor
        The sensitivity function. Shape (n_pairs, n_points).
    """
    n_pairs = source_pos.shape[0]
    n_points = pos.shape[0]

    # Initialize output tensor
    sensitivity = torch.zeros(
        (n_pairs, n_points), dtype=torch.float32, device=pos.device
    )

    # Process in batches
    for i in tqdm(range(0, n_pairs, batch_size), desc="Computing sensitivities"):
        end_idx = min(i + batch_size, n_pairs)
        batch_sources = source_pos[i:end_idx]
        batch_detectors = detector_pos[i:end_idx]

        # Call the original cw_sensitivity function for this batch
        batch_sensitivity = cw_sensitivity(pos, batch_sources, batch_detectors, mu_eff)

        # Store result
        sensitivity[i:end_idx] = batch_sensitivity

        # Optional: clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sensitivity


def get_valid_source_detector_pairs(
    sensor_positions_mm: torch.Tensor, max_dist: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get unique source-detector pairs that satisfy distance criteria using PyTorch.

    Parameters
    ----------
    sensor_positions_mm : torch.Tensor
        Sensor positions with shape (n_sensors, 3).
    max_dist : float
        Maximum allowed distance between source and detector.

    Returns
    -------
    sources : torch.Tensor
        Source positions for valid unordered pairs. Shape (n_pairs, 3).
    detectors : torch.Tensor
        Detector positions for valid unordered pairs. Shape (n_pairs, 3).
    """
    # Calculate pairwise distances
    d_mat = torch.norm(
        sensor_positions_mm[:, None, :] - sensor_positions_mm[None, :, :], dim=2
    )

    # Find each reciprocal S-D pair once.  CW measurements are reciprocal in
    # this model, so including both (i, j) and (j, i) double-counts channels.
    mask = torch.triu((d_mat <= max_dist) & (d_mat > 0), diagonal=1)
    src_idx, det_idx = torch.nonzero(mask, as_tuple=True)

    sources = sensor_positions_mm[src_idx]
    detectors = sensor_positions_mm[det_idx]

    return sources, detectors
