"""
Time-domain fNIRS utility functions.

Implements the time-domain Green's function for the diffusion equation
and sensitivity computation via the adjoint formulation.
"""

import numpy as np
import torch
from typing import Tuple
from tqdm.auto import tqdm


def greens_function_td(
    d: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
) -> torch.Tensor:
    """
    Time-domain Green's function for infinite homogeneous medium.

    G(r,t) = c / (4πDct)^(3/2) * exp(-μₐct - r²/(4Dct))

    Parameters
    ----------
    d : torch.Tensor
        Distances from source to field points. Shape (n_pairs, n_points) or similar.
    t : float
        Time in the same units as c (e.g., if c is in mm/ns, t should be in ns).
    D : float
        Diffusion coefficient = 1/(3(μₐ + μₛ')) in mm.
    mu_a : float
        Absorption coefficient in mm⁻¹.
    c : float
        Speed of light in tissue in mm/ns.

    Returns
    -------
    G : torch.Tensor
        Green's function values, same shape as d.
    """
    # Avoid division by zero at t=0
    t = max(t, 1e-12)

    prefactor = c / (4 * np.pi * D * c * t) ** 1.5
    exponent = -mu_a * c * t - d**2 / (4 * D * c * t)

    return prefactor * torch.exp(exponent)


def td_sensitivity(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    detector_pos: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
    n_integration_points: int = 20,
) -> torch.Tensor:
    """
    Calculate the time-domain sensitivity function using PyTorch.

    The sensitivity is computed via the adjoint formulation:
    J(t) = -∫₀ᵗ G(source→pos, t') · G(pos→detector, t-t') dt'

    This represents the sensitivity of the detected signal at time t
    to absorption changes at position pos.

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate sensitivity for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
    t : float
        Time gate in the same units as c.
    D : float
        Diffusion coefficient in mm.
    mu_a : float
        Absorption coefficient in mm⁻¹.
    c : float
        Speed of light in tissue in mm/ns.
    n_integration_points : int
        Number of points for trapezoidal integration over t'.

    Returns
    -------
    sensitivity : torch.Tensor
        The sensitivity function. Shape (n_pairs, n_points).
    """
    # Compute distances from sources and detectors to all grid points
    # source_pos: (n_pairs, 3), pos: (n_points, 3)
    # d_source: (n_pairs, n_points)
    d_source = torch.norm(source_pos[:, None, :] - pos[None, :, :], dim=2)
    d_detector = torch.norm(detector_pos[:, None, :] - pos[None, :, :], dim=2)

    # Integration over t' from small epsilon to t-epsilon
    # Avoid endpoints where one of the Green's functions would have t=0
    eps = t * 0.01
    t_prime = torch.linspace(eps, t - eps, n_integration_points, device=pos.device)
    dt = t_prime[1] - t_prime[0] if n_integration_points > 1 else t

    # Trapezoidal integration
    sensitivity = torch.zeros_like(d_source)
    for tp in t_prime:
        tp_val = tp.item()
        G_sv = greens_function_td(d_source, tp_val, D, mu_a, c)
        G_vd = greens_function_td(d_detector, t - tp_val, D, mu_a, c)
        sensitivity += G_sv * G_vd * dt

    return -sensitivity


def td_sensitivity_batched(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    detector_pos: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
    n_integration_points: int = 20,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Memory-efficient batched version of td_sensitivity.

    Processes source-detector pairs in batches to avoid GPU OOM errors.

    Parameters
    ----------
    pos : torch.Tensor
        The positions to calculate sensitivity for. Shape (n_points, 3).
    source_pos : torch.Tensor
        The source positions. Shape (n_pairs, 3).
    detector_pos : torch.Tensor
        The detector positions. Shape (n_pairs, 3).
    t : float
        Time gate.
    D : float
        Diffusion coefficient.
    mu_a : float
        Absorption coefficient.
    c : float
        Speed of light in tissue.
    n_integration_points : int
        Number of points for integration.
    batch_size : int
        Number of pairs to process at once.

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
    for i in tqdm(range(0, n_pairs, batch_size), desc=f"Computing TD sensitivity (t={t:.2f})"):
        end_idx = min(i + batch_size, n_pairs)
        batch_sources = source_pos[i:end_idx]
        batch_detectors = detector_pos[i:end_idx]

        batch_sensitivity = td_sensitivity(
            pos, batch_sources, batch_detectors, t, D, mu_a, c, n_integration_points
        )

        sensitivity[i:end_idx] = batch_sensitivity

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sensitivity


def get_valid_source_detector_pairs(
    sensor_positions_mm: torch.Tensor, max_dist: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get all valid source-detector pairs that satisfy distance criteria.

    Parameters
    ----------
    sensor_positions_mm : torch.Tensor
        Sensor positions with shape (n_sensors, 3).
    max_dist : float
        Maximum allowed distance between source and detector.

    Returns
    -------
    sources : torch.Tensor
        Source positions for valid pairs. Shape (n_pairs, 3).
    detectors : torch.Tensor
        Detector positions for valid pairs. Shape (n_pairs, 3).
    """
    # Calculate pairwise distances
    d_mat = torch.norm(
        sensor_positions_mm[:, None, :] - sensor_positions_mm[None, :, :], dim=2
    )

    # Find valid pairs (different sensors and within max distance)
    mask = (d_mat <= max_dist) & (d_mat > 0)
    src_idx, det_idx = torch.nonzero(mask, as_tuple=True)

    sources = sensor_positions_mm[src_idx]
    detectors = sensor_positions_mm[det_idx]

    return sources, detectors
