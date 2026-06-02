"""
Time-domain fNIRS utility functions.

Implements the time-domain Green's function for the diffusion equation
and sensitivity computation via the adjoint formulation.

Semi-infinite medium is modelled with the extrapolated boundary condition
using the method of images (Patterson, Chance & Wilson, Appl. Opt. 1989;
Arridge, Appl. Opt. 1995). A local tangent-plane approximation is used
at each optode on the curved hemisphere surface, so the real source is
placed one transport mean free path beneath the optode along its outward
normal and a negative image source is placed through the extrapolated
boundary along the same normal.
"""

import numpy as np
import torch
from typing import Tuple
from tqdm.auto import tqdm


# Default extrapolation-distance factor A for a tissue/air interface with
# n_tissue = 1.4, n_air = 1.0 (Haskell et al., JOSA A 1994).
# z_b = 2 * A * D.
DEFAULT_A = 2.948


def greens_function_td_infinite(
    d: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
) -> torch.Tensor:
    """
    Time-domain Green's function for an infinite homogeneous medium.

    G(r,t) = c / (4πDct)^(3/2) * exp(-μₐct - r²/(4Dct))
    """
    t = max(t, 1e-12)
    prefactor = c / (4 * np.pi * D * c * t) ** 1.5
    exponent = -mu_a * c * t - d**2 / (4 * D * c * t)
    return prefactor * torch.exp(exponent)


def greens_function_td_semi_infinite(
    optode_pos: torch.Tensor,
    optode_normal: torch.Tensor,
    voxel_pos: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
    mu_s_prime: float,
    A: float = DEFAULT_A,
) -> torch.Tensor:
    """
    Time-domain fluence Green's function for a semi-infinite medium.

    Uses the extrapolated-boundary-condition image-source construction:
      - real isotropic source at depth z0 = 1/μs' inside the tissue
      - image source reflected through the extrapolated boundary at
        distance zb = 2*A*D outside the surface

    Φ(optode → voxel, t) = G_inf(ρ_real, t) − G_inf(ρ_image, t)

    Parameters
    ----------
    optode_pos : (n_pairs, 3) positions of the optode on the surface.
    optode_normal : (n_pairs, 3) unit outward normal at each optode.
    voxel_pos : (n_points, 3) positions at which to evaluate the fluence.
    t, D, mu_a, c, mu_s_prime : float, optical/physical constants in mm, ns.
    A : extrapolation-distance factor for the refractive-index mismatch.

    Returns
    -------
    Φ : (n_pairs, n_points) fluence difference.
    """
    z0 = 1.0 / mu_s_prime
    zb = 2.0 * A * D

    # Real source: one transport MFP inside the tissue along inward normal.
    real_src = optode_pos - z0 * optode_normal  # (n_pairs, 3)
    # Image source: mirror through the extrapolated boundary at zb outside surface.
    image_src = optode_pos + (z0 + 2.0 * zb) * optode_normal  # (n_pairs, 3)

    d_real = torch.norm(real_src[:, None, :] - voxel_pos[None, :, :], dim=2)
    d_image = torch.norm(image_src[:, None, :] - voxel_pos[None, :, :], dim=2)

    G_real = greens_function_td_infinite(d_real, t, D, mu_a, c)
    G_image = greens_function_td_infinite(d_image, t, D, mu_a, c)
    return G_real - G_image


def td_sensitivity(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    source_normal: torch.Tensor,
    detector_pos: torch.Tensor,
    detector_normal: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
    mu_s_prime: float,
    A: float = DEFAULT_A,
    n_integration_points: int = 20,
) -> torch.Tensor:
    """
    Time-domain sensitivity function for a semi-infinite medium.

    J(r, t) = -∫₀ᵗ Φ_semi(s → r, t') · Φ_semi(r → d, t - t') dt'

    Both fluences use the image-source semi-infinite Green's function.

    Parameters
    ----------
    pos : (n_points, 3) voxel positions at which to evaluate sensitivity.
    source_pos, detector_pos : (n_pairs, 3) optode positions.
    source_normal, detector_normal : (n_pairs, 3) unit outward normals.
    t : float, time gate.
    D, mu_a, c, mu_s_prime : optical/physical constants in mm, ns.
    A : extrapolation-distance factor.
    n_integration_points : trapezoidal integration points over t'.

    Returns
    -------
    sensitivity : (n_pairs, n_points)
    """
    eps = t * 0.01
    t_prime = torch.linspace(eps, t - eps, n_integration_points, device=pos.device)
    dt = t_prime[1] - t_prime[0] if n_integration_points > 1 else t

    n_pairs = source_pos.shape[0]
    n_points = pos.shape[0]
    sensitivity = torch.zeros((n_pairs, n_points), dtype=pos.dtype, device=pos.device)

    for tp in t_prime:
        tp_val = tp.item()
        Phi_s = greens_function_td_semi_infinite(
            source_pos, source_normal, pos, tp_val, D, mu_a, c, mu_s_prime, A
        )
        Phi_d = greens_function_td_semi_infinite(
            detector_pos, detector_normal, pos, t - tp_val, D, mu_a, c, mu_s_prime, A
        )
        sensitivity += Phi_s * Phi_d * dt

    return -sensitivity


def td_sensitivity_batched(
    pos: torch.Tensor,
    source_pos: torch.Tensor,
    source_normal: torch.Tensor,
    detector_pos: torch.Tensor,
    detector_normal: torch.Tensor,
    t: float,
    D: float,
    mu_a: float,
    c: float,
    mu_s_prime: float,
    A: float = DEFAULT_A,
    n_integration_points: int = 20,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Memory-efficient batched version of td_sensitivity.
    """
    n_pairs = source_pos.shape[0]
    n_points = pos.shape[0]

    sensitivity = torch.zeros(
        (n_pairs, n_points), dtype=torch.float32, device=pos.device
    )

    for i in tqdm(range(0, n_pairs, batch_size), desc=f"Computing TD sensitivity (t={t:.2f})"):
        end = min(i + batch_size, n_pairs)
        sensitivity[i:end] = td_sensitivity(
            pos,
            source_pos[i:end],
            source_normal[i:end],
            detector_pos[i:end],
            detector_normal[i:end],
            t, D, mu_a, c, mu_s_prime, A, n_integration_points,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sensitivity


def get_valid_source_detector_pairs(
    sensor_positions_mm: torch.Tensor,
    max_dist: float,
    head_center: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get unique source-detector pairs satisfying the distance criterion,
    together with their outward normals on the hemisphere surface.

    Parameters
    ----------
    sensor_positions_mm : (n_sensors, 3) optode positions on the hemisphere.
    max_dist : maximum allowed source-detector distance.
    head_center : (3,) hemisphere centre. If None, inferred from the
        guti.core convention (BRAIN_RADIUS, BRAIN_RADIUS, 0).

    Returns
    -------
    sources, source_normals, detectors, detector_normals : each (n_pairs, 3)
        Source-detector pairs are unordered; reciprocal pairs are not duplicated.
    """
    if head_center is None:
        from guti.core import BRAIN_RADIUS
        head_center = torch.tensor(
            [BRAIN_RADIUS, BRAIN_RADIUS, 0.0],
            dtype=sensor_positions_mm.dtype,
            device=sensor_positions_mm.device,
        )

    # Outward normals on the hemisphere (optode pointing away from head centre).
    v = sensor_positions_mm - head_center
    normals = v / torch.norm(v, dim=1, keepdim=True)

    d_mat = torch.norm(
        sensor_positions_mm[:, None, :] - sensor_positions_mm[None, :, :], dim=2
    )
    # Count each reciprocal S-D pair once (matches the CW fNIRS convention).
    # Including both (i, j) and (j, i) would double-count channels.
    mask = torch.triu((d_mat <= max_dist) & (d_mat > 0), diagonal=1)
    src_idx, det_idx = torch.nonzero(mask, as_tuple=True)

    return (
        sensor_positions_mm[src_idx],
        normals[src_idx],
        sensor_positions_mm[det_idx],
        normals[det_idx],
    )
