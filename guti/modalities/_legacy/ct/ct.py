"""
Simple CT simulation & reconstruction utilities
===============================================

This module depends on *scikit‑image* (``pip install scikit-image``).

Workflow
--------
1. Build a 2‑D attenuation map from the brain geometry in ``guti.core``:
   • brain voxels get random μ values  
   • skull, scalp, and air get fixed coefficients  
2. Generate a parallel‑beam sinogram by forward‑projecting the slice.  
3. Reconstruct the slice via filtered back‑projection (FBP).

Example
-------
Run this file directly to see a demo::

    python -m guti.modalities.ct.ct
"""
# %%
from __future__ import annotations

import numpy as np
from skimage.transform import radon, iradon

from guti.core import get_voxel_mask, get_source_positions, BRAIN_RADIUS

# ----------------------------------------------------------------------
#  Linear attenuation coefficients (unit: mm⁻¹, but only relative matters)
# ----------------------------------------------------------------------
MU_AIR: float = 0.00
MU_SCALP: float = 0.15
MU_SKULL: float = 0.50
MU_BRAIN_MEAN: float = 0.20

# Gaussian‑blob parameters
N_SOURCES: int = 100          # number of point sources
GAUSS_SIGMA: float = 3.0      # mm, spatial spread (FWHM≈2.35σ)
MU_SOURCE_DELTA: float = 0.10 # added μ at the center of each blob


# ----------------------------------------------------------------------
def build_attenuation_map(
    resolution: float = 1.0,
    slice_index: int | None = None,
    source_positions: np.ndarray | None = None,
    source_strengths: np.ndarray | None = None,
    n_sources: int | None = None,
    source_sigma: float | None = None,
    rng: int | None = None,
) -> np.ndarray:
    """
    Create a 2‑D attenuation map (μ) for an axial slice.

    • Brain voxels start at a uniform ``MU_BRAIN_MEAN``.  
    • Add Gaussian blobs around specified sources or randomly generated ones.
    • Skull, scalp, and air retain fixed coefficients.

    Parameters
    ----------
    resolution
        Spatial resolution (mm/voxel) passed to :func:`guti.core.get_voxel_mask`.
    slice_index
        z‑index of the slice (defaults to the central slice).
    source_positions
        Array of shape (n_sources, 3) containing (x,y,z) positions of sources in mm.
        If None, sources will be randomly generated using n_sources and source_sigma.
    source_strengths
        Array of shape (n_sources,) containing the peak μ value for each source.
        If None, defaults to MU_SOURCE_DELTA for all sources.
    n_sources
        Number of sources to generate randomly. Only used if source_positions is None.
    source_sigma
        Standard deviation of Gaussian blobs in mm. Only used if source_positions is None.
    rng
        Seed for deterministic placement of the sources when generating randomly.

    Returns
    -------
    ndarray
        2‑D array (ny, nx) containing linear attenuation coefficients.
    """
    mask = get_voxel_mask(resolution)
    if slice_index is None:
        slice_index = mask.shape[2] // 2  # middle axial slice

    m2d = mask[:, :, slice_index].astype(np.int8)

    mu = np.empty_like(m2d, dtype=np.float32)
    mu[m2d == 0] = MU_AIR
    mu[m2d == 3] = MU_SCALP
    mu[m2d == 2] = MU_SKULL
    mu[m2d == 1] = MU_BRAIN_MEAN  # uniform brain background

    # Grid coordinates (mm) for this slice
    nx, ny = m2d.shape
    x_mm = np.linspace(-BRAIN_RADIUS, BRAIN_RADIUS, nx)
    y_mm = np.linspace(-BRAIN_RADIUS, BRAIN_RADIUS, ny)
    X2, Y2 = np.meshgrid(x_mm, y_mm, indexing="ij")

    # z‑coordinate of this slice
    nz = get_voxel_mask(resolution).shape[2]
    z_mm_slice = (slice_index / (nz - 1)) * BRAIN_RADIUS

    # Get source positions and strengths
    if source_positions is None:
        if n_sources is None:
            n_sources = N_SOURCES
        if source_sigma is None:
            source_sigma = GAUSS_SIGMA
        source_positions = get_source_positions(n_sources, rng=rng)
        source_strengths = np.full(n_sources, MU_SOURCE_DELTA)
    else:
        if source_strengths is None:
            source_strengths = np.full(len(source_positions), MU_SOURCE_DELTA)
        if len(source_positions) != len(source_strengths):
            raise ValueError("source_positions and source_strengths must have the same length")

    # Add Gaussian blobs around sources that lie close to this slice
    sigma2 = source_sigma ** 2
    for (xs, ys, zs), strength in zip(source_positions, source_strengths):
        if abs(zs - z_mm_slice) > 3 * source_sigma:  # skip distant sources
            continue
        x_phys = xs - BRAIN_RADIUS
        y_phys = ys - BRAIN_RADIUS
        gauss = strength * np.exp(
            -((X2 - x_phys) ** 2 + (Y2 - y_phys) ** 2) / (2 * sigma2)
        )
        mu[m2d == 1] += gauss[m2d == 1]

    return mu


# ----------------------------------------------------------------------
def simulate_projections(
    mu: np.ndarray,
    n_angles: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a parallel‑beam sinogram (Radon transform) for a slice.

    Parameters
    ----------
    mu
        2‑D attenuation map produced by :func:`build_attenuation_map`.
    n_angles
        Number of projection angles spanning 180°.

    Returns
    -------
    sinogram
        2‑D array (n_detectors, n_angles) of line integrals.
    theta
        1‑D array of projection angles in degrees.
    """
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(mu, theta=theta, circle=False)
    return sinogram, theta


# ----------------------------------------------------------------------
def reconstruct(
    sinogram: np.ndarray,
    theta: np.ndarray,
    filter_name: str = "ramp",
) -> np.ndarray:
    """
    Reconstruct a slice from its sinogram via filtered back‑projection.

    Parameters
    ----------
    sinogram
        Sinogram returned by :func:`simulate_projections`.
    theta
        Projection angles corresponding to the sinogram columns.
    filter_name
        Frequency filter (``'ramp'``, ``'shepp‑logan'``, ``'hann'``, ...).

    Returns
    -------
    ndarray
        Reconstructed 2‑D attenuation map.
    """
    return iradon(sinogram, theta=theta, filter_name=filter_name, circle=False)


# ----------------------------------------------------------------------
def demo(
    resolution: float = 1.0,
    n_angles: int = 360,
    rng: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience routine: forward project & reconstruct a slice.

    Now places Gaussian blobs in the brain region.

    Returns
    -------
    truth, sinogram, recon : (ndarray, ndarray, ndarray)
    """
    truth = build_attenuation_map(resolution=resolution, rng=rng)
    sinogram, theta = simulate_projections(truth, n_angles=n_angles)
    recon = reconstruct(sinogram, theta)
    return truth, sinogram, recon


# %%
# Quick visual sanity check
truth, sino, recon = demo()

import matplotlib.pyplot as plt  # Lazy import for demo only

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(truth, cmap="gray")
ax[0].set_title("Ground truth μ")
ax[0].axis("off")

ax[1].imshow(sino, cmap="gray", aspect="auto")
ax[1].set_title("Sinogram")
ax[1].axis("off")

ax[2].imshow(recon, cmap="gray")
ax[2].set_title("Reconstruction")
ax[2].axis("off")

plt.tight_layout()
plt.show()
# %%


