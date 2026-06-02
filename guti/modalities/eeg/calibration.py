"""Empirically anchored EEG capacity.

The OpenMEEG/BEM EEG lead field carries a reliable *shape* but an unreliable
*absolute* gain (V per A·mm): a few source voxels at the brain/skull boundary blow
up numerically (peak response ~10^5× their neighbours), dominating the singular
spectrum. Trusting that absolute gain gives a top-mode SNR of ~3 and ~1 kbit/s.

We instead use only the lead-field *shape* and anchor the absolute SNR from physics:

  1. drop near-boundary source voxels so the BEM blow-ups don't enter the SVD, then
  2. set the **best spatial mode's** amplitude SNR to a literature single-channel
     value boosted by the realistic array gain, and scale the other modes by the
     spectrum shape σ_i / σ_1.

Why anchor the *top mode*, not a deep reference source. An earlier version pinned a
20 mm-deep reference voxel to single-channel SNR ≈ 1. Because the BEM lead field has
a steep depth gradient, the much-better-coupled superficial modes then blew up to
amplitude SNR ~200 (power ~40,000:1) — physically impossible for a single EEG
snapshot, and ~14× over the √N coherent-combination ceiling. Pinning the *best*
mode caps per-mode SNR at a physical value instead of inflating from a weak source.

The array gain is √N_eff, **not** √(n_electrodes): EEG sensor noise is heavily
spatially correlated (volume conduction, reference, biological artifact), so the
effective number of independent channels — EEG's spatial degrees of freedom — is
~20–40 regardless of electrode count. Combining the per-channel SNR with this gain
gives the best mode's SNR; the lead-field shape distributes it across the rest.

    snr_mode_i = snr_ref · √N_eff · (σ_i / σ_1)

Only the shape (σ_i/σ_1) comes from the lead field; the absolute V/(A·mm) scale is
discarded. See ``docs/superpowers/specs/2026-06-01-eeg-empirical-field-scaling-design.md``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from guti.core import BRAIN_RADIUS, get_bitrate, get_grid_positions

_LEADFIELD_MAT = (
    Path(__file__).resolve().parent.parent
    / "leadfields"
    / "eeg"
    / "eeg_leadfield.mat"
)

# The cached lead field was built on the default 5 mm hemisphere source grid.
_GRID_SPACING_MM = 5.0
# Hemisphere centre in the core coordinate frame: (R, R, 0), z up.
_BRAIN_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

DEFAULT_MARGIN_MM = 4.0
# Effective number of independent EEG channels (spatial degrees of freedom). Sensor
# noise is spatially correlated, so the coherent array gain is √N_eff with N_eff far
# below the electrode count; ~20–40 is the standard estimate for EEG.
DEFAULT_EFFECTIVE_CHANNELS = 32


def load_eeg_leadfield(
    path: Path | str = _LEADFIELD_MAT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, pos)`` for the cached OpenMEEG EEG lead field.

    ``A`` has shape ``(n_sensors, 3 * n_sources)`` with the three Cartesian dipole
    orientations interleaved per source voxel. ``pos`` is the ``(n_sources, 3)``
    source grid in mm, ordered to match ``A``'s columns.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Stored transposed as (3*n_sources, n_sensors).
        linop = np.array(f["linop"])
    A = linop.T
    n_sources = A.shape[1] // 3
    pos = get_grid_positions(_GRID_SPACING_MM)
    if pos.shape[0] != n_sources:
        raise ValueError(
            f"grid/leadfield mismatch: {pos.shape[0]} grid points vs "
            f"{n_sources} lead-field sources (expected {_GRID_SPACING_MM} mm grid)"
        )
    return A, pos


def exclude_boundary_voxels(
    A: np.ndarray,
    pos: np.ndarray,
    margin_mm: float = DEFAULT_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop source voxels within ``margin_mm`` of the brain surface.

    Near-boundary voxels carry the BEM numerical blow-ups; removing them collapses
    the spectrum's top dynamic range (``s1/s10``) from ~6000 to ~4. Operates on the
    interleaved-orientation column layout of ``A``.
    """
    r = np.linalg.norm(pos - _BRAIN_CENTER, axis=1)
    keep = r <= (BRAIN_RADIUS - margin_mm)
    A3 = A.reshape(A.shape[0], -1, 3)
    A_clean = A3[:, keep, :].reshape(A.shape[0], -1)
    return A_clean, pos[keep]


def anchored_mode_snr(
    A_clean: np.ndarray,
    snr_ref: float,
    effective_channels: int = DEFAULT_EFFECTIVE_CHANNELS,
) -> np.ndarray:
    """Per-mode **amplitude** SNR for the anchored EEG channel.

    The best spatial mode is pinned to ``snr_ref · √effective_channels`` (literature
    single-channel SNR × realistic array gain); the rest follow the lead-field
    spectrum shape. Absolute lead-field scale is not used.
    """
    sigma = np.linalg.svd(A_clean, compute_uv=False)
    top_mode_snr = snr_ref * np.sqrt(effective_channels)
    return top_mode_snr * sigma / sigma[0]


def anchored_eeg_bitrate(
    snr_ref: float,
    margin_mm: float = DEFAULT_MARGIN_MM,
    effective_channels: int = DEFAULT_EFFECTIVE_CHANNELS,
    time_resolution: float = 0.01,
    leadfield: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Empirically anchored EEG capacity in bits/s.

    ``bitrate = (1/2T) · Σ_i log2(1 + snr_mode_i²)`` with the per-mode SNR from
    :func:`anchored_mode_snr` — the same ``get_bitrate`` prefactor convention as the
    other modes (T = 0.01 ⇒ 50). Pass ``leadfield=(A, pos)`` to reuse a loaded field.
    """
    A, pos = leadfield if leadfield is not None else load_eeg_leadfield()
    A_clean, _ = exclude_boundary_voxels(A, pos, margin_mm)
    snr_modes = anchored_mode_snr(A_clean, snr_ref, effective_channels)
    # get_bitrate computes (1/2T) Σ log2(1 + (s/noise)^2); feed amplitude SNR with
    # unit noise.
    return float(get_bitrate(snr_modes, noise=1.0, time_resolution=time_resolution))
