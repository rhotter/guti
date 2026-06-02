"""Empirically anchored EEG capacity.

The OpenMEEG/BEM EEG lead field carries a reliable *shape* but an unreliable
*absolute* gain (V per A·mm): a few source voxels at the brain/skull boundary blow
up numerically (peak response ~10^5× their neighbours), dominating the singular
spectrum. Trusting that absolute gain gives a top-mode SNR of ~3 and ~1 kbit/s,
~80× below an anchored estimate.

This module instead does what the sibling `guti2` pipeline does for EEG:

  1. drop near-boundary source voxels so the BEM blow-ups don't enter the SVD, then
  2. anchor the source amplitude so a canonical, fixed-depth cortical source
     produces a literature single-channel amplitude SNR (`snr_ref`). The SVD then
     spreads that budget across spatial modes.

Only the lead-field shape is used; the absolute V/(A·mm) scale is discarded.

See ``docs/superpowers/specs/2026-06-01-eeg-empirical-field-scaling-design.md``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from guti.capacity import get_bitrate, get_capacity
from guti.core import BRAIN_RADIUS, SCALP_RADIUS, get_grid_positions

_LEADFIELD_MAT = (
    Path(__file__).resolve().parent.parent
    / "leadfields"
    / "eeg"
    / "eeg_leadfield.mat"
)

# Hemisphere centre in the core coordinate frame: (R, R, 0), z up.
_BRAIN_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

# Known cached-leadfield layouts. The current repo cache has 984 source voxels,
# matching a 10 mm grid with a 5 mm surface margin; older/generated caches may
# use other grid layouts.
_GRID_LAYOUT_CANDIDATES = (
    (5.0, 0.0),
    (10.0, 5.0),
    (10.0, 0.0),
)

DEFAULT_MARGIN_MM = 4.0
DEFAULT_REF_DEPTH_MM = 20.0


def _candidate_grid_positions(spacing_mm: float, margin_mm: float) -> np.ndarray:
    positions = get_grid_positions(spacing_mm)
    if margin_mm <= 0.0:
        return positions
    r = np.linalg.norm(positions - _BRAIN_CENTER, axis=1)
    return positions[r < (BRAIN_RADIUS - margin_mm)]


def _source_positions_for_cached_leadfield(n_sources: int) -> np.ndarray:
    for spacing_mm, margin_mm in _GRID_LAYOUT_CANDIDATES:
        positions = _candidate_grid_positions(spacing_mm, margin_mm)
        if positions.shape[0] == n_sources:
            return positions
    raise ValueError(
        f"no known source grid matches cached lead field with {n_sources} sources"
    )


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
    pos = _source_positions_for_cached_leadfield(n_sources)
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


def _reference_column(pos_clean: np.ndarray, ref_depth_mm: float) -> int:
    """Index of the source voxel nearest ``ref_depth_mm`` below the scalp, on +z.

    A *fixed* depth (not "the most superficial voxel") keeps the reference-signal
    magnitude grid-invariant — otherwise the nearest-to-surface voxel drifts toward
    the sensors as the grid refines and its near-field response diverges.
    """
    target = _BRAIN_CENTER + np.array([0.0, 0.0, SCALP_RADIUS - ref_depth_mm])
    return int(np.argmin(np.linalg.norm(pos_clean - target, axis=1)))


def anchored_mode_snr(
    A_clean: np.ndarray,
    pos_clean: np.ndarray,
    snr_ref: float,
    ref_depth_mm: float = DEFAULT_REF_DEPTH_MM,
) -> np.ndarray:
    """Per-mode **amplitude** SNR for the anchored EEG channel.

    ``S0`` is chosen so the best single sensor sees amplitude SNR ``snr_ref`` for a
    unit source at the reference voxel; each SVD mode then inherits
    ``snr_ref * sigma_i / peak_ref``.
    """
    ref = _reference_column(pos_clean, ref_depth_mm)
    peak_ref = np.max(np.abs(A_clean[:, 3 * ref : 3 * ref + 3]))
    if peak_ref <= 0:
        raise ValueError("reference voxel has zero lead-field response")
    sigma = np.linalg.svd(A_clean, compute_uv=False)
    return snr_ref * sigma / peak_ref


def anchored_eeg_bitrate(
    snr_ref: float,
    margin_mm: float = DEFAULT_MARGIN_MM,
    ref_depth_mm: float = DEFAULT_REF_DEPTH_MM,
    time_resolution: float = 0.01,
    leadfield: tuple[np.ndarray, np.ndarray] | None = None,
    spectrum_kwargs: dict | None = None,
) -> float:
    """Empirically anchored EEG capacity in bits/s.

    ``bitrate = (1/2T) * sum_i log2(1 + (snr_ref * sigma_i / peak_ref)^2)`` — the
    same ``get_bitrate`` prefactor convention as the other modes (T = 0.01 ⇒ 50).
    If ``spectrum_kwargs`` are supplied, ``snr_ref`` is treated as a full-band
    amplitude SNR over ``1 / time_resolution`` Hz; the shared capacity code then
    scales the unit full-band noise to each frequency bin with
    ``sqrt(delta_f / bandwidth)``.
    Pass ``leadfield=(A, pos)`` to use a non-cached layout (e.g. from
    ``eeg.modality.build_eeg_gain``); otherwise the cached 256-sensor field is used.
    """
    A, pos = leadfield if leadfield is not None else load_eeg_leadfield()
    A_clean, pos_clean = exclude_boundary_voxels(A, pos, margin_mm)
    snr_modes = anchored_mode_snr(A_clean, pos_clean, snr_ref, ref_depth_mm)
    # get_bitrate computes (1/2T) sum log2(1 + (s/noise)^2); feed amplitude SNR
    # directly with unit noise and unit input power per mode.
    return float(
        get_bitrate(
            snr_modes,
            n_sources=len(snr_modes),
            total_input_power=float(len(snr_modes)),
            noise=1.0,
            time_resolution=time_resolution,
            **(spectrum_kwargs or {}),
        )
    )


def anchored_eeg_capacity(
    snr_ref: float,
    margin_mm: float = DEFAULT_MARGIN_MM,
    ref_depth_mm: float = DEFAULT_REF_DEPTH_MM,
    time_resolution: float = 0.01,
    leadfield: tuple[np.ndarray, np.ndarray] | None = None,
    spectrum_kwargs: dict | None = None,
) -> float:
    """Water-filled counterpart to :func:`anchored_eeg_bitrate`."""
    A, pos = leadfield if leadfield is not None else load_eeg_leadfield()
    A_clean, pos_clean = exclude_boundary_voxels(A, pos, margin_mm)
    snr_modes = anchored_mode_snr(A_clean, pos_clean, snr_ref, ref_depth_mm)
    return float(
        get_capacity(
            snr_modes,
            n_sources=len(snr_modes),
            total_input_power=float(len(snr_modes)),
            noise=1.0,
            time_resolution=time_resolution,
            **(spectrum_kwargs or {}),
        )
    )
