"""Active ultrasound free-field propagation helpers.

This module intentionally has no JAX/jwave dependency. The old heterogeneous
jwave ultrasound helpers live under ``guti.modalities._legacy.us_jwave``.
"""

from __future__ import annotations

import numpy as np
import torch

from guti.core import BRAIN_RADIUS


def simulate_free_field_propagation(
    source_positions: torch.Tensor,
    receiver_positions: torch.Tensor,
    source_signals: torch.Tensor,
    time_step: float,
    center_frequency: float,
    voxel_size: torch.Tensor,
    device: str = "cpu",
    compute_time_series: bool = False,
    temporal_sampling: int = 1,
) -> torch.Tensor:
    """Simulate homogeneous free-field ultrasound propagation.

    Positions are in meters. ``source_signals`` has shape
    ``(n_sources, n_time_steps)``. When ``compute_time_series`` is true, the
    result has shape ``(n_receivers, n_sources, ceil(n_time_steps / stride))``.
    """
    if temporal_sampling < 1:
        raise ValueError("temporal_sampling must be a positive integer >= 1")

    sound_speed = 1500.0  # m/s, water-like free-field approximation

    source_positions = source_positions.to(device)
    receiver_positions = receiver_positions.to(device)
    source_signals = source_signals.to(device)
    voxel_size = voxel_size.to(device)

    distances = torch.cdist(
        receiver_positions.float().unsqueeze(0),
        source_positions.float().unsqueeze(0),
    )[0]

    zero_distances = distances == 0
    if torch.any(zero_distances):
        eps = torch.tensor(1e-10, dtype=distances.dtype, device=device)
        distances = torch.where(zero_distances, eps, distances)

    # Scale by source-cell volume, not simulation voxel size. This keeps sweeps
    # tied to source discretization rather than medium-resolution dx.
    num_sources = source_signals.shape[0]
    source_volume_m3 = (2.0 / 3.0) * np.pi * (BRAIN_RADIUS * 1e-3) ** 3
    source_cell_volume = source_volume_m3 / float(num_sources)

    wavelength = sound_speed / center_frequency
    wavenumber = 2 * torch.pi / wavelength
    propagator_factor = (
        2 * wavenumber * source_cell_volume
    ) / (4 * torch.pi * distances)

    if not compute_time_series:
        return propagator_factor * torch.exp(-1j * wavenumber * distances)

    travel_times = distances / sound_speed
    delay_steps = torch.floor(travel_times / time_step).int()

    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]
    selected_time_indices = torch.arange(
        0,
        num_time_steps,
        temporal_sampling,
        device=device,
    )

    padded_source_signals = torch.cat(
        [
            torch.zeros(
                num_sources,
                1,
                device=device,
                dtype=source_signals.dtype,
            ),
            source_signals,
        ],
        dim=1,
    )

    source_idx = torch.arange(num_sources, device=device).unsqueeze(0).expand(
        num_receivers,
        num_sources,
    )
    pressure_field = torch.empty(
        (num_receivers, num_sources, selected_time_indices.shape[0]),
        dtype=padded_source_signals.dtype,
        device=device,
    )

    for idx, t_idx in enumerate(selected_time_indices):
        time_idx_matrix = t_idx - delay_steps + 1
        time_idx_matrix = torch.clamp(
            time_idx_matrix,
            min=0,
            max=padded_source_signals.shape[1] - 1,
        )
        delayed_signals = padded_source_signals[source_idx, time_idx_matrix]
        pressure_field[:, :, idx] = delayed_signals * propagator_factor

    return pressure_field
