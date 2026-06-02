"""Active ultrasound free-field propagation helpers.

This module intentionally has no JAX/jwave dependency. The old heterogeneous
jwave ultrasound helpers live under ``guti.modalities._legacy.us_jwave``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from guti.core import BRAIN_RADIUS, get_grid_positions, get_sensor_positions
from guti.mida_geometry import mida_brain_acoustic_speed_m_s, mida_brain_volume_mm3
from guti.linop import ChunkedForwardOperator

# Free-field constants shared by every entry point (the in-process USModality
# and the production analytical.py CLI), so the physics is defined exactly once.
try:
    SOUND_SPEED_M_S = mida_brain_acoustic_speed_m_s()
except (FileNotFoundError, KeyError):
    SOUND_SPEED_M_S = 1500.0      # water-like free-field approximation
POINTS_PER_WAVELENGTH = 24        # voxel size = c / (PPW * f)
TIME_DURATION_S = 120e-6          # simulated window
RECEIVER_OFFSET_MM = 8.0          # receiver standoff from the scalp
DEFAULT_CENTER_FREQ_HZ = 50e3
DEFAULT_RECEIVER_BATCH = 256      # receivers per matrix-free row block


def free_field_source_spacing_mm(n_sources: int) -> float:
    """Grid spacing (mm) giving ~``n_sources`` points in the brain volume."""
    try:
        volume_mm3 = mida_brain_volume_mm3()
    except (FileNotFoundError, KeyError, ValueError):
        volume_mm3 = (2.0 / 3.0) * np.pi * BRAIN_RADIUS**3
    return (volume_mm3 / n_sources) ** (1.0 / 3.0)


def create_free_field_sources(
    n_sources: int | None = None, *, source_spacing_mm: float | None = None
) -> np.ndarray:
    """Brain-grid source positions in **meters**.

    Provide either ``n_sources`` (spacing derived to hit that count) or an
    explicit ``source_spacing_mm``.
    """
    if (n_sources is None) == (source_spacing_mm is None):
        raise ValueError("Pass exactly one of n_sources or source_spacing_mm")
    if source_spacing_mm is None:
        source_spacing_mm = free_field_source_spacing_mm(n_sources)
    return get_grid_positions(grid_spacing_mm=source_spacing_mm) * 1e-3


def create_free_field_receivers(n_sensors: int) -> np.ndarray:
    """Ultrasound receiver positions in **meters**."""
    return get_sensor_positions(n_sensors=n_sensors, offset=RECEIVER_OFFSET_MM) * 1e-3


def free_field_time_axis(center_frequency: float) -> tuple[np.ndarray, float]:
    """Return ``(time_axis, time_step)`` for the free-field simulation window."""
    time_step = 1e-1 / center_frequency
    time_axis = np.arange(0, TIME_DURATION_S, time_step)
    return time_axis, time_step


def free_field_voxel_size(center_frequency: float) -> np.ndarray:
    """Isotropic voxel size (m) = ``c / (PPW * f)`` per axis."""
    dx_m = SOUND_SPEED_M_S / (POINTS_PER_WAVELENGTH * center_frequency)
    return np.array([dx_m, dx_m, dx_m])


def build_source_signal(
    time_axis: np.ndarray,
    center_frequency: float,
    signal_type: str = "tone_burst",
    signal_cycles: float = 2.0,
    signal_window: str = "hann",
) -> np.ndarray:
    """Per-source excitation waveform: continuous wave or windowed tone burst."""
    carrier = np.sin(2 * np.pi * time_axis * center_frequency)
    if signal_type == "cw":
        return carrier
    if signal_type != "tone_burst":
        raise ValueError(f"Unsupported signal_type={signal_type!r}")

    if time_axis.size == 0:
        return carrier
    if signal_cycles <= 0:
        raise ValueError("signal_cycles must be positive")

    if time_axis.size == 1:
        dt = 1.0 / (10.0 * center_frequency)
    else:
        dt = float(time_axis[1] - time_axis[0])
    active_duration = signal_cycles / center_frequency
    active_samples = max(1, min(time_axis.size, int(round(active_duration / dt))))

    envelope = np.ones(active_samples, dtype=np.float64)
    if signal_window == "hann":
        if active_samples > 1:
            envelope = np.hanning(active_samples)
    elif signal_window != "rect":
        raise ValueError(f"Unsupported signal_window={signal_window!r}")

    signal = np.zeros_like(carrier)
    signal[:active_samples] = carrier[:active_samples] * envelope
    return signal


def make_free_field_chunk_fn(
    source_positions_t: torch.Tensor,
    receiver_positions: np.ndarray,
    source_signals_t: torch.Tensor,
    *,
    time_step: float,
    center_frequency: float,
    voxel_size_t: torch.Tensor,
    temporal_sampling: int,
    device: str,
    num_sources: int,
    use_complex_amplitudes: bool = False,
):
    """Return ``compute_chunk(start, end)`` for a batch of receivers.

    The closure yields the sensor-time rows for receivers ``[start:end)`` as a
    ``((end-start) * nt, num_sources)`` float tensor — the single definition of
    the receiver-batch chunk shared by ``USModality`` and ``analytical.py``.
    """

    def compute_chunk(start: int, end: int) -> torch.Tensor:
        receivers_t = torch.as_tensor(receiver_positions[start:end], device=device)
        pf = simulate_free_field_propagation(
            source_positions_t,
            receivers_t,
            source_signals_t,
            time_step,
            center_frequency,
            voxel_size_t,
            device=device,
            compute_time_series=not use_complex_amplitudes,
            temporal_sampling=temporal_sampling,
        )
        if use_complex_amplitudes:
            return torch.cat([pf.real, pf.imag], dim=0).float()
        # (batch, n_sources, nt) -> (batch*nt, n_sources)
        return pf.permute(0, 2, 1).reshape(-1, num_sources).float()

    return compute_chunk


def build_free_field_operator(
    source_positions_m: np.ndarray,
    receiver_positions_m: np.ndarray,
    *,
    center_frequency: float = DEFAULT_CENTER_FREQ_HZ,
    temporal_sampling: int = 1,
    source_signals: np.ndarray | None = None,
    signal_type: str = "cw",
    signal_cycles: float = 2.0,
    signal_window: str = "hann",
    receiver_batch: int = DEFAULT_RECEIVER_BATCH,
    backend: str = "torch",
    device: str = "cpu",
    dtype=None,
) -> tuple[ChunkedForwardOperator, dict]:
    """Assemble the matrix-free free-field forward operator.

    Returns ``(operator, meta)`` where ``meta`` carries ``time_step``,
    ``time_resolution`` (``time_step * temporal_sampling``), ``nt`` and
    ``n_outputs``. Positions are in meters. When ``source_signals`` is omitted a
    per-source waveform is built from ``signal_type`` and tiled across sources.
    """
    if dtype is None:
        dtype = torch.float32
    time_axis, time_step = free_field_time_axis(center_frequency)
    nt = len(range(0, time_axis.shape[0], temporal_sampling))
    n_sources = len(source_positions_m)
    n_receivers = len(receiver_positions_m)

    if source_signals is None:
        waveform = build_source_signal(
            time_axis,
            center_frequency,
            signal_type=signal_type,
            signal_cycles=signal_cycles,
            signal_window=signal_window,
        )
        source_signals = np.tile(waveform, (n_sources, 1))

    source_positions_t = torch.as_tensor(source_positions_m, device=device)
    source_signals_t = torch.as_tensor(source_signals, device=device)
    voxel_size_t = torch.as_tensor(free_field_voxel_size(center_frequency), device=device)

    chunk_fn = make_free_field_chunk_fn(
        source_positions_t,
        receiver_positions_m,
        source_signals_t,
        time_step=time_step,
        center_frequency=center_frequency,
        voxel_size_t=voxel_size_t,
        temporal_sampling=temporal_sampling,
        device=device,
        num_sources=n_sources,
    )

    operator = ChunkedForwardOperator.from_item_batches(
        n_sources,
        n_receivers,
        nt,
        chunk_fn,
        batch_size=receiver_batch,
        backend=backend,
        dtype=dtype,
        device=device,
    )
    meta = {
        "time_step": time_step,
        "time_resolution": time_step * temporal_sampling,
        "nt": nt,
        "n_outputs": n_receivers * nt,
    }
    return operator, meta


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
