"""Capacity and bitrate utilities for linear imaging channels."""

from __future__ import annotations

import numpy as np


def _as_spectrum(s: np.ndarray, *, name: str = "s") -> np.ndarray:
    spectrum = np.asarray(s, dtype=float)
    if spectrum.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of singular values")
    if np.any(spectrum < 0):
        raise ValueError(f"{name} must contain non-negative singular values")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError(f"{name} must contain finite singular values")
    return spectrum


def _as_channel_matrix(channel: np.ndarray, *, name: str = "channel") -> np.ndarray:
    matrix = np.asarray(channel, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D output-by-input channel matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite entries")
    return matrix


def _spectrum_from_channel_or_spectrum(channel_or_spectrum: np.ndarray) -> np.ndarray:
    values = np.asarray(channel_or_spectrum, dtype=float)
    if values.ndim == 1:
        return _as_spectrum(values)
    if values.ndim == 2:
        return np.linalg.svd(_as_channel_matrix(values), compute_uv=False)
    raise ValueError("s must be either singular values or a 2D channel matrix")


def _validate_optional_matrix_shape(
    channel_or_spectrum: np.ndarray,
    *,
    n_sources: int | None = None,
    n_outputs: int | None = None,
) -> None:
    values = np.asarray(channel_or_spectrum)
    if values.ndim != 2:
        return
    if n_outputs is not None and values.shape[0] != n_outputs:
        raise ValueError("n_outputs must match the channel matrix output dimension")
    if n_sources is not None and values.shape[1] != n_sources:
        raise ValueError("n_sources must match the channel matrix input dimension")


def _as_output_noise_covariance(
    output_noise_covariance: np.ndarray,
    *,
    n_outputs: int,
) -> np.ndarray:
    covariance = np.asarray(output_noise_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("output_noise_covariance must be a square matrix")
    if covariance.shape != (n_outputs, n_outputs):
        raise ValueError(
            "output_noise_covariance shape must match the channel output dimension"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("output_noise_covariance must contain finite entries")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("output_noise_covariance must be symmetric")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0.0:
        raise ValueError("output_noise_covariance must be positive definite")
    return covariance


def _whitened_channel_spectrum(
    channel: np.ndarray,
    *,
    output_noise_covariance: np.ndarray,
) -> np.ndarray:
    matrix = _as_channel_matrix(channel)
    covariance = _as_output_noise_covariance(
        output_noise_covariance,
        n_outputs=matrix.shape[0],
    )

    noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(covariance)
    rotated_channel = noise_eigenvectors.T @ matrix
    whitened_channel = rotated_channel / np.sqrt(noise_eigenvalues)[:, None]
    return np.linalg.svd(whitened_channel, compute_uv=False)


def _noise_normalized_spectrum(
    channel_or_spectrum: np.ndarray,
    *,
    noise: float | None,
    output_noise_covariance: np.ndarray | None,
) -> np.ndarray:
    if output_noise_covariance is None:
        if noise is None:
            raise ValueError("noise must be provided when output_noise_covariance is not")
        if noise <= 0:
            raise ValueError("noise must be positive")
        return _spectrum_from_channel_or_spectrum(channel_or_spectrum) / noise

    if noise is not None:
        raise ValueError(
            "Pass either scalar noise or output_noise_covariance, not both"
        )
    return _whitened_channel_spectrum(
        channel_or_spectrum,
        output_noise_covariance=output_noise_covariance,
    )


def noise_normalized_singular_values(
    channel_or_spectrum: np.ndarray,
    *,
    noise: float | None = None,
    output_noise_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Return singular values after output-noise whitening.

    With scalar ``noise`` this is simply ``s / noise``.  With a full output
    covariance, ``channel_or_spectrum`` must be the output-by-input channel
    matrix and the result is the spectrum of ``K_N^{-1/2} H``.
    """
    return _noise_normalized_spectrum(
        channel_or_spectrum,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
    )


def sensor_noise_normalized_singular_values(
    channel: np.ndarray,
    *,
    sensor_noise_covariance: np.ndarray,
    outputs_per_sensor: int = 1,
) -> np.ndarray:
    """Return spectrum of ``K_N^{-1/2} H`` for separable sensor noise.

    The channel rows must be grouped by sensor, then by output index within
    each sensor.  This is equivalent to a full output covariance of
    ``kron(sensor_noise_covariance, I_outputs_per_sensor)`` without explicitly
    forming that larger matrix.
    """
    matrix = _as_channel_matrix(channel)
    covariance = np.asarray(sensor_noise_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("sensor_noise_covariance must be a square matrix")
    if outputs_per_sensor <= 0:
        raise ValueError("outputs_per_sensor must be positive")
    n_sensors = covariance.shape[0]
    if matrix.shape[0] != n_sensors * outputs_per_sensor:
        raise ValueError(
            "channel output rows must equal "
            "sensor_noise_covariance.shape[0] * outputs_per_sensor"
        )

    covariance = 0.5 * (covariance + covariance.T)
    noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(covariance)
    if noise_eigenvalues[0] <= 0.0:
        raise ValueError("sensor_noise_covariance must be positive definite")
    blocks = matrix.reshape(n_sensors, outputs_per_sensor, matrix.shape[1])
    rotated = np.einsum("ji,jok->iok", noise_eigenvectors, blocks, optimize=True)
    whitened = rotated / np.sqrt(noise_eigenvalues)[:, None, None]
    whitened_matrix = whitened.reshape(matrix.shape)
    m, n = whitened_matrix.shape
    if m <= n:
        gram = whitened_matrix @ whitened_matrix.T
    else:
        gram = whitened_matrix.T @ whitened_matrix
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    return np.sqrt(np.clip(eigenvalues[::-1], 0.0, None))


def total_input_power_from_average_output_power(
    s: np.ndarray,
    *,
    average_output_power: float,
    n_sources: int,
    n_outputs: int,
) -> float:
    """Estimate total input power from per-output-channel average signal power.

    Assumes i.i.d. source power, so

        P_in,total = n_sources * n_outputs * P_out,avg / sum_i s_i^2

    where ``s`` is either the forward operator singular-value spectrum or the
    forward operator itself, in the same units as the supplied output power.
    """
    _validate_optional_matrix_shape(s, n_sources=n_sources, n_outputs=n_outputs)
    values = np.asarray(s, dtype=float)
    if average_output_power < 0:
        raise ValueError("average_output_power must be non-negative")
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive")

    if values.ndim == 1:
        spectrum_power = float(np.sum(_as_spectrum(values) ** 2))
    elif values.ndim == 2:
        spectrum_power = float(np.sum(_as_channel_matrix(values) ** 2))
    else:
        raise ValueError("s must be either singular values or a 2D channel matrix")
    if spectrum_power <= 0.0:
        raise ValueError("sum(s_i^2) must be positive")

    return float(n_sources * n_outputs * average_output_power / spectrum_power)


def total_input_power_from_input_amplitude(
    input_amplitude: float,
    *,
    n_sources: int,
) -> float:
    """Return total input power from a per-source/channel input amplitude."""
    if input_amplitude < 0:
        raise ValueError("input_amplitude must be non-negative")
    if not np.isfinite(input_amplitude):
        raise ValueError("input_amplitude must be finite")
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")
    return float(n_sources * input_amplitude**2)


def resolve_total_input_power(
    s: np.ndarray | None = None,
    *,
    n_sources: int | None = None,
    n_outputs: int | None = None,
    total_input_power: float | None = None,
    input_power_per_source: float | None = None,
    input_amplitude: float | None = None,
    average_output_power: float | None = None,
) -> float:
    """Resolve exactly one input-power convention to total input power.

    Callers may provide an explicit total input power, a per-source input power,
    a per-source input amplitude, or an average per-output signal power.
    """
    provided = [
        total_input_power is not None,
        input_power_per_source is not None,
        input_amplitude is not None,
        average_output_power is not None,
    ]
    if sum(provided) != 1:
        raise ValueError(
            "Provide exactly one of total_input_power, input_power_per_source, "
            "input_amplitude, or average_output_power"
        )

    if total_input_power is not None:
        if total_input_power < 0:
            raise ValueError("total_input_power must be non-negative")
        if not np.isfinite(total_input_power):
            raise ValueError("total_input_power must be finite")
        return float(total_input_power)

    if n_sources is None:
        raise ValueError(
            "n_sources is required for input_power_per_source, input_amplitude, "
            "or average_output_power"
        )
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")

    if input_power_per_source is not None:
        if input_power_per_source < 0:
            raise ValueError("input_power_per_source must be non-negative")
        if not np.isfinite(input_power_per_source):
            raise ValueError("input_power_per_source must be finite")
        return float(n_sources * input_power_per_source)

    if input_amplitude is not None:
        return total_input_power_from_input_amplitude(
            input_amplitude,
            n_sources=n_sources,
        )

    if s is None:
        raise ValueError("s is required when deriving input power from output power")
    if n_outputs is None:
        raise ValueError("n_outputs is required for average_output_power")
    return total_input_power_from_average_output_power(
        s,
        average_output_power=average_output_power,
        n_sources=n_sources,
        n_outputs=n_outputs,
    )


def get_bitrate(
    s: np.ndarray,
    *,
    n_sources: int,
    total_input_power: float | None = None,
    input_power_per_source: float | None = None,
    input_amplitude: float | None = None,
    average_output_power: float | None = None,
    n_outputs: int | None = None,
    noise: float | None = None,
    time_resolution: float = 1.0,
    output_noise_covariance: np.ndarray | None = None,
) -> float:
    """Return i.i.d.-input bitrate for a linear Gaussian channel.

    The total input power is spread uniformly over ``n_sources`` source
    channels. By default ``noise`` is the per-output-channel i.i.d. noise
    standard deviation and ``s`` may be singular values or the channel matrix.
    If ``output_noise_covariance`` is provided, ``s`` must be the output-by-input
    channel matrix and the code uses the singular values of
    ``K_N^{-1/2} H``.
    """
    _validate_optional_matrix_shape(s, n_sources=n_sources, n_outputs=n_outputs)
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")
    if time_resolution <= 0:
        raise ValueError("time_resolution must be positive")

    resolved_total_input_power = resolve_total_input_power(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    gains = _noise_normalized_spectrum(
        s,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
    )
    resolved_power_per_source = resolved_total_input_power / n_sources
    snr_per_mode = (gains**2) * resolved_power_per_source
    return float(np.sum(np.log2(1.0 + snr_per_mode)) / (2.0 * time_resolution))


def water_filling_power_allocation(
    s: np.ndarray,
    *,
    total_input_power: float,
    noise: float | None = None,
    output_noise_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate input power across independent SVD modes by water filling."""
    if total_input_power < 0:
        raise ValueError("total_input_power must be non-negative")
    if not np.isfinite(total_input_power):
        raise ValueError("total_input_power must be finite")

    gains = _noise_normalized_spectrum(
        s,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
    )
    allocation = np.zeros_like(gains)
    active = gains > 0
    if total_input_power == 0 or not np.any(active):
        return allocation

    active_indices = np.flatnonzero(active)
    active_gains = gains[active_indices]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        floors = 1.0 / (active_gains**2)

    finite = np.isfinite(floors)
    if not np.any(finite):
        allocation[int(active_indices[np.argmax(active_gains)])] = total_input_power
        return allocation

    finite_indices = active_indices[finite]
    finite_floors = floors[finite]
    order = np.argsort(finite_floors)
    sorted_floors = finite_floors[order]
    prefix = np.cumsum(sorted_floors, dtype=float)

    active_count = len(sorted_floors)
    for k in range(1, len(sorted_floors) + 1):
        water_level = (total_input_power + prefix[k - 1]) / k
        if k == len(sorted_floors) or water_level <= sorted_floors[k]:
            active_count = k
            break

    sorted_allocation = np.zeros_like(sorted_floors)
    if active_count == 1:
        sorted_allocation[0] = total_input_power
    else:
        active_floors = sorted_floors[:active_count]
        mean_floor = prefix[active_count - 1] / active_count
        sorted_allocation[:active_count] = np.maximum(
            0.0,
            total_input_power / active_count + (mean_floor - active_floors),
        )
        allocated = float(np.sum(sorted_allocation[:active_count]))
        if allocated > 0.0 and np.isfinite(allocated):
            sorted_allocation[:active_count] *= total_input_power / allocated
        else:
            sorted_allocation[0] = total_input_power

    allocation[finite_indices[order]] = sorted_allocation
    return allocation


def get_capacity(
    s: np.ndarray,
    *,
    total_input_power: float | None = None,
    input_power_per_source: float | None = None,
    input_amplitude: float | None = None,
    average_output_power: float | None = None,
    n_sources: int | None = None,
    n_outputs: int | None = None,
    noise: float | None = None,
    time_resolution: float = 1.0,
    output_noise_covariance: np.ndarray | None = None,
) -> float:
    """Return water-filled channel capacity for a linear Gaussian channel."""
    _validate_optional_matrix_shape(s, n_sources=n_sources, n_outputs=n_outputs)
    if time_resolution <= 0:
        raise ValueError("time_resolution must be positive")

    resolved_total_input_power = resolve_total_input_power(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    gains = _noise_normalized_spectrum(
        s,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
    )
    input_power_per_mode = water_filling_power_allocation(
        gains,
        total_input_power=resolved_total_input_power,
        noise=1.0,
    )
    snr_per_mode = (gains**2) * input_power_per_mode
    return float(np.sum(np.log2(1.0 + snr_per_mode)) / (2.0 * time_resolution))


def get_bitrate_temporal_filter(
    s: np.ndarray,
    freqs: np.ndarray,
    H_magnitude: np.ndarray,
    *,
    n_sources: int,
    total_input_power: float | None = None,
    input_power_per_source: float | None = None,
    input_amplitude: float | None = None,
    average_output_power: float | None = None,
    n_outputs: int | None = None,
    noise: float,
) -> float:
    """Return bitrate for a spatial spectrum followed by a temporal filter."""
    spectrum = _as_spectrum(s)
    freqs = np.asarray(freqs, dtype=float)
    H_magnitude = np.asarray(H_magnitude, dtype=float)
    if freqs.ndim != 1 or H_magnitude.ndim != 1:
        raise ValueError("freqs and H_magnitude must be 1D arrays")
    if freqs.shape != H_magnitude.shape:
        raise ValueError("freqs and H_magnitude must have the same shape")
    if len(freqs) < 2:
        return 0.0
    if noise <= 0:
        raise ValueError("noise must be positive")

    resolved_total_input_power = resolve_total_input_power(
        spectrum,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    input_power_per_source = resolved_total_input_power / n_sources
    df = float(freqs[1] - freqs[0])
    gains = np.outer(spectrum, H_magnitude).ravel() / noise
    return df * float(np.sum(np.log2(1.0 + (gains**2) * input_power_per_source)))


def get_capacity_temporal_filter(
    s: np.ndarray,
    freqs: np.ndarray,
    H_magnitude: np.ndarray,
    *,
    n_sources: int,
    total_input_power: float | None = None,
    input_power_per_source: float | None = None,
    input_amplitude: float | None = None,
    average_output_power: float | None = None,
    n_outputs: int | None = None,
    noise: float,
) -> float:
    """Return water-filled capacity for a spatial spectrum followed by a temporal filter.

    Water-filling counterpart of :func:`get_bitrate_temporal_filter`. The
    (spatial mode × frequency bin) grid is treated as a bank of independent
    parallel Gaussian channels with effective gains ``s_k · |H(f_m)| / noise``;
    the total input power is allocated optimally across them (same convention as
    :func:`get_capacity`), rather than spread uniformly.

    Power budget matches the equal-power version exactly: the equal-power path
    puts ``total_input_power / n_sources`` on every one of the
    ``n_sources × n_freqs`` grid entries, so the aggregate budget water-filled
    here is ``(total_input_power / n_sources) × n_sources × n_freqs``. On that
    identical budget capacity is always ≥ the equal-power bitrate, and the
    result is invariant to the frequency resolution ``df`` (once fine enough to
    resolve ``H``).
    """
    spectrum = _as_spectrum(s)
    freqs = np.asarray(freqs, dtype=float)
    H_magnitude = np.asarray(H_magnitude, dtype=float)
    if freqs.ndim != 1 or H_magnitude.ndim != 1:
        raise ValueError("freqs and H_magnitude must be 1D arrays")
    if freqs.shape != H_magnitude.shape:
        raise ValueError("freqs and H_magnitude must have the same shape")
    if len(freqs) < 2:
        return 0.0
    if noise <= 0:
        raise ValueError("noise must be positive")

    resolved_total_input_power = resolve_total_input_power(
        spectrum,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    df = float(freqs[1] - freqs[0])
    gains = np.outer(spectrum, H_magnitude).ravel() / noise
    # Match the equal-power aggregate budget: per-source power applied to every
    # (mode × freq) grid entry, then water-filled over the whole grid.
    power_per_source = resolved_total_input_power / n_sources
    grid_total_power = power_per_source * gains.size
    input_power_per_mode = water_filling_power_allocation(
        gains,
        total_input_power=grid_total_power,
        noise=1.0,
    )
    return df * float(np.sum(np.log2(1.0 + (gains**2) * input_power_per_mode)))


def get_bitrate_from_average_output_power(
    s: np.ndarray,
    *,
    average_output_power: float,
    noise: float | None = None,
    n_sources: int,
    n_outputs: int,
    time_resolution: float = 1.0,
    output_noise_covariance: np.ndarray | None = None,
) -> float:
    """Return i.i.d.-input bitrate from observed output signal/noise levels."""
    return get_bitrate(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        average_output_power=average_output_power,
        noise=noise,
        time_resolution=time_resolution,
        output_noise_covariance=output_noise_covariance,
    )


def get_capacity_from_average_output_power(
    s: np.ndarray,
    *,
    average_output_power: float,
    noise: float | None = None,
    n_sources: int,
    n_outputs: int,
    time_resolution: float = 1.0,
    output_noise_covariance: np.ndarray | None = None,
) -> float:
    """Return water-filled capacity from observed output signal/noise levels."""
    return get_capacity(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        average_output_power=average_output_power,
        noise=noise,
        time_resolution=time_resolution,
        output_noise_covariance=output_noise_covariance,
    )
