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

    gains = _noise_normalized_spectrum(
        s,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
    )
    allocation = np.zeros_like(gains)
    active = gains > 0
    if total_input_power == 0 or not np.any(active):
        return allocation

    floors = 1.0 / (gains[active] ** 2)
    lo = float(np.min(floors))
    hi = float(np.max(floors) + total_input_power)

    # Monotone solve for mu where sum(max(0, mu - floor_i)) = P_total.
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        power = float(np.sum(np.maximum(0.0, mu - floors)))
        if power < total_input_power:
            lo = mu
        else:
            hi = mu

    mu = hi
    allocation[active] = np.maximum(0.0, mu - floors)
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
