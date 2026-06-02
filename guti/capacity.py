"""Capacity and bitrate utilities for linear imaging channels."""

from __future__ import annotations

import numpy as np


DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ = 1.0
DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ = 1.0
DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ = 100.0

_OUTPUT_FREQUENCY_SPECTRUM_KWARGS = frozenset(
    {
        "output_frequency_spectrum",
        "output_frequency_bin_width",
        "output_power_law_beta",
        "output_power_law_min_freq_hz",
        "output_power_law_max_freq_hz",
        "output_power_law_bin_width_hz",
    }
)


_NOISE_FREQUENCY_SPECTRUM_KWARGS = frozenset(
    {
        "noise_frequency_spectrum",
        "noise_frequency_bin_width",
        "noise_power_law_beta",
        "noise_power_law_min_freq_hz",
        "noise_power_law_max_freq_hz",
        "noise_power_law_bin_width_hz",
    }
)


def _as_spectrum(s: np.ndarray, *, name: str = "s") -> np.ndarray:
    spectrum = np.asarray(s, dtype=float)
    if spectrum.ndim != 1:
        raise ValueError(f"{name} must be a 1D array of singular values")
    if np.any(spectrum < 0):
        raise ValueError(f"{name} must contain non-negative singular values")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError(f"{name} must contain finite singular values")
    return spectrum


def _as_frequency_spectrum(
    spectrum: np.ndarray,
    *,
    name: str,
    allow_zero: bool = True,
) -> np.ndarray:
    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one bin")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if allow_zero:
        if np.any(values < 0.0):
            raise ValueError(f"{name} must contain non-negative values")
        if not np.any(values > 0.0):
            raise ValueError(f"{name} must contain at least one positive value")
    else:
        if np.any(values <= 0.0):
            raise ValueError(f"{name} must contain positive values")
    return values


def _as_positive_finite(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be positive and finite")
    return value


def power_law_frequency_spectrum(
    *,
    beta: float,
    min_freq_hz: float,
    max_freq_hz: float,
    freq_bin_width_hz: float,
) -> np.ndarray:
    """Return a 1D power-law frequency spectrum sampled on uniform bins.

    The returned values are proportional to ``1 / f**beta`` at bin centers.
    They are intentionally not normalized; bitrate/capacity callers normalize
    the spectrum to the supplied total output/input power.
    """
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    min_freq_hz = _as_positive_finite(min_freq_hz, name="min_freq_hz")
    max_freq_hz = _as_positive_finite(max_freq_hz, name="max_freq_hz")
    freq_bin_width_hz = _as_positive_finite(
        freq_bin_width_hz,
        name="freq_bin_width_hz",
    )
    if max_freq_hz <= min_freq_hz:
        raise ValueError("max_freq_hz must be greater than min_freq_hz")

    n_bins = int(np.ceil((max_freq_hz - min_freq_hz) / freq_bin_width_hz))
    freqs = min_freq_hz + (np.arange(n_bins, dtype=float) + 0.5) * freq_bin_width_hz
    freqs = np.minimum(freqs, max_freq_hz)
    return freqs ** (-beta)


def default_output_frequency_spectrum_kwargs(
    modality: str,
) -> dict[str, float | str]:
    """Return default output-spectrum kwargs for modality-level bitrate.

    Direct neural field modalities use a power-law output-power spectrum over the
    conventional 1--100 Hz band. Hemodynamic modalities already route through
    the HRF temporal spectrum, and the remaining modalities keep the historical
    single-band behavior unless a caller supplies an explicit spectrum.
    """
    if modality == "eeg":
        return {
            "output_power_law_beta": 1.4,
            "output_power_law_min_freq_hz": DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            "output_power_law_max_freq_hz": DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            "output_power_law_bin_width_hz": DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        }
    if modality in {"meg_opm", "meg_squid"}:
        return {
            "output_power_law_beta": 1.7,
            "output_power_law_min_freq_hz": DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            "output_power_law_max_freq_hz": DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            "output_power_law_bin_width_hz": DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        }
    return {}


def has_output_frequency_spectrum_kwargs(kwargs: dict) -> bool:
    """Return True if a kwargs dict contains explicit output-spectrum settings."""
    return any(
        key in kwargs and kwargs[key] is not None
        for key in _OUTPUT_FREQUENCY_SPECTRUM_KWARGS
    )


def output_frequency_spectrum_kwargs_from_params(params) -> dict[str, float | str]:
    """Extract output-spectrum kwargs from a ``Parameters``-like object."""
    spectrum_type = getattr(params, "output_spectrum_type", None)
    if spectrum_type is None:
        return {}
    if spectrum_type != "power_law":
        raise ValueError(f"Unsupported output_spectrum_type {spectrum_type!r}")
    return {
        "output_power_law_beta": getattr(params, "output_spectrum_beta"),
        "output_power_law_min_freq_hz": getattr(params, "output_spectrum_min_freq_hz"),
        "output_power_law_max_freq_hz": getattr(params, "output_spectrum_max_freq_hz"),
        "output_power_law_bin_width_hz": getattr(
            params,
            "output_spectrum_bin_width_hz",
        ),
    }


def noise_frequency_spectrum_kwargs_from_params(params) -> dict[str, float | str]:
    """Extract noise-spectrum kwargs from a ``Parameters``-like object."""
    spectrum_type = getattr(params, "noise_spectrum_type", None)
    if spectrum_type is None:
        return {}
    if spectrum_type != "power_law":
        raise ValueError(f"Unsupported noise_spectrum_type {spectrum_type!r}")
    return {
        "noise_power_law_beta": getattr(params, "noise_spectrum_beta"),
        "noise_power_law_min_freq_hz": getattr(params, "noise_spectrum_min_freq_hz"),
        "noise_power_law_max_freq_hz": getattr(params, "noise_spectrum_max_freq_hz"),
        "noise_power_law_bin_width_hz": getattr(
            params,
            "noise_spectrum_bin_width_hz",
        ),
    }


def frequency_spectrum_kwargs_from_params(params) -> dict[str, float | str]:
    """Extract output/noise frequency-spectrum kwargs from ``Parameters``."""
    return {
        **output_frequency_spectrum_kwargs_from_params(params),
        **noise_frequency_spectrum_kwargs_from_params(params),
    }


def _resolve_frequency_spectrum(
    *,
    spectrum: np.ndarray | None,
    frequency_bin_width: float | None,
    power_law_beta: float | None,
    power_law_min_freq_hz: float | None,
    power_law_max_freq_hz: float | None,
    power_law_bin_width_hz: float | None,
    spectrum_name: str,
    bin_width_name: str,
) -> tuple[np.ndarray | None, float | None]:
    has_array = spectrum is not None
    has_power_law = any(
        value is not None
        for value in (
            power_law_beta,
            power_law_min_freq_hz,
            power_law_max_freq_hz,
            power_law_bin_width_hz,
        )
    )
    if has_array and has_power_law:
        raise ValueError(
            f"Pass either {spectrum_name} or {spectrum_name} power-law args, not both"
        )
    if has_array:
        if frequency_bin_width is None:
            raise ValueError(f"{bin_width_name} is required with {spectrum_name}")
        return (
            _as_frequency_spectrum(spectrum, name=spectrum_name),
            _as_positive_finite(frequency_bin_width, name=bin_width_name),
        )
    if not has_power_law:
        if frequency_bin_width is not None:
            raise ValueError(f"{bin_width_name} requires {spectrum_name}")
        return None, None

    if (
        power_law_beta is None
        or power_law_min_freq_hz is None
        or power_law_max_freq_hz is None
        or power_law_bin_width_hz is None
    ):
        raise ValueError(
            f"{spectrum_name} power law requires beta, min_freq_hz, "
            "max_freq_hz, and bin_width_hz"
        )
    if frequency_bin_width is not None and not np.isclose(
        float(frequency_bin_width),
        float(power_law_bin_width_hz),
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError(
            f"{bin_width_name} must match {spectrum_name} power-law bin width"
        )
    return (
        power_law_frequency_spectrum(
            beta=power_law_beta,
            min_freq_hz=power_law_min_freq_hz,
            max_freq_hz=power_law_max_freq_hz,
            freq_bin_width_hz=power_law_bin_width_hz,
        ),
        _as_positive_finite(power_law_bin_width_hz, name=bin_width_name),
    )


def _normalized_frequency_power_weights(
    spectrum: np.ndarray,
    *,
    frequency_bin_width: float,
) -> np.ndarray:
    weighted = _as_frequency_spectrum(spectrum, name="frequency spectrum") * float(
        frequency_bin_width
    )
    total = float(np.sum(weighted))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("frequency spectrum integral must be positive and finite")
    return weighted / total


def _scalar_noise_std_for_frequency_bin(
    noise: float,
    *,
    frequency_bin_width: float,
    time_resolution: float,
) -> float:
    """Convert a full-band noise std to a frequency-bin noise std."""
    noise = _as_positive_finite(noise, name="noise")
    frequency_bin_width = _as_positive_finite(
        frequency_bin_width,
        name="output_frequency_bin_width",
    )
    time_resolution = _as_positive_finite(time_resolution, name="time_resolution")
    return float(noise * np.sqrt(frequency_bin_width * time_resolution))


def _noise_covariance_for_frequency_bin(
    output_noise_covariance: np.ndarray,
    *,
    frequency_bin_width: float,
    time_resolution: float,
) -> np.ndarray:
    """Convert a full-band noise covariance to a frequency-bin covariance."""
    frequency_bin_width = _as_positive_finite(
        frequency_bin_width,
        name="output_frequency_bin_width",
    )
    time_resolution = _as_positive_finite(time_resolution, name="time_resolution")
    return np.asarray(output_noise_covariance, dtype=float) * (
        frequency_bin_width * time_resolution
    )


def _resolve_spectral_bins(
    *,
    output_frequency_spectrum: np.ndarray | None,
    output_frequency_bin_width: float | None,
    noise_frequency_spectrum: np.ndarray | None,
    noise_frequency_bin_width: float | None,
    output_power_law_beta: float | None,
    output_power_law_min_freq_hz: float | None,
    output_power_law_max_freq_hz: float | None,
    output_power_law_bin_width_hz: float | None,
    noise_power_law_beta: float | None,
    noise_power_law_min_freq_hz: float | None,
    noise_power_law_max_freq_hz: float | None,
    noise_power_law_bin_width_hz: float | None,
    noise: float | None,
    output_noise_covariance: np.ndarray | None,
    time_resolution: float,
) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    output_spectrum, output_df = _resolve_frequency_spectrum(
        spectrum=output_frequency_spectrum,
        frequency_bin_width=output_frequency_bin_width,
        power_law_beta=output_power_law_beta,
        power_law_min_freq_hz=output_power_law_min_freq_hz,
        power_law_max_freq_hz=output_power_law_max_freq_hz,
        power_law_bin_width_hz=output_power_law_bin_width_hz,
        spectrum_name="output_frequency_spectrum",
        bin_width_name="output_frequency_bin_width",
    )
    noise_spectrum, noise_df = _resolve_frequency_spectrum(
        spectrum=noise_frequency_spectrum,
        frequency_bin_width=noise_frequency_bin_width,
        power_law_beta=noise_power_law_beta,
        power_law_min_freq_hz=noise_power_law_min_freq_hz,
        power_law_max_freq_hz=noise_power_law_max_freq_hz,
        power_law_bin_width_hz=noise_power_law_bin_width_hz,
        spectrum_name="noise_frequency_spectrum",
        bin_width_name="noise_frequency_bin_width",
    )
    if output_spectrum is None and noise_spectrum is None:
        return None
    if noise_spectrum is not None and output_noise_covariance is not None:
        raise ValueError(
            "noise_frequency_spectrum is not supported with output_noise_covariance"
        )

    if output_spectrum is None:
        assert noise_spectrum is not None
        output_spectrum = np.ones_like(noise_spectrum)
        output_df = noise_df
    if noise_spectrum is not None:
        if output_spectrum.shape != noise_spectrum.shape:
            raise ValueError(
                "output_frequency_spectrum and noise_frequency_spectrum must "
                "have the same number of bins"
            )
        if noise_df is None:
            noise_df = output_df
        if not np.isclose(float(output_df), float(noise_df), rtol=1e-12, atol=0.0):
            raise ValueError(
                "output_frequency_bin_width and noise_frequency_bin_width must match"
            )

    assert output_df is not None
    output_weights = _normalized_frequency_power_weights(
        output_spectrum,
        frequency_bin_width=output_df,
    )
    noise_per_bin = None
    if noise_spectrum is not None:
        noise_levels = _as_frequency_spectrum(
            noise_spectrum,
            name="noise_frequency_spectrum",
            allow_zero=False,
        )
        if noise is None:
            noise_per_bin = noise_levels
        else:
            bin_noise = _scalar_noise_std_for_frequency_bin(
                noise,
                frequency_bin_width=float(output_df),
                time_resolution=time_resolution,
            )
            noise_per_bin = bin_noise * noise_levels

    return output_weights, noise_per_bin, float(output_df)


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


def _get_bitrate_flat(
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
    output_frequency_spectrum: np.ndarray | None = None,
    output_frequency_bin_width: float | None = None,
    noise_frequency_spectrum: np.ndarray | None = None,
    noise_frequency_bin_width: float | None = None,
    output_power_law_beta: float | None = None,
    output_power_law_min_freq_hz: float | None = None,
    output_power_law_max_freq_hz: float | None = None,
    output_power_law_bin_width_hz: float | None = None,
    noise_power_law_beta: float | None = None,
    noise_power_law_min_freq_hz: float | None = None,
    noise_power_law_max_freq_hz: float | None = None,
    noise_power_law_bin_width_hz: float | None = None,
) -> float:
    """Return i.i.d.-input bitrate for a linear Gaussian channel.

    The total input power is spread uniformly over ``n_sources`` source
    channels. By default ``noise`` is the per-output-channel i.i.d. noise
    standard deviation and ``s`` may be singular values or the channel matrix.
    If ``output_noise_covariance`` is provided, ``s`` must be the output-by-input
    channel matrix and the code uses the singular values of
    ``K_N^{-1/2} H``.

    If ``output_frequency_spectrum`` (or the output power-law args) is supplied,
    the resolved total power is interpreted as power integrated over all
    frequency bins. The output spectrum is normalized to power weights, each bin
    is evaluated with ``time_resolution = 1 / output_frequency_bin_width``, and
    the bin bitrates are summed. A supplied ``noise_frequency_spectrum`` gives
    absolute per-bin noise stds when ``noise`` is omitted, or per-bin
    multipliers on the scalar noise std when ``noise`` is supplied. Scalar noise
    and noise covariances are interpreted as full-band values over
    ``1 / time_resolution`` Hz and converted to each bin by
    ``sqrt(output_frequency_bin_width * time_resolution)`` for standard
    deviations, or ``output_frequency_bin_width * time_resolution`` for
    covariances.
    """
    spectral_bins = _resolve_spectral_bins(
        output_frequency_spectrum=output_frequency_spectrum,
        output_frequency_bin_width=output_frequency_bin_width,
        noise_frequency_spectrum=noise_frequency_spectrum,
        noise_frequency_bin_width=noise_frequency_bin_width,
        output_power_law_beta=output_power_law_beta,
        output_power_law_min_freq_hz=output_power_law_min_freq_hz,
        output_power_law_max_freq_hz=output_power_law_max_freq_hz,
        output_power_law_bin_width_hz=output_power_law_bin_width_hz,
        noise_power_law_beta=noise_power_law_beta,
        noise_power_law_min_freq_hz=noise_power_law_min_freq_hz,
        noise_power_law_max_freq_hz=noise_power_law_max_freq_hz,
        noise_power_law_bin_width_hz=noise_power_law_bin_width_hz,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
        time_resolution=time_resolution,
    )
    if spectral_bins is None:
        return _get_bitrate_flat(
            s,
            n_sources=n_sources,
            total_input_power=total_input_power,
            input_power_per_source=input_power_per_source,
            input_amplitude=input_amplitude,
            average_output_power=average_output_power,
            n_outputs=n_outputs,
            noise=noise,
            time_resolution=time_resolution,
            output_noise_covariance=output_noise_covariance,
        )

    _validate_optional_matrix_shape(s, n_sources=n_sources, n_outputs=n_outputs)
    if n_sources <= 0:
        raise ValueError("n_sources must be positive")

    output_weights, noise_per_bin, freq_bin_width = spectral_bins
    bin_noise_from_scalar = None
    bin_output_noise_covariance = output_noise_covariance
    if noise_per_bin is None:
        if noise is not None:
            bin_noise_from_scalar = _scalar_noise_std_for_frequency_bin(
                noise,
                frequency_bin_width=freq_bin_width,
                time_resolution=time_resolution,
            )
        elif output_noise_covariance is not None:
            bin_output_noise_covariance = _noise_covariance_for_frequency_bin(
                output_noise_covariance,
                frequency_bin_width=freq_bin_width,
                time_resolution=time_resolution,
            )
    resolved_total_input_power = resolve_total_input_power(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    total = 0.0
    for i, output_weight in enumerate(output_weights):
        bin_power = resolved_total_input_power * float(output_weight)
        bin_noise = (
            bin_noise_from_scalar
            if noise_per_bin is None
            else float(noise_per_bin[i])
        )
        total += _get_bitrate_flat(
            s,
            n_sources=n_sources,
            total_input_power=bin_power,
            n_outputs=n_outputs,
            noise=bin_noise,
            time_resolution=1.0 / freq_bin_width,
            output_noise_covariance=bin_output_noise_covariance,
        )
    return float(total)


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


def _get_capacity_flat(
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
    output_frequency_spectrum: np.ndarray | None = None,
    output_frequency_bin_width: float | None = None,
    noise_frequency_spectrum: np.ndarray | None = None,
    noise_frequency_bin_width: float | None = None,
    output_power_law_beta: float | None = None,
    output_power_law_min_freq_hz: float | None = None,
    output_power_law_max_freq_hz: float | None = None,
    output_power_law_bin_width_hz: float | None = None,
    noise_power_law_beta: float | None = None,
    noise_power_law_min_freq_hz: float | None = None,
    noise_power_law_max_freq_hz: float | None = None,
    noise_power_law_bin_width_hz: float | None = None,
) -> float:
    """Return water-filled channel capacity for a linear Gaussian channel.

    Frequency-spectrum arguments have the same semantics as
    :func:`get_bitrate`: resolved total power is normalized across output
    frequency bins and each bin is evaluated at
    ``time_resolution = 1 / output_frequency_bin_width`` before summing.
    """
    spectral_bins = _resolve_spectral_bins(
        output_frequency_spectrum=output_frequency_spectrum,
        output_frequency_bin_width=output_frequency_bin_width,
        noise_frequency_spectrum=noise_frequency_spectrum,
        noise_frequency_bin_width=noise_frequency_bin_width,
        output_power_law_beta=output_power_law_beta,
        output_power_law_min_freq_hz=output_power_law_min_freq_hz,
        output_power_law_max_freq_hz=output_power_law_max_freq_hz,
        output_power_law_bin_width_hz=output_power_law_bin_width_hz,
        noise_power_law_beta=noise_power_law_beta,
        noise_power_law_min_freq_hz=noise_power_law_min_freq_hz,
        noise_power_law_max_freq_hz=noise_power_law_max_freq_hz,
        noise_power_law_bin_width_hz=noise_power_law_bin_width_hz,
        noise=noise,
        output_noise_covariance=output_noise_covariance,
        time_resolution=time_resolution,
    )
    if spectral_bins is None:
        return _get_capacity_flat(
            s,
            total_input_power=total_input_power,
            input_power_per_source=input_power_per_source,
            input_amplitude=input_amplitude,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=noise,
            time_resolution=time_resolution,
            output_noise_covariance=output_noise_covariance,
        )

    _validate_optional_matrix_shape(s, n_sources=n_sources, n_outputs=n_outputs)
    output_weights, noise_per_bin, freq_bin_width = spectral_bins
    bin_noise_from_scalar = None
    bin_output_noise_covariance = output_noise_covariance
    if noise_per_bin is None:
        if noise is not None:
            bin_noise_from_scalar = _scalar_noise_std_for_frequency_bin(
                noise,
                frequency_bin_width=freq_bin_width,
                time_resolution=time_resolution,
            )
        elif output_noise_covariance is not None:
            bin_output_noise_covariance = _noise_covariance_for_frequency_bin(
                output_noise_covariance,
                frequency_bin_width=freq_bin_width,
                time_resolution=time_resolution,
            )
    resolved_total_input_power = resolve_total_input_power(
        s,
        n_sources=n_sources,
        n_outputs=n_outputs,
        total_input_power=total_input_power,
        input_power_per_source=input_power_per_source,
        input_amplitude=input_amplitude,
        average_output_power=average_output_power,
    )
    total = 0.0
    for i, output_weight in enumerate(output_weights):
        bin_power = resolved_total_input_power * float(output_weight)
        bin_noise = (
            bin_noise_from_scalar
            if noise_per_bin is None
            else float(noise_per_bin[i])
        )
        total += _get_capacity_flat(
            s,
            total_input_power=bin_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=bin_noise,
            time_resolution=1.0 / freq_bin_width,
            output_noise_covariance=bin_output_noise_covariance,
        )
    return float(total)


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
    **kwargs,
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
        **kwargs,
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
    **kwargs,
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
        **kwargs,
    )
