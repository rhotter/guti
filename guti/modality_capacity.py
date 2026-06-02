"""Single source of truth for the modality bitrate/capacity algorithm.

This is the algorithm reported in the README modality summary table and the
website scaling charts: anchor the input power to the typical output signal,
use the modality's detector noise (with the saved spherical-Johnson covariance
when available, scalar diagonal otherwise) and an output temporal power-law
spectrum, then compute the achievable bitrate and the water-filled capacity from
a saved SVD spectrum.

``scripts/plot_modality_convergence.py`` (which writes the convergence metrics the
README summary reads) and ``scripts/export_svd_json.py`` (which feeds the website
charts) both call :func:`compute_bitrate_capacity`, so the two cannot drift.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from guti.capacity import (
    default_output_frequency_spectrum_kwargs,
    frequency_spectrum_kwargs_from_params,
    get_bitrate,
    get_capacity,
    total_input_power_from_average_output_power,
)
from guti.core import get_grid_positions
from guti.noise_models import (
    compute_average_output_power,
    compute_output_noise_std,
    get_noise_model,
    scale_singular_values_for_capacity,
)
from guti.parameters import Parameters

# --- Ultrasound 2 MHz RBC-backscatter constants (ported from the README summary) ---
_US_RBC_FREQ_HZ = 2_000_000.0
_US_RBC_REFERENCE_SVD_FREQ_HZ = 50_000.0
_US_RBC_RATE_BANDWIDTH_HZ = 1.0
_US_RBC_EXTERNAL_PRESSURE_PA = 1_000_000.0
_US_RBC_SKULL_TRANSMISSION_2MHZ = 0.03
_US_RBC_BSC_10MHZ_CM_INV_SR_INV = 3e-5
_US_RBC_CEREBRAL_BLOOD_VOLUME = 0.03
_US_RBC_VOXEL_VOLUME_MM3 = 24.0
_US_RBC_RANGE_M = 0.10
_US_SOUND_SPEED_M_S = 1540.0
_US_BRAIN_DEPTH_M = 0.150
_US_BODY_TEMP_K = 310.0
_US_K_B = 1.380649e-23
_US_TISSUE_DENSITY_KG_M3 = 1000.0
_US_ELECTRONIC_RESISTANCE_OHM = 50.0
_US_RX_SENSITIVITY_V_PA = 1e-3
_US_SCALP_AREA_MM2 = 2.0 * math.pi * 92.0**2
_US_N_SENSORS = 6000


def _us_rbc_noise_pressure_pa() -> float:
    prf_hz = _US_SOUND_SPEED_M_S / (2.0 * _US_BRAIN_DEPTH_M)
    noise_bandwidth_hz = _US_RBC_FREQ_HZ * _US_RBC_RATE_BANDWIDTH_HZ / prf_hz
    element_area_m2 = _US_SCALP_AREA_MM2 * 1e-6 / _US_N_SENSORS
    wavelength_m = _US_SOUND_SPEED_M_S / _US_RBC_FREQ_HZ
    mode_count = element_area_m2 * 2.0 * math.pi / wavelength_m**2
    acoustic_power_w = mode_count * _US_K_B * _US_BODY_TEMP_K * noise_bandwidth_hz
    acoustic_pressure_pa = math.sqrt(
        (acoustic_power_w / element_area_m2) * _US_TISSUE_DENSITY_KG_M3 * _US_SOUND_SPEED_M_S
    )
    electronic_pressure_pa = (
        math.sqrt(4.0 * _US_K_B * _US_BODY_TEMP_K * _US_ELECTRONIC_RESISTANCE_OHM * noise_bandwidth_hz)
        / _US_RX_SENSITIVITY_V_PA
    )
    return math.sqrt(acoustic_pressure_pa**2 + electronic_pressure_pa**2)


def _us_rbc_output_amplitude_pa() -> float:
    freq_mhz = _US_RBC_FREQ_HZ / 1e6
    bsc_blood_cm = _US_RBC_BSC_10MHZ_CM_INV_SR_INV * (freq_mhz / 10.0) ** 4
    bsc_brain_m = bsc_blood_cm * _US_RBC_CEREBRAL_BLOOD_VOLUME * 100.0
    voxel_volume_m3 = _US_RBC_VOXEL_VOLUME_MM3 * 1e-9
    ratio = (
        _US_RBC_SKULL_TRANSMISSION_2MHZ**2
        * math.sqrt(bsc_brain_m * voxel_volume_m3)
        / _US_RBC_RANGE_M
    )
    return ratio * _US_RBC_EXTERNAL_PRESSURE_PA


def _us_rbc_bitrate_capacity(
    s_capacity: np.ndarray,
    n_sources: int,
    n_outputs: int,
    svd_frequency_hz: float = _US_RBC_REFERENCE_SVD_FREQ_HZ,
) -> dict[str, Any]:
    """README 2 MHz RBC-backscatter US path with λ³ spatial scaling.

    The SVD spectrum is computed at ``svd_frequency_hz`` (the variant's own
    frequency); the spatial mode count is scaled to the 2 MHz imaging frequency by
    (f_image / f_svd)³, so a frequency sweep stays referenced to the same 2 MHz
    imaging target instead of a hardcoded 50 kHz.
    """
    output_amplitude_pa = _us_rbc_output_amplitude_pa()
    noise_pressure_pa = _us_rbc_noise_pressure_pa()
    time_resolution_s = 1.0 / _US_RBC_RATE_BANDWIDTH_HZ
    total_input_power = total_input_power_from_average_output_power(
        s_capacity, average_output_power=output_amplitude_pa**2,
        n_sources=n_sources, n_outputs=n_outputs,
    )
    lambda3_scale = (_US_RBC_FREQ_HZ / float(svd_frequency_hz)) ** 3
    bitrate = get_bitrate(
        s_capacity, n_sources=n_sources, total_input_power=total_input_power,
        noise=noise_pressure_pa, time_resolution=time_resolution_s,
    ) * lambda3_scale
    capacity = get_capacity(
        s_capacity[s_capacity > 0], total_input_power=total_input_power,
        noise=noise_pressure_pa, time_resolution=time_resolution_s,
    ) * lambda3_scale
    return {
        "bitrate_bits_per_s": float(bitrate),
        "channel_capacity_bits_per_s": float(capacity),
        "output_amplitude": float(output_amplitude_pa),
        "output_noise": float(noise_pressure_pa),
        "output_snr": float(output_amplitude_pa / noise_pressure_pa),
        "bandwidth_hz": _US_RBC_RATE_BANDWIDTH_HZ,
        "time_resolution_s": time_resolution_s,
        "total_input_power": float(total_input_power),
        "noise_model_type": "scalar_iid",
    }


def modality_bandwidth_hz(noise_model: str, params: Parameters) -> float:
    """Capacity bandwidth for a modality (US scales its band with frequency)."""
    model = get_noise_model(noise_model)
    if model.canonical_name == "us_analytical":
        freq = float(params.frequency_hz or 50_000.0)
        return float(model.reference_bandwidth_hz * freq / 50_000.0)
    return float(model.reference_bandwidth_hz)


def infer_shape(
    noise_model: str,
    params: Parameters,
    source_orientations: int = 1,
) -> tuple[int, int, int]:
    """Return ``(n_outputs, n_sources, n_voxels)`` for a variant."""
    if params.matrix_size is not None:
        n_outputs, n_sources = params.matrix_size
        n_voxels = int(params.num_brain_grid_points or n_sources)
        return int(n_outputs), int(n_sources), n_voxels

    if noise_model.startswith("meg_"):
        if params.num_sensors is None or params.source_spacing_mm is None:
            raise ValueError("MEG variant needs num_sensors and source_spacing_mm")
        n_voxels = len(get_grid_positions(grid_spacing_mm=float(params.source_spacing_mm)))
        return 3 * int(params.num_sensors), 3 * n_voxels, n_voxels

    if noise_model.startswith("eeg"):
        if params.num_sensors is None or params.num_brain_grid_points is None:
            raise ValueError("EEG variant needs num_sensors and num_brain_grid_points")
        n_voxels = int(params.num_brain_grid_points)
        return int(params.num_sensors), source_orientations * n_voxels, n_voxels

    if params.num_sensors is None or params.num_brain_grid_points is None:
        raise ValueError("variant needs num_sensors and num_brain_grid_points")
    n_voxels = int(params.num_brain_grid_points)
    return int(params.num_sensors), n_voxels, n_voxels


def compute_bitrate_capacity(
    singular_values: np.ndarray,
    params: Parameters,
    *,
    noise_model: str,
    source_orientations: int = 1,
    s_noise_normalized: np.ndarray | None = None,
    tier: str = "today",
) -> dict[str, Any]:
    """Bitrate + water-filled capacity for one SVD spectrum under the README method.

    ``s_noise_normalized`` is the covariance-whitened spectrum (σ/noise) saved by the
    Johnson-covariance sweeps. When present the noise is folded in already (noise=1),
    giving the full spherical-Johnson result; otherwise a scalar detector-noise
    diagonal is used. ``tier`` selects today vs. fundamental detector noise.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    n_outputs, n_sources, n_voxels = infer_shape(noise_model, params, source_orientations)
    n_sensors = int(params.num_sensors)

    s_capacity = scale_singular_values_for_capacity(
        s, noise_model, params=params, voxel_size_mm=params.grid_resolution_mm
    )

    if noise_model == "us_analytical":
        # Fixed 50 kHz → 2 MHz λ³ extrapolation (the calibrated proxy frequency).
        # Frequency is NOT a meaningful sweep axis here: the swept variants are
        # resolution-capped, so scaling λ³ off each variant's own frequency
        # produces a spurious slope. US sweeps are restricted to 50 kHz upstream.
        result = _us_rbc_bitrate_capacity(s_capacity, n_sources, n_outputs)
        result.update({"n_sensors": n_sensors, "n_voxels": n_voxels,
                       "n_outputs": n_outputs, "n_sources": n_sources})
        return result

    bandwidth_hz = modality_bandwidth_hz(noise_model, params)
    time_resolution_s = 1.0 / bandwidth_hz
    average_output_power = compute_average_output_power(noise_model)
    output_amplitude = math.sqrt(average_output_power)
    output_noise = compute_output_noise_std(
        noise_model,
        n_sensors=n_sensors,
        bandwidth_hz=bandwidth_hz,
        tier=tier,
        frequency_hz=params.frequency_hz,
        voxel_size_mm=params.grid_resolution_mm,
        tr_s=params.time_resolution,
        bold_contrast=params.bold_contrast,
        bold_snr=params.bold_snr,
    )
    spectrum_kwargs = frequency_spectrum_kwargs_from_params(
        params
    ) or default_output_frequency_spectrum_kwargs(noise_model)
    total_input_power = total_input_power_from_average_output_power(
        s_capacity,
        average_output_power=average_output_power,
        n_sources=n_sources,
        n_outputs=n_outputs,
    )

    if s_noise_normalized is not None and tier == "today":
        # Saved covariance-whitened spectrum already folds in (today) noise.
        snn = np.asarray(s_noise_normalized, dtype=np.float64)
        bitrate = get_bitrate(
            snn, n_sources=n_sources, total_input_power=total_input_power,
            noise=1.0, time_resolution=time_resolution_s, **spectrum_kwargs,
        )
        capacity = get_capacity(
            snn[snn > 0], total_input_power=total_input_power,
            noise=1.0, time_resolution=time_resolution_s, **spectrum_kwargs,
        )
        noise_model_type = "spatial_covariance"
    else:
        bitrate = get_bitrate(
            s_capacity, n_sources=n_sources, total_input_power=total_input_power,
            noise=output_noise, time_resolution=time_resolution_s, **spectrum_kwargs,
        )
        capacity = get_capacity(
            s_capacity[s_capacity > 0], total_input_power=total_input_power,
            noise=output_noise, time_resolution=time_resolution_s, **spectrum_kwargs,
        )
        noise_model_type = "scalar_iid"

    return {
        "bitrate_bits_per_s": float(bitrate),
        "channel_capacity_bits_per_s": float(capacity),
        "output_amplitude": float(output_amplitude),
        "output_noise": float(output_noise),
        "output_snr": float(output_amplitude / output_noise),
        "bandwidth_hz": bandwidth_hz,
        "time_resolution_s": time_resolution_s,
        "total_input_power": float(total_input_power),
        "noise_model_type": noise_model_type,
        "n_sensors": n_sensors,
        "n_voxels": n_voxels,
        "n_outputs": n_outputs,
        "n_sources": n_sources,
    }
