"""
Parameter sweep visualization utilities.
"""

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.core import get_bitrate, noise_floor_heuristic
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Literal
import os

os.makedirs("plots", exist_ok=True)

def normalize_singular_values(s: np.ndarray, params: Parameters, method: Literal["sqrtN", "s0"] = "s0") -> np.ndarray:
    if method == "s0":
        return s / s[0]
    elif method == "sqrtN":
        matrix_size = getattr(params, "matrix_size", None)
        if matrix_size is not None:
            Ninput = matrix_size[1]
            Noutput = matrix_size[0]
        else:
            Ninput = getattr(params, "num_brain_grid_points", None)
            if Ninput is None:
                # Try to get source_spacing_mm and generate grid
                source_spacing_mm = getattr(params, "source_spacing_mm", None)
                if source_spacing_mm is not None:
                    from guti.core import get_grid_positions
                    grid_positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
                    Ninput = len(grid_positions)
                else:
                    raise ValueError("Cannot normalize: missing matrix_size, num_brain_grid_points, or source_spacing_mm in parameters.")
            Noutput = getattr(params, "num_sensors", None)
            if Noutput is None:
                raise ValueError("Cannot normalize: missing num_sensors in parameters.")
        return s / np.sqrt(Ninput * Noutput)
    else:
        raise ValueError(f"Invalid normalization method: {method}")


def get_normalized_variants(modality_name: str, param_key: str, constant_params: Optional[Parameters] = None, normalization_method: Literal["sqrtN", "s0"] = "sqrtN"):
    """
    Get sorted variants with normalized singular values.

    Returns:
        list of tuples: (variant_dict, normalized_singular_values)
    """
    if constant_params is None:
        constant_params = Parameters()

    # list_svd_variants will filter and sort by param_key
    sorted_variants = list_svd_variants(
        modality_name,
        constant_params=constant_params,
        sort_by=param_key
    )


    # Normalize all singular values
    normalized_svs = []
    for _, v in sorted_variants:
        s_normalized = normalize_singular_values(v["s"], v["params"], method=normalization_method)
        normalized_svs.append((v, s_normalized))

    return normalized_svs


def plot_parameter_sweep_spectra(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6),
    ylim: tuple = (1e-5, 1e1),
    normalization_method: Literal["sqrtN", "s0"] = "sqrtN"
):
    normalized_svs = get_normalized_variants(modality_name, param_key, constant_params, normalization_method)

    if not normalized_svs:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    # Find the global maximum (first) singular value across all normalized variants
    max_sv = max(s_normalized[0] for _, s_normalized in normalized_svs)

    param_values = [getattr(v["params"], param_key) for v, _ in normalized_svs]
    min_val, max_val = min(param_values), max(param_values)
    # colors = plt.cm.viridis((np.array(param_values) - min_val) / (max_val - min_val))
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(normalized_svs))]

    plt.figure(figsize=figsize)
    for (v, s_normalized), color in zip(normalized_svs, colors):
        params = v["params"]
        param_value = getattr(params, param_key)
        s = s_normalized / max_sv  # Normalize by largest singular value across all params

        plt.plot(
            np.arange(1, len(s) + 1),
            s,
            label=f"{param_key}={param_value}",
            color=color
        )

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value (normalized)")
    plt.title(f"Singular value spectra - {param_key} sweep\n{modality_name}")
    plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/spectra.png")
    plt.show()


def plot_first_singular_value_vs_parameter(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6)
):
    normalized_svs = get_normalized_variants(modality_name, param_key, constant_params)

    if not normalized_svs:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    first_singular_values = []
    for v, s_normalized in normalized_svs:
        params = v["params"]
        param_value = getattr(params, param_key)
        first_singular_values.append((param_value, s_normalized[0]))

    param_values, s1_values = zip(*first_singular_values)

    plt.figure(figsize=figsize)
    plt.plot(param_values, s1_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_key)
    plt.ylabel('First Singular Value')
    plt.title(f'Maximum Gain vs {param_key}\n{modality_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/first_sv.png")
    plt.show()


def plot_bitrate_vs_parameter(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6),
    time_resolution: float = 1.0,
    snr: float = 10.0
):
    normalized_svs = get_normalized_variants(modality_name, param_key, constant_params)

    if not normalized_svs:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    param_values = []
    bitrates = []

    for v, s_normalized in normalized_svs:
        params = v["params"]
        param_value = getattr(params, param_key)
        n_sensors = params.num_sensors
        noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=snr, n_detectors=n_sensors)
        bitrate = get_bitrate(s_normalized, noise_level, time_resolution=time_resolution)
        param_values.append(param_value)
        bitrates.append(bitrate)

    plt.figure(figsize=figsize)
    plt.plot(param_values, bitrates, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_key)
    plt.ylabel('Bitrate (bits/s)')
    plt.title(f'Information Capacity vs {param_key}\n{modality_name}')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/bitrate.png")
    plt.show()


def plot_bitrate_vs_snr(
    modality_name: str,
    param_key: str,
    param_value: float,
    snr_values: np.ndarray,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6),
    time_resolution: float = 1.0
):
    """
    Plot bitrate vs SNR for a specific parameter value.

    Args:
        modality_name: Name of the imaging modality
        param_key: Parameter key being swept
        param_value: Specific value of the parameter to analyze
        snr_values: Array of SNR values to test
        constant_params: Fixed parameters
        figsize: Figure size
        time_resolution: Time resolution in seconds

    Example:
        >>> snrs = np.logspace(-1, 2, 50)  # SNR from 0.1 to 100
        >>> plot_bitrate_vs_snr("fnirs_analytical_cw", "grid_resolution_mm", 5.0, snrs)
    """
    if constant_params is None:
        constant_params = Parameters()

    # Get the variant with this parameter value
    normalized_svs = get_normalized_variants(modality_name, param_key, constant_params)

    if not normalized_svs:
        print(f"No variants found for {modality_name} with given parameters")
        return

    # Find the variant matching our param_value
    matching_variant = None
    for v, s_normalized in normalized_svs:
        if getattr(v["params"], param_key) == param_value:
            matching_variant = (v, s_normalized)
            break

    if matching_variant is None:
        print(f"No variant found with {param_key}={param_value}")
        return

    v, s_normalized = matching_variant
    params = v["params"]
    n_sensors = params.num_sensors

    # Compute bitrates for each SNR
    bitrates = []
    for snr in snr_values:
        noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=snr, n_detectors=n_sensors)
        bitrate = get_bitrate(s_normalized, noise_level, time_resolution=time_resolution)
        bitrates.append(bitrate)

    plt.figure(figsize=figsize)
    plt.plot(snr_values, bitrates, 'o-', linewidth=2, markersize=6)
    # plt.xscale('log')
    plt.xlabel('SNR')
    plt.ylabel('Bitrate (bits/s)')
    plt.title(f'Information Capacity vs SNR \n {modality_name} ({param_key}={param_value})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/bitrate_vs_snr.png")
    plt.show()


def show_sweep_results(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    plots: list = ["spectra", "first_sv", "bitrate"],
    figsize: tuple = (10, 6)
):
    """
    Show all scaling analysis plots for a parameter sweep.

    Example:
        >>> from guti.parameters import Parameters
        >>> from guti.scaling_utils import show_sweep_results
        >>>
        >>> show_sweep_results(
        ...     modality_name="fnirs_analytical_cw",
        ...     param_key="grid_resolution_mm",
        ...     constant_params=Parameters(num_sensors=400)
        ... )
    """
    if constant_params is None:
        constant_params = Parameters()

    print(f"\n{'='*60}")
    print(f"Parameter Sweep Results: {modality_name}")
    print(f"Varying: {param_key}")
    print(f"Constant parameters: {constant_params}")
    print(f"{'='*60}\n")

    variants = list_svd_variants(modality_name, constant_params=constant_params, sort_by=param_key)
    print(f"Found {len(variants)} variants:")
    for k, v in variants.items():
        print(f"  {k}: {v['params']}")
    print()

    if "spectra" in plots:
        plot_parameter_sweep_spectra(modality_name, param_key, constant_params, figsize)

    if "first_sv" in plots:
        plot_first_singular_value_vs_parameter(modality_name, param_key, constant_params, figsize)

    if "bitrate" in plots:
        plot_bitrate_vs_parameter(modality_name, param_key, constant_params, figsize)
