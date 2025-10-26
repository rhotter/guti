"""
Parameter sweep visualization utilities.
"""

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.core import get_bitrate, noise_floor_heuristic
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_parameter_sweep_spectra(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6),
    ylim: tuple = (1e-5, 1e1)
):
    if constant_params is None:
        constant_params = Parameters()

    variants = list_svd_variants(modality_name, constant_params=constant_params)

    if not variants:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    sorted_variants = sorted(
        variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
    )

    # Find the global maximum singular value across all variants
    max_sv = 0.0
    for k, v in sorted_variants:
        cur_max = np.max(v["s"])
        if cur_max > max_sv:
            max_sv = cur_max
    if max_sv == 0.0:
        max_sv = 1.0  # Avoid division by zero

    param_values = [getattr(v["params"], param_key) for k, v in sorted_variants]
    min_val, max_val = min(param_values), max(param_values)
    colors = plt.cm.viridis((np.array(param_values) - min_val) / (max_val - min_val))

    plt.figure(figsize=figsize)
    for (k, v), color in zip(sorted_variants, colors):
        params = v["params"]
        s = v["s"] / max_sv  # Normalize by largest singular value across all params
        param_value = getattr(params, param_key)

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
    plt.show()


def plot_first_singular_value_vs_parameter(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6)
):
    if constant_params is None:
        constant_params = Parameters()

    variants = list_svd_variants(modality_name, constant_params=constant_params)

    if not variants:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    sorted_variants = sorted(
        variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
    )

    first_singular_values = []
    for k, v in sorted_variants:
        params = v["params"]
        s = v["s"]
        param_value = getattr(params, param_key)
        first_singular_values.append((param_value, s[0]))

    param_values, s1_values = zip(*first_singular_values)

    plt.figure(figsize=figsize)
    plt.plot(param_values, s1_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_key)
    plt.ylabel('First Singular Value')
    plt.title(f'Maximum Gain vs {param_key}\n{modality_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bitrate_vs_parameter(
    modality_name: str,
    param_key: str,
    constant_params: Optional[Parameters] = None,
    figsize: tuple = (10, 6),
    time_resolution: float = 1.0
):
    if constant_params is None:
        constant_params = Parameters()

    variants = list_svd_variants(modality_name, constant_params=constant_params)

    if not variants:
        print(f"No variants found for {modality_name} with given constant parameters")
        return

    sorted_variants = sorted(
        variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
    )

    param_values = []
    bitrates = []

    for k, v in sorted_variants:
        params = v["params"]
        s = v["s"]
        param_value = getattr(params, param_key)
        n_sensors = params.num_sensors
        noise_level = noise_floor_heuristic(s, heuristic="power", n_detectors=n_sensors)

        bitrate = get_bitrate(s, noise_level, time_resolution=time_resolution, n_detectors=n_sensors)

        param_values.append(param_value)
        bitrates.append(bitrate)

    plt.figure(figsize=figsize)
    plt.plot(param_values, bitrates, 'o-', linewidth=2, markersize=8)
    plt.xlabel(param_key)
    plt.ylabel('Bitrate (bits/s)')
    plt.title(f'Information Capacity vs {param_key}\n{modality_name}')
    plt.grid(True)
    plt.tight_layout()
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

    variants = list_svd_variants(modality_name, constant_params=constant_params)
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
