"""
Parameter sweep visualization utilities.
"""

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.capacity import total_input_power_from_average_output_power
from guti.hrf import get_modality_bitrate
from guti.noise_models import (
    compute_average_output_power,
    compute_output_noise_std,
    get_noise_model,
    scale_singular_values_for_capacity,
)
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
        # Forward models stored as voxel-integrated transfer functions
        # (J = density * voxel_volume, units mm^-1) have singular values that
        # scale as sqrt(voxel_volume) under grid refinement. Since the number of
        # grid points Ninput ~ V_brain / voxel_volume, normalizing by sqrt(Ninput)
        # would leave a residual factor ~ voxel_volume ~ grid^3, so spectra would
        # not overlap across a grid sweep. Use the quadrature weight
        # sqrt(voxel_volume) on the input side instead: it is grid-convergent and
        # constant across non-grid sweeps (where it reduces to the old behaviour
        # up to a fixed constant).
        voxel_volume_mm3 = getattr(params, "voxel_volume_mm3", None)
        input_scale = voxel_volume_mm3 if voxel_volume_mm3 is not None else Ninput
        return s / np.sqrt(input_scale * Noutput)
    else:
        raise ValueError(f"Invalid normalization method: {method}")


def infer_matrix_shape_for_capacity(
    modality_name: str,
    params: Parameters,
    n_singular_values: int,
) -> tuple[int, int]:
    if getattr(params, "matrix_size", None) is not None:
        n_outputs, n_sources = params.matrix_size
        return int(n_outputs), int(n_sources)

    if modality_name.startswith("meg_"):
        if params.num_sensors is None or params.source_spacing_mm is None:
            raise ValueError("MEG capacity needs num_sensors and source_spacing_mm")
        from guti.core import get_grid_positions

        n_outputs = 3 * int(params.num_sensors)
        n_sources = 3 * len(get_grid_positions(grid_spacing_mm=params.source_spacing_mm))
        return n_outputs, n_sources

    if modality_name.startswith("eeg_"):
        if params.num_sensors is None or params.num_brain_grid_points is None:
            raise ValueError("EEG capacity needs num_sensors and num_brain_grid_points")
        return int(params.num_sensors), 3 * int(params.num_brain_grid_points)

    if params.num_sensors is not None and params.num_brain_grid_points is not None:
        return int(params.num_sensors), int(params.num_brain_grid_points)

    raise ValueError(
        f"Cannot infer matrix shape for {modality_name}; singular values alone "
        f"only give min(n_outputs, n_sources)={n_singular_values}"
    )


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

    seen_param_values = {}
    plt.figure(figsize=figsize)
    seen_param_values = {}
    for (v, s_normalized), color in zip(normalized_svs, colors):
        params = v["params"]
        param_value = getattr(params, param_key)
        s = s_normalized / max_sv  # Normalize by largest singular value across all params

        # Determine if this is a duplicate
        is_duplicate = param_value in seen_param_values
        linestyle = '--' if is_duplicate else '-'

        # Track this parameter value
        if param_value not in seen_param_values:
            seen_param_values[param_value] = 0
        seen_param_values[param_value] += 1

        plt.plot(
            np.arange(1, len(s) + 1),
            s,
            label=f"{param_key}={param_value}",
            color=color,
            linestyle=linestyle
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
        model = get_noise_model(modality_name)
        n_sensors = params.num_sensors or model.reference_sensor_count
        freq = getattr(params, "frequency_hz", None)
        s_capacity = scale_singular_values_for_capacity(
            v["s"],
            modality_name,
            params=params,
        )
        n_outputs, n_sources = infer_matrix_shape_for_capacity(
            modality_name,
            params,
            len(s_capacity),
        )
        average_output_power = compute_average_output_power(modality_name)
        total_input_power = total_input_power_from_average_output_power(
            s_capacity,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
        )
        bitrate = get_modality_bitrate(
            s_capacity,
            modality_name,
            n_sources=n_sources,
            total_input_power=total_input_power,
            noise=compute_output_noise_std(
                modality_name,
                n_sensors=n_sensors,
                frequency_hz=freq,
            ),
            time_resolution=time_resolution,
            hrf_type=getattr(params, "hrf_type", None),
        )
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
        ...     modality_name="cw_fnirs",
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
