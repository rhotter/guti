"""
Check harness for comparing the main-branch analytical bitrate computation
against the current branch's SLQ approximation on a reduced free-field problem.

This avoids importing ``guti.modalities.us.analytical`` directly, since that
file is currently a script with top-level execution. Instead, it loads the SLQ
and propagation functions from source so the comparison stays aligned with the
implementation under test.
"""

import argparse
import ast
import math
import time
from pathlib import Path

import numpy as np
import torch

from guti.core import (
    BRAIN_RADIUS,
    get_bitrate,
    get_grid_positions,
    get_sensor_positions,
    noise_floor_heuristic,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_function(relative_path: str, name: str, namespace: dict):
    source_path = REPO_ROOT / relative_path
    source = source_path.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise RuntimeError(f"Function {name} not found in {source_path}")


def build_free_field_matrix(
    build_source_signal,
    simulate_free_field_propagation,
    n_sources_target: int,
    n_sensors: int,
    center_frequency: float,
    temporal_sampling: int,
    signal_type: str,
    signal_cycles: float,
    signal_window: str,
) -> tuple[torch.Tensor, int, int]:
    min_speed_of_sound = 1500.0
    points_per_wavelength = 24
    dx_m = min_speed_of_sound / (points_per_wavelength * center_frequency)
    voxel_size = np.array([dx_m, dx_m, dx_m], dtype=np.float32)

    grid_spacing_mm = ((2.0 / 3.0) * np.pi * BRAIN_RADIUS**3 / n_sources_target) ** (
        1.0 / 3.0
    )
    source_positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm) * 1e-3
    sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8) * 1e-3

    time_step = 1e-1 / center_frequency
    time_duration = 120e-6
    time_axis = np.arange(0, time_duration, time_step)
    source_signals = build_source_signal(
        time_axis,
        center_frequency,
        signal_type=signal_type,
        signal_cycles=signal_cycles,
        signal_window=signal_window,
    )
    source_signals = np.tile(source_signals, (source_positions.shape[0], 1)).astype(
        np.float32
    )

    pressure_field = simulate_free_field_propagation(
        torch.tensor(source_positions, dtype=torch.float32),
        torch.tensor(sensor_positions, dtype=torch.float32),
        torch.tensor(source_signals, dtype=torch.float32),
        time_step,
        center_frequency,
        torch.tensor(voxel_size, dtype=torch.float32),
        device="cpu",
        compute_time_series=True,
        temporal_sampling=temporal_sampling,
    )
    matrix = pressure_field.permute(0, 2, 1).reshape(-1, source_positions.shape[0]).float()
    return matrix, int(source_positions.shape[0]), int(sensor_positions.shape[0])


def main():
    parser = argparse.ArgumentParser(
        description="Compare main-branch exact bitrate logic with this branch's SLQ approximation"
    )
    parser.add_argument("--n_sources_target", type=int, default=32)
    parser.add_argument("--n_sensors", type=int, default=24)
    parser.add_argument("--center_frequency", type=float, default=0.05e6)
    parser.add_argument("--temporal_sampling", type=int, default=5)
    parser.add_argument(
        "--signal_type",
        type=str,
        default="tone_burst",
        choices=["cw", "tone_burst"],
    )
    parser.add_argument("--signal_cycles", type=float, default=2.0)
    parser.add_argument(
        "--signal_window",
        type=str,
        default="hann",
        choices=["rect", "hann"],
    )
    parser.add_argument(
        "--noise_heuristic",
        type=str,
        default="power",
        choices=["power", "first"],
    )
    parser.add_argument("--noise_snr", type=float, default=2000.0)
    parser.add_argument("--slq_s", type=int, default=64)
    parser.add_argument("--slq_t", type=int, default=64)
    parser.add_argument("--slq_batch", type=int, default=16)
    parser.add_argument("--slq_chunk_rows", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")

    namespace = {"torch": torch, "np": np, "math": math, "time": time}
    bitrate_slq_torch_gpu_chunked = load_function(
        "guti/modalities/us/analytical.py",
        "bitrate_slq_torch_gpu_chunked",
        namespace,
    )
    build_source_signal = load_function(
        "guti/modalities/us/analytical.py",
        "build_source_signal",
        namespace,
    )
    simulate_free_field_propagation = load_function(
        "guti/modalities/us/utils.py",
        "simulate_free_field_propagation",
        namespace,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    build_start = time.perf_counter()
    matrix, n_sources_actual, n_sensors_actual = build_free_field_matrix(
        build_source_signal,
        simulate_free_field_propagation,
        n_sources_target=args.n_sources_target,
        n_sensors=args.n_sensors,
        center_frequency=args.center_frequency,
        temporal_sampling=args.temporal_sampling,
        signal_type=args.signal_type,
        signal_cycles=args.signal_cycles,
        signal_window=args.signal_window,
    )
    build_seconds = time.perf_counter() - build_start

    singular_values = torch.linalg.svdvals(matrix).cpu().numpy()
    normalize_scale = 1.0 / math.sqrt(n_sources_actual * n_sensors_actual)

    # This mirrors the normalization in main:guti/modalities/us/analytical.py.
    singular_values_main = singular_values * normalize_scale
    noise_level_main = noise_floor_heuristic(
        singular_values_main,
        heuristic=args.noise_heuristic,
        snr=args.noise_snr,
    )
    noise_level_current = noise_floor_heuristic(
        singular_values,
        heuristic=args.noise_heuristic,
        snr=args.noise_snr,
    )

    exact_main = get_bitrate(
        singular_values_main,
        noise_level_main,
        time_resolution=1.0,
    )
    exact_current = get_bitrate(
        singular_values,
        noise_level_current,
        time_resolution=1.0,
    )

    slq_start = time.perf_counter()
    slq_main = bitrate_slq_torch_gpu_chunked(
        matrix.cpu(),
        noise_std_full_brain=noise_level_main,
        time_resolution=1.0,
        s=args.slq_s,
        t=args.slq_t,
        batch=args.slq_batch,
        chunk_rows=args.slq_chunk_rows,
        normalize_scale=normalize_scale,
        verbose=args.verbose,
        device=args.device,
    )
    slq_current_both_mode = bitrate_slq_torch_gpu_chunked(
        matrix.cpu(),
        noise_std_full_brain=noise_level_current,
        time_resolution=1.0,
        s=args.slq_s,
        t=args.slq_t,
        batch=args.slq_batch,
        chunk_rows=args.slq_chunk_rows,
        normalize_scale=normalize_scale,
        verbose=args.verbose,
        device=args.device,
    )
    slq_seconds = time.perf_counter() - slq_start

    abs_diff_main = abs(slq_main - exact_main)
    rel_diff_main = abs_diff_main / exact_main if exact_main else float("inf")
    abs_diff_current = abs(slq_current_both_mode - exact_current)
    rel_diff_current = (
        abs_diff_current / exact_current if exact_current else float("inf")
    )

    print(
        {
            "n_sources_actual": n_sources_actual,
            "n_sensors_actual": n_sensors_actual,
            "matrix_shape": tuple(int(x) for x in matrix.shape),
            "normalize_scale": float(normalize_scale),
            "noise_level_main_style": float(noise_level_main),
            "noise_level_current_style": float(noise_level_current),
            "exact_main_style": float(exact_main),
            "slq_main_style": float(slq_main),
            "abs_diff_vs_main_style": float(abs_diff_main),
            "rel_diff_vs_main_style": float(rel_diff_main),
            "exact_current_style": float(exact_current),
            "slq_current_both_mode_style": float(slq_current_both_mode),
            "abs_diff_vs_current_style": float(abs_diff_current),
            "rel_diff_vs_current_style": float(rel_diff_current),
            "build_seconds": build_seconds,
            "slq_seconds": slq_seconds,
        }
    )


if __name__ == "__main__":
    main()
