#!/usr/bin/env python3
"""Run a clean EEG OpenMEEG SVD sweep over voxel and sensor counts."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from guti.capacity import sensor_noise_normalized_singular_values
from guti.core import (
    BRAIN_RADIUS,
    create_eeg_bem_model,
    get_grid_positions,
    get_sensor_positions,
)
from guti.modalities.eeg.compute_eeg_leadfield import (
    compute_eeg_leadfield_from_bem_dir,
)
from guti.modalities.eeg.scalp_resistance import (
    DEFAULT_JOHNSON_ELECTRODE_AREA_CM2,
    DEFAULT_JOHNSON_LMAX,
    surface_impedance_kernel_matrix,
)
from guti.noise_models import (
    DEFAULT_NOISE_CORRELATION_KERNEL,
    DEFAULT_NOISE_CORRELATION_LENGTH_MM,
    compute_johnson_noise_covariance,
    compute_output_noise_std,
    compute_sensor_noise_covariance,
    get_noise_model,
)
from guti.parameters import Parameters


DEFAULT_SOURCE_SPACING_MM = (40.0, 30.0, 20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0)
DEFAULT_SENSOR_COUNTS = (32, 64, 128, 256, 512, 1024, 2048, 10000)
DEFAULT_GRID_RESOLUTION_MM = 20.0
DEFAULT_SOURCE_RADIUS_MARGIN_MM = 5.0
DEFAULT_OUTPUT_DIR = Path("results/variants/eeg_openmeeg_clean_sweep_20260601_margin5mm")
DEFAULT_WORK_DIR = Path("results/tmp/eeg_openmeeg_clean_sweep_20260601")
DEFAULT_BEM_DIR = DEFAULT_WORK_DIR / "bem_model/eeg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EEG OpenMEEG singular-value spectra for a rectangular "
            "source-spacing by sensor-count sweep."
        )
    )
    parser.add_argument(
        "--source-spacing-mm",
        nargs="+",
        type=float,
        default=DEFAULT_SOURCE_SPACING_MM,
        help="Source-grid spacings in mm. These determine n_voxels.",
    )
    parser.add_argument(
        "--num-sensors",
        nargs="+",
        type=int,
        default=DEFAULT_SENSOR_COUNTS,
        help="EEG sensor counts.",
    )
    parser.add_argument(
        "--grid-resolution-mm",
        type=float,
        default=DEFAULT_GRID_RESOLUTION_MM,
        help="OpenMEEG spherical mesh resolution in mm.",
    )
    parser.add_argument(
        "--source-radius-margin-mm",
        type=float,
        default=DEFAULT_SOURCE_RADIUS_MARGIN_MM,
        help=(
            "Exclude source grid points within this margin of the brain boundary. "
            "The default keeps sources well inside the coarse OpenMEEG mesh."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for clean sweep NPZ/JSONL outputs.",
    )
    parser.add_argument(
        "--bem-dir",
        type=Path,
        default=DEFAULT_BEM_DIR,
        help="Scratch directory for OpenMEEG model files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of grid points to run, for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and dependency status without running OpenMEEG.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute spectra even when the target NPZ already exists.",
    )
    parser.add_argument(
        "--noise-correlation-length-mm",
        type=float,
        default=DEFAULT_NOISE_CORRELATION_LENGTH_MM,
        help=(
            "Scalp noise-correlation length in mm for "
            "--noise-covariance-model=distance_kernel. Default: 5."
        ),
    )
    parser.add_argument(
        "--noise-correlation-kernel",
        choices=("gaussian", "exponential"),
        default=DEFAULT_NOISE_CORRELATION_KERNEL,
        help=(
            "Spatial distance-correlation kernel for "
            "--noise-covariance-model=distance_kernel. Default: gaussian."
        ),
    )
    parser.add_argument(
        "--noise-covariance-model",
        choices=("spherical_johnson", "distance_kernel"),
        default="spherical_johnson",
        help=(
            "Noise covariance used for --save-noise-normalized. "
            "spherical_johnson uses the layered spherical-harmonic EEG "
            "surface-impedance kernel; distance_kernel keeps the older "
            "phenomenological Gaussian/exponential distance model."
        ),
    )
    parser.add_argument(
        "--johnson-electrode-area-cm2",
        type=float,
        default=DEFAULT_JOHNSON_ELECTRODE_AREA_CM2,
        help=(
            "Circular electrode patch area for the spherical Johnson kernel. "
            f"Default: {DEFAULT_JOHNSON_ELECTRODE_AREA_CM2:g} cm^2."
        ),
    )
    parser.add_argument(
        "--johnson-lmax",
        type=int,
        default=DEFAULT_JOHNSON_LMAX,
        help=(
            "Spherical-harmonic truncation for the Johnson impedance kernel. "
            f"Default: {DEFAULT_JOHNSON_LMAX}."
        ),
    )
    parser.add_argument(
        "--johnson-series-resistance-ohm",
        type=float,
        default=None,
        help=(
            "Optional independent per-electrode series resistance added before "
            "forming the Johnson covariance. By default only the layered volume "
            "conductor impedance determines the correlation structure."
        ),
    )
    parser.add_argument(
        "--johnson-absolute-scale",
        action="store_true",
        help=(
            "Use the absolute Johnson covariance diagonal. By default the "
            "spherical Johnson matrix supplies only the correlation structure, "
            "and the diagonal is matched to the existing EEG detector-noise model."
        ),
    )
    parser.add_argument(
        "--save-noise-normalized",
        action="store_true",
        help=(
            "Also save singular values whitened by the correlated sensor-noise "
            "covariance. This is much more expensive for large sensor counts."
        ),
    )
    parser.add_argument(
        "--max-save-noise-normalized-sensors",
        type=int,
        default=2048,
        help=(
            "Refuse --save-noise-normalized jobs above this sensor count unless "
            "explicitly raised. Default: 2048."
        ),
    )
    return parser


def check_dependencies() -> list[str]:
    try:
        import openmeeg  # noqa: F401
    except ImportError:
        return ["python package openmeeg"]
    return []


def eeg_grid_positions(source_spacing_mm: float, source_radius_margin_mm: float) -> np.ndarray:
    positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
    if source_radius_margin_mm <= 0:
        return positions
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
    distances = np.linalg.norm(positions - center, axis=1)
    return positions[distances < BRAIN_RADIUS - source_radius_margin_mm]


def count_dipoles(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def save_clean_svd(
    singular_values: np.ndarray,
    params: Parameters,
    output_dir: Path,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_hash = params.get_hash()
    path = output_dir / f"{params_hash}.npz"
    arrays = {} if extra_arrays is None else dict(extra_arrays)
    np.savez(path, singular_values=singular_values, parameters=asdict(params), **arrays)
    return path


def _float_key(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def sweep_key(params: Parameters | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(params, Parameters):
        data = asdict(params)
    else:
        data = params
    return (
        int(data["num_sensors"]),
        _float_key(data.get("source_spacing_mm")),
        _float_key(data.get("grid_resolution_mm")),
        int(data["num_brain_grid_points"]),
    )


def noise_key(params: Parameters | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(params, Parameters):
        data = asdict(params)
    else:
        data = params
    key = (
        data.get("noise_correlation_kernel"),
        _float_key(data.get("noise_correlation_length_mm")),
        data.get("noise_distance_metric"),
    )
    if data.get("noise_correlation_kernel") == "spherical_johnson":
        key = (*key, data.get("comment"))
    return key


def run_comment_for_args(args: argparse.Namespace) -> str:
    base = "clean EEG OpenMEEG n_voxels x n_sensors convergence sweep"
    if not args.save_noise_normalized:
        return base
    if args.noise_covariance_model == "distance_kernel":
        return (
            f"{base}; noise_covariance=distance_kernel "
            f"kernel={args.noise_correlation_kernel} "
            f"length_mm={float(args.noise_correlation_length_mm):g}"
        )
    scale = "absolute" if args.johnson_absolute_scale else "matched_detector_diagonal"
    series = (
        "none"
        if args.johnson_series_resistance_ohm is None
        else f"{float(args.johnson_series_resistance_ohm):g}"
    )
    return (
        f"{base}; noise_covariance=spherical_johnson "
        f"electrode_area_cm2={float(args.johnson_electrode_area_cm2):g} "
        f"lmax={int(args.johnson_lmax)} "
        f"series_resistance_ohm={series} "
        f"scale={scale}"
    )


def find_existing_npz(
    output_dir: Path,
    params: Parameters,
    *,
    require_noise_normalized: bool,
) -> Path | None:
    target_key = sweep_key(params)
    newest: tuple[float, str, Path] | None = None
    for path in output_dir.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                existing_params = data["parameters"].item()
                if sweep_key(existing_params) != target_key:
                    continue
                if (
                    require_noise_normalized
                    and "noise_normalized_singular_values" not in data.files
                ):
                    continue
                if require_noise_normalized and noise_key(existing_params) != noise_key(
                    params
                ):
                    continue
        except Exception:
            continue
        candidate = (path.stat().st_mtime, path.name, path)
        if newest is None or candidate > newest:
            newest = candidate
    return None if newest is None else newest[2]


def compute_singular_values(leadfield: np.ndarray) -> np.ndarray:
    """Compute singular values via the smaller Gram matrix."""
    if leadfield.shape[0] >= leadfield.shape[1]:
        gram = leadfield.T @ leadfield
    else:
        gram = leadfield @ leadfield.T
    eigvals = np.linalg.eigvalsh(gram)
    singular_values = np.sqrt(np.clip(eigvals, 0.0, None))
    return singular_values[::-1]


def noise_params_for_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.save_noise_normalized:
        return {
            "noise_correlation_length_mm": None,
            "noise_correlation_kernel": None,
            "noise_distance_metric": None,
        }
    if args.noise_covariance_model == "spherical_johnson":
        return {
            "noise_correlation_length_mm": None,
            "noise_correlation_kernel": "spherical_johnson",
            "noise_distance_metric": "spherical_harmonic",
        }
    return {
        "noise_correlation_length_mm": float(args.noise_correlation_length_mm),
        "noise_correlation_kernel": args.noise_correlation_kernel,
        "noise_distance_metric": "geodesic",
    }


def compute_sensor_noise_covariance_for_args(
    sensor_positions: np.ndarray,
    *,
    detector_noise: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.noise_covariance_model == "distance_kernel":
        return compute_sensor_noise_covariance(
            sensor_positions,
            detector_noise,
            correlation_length_mm=float(args.noise_correlation_length_mm),
            kernel=args.noise_correlation_kernel,
        )

    impedance = surface_impedance_kernel_matrix(
        sensor_positions,
        electrode_area_cm2=float(args.johnson_electrode_area_cm2),
        lmax=int(args.johnson_lmax),
    )
    noise_std = None if args.johnson_absolute_scale else detector_noise
    return compute_johnson_noise_covariance(
        impedance,
        bandwidth_hz=get_noise_model("eeg_openmeeg").reference_bandwidth_hz,
        series_resistance_ohm=args.johnson_series_resistance_ohm,
        noise_std=noise_std,
    )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (source_spacing_mm, num_sensors)
        for source_spacing_mm in args.source_spacing_mm
        for num_sensors in args.num_sensors
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.save_noise_normalized:
        too_large = [
            num_sensors
            for _, num_sensors in jobs
            if num_sensors > args.max_save_noise_normalized_sensors
        ]
        if too_large:
            max_requested = max(too_large)
            raise SystemExit(
                "--save-noise-normalized builds and diagonalizes a dense sensor "
                f"covariance; requested {max_requested} sensors exceeds "
                f"--max-save-noise-normalized-sensors="
                f"{args.max_save_noise_normalized_sensors}."
            )

    print(f"Planned EEG OpenMEEG jobs: {len(jobs)}")
    for source_spacing_mm, num_sensors in jobs:
        n_voxels = len(
            eeg_grid_positions(source_spacing_mm, args.source_radius_margin_mm)
        )
        print(
            f"  source_spacing_mm={source_spacing_mm:g} "
            f"n_voxels={n_voxels} num_sensors={num_sensors}"
        )

    missing = check_dependencies()
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        if not args.dry_run:
            return 2

    if args.dry_run:
        return 0

    records_path = output_dir / "runs.jsonl"
    for index, (source_spacing_mm, num_sensors) in enumerate(jobs, start=1):
        start_time = time.time()
        n_voxels = len(
            eeg_grid_positions(source_spacing_mm, args.source_radius_margin_mm)
        )
        expected_n_dipoles = 3 * n_voxels
        noise_params = noise_params_for_args(args)
        params = Parameters(
            num_sensors=int(num_sensors),
            source_spacing_mm=float(source_spacing_mm),
            grid_resolution_mm=float(args.grid_resolution_mm),
            num_brain_grid_points=int(n_voxels),
            voxel_volume_mm3=float(source_spacing_mm**3),
            matrix_size=(int(num_sensors), int(expected_n_dipoles)),
            noise_correlation_length_mm=noise_params["noise_correlation_length_mm"],
            noise_correlation_kernel=noise_params["noise_correlation_kernel"],
            noise_distance_metric=noise_params["noise_distance_metric"],
            comment=run_comment_for_args(args),
        )
        existing_npz_path = find_existing_npz(
            output_dir,
            params,
            require_noise_normalized=args.save_noise_normalized,
        )
        if existing_npz_path is not None and not args.force:
            print(f"[{index}/{len(jobs)}] skip existing {existing_npz_path}")
            continue

        print(
            f"[{index}/{len(jobs)}] EEG OpenMEEG "
            f"source_spacing_mm={source_spacing_mm:g} "
            f"n_voxels={n_voxels} num_sensors={num_sensors}"
        )

        create_eeg_bem_model(
            source_spacing_mm=source_spacing_mm,
            n_sensors=num_sensors,
            grid_resolution=args.grid_resolution_mm,
            output_dir=str(args.bem_dir),
            use_radial_orientations=False,
            source_radius_margin_mm=args.source_radius_margin_mm,
        )

        leadfield = compute_eeg_leadfield_from_bem_dir(args.bem_dir)
        n_dipoles = count_dipoles(args.bem_dir / "dipole_locations.txt")
        if n_dipoles != 3 * n_voxels:
            raise ValueError(
                f"Expected 3 orientations per voxel; got {n_dipoles} dipoles "
                f"for {n_voxels} voxels"
            )

        singular_values = compute_singular_values(leadfield)
        s_noise_normalized = None
        noise_extra_arrays: dict[str, np.ndarray] = {}
        if args.save_noise_normalized:
            detector_noise = compute_output_noise_std(
                "eeg_openmeeg",
                n_sensors=num_sensors,
            )
            sensor_noise_covariance = compute_sensor_noise_covariance_for_args(
                get_sensor_positions(num_sensors),
                detector_noise=detector_noise,
                args=args,
            )
            s_noise_normalized = sensor_noise_normalized_singular_values(
                leadfield.T,
                sensor_noise_covariance=sensor_noise_covariance,
                outputs_per_sensor=1,
            )
            noise_extra_arrays = {
                "noise_covariance_model": np.array(args.noise_covariance_model),
                "noise_detector_std_v": np.array(detector_noise, dtype=np.float64),
                "noise_absolute_scale": np.array(
                    bool(args.johnson_absolute_scale),
                    dtype=bool,
                ),
            }
            if args.noise_covariance_model == "spherical_johnson":
                noise_extra_arrays.update(
                    {
                        "johnson_electrode_area_cm2": np.array(
                            args.johnson_electrode_area_cm2,
                            dtype=np.float64,
                        ),
                        "johnson_lmax": np.array(args.johnson_lmax, dtype=np.int64),
                        "johnson_series_resistance_ohm": np.array(
                            np.nan
                            if args.johnson_series_resistance_ohm is None
                            else args.johnson_series_resistance_ohm,
                            dtype=np.float64,
                        ),
                    }
                )
        extra_arrays: dict[str, np.ndarray] = {}
        if s_noise_normalized is not None:
            extra_arrays.update(
                {
                    "noise_normalized_singular_values": s_noise_normalized,
                    "noise_correlation_length_mm": np.array(
                        np.nan
                        if noise_params["noise_correlation_length_mm"] is None
                        else noise_params["noise_correlation_length_mm"],
                        dtype=np.float64,
                    ),
                    "noise_correlation_kernel": np.array(
                        noise_params["noise_correlation_kernel"]
                    ),
                }
            )
            extra_arrays.update(noise_extra_arrays)
        npz_path = save_clean_svd(
            singular_values,
            params,
            output_dir,
            extra_arrays=extra_arrays,
        )
        record: dict[str, Any] = {
            "index": index,
            "status": "ok",
            "npz_path": str(npz_path),
            "source_spacing_mm": source_spacing_mm,
            "grid_resolution_mm": args.grid_resolution_mm,
            "source_radius_margin_mm": args.source_radius_margin_mm,
            "num_sensors": num_sensors,
            "num_brain_grid_points": n_voxels,
            "n_dipoles": n_dipoles,
            "leadfield_shape": tuple(int(v) for v in leadfield.shape),
            "matrix_size": params.matrix_size,
            "n_singular_values": int(len(singular_values)),
            "svd_method": "gram_eigvalsh",
            "noise_correlation_length_mm": noise_params["noise_correlation_length_mm"],
            "noise_correlation_kernel": noise_params["noise_correlation_kernel"],
            "noise_distance_metric": noise_params["noise_distance_metric"],
            "noise_covariance_model": args.noise_covariance_model,
            "johnson_electrode_area_cm2": (
                args.johnson_electrode_area_cm2
                if args.noise_covariance_model == "spherical_johnson"
                else None
            ),
            "johnson_lmax": (
                args.johnson_lmax
                if args.noise_covariance_model == "spherical_johnson"
                else None
            ),
            "johnson_series_resistance_ohm": (
                args.johnson_series_resistance_ohm
                if args.noise_covariance_model == "spherical_johnson"
                else None
            ),
            "johnson_absolute_scale": (
                bool(args.johnson_absolute_scale)
                if args.noise_covariance_model == "spherical_johnson"
                else None
            ),
            "elapsed_s": time.time() - start_time,
        }
        if s_noise_normalized is not None:
            record["noise_normalized_first_singular_value"] = float(
                s_noise_normalized[0]
            )
        with records_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"  saved {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
