#!/usr/bin/env python3
"""Run local Sarvas MEG SVD sweeps for OPM and SQUID variants."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-guti")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from guti.capacity import (
    noise_normalized_singular_values,
    sensor_noise_normalized_singular_values,
)
from guti.core import BRAIN_RADIUS, get_grid_positions, get_sensor_positions
from guti.data_utils import save_svd
from guti.modalities.meg.johnson_noise import (
    compute_meg_johnson_noise_covariance,
)
from guti.noise_models import (
    DEFAULT_NOISE_CORRELATION_KERNEL,
    DEFAULT_NOISE_CORRELATION_LENGTH_MM,
    compute_output_noise_std,
    compute_sensor_noise_covariance,
    get_noise_model,
)
from guti.parameters import Parameters


HEAD_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
MEG_OFFSETS_MM = {
    "meg_opm": 7.0,
    "meg_squid": 25.0,
}


def parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Sarvas MEG singular-value variants and save them under "
            "results/variants/{meg_opm,meg_squid}."
        )
    )
    parser.add_argument(
        "--modalities",
        default="meg_opm,meg_squid",
        help="Comma-separated modalities: meg_opm, meg_squid. Default: both.",
    )
    parser.add_argument(
        "--sensor-counts",
        type=parse_int_csv,
        required=True,
        help="Comma-separated sensor counts to compute, e.g. 1200,1500,2000.",
    )
    parser.add_argument(
        "--source-spacing-mm",
        type=parse_float_csv,
        default=[5.0],
        help="Comma-separated source grid spacings in mm. Default: 5.0.",
    )
    parser.add_argument(
        "--svd-method",
        choices=("gram", "direct"),
        default="gram",
        help="SVD method. gram eigensolves the smaller covariance matrix. Default: gram.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute variants even if the target hash already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without computing.",
    )
    parser.add_argument(
        "--max-matrix-gb",
        type=float,
        default=8.0,
        help="Abort jobs whose dense forward matrix would exceed this size. Default: 8.",
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
        choices=("meg_johnson_radial", "meg_johnson_xyz", "distance_kernel"),
        default="meg_johnson_radial",
        help=(
            "Noise covariance used for saved noise-normalized spectra. "
            "meg_johnson_radial uses the finite-volume MEG body-Johnson "
            "correlation for radial point sensors and expands it across x/y/z "
            "components; meg_johnson_xyz uses the full 3-component covariance; "
            "distance_kernel keeps the older phenomenological model. "
            "Default: meg_johnson_radial."
        ),
    )
    parser.add_argument(
        "--johnson-voxel-resolution-mm",
        type=float,
        default=4.0,
        help="Voxel size for MEG Johnson covariance. Default: 4.",
    )
    parser.add_argument(
        "--johnson-solver",
        choices=("finite_volume", "vector_potential"),
        default="finite_volume",
        help="MEG Johnson reciprocal solver. Default: finite_volume.",
    )
    parser.add_argument(
        "--johnson-absolute-scale",
        action="store_true",
        help=(
            "Use the absolute body-Johnson covariance diagonal. By default the "
            "Johnson matrix supplies the correlation structure and the diagonal "
            "is matched to the existing MEG detector-noise model."
        ),
    )
    parser.add_argument(
        "--skip-noise-normalized",
        action="store_true",
        help=(
            "Save only the raw forward singular values. Use this for compatibility "
            "with existing scalar-noise convergence sweeps."
        ),
    )
    parser.add_argument(
        "--extrapolate-from-sensors",
        type=int,
        default=None,
        help=(
            "Instead of recomputing dense high-sensor spectra, scale the matching "
            "spectrum from this sensor count by sqrt(N / N_ref). This is intended "
            "for uniform dense sensor-count extensions after validating the "
            "sensor-sampling regime."
        ),
    )
    return parser


def compute_meg_forward_matrix(
    n_sensors: int,
    grid_spacing_mm: float,
    offset_mm: float,
) -> np.ndarray:
    """Fully-vectorized Sarvas MEG matrix, 3 field components x 3 dipole components."""
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    coeff = 1e-7

    r_all = (sensors - HEAD_CENTER) * 1e-3   # (n_sensors, 3)
    r0_all = (sources - HEAD_CENTER) * 1e-3  # (n_sources, 3)

    # Chunk over sensors to cap peak memory at ~12 GB
    bytes_per_sensor = n_sources * 3 * 8 * 12  # ~12 arrays of size (n_sources, 3)
    chunk = max(1, int(12e9 / bytes_per_sensor))

    A = np.empty((3 * n_sensors, 3 * n_sources), dtype=np.float64)

    # r0_cross is source-only, compute once
    r0_cross = np.zeros((n_sources, 3, 3), dtype=np.float64)
    r0_cross[:, 0, 1] = -r0_all[:, 2]
    r0_cross[:, 0, 2] = r0_all[:, 1]
    r0_cross[:, 1, 0] = r0_all[:, 2]
    r0_cross[:, 1, 2] = -r0_all[:, 0]
    r0_cross[:, 2, 0] = -r0_all[:, 1]
    r0_cross[:, 2, 1] = r0_all[:, 0]

    for start in range(0, n_sensors, chunk):
        end = min(start + chunk, n_sensors)
        r = r_all[start:end]          # (c, 3)
        c = end - start

        a_vec = r[:, None, :] - r0_all[None, :, :]          # (c, n_q, 3)
        a = np.linalg.norm(a_vec, axis=2)                    # (c, n_q)
        r_norm = np.linalg.norm(r, axis=1)                   # (c,)

        r0_dot_r = (r0_all @ r.T).T                          # (c, n_q)
        a_dot_r = (a_vec * r[:, None, :]).sum(axis=2)        # (c, n_q)

        F = a * (a * r_norm[:, None] + r_norm[:, None] ** 2 - r0_dot_r)  # (c, n_q)

        c1 = a ** 2 / r_norm[:, None] + a_dot_r / a + 2 * a + 2 * r_norm[:, None]
        c2 = a + 2 * r_norm[:, None] + a_dot_r / a
        nabla_F = c1[:, :, None] * r[:, None, :] - c2[:, :, None] * r0_all[None, :, :]  # (c, n_q, 3)

        r0_cross_nabla_F = np.cross(r0_all[None, :, :], nabla_F)  # (c, n_q, 3)

        # M[s,q,i,j] = coeff * (-F[s,q]*r0_cross[q,i,j] - r[s,i]*r0xnF[s,q,j]) / F[s,q]^2
        F2 = F[:, :, None, None] ** 2
        term1 = -F[:, :, None, None] * r0_cross[None, :, :, :]   # (c, n_q, 3, 3)
        term2 = r[:, None, :, None] * r0_cross_nabla_F[:, :, None, :]  # (c, n_q, 3, 3)
        M = coeff * (term1 - term2) / F2  # (c, n_q, 3, 3)

        # zero invalid entries
        invalid = (a < 1e-12) | (r_norm[:, None] < 1e-12) | (np.abs(F) < 1e-20)
        M[invalid] = 0.0

        # A[3s:3s+3, 3q:3q+3] = M[s,q,:,:]; layout: M.T(0,2,1,3).reshape
        A[3 * start : 3 * end] = M.transpose(0, 2, 1, 3).reshape(3 * c, 3 * n_sources)

    return A


def singular_values(A: np.ndarray, method: str) -> np.ndarray:
    if method == "direct":
        return np.linalg.svd(A, full_matrices=False, compute_uv=False)

    m, n = A.shape
    if m <= n:
        covariance = A @ A.T
    else:
        covariance = A.T @ A
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.sqrt(np.clip(eigenvalues[::-1], 0.0, None))


def target_path(modality: str, params: Parameters) -> Path:
    return Path("results/variants") / modality / f"{params.get_hash()}.npz"


def load_reference_spectrum(
    modality: str,
    *,
    n_sensors: int,
    spacing_mm: float,
    offset_mm: float,
    skip_noise_normalized: bool,
) -> tuple[np.ndarray, Parameters, Path]:
    target_params = Parameters(
        num_sensors=int(n_sensors),
        source_spacing_mm=float(spacing_mm),
        sensor_offset_mm=float(offset_mm),
    )
    target = target_path(modality, target_params)
    candidates = [target] if target.exists() else []
    candidates.extend(sorted((Path("results/variants") / modality).glob("*.npz")))

    for path in candidates:
        try:
            data = np.load(path, allow_pickle=True)
            params = Parameters.from_dict(data["parameters"].item())
        except Exception:
            continue
        if (
            params.num_sensors == int(n_sensors)
            and params.source_spacing_mm == float(spacing_mm)
            and params.sensor_offset_mm == float(offset_mm)
        ):
            if skip_noise_normalized and "noise_normalized_singular_values" in data.files:
                continue
            return np.asarray(data["singular_values"], dtype=np.float64), params, path

    raise FileNotFoundError(
        "No matching reference spectrum found for "
        f"{modality} sensors={n_sensors} spacing={spacing_mm:g}mm "
        f"offset={offset_mm:g}mm"
    )


def matrix_size_gb(n_sensors: int, grid_spacing_mm: float) -> tuple[int, int, int, float]:
    n_voxels = len(get_grid_positions(grid_spacing_mm=grid_spacing_mm))
    n_outputs = 3 * n_sensors
    n_sources = 3 * n_voxels
    size_gb = n_outputs * n_sources * np.dtype(np.float64).itemsize / 1e9
    return n_outputs, n_sources, n_voxels, size_gb


def selected_modalities(value: str) -> list[str]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(MEG_OFFSETS_MM))
    if unknown:
        raise ValueError(f"Unknown MEG modalities: {unknown}")
    return requested


def noise_params_for_args(args: argparse.Namespace) -> dict[str, object]:
    if args.noise_covariance_model == "distance_kernel":
        return {
            "noise_covariance_model": args.noise_covariance_model,
            "noise_correlation_length_mm": float(args.noise_correlation_length_mm),
            "noise_correlation_kernel": args.noise_correlation_kernel,
            "noise_distance_metric": "geodesic",
            "noise_voxel_resolution_mm": None,
            "noise_solver": None,
            "noise_absolute_scale": None,
            "noise_sensor_components": None,
        }

    sensor_components = (
        "xyz" if args.noise_covariance_model == "meg_johnson_xyz" else "radial"
    )
    return {
        "noise_covariance_model": args.noise_covariance_model,
        "noise_correlation_length_mm": None,
        "noise_correlation_kernel": "meg_johnson",
        "noise_distance_metric": "finite_volume_head",
        "noise_voxel_resolution_mm": float(args.johnson_voxel_resolution_mm),
        "noise_solver": args.johnson_solver,
        "noise_absolute_scale": bool(args.johnson_absolute_scale),
        "noise_sensor_components": sensor_components,
    }


def apply_noise_params(params: Parameters, noise_params: dict[str, object]) -> None:
    params.noise_covariance_model = noise_params["noise_covariance_model"]
    params.noise_correlation_length_mm = noise_params["noise_correlation_length_mm"]
    params.noise_correlation_kernel = noise_params["noise_correlation_kernel"]
    params.noise_distance_metric = noise_params["noise_distance_metric"]
    params.noise_voxel_resolution_mm = noise_params["noise_voxel_resolution_mm"]
    params.noise_solver = noise_params["noise_solver"]
    params.noise_absolute_scale = noise_params["noise_absolute_scale"]
    params.noise_sensor_components = noise_params["noise_sensor_components"]


def compute_noise_normalized_spectrum_for_args(
    A: np.ndarray,
    sensor_positions: np.ndarray,
    *,
    modality: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    detector_noise = compute_output_noise_std(
        modality,
        n_sensors=int(sensor_positions.shape[0]),
    )
    if args.noise_covariance_model == "distance_kernel":
        sensor_noise_covariance = compute_sensor_noise_covariance(
            sensor_positions,
            detector_noise,
            correlation_length_mm=float(args.noise_correlation_length_mm),
            kernel=args.noise_correlation_kernel,
        )
        s_noise_normalized = sensor_noise_normalized_singular_values(
            A,
            sensor_noise_covariance=sensor_noise_covariance,
            outputs_per_sensor=3,
        )
        extra_arrays = {
            "noise_covariance_model": np.array(args.noise_covariance_model),
            "noise_detector_std_t": np.array(detector_noise, dtype=np.float64),
            "noise_absolute_scale": np.array(False, dtype=bool),
        }
        return (
            s_noise_normalized,
            extra_arrays,
            f", noise_normalized_s0={s_noise_normalized[0]:.6g}",
        )

    bandwidth_hz = get_noise_model(modality).reference_bandwidth_hz
    noise_std = None if args.johnson_absolute_scale else detector_noise
    sensor_components = (
        "xyz" if args.noise_covariance_model == "meg_johnson_xyz" else "radial"
    )
    covariance, metadata, scale_info = compute_meg_johnson_noise_covariance(
        sensor_positions,
        noise_std=noise_std,
        sensor_components=sensor_components,
        voxel_resolution_mm=float(args.johnson_voxel_resolution_mm),
        bandwidth_hz=bandwidth_hz,
        solver=args.johnson_solver,
        return_metadata=True,
    )
    if sensor_components == "radial":
        s_noise_normalized = sensor_noise_normalized_singular_values(
            A,
            sensor_noise_covariance=covariance,
            outputs_per_sensor=3,
        )
    else:
        s_noise_normalized = noise_normalized_singular_values(
            A,
            output_noise_covariance=covariance,
        )

    extra_arrays = {
        "noise_covariance_model": np.array(args.noise_covariance_model),
        "noise_detector_std_t": np.array(detector_noise, dtype=np.float64),
        "noise_absolute_scale": np.array(bool(args.johnson_absolute_scale), dtype=bool),
        "johnson_voxel_resolution_mm": np.array(
            args.johnson_voxel_resolution_mm,
            dtype=np.float64,
        ),
        "johnson_solver": np.array(args.johnson_solver),
        "johnson_sensor_components": np.array(sensor_components),
        "johnson_n_voxels": np.array(metadata.n_voxels, dtype=np.int64),
        "johnson_bandwidth_hz": np.array(bandwidth_hz, dtype=np.float64),
        "johnson_raw_body_noise_median_T_per_sqrtHz": np.array(
            scale_info["raw_body_noise_median_T_per_sqrtHz"],
            dtype=np.float64,
        ),
        "johnson_regularization_jitter_T2": np.array(
            scale_info["regularization_jitter_T2"],
            dtype=np.float64,
        ),
    }
    noise_text = (
        f", noise_normalized_s0={s_noise_normalized[0]:.6g}, "
        "johnson_raw_median="
        f"{scale_info['raw_body_noise_median_T_per_sqrtHz'] * 1e15:.4g} "
        "fT/sqrtHz"
    )
    return s_noise_normalized, extra_arrays, noise_text


def main() -> int:
    args = build_parser().parse_args()
    modalities = selected_modalities(args.modalities)
    if args.extrapolate_from_sensors is not None and not args.skip_noise_normalized:
        raise SystemExit(
            "--extrapolate-from-sensors is only defined for scalar-noise spectra; "
            "pass --skip-noise-normalized"
        )
    if args.johnson_voxel_resolution_mm <= 0.0:
        raise SystemExit("--johnson-voxel-resolution-mm must be positive")

    jobs = []
    noise_params = noise_params_for_args(args)
    for modality in modalities:
        for n_sensors in args.sensor_counts:
            for spacing in args.source_spacing_mm:
                params = Parameters(
                    num_sensors=int(n_sensors),
                    source_spacing_mm=float(spacing),
                    sensor_offset_mm=MEG_OFFSETS_MM[modality],
                )
                if not args.skip_noise_normalized:
                    apply_noise_params(params, noise_params)
                n_outputs, n_sources, n_voxels, size_gb = matrix_size_gb(
                    n_sensors,
                    spacing,
                )
                jobs.append((modality, params, n_outputs, n_sources, n_voxels, size_gb))

    for modality, params, n_outputs, n_sources, n_voxels, size_gb in jobs:
        if args.extrapolate_from_sensors is not None:
            params.comment = (
                "sensor_count_sqrt_extrapolated_from_"
                f"{int(args.extrapolate_from_sensors)}_sensor_sarvas_spectrum"
            )
        elif not args.skip_noise_normalized:
            params.comment = (
                "sarvas_meg_noise_normalized_"
                f"{args.noise_covariance_model}_covariance"
            )
        out_path = target_path(modality, params)
        label = (
            f"{modality} sensors={params.num_sensors} "
            f"spacing={params.source_spacing_mm:g}mm voxels={n_voxels} "
            f"matrix={n_outputs}x{n_sources} ({size_gb:.2f} GB)"
        )
        if out_path.exists() and not args.force:
            print(f"Skipping existing {label}: {out_path}", flush=True)
            continue
        if size_gb > args.max_matrix_gb:
            raise SystemExit(
                f"Refusing {label}; dense matrix exceeds --max-matrix-gb={args.max_matrix_gb}"
            )
        print(f"Computing {label}: {out_path}", flush=True)
        if args.dry_run:
            continue

        if args.extrapolate_from_sensors is not None:
            reference_n = int(args.extrapolate_from_sensors)
            if int(params.num_sensors) <= reference_n:
                raise SystemExit(
                    "--extrapolate-from-sensors requires target sensor counts "
                    "larger than the reference count"
                )
            s_ref, ref_params, ref_path = load_reference_spectrum(
                modality,
                n_sensors=reference_n,
                spacing_mm=float(params.source_spacing_mm),
                offset_mm=float(params.sensor_offset_mm),
                skip_noise_normalized=True,
            )
            scale = math.sqrt(float(params.num_sensors) / reference_n)
            s = s_ref * scale
            extra_arrays = {
                "spectrum_estimate_method": np.array(
                    "sensor_count_sqrt_scaling_extrapolation"
                ),
                "spectrum_reference_num_sensors": np.array(reference_n, dtype=np.int64),
                "spectrum_reference_path": np.array(str(ref_path)),
                "spectrum_reference_hash": np.array(ref_path.stem),
                "spectrum_reference_singular_value_count": np.array(
                    len(s_ref),
                    dtype=np.int64,
                ),
                "spectrum_scale_factor": np.array(scale, dtype=np.float64),
            }
            save_svd(s, modality, params, extra_arrays=extra_arrays)
            print(
                f"Saved extrapolated {out_path} with {len(s)} singular values "
                f"from {ref_path} (scale={scale:.6g}, "
                f"ref_s0={s_ref[0]:.6g}, s0={s[0]:.6g})",
                flush=True,
            )
            continue

        A = compute_meg_forward_matrix(
            n_sensors=int(params.num_sensors),
            grid_spacing_mm=float(params.source_spacing_mm),
            offset_mm=float(params.sensor_offset_mm),
        )
        s = singular_values(A, args.svd_method)
        extra_arrays = None
        noise_text = ""
        if not args.skip_noise_normalized:
            sensors = get_sensor_positions(
                int(params.num_sensors),
                offset=float(params.sensor_offset_mm),
            )
            s_noise_normalized, extra_arrays, noise_text = (
                compute_noise_normalized_spectrum_for_args(
                    A,
                    sensors,
                    modality=modality,
                    args=args,
                )
            )
            extra_arrays = {
                **extra_arrays,
                "noise_normalized_singular_values": s_noise_normalized,
            }
        save_svd(s, modality, params, extra_arrays=extra_arrays)
        print(
            f"Saved {out_path} with {len(s)} singular values "
            f"(s0={s[0]:.6g}, s_last={s[-1]:.6g}{noise_text})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
