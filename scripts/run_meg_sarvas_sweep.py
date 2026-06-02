#!/usr/bin/env python3
"""Run local Sarvas MEG SVD sweeps for OPM and SQUID variants."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-guti")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from guti.capacity import sensor_noise_normalized_singular_values
from guti.core import BRAIN_RADIUS, get_grid_positions, get_sensor_positions
from guti.data_utils import save_svd
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
)
from guti.parameters import Parameters


HEAD_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
MEG_OFFSETS_MM = {
    "meg_opm": 7.0,
    "meg_squid": 25.0,
}
DEFAULT_JOHNSON_BANDWIDTH_HZ = 100.0
JOHNSON_COVARIANCE_MODELS = {"spherical_johnson", "johnson_volume"}


def is_johnson_covariance_model(model: str) -> bool:
    return model in JOHNSON_COVARIANCE_MODELS


def saved_noise_kernel(model: str) -> str:
    if is_johnson_covariance_model(model):
        return "spherical_johnson"
    return model


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
        help="Scalp noise-correlation length in mm. Default: 5.",
    )
    parser.add_argument(
        "--noise-correlation-kernel",
        choices=("gaussian", "exponential"),
        default=DEFAULT_NOISE_CORRELATION_KERNEL,
        help="Spatial noise-correlation kernel. Default: gaussian.",
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
        "--noise-covariance-model",
        choices=("distance_kernel", "spherical_johnson", "johnson_volume"),
        default="distance_kernel",
        help=(
            "Sensor covariance model used unless --skip-noise-normalized is set. "
            "distance_kernel uses the length-scale kernel; spherical_johnson "
            "uses the same layered spherical EEG Johnson covariance correlation "
            "structure and applies the MEG scalar detector-noise diagonal. "
            "johnson_volume is accepted as a legacy alias."
        ),
    )
    parser.add_argument(
        "--johnson-electrode-area-cm2",
        type=float,
        default=DEFAULT_JOHNSON_ELECTRODE_AREA_CM2,
        help="Circular scalp electrode area for spherical_johnson covariance.",
    )
    parser.add_argument(
        "--johnson-lmax",
        type=int,
        default=DEFAULT_JOHNSON_LMAX,
        help="Spherical-harmonic truncation for spherical_johnson covariance.",
    )
    parser.add_argument(
        "--johnson-bandwidth-hz",
        type=float,
        default=DEFAULT_JOHNSON_BANDWIDTH_HZ,
        help=(
            "Bandwidth used to form the Johnson covariance. With the MEG scalar "
            "noise diagonal applied, this only affects the intermediate "
            "correlation normalization."
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
    """Vectorized Sarvas MEG matrix, 3 field components x 3 dipole components."""
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    A = np.empty((3 * n_sensors, 3 * n_sources), dtype=np.float64)
    coeff = 1e-7

    sources_m = (sources - HEAD_CENTER) * 1e-3
    for i, sensor in enumerate(sensors):
        r = (sensor - HEAD_CENTER) * 1e-3
        a_vec = r[None, :] - sources_m
        a = np.linalg.norm(a_vec, axis=1)
        r_norm = np.linalg.norm(r)
        valid = (a >= 1e-12) & (r_norm >= 1e-12)

        M = np.zeros((n_sources, 3, 3), dtype=np.float64)
        if np.any(valid):
            r0 = sources_m[valid]
            av = a_vec[valid]
            aa = a[valid]
            a_dot_r = av @ r
            r0_dot_r = r0 @ r
            F = aa * (aa * r_norm + r_norm**2 - r0_dot_r)
            valid_f = np.abs(F) >= 1e-20
            if np.any(valid_f):
                r0 = r0[valid_f]
                aa = aa[valid_f]
                a_dot_r = a_dot_r[valid_f]
                F = F[valid_f]
                nabla_F = (
                    (aa**2 / r_norm + a_dot_r / aa + 2 * aa + 2 * r_norm)[:, None]
                    * r[None, :]
                    - (aa + 2 * r_norm + a_dot_r / aa)[:, None] * r0
                )
                r0_cross = np.zeros((len(r0), 3, 3), dtype=np.float64)
                r0_cross[:, 0, 1] = -r0[:, 2]
                r0_cross[:, 0, 2] = r0[:, 1]
                r0_cross[:, 1, 0] = r0[:, 2]
                r0_cross[:, 1, 2] = -r0[:, 0]
                r0_cross[:, 2, 0] = -r0[:, 1]
                r0_cross[:, 2, 1] = r0[:, 0]
                r0xr = np.cross(r0, r[None, :])
                source_mats = (
                    coeff
                    * (
                        -F[:, None, None] * r0_cross
                        - nabla_F[:, :, None] * r0xr[:, None, :]
                    )
                    / (F[:, None, None] ** 2)
                )
                valid_indices = np.flatnonzero(valid)[valid_f]
                M[valid_indices] = source_mats

        A[3 * i : 3 * (i + 1)] = np.transpose(M, (1, 0, 2)).reshape(
            3,
            3 * n_sources,
        )
    return A


def build_sensor_noise_covariance(
    args: argparse.Namespace,
    *,
    modality: str,
    params: Parameters,
    cache: dict[tuple[Any, ...], np.ndarray],
) -> np.ndarray:
    n_sensors = int(params.num_sensors)
    noise_std = compute_output_noise_std(modality, n_sensors=n_sensors)
    if args.noise_covariance_model == "distance_kernel":
        key = (
            modality,
            n_sensors,
            "distance_kernel",
            float(params.sensor_offset_mm),
            float(args.noise_correlation_length_mm),
            args.noise_correlation_kernel,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        sensors = get_sensor_positions(n_sensors, offset=float(params.sensor_offset_mm))
        covariance = compute_sensor_noise_covariance(
            sensors,
            noise_std,
            correlation_length_mm=float(args.noise_correlation_length_mm),
            kernel=args.noise_correlation_kernel,
        )
    elif args.noise_covariance_model in JOHNSON_COVARIANCE_MODELS:
        key = (
            modality,
            n_sensors,
            "spherical_johnson",
            float(args.johnson_electrode_area_cm2),
            int(args.johnson_lmax),
            float(args.johnson_bandwidth_hz),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        scalp_sensors = get_sensor_positions(n_sensors)
        impedance = surface_impedance_kernel_matrix(
            scalp_sensors,
            electrode_area_cm2=float(args.johnson_electrode_area_cm2),
            lmax=int(args.johnson_lmax),
        )
        covariance = compute_johnson_noise_covariance(
            impedance,
            bandwidth_hz=float(args.johnson_bandwidth_hz),
            noise_std=noise_std,
        )
    else:
        raise ValueError(f"Unsupported noise covariance model {args.noise_covariance_model!r}")

    cache[key] = covariance
    return covariance


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


def main() -> int:
    args = build_parser().parse_args()
    modalities = selected_modalities(args.modalities)
    if args.extrapolate_from_sensors is not None and not args.skip_noise_normalized:
        raise SystemExit(
            "--extrapolate-from-sensors is only defined for scalar-noise spectra; "
            "pass --skip-noise-normalized"
        )
    if is_johnson_covariance_model(args.noise_covariance_model):
        if args.johnson_electrode_area_cm2 <= 0.0:
            raise SystemExit("--johnson-electrode-area-cm2 must be positive")
        if args.johnson_lmax <= 1:
            raise SystemExit("--johnson-lmax must be greater than 1")
        if args.johnson_bandwidth_hz <= 0.0:
            raise SystemExit("--johnson-bandwidth-hz must be positive")

    jobs = []
    for modality in modalities:
        for n_sensors in args.sensor_counts:
            for spacing in args.source_spacing_mm:
                params = Parameters(
                    num_sensors=int(n_sensors),
                    source_spacing_mm=float(spacing),
                    sensor_offset_mm=MEG_OFFSETS_MM[modality],
                )
                if not args.skip_noise_normalized:
                    params.noise_correlation_length_mm = (
                        None
                        if is_johnson_covariance_model(args.noise_covariance_model)
                        else float(args.noise_correlation_length_mm)
                    )
                    params.noise_correlation_kernel = (
                        saved_noise_kernel(args.noise_covariance_model)
                        if is_johnson_covariance_model(args.noise_covariance_model)
                        else args.noise_correlation_kernel
                    )
                    params.noise_distance_metric = (
                        "spherical_harmonic"
                        if is_johnson_covariance_model(args.noise_covariance_model)
                        else "geodesic"
                    )
                n_outputs, n_sources, n_voxels, size_gb = matrix_size_gb(
                    n_sensors,
                    spacing,
                )
                jobs.append((modality, params, n_outputs, n_sources, n_voxels, size_gb))

    noise_covariance_cache: dict[tuple[Any, ...], np.ndarray] = {}
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
            sensor_noise_covariance = build_sensor_noise_covariance(
                args,
                modality=modality,
                params=params,
                cache=noise_covariance_cache,
            )
            s_noise_normalized = sensor_noise_normalized_singular_values(
                A,
                sensor_noise_covariance=sensor_noise_covariance,
                outputs_per_sensor=3,
            )
            extra_arrays = {
                "noise_normalized_singular_values": s_noise_normalized,
                "noise_covariance_model": np.array(
                    saved_noise_kernel(args.noise_covariance_model)
                ),
                "noise_correlation_kernel": np.array(params.noise_correlation_kernel),
            }
            if params.noise_correlation_length_mm is not None:
                extra_arrays["noise_correlation_length_mm"] = np.array(
                    params.noise_correlation_length_mm,
                    dtype=np.float64,
                )
            if is_johnson_covariance_model(args.noise_covariance_model):
                extra_arrays.update(
                    {
                        "johnson_electrode_area_cm2": np.array(
                            args.johnson_electrode_area_cm2,
                            dtype=np.float64,
                        ),
                        "johnson_lmax": np.array(args.johnson_lmax, dtype=np.int64),
                        "johnson_bandwidth_hz": np.array(
                            args.johnson_bandwidth_hz,
                            dtype=np.float64,
                        ),
                        "johnson_noise_diagonal": np.array(
                            f"{modality}_scalar_noise_std"
                        ),
                    }
                )
            noise_text = f", noise_normalized_s0={s_noise_normalized[0]:.6g}"
        save_svd(s, modality, params, extra_arrays=extra_arrays)
        print(
            f"Saved {out_path} with {len(s)} singular values "
            f"(s0={s[0]:.6g}, s_last={s[-1]:.6g}{noise_text})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
