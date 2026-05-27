#!/usr/bin/env python3
"""Compute posterior information maps for representative GUTI modalities.

The model is:

    y = A x + n
    x ~ N(0, I)
    n ~ N(0, noise^2 I)

Each column of A is scaled so one unit of x corresponds to the reference source
amplitude in the modality noise model.  For EEG/MEG, each voxel has a 3-vector
dipole source and the voxel score is the mutual information for that 3D block.
For fNIRS, each voxel is scalar absorption contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from guti.core import BRAIN_RADIUS, get_grid_positions, get_sensor_positions
from guti.modalities.fnirs_analytical.modality import fNIRSAnalytical
from guti.noise_models import compute_detector_noise_std, get_noise_model
from guti.parameters import Parameters
from recompute_meg_variants import compute_forward_matrix as compute_meg_forward_matrix


OUT_DIR = Path("results/information_maps")
HEAD_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])


def depth_mm(positions: np.ndarray) -> np.ndarray:
    """Depth inward from the spherical brain surface."""
    radius = np.linalg.norm(positions - HEAD_CENTER, axis=1)
    return BRAIN_RADIUS - radius


def compute_eeg_forward_matrix(
    n_sensors: int,
    grid_spacing_mm: float,
    conductivity_s_per_m: float = 0.33,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple homogeneous-sphere EEG dipole potential matrix.

    This is a lightweight approximation used for information-map prototyping.
    It maps x/y/z current dipoles at each grid point to scalar scalp voltages.
    """
    sensors_m = (get_sensor_positions(n_sensors) - HEAD_CENTER) * 1e-3
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    sources_m = (sources - HEAD_CENTER) * 1e-3

    rows = []
    coeff = 1.0 / (4.0 * np.pi * conductivity_s_per_m)
    for sensor in sensors_m:
        r = sensor[None, :] - sources_m
        dist = np.linalg.norm(r, axis=1)
        block = coeff * r / np.maximum(dist[:, None], 1e-12) ** 3
        rows.append(block.reshape(1, -1))
    return np.vstack(rows), sources


def posterior_info_scalar(
    A: np.ndarray,
    noise: float,
    chunk_cols: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar voxel posterior variance and information in bits/sample."""
    A64 = np.asarray(A, dtype=np.float64)
    m, n = A64.shape
    gram_y = A64 @ A64.T
    gram_y.flat[:: m + 1] += noise**2
    chol = cho_factor(gram_y, lower=True, check_finite=False)

    posterior_var = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_cols):
        stop = min(start + chunk_cols, n)
        block = A64[:, start:stop]
        solved = cho_solve(chol, block, check_finite=False)
        explained = np.sum(block * solved, axis=0)
        posterior_var[start:stop] = np.clip(1.0 - explained, 1e-15, 1.0)

    info_bits = -0.5 * np.log2(posterior_var)
    return posterior_var, info_bits


def posterior_info_vector3(
    A: np.ndarray,
    noise: float,
    n_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """3D dipole posterior covariance determinant and information by voxel."""
    A64 = np.asarray(A, dtype=np.float64)
    m = A64.shape[0]
    gram_y = A64 @ A64.T
    gram_y.flat[:: m + 1] += noise**2
    chol = cho_factor(gram_y, lower=True, check_finite=False)

    posterior_det = np.empty(n_voxels, dtype=np.float64)
    info_bits = np.empty(n_voxels, dtype=np.float64)
    eye3 = np.eye(3)

    for i in range(n_voxels):
        cols = slice(3 * i, 3 * i + 3)
        block = A64[:, cols]
        solved = cho_solve(chol, block, check_finite=False)
        posterior_block = eye3 - block.T @ solved
        posterior_block = 0.5 * (posterior_block + posterior_block.T)
        sign, logdet = np.linalg.slogdet(posterior_block)
        if sign <= 0:
            eig = np.linalg.eigvalsh(posterior_block)
            logdet = np.sum(np.log(np.clip(eig, 1e-15, 1.0)))
        posterior_det[i] = np.exp(logdet)
        info_bits[i] = -0.5 * logdet / np.log(2.0)

    return posterior_det, info_bits


def summarize_by_depth(depth: np.ndarray, info_bits: np.ndarray, bin_width_mm: float = 5.0):
    bins = np.arange(0, BRAIN_RADIUS + bin_width_mm, bin_width_mm)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full_like(centers, np.nan, dtype=np.float64)
    median = np.full_like(centers, np.nan, dtype=np.float64)
    p90 = np.full_like(centers, np.nan, dtype=np.float64)
    count = np.zeros_like(centers, dtype=np.int64)
    for i in range(len(centers)):
        mask = (depth >= bins[i]) & (depth < bins[i + 1])
        count[i] = int(mask.sum())
        if count[i]:
            vals = info_bits[mask]
            mean[i] = float(np.mean(vals))
            median[i] = float(np.median(vals))
            p90[i] = float(np.percentile(vals, 90))
    return {
        "bin_centers_mm": centers,
        "mean_bits": mean,
        "median_bits": median,
        "p90_bits": p90,
        "count": count,
    }


def save_result(
    name: str,
    positions: np.ndarray,
    info_bits: np.ndarray,
    posterior_measure: np.ndarray,
    params: dict,
    posterior_measure_name: str,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = depth_mm(positions)
    profile = summarize_by_depth(d, info_bits)
    out_path = OUT_DIR / f"{name}_posterior_info.npz"
    np.savez(
        out_path,
        positions_mm=positions,
        depth_mm=d,
        info_bits_per_sample=info_bits,
        **{posterior_measure_name: posterior_measure},
        depth_bin_centers_mm=profile["bin_centers_mm"],
        depth_mean_bits=profile["mean_bits"],
        depth_median_bits=profile["median_bits"],
        depth_p90_bits=profile["p90_bits"],
        depth_bin_count=profile["count"],
        params_json=json.dumps(params, sort_keys=True),
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(profile["bin_centers_mm"], profile["mean_bits"], label="mean")
    ax.plot(profile["bin_centers_mm"], profile["median_bits"], label="median")
    ax.plot(profile["bin_centers_mm"], profile["p90_bits"], label="p90")
    ax.set_xlabel("Depth from brain surface (mm)")
    ax.set_ylabel("Information (bits/sample/voxel)")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_depth_profile.png", dpi=180)
    plt.close(fig)

    return out_path


def run_meg(name: str, n_sensors: int, grid_spacing_mm: float, offset_mm: float) -> Path:
    model = get_noise_model(name)
    noise = compute_detector_noise_std(name, n_sensors=n_sensors, tier="today")
    A_raw = compute_meg_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
    positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    A = A_raw * model.source_amplitude
    posterior_det, info_bits = posterior_info_vector3(A, noise, len(positions))
    return save_result(
        name,
        positions,
        info_bits,
        posterior_det,
        {
            "modality": name,
            "n_sensors": n_sensors,
            "grid_spacing_mm": grid_spacing_mm,
            "sensor_offset_mm": offset_mm,
            "noise_today": noise,
            "source_amplitude": model.source_amplitude,
            "model": "Sarvas MEG, 3 dipole orientations per voxel",
        },
        "posterior_cov_det",
    )


def run_eeg(n_sensors: int, grid_spacing_mm: float) -> Path:
    noise = compute_detector_noise_std("eeg_openmeeg", n_sensors=n_sensors, tier="today")
    source_amplitude_a_m = 10e-9
    A_raw, positions = compute_eeg_forward_matrix(n_sensors, grid_spacing_mm)
    A = A_raw * source_amplitude_a_m
    posterior_det, info_bits = posterior_info_vector3(A, noise, len(positions))
    return save_result(
        "eeg_homogeneous",
        positions,
        info_bits,
        posterior_det,
        {
            "modality": "eeg_homogeneous",
            "n_sensors": n_sensors,
            "grid_spacing_mm": grid_spacing_mm,
            "noise_today": noise,
            "source_amplitude_a_m": source_amplitude_a_m,
            "model": "homogeneous quasi-static dipole potential, prototype map",
        },
        "posterior_cov_det",
    )


def run_fnirs(n_sensors: int, grid_spacing_mm: float, max_dist_mm: float) -> Path:
    model = get_noise_model("fnirs_analytical_cw")
    noise = compute_detector_noise_std(
        "fnirs_analytical_cw",
        n_sensors=n_sensors,
        tier="today",
    )
    modality = fNIRSAnalytical(
        Parameters(
            num_sensors=n_sensors,
            grid_resolution_mm=grid_spacing_mm,
            max_dist=max_dist_mm,
        )
    )
    modality.setup_geometry()
    A_raw = modality.compute_forward_model()
    positions = modality.grid_points
    A = np.asarray(A_raw, dtype=np.float32) * model.source_amplitude
    posterior_var, info_bits = posterior_info_scalar(A, noise)
    return save_result(
        "fnirs_analytical_cw",
        positions,
        info_bits,
        posterior_var,
        {
            "modality": "fnirs_analytical_cw",
            "n_sensors": n_sensors,
            "grid_spacing_mm": grid_spacing_mm,
            "max_dist_mm": max_dist_mm,
            "noise_today": noise,
            "source_amplitude": model.source_amplitude,
            "model": "CW fNIRS analytical diffusion sensitivity, scalar absorption per voxel",
        },
        "posterior_variance",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-spacing-mm", type=float, default=10.0)
    parser.add_argument("--eeg-sensors", type=int, default=256)
    parser.add_argument("--meg-sensors", type=int, default=200)
    parser.add_argument("--fnirs-sensors", type=int, default=100)
    parser.add_argument("--fnirs-grid-spacing-mm", type=float, default=6.0)
    parser.add_argument("--fnirs-max-dist-mm", type=float, default=50.0)
    args = parser.parse_args()

    outputs = []
    outputs.append(run_eeg(args.eeg_sensors, args.grid_spacing_mm))
    outputs.append(run_meg("meg_opm", args.meg_sensors, args.grid_spacing_mm, 7.0))
    outputs.append(run_meg("meg_squid", args.meg_sensors, args.grid_spacing_mm, 25.0))
    outputs.append(
        run_fnirs(
            args.fnirs_sensors,
            args.fnirs_grid_spacing_mm,
            args.fnirs_max_dist_mm,
        )
    )
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
