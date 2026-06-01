#!/usr/bin/env python3
"""Compute posterior information maps for representative GUTI modalities.

The model is:

    y = A x + n
    x ~ N(0, I)
    n ~ N(0, noise^2 I)

By default the maps use the same empirical-SNR normalization as the web bitrate
export: the whole forward matrix is scaled relative to a noise floor such that
sqrt(sum(s_i^2)) / noise equals the modality's empirical SNR.  This preserves the
spatial structure of the forward model while avoiding claims that depend on an
unvalidated absolute Jacobian gain.

Use --scaling physical to inspect the raw source-amplitude/noise model instead.
For EEG/MEG, each voxel has a 3-vector dipole source and the voxel score is the
mutual information for that 3D block.  For fNIRS, each voxel is scalar absorption
contrast integrated over the voxel volume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from guti.core import BRAIN_RADIUS, get_grid_positions, get_sensor_positions
from guti.modalities.cw_fnirs.modality import CWfNIRS
from guti.noise_models import (
    compute_detector_noise_std,
    compute_empirical_snr,
    get_noise_model,
)
from guti.parameters import Parameters
from recompute_meg_variants import compute_forward_matrix as compute_meg_forward_matrix


OUT_DIR = Path("results/information_maps")
HEAD_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
SCALING_CHOICES = ("empirical", "physical")
DEFAULT_SCALING = "empirical"
COMPARISON_LABELS = {
    "eeg_homogeneous": "EEG homogeneous",
    "meg_opm": "MEG OPM",
    "meg_squid": "MEG SQUID",
    "cw_fnirs": "fNIRS CW",
}
MODALITY_COLORS = {
    "EEG homogeneous": "#2563eb",
    "MEG OPM": "#ea580c",
    "MEG SQUID": "#16a34a",
    "fNIRS CW": "#dc2626",
}
ANATOMICAL_DEPTH_BANDS = [
    {
        "label": "cortex",
        "start_mm": 0.0,
        "end_mm": 3.0,
        "color": "#f2c88f",
    },
    {
        "label": "white matter / subcortical",
        "start_mm": 3.0,
        "end_mm": 25.0,
        "color": "#cde8d2",
    },
    {
        "label": "deep brain",
        "start_mm": 25.0,
        "end_mm": float(BRAIN_RADIUS),
        "color": "#d9e2f3",
    },
]
ANATOMICAL_DEPTH_BAND_META = [
    {k: band[k] for k in ("label", "start_mm", "end_mm")}
    for band in ANATOMICAL_DEPTH_BANDS
]


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


def as_numpy(array) -> np.ndarray:
    """Convert numpy/torch-like arrays to a CPU numpy array."""
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def empirical_noise_for_matrix(
    A: np.ndarray,
    modality_name: str,
    n_sensors: int,
    tier: str = "today",
) -> tuple[float, dict]:
    """Noise floor matching the repo's empirical bitrate normalization.

    The web bitrate export uses compute_noise_empirical(), equivalent to
    noise = sqrt(sum(s_i^2)) / empirical_snr.  The Frobenius norm of A equals
    sqrt(sum(s_i^2)), so this avoids an expensive SVD while producing the same
    channel ratios.
    """
    empirical_snr = compute_empirical_snr(
        modality_name,
        n_sensors=n_sensors,
        tier=tier,
    )
    matrix_norm = float(np.linalg.norm(np.asarray(A, dtype=np.float64)))
    if matrix_norm <= 0:
        raise ValueError(f"Cannot empirically normalize zero matrix for {modality_name}")

    noise = matrix_norm / empirical_snr
    model = get_noise_model(modality_name)
    return noise, {
        "scaling": "empirical",
        "empirical_snr": empirical_snr,
        "typical_signal_amplitude": model.typical_signal_amplitude,
        "matrix_frobenius_norm": matrix_norm,
        "noise_used": noise,
        "noise_interpretation": (
            "Frobenius(A)/empirical_snr; matches export_svd_json.py "
            "empirical bitrate normalization"
        ),
    }


def choose_noise(
    A: np.ndarray,
    modality_name: str,
    n_sensors: int,
    physical_noise: float,
    scaling: str,
) -> tuple[float, dict]:
    if scaling == "empirical":
        return empirical_noise_for_matrix(A, modality_name, n_sensors)
    if scaling == "physical":
        return physical_noise, {
            "scaling": "physical",
            "noise_used": physical_noise,
            "noise_interpretation": "detector noise in forward-model measurement units",
        }
    raise ValueError(f"Unknown scaling {scaling!r}; expected one of {SCALING_CHOICES}")


def summarize_by_depth(depth: np.ndarray, info_bits: np.ndarray, bin_width_mm: float):
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


def plot_depth_profile(name: str, profile: dict[str, np.ndarray]) -> None:
    x = profile["bin_centers_mm"]
    series = [
        ("mean", profile["mean_bits"]),
        ("median", profile["median_bits"]),
        ("p90", profile["p90_bits"]),
    ]

    for scale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        add_anatomy_depth_bands(ax)
        for label, values in series:
            y = values.copy()
            if scale == "log":
                y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, label=label)

        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylabel("Information (bits/sample/voxel, log scale)")
        else:
            ax.set_ylabel("Information (bits/sample/voxel)")

        ax.set_xlabel("Depth from brain surface (mm)")
        ax.set_title(f"{name}: information by radial depth", pad=34)
        ax.set_xlim(0, BRAIN_RADIUS)
        ax.grid(True, alpha=0.3, which="both")
        add_plot_legends(ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}_depth_profile_{scale}.png", dpi=180, bbox_inches="tight")
        if scale == "linear":
            fig.savefig(OUT_DIR / f"{name}_depth_profile.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def anatomy_band_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=band["color"],
            edgecolor="none",
            alpha=0.35,
            label=f"{band['label']} ({band['start_mm']:g}-{band['end_mm']:g} mm)",
        )
        for band in ANATOMICAL_DEPTH_BANDS
    ]


def add_anatomy_depth_bands(ax) -> None:
    """Add approximate radial-depth anatomy bands behind the data."""
    xaxis_transform = ax.get_xaxis_transform()
    for band in ANATOMICAL_DEPTH_BANDS:
        start = band["start_mm"]
        end = band["end_mm"]
        width = end - start
        ax.axvspan(start, end, color=band["color"], alpha=0.38, lw=0, zorder=0)
        ax.axvline(end, color="#4b5563", linewidth=1.0, alpha=0.55, zorder=1)
        ax.add_patch(
            Rectangle(
                (start, 1.015),
                width,
                0.055,
                transform=xaxis_transform,
                facecolor=band["color"],
                edgecolor="#4b5563",
                linewidth=0.6,
                alpha=0.92,
                clip_on=False,
                zorder=5,
            )
        )

        label = f"{band['label']}\n{start:g}-{end:g} mm"
        ax.text(
            start + width / 2,
            1.045,
            label,
            transform=xaxis_transform,
            ha="center",
            va="center",
            rotation=90 if width < 8 else 0,
            fontsize=8.0 if width < 8 else 8.7,
            color="#111827",
            clip_on=False,
            zorder=6,
        )

    ax.text(
        1.0,
        -0.34,
        "Anatomical bands are approximate radial-depth regions, not segmented anatomy.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#6b7280",
    )


def add_plot_legends(ax) -> None:
    data_legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
    )
    ax.add_artist(data_legend)
    ax.legend(
        handles=anatomy_band_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.6,
    )


def plot_combined_depth_profiles(paths: list[Path], scaling: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        params = json.loads(str(data["params_json"]))
        name = params["modality"]
        loaded.append(
            (
                COMPARISON_LABELS.get(name, name),
                data["depth_bin_centers_mm"],
                data["depth_mean_bits"],
            )
        )

    for scale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        add_anatomy_depth_bands(ax)
        for label, x, values in loaded:
            y = values.copy()
            if scale == "log":
                y = np.where(y > 0, y, np.nan)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                label=label,
                color=MODALITY_COLORS.get(label),
            )

        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylabel("Mean information (bits/sample/voxel, log scale)")
        else:
            ax.set_ylabel("Mean information (bits/sample/voxel)")

        ax.set_xlabel("Depth from brain surface (mm)")
        ax.set_title(f"Posterior information by radial depth ({scaling} scaling)", pad=34)
        ax.set_xlim(0, BRAIN_RADIUS)
        ax.grid(True, alpha=0.3, which="both")
        add_plot_legends(ax)
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / f"all_modalities_depth_mean_{scaling}_{scale}.png",
            dpi=180,
            bbox_inches="tight",
        )
        if scaling == DEFAULT_SCALING:
            fig.savefig(
                OUT_DIR / f"all_modalities_depth_mean_{scale}.png",
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(fig)


def save_result(
    name: str,
    positions: np.ndarray,
    info_bits: np.ndarray,
    posterior_measure: np.ndarray,
    params: dict,
    posterior_measure_name: str,
    depth_bin_width_mm: float,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = depth_mm(positions)
    profile = summarize_by_depth(d, info_bits, depth_bin_width_mm)
    out_path = OUT_DIR / f"{name}_posterior_info.npz"
    params_with_context = {
        **params,
        "anatomical_depth_bands_mm": ANATOMICAL_DEPTH_BAND_META,
    }
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
        depth_bin_width_mm=depth_bin_width_mm,
        params_json=json.dumps(params_with_context, sort_keys=True),
    )

    plot_depth_profile(name, profile)

    return out_path


def run_meg(
    name: str,
    n_sensors: int,
    grid_spacing_mm: float,
    offset_mm: float,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    model = get_noise_model(name)
    physical_noise = compute_detector_noise_std(name, n_sensors=n_sensors, tier="today")
    A_raw = compute_meg_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
    positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    A = A_raw * model.source_amplitude
    noise, scaling_params = choose_noise(A, name, n_sensors, physical_noise, scaling)
    posterior_det, info_bits = posterior_info_vector3(A, noise, len(positions))
    return save_result(
        name,
        positions,
        info_bits,
        posterior_det,
        {
            "modality": name,
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "grid_spacing_mm": grid_spacing_mm,
            "sensor_offset_mm": offset_mm,
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "model": "Sarvas MEG, 3 dipole orientations per voxel",
            "sensor_model": "triaxial magnetic field components at each sensor position",
            **scaling_params,
        },
        "posterior_cov_det",
        depth_bin_width_mm,
    )


def run_eeg(
    n_sensors: int,
    grid_spacing_mm: float,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    physical_noise = compute_detector_noise_std(
        "eeg_openmeeg",
        n_sensors=n_sensors,
        tier="today",
    )
    source_amplitude_a_m = 10e-9
    A_raw, positions = compute_eeg_forward_matrix(n_sensors, grid_spacing_mm)
    A = A_raw * source_amplitude_a_m
    noise, scaling_params = choose_noise(
        A,
        "eeg_openmeeg",
        n_sensors,
        physical_noise,
        scaling,
    )
    posterior_det, info_bits = posterior_info_vector3(A, noise, len(positions))
    return save_result(
        "eeg_homogeneous",
        positions,
        info_bits,
        posterior_det,
        {
            "modality": "eeg_homogeneous",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "grid_spacing_mm": grid_spacing_mm,
            "detector_noise_today": physical_noise,
            "source_amplitude_a_m": source_amplitude_a_m,
            "model": (
                "homogeneous quasi-static dipole potential; OpenMEEG is not "
                "used because leadfields/OpenMEEG binaries are unavailable"
            ),
            "calibration_modality": "eeg_openmeeg",
            **scaling_params,
        },
        "posterior_cov_det",
        depth_bin_width_mm,
    )


def run_fnirs(
    n_sensors: int,
    grid_spacing_mm: float,
    max_dist_mm: float,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    model = get_noise_model("cw_fnirs")
    physical_noise = compute_detector_noise_std(
        "cw_fnirs",
        n_sensors=n_sensors,
        tier="today",
    )
    modality = CWfNIRS(
        Parameters(
            num_sensors=n_sensors,
            grid_resolution_mm=grid_spacing_mm,
            max_dist=max_dist_mm,
        )
    )
    modality.setup_geometry()
    A_transfer = as_numpy(modality.compute_forward_model())
    positions = modality.grid_points
    voxel_volume_mm3 = grid_spacing_mm**3
    A = np.asarray(A_transfer, dtype=np.float64) * model.source_amplitude
    noise, scaling_params = choose_noise(
        A,
        "cw_fnirs",
        n_sensors,
        physical_noise,
        scaling,
    )
    posterior_var, info_bits = posterior_info_scalar(A, noise)
    return save_result(
        "cw_fnirs",
        positions,
        info_bits,
        posterior_var,
        {
            "modality": "cw_fnirs",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "n_unique_source_detector_pairs": int(A.shape[0]),
            "grid_spacing_mm": grid_spacing_mm,
            "max_dist_mm": max_dist_mm,
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "voxel_volume_mm3": voxel_volume_mm3,
            "transfer_function_units": "mm^-1",
            "transfer_function_convention": "voxel-integrated before absorption scaling",
            "model": (
                "CW fNIRS analytical diffusion sensitivity; scalar absorption "
                "contrast integrated over each voxel"
            ),
            **scaling_params,
        },
        "posterior_variance",
        depth_bin_width_mm,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-spacing-mm", type=float, default=10.0)
    parser.add_argument("--eeg-sensors", type=int, default=256)
    parser.add_argument("--meg-sensors", type=int, default=200)
    parser.add_argument("--fnirs-sensors", type=int, default=100)
    parser.add_argument("--fnirs-grid-spacing-mm", type=float, default=6.0)
    parser.add_argument("--fnirs-max-dist-mm", type=float, default=50.0)
    parser.add_argument("--depth-bin-width-mm", type=float, default=2.0)
    parser.add_argument(
        "--scaling",
        choices=SCALING_CHOICES,
        default=DEFAULT_SCALING,
        help=(
            "empirical matches the web bitrate normalization; physical uses "
            "raw source-amplitude and detector-noise units"
        ),
    )
    args = parser.parse_args()

    outputs = []
    outputs.append(
        run_eeg(
            args.eeg_sensors,
            args.grid_spacing_mm,
            args.depth_bin_width_mm,
            args.scaling,
        )
    )
    outputs.append(
        run_meg(
            "meg_opm",
            args.meg_sensors,
            args.grid_spacing_mm,
            7.0,
            args.depth_bin_width_mm,
            args.scaling,
        )
    )
    outputs.append(
        run_meg(
            "meg_squid",
            args.meg_sensors,
            args.grid_spacing_mm,
            25.0,
            args.depth_bin_width_mm,
            args.scaling,
        )
    )
    outputs.append(
        run_fnirs(
            args.fnirs_sensors,
            args.fnirs_grid_spacing_mm,
            args.fnirs_max_dist_mm,
            args.depth_bin_width_mm,
            args.scaling,
        )
    )
    plot_combined_depth_profiles(outputs, args.scaling)
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
