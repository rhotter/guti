#!/usr/bin/env python3
"""Compute posterior information maps for representative GUTI modalities.

The model is:

    y = A x + n
    x ~ N(0, I)
    n ~ N(0, noise^2 I)

By default the maps use the physical detector-floor path: the forward model is
scaled by the source amplitude and compared directly to the detector noise floor.
This preserves the raw forward gain, matching the first-principles capacity
interpretation used for the blog figures.

Use --scaling empirical to inspect the observed-SNR diagnostic path instead.  In
that mode the whole forward matrix is scaled relative to a noise floor such that
sqrt(sum(s_i^2)) / noise equals the modality's empirical SNR, which normalizes
away absolute forward gain.
For EEG/MEG, each voxel has a 3-vector dipole source and the voxel score is the
mutual information for that 3D block.  For fNIRS, each voxel is scalar absorption
contrast integrated over the voxel volume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from guti.core import (
    BRAIN_RADIUS,
    CSF_RADIUS,
    SCALP_RADIUS,
    SKULL_RADIUS,
    get_grid_positions,
    get_sensor_positions,
)
from guti.modalities.cw_fnirs.modality import CWfNIRS
from guti.modalities.td_fnirs.modality import TDfNIRSAnalytical
from guti.noise_models import (
    compute_detector_noise_std,
    compute_empirical_snr,
    get_noise_model,
)
from guti.parameters import Parameters


OUT_DIR = Path("results/information_maps")
HEAD_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
OUTER_SCALP_DEPTH_MM = BRAIN_RADIUS - SCALP_RADIUS
SCALING_CHOICES = ("empirical", "physical")
MAP_KIND_CHOICES = ("posterior", "capacity", "both")
DEFAULT_SCALING = "physical"
SCALING_LABELS = {
    "physical": "physical detector-floor",
    "empirical": "empirical observed-SNR",
}
TIME_RESOLUTION_S = {
    "eeg_openmeeg": 0.01,
    "eeg_homogeneous": 0.01,
    "meg_opm": 0.01,
    "meg_squid": 0.01,
    "cw_fnirs": 0.1,
    "td_fnirs_analytical": 0.1,
}
TABLE_DEFAULTS = {
    "eeg_openmeeg": {
        "n_sensors": 256,
        "grid_spacing_mm": 5.0,
        "mesh_resolution_mm": 10.0,
        "leadfield_path": "guti/modalities/leadfields/eeg/eeg_leadfield.mat",
        "dipoles_path": "guti/modalities/bem_model/eeg/dipole_locations.txt",
        "reference": "results/variants/eeg_openmeeg/cedd0dd5.npz",
    },
    "eeg_homogeneous": {
        "n_sensors": 256,
        "grid_spacing_mm": 5.0,
        "reference": (
            "Lightweight homogeneous-sphere EEG approximation; retained as a "
            "diagnostic fallback, not used in the default combined chart."
        ),
    },
    "meg_opm": {
        "n_sensors": 1000,
        "grid_spacing_mm": 3.0,
        "sensor_offset_mm": 5.0,
        "reference": "results/meg_opm_svd_spectrum.npz",
    },
    "meg_squid": {
        "n_sensors": 1000,
        "grid_spacing_mm": 3.0,
        "sensor_offset_mm": 25.0,
        "reference": "results/meg_squid_svd_spectrum.npz",
    },
    "cw_fnirs": {
        "n_sensors": 800,
        "grid_spacing_mm": 6.0,
        "max_dist_mm": 50.0,
        "reference": "results/cw_fnirs_svd_spectrum.npz",
    },
    "td_fnirs_analytical": {
        "n_sensors": 400,
        "grid_spacing_mm": 4.0,
        "max_dist_mm": 50.0,
        "n_time_gates": 6,
        "reference": "results/td_fnirs_analytical_svd_spectrum.npz",
    },
}
MODALITY_ALIASES = {
    "all": "all",
    "eeg": "eeg_openmeeg",
    "eeg_openmeeg": "eeg_openmeeg",
    "eeg_homogeneous": "eeg_homogeneous",
    "homogeneous_eeg": "eeg_homogeneous",
    "meg_opm": "meg_opm",
    "opm": "meg_opm",
    "meg_squid": "meg_squid",
    "squid": "meg_squid",
    "fnirs": "cw_fnirs",
    "fnirs_cw": "cw_fnirs",
    "cw_fnirs": "cw_fnirs",
    "cw_fnirs": "cw_fnirs",
    "td_fnirs": "td_fnirs_analytical",
    "td-fnirs": "td_fnirs_analytical",
    "td_fnirs_analytical": "td_fnirs_analytical",
}
DEFAULT_MODALITIES = (
    "eeg_openmeeg",
    "meg_opm",
    "meg_squid",
    "cw_fnirs",
    "td_fnirs_analytical",
)
CAPACITY_ATTRIBUTION_DEFAULT_MODALITIES = (
    "eeg_openmeeg",
    "meg_opm",
    "meg_squid",
    "cw_fnirs",
)
MODALITY_ORDER = (
    "eeg_openmeeg",
    "eeg_homogeneous",
    "meg_opm",
    "meg_squid",
    "cw_fnirs",
    "td_fnirs_analytical",
)
COMPARISON_LABELS = {
    "eeg_openmeeg": "EEG OpenMEEG",
    "eeg_homogeneous": "EEG homogeneous",
    "meg_opm": "MEG OPM",
    "meg_squid": "MEG SQUID",
    "cw_fnirs": "fNIRS CW",
    "td_fnirs_analytical": "TD-fNIRS",
}
MODALITY_COLORS = {
    "EEG OpenMEEG": "#2563eb",
    "EEG homogeneous": "#60a5fa",
    "MEG OPM": "#ea580c",
    "MEG SQUID": "#16a34a",
    "fNIRS CW": "#dc2626",
    "TD-fNIRS": "#9333ea",
}
ANATOMICAL_DEPTH_BANDS = [
    {
        "label": "scalp / skin",
        "start_mm": float(BRAIN_RADIUS - SCALP_RADIUS),
        "end_mm": float(BRAIN_RADIUS - SKULL_RADIUS),
        "color": "#f5d7b5",
    },
    {
        "label": "skull",
        "start_mm": float(BRAIN_RADIUS - SKULL_RADIUS),
        "end_mm": float(BRAIN_RADIUS - CSF_RADIUS),
        "color": "#d7c2a6",
    },
    {
        "label": "CSF / meninges",
        "start_mm": float(BRAIN_RADIUS - CSF_RADIUS),
        "end_mm": 0.0,
        "color": "#b7d7ea",
    },
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


def load_openmeeg_leadfield(leadfield_path: str | Path) -> np.ndarray:
    """Load an OpenMEEG HDF5 leadfield matrix from a .mat file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading OpenMEEG .mat leadfields requires h5py. Install it in the "
            "active environment or run with an environment that includes h5py."
        ) from exc

    path = Path(leadfield_path)
    with h5py.File(path, "r") as f:
        for key in ("linop", "matrix"):
            if key in f:
                return np.asarray(f[key], dtype=np.float64)
        raise ValueError(
            f"{path} does not contain an OpenMEEG leadfield dataset named "
            "'linop' or 'matrix'; found {sorted(f.keys())}"
        )


def compute_eeg_openmeeg_forward_matrix(
    leadfield_path: str | Path,
    dipoles_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load OpenMEEG EEG leadfield as sensors x (3 orientations per voxel)."""
    leadfield = load_openmeeg_leadfield(leadfield_path)
    dipoles = np.loadtxt(dipoles_path, dtype=np.float64)
    if dipoles.ndim != 2 or dipoles.shape[1] != 6:
        raise ValueError(
            f"Expected dipoles file with columns x y z ox oy oz; got {dipoles.shape}"
        )
    if len(dipoles) % 3:
        raise ValueError(
            f"OpenMEEG dipoles must come in x/y/z orientation triples; got {len(dipoles)}"
        )

    n_dipoles = len(dipoles)
    if leadfield.shape[0] == n_dipoles:
        A_raw = leadfield.T
    elif leadfield.shape[1] == n_dipoles:
        A_raw = leadfield
    else:
        raise ValueError(
            "OpenMEEG leadfield shape does not match dipole count: "
            f"leadfield={leadfield.shape}, dipoles={n_dipoles}"
        )

    grouped_positions = dipoles[:, :3].reshape(-1, 3, 3)
    if not np.allclose(grouped_positions, grouped_positions[:, :1, :]):
        raise ValueError("OpenMEEG dipole positions are not grouped in orientation triples")

    grouped_orientations = dipoles[:, 3:].reshape(-1, 3, 3)
    if not np.allclose(grouped_orientations, np.eye(3)[None, :, :]):
        raise ValueError("OpenMEEG dipole orientations are not x/y/z triples")

    return np.asarray(A_raw, dtype=np.float64), grouped_positions[:, 0, :]


def reference_svd_metadata(A_raw: np.ndarray, reference_path: str | Path) -> dict:
    """Compare a loaded OpenMEEG leadfield against its saved SVD reference."""
    path = Path(reference_path)
    if not path.exists():
        return {"reference_svd_path": str(path), "reference_svd_status": "missing"}

    saved = np.load(path, allow_pickle=True)["singular_values"]
    current = np.linalg.svd(np.asarray(A_raw, dtype=np.float64), compute_uv=False)
    width = min(len(current), len(saved))
    rel = np.abs(current[:width] - saved[:width]) / np.maximum(
        np.abs(saved[:width]),
        1e-300,
    )
    return {
        "reference_svd_path": str(path),
        "reference_svd_status": "compared",
        "reference_svd_max_relative_error": float(np.max(rel)),
        "reference_svd_max_absolute_error": float(
            np.max(np.abs(current[:width] - saved[:width]))
        ),
        "reference_svd_current_first3": list(map(float, current[:3])),
        "reference_svd_saved_first3": list(map(float, saved[:3])),
    }


def compute_meg_forward_matrix(
    n_sensors: int,
    grid_spacing_mm: float,
    offset_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Sarvas MEG matrix, 3 field components x 3 dipole components."""
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    A = np.empty((3 * n_sensors, 3 * n_sources), dtype=np.float64)
    center = HEAD_CENTER
    coeff = 1e-7

    sources_m = (sources - center) * 1e-3
    for i, sensor in enumerate(sensors):
        r = (sensor - center) * 1e-3
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
                av = av[valid_f]
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
            3, 3 * n_sources
        )
    return A, sources


def posterior_diag_from_precision(
    precision: np.ndarray,
    chunk_cols: int,
) -> np.ndarray:
    """Diagonal of inv(precision), computed in chunks after one Cholesky."""
    n = precision.shape[0]
    chol = cho_factor(precision, lower=True, check_finite=False)
    diag = np.empty(n, dtype=np.float64)
    eye_chunk = np.zeros((n, min(chunk_cols, n)), dtype=np.float64)
    for start in range(0, n, chunk_cols):
        stop = min(start + chunk_cols, n)
        width = stop - start
        rhs = eye_chunk[:, :width]
        rhs.fill(0.0)
        rhs[start + np.arange(width), np.arange(width)] = 1.0
        solved = cho_solve(chol, rhs, check_finite=False)
        diag[start:stop] = solved[np.arange(start, stop), np.arange(width)]
    return diag


def posterior_info_scalar(
    A: np.ndarray,
    noise: float,
    chunk_cols: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar voxel posterior variance and information in bits/sample."""
    A64 = np.asarray(A, dtype=np.float64)
    m, n = A64.shape
    if n <= m:
        precision = np.eye(n, dtype=np.float64)
        precision += (A64.T @ A64) / noise**2
        posterior_var = posterior_diag_from_precision(precision, chunk_cols)
        posterior_var = np.clip(posterior_var, 1e-15, 1.0)
        info_bits = -0.5 * np.log2(posterior_var)
        return posterior_var, info_bits

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
    chunk_voxels: int = 512,
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

    for start in range(0, n_voxels, chunk_voxels):
        stop = min(start + chunk_voxels, n_voxels)
        n_chunk = stop - start
        cols = slice(3 * start, 3 * stop)
        block = A64[:, cols]
        solved = cho_solve(chol, block, check_finite=False)
        block3 = block.reshape(m, n_chunk, 3)
        solved3 = solved.reshape(m, n_chunk, 3)
        posterior_blocks = eye3[None, :, :] - np.einsum(
            "mvi,mvj->vij",
            block3,
            solved3,
            optimize=True,
        )
        posterior_blocks = 0.5 * (
            posterior_blocks + np.swapaxes(posterior_blocks, 1, 2)
        )
        sign, logdet = np.linalg.slogdet(posterior_blocks)
        bad = sign <= 0
        if np.any(bad):
            eig = np.linalg.eigvalsh(posterior_blocks[bad])
            logdet[bad] = np.sum(np.log(np.clip(eig, 1e-15, 1.0)), axis=1)
        posterior_det[start:stop] = np.exp(logdet)
        info_bits[start:stop] = -0.5 * logdet / np.log(2.0)

    return posterior_det, info_bits


def mode_bits_from_singular_values(
    singular_values: np.ndarray,
    noise: float,
) -> np.ndarray:
    """Capacity contribution of each singular mode in bits/sample."""
    s = np.asarray(singular_values, dtype=np.float64)
    return 0.5 * np.log2(1.0 + (s / noise) ** 2)


def _positive_eigen_spectrum(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(eigvals)[::-1]
    eigvals = np.asarray(eigvals[order], dtype=np.float64)
    eigvecs = np.asarray(eigvecs[:, order], dtype=np.float64)
    eig_floor = max(float(eigvals[0]) * 1e-14, 0.0)
    positive = eigvals > eig_floor
    return np.sqrt(np.clip(eigvals[positive], 0.0, None)), eigvecs[:, positive]


def capacity_attribution_vector3(
    A: np.ndarray,
    noise: float,
    n_voxels: int,
    chunk_voxels: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attribute SVD mode capacity to 3-orientation voxels by right-mode energy.

    The returned voxel attributions sum to ``sum(mode_bits)`` up to numerical
    precision.  This is a spatial attribution of the joint SVD capacity, not a
    marginal posterior-identifiability score.
    """
    A64 = np.asarray(A, dtype=np.float64)
    m, n = A64.shape
    if n != 3 * n_voxels:
        raise ValueError(f"Expected {3 * n_voxels} source columns, got {n}")

    if m <= n:
        eigvals, U = np.linalg.eigh(A64 @ A64.T)
        singular_values, U = _positive_eigen_spectrum(eigvals, U)
        mode_bits = mode_bits_from_singular_values(singular_values, noise)
        attribution = np.zeros(n_voxels, dtype=np.float64)
        inv_s = 1.0 / singular_values
        for start in range(0, n_voxels, chunk_voxels):
            stop = min(start + chunk_voxels, n_voxels)
            cols = slice(3 * start, 3 * stop)
            projected = U.T @ A64[:, cols]
            right_energy = (projected * inv_s[:, None]) ** 2
            right_energy = right_energy.reshape(len(singular_values), stop - start, 3)
            attribution[start:stop] = mode_bits @ right_energy.sum(axis=2)
        return attribution, singular_values, mode_bits

    eigvals, V = np.linalg.eigh(A64.T @ A64)
    singular_values, V = _positive_eigen_spectrum(eigvals, V)
    mode_bits = mode_bits_from_singular_values(singular_values, noise)
    right_energy = (V**2).reshape(n_voxels, 3, len(singular_values)).sum(axis=1)
    return right_energy @ mode_bits, singular_values, mode_bits


def capacity_attribution_scalar(
    A: np.ndarray,
    noise: float,
    chunk_cols: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attribute SVD mode capacity to scalar voxels by right-mode energy."""
    A64 = np.asarray(A, dtype=np.float64)
    m, n = A64.shape

    if n <= m:
        eigvals, V = np.linalg.eigh(A64.T @ A64)
        singular_values, V = _positive_eigen_spectrum(eigvals, V)
        mode_bits = mode_bits_from_singular_values(singular_values, noise)
        return (V**2) @ mode_bits, singular_values, mode_bits

    eigvals, U = np.linalg.eigh(A64 @ A64.T)
    singular_values, U = _positive_eigen_spectrum(eigvals, U)
    mode_bits = mode_bits_from_singular_values(singular_values, noise)
    attribution = np.zeros(n, dtype=np.float64)
    inv_s = 1.0 / singular_values
    for start in range(0, n, chunk_cols):
        stop = min(start + chunk_cols, n)
        projected = U.T @ A64[:, start:stop]
        right_energy = (projected * inv_s[:, None]) ** 2
        attribution[start:stop] = mode_bits @ right_energy
    return attribution, singular_values, mode_bits


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
        "noise_model": "empirical_observed_snr",
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
            "noise_model": "physical_detector_floor",
            "detector_noise_std": physical_noise,
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
        if i == len(centers) - 1:
            mask = (depth >= bins[i]) & (depth <= bins[i + 1])
        else:
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


def normalize_to_first_in_brain_value(
    x: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalize a depth profile to the first finite positive in-brain value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(y) & (y > 0) & (x >= 0)
    if not np.any(valid):
        valid = np.isfinite(y) & (y > 0)
    if not np.any(valid):
        return np.full_like(y, np.nan, dtype=np.float64), np.nan

    reference = float(y[np.flatnonzero(valid)[0]])
    return y / reference, reference


def plot_depth_profile(name: str, profile: dict[str, np.ndarray], scaling: str) -> None:
    x = profile["bin_centers_mm"]
    series = [
        ("mean", profile["mean_bits"]),
        ("median", profile["median_bits"]),
        ("p90", profile["p90_bits"]),
    ]
    stems = [f"{name}_{scaling}"]
    if scaling == DEFAULT_SCALING:
        stems.append(name)

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

        ax.set_xlabel("Radial position relative to brain surface (mm)")
        ax.set_title(
            f"{name}: information by radial depth "
            f"({SCALING_LABELS.get(scaling, scaling)} scaling)",
            pad=14,
        )
        ax.set_xlim(OUTER_SCALP_DEPTH_MM, BRAIN_RADIUS)
        ax.grid(True, alpha=0.3, which="both")
        add_plot_legends(ax)
        fig.tight_layout()
        for stem in stems:
            fig.savefig(
                OUT_DIR / f"{stem}_depth_profile_{scale}.png",
                dpi=180,
                bbox_inches="tight",
            )
            if scale == "linear":
                fig.savefig(
                    OUT_DIR / f"{stem}_depth_profile.png",
                    dpi=180,
                    bbox_inches="tight",
                )
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
    for band in ANATOMICAL_DEPTH_BANDS:
        start = band["start_mm"]
        end = band["end_mm"]
        ax.axvspan(start, end, color=band["color"], alpha=0.32, lw=0, zorder=0)
        ax.axvline(end, color="#4b5563", linewidth=0.9, alpha=0.45, zorder=1)

    ax.axvline(0, color="#111827", linewidth=1.3, alpha=0.75, zorder=1)
    ax.text(
        1.0,
        -0.38,
        "Negative depths are outside brain. Shaded bands are approximate spherical head layers, not segmented anatomy.",
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
        fontsize=8.0,
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
                data["depth_bin_count"],
            )
        )

    def plot_combined(normalized: bool, aggregation: str) -> None:
        if aggregation == "mean":
            title_base = "posterior information"
            y_base = "Mean information"
            filename_metric = "mean"
        elif aggregation == "total":
            title_base = "total posterior information"
            y_base = "Total information"
            filename_metric = "total"
        else:
            raise ValueError(f"Unknown aggregation {aggregation!r}")

        for scale in ("linear", "log"):
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            add_anatomy_depth_bands(ax)
            for label, x, mean_values, counts in loaded:
                if aggregation == "mean":
                    y = mean_values.copy()
                else:
                    y = mean_values * counts
                if normalized:
                    y, _ = normalize_to_first_in_brain_value(x, y)
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

            if normalized:
                if scale == "log":
                    ax.set_yscale("log")
                    ax.set_ylabel(f"Relative {y_base.lower()} (first bin = 1, log scale)")
                else:
                    ax.set_ylabel(f"Relative {y_base.lower()} (first bin = 1)")
                title = f"Relative {title_base} by radial depth"
                suffix = "normalized"
            else:
                if scale == "log":
                    ax.set_yscale("log")
                    if aggregation == "mean":
                        ax.set_ylabel("Mean information (bits/sample/voxel, log scale)")
                    else:
                        ax.set_ylabel("Total information (bits/sample/depth bin, log scale)")
                else:
                    if aggregation == "mean":
                        ax.set_ylabel("Mean information (bits/sample/voxel)")
                    else:
                        ax.set_ylabel("Total information (bits/sample/depth bin)")
                title = f"{title_base.capitalize()} by radial depth"
                suffix = None

            ax.set_xlabel("Radial position relative to brain surface (mm)")
            ax.set_title(f"{title} ({SCALING_LABELS.get(scaling, scaling)} scaling)", pad=14)
            ax.set_xlim(OUTER_SCALP_DEPTH_MM, BRAIN_RADIUS)
            ax.grid(True, alpha=0.3, which="both")
            add_plot_legends(ax)
            fig.tight_layout()
            filename_parts = ["all_modalities_depth", filename_metric]
            if suffix is not None:
                filename_parts.append(suffix)
            filename_parts.extend([scaling, scale])
            fig.savefig(
                OUT_DIR / f"{'_'.join(filename_parts)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            if scaling == DEFAULT_SCALING:
                alias_parts = ["all_modalities_depth", filename_metric]
                if suffix is not None:
                    alias_parts.append(suffix)
                alias_parts.append(scale)
                fig.savefig(
                    OUT_DIR / f"{'_'.join(alias_parts)}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
            plt.close(fig)

    for aggregation in ("mean", "total"):
        plot_combined(normalized=False, aggregation=aggregation)
        plot_combined(normalized=True, aggregation=aggregation)


def plot_combined_capacity_attribution_profiles(paths: list[Path], scaling: str) -> None:
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
                data["depth_bin_count"],
            )
        )

    def plot_combined(normalized: bool, aggregation: str) -> None:
        if aggregation == "mean":
            title_base = "SVD capacity attribution"
            y_base = "Mean attributed capacity"
            filename_metric = "mean"
        elif aggregation == "total":
            title_base = "SVD capacity attribution"
            y_base = "Attributed capacity"
            filename_metric = "total"
        else:
            raise ValueError(f"Unknown aggregation {aggregation!r}")

        for scale in ("linear", "log"):
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            add_anatomy_depth_bands(ax)
            for label, x, mean_values, counts in loaded:
                if aggregation == "mean":
                    y = mean_values.copy()
                else:
                    y = mean_values * counts
                if normalized:
                    y, _ = normalize_to_first_in_brain_value(x, y)
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

            if normalized:
                if scale == "log":
                    ax.set_yscale("log")
                    ax.set_ylabel(f"Relative {y_base.lower()} (first bin = 1, log scale)")
                else:
                    ax.set_ylabel(f"Relative {y_base.lower()} (first bin = 1)")
                title = f"Relative {title_base} by radial depth"
                suffix = "normalized"
            else:
                if scale == "log":
                    ax.set_yscale("log")
                    if aggregation == "mean":
                        ax.set_ylabel(
                            "Mean attributed capacity (bits/sample/voxel, log scale)"
                        )
                    else:
                        ax.set_ylabel(
                            "Attributed capacity (bits/sample/depth bin, log scale)"
                        )
                else:
                    if aggregation == "mean":
                        ax.set_ylabel("Mean attributed capacity (bits/sample/voxel)")
                    else:
                        ax.set_ylabel("Attributed capacity (bits/sample/depth bin)")
                title = f"{title_base} by radial depth"
                suffix = None

            ax.set_xlabel("Radial position relative to brain surface (mm)")
            ax.set_title(f"{title} ({SCALING_LABELS.get(scaling, scaling)} scaling)", pad=14)
            ax.set_xlim(OUTER_SCALP_DEPTH_MM, BRAIN_RADIUS)
            ax.grid(True, alpha=0.3, which="both")
            add_plot_legends(ax)
            fig.tight_layout()
            filename_parts = ["all_modalities_depth_capacity", filename_metric]
            if suffix is not None:
                filename_parts.append(suffix)
            filename_parts.extend([scaling, scale])
            fig.savefig(
                OUT_DIR / f"{'_'.join(filename_parts)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            if scaling == DEFAULT_SCALING:
                alias_parts = ["all_modalities_depth_capacity", filename_metric]
                if suffix is not None:
                    alias_parts.append(suffix)
                alias_parts.append(scale)
                fig.savefig(
                    OUT_DIR / f"{'_'.join(alias_parts)}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
            plt.close(fig)

    for aggregation in ("mean", "total"):
        plot_combined(normalized=False, aggregation=aggregation)
        plot_combined(normalized=True, aggregation=aggregation)


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
    scaling = params.get("scaling", DEFAULT_SCALING)
    out_path = OUT_DIR / f"{name}_posterior_info_{scaling}.npz"
    params_with_context = {
        **params,
        "anatomical_depth_bands_mm": ANATOMICAL_DEPTH_BAND_META,
    }
    payload = dict(
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
    np.savez(out_path, **payload)
    if scaling == DEFAULT_SCALING:
        np.savez(OUT_DIR / f"{name}_posterior_info.npz", **payload)

    plot_depth_profile(name, profile, scaling)

    return out_path


def save_capacity_attribution_result(
    name: str,
    positions: np.ndarray,
    capacity_bits: np.ndarray,
    singular_values: np.ndarray,
    mode_bits: np.ndarray,
    params: dict,
    depth_bin_width_mm: float,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = depth_mm(positions)
    profile = summarize_by_depth(d, capacity_bits, depth_bin_width_mm)
    scaling = params.get("scaling", DEFAULT_SCALING)
    out_path = OUT_DIR / f"{name}_capacity_attribution_{scaling}.npz"
    params_with_context = {
        **params,
        "map_kind": "svd_capacity_attribution",
        "anatomical_depth_bands_mm": ANATOMICAL_DEPTH_BAND_META,
        "capacity_total_bits_per_sample": float(np.sum(mode_bits)),
        "attribution_total_bits_per_sample": float(np.sum(capacity_bits)),
        "capacity_total_bits_per_second": float(
            np.sum(mode_bits) / TIME_RESOLUTION_S.get(name, 1.0)
        ),
        "time_resolution_s": TIME_RESOLUTION_S.get(name),
    }
    payload = dict(
        positions_mm=positions,
        depth_mm=d,
        capacity_bits_per_sample=capacity_bits,
        info_bits_per_sample=capacity_bits,
        singular_values=singular_values,
        mode_bits_per_sample=mode_bits,
        depth_bin_centers_mm=profile["bin_centers_mm"],
        depth_mean_bits=profile["mean_bits"],
        depth_median_bits=profile["median_bits"],
        depth_p90_bits=profile["p90_bits"],
        depth_bin_count=profile["count"],
        depth_bin_width_mm=depth_bin_width_mm,
        params_json=json.dumps(params_with_context, sort_keys=True),
    )
    np.savez(out_path, **payload)
    if scaling == DEFAULT_SCALING:
        np.savez(OUT_DIR / f"{name}_capacity_attribution.npz", **payload)
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
    A_raw, positions = compute_meg_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
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
            "table_default_reference": TABLE_DEFAULTS[name]["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "model": "Sarvas MEG, 3 dipole orientations per voxel",
            "sensor_model": "triaxial magnetic field components at each sensor position",
            **scaling_params,
        },
        "posterior_cov_det",
        depth_bin_width_mm,
    )


def run_meg_capacity_attribution(
    name: str,
    n_sensors: int,
    grid_spacing_mm: float,
    offset_mm: float,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    model = get_noise_model(name)
    physical_noise = compute_detector_noise_std(name, n_sensors=n_sensors, tier="today")
    A_raw, positions = compute_meg_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
    A = A_raw * model.source_amplitude
    noise, scaling_params = choose_noise(A, name, n_sensors, physical_noise, scaling)
    capacity_bits, singular_values, mode_bits = capacity_attribution_vector3(
        A,
        noise,
        len(positions),
    )
    return save_capacity_attribution_result(
        name,
        positions,
        capacity_bits,
        singular_values,
        mode_bits,
        {
            "modality": name,
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "grid_spacing_mm": grid_spacing_mm,
            "sensor_offset_mm": offset_mm,
            "table_default_reference": TABLE_DEFAULTS[name]["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "model": (
                "Sarvas MEG; SVD mode capacity attributed to 3-orientation "
                "voxels by right-singular-vector energy"
            ),
            **scaling_params,
        },
        depth_bin_width_mm,
    )


def run_eeg_openmeeg(
    n_sensors: int,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    defaults = TABLE_DEFAULTS["eeg_openmeeg"]
    model = get_noise_model("eeg_openmeeg")
    if n_sensors != defaults["n_sensors"]:
        raise ValueError(
            "The saved OpenMEEG EEG leadfield has "
            f"{defaults['n_sensors']} sensors; got --eeg-sensors={n_sensors}."
        )
    physical_noise = compute_detector_noise_std(
        "eeg_openmeeg",
        n_sensors=n_sensors,
        tier="today",
    )
    A_raw, positions = compute_eeg_openmeeg_forward_matrix(
        defaults["leadfield_path"],
        defaults["dipoles_path"],
    )
    A = A_raw * model.source_amplitude
    noise, scaling_params = choose_noise(
        A,
        "eeg_openmeeg",
        n_sensors,
        physical_noise,
        scaling,
    )
    posterior_det, info_bits = posterior_info_vector3(A, noise, len(positions))
    return save_result(
        "eeg_openmeeg",
        positions,
        info_bits,
        posterior_det,
        {
            "modality": "eeg_openmeeg",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "n_source_locations": int(len(positions)),
            "n_dipole_orientations_per_location": 3,
            "grid_spacing_mm": defaults["grid_spacing_mm"],
            "mesh_resolution_mm": defaults["mesh_resolution_mm"],
            "mesh_generator": "legacy latitude/longitude sphere mesh",
            "leadfield_path": defaults["leadfield_path"],
            "dipoles_path": defaults["dipoles_path"],
            "table_default_reference": defaults["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "source_amplitude_units": model.source_amplitude_units,
            "model": "OpenMEEG BEM EEG leadfield, 3 dipole orientations per voxel",
            **reference_svd_metadata(A_raw, defaults["reference"]),
            **scaling_params,
        },
        "posterior_cov_det",
        depth_bin_width_mm,
    )


def run_eeg_openmeeg_capacity_attribution(
    n_sensors: int,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    defaults = TABLE_DEFAULTS["eeg_openmeeg"]
    model = get_noise_model("eeg_openmeeg")
    if n_sensors != defaults["n_sensors"]:
        raise ValueError(
            "The saved OpenMEEG EEG leadfield has "
            f"{defaults['n_sensors']} sensors; got --eeg-sensors={n_sensors}."
        )
    physical_noise = compute_detector_noise_std(
        "eeg_openmeeg",
        n_sensors=n_sensors,
        tier="today",
    )
    A_raw, positions = compute_eeg_openmeeg_forward_matrix(
        defaults["leadfield_path"],
        defaults["dipoles_path"],
    )
    A = A_raw * model.source_amplitude
    noise, scaling_params = choose_noise(
        A,
        "eeg_openmeeg",
        n_sensors,
        physical_noise,
        scaling,
    )
    capacity_bits, singular_values, mode_bits = capacity_attribution_vector3(
        A,
        noise,
        len(positions),
    )
    return save_capacity_attribution_result(
        "eeg_openmeeg",
        positions,
        capacity_bits,
        singular_values,
        mode_bits,
        {
            "modality": "eeg_openmeeg",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "n_source_locations": int(len(positions)),
            "n_dipole_orientations_per_location": 3,
            "grid_spacing_mm": defaults["grid_spacing_mm"],
            "mesh_resolution_mm": defaults["mesh_resolution_mm"],
            "mesh_generator": "legacy latitude/longitude sphere mesh",
            "leadfield_path": defaults["leadfield_path"],
            "dipoles_path": defaults["dipoles_path"],
            "table_default_reference": defaults["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "source_amplitude_units": model.source_amplitude_units,
            "model": (
                "OpenMEEG BEM EEG; SVD mode capacity attributed to "
                "3-orientation voxels by right-singular-vector energy"
            ),
            **reference_svd_metadata(A_raw, defaults["reference"]),
            **scaling_params,
        },
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
            "table_default_reference": TABLE_DEFAULTS["eeg_homogeneous"]["reference"],
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
            "table_default_reference": TABLE_DEFAULTS["cw_fnirs"]["reference"],
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


def run_fnirs_capacity_attribution(
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
    capacity_bits, singular_values, mode_bits = capacity_attribution_scalar(A, noise)
    return save_capacity_attribution_result(
        "cw_fnirs",
        positions,
        capacity_bits,
        singular_values,
        mode_bits,
        {
            "modality": "cw_fnirs",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "n_unique_source_detector_pairs": int(A.shape[0]),
            "grid_spacing_mm": grid_spacing_mm,
            "max_dist_mm": max_dist_mm,
            "table_default_reference": TABLE_DEFAULTS["cw_fnirs"]["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "voxel_volume_mm3": voxel_volume_mm3,
            "transfer_function_units": "mm^-1",
            "transfer_function_convention": "voxel-integrated before absorption scaling",
            "model": (
                "CW fNIRS analytical diffusion sensitivity; SVD mode capacity "
                "attributed to scalar absorption voxels by right-singular-vector energy"
            ),
            **scaling_params,
        },
        depth_bin_width_mm,
    )


def run_td_fnirs(
    n_sensors: int,
    grid_spacing_mm: float,
    max_dist_mm: float,
    n_time_gates: int,
    depth_bin_width_mm: float,
    scaling: str,
) -> Path:
    model = get_noise_model("td_fnirs_analytical")
    physical_noise = compute_detector_noise_std(
        "td_fnirs_analytical",
        n_sensors=n_sensors,
        tier="today",
    )
    modality = TDfNIRSAnalytical(
        Parameters(
            num_sensors=n_sensors,
            grid_resolution_mm=grid_spacing_mm,
            max_dist=max_dist_mm,
            n_time_gates=n_time_gates,
        )
    )
    modality.setup_geometry()
    A_transfer = as_numpy(modality.compute_forward_model())
    positions = modality.grid_points
    voxel_volume_mm3 = grid_spacing_mm**3
    A = np.asarray(A_transfer, dtype=np.float64) * model.source_amplitude
    noise, scaling_params = choose_noise(
        A,
        "td_fnirs_analytical",
        n_sensors,
        physical_noise,
        scaling,
    )
    posterior_var, info_bits = posterior_info_scalar(A, noise)
    n_pairs = int(A.shape[0] // max(n_time_gates, 1))
    return save_result(
        "td_fnirs_analytical",
        positions,
        info_bits,
        posterior_var,
        {
            "modality": "td_fnirs_analytical",
            "n_sensors": n_sensors,
            "forward_matrix_shape": list(A.shape),
            "n_unique_source_detector_pairs": n_pairs,
            "n_time_gates": n_time_gates,
            "time_gates_ns": list(map(float, modality.time_gates_ns)),
            "grid_spacing_mm": grid_spacing_mm,
            "max_dist_mm": max_dist_mm,
            "table_default_reference": TABLE_DEFAULTS["td_fnirs_analytical"]["reference"],
            "detector_noise_today": physical_noise,
            "source_amplitude": model.source_amplitude,
            "voxel_volume_mm3": voxel_volume_mm3,
            "transfer_function_units": "mm^-1",
            "transfer_function_convention": "voxel-integrated before absorption scaling",
            "model": (
                "TD-fNIRS analytical semi-infinite diffusion sensitivity; scalar "
                "absorption contrast integrated over each voxel"
            ),
            **scaling_params,
        },
        "posterior_variance",
        depth_bin_width_mm,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modalities",
        default="all",
        help=(
            "Comma-separated subset to compute: eeg/eeg_openmeeg, "
            "eeg_homogeneous, meg_opm, meg_squid, fnirs_cw, td_fnirs, or all."
        ),
    )
    parser.add_argument(
        "--grid-spacing-mm",
        type=float,
        default=None,
        help="Override both EEG and MEG source-grid spacing.",
    )
    parser.add_argument(
        "--eeg-sensors",
        type=int,
        default=TABLE_DEFAULTS["eeg_openmeeg"]["n_sensors"],
    )
    parser.add_argument(
        "--eeg-grid-spacing-mm",
        type=float,
        default=TABLE_DEFAULTS["eeg_homogeneous"]["grid_spacing_mm"],
        help="Only affects eeg_homogeneous; eeg_openmeeg uses the saved BEM grid.",
    )
    parser.add_argument(
        "--meg-sensors",
        type=int,
        default=TABLE_DEFAULTS["meg_opm"]["n_sensors"],
    )
    parser.add_argument(
        "--meg-grid-spacing-mm",
        type=float,
        default=TABLE_DEFAULTS["meg_opm"]["grid_spacing_mm"],
    )
    parser.add_argument(
        "--meg-opm-offset-mm",
        type=float,
        default=TABLE_DEFAULTS["meg_opm"]["sensor_offset_mm"],
    )
    parser.add_argument(
        "--meg-squid-offset-mm",
        type=float,
        default=TABLE_DEFAULTS["meg_squid"]["sensor_offset_mm"],
    )
    parser.add_argument(
        "--fnirs-sensors",
        type=int,
        default=TABLE_DEFAULTS["cw_fnirs"]["n_sensors"],
    )
    parser.add_argument(
        "--fnirs-grid-spacing-mm",
        type=float,
        default=TABLE_DEFAULTS["cw_fnirs"]["grid_spacing_mm"],
    )
    parser.add_argument(
        "--fnirs-max-dist-mm",
        type=float,
        default=TABLE_DEFAULTS["cw_fnirs"]["max_dist_mm"],
    )
    parser.add_argument(
        "--td-fnirs-sensors",
        type=int,
        default=TABLE_DEFAULTS["td_fnirs_analytical"]["n_sensors"],
    )
    parser.add_argument(
        "--td-fnirs-grid-spacing-mm",
        type=float,
        default=TABLE_DEFAULTS["td_fnirs_analytical"]["grid_spacing_mm"],
    )
    parser.add_argument(
        "--td-fnirs-max-dist-mm",
        type=float,
        default=TABLE_DEFAULTS["td_fnirs_analytical"]["max_dist_mm"],
    )
    parser.add_argument(
        "--td-fnirs-gates",
        type=int,
        default=TABLE_DEFAULTS["td_fnirs_analytical"]["n_time_gates"],
    )
    parser.add_argument("--depth-bin-width-mm", type=float, default=2.0)
    parser.add_argument(
        "--scaling",
        choices=SCALING_CHOICES,
        default=DEFAULT_SCALING,
        help=(
            "physical uses raw source-amplitude and detector-noise units; "
            "empirical normalizes the full forward matrix to observed SNR"
        ),
    )
    parser.add_argument(
        "--map-kind",
        choices=MAP_KIND_CHOICES,
        default="posterior",
        help=(
            "posterior computes marginal voxel posterior information; capacity "
            "attributes SVD mode capacity to voxels so depth-bin totals sum to "
            "the joint SVD capacity; both computes both families"
        ),
    )
    args = parser.parse_args()
    eeg_grid_spacing_mm = args.grid_spacing_mm or args.eeg_grid_spacing_mm
    meg_grid_spacing_mm = args.grid_spacing_mm or args.meg_grid_spacing_mm

    requested = [item.strip() for item in args.modalities.split(",") if item.strip()]
    selected = []
    requested_all = False
    for item in requested:
        alias = MODALITY_ALIASES.get(item)
        if alias is None:
            raise ValueError(f"Unknown modality {item!r}; expected one of {sorted(MODALITY_ALIASES)}")
        if alias == "all":
            selected = list(DEFAULT_MODALITIES)
            requested_all = True
            break
        selected.append(alias)
    selected = [name for name in MODALITY_ORDER if name in set(selected)]

    posterior_outputs = []
    if args.map_kind in ("posterior", "both") and "eeg_openmeeg" in selected:
        posterior_outputs.append(
            run_eeg_openmeeg(
                args.eeg_sensors,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("posterior", "both") and "eeg_homogeneous" in selected:
        posterior_outputs.append(
            run_eeg(
                args.eeg_sensors,
                eeg_grid_spacing_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("posterior", "both") and "meg_opm" in selected:
        posterior_outputs.append(
            run_meg(
                "meg_opm",
                args.meg_sensors,
                meg_grid_spacing_mm,
                args.meg_opm_offset_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("posterior", "both") and "meg_squid" in selected:
        posterior_outputs.append(
            run_meg(
                "meg_squid",
                args.meg_sensors,
                meg_grid_spacing_mm,
                args.meg_squid_offset_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("posterior", "both") and "cw_fnirs" in selected:
        posterior_outputs.append(
            run_fnirs(
                args.fnirs_sensors,
                args.fnirs_grid_spacing_mm,
                args.fnirs_max_dist_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("posterior", "both") and "td_fnirs_analytical" in selected:
        posterior_outputs.append(
            run_td_fnirs(
                args.td_fnirs_sensors,
                args.td_fnirs_grid_spacing_mm,
                args.td_fnirs_max_dist_mm,
                args.td_fnirs_gates,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if posterior_outputs:
        plot_combined_depth_profiles(posterior_outputs, args.scaling)

    capacity_selected = list(selected)
    if requested_all:
        capacity_selected = list(CAPACITY_ATTRIBUTION_DEFAULT_MODALITIES)
    elif (
        args.map_kind in ("capacity", "both")
        and "td_fnirs_analytical" in capacity_selected
    ):
        raise NotImplementedError(
            "Exact TD-fNIRS capacity attribution requires a large dense right-SVD "
            "that is disabled in the local default path. Use CW-fNIRS or run TD "
            "on a larger compute host."
        )

    capacity_outputs = []
    if args.map_kind in ("capacity", "both") and "eeg_openmeeg" in capacity_selected:
        capacity_outputs.append(
            run_eeg_openmeeg_capacity_attribution(
                args.eeg_sensors,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("capacity", "both") and "meg_opm" in capacity_selected:
        capacity_outputs.append(
            run_meg_capacity_attribution(
                "meg_opm",
                args.meg_sensors,
                meg_grid_spacing_mm,
                args.meg_opm_offset_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("capacity", "both") and "meg_squid" in capacity_selected:
        capacity_outputs.append(
            run_meg_capacity_attribution(
                "meg_squid",
                args.meg_sensors,
                meg_grid_spacing_mm,
                args.meg_squid_offset_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if args.map_kind in ("capacity", "both") and "cw_fnirs" in capacity_selected:
        capacity_outputs.append(
            run_fnirs_capacity_attribution(
                args.fnirs_sensors,
                args.fnirs_grid_spacing_mm,
                args.fnirs_max_dist_mm,
                args.depth_bin_width_mm,
                args.scaling,
            )
        )
    if capacity_outputs:
        plot_combined_capacity_attribution_profiles(capacity_outputs, args.scaling)

    print("Wrote:")
    for path in posterior_outputs + capacity_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
