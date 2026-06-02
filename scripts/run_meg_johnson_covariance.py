#!/usr/bin/env python3
"""Compute a MEG body-Johnson noise covariance matrix."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-config")

from guti.core import get_sensor_positions
from guti.modalities.meg.johnson_noise import (
    compute_meg_johnson_noise_spectral_density,
    match_covariance_diagonal,
)
from guti.modalities.meg.meg import OPM_OFFSET_MM, SQUID_OFFSET_MM
from guti.noise_models import (
    BODY_TEMP_K,
    compute_output_noise_std,
    covariance_to_correlation_matrix,
)


MEG_OFFSETS_MM = {
    "meg_opm": OPM_OFFSET_MM,
    "meg_squid": SQUID_OFFSET_MM,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modality",
        choices=tuple(MEG_OFFSETS_MM),
        default="meg_opm",
        help="MEG sensor standoff/noise model. Default: meg_opm.",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=128,
        help="Number of sensor locations. Default: 128.",
    )
    parser.add_argument(
        "--sensor-components",
        choices=("xyz", "radial"),
        default="xyz",
        help="Use 3 global components per sensor or one radial channel. Default: xyz.",
    )
    parser.add_argument(
        "--voxel-resolution-mm",
        type=float,
        default=8.0,
        help="Voxel size for the layered head integration. Default: 8.",
    )
    parser.add_argument(
        "--solver",
        choices=("finite_volume", "vector_potential"),
        default="finite_volume",
        help=(
            "Reciprocal solver. finite_volume solves the scalar-potential "
            "charge-conservation correction; vector_potential is the older "
            "uncorrected shortcut. Default: finite_volume."
        ),
    )
    parser.add_argument(
        "--bandwidth-hz",
        type=float,
        default=100.0,
        help="Band-integrated covariance bandwidth. Default: 100.",
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=BODY_TEMP_K,
        help=f"Conductive-tissue temperature. Default: {BODY_TEMP_K}.",
    )
    parser.add_argument(
        "--tier",
        choices=("today", "fundamental"),
        default="fundamental",
        help=(
            "Scalar detector-noise tier used for the matched covariance. "
            "Default: fundamental."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Channel batch size for reciprocal features. Default: 64.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/meg_johnson_noise"),
        help="Directory for NPZ and markdown summary outputs.",
    )
    return parser


def offdiag_stats(matrix: np.ndarray) -> dict[str, float]:
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    values = matrix[mask]
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_abs": float(np.mean(np.abs(values))),
    }


def write_summary(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    corr = result["correlation_stats"]
    lines = [
        "# MEG Johnson Noise Covariance",
        "",
        "Reciprocal-field overlap calculation for body/tissue magnetic",
        "Johnson noise in the GUTI layered hemispherical head.",
        "",
        "## Configuration",
        "",
        f"- Modality: `{result['modality']}`",
        f"- Sensors: {result['n_sensors']}",
        f"- Channels: {result['n_channels']} (`{result['sensor_components']}`)",
        f"- Sensor offset: {result['sensor_offset_mm']} mm",
        f"- Voxel resolution: {result['voxel_resolution_mm']} mm",
        f"- Voxels: {result['n_voxels']}",
        f"- Solver: `{result['solver']}`",
        f"- Approximation: {result['approximation']}",
        f"- Bandwidth: {result['bandwidth_hz']} Hz",
        f"- Temperature: {result['temperature_k']} K",
        f"- Detector-noise tier for matched covariance: `{result['tier']}`",
        "",
        "## Noise Scale",
        "",
        (
            "- Raw body-Johnson median over selected channels: "
            f"{result['raw_body_noise_fT_per_sqrtHz']:.4g} fT/sqrt(Hz)"
        ),
        (
            "- Raw body-Johnson median band noise: "
            f"{result['raw_body_noise_fT_rms']:.4g} fT RMS"
        ),
        (
        "- Matched scalar detector noise: "
            f"{result['matched_detector_noise_fT_rms']:.4g} fT RMS"
        ),
        (
            "- Note: `xyz` reports the median across global x/y/z output rows; "
            "for a scalar radial OPM/SQUID comparison, use "
            "`--sensor-components radial`."
        ),
        "",
        "## Correlation",
        "",
        "| off-diagonal rho | value |",
        "|---|---:|",
        f"| min | {corr['min']:.5g} |",
        f"| max | {corr['max']:.5g} |",
        f"| mean | {corr['mean']:.5g} |",
        f"| median | {corr['median']:.5g} |",
        f"| mean abs | {corr['mean_abs']:.5g} |",
        "",
        "## Caveat",
        "",
        "This is a low-frequency reciprocal finite-volume solve for the",
        "conductive head, not a full detector-geometry FEM model. The default",
        "`finite_volume` solver enforces the scalar-potential correction",
        "`div(sigma E)=0` on the voxelized head. It still uses point-dipole",
        "detector channels rather than finite SQUID pickup loops, axial",
        "gradiometers, OPM cell volumes, dewar conductors, or MSR shields.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.n_sensors <= 0:
        raise SystemExit("--n-sensors must be positive")
    if args.bandwidth_hz <= 0.0:
        raise SystemExit("--bandwidth-hz must be positive")
    if args.voxel_resolution_mm <= 0.0:
        raise SystemExit("--voxel-resolution-mm must be positive")

    sensor_offset_mm = MEG_OFFSETS_MM[args.modality]
    sensors = get_sensor_positions(args.n_sensors, offset=sensor_offset_mm)
    spectral_density, metadata = compute_meg_johnson_noise_spectral_density(
        sensors,
        sensor_components=args.sensor_components,
        voxel_resolution_mm=args.voxel_resolution_mm,
        temperature_k=args.temperature_k,
        solver=args.solver,
        batch_size=args.batch_size,
        return_metadata=True,
    )

    band_covariance = spectral_density * args.bandwidth_hz
    correlation = covariance_to_correlation_matrix(spectral_density)
    detector_noise = compute_output_noise_std(
        args.modality,
        n_sensors=args.n_sensors,
        bandwidth_hz=args.bandwidth_hz,
        tier=args.tier,
    )
    matched_detector_covariance = match_covariance_diagonal(
        spectral_density,
        detector_noise,
    )

    diag_spectral = np.diag(spectral_density)
    raw_body_noise_fT_per_sqrtHz = float(np.median(np.sqrt(diag_spectral)) * 1e15)
    raw_body_noise_fT_rms = raw_body_noise_fT_per_sqrtHz * np.sqrt(args.bandwidth_hz)
    result = {
        "modality": args.modality,
        "n_sensors": args.n_sensors,
        "n_channels": metadata.n_channels,
        "sensor_components": args.sensor_components,
        "sensor_offset_mm": sensor_offset_mm,
        "voxel_resolution_mm": args.voxel_resolution_mm,
        "n_voxels": metadata.n_voxels,
        "solver": metadata.solver,
        "approximation": metadata.approximation,
        "bandwidth_hz": args.bandwidth_hz,
        "temperature_k": args.temperature_k,
        "tier": args.tier,
        "raw_body_noise_fT_per_sqrtHz": raw_body_noise_fT_per_sqrtHz,
        "raw_body_noise_fT_rms": raw_body_noise_fT_rms,
        "matched_detector_noise_fT_rms": float(detector_noise * 1e15),
        "correlation_stats": offdiag_stats(correlation),
        "metadata": asdict(metadata),
    }

    stem = (
        f"{args.modality}_{args.n_sensors}sensors_{args.sensor_components}_"
        f"{args.solver}_{args.voxel_resolution_mm:g}mm_{args.bandwidth_hz:g}hz"
    ).replace(".", "p")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"{stem}.npz"
    md_path = args.output_dir / f"{stem}.md"

    np.savez_compressed(
        npz_path,
        sensor_positions_mm=sensors,
        johnson_spectral_density_T2_per_hz=spectral_density,
        johnson_band_covariance_T2=band_covariance,
        johnson_correlation=correlation,
        matched_detector_covariance_T2=matched_detector_covariance,
        metadata_json=np.array(json.dumps(result, indent=2, sort_keys=True)),
    )
    write_summary(md_path, result)

    print(
        "raw body Johnson median "
        f"{raw_body_noise_fT_per_sqrtHz:.4g} fT/sqrt(Hz); "
        f"offdiag rho mean={result['correlation_stats']['mean']:.4g}, "
        f"mean_abs={result['correlation_stats']['mean_abs']:.4g}"
    )
    print(f"wrote {npz_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
