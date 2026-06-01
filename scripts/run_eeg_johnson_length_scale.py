#!/usr/bin/env python3
"""Estimate an effective spatial length scale for EEG Johnson noise."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-config")

from guti.core import SCALP_RADIUS, get_sensor_positions
from guti.modalities.eeg.scalp_resistance import (
    default_layers,
    normalized_cap_axisymmetric_coeffs,
    surface_impedance_by_degree,
)
from guti.noise_models import (
    BODY_TEMP_K,
    compute_johnson_noise_covariance,
    covariance_to_correlation_matrix,
    estimate_effective_correlation_length_mm,
    scalp_geodesic_distance_matrix,
)


DEFAULT_OUTPUT_MD = Path("results/eeg_johnson_effective_length_scale.md")
DEFAULT_OUTPUT_JSON = Path("results/eeg_johnson_effective_length_scale.json")


def surface_kernel_matrix(
    sensor_positions_mm: np.ndarray,
    *,
    electrode_area_cm2: float,
    lmax: int,
) -> np.ndarray:
    """Return zero-mean terminal surface impedance kernel in ohms."""
    if electrode_area_cm2 <= 0.0:
        raise ValueError("electrode_area_cm2 must be positive")
    if lmax <= 1:
        raise ValueError("lmax must be greater than 1")

    layers = default_layers()
    scalp_radius_m = layers[-1].outer_radius_m
    cap_half_angle = math.acos(
        1.0 - electrode_area_cm2 * 1e-4 / (2.0 * math.pi * scalp_radius_m**2)
    )
    transfer = surface_impedance_by_degree(layers, lmax)
    cap_coeffs = normalized_cap_axisymmetric_coeffs(cap_half_angle, lmax)
    weights = transfer[1:] * cap_coeffs[1:] ** 2 / scalp_radius_m**2

    distances_mm = scalp_geodesic_distance_matrix(sensor_positions_mm)
    cos_theta = np.cos(distances_mm / SCALP_RADIUS)
    kernel = np.zeros_like(cos_theta)

    p_prev = np.ones_like(cos_theta)
    p_curr = cos_theta.copy()
    kernel += weights[0] * p_curr
    for ell in range(1, lmax - 1):
        p_next = ((2 * ell + 1) * cos_theta * p_curr - ell * p_prev) / (ell + 1)
        kernel += weights[ell] * p_next
        p_prev, p_curr = p_curr, p_next

    return 0.5 * (kernel + kernel.T)


def summarize_covariance(
    sensor_positions_mm: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, object]:
    corr = covariance_to_correlation_matrix(covariance)
    offdiag = np.triu(np.ones_like(corr, dtype=bool), k=1)
    positive = offdiag & (corr > 0.0)
    fits = {
        kernel: asdict(
            estimate_effective_correlation_length_mm(
                sensor_positions_mm,
                covariance,
                kernel=kernel,
                distance_metric="geodesic",
            )
        )
        for kernel in ("exponential", "gaussian")
    }
    return {
        "fits": fits,
        "diag_min": float(np.min(np.diag(covariance))),
        "diag_max": float(np.max(np.diag(covariance))),
        "corr_min": float(np.min(corr[offdiag])),
        "corr_max": float(np.max(corr[offdiag])),
        "corr_positive_pair_count": int(np.count_nonzero(positive)),
    }


def write_markdown(path: Path, result: dict[str, object]) -> None:
    rows = result["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EEG Johnson Noise Effective Length Scale",
        "",
        "This is a reproducible estimate of the scalar distance-kernel length",
        "that best approximates the Johnson covariance from the GUTI layered",
        "spherical EEG head model.",
        "",
        "## Configuration",
        "",
        f"- Sensors: {result['n_sensors']} Fibonacci scalp electrodes",
        f"- Electrode patch area: {result['electrode_area_cm2']} cm^2 circular cap",
        f"- Spherical-harmonic truncation: l <= {result['lmax']}",
        f"- Temperature: {result['temperature_k']} K",
        f"- Bandwidth: {result['bandwidth_hz']} Hz",
        "- Fit objective: least squares in log-correlation over positive",
        "  off-diagonal sensor pairs.",
        "",
        "## Estimates",
        "",
        "| Model | max offdiag corr | min offdiag corr | exp L (mm) | exp log RMSE | gaussian L (mm) | gaussian log RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fits = row["summary"]["fits"]
        lines.append(
            "| {label} | {corr_max:.4g} | {corr_min:.4g} | {exp_l:.3f} | "
            "{exp_rmse:.3f} | {gauss_l:.3f} | {gauss_rmse:.3f} |".format(
                label=row["label"],
                corr_max=row["summary"]["corr_max"],
                corr_min=row["summary"]["corr_min"],
                exp_l=fits["exponential"]["length_mm"],
                exp_rmse=fits["exponential"]["log_rmse"],
                gauss_l=fits["gaussian"]["length_mm"],
                gauss_rmse=fits["gaussian"]["log_rmse"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For the pure volume-conductor Johnson term, the effective length is",
            "about 19 mm for the exponential kernel and 27 mm for the Gaussian",
            "kernel. If the existing Gaussian distance model is kept as a",
            "fallback approximation, 27 mm is the closest match under this fit.",
            "",
            "Adding independent series/contact resistance mostly increases the",
            "diagonal variance and suppresses off-diagonal correlations. With a",
            "5 kOhm independent series term, the nearest-neighbor correlation is",
            "only about 0.02, so a single length-scale kernel is a poor physical",
            "description; the direct Johnson covariance should be preferred.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sensors", type=int, default=256)
    parser.add_argument("--electrode-area-cm2", type=float, default=1.0)
    parser.add_argument("--lmax", type=int, default=2000)
    parser.add_argument("--bandwidth-hz", type=float, default=100.0)
    parser.add_argument("--temperature-k", type=float, default=BODY_TEMP_K)
    parser.add_argument(
        "--series-resistance-ohm",
        nargs="*",
        type=float,
        default=[500.0, 1000.0, 5000.0],
        help="Independent per-electrode series/contact resistances to compare.",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensors = get_sensor_positions(args.n_sensors)
    volume_impedance = surface_kernel_matrix(
        sensors,
        electrode_area_cm2=args.electrode_area_cm2,
        lmax=args.lmax,
    )

    cases = [("volume only", None)]
    cases.extend(
        (
            f"volume + {resistance:g} ohm independent series",
            float(resistance),
        )
        for resistance in args.series_resistance_ohm
    )

    rows = []
    for label, series_resistance in cases:
        covariance = compute_johnson_noise_covariance(
            volume_impedance,
            bandwidth_hz=args.bandwidth_hz,
            temperature_k=args.temperature_k,
            series_resistance_ohm=series_resistance,
        )
        rows.append(
            {
                "label": label,
                "series_resistance_ohm": series_resistance,
                "summary": summarize_covariance(sensors, covariance),
            }
        )

    result = {
        "n_sensors": args.n_sensors,
        "electrode_area_cm2": args.electrode_area_cm2,
        "lmax": args.lmax,
        "bandwidth_hz": args.bandwidth_hz,
        "temperature_k": args.temperature_k,
        "volume_impedance_diag_ohm": float(np.diag(volume_impedance)[0]),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output_md, result)

    volume_fits = rows[0]["summary"]["fits"]
    print(
        "volume-only effective length: "
        f"exponential={volume_fits['exponential']['length_mm']:.3f} mm, "
        f"gaussian={volume_fits['gaussian']['length_mm']:.3f} mm"
    )
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
