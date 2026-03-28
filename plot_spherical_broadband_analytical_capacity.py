#!/usr/bin/env python3
"""Estimate broadband spherical-model bitrate or capacity vs center frequency."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn, spherical_yn

from guti.core import (
    get_bitrate,
    noise_floor_heuristic,
    water_filling_spectrum,
)

US_ANALYTICAL_SOURCE_RADIUS_M = 0.08
US_ANALYTICAL_RECEIVER_RADIUS_M = 0.10
US_ANALYTICAL_SOUND_SPEED_MPS = 1500.0
# Gaussian spectrum setting chosen to match the half-power fractional bandwidth
# of the 2-cycle Hann tone burst used in guti.modalities.us.analytical.
US_ANALYTICAL_MATCHED_FRACTIONAL_BANDWIDTH = 0.909004266


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D grid with at least two points")
    w = np.empty_like(x, dtype=float)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("x and y must be 1D arrays of the same length")
    if x.size < 2:
        raise ValueError("x and y must contain at least two points")
    dx = np.diff(x)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * dx))


def gaussian_analytic_spectrum(
    omega: np.ndarray,
    omega0: float,
    sigma_omega: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((omega - omega0) / sigma_omega) ** 2)


def spherical_block_matrix(
    ell: int,
    a: float,
    c: float,
    omega_grid: np.ndarray,
    s_tilde: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    w_r = trapz_weights(r_grid) * (r_grid**2)
    w_omega = trapz_weights(omega_grid) / (2.0 * np.pi)

    k = omega_grid / c
    x_a = k * a
    x_r = np.outer(k, r_grid)

    with np.errstate(all="ignore"):
        h_l = spherical_jn(ell, x_a) + 1j * spherical_yn(ell, x_a)
        J = spherical_jn(ell, x_r)
        kernel = (1j * k * h_l)[:, None] * J

    # In the low-frequency / high-order tail, h_l and j_l can overflow/underflow
    # separately even though their product has a finite quasistatic limit:
    #   i k h_l^(1)(k a) j_l(k r) -> (r/a)^ell / ((2 ell + 1) a)
    invalid = ~np.isfinite(kernel)
    if np.any(invalid):
        static_kernel = ((r_grid[None, :] / a) ** ell) / ((2 * ell + 1) * a)
        kernel = np.where(invalid, static_kernel.astype(np.complex128), kernel)

    return (
        np.sqrt(w_omega)[:, None]
        * s_tilde[:, None]
        * kernel
        * np.sqrt(w_r)[None, :]
    )


def singular_values_for_ell(
    ell: int,
    a: float,
    R: float,
    c: float,
    omega_grid: np.ndarray,
    s_tilde: np.ndarray,
    n_r: int,
    n_sv: int,
) -> np.ndarray:
    r_grid = np.linspace(0.0, R, n_r)
    M = spherical_block_matrix(
        ell=ell,
        a=a,
        c=c,
        omega_grid=omega_grid,
        s_tilde=s_tilde,
        r_grid=r_grid,
    )
    svals = np.linalg.svd(M, compute_uv=False)
    return svals[:n_sv].real


def full_singular_value_list(
    ell_max: int,
    n_sv_per_ell: int,
    a: float,
    R: float,
    c: float,
    omega_grid: np.ndarray,
    s_tilde: np.ndarray,
    n_r: int,
) -> np.ndarray:
    all_svals: list[float] = []
    for ell in range(ell_max + 1):
        svals_ell = singular_values_for_ell(
            ell=ell,
            a=a,
            R=R,
            c=c,
            omega_grid=omega_grid,
            s_tilde=s_tilde,
            n_r=n_r,
            n_sv=n_sv_per_ell,
        )
        multiplicity = 2 * ell + 1
        for sval in svals_ell:
            all_svals.extend([float(sval)] * multiplicity)
    svals = np.asarray(all_svals, dtype=float)
    return np.sort(svals)[::-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate bitrate or channel capacity versus center frequency using "
            "the spherical broadband pulse model."
        )
    )
    parser.add_argument(
        "--frequencies-khz",
        type=parse_float_csv,
        default=None,
        help="Explicit comma-separated center frequencies in kHz. Overrides the range sweep.",
    )
    parser.add_argument("--fmin-khz", type=float, default=50.0)
    parser.add_argument("--fmax-khz", type=float, default=1000.0)
    parser.add_argument("--num-frequencies", type=int, default=20)
    parser.add_argument(
        "--receiver-radius-m",
        type=float,
        default=US_ANALYTICAL_RECEIVER_RADIUS_M,
        help=(
            "Receiver shell radius in meters. Default matches the free-field "
            "US analytical model's 100 mm sensor radius."
        ),
    )
    parser.add_argument(
        "--source-radius-m",
        type=float,
        default=US_ANALYTICAL_SOURCE_RADIUS_M,
        help=(
            "Source domain radius in meters. Default matches the free-field "
            "US analytical model's 80 mm brain radius."
        ),
    )
    parser.add_argument(
        "--sound-speed",
        type=float,
        default=US_ANALYTICAL_SOUND_SPEED_MPS,
        help=(
            "Sound speed in m/s. Default matches guti.modalities.us.analytical."
        ),
    )
    parser.add_argument(
        "--fractional-bandwidth",
        type=float,
        default=US_ANALYTICAL_MATCHED_FRACTIONAL_BANDWIDTH,
        help=(
            "Gaussian-spectrum fractional bandwidth. Default matches the "
            "half-power bandwidth of the 2-cycle Hann tone burst used in "
            "guti.modalities.us.analytical."
        ),
    )
    parser.add_argument("--fixed-resolution", action="store_true", help="Disable adaptive discretization and use the max values directly.")
    parser.add_argument("--base-frequency-khz", type=float, default=50.0, help="Reference frequency for adaptive scaling.")
    parser.add_argument(
        "--ell-scale-exponent",
        type=float,
        default=0.5,
        help="Frequency scaling exponent for ell_max in adaptive mode. Default: 0.5",
    )
    parser.add_argument(
        "--n-omega-scale-exponent",
        type=float,
        default=0.5,
        help="Frequency scaling exponent for n_omega in adaptive mode. Default: 0.5",
    )
    parser.add_argument(
        "--n-r-scale-exponent",
        type=float,
        default=0.5,
        help="Frequency scaling exponent for n_r in adaptive mode. Default: 0.5",
    )
    parser.add_argument(
        "--n-sv-scale-exponent",
        type=float,
        default=0.5,
        help="Frequency scaling exponent for n_sv_per_ell in adaptive mode. Default: 0.5",
    )
    parser.add_argument("--n-omega-min", type=int, default=64)
    parser.add_argument("--n-r-min", type=int, default=48)
    parser.add_argument("--n-sv-per-ell-min", type=int, default=2)
    parser.add_argument("--ell-max-min", type=int, default=40)
    parser.add_argument("--n-omega", type=int, default=192, help="Maximum n_omega in adaptive mode, or fixed n_omega with --fixed-resolution.")
    parser.add_argument("--n-r", type=int, default=96, help="Maximum n_r in adaptive mode, or fixed n_r with --fixed-resolution.")
    parser.add_argument("--n-sv-per-ell", type=int, default=4, help="Maximum n_sv_per_ell in adaptive mode, or fixed value with --fixed-resolution.")
    parser.add_argument("--ell-padding", type=int, default=10)
    parser.add_argument(
        "--ell-max-cap",
        type=int,
        default=320,
        help=(
            "Hard cap on ell_max for runtime. Increase this for better convergence "
            "at high frequency."
        ),
    )
    parser.add_argument(
        "--convergence-growth-factor",
        type=float,
        default=1.5,
        help="Multiplicative refinement factor between convergence rounds.",
    )
    parser.add_argument(
        "--min-convergence-rounds",
        type=int,
        default=2,
        help="Minimum number of resolution rounds per frequency. Default: 2",
    )
    parser.add_argument(
        "--max-convergence-rounds",
        type=int,
        default=8,
        help="Maximum number of resolution rounds per frequency. Default: 8",
    )
    parser.add_argument(
        "--capacity-rtol",
        type=float,
        default=0.02,
        help="Relative tolerance on the primary metric between rounds. Default: 0.02",
    )
    parser.add_argument(
        "--top-sv-rtol",
        type=float,
        default=0.005,
        help="Relative tolerance on the top singular value between rounds. Default: 0.005",
    )
    parser.add_argument(
        "--require-convergence",
        action="store_true",
        help="Raise an error if a frequency does not converge before hitting the resolution caps.",
    )
    parser.add_argument(
        "--capacity-snr",
        type=float,
        default=2000.0,
        help="SNR passed to get_bitrate_channel_capacity().",
    )
    parser.add_argument(
        "--noise-heuristic",
        choices=("power", "first"),
        default="power",
        help="Noise heuristic used with get_bitrate(). Matches analytical.py.",
    )
    parser.add_argument(
        "--noise-snr",
        type=float,
        default=2000.0,
        help="SNR passed to noise_floor_heuristic() for get_bitrate().",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help=(
            "Absolute noise level override for get_bitrate(). If provided, "
            "--noise-heuristic and --noise-snr are ignored."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("bitrate", "channel_capacity", "mode_scaled_capacity"),
        default="bitrate",
        help="Primary metric to plot and use for convergence. Default: bitrate",
    )
    parser.add_argument(
        "--mode-threshold-frac",
        type=float,
        default=1e-3,
        help="Relative threshold for counting significant modes: sigma_i >= frac * sigma_max.",
    )
    parser.add_argument(
        "--mode-snr-target",
        type=float,
        default=1.0,
        help=(
            "Target average output SNR per significant mode for mode_scaled_capacity. "
            "The total output SNR target becomes mode_snr_target * M(f)."
        ),
    )
    parser.add_argument(
        "--time-resolution",
        type=float,
        default=1.0,
        help="time_resolution passed to bitrate/capacity helpers. Default: 1.0",
    )
    parser.add_argument(
        "--outdir",
        default="plots/spherical_broadband_analytical",
        help="Directory for plots and saved arrays.",
    )
    return parser


def choose_resolution(freq_khz: float, ell_max_est: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.fixed_resolution:
        return (
            min(ell_max_est, args.ell_max_cap),
            args.n_omega,
            args.n_r,
            args.n_sv_per_ell,
        )

    freq_scale = max(1.0, freq_khz / args.base_frequency_khz)
    ell_scale = freq_scale ** args.ell_scale_exponent
    omega_scale = freq_scale ** args.n_omega_scale_exponent
    r_scale = freq_scale ** args.n_r_scale_exponent
    sv_scale = freq_scale ** args.n_sv_scale_exponent

    ell_cap_adaptive = min(
        args.ell_max_cap,
        max(args.ell_max_min, int(math.ceil(args.ell_max_min * ell_scale))),
    )
    n_omega = min(
        args.n_omega,
        max(args.n_omega_min, int(math.ceil(args.n_omega_min * omega_scale))),
    )
    n_r = min(
        args.n_r,
        max(args.n_r_min, int(math.ceil(args.n_r_min * r_scale))),
    )
    n_sv_per_ell = min(
        args.n_sv_per_ell,
        max(args.n_sv_per_ell_min, int(math.ceil(args.n_sv_per_ell_min * sv_scale))),
    )
    ell_max = min(ell_max_est, ell_cap_adaptive)
    return ell_max, n_omega, n_r, n_sv_per_ell


def refine_resolution(
    ell_max: int,
    n_omega: int,
    n_r: int,
    n_sv_per_ell: int,
    ell_max_est: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    growth = args.convergence_growth_factor
    ell_next = min(
        ell_max_est,
        args.ell_max_cap,
        max(ell_max + 1, int(math.ceil(ell_max * growth))),
    )
    n_omega_next = min(
        args.n_omega,
        max(n_omega + 1, int(math.ceil(n_omega * growth))),
    )
    n_r_next = min(
        args.n_r,
        max(n_r + 1, int(math.ceil(n_r * growth))),
    )
    n_sv_next = min(
        args.n_sv_per_ell,
        max(n_sv_per_ell + 1, int(math.ceil(n_sv_per_ell * growth))),
    )
    return ell_next, n_omega_next, n_r_next, n_sv_next


def resolve_frequencies(args: argparse.Namespace) -> np.ndarray:
    if args.frequencies_khz is not None:
        if not args.frequencies_khz:
            raise ValueError("--frequencies-khz must contain at least one frequency")
        if any(freq <= 0 for freq in args.frequencies_khz):
            raise ValueError("--frequencies-khz values must be positive")
        return np.asarray(args.frequencies_khz, dtype=float)

    if args.num_frequencies < 2:
        raise ValueError("--num-frequencies must be at least 2")
    if args.fmin_khz <= 0 or args.fmax_khz <= 0:
        raise ValueError("frequencies must be positive")
    if args.fmin_khz >= args.fmax_khz:
        raise ValueError("--fmin-khz must be less than --fmax-khz")
    return np.linspace(args.fmin_khz, args.fmax_khz, args.num_frequencies)


def relative_change(current: float, previous: float, floor: float = 1e-12) -> float:
    scale = max(abs(current), abs(previous), floor)
    return abs(current - previous) / scale


def get_mode_count(svals: np.ndarray, threshold_frac: float) -> tuple[int, float]:
    if svals.size == 0:
        return 0, float("nan")
    threshold = float(threshold_frac * svals[0])
    return int(np.count_nonzero(svals >= threshold)), threshold


def capacity_from_power_allocation_over_noise(
    svals: np.ndarray,
    power_allocation_over_noise: np.ndarray,
    time_resolution: float,
) -> float:
    return float(
        (1.0 / (2.0 * time_resolution))
        * np.sum(np.log2(1.0 + power_allocation_over_noise * (svals**2)))
    )


def evaluate_frequency_at_resolution(
    freq_khz: float,
    ell_max_est: int,
    ell_max: int,
    n_omega_used: int,
    n_r_used: int,
    n_sv_per_ell_used: int,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    f0 = freq_khz * 1e3
    omega0 = 2.0 * np.pi * f0
    sigma_omega = 0.5 * args.fractional_bandwidth * omega0
    omega_min = max(1e-6, omega0 - 4.0 * sigma_omega)
    omega_max = omega0 + 4.0 * sigma_omega
    omega_grid = np.linspace(omega_min, omega_max, n_omega_used)
    s_tilde = gaussian_analytic_spectrum(
        omega=omega_grid,
        omega0=omega0,
        sigma_omega=sigma_omega,
        amplitude=1.0,
    )

    norm = math.sqrt(trapezoid_integral(np.abs(s_tilde) ** 2, omega_grid) / (2.0 * np.pi))
    s_tilde = s_tilde / norm

    svals = full_singular_value_list(
        ell_max=ell_max,
        n_sv_per_ell=n_sv_per_ell_used,
        a=args.receiver_radius_m,
        R=args.source_radius_m,
        c=args.sound_speed,
        omega_grid=omega_grid,
        s_tilde=s_tilde,
        n_r=n_r_used,
    )
    svals = svals[np.abs(svals) > 0]
    top_singular_value = float(svals[0]) if svals.size else 0.0
    mode_count, mode_threshold = get_mode_count(svals, args.mode_threshold_frac)
    if args.noise_level is None:
        noise_level = float(
            noise_floor_heuristic(
                svals,
                heuristic=args.noise_heuristic,
                snr=args.noise_snr,
            )
        )
    else:
        noise_level = float(args.noise_level)
    bitrate = float(
        get_bitrate(
            svals,
            noise_level,
            time_resolution=args.time_resolution,
        )
    )
    current_power_allocation_over_noise = water_filling_spectrum(
        svals.astype(np.float64),
        args.capacity_snr,
    )
    current_output_power_over_noise = float(
        np.sum((svals.astype(np.float64) ** 2) * current_power_allocation_over_noise)
    )
    channel_capacity = float(
        capacity_from_power_allocation_over_noise(
            svals.astype(np.float64),
            current_power_allocation_over_noise,
            time_resolution=args.time_resolution,
        )
    )
    average_mode_snr_current = float(current_output_power_over_noise / mode_count) if mode_count > 0 else float("nan")

    mode_scaled_output_power_over_noise = float(args.mode_snr_target * mode_count)
    mode_scaled_capacity = 0.0
    average_mode_snr_mode_scaled = float("nan")
    if mode_count > 0 and mode_scaled_output_power_over_noise > 0:
        mode_scaled_snr = math.sqrt(mode_scaled_output_power_over_noise)
        mode_scaled_power_allocation_over_noise = water_filling_spectrum(
            svals.astype(np.float64),
            mode_scaled_snr,
        )
        mode_scaled_capacity = float(
            capacity_from_power_allocation_over_noise(
                svals.astype(np.float64),
                mode_scaled_power_allocation_over_noise,
                time_resolution=args.time_resolution,
            )
        )
        realized_output_power_over_noise = float(
            np.sum((svals.astype(np.float64) ** 2) * mode_scaled_power_allocation_over_noise)
        )
        average_mode_snr_mode_scaled = float(realized_output_power_over_noise / mode_count)

    if args.metric == "bitrate":
        metric_value = bitrate
    elif args.metric == "channel_capacity":
        metric_value = channel_capacity
    else:
        metric_value = mode_scaled_capacity
    return {
        "frequency_khz": float(freq_khz),
        "bitrate": bitrate,
        "channel_capacity": channel_capacity,
        "mode_scaled_capacity": mode_scaled_capacity,
        "noise_level": noise_level,
        "metric_value": metric_value,
        "top_singular_value": top_singular_value,
        "mode_count": int(mode_count),
        "mode_threshold": float(mode_threshold),
        "current_output_power_over_noise": current_output_power_over_noise,
        "average_mode_snr_current": average_mode_snr_current,
        "mode_scaled_output_power_over_noise": mode_scaled_output_power_over_noise,
        "average_mode_snr_mode_scaled": average_mode_snr_mode_scaled,
        "ell_max_est": int(ell_max_est),
        "ell_max": int(ell_max),
        "n_omega": int(n_omega_used),
        "n_r": int(n_r_used),
        "n_sv_per_ell": int(n_sv_per_ell_used),
    }


def estimate_frequency_result(
    freq_khz: float,
    args: argparse.Namespace,
    *,
    progress_prefix: str | None = None,
) -> dict[str, float | int]:
    f0 = freq_khz * 1e3
    omega0 = 2.0 * np.pi * f0
    sigma_omega = 0.5 * args.fractional_bandwidth * omega0
    omega_max = omega0 + 4.0 * sigma_omega
    k_max = omega_max / args.sound_speed
    ell_max_est = int(math.ceil(k_max * args.receiver_radius_m + args.ell_padding))
    resolution = choose_resolution(
        freq_khz=freq_khz,
        ell_max_est=ell_max_est,
        args=args,
    )
    previous_result: dict[str, float | int] | None = None
    final_result: dict[str, float | int] | None = None
    converged = False
    metric_rel_change = float("nan")
    top_sv_rel_change = float("nan")
    stop_reason = "max_convergence_rounds"

    for round_idx in range(1, args.max_convergence_rounds + 1):
        ell_max, n_omega_used, n_r_used, n_sv_per_ell_used = resolution
        if progress_prefix is not None:
            print(
                f"{progress_prefix} "
                f"freq={freq_khz:.1f} kHz "
                f"round={round_idx} "
                f"ell_max_est={ell_max_est} ell_max_used={ell_max} "
                f"n_omega={n_omega_used} n_r={n_r_used} n_sv_per_ell={n_sv_per_ell_used}"
            )
        current_result = evaluate_frequency_at_resolution(
            freq_khz=freq_khz,
            ell_max_est=ell_max_est,
            ell_max=ell_max,
            n_omega_used=n_omega_used,
            n_r_used=n_r_used,
            n_sv_per_ell_used=n_sv_per_ell_used,
            args=args,
        )
        current_result["rounds"] = int(round_idx)

        if previous_result is not None:
            metric_rel_change = relative_change(
                float(current_result["metric_value"]),
                float(previous_result["metric_value"]),
            )
            top_sv_rel_change = relative_change(
                float(current_result["top_singular_value"]),
                float(previous_result["top_singular_value"]),
            )
            current_result["metric_rel_change"] = float(metric_rel_change)
            current_result["top_sv_rel_change"] = float(top_sv_rel_change)
            if (
                round_idx >= args.min_convergence_rounds
                and metric_rel_change <= args.capacity_rtol
                and top_sv_rel_change <= args.top_sv_rtol
            ):
                converged = True
                final_result = current_result
                stop_reason = "converged"
                break

        next_resolution = refine_resolution(
            ell_max=ell_max,
            n_omega=n_omega_used,
            n_r=n_r_used,
            n_sv_per_ell=n_sv_per_ell_used,
            ell_max_est=ell_max_est,
            args=args,
        )
        final_result = current_result
        previous_result = current_result
        if next_resolution == resolution:
            stop_reason = "resolution_caps"
            break
        resolution = next_resolution

    if final_result is None:
        raise RuntimeError("No result produced during convergence loop")

    final_result["converged"] = int(converged)
    final_result["metric_rel_change"] = float(metric_rel_change)
    final_result["top_sv_rel_change"] = float(top_sv_rel_change)
    final_result["stop_reason"] = stop_reason
    if progress_prefix is not None:
        print(
            f"{progress_prefix} "
            f"done freq={freq_khz:.1f} kHz "
            f"bitrate={float(final_result['bitrate']):.3f} "
            f"channel_capacity={float(final_result['channel_capacity']):.3f} "
            f"mode_scaled_capacity={float(final_result['mode_scaled_capacity']):.3f} "
            f"mode_count={int(final_result['mode_count'])} "
            f"top_sv={float(final_result['top_singular_value']):.6g} "
            f"converged={bool(final_result['converged'])} "
            f"rounds={int(final_result['rounds'])}"
        )
    if args.require_convergence and not converged:
        raise RuntimeError(
            "Frequency "
            f"{freq_khz:.1f} kHz did not converge "
            f"(stop_reason={final_result['stop_reason']}, "
            f"metric_rel_change={final_result['metric_rel_change']:.3g}, "
            f"top_sv_rel_change={final_result['top_sv_rel_change']:.3g}, "
            f"ell_max={final_result['ell_max']}, n_omega={final_result['n_omega']}, "
            f"n_r={final_result['n_r']}, n_sv_per_ell={final_result['n_sv_per_ell']})."
        )

    return final_result


def write_output_artifacts(results: list[dict[str, float | int]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("No results to write")

    results_sorted = sorted(results, key=lambda row: float(row["frequency_khz"]))
    frequencies_khz = np.asarray([row["frequency_khz"] for row in results_sorted], dtype=float)
    bitrates = np.asarray([row["bitrate"] for row in results_sorted], dtype=float)
    channel_capacities = np.asarray(
        [row["channel_capacity"] for row in results_sorted],
        dtype=float,
    )
    mode_scaled_capacities = np.asarray(
        [row["mode_scaled_capacity"] for row in results_sorted],
        dtype=float,
    )
    metric_values = np.asarray([row["metric_value"] for row in results_sorted], dtype=float)
    top_singular_values = np.asarray(
        [row["top_singular_value"] for row in results_sorted],
        dtype=float,
    )
    mode_counts = np.asarray([row["mode_count"] for row in results_sorted], dtype=int)
    mode_thresholds = np.asarray([row["mode_threshold"] for row in results_sorted], dtype=float)
    current_output_powers_over_noise = np.asarray(
        [row["current_output_power_over_noise"] for row in results_sorted],
        dtype=float,
    )
    average_mode_snrs_current = np.asarray(
        [row["average_mode_snr_current"] for row in results_sorted],
        dtype=float,
    )
    mode_scaled_output_powers_over_noise = np.asarray(
        [row["mode_scaled_output_power_over_noise"] for row in results_sorted],
        dtype=float,
    )
    average_mode_snrs_mode_scaled = np.asarray(
        [row["average_mode_snr_mode_scaled"] for row in results_sorted],
        dtype=float,
    )
    ell_maxes = np.asarray([row["ell_max"] for row in results_sorted], dtype=int)
    n_omegas = np.asarray([row["n_omega"] for row in results_sorted], dtype=int)
    n_rs = np.asarray([row["n_r"] for row in results_sorted], dtype=int)
    n_sv_per_ells = np.asarray(
        [row["n_sv_per_ell"] for row in results_sorted],
        dtype=int,
    )
    converged = np.asarray([row.get("converged", 0) for row in results_sorted], dtype=int)
    rounds = np.asarray([row.get("rounds", 1) for row in results_sorted], dtype=int)
    metric_rel_changes = np.asarray(
        [row.get("metric_rel_change", np.nan) for row in results_sorted],
        dtype=float,
    )
    top_sv_rel_changes = np.asarray(
        [row.get("top_sv_rel_change", np.nan) for row in results_sorted],
        dtype=float,
    )
    noise_levels = np.asarray(
        [row.get("noise_level", np.nan) for row in results_sorted],
        dtype=float,
    )

    np.savez(
        outdir / "spherical_broadband_capacity_vs_frequency.npz",
        frequencies_khz=frequencies_khz,
        frequencies_hz=frequencies_khz * 1e3,
        capacities=channel_capacities,
        bitrates=bitrates,
        channel_capacities=channel_capacities,
        mode_scaled_capacities=mode_scaled_capacities,
        metric_values=metric_values,
        top_singular_values=top_singular_values,
        mode_counts=mode_counts,
        mode_thresholds=mode_thresholds,
        current_output_powers_over_noise=current_output_powers_over_noise,
        average_mode_snrs_current=average_mode_snrs_current,
        mode_scaled_output_powers_over_noise=mode_scaled_output_powers_over_noise,
        average_mode_snrs_mode_scaled=average_mode_snrs_mode_scaled,
        ell_maxes=ell_maxes,
        n_omegas=n_omegas,
        n_rs=n_rs,
        n_sv_per_ells=n_sv_per_ells,
        converged=converged,
        rounds=rounds,
        capacity_rel_changes=metric_rel_changes,
        metric_rel_changes=metric_rel_changes,
        top_sv_rel_changes=top_sv_rel_changes,
        noise_levels=noise_levels,
    )

    frequencies_hz = frequencies_khz * 1e3
    bitrate_over_f3 = bitrates / np.maximum(frequencies_hz**3, 1e-30)
    channel_capacity_over_f3 = channel_capacities / np.maximum(frequencies_hz**3, 1e-30)
    mode_scaled_capacity_over_f3 = mode_scaled_capacities / np.maximum(frequencies_hz**3, 1e-30)
    mode_count_over_f3 = mode_counts / np.maximum(frequencies_hz**3, 1e-30)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, bitrates, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("bitrate")
    ax.set_title("Spherical broadband analytical bitrate vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "bitrate_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, bitrate_over_f3, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("bitrate / f^3")
    ax.set_title("Spherical broadband analytical bitrate / f^3")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "bitrate_over_f3_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, channel_capacities, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("channel capacity")
    ax.set_title("Spherical broadband analytical channel capacity vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "channel_capacity_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, channel_capacity_over_f3, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("channel capacity / f^3")
    ax.set_title("Spherical broadband analytical channel capacity / f^3")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "channel_capacity_over_f3_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, mode_scaled_capacities, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("mode-scaled channel capacity")
    ax.set_title("Mode-scaled channel capacity vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "mode_scaled_capacity_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, mode_scaled_capacity_over_f3, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("mode-scaled capacity / f^3")
    ax.set_title("Mode-scaled channel capacity / f^3")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "mode_scaled_capacity_over_f3_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, mode_counts, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("significant mode count")
    ax.set_title("Significant mode count vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "mode_count_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, mode_count_over_f3, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("mode count / f^3")
    ax.set_title("Significant mode count / f^3")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "mode_count_over_f3_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, average_mode_snrs_current, marker="o", linewidth=2.0, label="current")
    ax.plot(
        frequencies_khz,
        average_mode_snrs_mode_scaled,
        marker="s",
        linewidth=2.0,
        label="mode-scaled",
    )
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("average per-mode output SNR")
    ax.set_title("Average per-mode output SNR vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "average_mode_snr_vs_frequency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(frequencies_khz, top_singular_values, marker="o", linewidth=2.0)
    ax.set_xlabel("center frequency (kHz)")
    ax.set_ylabel("top singular value")
    ax.set_title("Top singular value vs frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "top_singular_value_vs_frequency.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    frequencies_khz = resolve_frequencies(args)
    results: list[dict[str, float | int]] = []
    for idx, freq_khz in enumerate(frequencies_khz, start=1):
        results.append(
            estimate_frequency_result(
                float(freq_khz),
                args,
                progress_prefix=f"[{idx}/{len(frequencies_khz)}]",
            )
        )

    write_output_artifacts(results, outdir)
    print(f"Wrote {outdir / 'bitrate_vs_frequency.png'}")
    print(f"Wrote {outdir / 'bitrate_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'channel_capacity_vs_frequency.png'}")
    print(f"Wrote {outdir / 'channel_capacity_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_scaled_capacity_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_scaled_capacity_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_count_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_count_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'average_mode_snr_vs_frequency.png'}")
    print(f"Wrote {outdir / 'top_singular_value_vs_frequency.png'}")
    print(f"Wrote {outdir / 'spherical_broadband_capacity_vs_frequency.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
