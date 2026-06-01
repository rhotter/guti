#!/usr/bin/env python3
"""Plot bitrate and first-singular-value heatmaps for the US analytical sweep."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from guti.capacity import (
    get_bitrate,
    get_capacity,
    total_input_power_from_average_output_power,
)
from guti.noise_models import compute_average_output_power, compute_output_noise_std
from guti.parameters import Parameters


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved US analytical sweep results as per-frequency heatmaps."
    )
    parser.add_argument(
        "--modality",
        default="us_free_field_analytical_frequency_sweep",
        help="Variant modality directory under results/variants.",
    )
    parser.add_argument(
        "--frequencies-khz",
        type=parse_int_csv,
        default=[50, 100, 150, 250],
        help="Comma-separated frequencies in kHz to plot.",
    )
    parser.add_argument(
        "--sensor-counts",
        type=parse_int_csv,
        default=[1000, 3250, 5500, 7750, 10000],
        help="Comma-separated sensor counts to plot on the x-axis.",
    )
    parser.add_argument(
        "--comment-substring",
        default=None,
        help="Optional substring filter on Parameters.comment to isolate one sweep.",
    )
    parser.add_argument(
        "--outdir",
        default="plots/us_analytical_heatmaps",
        help="Directory to write the PNG files into.",
    )
    parser.add_argument(
        "--capacity-time-resolution",
        type=float,
        default=1.0,
        help="time_resolution passed to get_capacity(). Default: 1.0",
    )
    return parser


def load_variants(modality: str) -> list[dict]:
    variant_dir = Path("results") / "variants" / modality
    if not variant_dir.exists():
        raise FileNotFoundError(f"No variant directory found at {variant_dir}")

    variants: list[dict] = []
    for path in sorted(variant_dir.glob("*.npz")):
        if len(path.stem) != 8:
            continue
        data = np.load(path, allow_pickle=True)
        if "parameters" not in data or "singular_values" not in data:
            continue
        params_dict = data["parameters"].item()
        params = Parameters.from_dict(params_dict) if params_dict is not None else Parameters()
        variants.append(
            {
                "path": path,
                "mtime_ns": path.stat().st_mtime_ns,
                "params": params,
                "s": data["singular_values"],
            }
        )
    return variants


def filter_variants(
    variants: list[dict],
    frequencies_khz: list[int],
    sensor_counts: list[int],
    comment_substring: str | None,
) -> list[dict]:
    wanted_freqs_hz = {freq * 1000 for freq in frequencies_khz}
    wanted_sensors = set(sensor_counts)
    filtered: list[dict] = []
    for variant in variants:
        params = variant["params"]
        if params.frequency_hz is None or params.num_sensors is None or params.num_brain_grid_points is None:
            continue
        freq_hz = int(round(params.frequency_hz))
        if freq_hz not in wanted_freqs_hz:
            continue
        if params.num_sensors not in wanted_sensors:
            continue
        if comment_substring is not None:
            comment = params.comment or ""
            if comment_substring not in comment:
                continue
        filtered.append(variant)
    return filtered


def dedupe_variants(variants: list[dict]) -> list[dict]:
    latest_by_key: dict[tuple[int, int, int], dict] = {}
    for variant in variants:
        params = variant["params"]
        key = (
            int(round(params.frequency_hz)),
            int(params.num_brain_grid_points),
            int(params.num_sensors),
        )
        current = latest_by_key.get(key)
        if current is None or variant["mtime_ns"] > current["mtime_ns"]:
            latest_by_key[key] = variant
    return list(latest_by_key.values())


def plot_grid(
    grid: np.ndarray,
    sensor_counts: list[int],
    source_counts: list[int],
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(sensor_counts)), sensor_counts, rotation=45, ha="right")
    ax.set_yticks(range(len(source_counts)), source_counts)
    ax.set_xlabel("num_sensors")
    ax.set_ylabel("num_brain_grid_points (realized)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_max_summary(
    frequencies_khz: list[int],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(frequencies_khz, values, marker="o", linewidth=2.0)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()

    variants = load_variants(args.modality)
    variants = filter_variants(
        variants,
        frequencies_khz=args.frequencies_khz,
        sensor_counts=args.sensor_counts,
        comment_substring=args.comment_substring,
    )
    variants = dedupe_variants(variants)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not variants:
        print("No matching variants found.")
        return 1

    summary_frequencies_khz: list[int] = []
    summary_max_bitrates: list[float] = []
    summary_max_capacities: list[float] = []
    average_output_power = compute_average_output_power("us_analytical")

    for freq_khz in args.frequencies_khz:
        freq_hz = freq_khz * 1000
        freq_variants = [
            variant
            for variant in variants
            if int(round(variant["params"].frequency_hz)) == freq_hz
        ]
        if not freq_variants:
            print(f"{freq_khz} kHz: no matching variants")
            continue

        source_counts = sorted(
            {
                int(variant["params"].num_brain_grid_points)
                for variant in freq_variants
            }
        )
        bitrate_grid = np.full((len(source_counts), len(args.sensor_counts)), np.nan)
        first_sv_grid = np.full((len(source_counts), len(args.sensor_counts)), np.nan)
        capacity_grid = np.full((len(source_counts), len(args.sensor_counts)), np.nan)

        for variant in freq_variants:
            params = variant["params"]
            s = variant["s"]
            source_count = int(params.num_brain_grid_points)
            sensor_count = int(params.num_sensors)
            i = source_counts.index(source_count)
            j = args.sensor_counts.index(sensor_count)

            s_normalized = s / math.sqrt(source_count * sensor_count)
            matrix_size = getattr(params, "matrix_size", None)
            if matrix_size is None:
                n_outputs, n_sources = sensor_count, source_count
            else:
                n_outputs, n_sources = matrix_size
            output_noise = compute_output_noise_std(
                "us_analytical",
                n_sensors=sensor_count,
                frequency_hz=freq_hz,
            )
            total_input_power = total_input_power_from_average_output_power(
                s,
                average_output_power=average_output_power,
                n_sources=int(n_sources),
                n_outputs=int(n_outputs),
            )

            first_sv_grid[i, j] = float(s_normalized[0])
            bitrate_grid[i, j] = float(
                get_bitrate(
                    s,
                    n_sources=int(n_sources),
                    total_input_power=total_input_power,
                    noise=output_noise,
                    time_resolution=1.0,
                )
            )
            s_for_capacity = s[np.abs(s) > 0]
            capacity_grid[i, j] = float(
                get_capacity(
                    s_for_capacity.astype(np.float64),
                    total_input_power=total_input_power,
                    noise=output_noise,
                    time_resolution=args.capacity_time_resolution,
                )
            )

        filled = int(np.isfinite(bitrate_grid).sum())
        total = int(bitrate_grid.size)
        plot_grid(
            bitrate_grid,
            sensor_counts=args.sensor_counts,
            source_counts=source_counts,
            title=f"bitrate at {freq_khz} kHz ({filled}/{total} filled)",
            cbar_label="bitrate",
            output_path=outdir / f"bitrate_{freq_khz}khz.png",
        )
        plot_grid(
            first_sv_grid,
            sensor_counts=args.sensor_counts,
            source_counts=source_counts,
            title=f"normalized first singular value at {freq_khz} kHz ({filled}/{total} filled)",
            cbar_label="normalized first singular value",
            output_path=outdir / f"first_sv_{freq_khz}khz.png",
        )
        if filled > 0:
            max_flat_index = int(np.nanargmax(bitrate_grid))
            max_i, max_j = np.unravel_index(max_flat_index, bitrate_grid.shape)
            max_bitrate = float(bitrate_grid[max_i, max_j])
            max_capacity_flat_index = int(np.nanargmax(capacity_grid))
            max_capacity_i, max_capacity_j = np.unravel_index(
                max_capacity_flat_index,
                capacity_grid.shape,
            )
            max_capacity = float(capacity_grid[max_capacity_i, max_capacity_j])
            summary_frequencies_khz.append(freq_khz)
            summary_max_bitrates.append(max_bitrate)
            summary_max_capacities.append(max_capacity)
            print(
                f"{freq_khz} kHz: wrote plots with {filled}/{total} populated cells; "
                f"max bitrate={max_bitrate:.3f} at "
                f"sources={source_counts[max_i]}, sensors={args.sensor_counts[max_j]}; "
                f"max channel capacity={max_capacity:.3f} at "
                f"sources={source_counts[max_capacity_i]}, sensors={args.sensor_counts[max_capacity_j]}"
            )
        else:
            print(f"{freq_khz} kHz: wrote plots with 0/{total} populated cells")

    if summary_frequencies_khz:
        plot_max_summary(
            summary_frequencies_khz,
            summary_max_bitrates,
            ylabel="max bitrate",
            title="Maximum bitrate by frequency",
            output_path=outdir / "max_bitrate_by_frequency.png",
        )
        print(
            "Wrote max-bitrate summary to "
            f"{outdir / 'max_bitrate_by_frequency.png'}"
        )
        plot_max_summary(
            summary_frequencies_khz,
            summary_max_capacities,
            ylabel="max channel capacity (bits/use)",
            title="Maximum channel capacity by frequency",
            output_path=outdir / "max_channel_capacity_by_frequency.png",
        )
        print(
            "Wrote max-channel-capacity summary to "
            f"{outdir / 'max_channel_capacity_by_frequency.png'}"
        )

    print(f"Wrote heatmaps to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
