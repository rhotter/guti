#!/usr/bin/env python3
"""Plot bitrate trends from downloaded ultrasound analytical sweep JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot bitrate vs number of sources and bitrate vs number of sensors "
            "from downloaded per-job JSON results."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="results/variants/us_free_field_analytical_frequency_sweep/json",
        help="Directory containing per-job JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/variants/us_free_field_analytical_frequency_sweep/plots",
        help="Directory where plot PNGs will be written.",
    )
    parser.add_argument(
        "--frequency-khz",
        type=int,
        default=None,
        help="Optional frequency to plot. Default: plot every frequency present.",
    )
    parser.add_argument(
        "--fixed-sources",
        type=int,
        default=None,
        help=(
            "When set together with --fixed-sensors, plot bitrate vs frequency "
            "for the fixed source/sensor pair on log-log axes with a linear fit."
        ),
    )
    parser.add_argument(
        "--fixed-sensors",
        type=int,
        default=None,
        help=(
            "When set together with --fixed-sources, plot bitrate vs frequency "
            "for the fixed source/sensor pair on log-log axes with a linear fit."
        ),
    )
    return parser


def load_latest_rows(input_dir: Path) -> list[dict]:
    latest_by_tuple: dict[tuple[int, int, int], tuple[float, dict]] = {}
    for json_path in sorted(input_dir.glob("*.json")):
        row = json.loads(json_path.read_text(encoding="utf-8"))
        if row.get("status") != "ok":
            continue
        bitrate = row.get("bitrate")
        if bitrate is None:
            continue
        key = (
            int(row["frequency_khz"]),
            int(row["n_sources"]),
            int(row["n_sensors"]),
        )
        current = latest_by_tuple.get(key)
        mtime = json_path.stat().st_mtime
        if current is None or mtime >= current[0]:
            row = dict(row)
            row["_json_path"] = str(json_path)
            latest_by_tuple[key] = (mtime, row)
    return [row for _, row in latest_by_tuple.values()]


def plot_for_frequency(rows: list[dict], output_dir: Path, frequency_khz: int) -> list[Path]:
    freq_rows = [row for row in rows if int(row["frequency_khz"]) == frequency_khz]
    if not freq_rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    all_sources = sorted({int(row["n_sources"]) for row in freq_rows})
    all_sensors = sorted({int(row["n_sensors"]) for row in freq_rows})
    expected = len(all_sources) * len(all_sensors)
    completed = len(freq_rows)
    saved_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    for n_sensors in all_sensors:
        series = sorted(
            (
                int(row["n_sources"]),
                float(row["bitrate"]),
            )
            for row in freq_rows
            if int(row["n_sensors"]) == n_sensors
        )
        if not series:
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"{n_sensors} sensors")
    ax.set_title(
        f"US Analytical Bitrate vs Sources at {frequency_khz} kHz "
        f"({completed}/{expected} completed)"
    )
    ax.set_xlabel("Number of Sources")
    ax.set_ylabel("Bitrate")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Series")
    fig.tight_layout()
    sources_path = output_dir / f"{frequency_khz}khz_bitrate_vs_sources_by_sensor.png"
    fig.savefig(sources_path, dpi=200)
    plt.close(fig)
    saved_paths.append(sources_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for n_sources in all_sources:
        series = sorted(
            (
                int(row["n_sensors"]),
                float(row["bitrate"]),
            )
            for row in freq_rows
            if int(row["n_sources"]) == n_sources
        )
        if not series:
            continue
        xs, ys = zip(*series)
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"{n_sources} sources")
    ax.set_title(
        f"US Analytical Bitrate vs Sensors at {frequency_khz} kHz "
        f"({completed}/{expected} completed)"
    )
    ax.set_xlabel("Number of Sensors")
    ax.set_ylabel("Bitrate")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Series")
    fig.tight_layout()
    sensors_path = output_dir / f"{frequency_khz}khz_bitrate_vs_sensors_by_source.png"
    fig.savefig(sensors_path, dpi=200)
    plt.close(fig)
    saved_paths.append(sensors_path)

    return saved_paths


def plot_frequency_trend(
    rows: list[dict],
    output_dir: Path,
    fixed_sources: int,
    fixed_sensors: int,
) -> list[Path]:
    trend_rows = sorted(
        (
            row
            for row in rows
            if int(row["n_sources"]) == fixed_sources
            and int(row["n_sensors"]) == fixed_sensors
        ),
        key=lambda row: float(row["frequency_khz"]),
    )
    if not trend_rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies_khz = np.array(
        [float(row["frequency_khz"]) for row in trend_rows],
        dtype=float,
    )
    bitrates = np.array([float(row["bitrate"]) for row in trend_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(frequencies_khz, bitrates, marker="o", linewidth=2, label="Sweep results")

    if len(trend_rows) >= 2:
        logx = np.log10(frequencies_khz)
        logy = np.log10(bitrates)
        slope, intercept = np.polyfit(logx, logy, 1)
        residual = logy - (slope * logx + intercept)
        total = logy - logy.mean()
        r2 = 1.0 - float(np.sum(residual**2) / np.sum(total**2))

        xfit = np.logspace(np.log10(frequencies_khz.min()), np.log10(frequencies_khz.max()), 200)
        yfit = 10 ** (intercept + slope * np.log10(xfit))
        ax.loglog(xfit, yfit, linewidth=2, label=f"Best fit: slope = {slope:.4f}")
        ax.text(
            0.04,
            0.06,
            f"log10 fit: y = {slope:.4f} x + {intercept:.4f}\n$R^2$ = {r2:.6f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    for x, y in zip(frequencies_khz, bitrates):
        ax.annotate(f"{int(x)} kHz", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_title(
        f"US Analytical Bitrate vs Frequency\n"
        f"{fixed_sources} sources, {fixed_sensors} sensors"
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Bitrate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / (
        f"bitrate_vs_frequency_{fixed_sources}src_{fixed_sensors}sensors_loglog_fit.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return [output_path]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    rows = load_latest_rows(input_dir)
    if not rows:
        parser.error(f"No successful JSON results with bitrate found in {input_dir}")

    saved_paths: list[Path] = []
    if (args.fixed_sources is None) != (args.fixed_sensors is None):
        parser.error("--fixed-sources and --fixed-sensors must be provided together")

    if args.fixed_sources is not None and args.fixed_sensors is not None:
        saved_paths.extend(
            plot_frequency_trend(
                rows,
                output_dir,
                fixed_sources=args.fixed_sources,
                fixed_sensors=args.fixed_sensors,
            )
        )
    else:
        if args.frequency_khz is not None:
            frequencies = [args.frequency_khz]
        else:
            frequencies = sorted({int(row["frequency_khz"]) for row in rows})

        for frequency_khz in frequencies:
            saved_paths.extend(plot_for_frequency(rows, output_dir, frequency_khz))

    if not saved_paths:
        parser.error("No plots were produced for the requested frequency selection")

    print(f"Loaded {len(rows)} deduplicated result points")
    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
