#!/usr/bin/env python3
"""Analyze a 50 kHz Modal ultrasound source/sensor convergence sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guti.capacity import (
    get_bitrate,
    get_capacity,
    total_input_power_from_average_output_power,
)
from guti.noise_models import compute_average_output_power, compute_output_noise_std
from guti.noise_models import compute_input_amplitude
from guti.parameters import Parameters


FREQUENCY_HZ = 50_000.0
CANONICAL_PATH = Path("results/us_svd_spectrum.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make convergence plots and canonical US spectrum from a Modal "
            "50 kHz source/sensor sweep output directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing downloaded Modal .npz files and json/all_job_results.jsonl.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Analysis output directory. Default: <input-dir>/analysis",
    )
    parser.add_argument(
        "--canonical-path",
        default=str(CANONICAL_PATH),
        help="Canonical NPZ path to overwrite with the largest completed run.",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Optional exact sweep command to include in README.md.",
    )
    parser.add_argument(
        "--input-power-convention",
        choices=["auto", "average_output_power", "fixed_total_source_power"],
        default="auto",
        help="Override input power convention for bitrate/capacity analysis.",
    )
    return parser


def load_json_records(input_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    aggregate_path = input_dir / "all_job_results.jsonl"
    paths: list[Path] = []
    if aggregate_path.exists():
        paths.append(aggregate_path)
    json_dir = input_dir / "json"
    if json_dir.exists():
        paths.extend(sorted(json_dir.glob("*.json")))

    for path in paths:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = [json.loads(path.read_text())]
        for row in rows:
            if row.get("status") not in (None, "ok"):
                continue
            n_sources = row.get("n_sources")
            n_sensors = row.get("n_sensors")
            if n_sources is None or n_sensors is None:
                continue
            records[(int(n_sources), int(n_sensors))] = row
            matrix_size = row.get("matrix_size")
            if matrix_size is not None and len(matrix_size) == 2:
                records[(int(matrix_size[1]), int(n_sensors))] = row
    return records


def parse_label_counts(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"_(\d+)src_(\d+)sensors", path.name)
    if match is None:
        match = re.search(r"(\d+)src_(\d+)sensors", path.name)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def matrix_shape_from_params(params: Parameters, n_singular_values: int) -> tuple[int, int]:
    if params.matrix_size is not None:
        return int(params.matrix_size[0]), int(params.matrix_size[1])
    if params.num_sensors is None or params.num_brain_grid_points is None:
        raise ValueError("US sweep result is missing matrix_size and count metadata")
    # Fallback for older files. This undercounts time-resolved rows but keeps the
    # script usable for inspection; new Modal runs save matrix_size explicitly.
    return max(int(params.num_sensors), n_singular_values), int(params.num_brain_grid_points)


def load_sweep_rows(input_dir: Path, input_power_convention_override: str = "auto") -> list[dict[str, Any]]:
    json_records = load_json_records(input_dir)
    rows: list[dict[str, Any]] = []
    average_output_power = compute_average_output_power("us")
    physical_total_source_power = compute_input_amplitude("us") ** 2

    for path in sorted(input_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "singular_values" not in data.files:
            continue
        params_dict = data["parameters"].item() if "parameters" in data.files else {}
        params = Parameters.from_dict(params_dict or {})
        singular_values = np.asarray(data["singular_values"], dtype=np.float64)
        noise_normalized_singular_values = None
        if "noise_normalized_singular_values" in data.files:
            noise_normalized_singular_values = np.asarray(
                data["noise_normalized_singular_values"],
                dtype=np.float64,
            )
        requested_sources, requested_sensors = parse_label_counts(path)

        n_sensors = params.num_sensors or requested_sensors
        n_brain = params.num_brain_grid_points or requested_sources
        if n_sensors is None or n_brain is None:
            continue
        frequency_hz = float(params.frequency_hz or FREQUENCY_HZ)
        if int(round(frequency_hz)) != int(FREQUENCY_HZ):
            continue

        n_outputs, n_sources = matrix_shape_from_params(params, len(singular_values))
        time_resolution = float(params.time_resolution or 1.0)
        output_noise = compute_output_noise_std(
            "us",
            n_sensors=int(n_sensors),
            frequency_hz=frequency_hz,
        )
        record = json_records.get((int(n_brain), int(n_sensors)), {})
        source_amplitude_scale = float(record.get("source_amplitude_scale") or 1.0)
        input_power_convention = record.get("input_power_convention") or "average_output_power"
        if input_power_convention_override != "auto":
            input_power_convention = input_power_convention_override
        if input_power_convention == "fixed_total_source_power":
            total_input_power = physical_total_source_power / (source_amplitude_scale**2)
        else:
            total_input_power = total_input_power_from_average_output_power(
                singular_values,
                average_output_power=average_output_power,
                n_sources=n_sources,
                n_outputs=n_outputs,
            )
        if noise_normalized_singular_values is None:
            noise_model_type = "scalar_iid"
            bitrate = float(
                get_bitrate(
                    singular_values,
                    n_sources=n_sources,
                    total_input_power=total_input_power,
                    noise=output_noise,
                    time_resolution=time_resolution,
                )
            )
            capacity = float(
                get_capacity(
                    singular_values[singular_values > 0],
                    total_input_power=total_input_power,
                    noise=output_noise,
                    time_resolution=time_resolution,
                )
            )
        else:
            noise_model_type = "spatial_covariance"
            bitrate = float(
                get_bitrate(
                    noise_normalized_singular_values,
                    n_sources=n_sources,
                    total_input_power=total_input_power,
                    noise=1.0,
                    time_resolution=time_resolution,
                )
            )
            capacity = float(
                get_capacity(
                    noise_normalized_singular_values[
                        noise_normalized_singular_values > 0
                    ],
                    total_input_power=total_input_power,
                    noise=1.0,
                    time_resolution=time_resolution,
                )
            )
        rows.append(
            {
                "path": path,
                "params": params,
                "singular_values": singular_values,
                "requested_sources": requested_sources,
                "requested_sensors": requested_sensors,
                "num_brain_grid_points": int(n_brain),
                "num_sensors": int(n_sensors),
                "frequency_hz": frequency_hz,
                "n_outputs": n_outputs,
                "n_sources": n_sources,
                "n_singular_values": int(len(singular_values)),
                "time_resolution_seconds": time_resolution,
                "first_singular_value": float(singular_values[0]),
                "rank_gt_1pct": int(np.sum(singular_values / singular_values[0] > 0.01)),
                "noise_std": float(output_noise),
                "noise_model_type": noise_model_type,
                "noise_correlation_length_mm": params.noise_correlation_length_mm,
                "noise_correlation_kernel": params.noise_correlation_kernel,
                "source_power_normalization": record.get("source_power_normalization") or "none",
                "source_amplitude_scale": source_amplitude_scale,
                "input_power_convention": input_power_convention,
                "total_input_power": float(total_input_power),
                "bitrate_bits_per_s": bitrate,
                "channel_capacity_bits_per_s": capacity,
                "modal_json_bitrate": record.get("bitrate"),
                "modal_gram_output_path": record.get("modal_gram_output_path")
                or record.get("gram_output_path"),
                "modal_gram_size_bytes": record.get("modal_gram_size_bytes"),
            }
        )

    rows.sort(key=lambda r: (r["num_sensors"], r["num_brain_grid_points"]))
    return rows


def write_metrics(rows: list[dict[str, Any]], outdir: Path) -> None:
    csv_path = outdir / "metrics.csv"
    fields = [
        "num_sensors",
        "num_brain_grid_points",
        "n_outputs",
        "n_sources",
        "n_singular_values",
        "time_resolution_seconds",
        "first_singular_value",
        "rank_gt_1pct",
        "noise_std",
        "noise_model_type",
        "noise_correlation_length_mm",
        "noise_correlation_kernel",
        "source_power_normalization",
        "source_amplitude_scale",
        "input_power_convention",
        "total_input_power",
        "bitrate_bits_per_s",
        "channel_capacity_bits_per_s",
        "modal_gram_output_path",
        "modal_gram_size_bytes",
        "source_npz",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field != "source_npz"},
                    "source_npz": str(row["path"]),
                }
            )

    json_rows = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in {"singular_values", "params"}}
        item["path"] = str(item["path"])
        json_rows.append(item)
    (outdir / "metrics.json").write_text(json.dumps(json_rows, indent=2), encoding="utf-8")


def plot_metric_vs_sources(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    relative: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for n_sensors in sorted({row["num_sensors"] for row in rows}):
        sensor_rows = [row for row in rows if row["num_sensors"] == n_sensors]
        sensor_rows.sort(key=lambda r: r["num_brain_grid_points"])
        xs = np.asarray([row["num_brain_grid_points"] for row in sensor_rows], dtype=float)
        ys = np.asarray([row[metric_key] for row in sensor_rows], dtype=float)
        if relative:
            ys = ys / ys[-1]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{n_sensors} sensors")
    ax.set_xscale("log")
    ax.set_xlabel("realized source points")
    ax.set_ylabel("relative to largest source count" if relative else ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def largest_common_source_count(rows: list[dict[str, Any]]) -> int:
    sensor_counts = sorted({row["num_sensors"] for row in rows})
    common: set[int] | None = None
    for n_sensors in sensor_counts:
        values = {row["num_brain_grid_points"] for row in rows if row["num_sensors"] == n_sensors}
        common = values if common is None else common & values
    if common:
        return max(common)
    return max(row["num_brain_grid_points"] for row in rows)


def rows_for_sensor_convergence(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    source_count = largest_common_source_count(rows)
    selected = [row for row in rows if row["num_brain_grid_points"] == source_count]
    if selected:
        return source_count, sorted(selected, key=lambda r: r["num_sensors"])

    # Fallback: choose the largest source-count row for each sensor count.
    selected = []
    for n_sensors in sorted({row["num_sensors"] for row in rows}):
        sensor_rows = [row for row in rows if row["num_sensors"] == n_sensors]
        selected.append(max(sensor_rows, key=lambda r: r["num_brain_grid_points"]))
    return source_count, selected


def plot_metric_vs_sensors(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.asarray([row["num_sensors"] for row in rows], dtype=float)
    ys = np.asarray([row[metric_key] for row in rows], dtype=float)
    ax.plot(xs, ys, marker="o", linewidth=2.0)
    ax.set_xlabel("sensors")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_canonical(row: dict[str, Any], canonical_path: Path) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    params = row["params"]
    params.frequency_hz = FREQUENCY_HZ
    gram_note = ""
    if row.get("modal_gram_output_path"):
        gram_note = (
            f"; modal_gram={row['modal_gram_output_path']} "
            f"({row.get('modal_gram_size_bytes')} bytes)"
        )
    params.comment = (
        f"Canonical 50 kHz ultrasound analytical spectrum from Modal convergence sweep; "
        f"source_npz={row['path']}{gram_note}"
    )
    np.savez(
        canonical_path,
        singular_values=row["singular_values"],
        parameters=asdict(params),
    )


def markdown_link(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return f"[{rel.as_posix()}]({rel.as_posix()})"


def write_readme(
    rows: list[dict[str, Any]],
    outdir: Path,
    canonical_row: dict[str, Any],
    sensor_source_count: int,
    sensor_rows: list[dict[str, Any]],
    command: str | None,
    canonical_path: Path,
) -> None:
    lines = [
        "# Ultrasound 50 kHz Convergence Check",
        "",
        "This directory contains the local analysis for a Modal sweep of the analytical ultrasound forward model at 50 kHz.",
        "",
        "## Sweep",
        "",
    ]
    if command:
        lines.extend(["Command:", "", f"```bash\n{command}\n```", ""])
    lines.extend(
        [
            f"- Completed NPZ results analyzed: {len(rows)}",
            f"- Sensor counts: {', '.join(map(str, sorted({row['num_sensors'] for row in rows})))}",
            f"- Realized source counts: {', '.join(map(str, sorted({row['num_brain_grid_points'] for row in rows})))}",
            f"- Input power convention: {canonical_row['input_power_convention']}",
            f"- Source power normalization: {canonical_row['source_power_normalization']}",
            f"- Noise model type: {canonical_row['noise_model_type']}",
            "",
            "## Plots",
            "",
            f"- {markdown_link(outdir / 'bitrate_vs_sources.png', outdir)}",
            f"- {markdown_link(outdir / 'capacity_vs_sources.png', outdir)}",
            f"- {markdown_link(outdir / 'relative_bitrate_vs_sources.png', outdir)}",
            f"- {markdown_link(outdir / 'relative_capacity_vs_sources.png', outdir)}",
            f"- {markdown_link(outdir / 'bitrate_vs_sensors_largest_sources.png', outdir)}",
            f"- {markdown_link(outdir / 'capacity_vs_sensors_largest_sources.png', outdir)}",
            "",
            "## Canonical Result",
            "",
            f"The canonical result was updated at `{canonical_path}` from `{canonical_row['path']}`.",
            "",
            "| quantity | value |",
            "| --- | ---: |",
            f"| sensors | {canonical_row['num_sensors']} |",
            f"| realized source points | {canonical_row['num_brain_grid_points']} |",
            f"| matrix shape | {canonical_row['n_outputs']} x {canonical_row['n_sources']} |",
            f"| singular values | {canonical_row['n_singular_values']} |",
            f"| first singular value | {canonical_row['first_singular_value']:.6g} |",
            f"| rank > 1% first SV | {canonical_row['rank_gt_1pct']} |",
            f"| source amplitude scale | {canonical_row['source_amplitude_scale']:.6g} |",
            f"| total input power | {canonical_row['total_input_power']:.6g} |",
            f"| bitrate | {canonical_row['bitrate_bits_per_s']:.6g} bit/s |",
            f"| water-filled channel capacity | {canonical_row['channel_capacity_bits_per_s']:.6g} bit/s |",
            f"| Modal Gram path | `{canonical_row.get('modal_gram_output_path')}` |",
            f"| Modal Gram size | {canonical_row.get('modal_gram_size_bytes')} bytes |",
            "",
            f"Sensor-scaling plots use the largest common realized source count, `{sensor_source_count}`, when available.",
            "",
            "| sensors | sources | bitrate bit/s | channel capacity bit/s |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sensor_rows:
        lines.append(
            f"| {row['num_sensors']} | {row['num_brain_grid_points']} | "
            f"{row['bitrate_bits_per_s']:.6g} | {row['channel_capacity_bits_per_s']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            f"- {markdown_link(outdir / 'metrics.csv', outdir)}",
            f"- {markdown_link(outdir / 'metrics.json', outdir)}",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir) if args.outdir is not None else input_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_sweep_rows(input_dir, args.input_power_convention)
    if not rows:
        raise SystemExit(f"No 50 kHz sweep NPZ files found in {input_dir}")

    write_metrics(rows, outdir)
    plot_metric_vs_sources(
        rows,
        metric_key="bitrate_bits_per_s",
        ylabel="bitrate (bit/s)",
        title="50 kHz US bitrate convergence vs source count",
        output_path=outdir / "bitrate_vs_sources.png",
    )
    plot_metric_vs_sources(
        rows,
        metric_key="channel_capacity_bits_per_s",
        ylabel="channel capacity (bit/s)",
        title="50 kHz US water-filled capacity convergence vs source count",
        output_path=outdir / "capacity_vs_sources.png",
    )
    plot_metric_vs_sources(
        rows,
        metric_key="bitrate_bits_per_s",
        ylabel="relative bitrate",
        title="50 kHz US relative bitrate convergence vs source count",
        output_path=outdir / "relative_bitrate_vs_sources.png",
        relative=True,
    )
    plot_metric_vs_sources(
        rows,
        metric_key="channel_capacity_bits_per_s",
        ylabel="relative capacity",
        title="50 kHz US relative capacity convergence vs source count",
        output_path=outdir / "relative_capacity_vs_sources.png",
        relative=True,
    )

    sensor_source_count, sensor_rows = rows_for_sensor_convergence(rows)
    plot_metric_vs_sensors(
        sensor_rows,
        metric_key="bitrate_bits_per_s",
        ylabel="bitrate (bit/s)",
        title=f"50 kHz US bitrate vs sensors at {sensor_source_count} sources",
        output_path=outdir / "bitrate_vs_sensors_largest_sources.png",
    )
    plot_metric_vs_sensors(
        sensor_rows,
        metric_key="channel_capacity_bits_per_s",
        ylabel="channel capacity (bit/s)",
        title=f"50 kHz US water-filled capacity vs sensors at {sensor_source_count} sources",
        output_path=outdir / "capacity_vs_sensors_largest_sources.png",
    )

    canonical_row = max(rows, key=lambda r: (r["num_brain_grid_points"], r["num_sensors"]))
    canonical_path = Path(args.canonical_path)
    save_canonical(canonical_row, canonical_path)
    write_readme(
        rows,
        outdir,
        canonical_row,
        sensor_source_count,
        sensor_rows,
        args.command,
        canonical_path,
    )

    print(f"Analyzed {len(rows)} 50 kHz sweep results")
    print(f"Wrote analysis to {outdir}")
    print(f"Updated canonical spectrum at {canonical_path}")
    print(
        "Canonical: "
        f"sensors={canonical_row['num_sensors']} "
        f"sources={canonical_row['num_brain_grid_points']} "
        f"bitrate={canonical_row['bitrate_bits_per_s']:.6g} "
        f"capacity={canonical_row['channel_capacity_bits_per_s']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
