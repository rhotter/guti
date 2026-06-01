#!/usr/bin/env python3
"""Analyze JSON-only 50 kHz ultrasound SLQ convergence runs.

The exact SVD sweep writes NPZ spectra. The high-source SLQ sweep only writes
JSON records with estimated equal-power bitrate. This script combines both so
the high-source estimator can be compared against the exact low-source curve.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-metrics",
        required=True,
        help="metrics.csv from analyze_us_50khz_convergence.py.",
    )
    parser.add_argument(
        "--slq-dir",
        required=True,
        help="Directory containing SLQ JSON records under json/*.json.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Analysis output directory. Default: <slq-dir>/analysis",
    )
    return parser


def load_exact_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "method": "exact_svd",
                    "num_sensors": int(row["num_sensors"]),
                    "num_brain_grid_points": int(row["num_brain_grid_points"]),
                    "n_outputs": int(row["n_outputs"]),
                    "n_sources": int(row["n_sources"]),
                    "bitrate_bits_per_s": float(row["bitrate_bits_per_s"]),
                    "source": row.get("source_npz", ""),
                }
            )
    rows.sort(key=lambda row: (row["num_sensors"], row["num_brain_grid_points"]))
    return rows


def load_slq_rows(slq_dir: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    json_dir = slq_dir / "json"
    for path in sorted(json_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        if record.get("bitrate_slq") is None and record.get("bitrate") is None:
            continue
        matrix_size = record.get("matrix_size") or [None, None]
        realized_sources = int(matrix_size[1] or record["n_sources"])
        n_sensors = int(record["n_sensors"])
        rows_by_key[(realized_sources, n_sensors)] = {
            "method": "slq",
            "num_sensors": n_sensors,
            "num_brain_grid_points": realized_sources,
            "n_outputs": int(matrix_size[0] or 0),
            "n_sources": realized_sources,
            "requested_sources": int(record["n_sources"]),
            "bitrate_bits_per_s": float(record.get("bitrate_slq") or record["bitrate"]),
            "slq_s": _arg_value(record.get("analytical_args") or [], "--slq_s"),
            "slq_t": _arg_value(record.get("analytical_args") or [], "--slq_t"),
            "slq_batch": _arg_value(record.get("analytical_args") or [], "--slq_batch"),
            "slq_frobenius_norm_sq": record.get("slq_frobenius_norm_sq"),
            "slq_logdet_alpha": record.get("slq_logdet_alpha"),
            "total_input_power": record.get("total_input_power"),
            "noise_level": record.get("noise_level"),
            "raw_noise_level": record.get("raw_noise_level"),
            "noise_multiplier": record.get("noise_multiplier", 1.0),
            "average_output_signal_amplitude": record.get("average_output_signal_amplitude", 1e-3),
            "average_output_power": record.get("average_output_power"),
            "effective_time_resolution_seconds": record.get("effective_time_resolution_seconds"),
            "time_step_seconds": record.get("time_step_seconds"),
            "input_power_convention": record.get("input_power_convention"),
            "matrix_normalization": record.get("matrix_normalization"),
            "source": str(path),
        }
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (row["num_sensors"], row["num_brain_grid_points"]))
    return rows


def _arg_value(args: list[str], flag: str) -> int | None:
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return int(args[idx + 1])


def write_metrics(rows: list[dict[str, Any]], outdir: Path) -> None:
    fields = [
        "method",
        "num_sensors",
        "num_brain_grid_points",
        "n_outputs",
        "n_sources",
        "requested_sources",
        "bitrate_bits_per_s",
        "slq_s",
        "slq_t",
        "slq_batch",
        "slq_frobenius_norm_sq",
        "slq_logdet_alpha",
        "total_input_power",
        "noise_level",
        "raw_noise_level",
        "noise_multiplier",
        "average_output_signal_amplitude",
        "average_output_power",
        "effective_time_resolution_seconds",
        "time_step_seconds",
        "input_power_convention",
        "matrix_normalization",
        "source",
    ]
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (outdir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_bitrate_vs_sources(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sensor_counts = sorted({row["num_sensors"] for row in rows})
    for idx, n_sensors in enumerate(sensor_counts):
        color = color_cycle[idx % len(color_cycle)]
        for method, marker, linestyle, label_suffix in [
            ("exact_svd", "o", "-", "exact SVD"),
            ("slq", "x", "--", "SLQ"),
        ]:
            sensor_rows = [
                row
                for row in rows
                if row["num_sensors"] == n_sensors and row["method"] == method
            ]
            if not sensor_rows:
                continue
            sensor_rows.sort(key=lambda row: row["num_brain_grid_points"])
            ax.plot(
                [row["num_brain_grid_points"] for row in sensor_rows],
                [row["bitrate_bits_per_s"] for row in sensor_rows],
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                color=color,
                label=f"{n_sensors} sensors {label_suffix}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("realized source points")
    ax.set_ylabel("bitrate (bit/s)")
    ax.set_title("50 kHz US bitrate convergence: exact SVD plus streaming SLQ")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_relative_bitrate_vs_sources(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sensor_counts = sorted({row["num_sensors"] for row in rows})
    for idx, n_sensors in enumerate(sensor_counts):
        color = color_cycle[idx % len(color_cycle)]
        sensor_rows_all = [row for row in rows if row["num_sensors"] == n_sensors]
        if not sensor_rows_all:
            continue
        reference = max(sensor_rows_all, key=lambda row: row["num_brain_grid_points"])[
            "bitrate_bits_per_s"
        ]
        for method, marker, linestyle, label_suffix in [
            ("exact_svd", "o", "-", "exact SVD"),
            ("slq", "x", "--", "SLQ"),
        ]:
            sensor_rows = [row for row in sensor_rows_all if row["method"] == method]
            if not sensor_rows:
                continue
            sensor_rows.sort(key=lambda row: row["num_brain_grid_points"])
            ax.plot(
                [row["num_brain_grid_points"] for row in sensor_rows],
                [row["bitrate_bits_per_s"] / reference for row in sensor_rows],
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                color=color,
                label=f"{n_sensors} sensors {label_suffix}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("realized source points")
    ax.set_ylabel("relative to largest available source count")
    ax.set_title("50 kHz US relative bitrate convergence")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bitrate_vs_sensors(slq_rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    if not slq_rows:
        return []
    max_source = max(row["num_brain_grid_points"] for row in slq_rows)
    selected = [row for row in slq_rows if row["num_brain_grid_points"] == max_source]
    if not selected:
        selected = []
        for n_sensors in sorted({row["num_sensors"] for row in slq_rows}):
            sensor_rows = [row for row in slq_rows if row["num_sensors"] == n_sensors]
            selected.append(max(sensor_rows, key=lambda row: row["num_brain_grid_points"]))
    selected.sort(key=lambda row: row["num_sensors"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        [row["num_sensors"] for row in selected],
        [row["bitrate_bits_per_s"] for row in selected],
        marker="x",
        linestyle="--",
        linewidth=2.0,
    )
    ax.set_xlabel("sensors")
    ax.set_ylabel("SLQ bitrate estimate (bit/s)")
    ax.set_title(f"50 kHz US SLQ bitrate vs sensors near {max_source} sources")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return selected


def write_readme(
    exact_rows: list[dict[str, Any]],
    slq_rows: list[dict[str, Any]],
    selected_sensor_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    slq_time_resolutions = sorted(
        {
            row.get("effective_time_resolution_seconds")
            for row in slq_rows
            if row.get("effective_time_resolution_seconds") is not None
        }
    )
    slq_noise_multipliers = sorted(
        {
            row.get("noise_multiplier")
            for row in slq_rows
            if row.get("noise_multiplier") is not None
        }
    )
    slq_signal_amplitudes = sorted(
        {
            row.get("average_output_signal_amplitude")
            for row in slq_rows
            if row.get("average_output_signal_amplitude") is not None
        }
    )
    lines = [
        "# Ultrasound 50 kHz Streaming SLQ Convergence",
        "",
        "This analysis extends the exact 50 kHz SVD convergence sweep with JSON-only streaming SLQ bitrate estimates at higher source counts.",
        "",
        "SLQ estimates the equal-power trace-log bitrate. It does not save singular spectra and this script does not estimate the water-filled channel capacity.",
        "",
        "## Bitrate Parameters",
        "",
        "These rows use the equal-input-power bitrate formula `sum log2(1 + sigma_i^2 P_source / noise^2) / (2T)`.",
        "",
        f"- Input power convention: `{slq_rows[0].get('input_power_convention', 'average_output_power') if slq_rows else 'n/a'}`",
        f"- Average output signal amplitude(s): {', '.join(f'{value:.6g}' for value in slq_signal_amplitudes) if slq_signal_amplitudes else 'n/a'}",
        f"- Noise multiplier(s): {', '.join(f'{value:.6g}' for value in slq_noise_multipliers) if slq_noise_multipliers else 'n/a'}",
        f"- Bitrate time resolution(s): {', '.join(f'{value:.6g} s' for value in slq_time_resolutions) if slq_time_resolutions else 'n/a'}",
        "",
        "The current completed high-source rows used the very high-SNR default (`1e-3` output amplitude with the modeled acoustic/electronic noise floor) and `T=2e-6 s`. That convention makes weak high-index modes count strongly and can delay apparent source-count convergence.",
        "",
        "## Inputs",
        "",
        f"- Exact SVD rows: {len(exact_rows)}",
        f"- SLQ rows: {len(slq_rows)}",
        f"- Sensor counts: {', '.join(map(str, sorted({row['num_sensors'] for row in exact_rows + slq_rows})))}",
        f"- Largest SLQ realized source count: {max((row['num_brain_grid_points'] for row in slq_rows), default='n/a')}",
        "",
        "## Plots",
        "",
        "- [bitrate_vs_sources_exact_plus_slq.png](bitrate_vs_sources_exact_plus_slq.png)",
        "- [relative_bitrate_vs_sources_exact_plus_slq.png](relative_bitrate_vs_sources_exact_plus_slq.png)",
        "- [bitrate_vs_sensors_largest_slq_sources.png](bitrate_vs_sensors_largest_slq_sources.png)",
        "",
        "## Largest-Source Sensor Sweep",
        "",
        "| sensors | realized sources | SLQ bitrate bit/s |",
        "| ---: | ---: | ---: |",
    ]
    for row in selected_sensor_rows:
        lines.append(
            f"| {row['num_sensors']} | {row['num_brain_grid_points']} | {row['bitrate_bits_per_s']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            "- [metrics.csv](metrics.csv)",
            "- [metrics.json](metrics.json)",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    exact_rows = load_exact_rows(Path(args.exact_metrics))
    slq_dir = Path(args.slq_dir)
    slq_rows = load_slq_rows(slq_dir)
    outdir = Path(args.outdir) if args.outdir else slq_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = exact_rows + slq_rows
    if not all_rows:
        raise SystemExit("No exact or SLQ rows found")
    write_metrics(all_rows, outdir)
    plot_bitrate_vs_sources(all_rows, outdir / "bitrate_vs_sources_exact_plus_slq.png")
    plot_relative_bitrate_vs_sources(
        all_rows,
        outdir / "relative_bitrate_vs_sources_exact_plus_slq.png",
    )
    selected_sensor_rows = plot_bitrate_vs_sensors(
        slq_rows,
        outdir / "bitrate_vs_sensors_largest_slq_sources.png",
    )
    write_readme(exact_rows, slq_rows, selected_sensor_rows, outdir)
    print(f"Wrote SLQ analysis to {outdir}")
    print(f"Exact rows: {len(exact_rows)}")
    print(f"SLQ rows: {len(slq_rows)}")
    if slq_rows:
        largest = max(slq_rows, key=lambda row: (row["num_brain_grid_points"], row["num_sensors"]))
        print(
            "Largest SLQ row: "
            f"sources={largest['num_brain_grid_points']} "
            f"sensors={largest['num_sensors']} "
            f"bitrate={largest['bitrate_bits_per_s']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
