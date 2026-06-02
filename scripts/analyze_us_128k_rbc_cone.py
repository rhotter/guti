#!/usr/bin/env python3
"""Analyze the 128k-source 2 MHz RBC cone-transmission ultrasound sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = Path(
    "results/variants/us_free_field_analytical_50khz_128k_rbc_cone_20260602"
)

# The 6000-sensor Gram completed before the worker disappeared during its first
# bitrate pass, so no successful JSON row carries this path.
EXTRA_GRAM_METADATA = {
    6000: {
        "modal_gram_output_path": "/modal_results/us_analytical_grams/002_50khz_128000src_6000sensors_gram.npy",
        "modal_gram_size_bytes": 68_015_596_932,
        "gram_shape": [130399, 130399],
        "gram_side": "G^T G",
        "gram_save_elapsed_seconds": 268.491,
        "note": "Gram saved in the interrupted save+SLQ run; bitrate came from a follow-up bitrate-only run.",
    }
}

OBSERVED_SLQ_ELAPSED_SECONDS = {
    1000: 151.840,
    3000: 234.618,
    6000: 558.151,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def load_json_records(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((results_dir / "json").glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            row = json.load(fh)
        row["_json_path"] = str(path)
        rows.append(row)
    return rows


def selected_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {}
    for row in records:
        if row.get("status") != "ok" or row.get("bitrate") is None:
            continue
        n_sensors = int(row["n_sensors"])
        current = selected.get(n_sensors)
        has_gram = row.get("modal_gram_output_path") or row.get("gram_output_path")
        current_has_gram = (
            current
            and (current.get("modal_gram_output_path") or current.get("gram_output_path"))
        )
        if current is None or (has_gram and not current_has_gram):
            selected[n_sensors] = dict(row)
    for n_sensors, gram_meta in EXTRA_GRAM_METADATA.items():
        if n_sensors in selected:
            selected[n_sensors].update(
                {
                    key: value
                    for key, value in gram_meta.items()
                    if selected[n_sensors].get(key) in (None, "")
                }
            )
            selected[n_sensors]["gram_note"] = gram_meta.get("note")
    for n_sensors, elapsed in OBSERVED_SLQ_ELAPSED_SECONDS.items():
        if n_sensors in selected and selected[n_sensors].get("slq_elapsed_seconds") is None:
            selected[n_sensors]["slq_elapsed_seconds"] = elapsed
    return [selected[key] for key in sorted(selected)]


def metric_row(row: dict[str, Any]) -> dict[str, Any]:
    rbc = row.get("rbc_scaling") or {}
    return {
        "n_sensors": int(row["n_sensors"]),
        "requested_n_sources": int(row.get("requested_n_sources") or 128000),
        "realized_n_sources": int(row["n_sources"]),
        "n_outputs": int((row.get("matrix_size") or [0, 0])[0]),
        "bitrate_bits_per_s": float(row["bitrate"]),
        "noise_pa": float(row["noise_level"]),
        "bandwidth_hz": 1.0,
        "input_power_convention": row.get("input_power_convention"),
        "input_power_per_source": row.get("input_power_per_source"),
        "slq_s": _arg_value(row.get("analytical_args") or [], "--slq_s"),
        "slq_t": _arg_value(row.get("analytical_args") or [], "--slq_t"),
        "slq_batch": _arg_value(row.get("analytical_args") or [], "--slq_batch"),
        "slq_probe_parallel": bool(row.get("slq_probe_parallel")),
        "slq_elapsed_seconds": row.get("slq_elapsed_seconds"),
        "slq_frobenius_norm_sq": row.get("slq_frobenius_norm_sq"),
        "gram_side": row.get("gram_side"),
        "gram_shape": "x".join(map(str, row.get("gram_shape") or [])),
        "gram_size_bytes": row.get("modal_gram_size_bytes"),
        "gram_save_elapsed_seconds": row.get("gram_save_elapsed_seconds"),
        "modal_gram_output_path": row.get("modal_gram_output_path") or row.get("gram_output_path"),
        "sensor_in_cone_count": rbc.get("sensor_in_cone_count"),
        "sensor_in_cone_fraction": rbc.get("sensor_in_cone_fraction"),
        "source_in_cone_count": rbc.get("source_in_cone_count"),
        "source_in_cone_fraction": rbc.get("source_in_cone_fraction"),
        "reference_pressure_out_out_pa": rbc.get("reference_pressure_out_out_pa"),
        "reference_pressure_in_out_pa": rbc.get("reference_pressure_in_out_pa"),
        "reference_pressure_in_in_pa": rbc.get("reference_pressure_in_in_pa"),
        "json_path": row.get("_json_path"),
        "gram_note": row.get("gram_note"),
    }


def _arg_value(args: list[str], flag: str) -> str | None:
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def write_metrics(rows: list[dict[str, Any]], outdir: Path) -> list[dict[str, Any]]:
    metrics = [metric_row(row) for row in rows]
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")
    fieldnames = list(metrics[0].keys()) if metrics else []
    with (outdir / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def plot_bitrate(metrics: list[dict[str, Any]], outdir: Path) -> None:
    xs = [row["n_sensors"] for row in metrics]
    ys = [row["bitrate_bits_per_s"] for row in metrics]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel("number of sensors")
    ax.set_ylabel("bitrate (bit/s)")
    ax.set_title("US 128k sources: 2 MHz RBC cone transmission")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "bitrate_vs_sensors_rbc_cone.png", dpi=200)
    plt.close(fig)


def write_readme(metrics: list[dict[str, Any]], outdir: Path) -> None:
    if not metrics:
        return
    first = metrics[0]
    lines = [
        "# 128k-source ultrasound RBC cone sweep",
        "",
        "This run recomputed the three 50 kHz analytical ultrasound sensor counts with a 2 MHz RBC pressure-amplitude model and cone-dependent skull transmission.",
        "",
        "## Model",
        "",
        "- Requested sources: `128000`; realized source grid: `130399` points.",
        "- Bitrate estimator: streaming SLQ, `s=8`, `t=32`, equal unit input power per source.",
        "- Bitrate bandwidth convention: `1 Hz` via `bitrate_time_resolution=1.0 s`.",
        "- Output noise: `5 mPa = 0.005 Pa`, scalar iid.",
        "- Matrix normalization: disabled; the operator is scaled into output-pressure units.",
        "- Cone: half angle `15 deg` around the positive x-axis after subtracting the head center.",
        "- Skull pressure transmission: `0.5` inside the cone, `0.1` outside.",
        "",
        "The pair scaling replaces the raw free-field amplitude with:",
        "",
        "```text",
        "p_ij(t) = G_delay_ij(t) * P_external * sqrt(eta * V) / r_ij * T_source_i * T_sensor_j",
        "eta = CBV * BSC_10MHz * (2 MHz / 10 MHz)^4",
        "```",
        "",
        "Equivalently, because the raw free-field operator already has a `1/r_ij` factor, each streamed chunk is scaled by a common RBC pressure gain, source transmission, and sensor/output transmission before Gram accumulation and SLQ.",
        "",
        "Reference pair pressures at `r=0.10 m`:",
        "",
        f"- outside/outside: `{first['reference_pressure_out_out_pa']:.6g} Pa` (`{first['reference_pressure_out_out_pa'] * 1e3:.6g} mPa`)",
        f"- inside/outside: `{first['reference_pressure_in_out_pa']:.6g} Pa` (`{first['reference_pressure_in_out_pa'] * 1e3:.6g} mPa`)",
        f"- inside/inside: `{first['reference_pressure_in_in_pa']:.6g} Pa` (`{first['reference_pressure_in_in_pa'] * 1e3:.6g} mPa`)",
        "",
        "## Outputs",
        "",
        "- [metrics.csv](metrics.csv)",
        "- [metrics.json](metrics.json)",
        "- [bitrate_vs_sensors_rbc_cone.png](bitrate_vs_sensors_rbc_cone.png)",
        "",
        "## Results",
        "",
        "| sensors | realized sources | n_outputs | bitrate bit/s | Gram side | Gram shape | Gram size | Gram Modal path |",
        "|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            "| {n_sensors} | {realized_n_sources} | {n_outputs} | {bitrate:.6g} | {gram_side} | {gram_shape} | {size} | `{path}` |".format(
                n_sensors=row["n_sensors"],
                realized_n_sources=row["realized_n_sources"],
                n_outputs=row["n_outputs"],
                bitrate=row["bitrate_bits_per_s"],
                gram_side=row.get("gram_side") or "",
                gram_shape=row.get("gram_shape") or "",
                size=row.get("gram_size_bytes") or "",
                path=row.get("modal_gram_output_path") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "| sensors | Gram save seconds | SLQ seconds | note |",
            "|---:|---:|---:|---|",
        ]
    )
    for row in metrics:
        lines.append(
            "| {n_sensors} | {gram_elapsed} | {slq_elapsed} | {note} |".format(
                n_sensors=row["n_sensors"],
                gram_elapsed=_fmt_optional(row.get("gram_save_elapsed_seconds")),
                slq_elapsed=_fmt_optional(row.get("slq_elapsed_seconds")),
                note=row.get("gram_note") or "",
            )
        )
    lines.append("")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt_optional(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.results_dir / "analysis"
    records = load_json_records(args.results_dir)
    rows = selected_rows(records)
    metrics = write_metrics(rows, outdir)
    plot_bitrate(metrics, outdir)
    write_readme(metrics, outdir)
    print(f"selected rows: {len(metrics)}")
    for row in metrics:
        print(
            f"{row['n_sensors']} sensors: bitrate={row['bitrate_bits_per_s']:.6g}, "
            f"gram={row.get('modal_gram_output_path')}"
        )
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
