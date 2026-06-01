#!/usr/bin/env python3
"""Plot bitrate/capacity convergence from saved modality SVD sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
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
from guti.core import get_grid_positions
from guti.noise_models import (
    compute_average_output_power,
    compute_output_noise_std,
    get_noise_model,
    scale_singular_values_for_capacity,
)
from guti.parameters import Parameters


@dataclass(frozen=True)
class SweepSpec:
    name: str
    label: str
    noise_model: str
    variant_dirs: tuple[str, ...]
    source_orientations: int = 1


SWEEP_SPECS = (
    SweepSpec(
        name="eeg_openmeeg",
        label="EEG OpenMEEG",
        noise_model="eeg_openmeeg",
        variant_dirs=(
            "results/variants/eeg_openmeeg_clean_sweep_20260601_margin5mm",
            "results/variants/eeg_openmeeg_correlated_noise_20260601_margin5mm",
        ),
        source_orientations=3,
    ),
    SweepSpec(
        name="meg_opm",
        label="MEG OPM",
        noise_model="meg_opm",
        variant_dirs=("results/variants/meg_opm",),
        source_orientations=3,
    ),
    SweepSpec(
        name="meg_squid",
        label="MEG SQUID",
        noise_model="meg_squid",
        variant_dirs=("results/variants/meg_squid",),
        source_orientations=3,
    ),
    SweepSpec(
        name="fnirs_analytical_cw",
        label="fNIRS CW",
        noise_model="fnirs_analytical_cw",
        variant_dirs=("results/variants/fnirs_analytical_cw",),
    ),
    SweepSpec(
        name="us_analytical_50khz",
        label="US 50 kHz",
        noise_model="us_analytical",
        variant_dirs=(
            "results/variants/us_free_field_analytical_50khz_convergence_20260531",
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make convergence plots from saved SVD sweep variants."
    )
    parser.add_argument(
        "--outdir",
        default="results/convergence_plots",
        help="Output directory for metrics and plots.",
    )
    parser.add_argument(
        "--modalities",
        default="all",
        help=(
            "Comma-separated sweep names to include, or 'all'. "
            f"Known: {', '.join(spec.name for spec in SWEEP_SPECS)}"
        ),
    )
    parser.add_argument(
        "--noise-model-types",
        default="all",
        help=(
            "Comma-separated metric noise types to include, or 'all'. "
            "Known: scalar_iid, spatial_covariance."
        ),
    )
    return parser


def _optional_np_scalar(data: np.lib.npyio.NpzFile, name: str) -> Any:
    if name not in data.files:
        return None
    value = data[name]
    try:
        value = value.item()
    except ValueError:
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_variant(
    path: Path,
) -> tuple[np.ndarray, Parameters, np.ndarray | None, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "singular_values" not in data.files:
        raise ValueError("missing singular_values")
    params = Parameters()
    if "parameters" in data.files:
        try:
            raw_params = data["parameters"].item()
        except Exception:
            raw_params = None
        if raw_params:
            params = Parameters.from_dict(raw_params)
    noise_normalized = None
    if "noise_normalized_singular_values" in data.files:
        noise_normalized = np.asarray(
            data["noise_normalized_singular_values"],
            dtype=np.float64,
        )
    metadata = {
        "spectrum_estimate_method": _optional_np_scalar(
            data,
            "spectrum_estimate_method",
        ),
        "spectrum_reference_num_sensors": _optional_np_scalar(
            data,
            "spectrum_reference_num_sensors",
        ),
        "spectrum_reference_path": _optional_np_scalar(
            data,
            "spectrum_reference_path",
        ),
        "spectrum_reference_hash": _optional_np_scalar(
            data,
            "spectrum_reference_hash",
        ),
        "spectrum_reference_singular_value_count": _optional_np_scalar(
            data,
            "spectrum_reference_singular_value_count",
        ),
        "spectrum_scale_factor": _optional_np_scalar(
            data,
            "spectrum_scale_factor",
        ),
    }
    return (
        np.asarray(data["singular_values"], dtype=np.float64),
        params,
        noise_normalized,
        metadata,
    )


def infer_shape(
    spec: SweepSpec,
    params: Parameters,
) -> tuple[int, int, int]:
    """Return (n_outputs, n_sources, n_voxels)."""
    if params.matrix_size is not None:
        n_outputs, n_sources = params.matrix_size
        n_outputs = int(n_outputs)
        n_sources = int(n_sources)
        n_voxels = int(params.num_brain_grid_points or n_sources)
        return n_outputs, n_sources, n_voxels

    if spec.name.startswith("meg_"):
        if params.num_sensors is None or params.source_spacing_mm is None:
            raise ValueError("MEG sweep variant needs num_sensors and source_spacing_mm")
        n_voxels = len(get_grid_positions(grid_spacing_mm=float(params.source_spacing_mm)))
        return 3 * int(params.num_sensors), 3 * n_voxels, n_voxels

    if spec.name.startswith("eeg_"):
        if params.num_sensors is None or params.num_brain_grid_points is None:
            raise ValueError("EEG sweep variant needs num_sensors and num_brain_grid_points")
        n_voxels = int(params.num_brain_grid_points)
        return int(params.num_sensors), spec.source_orientations * n_voxels, n_voxels

    if params.num_sensors is None or params.num_brain_grid_points is None:
        raise ValueError("variant needs num_sensors and num_brain_grid_points")
    n_voxels = int(params.num_brain_grid_points)
    return int(params.num_sensors), n_voxels, n_voxels


def modality_bandwidth_hz(noise_model: str, params: Parameters) -> float:
    model = get_noise_model(noise_model)
    if model.canonical_name == "us_analytical":
        freq = float(params.frequency_hz or 50_000.0)
        return float(model.reference_bandwidth_hz * freq / 50_000.0)
    return float(model.reference_bandwidth_hz)


def variant_record(spec: SweepSpec, path: Path) -> dict[str, Any]:
    s, params, s_noise_normalized, spectrum_metadata = load_variant(path)
    n_outputs, n_sources, n_voxels = infer_shape(spec, params)
    n_sensors = int(params.num_sensors) if params.num_sensors is not None else None
    if n_sensors is None:
        raise ValueError("variant needs num_sensors")

    s_capacity = scale_singular_values_for_capacity(
        s,
        spec.noise_model,
        params=params,
        voxel_size_mm=params.grid_resolution_mm,
    )
    bandwidth_hz = modality_bandwidth_hz(spec.noise_model, params)
    time_resolution_s = 1.0 / bandwidth_hz
    average_output_power = compute_average_output_power(spec.noise_model)
    output_amplitude = math.sqrt(average_output_power)
    output_noise = compute_output_noise_std(
        spec.noise_model,
        n_sensors=n_sensors,
        bandwidth_hz=bandwidth_hz,
        frequency_hz=params.frequency_hz,
        voxel_size_mm=params.grid_resolution_mm,
        tr_s=params.time_resolution,
        bold_contrast=params.bold_contrast,
        bold_snr=params.bold_snr,
    )
    noise_model_type = "scalar_iid"
    if s_noise_normalized is not None:
        noise_model_type = "spatial_covariance"
        total_input_power = total_input_power_from_average_output_power(
            s_capacity,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
        )
        bitrate = get_bitrate(
            s_noise_normalized,
            n_sources=n_sources,
            total_input_power=total_input_power,
            noise=1.0,
            time_resolution=time_resolution_s,
        )
        capacity = get_capacity(
            s_noise_normalized[s_noise_normalized > 0],
            total_input_power=total_input_power,
            noise=1.0,
            time_resolution=time_resolution_s,
        )
    else:
        total_input_power = total_input_power_from_average_output_power(
            s_capacity,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
        )
        bitrate = get_bitrate(
            s_capacity,
            n_sources=n_sources,
            total_input_power=total_input_power,
            noise=output_noise,
            time_resolution=time_resolution_s,
        )
        capacity = get_capacity(
            s_capacity[s_capacity > 0],
            total_input_power=total_input_power,
            noise=output_noise,
            time_resolution=time_resolution_s,
        )
    params_dict = asdict(params)
    return {
        "modality": spec.name,
        "label": spec.label,
        "noise_model": spec.noise_model,
        "path": str(path),
        "mtime": os.path.getmtime(path),
        "n_sensors": n_sensors,
        "n_voxels": n_voxels,
        "n_outputs": n_outputs,
        "n_sources": n_sources,
        "n_singular_values": int(len(s_capacity)),
        "bandwidth_hz": bandwidth_hz,
        "time_resolution_s": time_resolution_s,
        "output_amplitude": output_amplitude,
        "output_noise": float(output_noise),
        "output_snr": float(output_amplitude / output_noise),
        "noise_model_type": noise_model_type,
        "noise_correlation_length_mm": params.noise_correlation_length_mm,
        "noise_correlation_kernel": params.noise_correlation_kernel,
        "total_input_power": float(total_input_power),
        "bitrate_bits_per_s": float(bitrate),
        "channel_capacity_bits_per_s": float(capacity),
        "frequency_hz": params.frequency_hz,
        "source_spacing_mm": params.source_spacing_mm,
        "grid_resolution_mm": params.grid_resolution_mm,
        "matrix_size": params.matrix_size,
        "params": {key: value for key, value in params_dict.items() if value is not None},
        **spectrum_metadata,
    }


def _noise_length_key(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _noise_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    if record["noise_model_type"] != "spatial_covariance":
        return (record["noise_model_type"], None, None)
    return (
        record["noise_model_type"],
        str(record.get("noise_correlation_kernel") or "unknown").lower(),
        _noise_length_key(record.get("noise_correlation_length_mm")),
    )


def load_records(spec: SweepSpec) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for variant_dir in spec.variant_dirs:
        for path in sorted(Path(variant_dir).glob("*.npz")):
            try:
                records.append(variant_record(spec, path))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})

    # Deduplicate repeated runs with the same plotted coordinates, preserving
    # exact spectra over extrapolated estimates and then the newest usable file.
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    def record_rank(record: dict[str, Any]) -> tuple[bool, float, str]:
        exact_spectrum = record.get("spectrum_estimate_method") in (None, "")
        return exact_spectrum, record["mtime"], record["path"]

    for record in records:
        key = (
            record["modality"],
            record["n_voxels"],
            record["n_sensors"],
            record.get("frequency_hz"),
            *_noise_record_key(record),
        )
        existing = latest.get(key)
        if existing is None or record_rank(record) > record_rank(existing):
            latest[key] = record
    return sorted(
        latest.values(),
        key=lambda row: (row["modality"], row["n_sensors"], row["n_voxels"]),
    ), errors


def write_metrics(records: list[dict[str, Any]], errors: list[dict[str, str]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    json_records = [{k: v for k, v in row.items() if k != "mtime"} for row in records]
    (outdir / "metrics.json").write_text(json.dumps(json_records, indent=2), encoding="utf-8")
    (outdir / "load_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    fields = [
        "modality",
        "label",
        "n_voxels",
        "n_sensors",
        "n_outputs",
        "n_sources",
        "n_singular_values",
        "bandwidth_hz",
        "output_amplitude",
        "output_noise",
        "output_snr",
        "noise_model_type",
        "noise_correlation_length_mm",
        "noise_correlation_kernel",
        "total_input_power",
        "bitrate_bits_per_s",
        "channel_capacity_bits_per_s",
        "frequency_hz",
        "source_spacing_mm",
        "grid_resolution_mm",
        "spectrum_estimate_method",
        "spectrum_reference_num_sensors",
        "spectrum_reference_path",
        "spectrum_reference_hash",
        "spectrum_reference_singular_value_count",
        "spectrum_scale_factor",
        "path",
    ]
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def plot_grouped_lines(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    group_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if len({row[x_key] for row in rows}) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for group_value in sorted({row[group_key] for row in rows}):
        group_rows = [row for row in rows if row[group_key] == group_value]
        if len(group_rows) < 2:
            continue
        group_rows.sort(key=lambda row: row[x_key])
        xs = np.asarray([row[x_key] for row in group_rows], dtype=float)
        ys = np.asarray([row[y_key] for row in group_rows], dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{group_value:g}")

    if not ax.lines:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(title=group_key, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_single_line(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    if len({row[x_key] for row in rows}) < 2:
        return False

    rows = sorted(rows, key=lambda row: row[x_key])
    xs = np.asarray([row[x_key] for row in rows], dtype=float)
    ys = np.asarray([row[y_key] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(xs, ys, marker="o", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_filename(filename: str, plot_tag: str | None) -> str:
    if plot_tag is None:
        return filename
    stem, suffix = filename.rsplit(".", 1)
    return f"{stem}_{plot_tag}.{suffix}"


def _plot_title(label: str, plot_tag: str | None, ylabel: str, xlabel: str) -> str:
    title_label = label if plot_tag is None else f"{label} {plot_tag}"
    return f"{title_label}: {ylabel} vs {xlabel}"


def _format_noise_length_tag(value: Any) -> str | None:
    if value is None:
        return None
    try:
        length_text = f"{float(value):.6g}"
    except (TypeError, ValueError):
        length_text = str(value)
    return length_text.replace("-", "m").replace(".", "p")


def noise_plot_tag(row: dict[str, Any]) -> str | None:
    if row["noise_model_type"] != "spatial_covariance":
        return None

    kernel = str(row.get("noise_correlation_kernel") or "unknown").lower()
    length_mm = row.get("noise_correlation_length_mm")
    length_key = _noise_length_key(length_mm)
    if kernel == "gaussian" and length_key == 5.0:
        return "correlated_noise"

    length_tag = _format_noise_length_tag(length_mm)
    if length_tag is None:
        return f"correlated_noise_{kernel}"
    return f"correlated_noise_{kernel}_L{length_tag}mm"


def plot_modality(
    rows: list[dict[str, Any]],
    outdir: Path,
    *,
    plot_tag: str | None = None,
) -> list[str]:
    modality = rows[0]["modality"]
    label = rows[0]["label"]
    modality_dir = outdir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    voxel_plot_specs = [
        (
            "n_voxels",
            "n_sensors",
            "bitrate_bits_per_s",
            "voxel/source points",
            "bitrate (bit/s)",
            "bitrate_vs_n_voxels.png",
        ),
        (
            "n_voxels",
            "n_sensors",
            "channel_capacity_bits_per_s",
            "voxel/source points",
            "channel capacity (bit/s)",
            "capacity_vs_n_voxels.png",
        ),
    ]
    for x_key, group_key, y_key, xlabel, ylabel, filename in voxel_plot_specs:
        path = modality_dir / _plot_filename(filename, plot_tag)
        ok = plot_grouped_lines(
            rows,
            x_key=x_key,
            group_key=group_key,
            y_key=y_key,
            xlabel=xlabel,
            ylabel=ylabel,
            title=_plot_title(label, plot_tag, ylabel, xlabel),
            output_path=path,
        )
        if ok:
            written.append(str(path))

    max_n_voxels = max(row["n_voxels"] for row in rows)
    highest_voxel_rows = [row for row in rows if row["n_voxels"] == max_n_voxels]
    for y_key, ylabel, filename in (
        ("bitrate_bits_per_s", "bitrate (bit/s)", "bitrate_vs_n_sensors.png"),
        (
            "channel_capacity_bits_per_s",
            "channel capacity (bit/s)",
            "capacity_vs_n_sensors.png",
        ),
    ):
        path = modality_dir / _plot_filename(filename, plot_tag)
        title_label = label if plot_tag is None else f"{label} {plot_tag}"
        ok = plot_single_line(
            highest_voxel_rows,
            x_key="n_sensors",
            y_key=y_key,
            xlabel="sensors",
            ylabel=ylabel,
            title=(
                f"{title_label}: {ylabel} vs sensors "
                f"at n_voxels={max_n_voxels:g}"
            ),
            output_path=path,
        )
        if ok:
            written.append(str(path))
    return written


def write_readme(
    records: list[dict[str, Any]],
    errors: list[dict[str, str]],
    plot_paths: list[str],
    outdir: Path,
) -> None:
    lines = [
        "# Modality SVD Convergence Plots",
        "",
        "Metrics are recomputed from saved singular-value spectra using the average-output-power workflow.",
        "When a sweep file contains `noise_normalized_singular_values`, metrics use the spatially correlated noise model saved with that file.",
        "`time_resolution_s` is set to `1 / bandwidth_hz` for each modality.",
        "",
        "## Coverage",
        "",
        "| modality | rows | voxel counts | sensor counts |",
        "| --- | ---: | --- | --- |",
    ]
    for modality in sorted({row["modality"] for row in records}):
        rows = [row for row in records if row["modality"] == modality]
        voxels = ", ".join(str(v) for v in sorted({row["n_voxels"] for row in rows}))
        sensors = ", ".join(str(v) for v in sorted({row["n_sensors"] for row in rows}))
        lines.append(f"| {modality} | {len(rows)} | {voxels} | {sensors} |")

    estimated_rows = [
        row for row in records if row.get("spectrum_estimate_method") not in (None, "")
    ]
    if estimated_rows:
        lines.extend(
            [
                "",
                (
                    f"{len(estimated_rows)} rows use explicit spectrum estimates; "
                    "see `spectrum_estimate_method` and reference fields in the metrics."
                ),
            ]
        )

    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        rel = Path(path).relative_to(outdir)
        lines.append(f"- [{rel.as_posix()}]({rel.as_posix()})")

    lines.extend(
        [
            "",
            "## Data",
            "",
            "- [metrics.csv](metrics.csv)",
            "- [metrics.json](metrics.json)",
            "- [load_errors.json](load_errors.json)",
        ]
    )
    if errors:
        lines.extend(["", f"Skipped {len(errors)} files with load/metadata errors."])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_specs(names: str) -> list[SweepSpec]:
    if names == "all":
        return list(SWEEP_SPECS)
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    specs_by_name = {spec.name: spec for spec in SWEEP_SPECS}
    unknown = sorted(wanted - set(specs_by_name))
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")
    return [specs_by_name[name] for name in wanted]


def select_noise_model_types(names: str) -> set[str] | None:
    if names == "all":
        return None
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    known = {"scalar_iid", "spatial_covariance"}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown noise model types: {unknown}")
    return wanted


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    selected_noise_model_types = select_noise_model_types(args.noise_model_types)

    all_records: list[dict[str, Any]] = []
    all_errors: list[dict[str, str]] = []
    for spec in select_specs(args.modalities):
        records, errors = load_records(spec)
        all_records.extend(records)
        all_errors.extend(errors)

    if not all_records:
        raise SystemExit("No usable sweep records found")
    if selected_noise_model_types is not None:
        all_records = [
            record
            for record in all_records
            if record["noise_model_type"] in selected_noise_model_types
        ]
    if not all_records:
        raise SystemExit("No sweep records matched the selected noise model types")

    write_metrics(all_records, all_errors, outdir)

    plot_paths: list[str] = []
    for modality in sorted({record["modality"] for record in all_records}):
        rows = [record for record in all_records if record["modality"] == modality]
        for plot_group in sorted(
            {(row["noise_model_type"], noise_plot_tag(row) or "") for row in rows}
        ):
            type_rows = [
                row
                for row in rows
                if (row["noise_model_type"], noise_plot_tag(row) or "") == plot_group
            ]
            plot_tag = noise_plot_tag(type_rows[0])
            plot_paths.extend(
                plot_modality(type_rows, outdir, plot_tag=plot_tag)
            )

    write_readme(all_records, all_errors, plot_paths, outdir)
    print(f"Wrote {len(all_records)} convergence records to {outdir}")
    print(f"Wrote {len(plot_paths)} plots")
    if all_errors:
        print(f"Skipped {len(all_errors)} files; see {outdir / 'load_errors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
