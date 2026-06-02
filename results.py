"""Generate GUTI result summary figures from saved SVD spectra."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from labellines import labelLines
except ImportError:  # pragma: no cover - optional plotting nicety
    labelLines = None

from guti.capacity import get_capacity_from_average_output_power
from guti.core import get_grid_positions
from guti.noise_models import compute_average_output_power, compute_output_noise_std
from guti.parameters import Parameters


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

RESULT_SOURCES = {
    "eeg": ("eeg",),
    "cw_fnirs": ("cw_fnirs",),
    "td_fnirs": ("td_fnirs",),
    "us": ("us",),
    "meg_opm": ("meg_opm",),
    "meg_squid": ("meg_squid",),
}

MODALITY_LABELS = {
    "eeg": "EEG",
    "cw_fnirs": "fNIRS (CW)",
    "td_fnirs": "fNIRS (TD)",
    "us": "Ultrasound (40 kHz)",
    "meg_opm": "MEG (OPM)",
    "meg_squid": "MEG (SQUID)",
}

TIME_RESOLUTION_PER_MODALITY = {
    "eeg": 10e-3,
    "cw_fnirs": 1.0,
    "td_fnirs": 1.0,
    "meg_opm": 10e-3,
    "meg_squid": 10e-3,
    "us": 1e-3,
}

SVDResults = Mapping[str, Tuple[np.ndarray, Optional[Parameters]]]


def load_svd_result(source_name: str) -> Tuple[np.ndarray, Optional[Parameters]]:
    filepath = RESULTS_DIR / f"{source_name}_svd_spectrum.npz"
    data = np.load(filepath, allow_pickle=True)
    params = None
    if "parameters" in data:
        params_dict = data["parameters"].item()
        if params_dict is not None:
            params = Parameters.from_dict(params_dict)
    return data["singular_values"], params


def load_latest_svds() -> Dict[str, Tuple[np.ndarray, Optional[Parameters]]]:
    results = {}
    missing = []
    for modality_name, source_names in RESULT_SOURCES.items():
        for source_name in source_names:
            if (RESULTS_DIR / f"{source_name}_svd_spectrum.npz").exists():
                results[modality_name] = load_svd_result(source_name)
                break
        else:
            missing.append(modality_name)

    if missing:
        raise FileNotFoundError(
            "Missing SVD result files for current modalities: " + ", ".join(missing)
        )
    return results


def infer_shape(
    modality_name: str,
    params: Optional[Parameters],
    singular_values: np.ndarray,
) -> Tuple[int, int]:
    """Infer the forward model shape stored behind a singular-value spectrum."""
    if params is not None and params.matrix_size is not None:
        return params.matrix_size
    if params is not None and modality_name.startswith("meg_"):
        return (
            3 * params.num_sensors,
            3 * len(get_grid_positions(params.source_spacing_mm)),
        )
    if params is not None and modality_name == "eeg":
        return params.num_sensors, 3 * params.num_brain_grid_points
    if (
        params is not None
        and params.num_sensors is not None
        and params.num_brain_grid_points is not None
    ):
        return params.num_sensors, params.num_brain_grid_points
    return len(singular_values), len(singular_values)


def plot_spectrum(all_svds: SVDResults) -> None:
    fig, ax = plt.subplots()
    for modality_name, (singular_values, _params) in all_svds.items():
        ax.plot(
            np.arange(1, len(singular_values) + 1),
            singular_values / singular_values[0],
            label=MODALITY_LABELS.get(modality_name, modality_name),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Spectrum of Imaging Modalities")
    ax.set_ylim(1e-4, 2)

    if labelLines is not None:
        labelLines(ax.get_lines(), zorder=2.5)
    else:
        ax.legend()

    fig.savefig(RESULTS_DIR / "spectrum.png")
    plt.close(fig)


def compute_channel_capacities(all_svds: SVDResults) -> Dict[str, float]:
    channel_capacities = {}
    for modality_name, (singular_values, params) in all_svds.items():
        n_sensors = None if params is None else params.num_sensors
        n_outputs, n_sources = infer_shape(modality_name, params, singular_values)
        try:
            time_resolution = TIME_RESOLUTION_PER_MODALITY[modality_name]
        except KeyError as exc:
            raise KeyError(
                f"Missing time resolution for modality {modality_name!r}"
            ) from exc

        channel_capacity = get_capacity_from_average_output_power(
            singular_values.astype(np.float64),
            average_output_power=compute_average_output_power(modality_name),
            noise=compute_output_noise_std(modality_name, n_sensors=n_sensors),
            n_sources=n_sources,
            n_outputs=n_outputs,
            time_resolution=time_resolution,
        )
        channel_capacities[modality_name] = channel_capacity
        print(f"{modality_name}: {channel_capacity:.2f} bits/s")

    return channel_capacities


def save_table(
    rows: List[List[str]],
    col_labels: List[str],
    output_path: Path,
    *,
    figsize: Tuple[float, float],
    pad_inches: float,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.4],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for i in range(2):
        table[(0, i)].set_facecolor("#404040")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(rows) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_bitrate_table(channel_capacities: Mapping[str, float]) -> None:
    table_labels = {**MODALITY_LABELS, "us": "Ultrasound (2 MHz)"}
    rows = []
    for modality_name in sorted(
        channel_capacities,
        key=lambda name: channel_capacities[name],
        reverse=True,
    ):
        label = table_labels.get(modality_name, modality_name)
        rows.append([label, f"{channel_capacities[modality_name]:,.0f}"])

    save_table(
        rows,
        ["Modality", "Channel Capacity (bits/s)"],
        RESULTS_DIR / "bitrate.png",
        figsize=(8, 3),
        pad_inches=0,
    )


def plot_spatial_channel_capacity_table(
    channel_capacities: Mapping[str, float],
) -> None:
    table_labels = {**MODALITY_LABELS, "us": "Ultrasound (2 MHz)"}
    channel_capacities_per_sample = {}
    for modality_name, bits_per_second in channel_capacities.items():
        bits_per_sample = bits_per_second * TIME_RESOLUTION_PER_MODALITY[modality_name]
        channel_capacities_per_sample[modality_name] = bits_per_sample
        print(f"{modality_name}: {bits_per_sample:.0f} bits/sample")

    rows = []
    for modality_name in sorted(
        channel_capacities_per_sample,
        key=lambda name: channel_capacities_per_sample[name],
        reverse=True,
    ):
        label = table_labels.get(modality_name, modality_name)
        bits_per_sample = channel_capacities_per_sample[modality_name]
        rows.append([label, f"{bits_per_sample:,.0f}"])

    save_table(
        rows,
        ["Modality", "Channel Capacity (bits/sample)"],
        RESULTS_DIR / "spatial_channel_capacity.png",
        figsize=(8, 3.5),
        pad_inches=0.1,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_svds = load_latest_svds()
    plot_spectrum(all_svds)
    channel_capacities = compute_channel_capacities(all_svds)
    plot_bitrate_table(channel_capacities)
    plot_spatial_channel_capacity_table(channel_capacities)


if __name__ == "__main__":
    main()
