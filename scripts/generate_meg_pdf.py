"""
Generate a multi-page PDF with all MEG scaling plots.

Pages:
  For each modality in [meg_opm, meg_squid]:
    Sweep num_sensors (constant source_spacing_mm=5.0):
      1. SVD spectra – unnormalized (raw singular values)
      2. SVD spectra – normalized by the largest singular value across all variants
      3. First singular value vs num_sensors
      4. Bitrate vs num_sensors
    Sweep source_spacing_mm (constant num_sensors=500):
      5. SVD spectra – unnormalized
      6. SVD spectra – normalized
      7. First singular value vs source_spacing_mm
      8. Bitrate vs source_spacing_mm

Output: plots/meg_scaling.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import viridis

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.capacity import get_bitrate, total_input_power_from_average_output_power
from guti.core import get_grid_positions
from guti.noise_models import (
    compute_average_output_power,
    compute_output_noise_std,
    get_noise_model,
)

os.makedirs("plots", exist_ok=True)

MODALITIES = ["meg_opm", "meg_squid"]
MODALITY_LABELS = {"meg_opm": "MEG OPM", "meg_squid": "MEG SQUID"}

# ── helpers ────────────────────────────────────────────────────────────────────

def get_variants(modality, sweep_key, constant_params):
    return list_svd_variants(modality, constant_params=constant_params, sort_by=sweep_key)


def compute_bitrate(s, params, modality, freq=None, time_resolution=0.01):
    """Bitrate from per-output average signal power and per-output noise."""
    n_sensors = params.num_sensors
    n_outputs = 3 * n_sensors
    n_sources = 3 * len(get_grid_positions(grid_spacing_mm=params.source_spacing_mm))
    noise = compute_output_noise_std(
        modality,
        n_sensors=n_sensors,
        frequency_hz=freq,
    )
    average_output_power = compute_average_output_power(modality)
    total_input_power = total_input_power_from_average_output_power(
        s,
        average_output_power=average_output_power,
        n_sources=n_sources,
        n_outputs=n_outputs,
    )
    return float(
        get_bitrate(
            s,
            n_sources=n_sources,
            total_input_power=total_input_power,
            noise=noise,
            time_resolution=time_resolution,
        )
    )


def colormap(n):
    return [viridis(i) for i in np.linspace(0, 1, max(n, 1))]


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_spectra(ax, variants, sweep_key, normalized, title):
    if not variants:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    colors = colormap(len(variants))
    # global max for normalization
    global_max = max(v["s"][0] for _, v in variants)

    for (_, v), color in zip(variants, colors):
        s = v["s"]
        label_val = getattr(v["params"], sweep_key)
        y = s / global_max if normalized else s
        ax.plot(np.arange(1, len(y) + 1), y, color=color,
                label=f"{sweep_key}={label_val}", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value" + (" (normalized)" if normalized else ""))
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_first_sv(ax, variants, sweep_key, title):
    if not variants:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    xs = [getattr(v["params"], sweep_key) for _, v in variants]
    ys = [v["s"][0] for _, v in variants]
    ax.plot(xs, ys, "o-", linewidth=2, markersize=6, color="steelblue")
    ax.set_xlabel(sweep_key)
    ax.set_ylabel("First singular value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_bitrate(ax, variants, sweep_key, modality, title, time_resolution=0.01):
    if not variants:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    xs, ys = [], []
    model = get_noise_model(modality)
    for _, v in variants:
        freq = getattr(v["params"], "frequency_hz", None)
        xs.append(getattr(v["params"], sweep_key))
        ys.append(
            compute_bitrate(
                v["s"],
                v["params"],
                modality,
                freq=freq,
                time_resolution=time_resolution,
            )
        )

    ax.plot(xs, ys, "o-", linewidth=2, markersize=6, color="darkorange",
            label="Output-power workflow")
    ax.set_xlabel(sweep_key)
    ax.set_ylabel("Bitrate (bits/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # annotate empirical SNR
    ax.text(0.02, 0.97, f"Output SNR today ~= {model.typical_signal_amplitude / model.today_best_noise:.2f}",
            transform=ax.transAxes, va="top", fontsize=7, color="darkorange")


# ── main ───────────────────────────────────────────────────────────────────────

def build_pdf():
    pdf_path = "plots/meg_scaling.pdf"
    with PdfPages(pdf_path) as pdf:
        for modality in MODALITIES:
            label = MODALITY_LABELS[modality]

            # ── Sweep num_sensors (source_spacing=5mm) ───────────────────────
            sweep_key = "num_sensors"
            variants = get_variants(modality, sweep_key, Parameters(source_spacing_mm=5.0))
            print(f"{modality} num_sensors sweep: {len(variants)} variants")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"{label} — sweep: {sweep_key} (source_spacing_mm=5.0)", fontsize=13)

            plot_spectra(axes[0, 0], variants, sweep_key, normalized=False,
                         title="SVD spectra (unnormalized)")
            plot_spectra(axes[0, 1], variants, sweep_key, normalized=True,
                         title="SVD spectra (normalized by max)")
            plot_first_sv(axes[1, 0], variants, sweep_key,
                          title="First singular value vs num_sensors")
            plot_bitrate(axes[1, 1], variants, sweep_key, modality,
                         title="Bitrate vs num_sensors")

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # ── Sweep source_spacing_mm (num_sensors=500) ────────────────────
            sweep_key = "source_spacing_mm"
            variants = get_variants(modality, sweep_key, Parameters(num_sensors=500))
            print(f"{modality} source_spacing_mm sweep: {len(variants)} variants")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"{label} — sweep: {sweep_key} (num_sensors=500)", fontsize=13)

            plot_spectra(axes[0, 0], variants, sweep_key, normalized=False,
                         title="SVD spectra (unnormalized)")
            plot_spectra(axes[0, 1], variants, sweep_key, normalized=True,
                         title="SVD spectra (normalized by max)")
            plot_first_sv(axes[1, 0], variants, sweep_key,
                          title="First singular value vs source_spacing_mm")
            plot_bitrate(axes[1, 1], variants, sweep_key, modality,
                         title="Bitrate vs source_spacing_mm")

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ── metadata
        d = pdf.infodict()
        d["Title"] = "MEG Scaling Plots"
        d["Subject"] = "MEG OPM and SQUID SVD scaling vs num_sensors and source_spacing_mm"

    print(f"\nPDF saved to {pdf_path}")


if __name__ == "__main__":
    build_pdf()
