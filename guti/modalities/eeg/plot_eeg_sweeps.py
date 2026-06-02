# %%
"""Plot SV spectra for the three EEG-OpenMEEG radial-line sweeps.

Reads saved variants in results/variants/eeg_openmeeg/ that match the three
(n_radial_lines / num_sensors / grid_resolution_mm) sweep configurations and
saves one figure per sweep under guti/modalities/eeg/results/.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the top-level `guti` package importable when running this file directly.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parents[2]))

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.scaling_utils import normalize_singular_values

OUT_DIR = THIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared defaults that pin down each sweep's "other two" axes.
N_DIPOLES_PER_LINE = 5

SWEEPS = [
    dict(
        name="n_radial_lines",
        param_key="n_radial_lines",
        constant_params=Parameters(
            num_sensors=2048,
            grid_resolution_mm=8.0,
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
        ),
        title="EEG (OpenMEEG) — n_radial_lines sweep",
        fname="svd_spectrum_n_radial_lines.png",
    ),
    dict(
        name="num_sensors",
        param_key="num_sensors",
        constant_params=Parameters(
            grid_resolution_mm=8.0,
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
            n_radial_lines=409,
        ),
        title="EEG (OpenMEEG) — num_sensors sweep",
        fname="svd_spectrum_num_sensors.png",
    ),
    dict(
        name="grid_resolution_mm",
        param_key="grid_resolution_mm",
        constant_params=Parameters(
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
            n_radial_lines=409,
            num_sensors=2048,
        ),
        title="EEG (OpenMEEG) — grid_resolution_mm sweep",
        fname="svd_spectrum_grid_resolution_mm.png",
    ),
]


def plot_sweep(name, param_key, constant_params, title, fname,
               normalization_method="sqrtN", ylim=(1e-5, 2)):
    variants = list_svd_variants(
        "eeg_openmeeg", constant_params=constant_params, sort_by=param_key
    )
    if not variants:
        print(f"[skip] no variants for sweep '{name}' with {constant_params}")
        return

    # Keep the last variant written per swept value.
    by_value = {}
    for _, v in variants:
        by_value[getattr(v["params"], param_key)] = v
    items = sorted(by_value.items())

    norm_svs = [
        normalize_singular_values(v["s"], v["params"], method=normalization_method)
        for _, v in items
    ]
    s_ref = max(s[0] for s in norm_svs)

    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(items))]
    fig, ax = plt.subplots(figsize=(9, 6))
    for (pv, v), s_norm, c in zip(items, norm_svs, colors):
        s = s_norm / s_ref
        ax.plot(np.arange(1, len(s) + 1), s, color=c,
                label=f"{param_key}={pv}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value (normalized)")
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    out = OUT_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(items)} traces)")


# %%
if __name__ == "__main__":
    for s in SWEEPS:
        plot_sweep(**s)
