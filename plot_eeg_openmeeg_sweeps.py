# %%
"""Plot SV spectra for the three EEG-OpenMEEG sweeps (sensors, sources, grid)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
from guti.scaling_utils import normalize_singular_values

OUT_DIR = "plots/eeg_openmeeg"
os.makedirs(OUT_DIR, exist_ok=True)

SWEEPS = [
    dict(
        param_key="num_sensors",
        constant_params=Parameters(source_spacing_mm=5.0, grid_resolution_mm=20.0),
        title="EEG (OpenMEEG) — sensor sweep",
        fname="svd_spectrum_sensors.png",
    ),
    dict(
        param_key="source_spacing_mm",
        constant_params=Parameters(num_sensors=256, grid_resolution_mm=20.0),
        title="EEG (OpenMEEG) — source-spacing sweep",
        fname="svd_spectrum_source_spacing.png",
    ),
    dict(
        param_key="grid_resolution_mm",
        constant_params=Parameters(num_sensors=256, source_spacing_mm=5.0),
        title="EEG (OpenMEEG) — mesh (grid) resolution sweep",
        fname="svd_spectrum_grid_resolution.png",
    ),
]


def plot_sweep(param_key, constant_params, title, fname,
               normalization_method="sqrtN", ylim=(1e-5, 2)):
    variants = list_svd_variants("eeg_openmeeg",
                                 constant_params=constant_params,
                                 sort_by=param_key)
    if not variants:
        print(f"[skip] no variants for {param_key}")
        return

    # Dedup: pick last variant per param value (handles duplicates across runs)
    by_value = {}
    for h, v in variants:
        by_value[getattr(v["params"], param_key)] = v
    items = sorted(by_value.items())

    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(items))]
    fig, ax = plt.subplots(figsize=(9, 6))

    # Global normalization so traces share a y-axis
    norm_svs = [normalize_singular_values(v["s"], v["params"], method=normalization_method)
                for _, v in items]
    s_ref = max(s[0] for s in norm_svs)

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
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(items)} traces)")


# %%
for s in SWEEPS:
    plot_sweep(**s)
