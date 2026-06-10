"""Plot SV spectra for the three fNIRS-CW analytical sweeps.

Saves one figure per sweep under guti/modalities/cw_fnirs/results/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parents[2]))

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters


def normalize_singular_values(s, params, method="sqrtN"):
    if method == "none":
        return s
    if method == "s0":
        return s / s[0]
    if method == "sqrtN":
        M, N = params.matrix_size
        # For voxel-integrated transfer functions (units mm^-1) the input-side
        # quadrature weight is sqrt(voxel_volume), not sqrt(N_grid_points); the
        # latter leaves a residual ~ voxel_volume ~ grid^3 that prevents the
        # spectra from overlapping across a grid-resolution sweep.
        vv = getattr(params, "voxel_volume_mm3", None)
        input_scale = vv if vv is not None else N
        return s / np.sqrt(input_scale * M)
    raise ValueError(method)

OUT_DIR = THIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIN_NUM_SENSORS = 800
PIN_GRID_RES_MM = 6.0
PIN_MAX_DIST = 40.0

SWEEPS = [
    dict(
        param_key="num_sensors",
        constant_params=Parameters(
            grid_resolution_mm=PIN_GRID_RES_MM, max_dist=PIN_MAX_DIST,
        ),
        title="fNIRS CW — num_sensors sweep",
        fname="svd_spectrum_num_sensors.png",
    ),
    dict(
        param_key="grid_resolution_mm",
        constant_params=Parameters(
            num_sensors=256, max_dist=PIN_MAX_DIST,
        ),
        title="fNIRS CW — grid_resolution_mm sweep",
        fname="svd_spectrum_grid_resolution_mm.png",
        # Show only the converged fine-grid regime (<=8 mm); legacy coarse
        # variants remain on disk but are far from the continuum limit.
        value_filter=lambda v: v <= 8.0,
    ),
    dict(
        param_key="max_dist",
        constant_params=Parameters(
            num_sensors=PIN_NUM_SENSORS, grid_resolution_mm=PIN_GRID_RES_MM,
        ),
        title="fNIRS CW — max_dist sweep",
        fname="svd_spectrum_max_dist.png",
    ),
]


def plot_sweep(param_key, constant_params, title, fname,
               normalization_method="s0", ylim=(1e-5, 2), value_filter=None):
    variants = list_svd_variants(
        "cw_fnirs", constant_params=constant_params, sort_by=param_key,
    )
    if not variants:
        print(f"[skip] no variants for {param_key}")
        return

    by_value = {}
    for _, v in variants:
        by_value[getattr(v["params"], param_key)] = v
    items = sorted(by_value.items())
    if value_filter is not None:
        items = [(pv, v) for pv, v in items if value_filter(pv)]
    # Drop variants with non-finite / degenerate spectra so one bad run can't
    # blank the whole figure.
    items = [(pv, v) for pv, v in items
             if np.all(np.isfinite(v["s"])) and v["s"][0] > 0]

    norm_svs = [
        normalize_singular_values(v["s"], v["params"], method=normalization_method)
        for _, v in items
    ]
    # "none" = raw singular values: no global rescale, autoscale the y-axis.
    if normalization_method == "none":
        s_ref = 1.0
    else:
        s_ref = max(s[0] for s in norm_svs)

    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(items))]
    fig, ax = plt.subplots(figsize=(9, 6))
    for (pv, v), s_norm, c in zip(items, norm_svs, colors):
        s = s_norm / s_ref
        ax.plot(np.arange(1, len(s) + 1), s, color=c, label=f"{param_key}={pv}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value" if normalization_method == "none"
                   else "Singular value (normalized)")
    ax.set_title(title)
    if normalization_method != "none":
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    out = OUT_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(items)} traces)")


if __name__ == "__main__":
    for method in ["s0", "sqrtN", "none"]:
        for s in SWEEPS:
            kwargs = dict(s)
            base, ext = kwargs["fname"].rsplit(".", 1)
            kwargs["fname"] = f"{base}_{method}.{ext}"
            kwargs["title"] = f"{kwargs['title']}  [{method}]"
            plot_sweep(normalization_method=method, **kwargs)
