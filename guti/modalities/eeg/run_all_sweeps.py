"""Run all three EEG-OpenMEEG sweeps back-to-back and save SVDs.

Mirrors the three config blocks in compute_eeg_svds.py but runs them in a
single process. Uses the OpenMEEG Python API (no CLI required).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guti.core import create_eeg_bem_model
from guti.data_utils import save_svd
from guti.parameters import Parameters
from guti.modalities.eeg.compute_eeg_leadfield import compute_eeg_leadfield

BEM_DIR = REPO_ROOT / "guti" / "modalities" / "bem_model" / "eeg"

N_DIPOLES_PER_LINE = 5

SWEEPS = [
    dict(
        name="n_radial_lines",
        sweep_param="n_radial_lines",
        sweep_values=[
            int(v) for v in np.linspace(5, 2048 / N_DIPOLES_PER_LINE, 7)
        ],
        constants=dict(
            grid_resolution_mm=8.0,
            num_sensors=2048,
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
        ),
    ),
    dict(
        name="num_sensors",
        sweep_param="num_sensors",
        sweep_values=[int(v) for v in np.linspace(5, 2048, 7)],
        constants=dict(
            grid_resolution_mm=8.0,
            n_radial_lines=409,
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
        ),
    ),
    dict(
        name="grid_resolution_mm",
        sweep_param="grid_resolution_mm",
        sweep_values=list(np.linspace(20.0, 8.0, 7)),
        constants=dict(
            num_sensors=2048,
            n_radial_lines=409,
            n_dipoles_per_line=N_DIPOLES_PER_LINE,
        ),
    ),
]

BEM_KWARG_BY_SWEEP = {
    "n_radial_lines": "n_radial_lines",
    "num_sensors": "n_sensors",
    "grid_resolution_mm": "grid_resolution",
}


def run_sweep(name, sweep_param, sweep_values, constants):
    print(f"\n{'#' * 80}\n# Sweep '{name}' over {sweep_param}: {sweep_values}\n{'#' * 80}")

    for sweep_value in sweep_values:
        t0 = time.time()
        params_dict = dict(constants)
        params_dict[sweep_param] = sweep_value
        params = Parameters.from_dict(params_dict)

        # BEM-model kwargs: start from constants + swept value, translate names.
        bem_kwargs = dict(
            n_radial_lines=params.n_radial_lines,
            n_dipoles_per_line=params.n_dipoles_per_line,
            n_sensors=params.num_sensors,
            grid_resolution=params.grid_resolution_mm,
            output_dir=str(BEM_DIR),
        )
        create_eeg_bem_model(**bem_kwargs)

        G = compute_eeg_leadfield(
            BEM_DIR / "sphere_head.geom",
            BEM_DIR / "sphere_head.cond",
            BEM_DIR / "dipole_locations.txt",
            BEM_DIR / "sensor_locations.txt",
        )
        s = np.linalg.svdvals(G)
        params.num_brain_grid_points = int(G.shape[1])
        save_svd(s, modality_name="eeg_openmeeg", params=params)

        dt = time.time() - t0
        print(
            f"[{name}] {sweep_param}={sweep_value}  G={G.shape}  "
            f"cond={s[0]/s[-1]:.2e}  {dt:.1f}s"
        )


if __name__ == "__main__":
    for s in SWEEPS:
        run_sweep(**s)
    print("\nAll sweeps done.")
