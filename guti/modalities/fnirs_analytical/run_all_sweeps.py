"""Run three CW-fNIRS (analytical) sweeps back-to-back and save SVDs.

Sweeps {num_sensors, grid_resolution_mm, max_dist} over the fNIRSAnalytical
modality. Uses the standard ImagingModality.run() pipeline (GPU SVD).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guti.parameters import Parameters
from guti.modalities.fnirs_analytical.modality import fNIRSAnalytical

# Pinned "center" values shared across sweeps.
PIN_NUM_SENSORS = 800
PIN_GRID_RES_MM = 6.0
PIN_MAX_DIST = 40.0
# Grid-resolution convergence is independent of sensor count, so the grid sweep
# uses a moderate (well-sampled) sensor count where the fine-grid full SVD stays
# accurate and fast (the 800-sensor SVD is intractable below ~4 mm).
GRID_SWEEP_NUM_SENSORS = 256

SWEEPS = [
    dict(
        name="num_sensors",
        sweep_param="num_sensors",
        sweep_values=[32, 64, 128, 256, 512, 1024, 2048],
        constants=dict(
            grid_resolution_mm=PIN_GRID_RES_MM,
            max_dist=PIN_MAX_DIST,
        ),
    ),
    dict(
        name="grid_resolution_mm",
        sweep_param="grid_resolution_mm",
        # Fine grids pushed to 2.5 mm to show full quadrature convergence of
        # sigma/sqrt(voxel_volume). Run at GRID_SWEEP_NUM_SENSORS so the fine-grid
        # full SVD stays tractable/accurate (convergence is sensor-independent).
        sweep_values=[8.0, 6.0, 5.0, 4.0, 3.0, 2.5],
        constants=dict(
            num_sensors=GRID_SWEEP_NUM_SENSORS,
            max_dist=PIN_MAX_DIST,
        ),
    ),
    dict(
        name="max_dist",
        sweep_param="max_dist",
        sweep_values=[float(v) for v in np.linspace(10.0, 70.0, 7)],
        constants=dict(
            num_sensors=PIN_NUM_SENSORS,
            grid_resolution_mm=PIN_GRID_RES_MM,
        ),
    ),
]


def run_sweep(name, sweep_param, sweep_values, constants):
    print(f"\n{'#' * 80}\n# fNIRS sweep '{name}' over {sweep_param}: {sweep_values}\n{'#' * 80}")
    for v in sweep_values:
        t0 = time.time()
        params = Parameters.from_dict({**constants, sweep_param: v})
        modality = fNIRSAnalytical(params=params)
        s = modality.run()
        print(
            f"[{name}] {sweep_param}={v}  matrix={modality.params.matrix_size}  "
            f"cond={s[0]/s[-1]:.2e}  {time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    for s in SWEEPS:
        run_sweep(**s)
    print("\nAll fNIRS sweeps done.")
