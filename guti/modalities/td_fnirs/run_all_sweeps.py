"""Run four TD-fNIRS (analytical) sweeps back-to-back and save SVDs.

Sweeps {num_sensors, grid_resolution_mm, max_dist, n_time_gates} over the
TDfNIRSAnalytical modality. Uses the standard ImagingModality.run() pipeline.

Pins mirror the CW fNIRS sweep (run_all_sweeps.py) so results are comparable.
num_sensors range is capped at 1024 (instead of 2048) to keep peak GPU memory
manageable when this runs concurrently with other GPU workloads.
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
from guti.modalities.td_fnirs.modality import TDfNIRSAnalytical

PIN_NUM_SENSORS = 800   # match the CW converged config
PIN_GRID_RES_MM = 3.0   # converged grid (see grid_resolution sweep)
PIN_MAX_DIST = 40.0     # CW-converged max source-detector distance
PIN_N_TIME_GATES = 6
# Grid-resolution convergence is independent of sensor count, so the grid sweep
# uses a moderate (well-sampled) sensor count where the fine-grid full SVD stays
# accurate and fast (the 800-sensor x 6-gate SVD is intractable below ~4 mm).
GRID_SWEEP_NUM_SENSORS = 256

SWEEPS = [
    dict(
        name="num_sensors",
        sweep_param="num_sensors",
        # 1024 dropped: at 3 mm it is a ~44 GB matrix. 32-800 captures the trend.
        sweep_values=[32, 64, 128, 256, 512, 800],
        constants=dict(
            grid_resolution_mm=PIN_GRID_RES_MM,
            max_dist=PIN_MAX_DIST,
            n_time_gates=PIN_N_TIME_GATES,
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
            n_time_gates=PIN_N_TIME_GATES,
        ),
    ),
    dict(
        name="max_dist",
        sweep_param="max_dist",
        # 800-sensor spacing (~8 mm) supports 10 mm; matches the CW max_dist sweep.
        sweep_values=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        constants=dict(
            num_sensors=PIN_NUM_SENSORS,
            grid_resolution_mm=PIN_GRID_RES_MM,
            n_time_gates=PIN_N_TIME_GATES,
        ),
    ),
    dict(
        name="n_time_gates",
        sweep_param="n_time_gates",
        sweep_values=[1, 2, 4, 6, 8, 12],
        constants=dict(
            num_sensors=PIN_NUM_SENSORS,
            grid_resolution_mm=PIN_GRID_RES_MM,
            max_dist=PIN_MAX_DIST,
        ),
    ),
]


def run_sweep(name, sweep_param, sweep_values, constants):
    print(f"\n{'#' * 80}\n# TD-fNIRS sweep '{name}' over {sweep_param}: {sweep_values}\n{'#' * 80}", flush=True)
    for v in sweep_values:
        t0 = time.time()
        params = Parameters.from_dict({**constants, sweep_param: v})
        modality = TDfNIRSAnalytical(params=params)
        s = modality.run()
        print(
            f"[{name}] {sweep_param}={v}  matrix={modality.params.matrix_size}  "
            f"cond={s[0]/s[-1]:.2e}  {time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    for s in SWEEPS:
        run_sweep(**s)
    print("\nAll TD-fNIRS sweeps done.")
