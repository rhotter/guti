"""Run the TD-fNIRS n_time_gates sweep (missing values: 1, 2, 4, 8, 12).

n_time_gates=6 already exists as the shared pin value, so it is skipped.
Pins: num_sensors=800, grid_resolution_mm=6.0, max_dist=40.0 (same as run_all_sweeps.py).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from guti.parameters import Parameters
from guti.modalities.td_fnirs.modality import TDfNIRSAnalytical

PIN_NUM_SENSORS = 800
PIN_GRID_RES_MM = 6.0
PIN_MAX_DIST = 40.0
N_TIME_GATES = [1, 2, 4, 8, 12]


def main():
    for ng in N_TIME_GATES:
        t0 = time.time()
        params = Parameters.from_dict(
            dict(
                num_sensors=PIN_NUM_SENSORS,
                grid_resolution_mm=PIN_GRID_RES_MM,
                max_dist=PIN_MAX_DIST,
                n_time_gates=ng,
            )
        )
        s = TDfNIRSAnalytical(params=params).run()
        print(
            f"n_time_gates={ng}  matrix={params.matrix_size}  "
            f"cond={s[0]/s[-1]:.2e}  {time.time()-t0:.1f}s",
            flush=True,
        )
    print("done")


if __name__ == "__main__":
    main()
