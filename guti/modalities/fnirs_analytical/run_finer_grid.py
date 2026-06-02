"""Push fNIRS-CW grid_resolution_mm sweep finer (5, 4, 3 mm).

Holds num_sensors=800 and max_dist=40 mm fixed — same pin as run_all_sweeps.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from guti.parameters import Parameters
from guti.modalities.fnirs_analytical.modality import fNIRSAnalytical

PIN_NUM_SENSORS = 800
PIN_MAX_DIST = 40.0
EXTRA_GRID_RES_MM = [5.0, 4.0, 3.0]


def main():
    for gr in EXTRA_GRID_RES_MM:
        t0 = time.time()
        params = Parameters.from_dict(
            dict(num_sensors=PIN_NUM_SENSORS, grid_resolution_mm=gr, max_dist=PIN_MAX_DIST)
        )
        s = fNIRSAnalytical(params=params).run()
        print(
            f"gr={gr}  matrix={params.matrix_size}  cond={s[0]/s[-1]:.2e}  "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )
    print("done")


if __name__ == "__main__":
    main()
