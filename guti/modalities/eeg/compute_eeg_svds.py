"""
Compute SVDs of EEG leadfields using OpenMEEG with single-parameter sweeps.

Uses the OpenMEEG Python API (no CLI binaries required).
Compatible with the centralized Parameters class and visualization in scaling.py.
"""

#%%
import sys
from pathlib import Path

# Make the top-level `guti` package importable when running this file directly.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from guti.utils import enable_autoreload
from guti.parameters import Parameters
from guti.data_utils import save_svd, list_svd_variants
from guti.core import create_eeg_bem_model
from guti.modalities.eeg.compute_eeg_leadfield import compute_eeg_leadfield

enable_autoreload()

print(__doc__)

#%%
# ============================================================================
# SWEEP CONFIGURATION (radial-line dipole method)
# ============================================================================
# Three sweep configs over {n_radial_lines, num_sensors, grid_resolution_mm}
# while holding the others (plus n_dipoles_per_line) fixed. Uncomment one.

# --- Sweep 1: n_radial_lines ---
# SWEEP_PARAM = "n_radial_lines"
# GRID_RESOLUTION_MM = 8.0
# N_RADIAL_LINES = 5
# N_DIPOLES_PER_LINE = 5
# N_SENSORS = 2048
# source_values = np.linspace(N_RADIAL_LINES, N_SENSORS / N_DIPOLES_PER_LINE, 7)
# sweep_values = [int(v) for v in source_values]
# CONSTANT_PARAMS = Parameters(
#     num_sensors=N_SENSORS,
#     grid_resolution_mm=GRID_RESOLUTION_MM,
#     n_dipoles_per_line=N_DIPOLES_PER_LINE,
# )

# --- Sweep 2: num_sensors ---
# SWEEP_PARAM = "num_sensors"
# GRID_RESOLUTION_MM = 8.0
# N_RADIAL_LINES = 409
# N_DIPOLES_PER_LINE = 5
# N_SENSORS = 5
# sensor_values = np.linspace(N_SENSORS, 2048, 7)
# sweep_values = [int(v) for v in sensor_values]
# CONSTANT_PARAMS = Parameters(
#     grid_resolution_mm=GRID_RESOLUTION_MM,
#     n_dipoles_per_line=N_DIPOLES_PER_LINE,
#     n_radial_lines=N_RADIAL_LINES,
# )

# --- Sweep 3: grid_resolution_mm ---
SWEEP_PARAM = "grid_resolution_mm"
GRID_RESOLUTION_MM = 20.0
N_RADIAL_LINES = 409
N_DIPOLES_PER_LINE = 5
N_SENSORS = 2048
sweep_values = np.linspace(GRID_RESOLUTION_MM, 8.0, 7)
CONSTANT_PARAMS = Parameters(
    n_dipoles_per_line=N_DIPOLES_PER_LINE,
    n_radial_lines=N_RADIAL_LINES,
    num_sensors=N_SENSORS,
)

print(f"Sweep parameter: {SWEEP_PARAM}")
print(f"Sweep values: {list(sweep_values)}")

# BEM model output lives alongside the per-modality folders.
BEM_DIR = REPO_ROOT / "guti" / "modalities" / "bem_model" / "eeg"

BEM_KWARG_BY_SWEEP = {
    "n_radial_lines": "n_radial_lines",
    "num_sensors": "n_sensors",
    "grid_resolution_mm": "grid_resolution",
}

#%%
# ============================================================================
# PARAMETER SWEEP
# ============================================================================
print(f"\nSweeping {SWEEP_PARAM}")
print(f"Constant parameters: {CONSTANT_PARAMS}\n")

for sweep_value in sweep_values:
    print(f"\n{'='*80}")
    print(f"Computing EEG leadfield: {SWEEP_PARAM}={sweep_value}")
    print(f"{'='*80}\n")

    # Build complete parameters by combining constant params + current sweep value.
    params_dict = CONSTANT_PARAMS.to_dict()
    params_dict[SWEEP_PARAM] = sweep_value
    params = Parameters.from_dict(params_dict)

    # Build the BEM-model kwargs from constants, then overwrite the swept one.
    bem_kwargs = dict(
        n_radial_lines=N_RADIAL_LINES,
        n_dipoles_per_line=N_DIPOLES_PER_LINE,
        n_sensors=N_SENSORS,
        grid_resolution=GRID_RESOLUTION_MM,
        output_dir=str(BEM_DIR),
    )
    bem_kwargs[BEM_KWARG_BY_SWEEP[SWEEP_PARAM]] = sweep_value
    create_eeg_bem_model(**bem_kwargs)

    # Compute leadfield via the OpenMEEG Python API.
    G_eeg = compute_eeg_leadfield(
        geom_file=BEM_DIR / "sphere_head.geom",
        cond_file=BEM_DIR / "sphere_head.cond",
        dipole_file=BEM_DIR / "dipole_locations.txt",
        sensor_file=BEM_DIR / "sensor_locations.txt",
    )
    print(f"Leadfield shape: {G_eeg.shape}")

    # SVD of the (n_sensors × n_dipoles) gain matrix.
    s_eeg = np.linalg.svdvals(G_eeg)

    # Record the actual number of sources produced by the BEM builder.
    params.num_brain_grid_points = int(G_eeg.shape[1])

    save_svd(s_eeg, modality_name="eeg_openmeeg", params=params)

    print(f"Condition number: {s_eeg[0] / s_eeg[-1]:.2e}")
    print(f"Saved SVD with parameters: {params}")

#%%
# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EEG Parameter Sweep Summary")
print("=" * 80)

variants = list_svd_variants("eeg_openmeeg", constant_params=CONSTANT_PARAMS)

print(f"\nFound {len(variants)} saved variants:")
for key, variant in variants.items():
    params = variant["params"]
    s = variant["s"]
    print(f"\n{key}:")
    print(f"  Parameters: {params}")
    print(f"  SVD shape: {s.shape}")
    print(f"  Condition number: {s[0] / s[-1]:.2e}")
    print(f"  Effective rank (>1% of max): {np.sum(s / s[0] > 0.01)}")

print(f"\n{'='*80}")
print("Sweep complete! Visualize with guti/modalities/eeg/plot_eeg_sweeps.py")
print(f"{'='*80}\n")

# %%
