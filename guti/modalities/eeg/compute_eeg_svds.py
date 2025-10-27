"""
Compute SVDs of EEG leadfields using OpenMEEG with single-parameter sweeps.

This script performs a parameter sweep over ONE parameter while holding others constant.
Compatible with the centralized Parameters class and visualization in scaling.py.
"""

#%%
import sys
sys.path.append('../..')

import os
import os.path as op
import subprocess
import numpy as np
import h5py
from pathlib import Path

# Add autoreload for interactive development
from guti.utils import enable_autoreload
from guti.parameters import Parameters
from guti.data_utils import save_svd, list_svd_variants

enable_autoreload()

print(__doc__)

#%%
# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================
# Specify which parameter to sweep (one of: num_sensors, source_spacing_mm, grid_resolution_mm)

# grid_resolution_mm
# source_spacing_mm
# num_sensors

# SWEEP_PARAM = "num_sensors"
# SWEEP_PARAM = "source_spacing_mm"
SWEEP_PARAM = "grid_resolution_mm"

# Define sweep range using linspace
SWEEP_MIN = 10.0      # Minimum value
SWEEP_MAX = 20.0      # Maximum value
SWEEP_N_POINTS = 7    # Number of points in sweep

# Constant parameters (held fixed during sweep)
# CONSTANT_PARAMS = Parameters(
#     num_sensors=256,
#     grid_resolution_mm=20.0,
# )

# CONSTANT_PARAMS = Parameters(
    # source_spacing_mm=5.0,
    # grid_resolution_mm=20.0,
# )

CONSTANT_PARAMS = Parameters(
    num_sensors=256,
    source_spacing_mm=5.0, #5
)

# Generate sweep values
sweep_values = np.linspace(SWEEP_MIN, SWEEP_MAX, SWEEP_N_POINTS)
# sweep_values = [5, 25, 50, 100, 150, 200, 250]
print(sweep_values)

# Get the path to the virtual environment
venv_path = op.join(op.dirname(__file__), '.venv')
venv_activate = op.join(venv_path, 'bin', 'activate')

# Set up environment to ensure conda tools are accessible
env = os.environ.copy()
conda_bin = '/Users/thomasribeiro/miniconda3/bin'
if conda_bin not in env.get('PATH', ''):
    env['PATH'] = f"{conda_bin}:{env.get('PATH', '')}"

#%%
def load_mat73(file_path):
    """Load MATLAB v7.3 format files."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            var = f[key][()]
            if isinstance(var, h5py.Reference):
                var = f[var][()]
            data[key] = var
    return data

#%%
# ============================================================================
# VISUALIZATION SETUP
# ============================================================================
import pyvista as pv
from guti.tri_view import visualize_bem_layers

pv.set_jupyter_backend(None)
pv.set_plot_theme("document")
pv.OFF_SCREEN = False

#%%
# ============================================================================
# PARAMETER SWEEP
# ============================================================================
print(f"\nSweeping {SWEEP_PARAM} from {SWEEP_MIN} to {SWEEP_MAX} ({SWEEP_N_POINTS} points)")
print(f"Constant parameters: {CONSTANT_PARAMS}\n")

for sweep_value in sweep_values:
    print(f"\n{'='*80}")
    print(f"Computing EEG leadfield: {SWEEP_PARAM}={sweep_value}")
    print(f"{'='*80}\n")

    # Build complete parameters by combining constant params + current sweep value
    params_dict = CONSTANT_PARAMS.to_dict()
    params_dict[SWEEP_PARAM] = sweep_value
    params = Parameters.from_dict(params_dict)

    # Extract individual parameters for create_eeg_bem_model
    # (need to provide defaults if not specified in params)
    source_spacing = params.source_spacing_mm if params.source_spacing_mm is not None else 10.0
    n_sensors = params.num_sensors if params.num_sensors is not None else 64
    grid_res = params.grid_resolution_mm if params.grid_resolution_mm is not None else 20.0

    print(f"Parameters: source_spacing={source_spacing}mm, n_sensors={n_sensors}, grid_resolution={grid_res}mm")

    # Create BEM model with current parameters
    from guti.core import create_eeg_bem_model
    create_eeg_bem_model(
        source_spacing_mm=source_spacing,
        n_sensors=n_sensors,
        grid_resolution=grid_res
    )

    # Visualize the BEM model
    print(f"\nVisualizing BEM model for {SWEEP_PARAM}={sweep_value}...")
    visualize_bem_layers(
        geom_path="guti/modalities/bem_model/eeg/sphere_head.geom",
        dipole_path="guti/modalities/bem_model/eeg/dipole_locations.txt",
        sensor_path="guti/modalities/bem_model/eeg/sensor_locations.txt",
        show_edges=True,
        layer_opacity={"Brain": 1, "Skull": 0.3, "Scalp": 0.2},
        layer_colors={"Brain": "green", "Skull": "blue", "Scalp": "peachpuff"},
    )

    # Compute leadfield matrix using bash script
    try:
        cmd = f'source {venv_activate} && bash guti/modalities/compute_eeg_leadfield.sh'
        result = subprocess.run(
            ['bash', '-c', cmd],
            capture_output=True,
            text=True,
            cwd='.',
            env=env,
            check=True
        )
        print("Leadfield computation successful")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        continue

    # Load the leadfield matrix
    try:
        G_eeg = load_mat73('guti/modalities/leadfields/eeg/eeg_leadfield.mat')['linop']
    except Exception as e:
        print(f"Failed to load leadfield: {e}")
        continue

    print(f"G_eeg shape: {G_eeg.shape}")

    # Compute SVD
    s_eeg = np.linalg.svdvals(G_eeg)

    # Update parameters with actual number of sources (convert to Python int for JSON serialization)
    params.num_brain_grid_points = G_eeg.shape[0]

    # Save using the centralized infrastructure
    save_svd(s_eeg, modality_name='eeg_openmeeg', params=params)

    print(f"Leadfield shape: {G_eeg.shape}")
    print(f"Condition number: {s_eeg[0] / s_eeg[-1]:.2e}")
    print(f"Saved SVD with parameters: {params}")

#%%
# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EEG Parameter Sweep Summary")
print("="*80)

# Load all saved variants for this modality with current constant params
variants = list_svd_variants('eeg_openmeeg', constant_params=CONSTANT_PARAMS)

print(f"\nFound {len(variants)} saved variants:")
for key, variant in variants.items():
    params = variant['params']
    s = variant['s']
    print(f"\n{key}:")
    print(f"  Parameters: {params}")
    print(f"  SVD shape: {s.shape}")
    print(f"  Condition number: {s[0] / s[-1]:.2e}")
    print(f"  Effective rank (>1% of max): {np.sum(s / s[0] > 0.01)}")

print(f"\n{'='*80}")
print("Sweep complete! Visualize results using scaling.py")
print(f"{'='*80}\n")

# %%
