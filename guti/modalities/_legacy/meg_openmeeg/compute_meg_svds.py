"""
Compute SVDs of MEG leadfields using OpenMEEG with parameter sweeps.

This script sweeps over:
- Source spacing (controls number of dipole sources)
- Number of MEG sensors
- Mesh resolution
- Sensor offset (OPMs vs SQUIDs)

For each parameter combination, it creates a BEM model, computes the leadfield
matrix, and analyzes the SVD spectrum.
"""

#%%
import sys
sys.path.append('../..')

import os
import os.path as op
import subprocess
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Add autoreload for interactive development
from guti.utils import enable_autoreload
enable_autoreload()

print(__doc__)

#%%
# Define parameter sweep ranges (in decreasing order for finer resolution)
SOURCE_SPACING_RANGE = [20, 10, 5]    # Spacing between dipoles in mm (smaller = more sources)
N_SENSORS_RANGE = [32, 64, 128]       # Number of MEG sensors
GRID_RESOLUTION_RANGE = [40, 20, 10]  # Grid resolution in mm (smaller = finer mesh)
SENSOR_OFFSET_RANGE = [20, 5]         # Distance from scalp in mm (20mm=SQUIDs, 5mm=OPMs)

# Get the path to the virtual environment
venv_path = op.join(op.dirname(__file__), '..', '..', '.venv')
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
# Create results directory
results_dir = Path('results/meg_parameter_sweep')
results_dir.mkdir(parents=True, exist_ok=True)

# Storage for all results
all_results = {}

#%%
# Parameter sweep
for source_spacing in SOURCE_SPACING_RANGE:
    for n_sensors in N_SENSORS_RANGE:
        for grid_res in GRID_RESOLUTION_RANGE:
            for sensor_offset in SENSOR_OFFSET_RANGE:
                sensor_type = "SQUIDs" if sensor_offset == 20 else "OPMs"
                print(f"\n{'='*80}")
                print(f"Computing MEG leadfield: {source_spacing}mm spacing, {n_sensors} sensors, "
                      f"{grid_res}mm resolution, {sensor_offset}mm offset ({sensor_type})")
                print(f"{'='*80}\n")

                # Create BEM model with current parameters
                from guti.core import create_meg_bem_model
                create_meg_bem_model(
                    source_spacing_mm=source_spacing,
                    n_sensors=n_sensors,
                    grid_resolution=grid_res,
                    sensor_offset=sensor_offset
                )

                # Compute leadfield matrix using bash script
                try:
                    cmd = f'source {venv_activate} && bash compute_meg_leadfield.sh'
                    result = subprocess.run(
                        ['bash', '-c', cmd],
                        capture_output=True,
                        text=True,
                        cwd='.',
                        env=env,
                        check=True
                    )
                    print("Leadfield computation successful")
                    if result.stdout:
                        print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Script failed with return code {e.returncode}")
                    print(f"Error output: {e.stderr}")
                    print(f"Standard output: {e.stdout}")
                    continue

                # Load the leadfield matrix
                try:
                    G_meg = load_mat73('leadfields/meg/meg_leadfield.mat')['linop']
                except Exception as e:
                    print(f"Failed to load leadfield: {e}")
                    continue

                # Compute SVD
                s_meg = np.linalg.svdvals(G_meg)
                s_meg_normalized = s_meg / s_meg[0]

                # Save using the centralized infrastructure
                from guti.data_utils import save_svd
                from guti.parameters import Parameters

                params = Parameters(
                    num_sensors=n_sensors,
                    source_spacing_mm=source_spacing,
                    grid_resolution_mm=grid_res,
                    sensor_offset_mm=sensor_offset,
                    num_brain_grid_points=G_meg.shape[1],  # Number of sources
                )
                save_svd(s_meg, modality_name='meg_openmeeg', params=params)

                # Store results for immediate plotting (optional)
                param_key = f"spacing{source_spacing}_sen{n_sensors}_res{grid_res}_off{sensor_offset}"
                all_results[param_key] = {
                    'source_spacing': source_spacing,
                    'n_sensors': n_sensors,
                    'grid_resolution': grid_res,
                    'sensor_offset': sensor_offset,
                    'singular_values': s_meg,
                    'singular_values_normalized': s_meg_normalized,
                    'leadfield_shape': G_meg.shape
                }

                print(f"Leadfield shape: {G_meg.shape}")
                print(f"Condition number: {s_meg[0] / s_meg[-1]:.2e}")

#%%
# Plot results - varying sensor offset (OPMs vs SQUIDs)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

source_spacing_fixed = SOURCE_SPACING_RANGE[1]  # Use middle value (10mm)
grid_res_fixed = GRID_RESOLUTION_RANGE[1]  # Use middle value (20mm)

for ax, n_sensors in zip(axes, N_SENSORS_RANGE):
    for sensor_offset in SENSOR_OFFSET_RANGE:
        sensor_type = "SQUIDs" if sensor_offset == 20 else "OPMs"
        param_key = f"spacing{source_spacing_fixed}_sen{n_sensors}_res{grid_res_fixed}_off{sensor_offset}"
        if param_key in all_results:
            s = all_results[param_key]['singular_values_normalized']
            ax.semilogy(s, label=f'{sensor_type} ({sensor_offset}mm)', alpha=0.7)

    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title(f'{n_sensors} sensors ({source_spacing_fixed}mm spacing, {grid_res_fixed}mm grid)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('MEG SVD Spectra - OPMs vs SQUIDs')
plt.tight_layout()
plt.savefig(results_dir / 'meg_svd_offset_comparison.png', dpi=150)
plt.show()

#%%
# Plot results - varying source spacing and sensor count
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sensor_offset_fixed = SENSOR_OFFSET_RANGE[0]  # Use SQUIDs (20mm)

for ax, grid_res in zip(axes, GRID_RESOLUTION_RANGE):
    for n_sensors in N_SENSORS_RANGE:
        for source_spacing in SOURCE_SPACING_RANGE:
            param_key = f"spacing{source_spacing}_sen{n_sensors}_res{grid_res}_off{sensor_offset_fixed}"
            if param_key in all_results:
                s = all_results[param_key]['singular_values_normalized']
                ax.semilogy(s, label=f'{source_spacing}mm spacing, {n_sensors} sensors', alpha=0.7)

    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title(f'Grid: {grid_res}mm (SQUIDs)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('MEG SVD Spectra - Varying Source Spacing and Sensor Count')
plt.tight_layout()
plt.savefig(results_dir / 'meg_svd_sources_comparison.png', dpi=150)
plt.show()

#%%
# Summary statistics
print("\n" + "="*80)
print("MEG Parameter Sweep Summary")
print("="*80)
for param_key, result in all_results.items():
    print(f"\n{param_key}:")
    print(f"  Leadfield shape: {result['leadfield_shape']}")
    print(f"  Condition number: {result['singular_values'][0] / result['singular_values'][-1]:.2e}")
    print(f"  Effective rank (>1% of max): {np.sum(result['singular_values_normalized'] > 0.01)}")

# Save all results together
np.savez(
    results_dir / 'meg_all_results.npz',
    **{key: val['singular_values'] for key, val in all_results.items()}
)

print(f"\nResults saved to {results_dir}")

# %%
