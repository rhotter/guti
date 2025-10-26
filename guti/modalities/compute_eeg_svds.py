"""
Compute SVDs of EEG leadfields using OpenMEEG with parameter sweeps.

This script sweeps over:
- Source spacing (controls number of dipole sources)
- Number of EEG sensors
- Mesh resolution

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
N_SENSORS_RANGE = [32, 64, 128]       # Number of EEG sensors
GRID_RESOLUTION_RANGE = [40, 20, 10]  # Grid resolution in mm (smaller = finer mesh)

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
results_dir = Path('results/eeg_parameter_sweep')
results_dir.mkdir(parents=True, exist_ok=True)

# Storage for all results
all_results = {}

#%%
# Parameter sweep
for source_spacing in SOURCE_SPACING_RANGE:
    for n_sensors in N_SENSORS_RANGE:
        for grid_res in GRID_RESOLUTION_RANGE:
            print(f"\n{'='*80}")
            print(f"Computing EEG leadfield: {source_spacing}mm spacing, {n_sensors} sensors, {grid_res}mm resolution")
            print(f"{'='*80}\n")

            # Create BEM model with current parameters
            from guti.core import create_eeg_bem_model
            create_eeg_bem_model(
                source_spacing_mm=source_spacing,
                n_sensors=n_sensors,
                grid_resolution=grid_res
            )

            # Compute leadfield matrix using bash script
            try:
                cmd = f'source {venv_activate} && bash compute_eeg_leadfield.sh'
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
                G_eeg = load_mat73('leadfields/eeg/eeg_leadfield.mat')['linop']
            except Exception as e:
                print(f"Failed to load leadfield: {e}")
                continue

            # Compute SVD
            s_eeg = np.linalg.svdvals(G_eeg)
            s_eeg_normalized = s_eeg / s_eeg[0]

            # Save using the centralized infrastructure
            from guti.data_utils import save_svd
            from guti.parameters import Parameters

            params = Parameters(
                num_sensors=n_sensors,
                source_spacing_mm=source_spacing,
                grid_resolution_mm=grid_res,
                num_brain_grid_points=G_eeg.shape[1],  # Number of sources
            )
            save_svd(s_eeg, modality_name='eeg_openmeeg', params=params)

            # Store results for immediate plotting (optional)
            param_key = f"spacing{source_spacing}_sen{n_sensors}_res{grid_res}"
            all_results[param_key] = {
                'source_spacing': source_spacing,
                'n_sensors': n_sensors,
                'grid_resolution': grid_res,
                'singular_values': s_eeg,
                'singular_values_normalized': s_eeg_normalized,
                'leadfield_shape': G_eeg.shape
            }

            print(f"Leadfield shape: {G_eeg.shape}")
            print(f"Condition number: {s_eeg[0] / s_eeg[-1]:.2e}")

#%%
# Plot results - varying source spacing
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, grid_res in zip(axes, GRID_RESOLUTION_RANGE):
    for n_sensors in N_SENSORS_RANGE:
        for source_spacing in SOURCE_SPACING_RANGE:
            param_key = f"spacing{source_spacing}_sen{n_sensors}_res{grid_res}"
            if param_key in all_results:
                s = all_results[param_key]['singular_values_normalized']
                ax.semilogy(s, label=f'{source_spacing}mm spacing, {n_sensors} sensors', alpha=0.7)

    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title(f'Grid resolution: {grid_res}mm')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('EEG SVD Spectra - Varying Source Spacing and Sensor Count')
plt.tight_layout()
plt.savefig(results_dir / 'eeg_svd_comparison.png', dpi=150)
plt.show()

#%%
# Plot results - varying grid resolution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, source_spacing in zip(axes, SOURCE_SPACING_RANGE):
    for n_sensors in N_SENSORS_RANGE:
        for grid_res in GRID_RESOLUTION_RANGE:
            param_key = f"spacing{source_spacing}_sen{n_sensors}_res{grid_res}"
            if param_key in all_results:
                s = all_results[param_key]['singular_values_normalized']
                ax.semilogy(s, label=f'{n_sensors} sensors, {grid_res}mm', alpha=0.7)

    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title(f'Source spacing: {source_spacing}mm')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('EEG SVD Spectra - Varying Sensor Count and Grid Resolution')
plt.tight_layout()
plt.savefig(results_dir / 'eeg_svd_grid_comparison.png', dpi=150)
plt.show()

#%%
# Summary statistics
print("\n" + "="*80)
print("EEG Parameter Sweep Summary")
print("="*80)
for param_key, result in all_results.items():
    print(f"\n{param_key}:")
    print(f"  Leadfield shape: {result['leadfield_shape']}")
    print(f"  Condition number: {result['singular_values'][0] / result['singular_values'][-1]:.2e}")
    print(f"  Effective rank (>1% of max): {np.sum(result['singular_values_normalized'] > 0.01)}")

# Save all results together
np.savez(
    results_dir / 'eeg_all_results.npz',
    **{key: val['singular_values'] for key, val in all_results.items()}
)

print(f"\nResults saved to {results_dir}")

# %%
