"""
Compute SVDs of leadfields using OpenMEEG.
"""

# %%
import sys

sys.path.append("../..")

import os.path as op
import openmeeg as om
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Add autoreload for interactive development
from guti.utils import enable_autoreload

enable_autoreload()

print(__doc__)

# %%
# Create the BEM brain model
from guti.core import (
    create_bem_model,
    create_bem_model_hemisphere,
    create_sphere,
    BRAIN_RADIUS,
)

create_bem_model()
# create_bem_model_hemisphere()

# %%
# Visualize the BEM model
import sys
from pathlib import Path
import pyvista as pv

# Set PyVista to use interactive GUI window (not inline/notebook)
pv.set_jupyter_backend(None)  # Disable notebook backend to avoid trame warning
pv.set_plot_theme("document")
pv.OFF_SCREEN = False

# Add bem_model directory to path to import tri_view
bem_model_path = Path(__file__).parent / "guti/modalities/bem_model"
sys.path.insert(0, str(bem_model_path))

from guti.tri_view import visualize_bem_layers

# Visualize all BEM layers with dipoles and sensors
visualize_bem_layers(
    geom_path="guti/modalities/bem_model/sphere_head.geom",
    dipole_path="guti/modalities/bem_model/dipole_locations.txt",
    # sensor_path="guti/modalities/bem_model/sensor_locations.txt",
    show_edges=True,
    layer_opacity={"Brain": 1, "Skull": 0.3, "Scalp": 0.2},
    layer_colors={"Brain": "green", "Skull": "blue", "Scalp": "peachpuff"},
)

# %%
# Compute the leadfield matrices
import subprocess
import os

# Get the path to the virtual environment
venv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".venv")
venv_activate = os.path.join(venv_path, "bin", "activate")

# Set up environment to ensure conda tools are accessible
env = os.environ.copy()
# Add conda bin to PATH to ensure OpenMEEG tools are found
conda_bin = "/workspace/miniconda3/bin"
if conda_bin not in env.get("PATH", ""):
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

try:
    # Run the bash script with the virtual environment activated and conda tools in PATH
    cmd = f"source {venv_activate} && bash compute_leadfields.sh"
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        cwd=".",
        env=env,
        check=True,
    )
    print("Script output:")
    print(result.stdout)
    if result.stderr:
        print("Script errors:")
        print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Script failed with return code {e.returncode}")
    print(f"Error output: {e.stderr}")
    print(f"Standard output: {e.stdout}")
except FileNotFoundError:
    print("Error: compute_leadfields.sh not found in current directory")


# %%
# Load the data
def load_mat73(file_path):
    data = {}
    with h5py.File(file_path, "r") as f:
        # Iterate through all variables in the .mat file
        for key in f.keys():
            # Get the variable data
            var = f[key][()]
            # If the variable is a reference to another dataset, follow the reference
            if isinstance(var, h5py.Reference):
                var = f[var][()]
            # Store in dictionary
            data[key] = var
    return data


G_meg = load_mat73("leadfields/meg_leadfield.mat")["linop"]
G_eeg = load_mat73("leadfields/eeg_leadfield.mat")["linop"]
G_eit = load_mat73("leadfields/eit_leadfield.mat")["linop"]
# G_ip = load_mat73('leadfields/ip_leadfield.mat')['linop']
# G_ecog = load_mat73('leadfields/ecog_leadfield.mat')['linop']

# %%
# Compute SVDs
s_meg = np.linalg.svdvals(G_meg)
s_eeg = np.linalg.svdvals(G_eeg)
s_eit = np.linalg.svdvals(G_eit)
# s_ip = np.linalg.svdvals(G_ip)
# s_ecog = np.linalg.svdvals(G_ecog)

# normalize by the first singular value
s_meg = s_meg / s_meg[0]
s_eeg = s_eeg / s_eeg[0]
s_eit = s_eit / s_eit[0]
# s_ip = s_ip / s_ip[0]
# s_ecog = s_ecog / s_ecog[0]

# %%
# Plot SVDs
plt.semilogy(s_meg, label="MEG")
plt.semilogy(s_eeg, label="EEG")
plt.semilogy(s_eit, label="EIT")
# plt.semilogy(s_ip, label='IP')
# plt.semilogy(s_ecog, label='ECoG')
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.grid(True)
plt.legend()
plt.show()

# %%
# Save the SVDs

from guti.data_utils import save_svd

save_svd(s_meg, "meg_openmeeg")
save_svd(s_eeg, "eeg_openmeeg")
save_svd(s_eit, "eit_openmeeg")
# save_svd(s_ip, 'ip_openmeeg')
# save_svd(s_ecog, 'ecog_openmeeg')

# %%
