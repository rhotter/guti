# %%
try:
    import IPython

    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
except:
    pass

from guti.core import get_grid_positions, get_sensor_positions

import numpy as np
import torch
from guti.modalities.fnirs_analytical.utils import (
    cw_sensitivity_batched,
    get_valid_source_detector_pairs,
)

# %%
grid_spacing_mm = 5.0
noptodes = 800

grid_points_mm = get_grid_positions(grid_spacing_mm)
sensor_positions_mm = get_sensor_positions(noptodes)
mu_a = 0.02  # cm^-1
mu_s_prime = 6.7  # cm^-1
mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))
mu_eff = mu_eff * 1e-1  # mm^-1

max_dist = 50  # mm

# Convert to torch tensors and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_points_torch = torch.from_numpy(grid_points_mm).float().to(device)
sensor_positions_torch = torch.from_numpy(sensor_positions_mm).float().to(device)

# %%
# Get all valid source-detector pairs
sources, detectors = get_valid_source_detector_pairs(sensor_positions_torch, max_dist)

print(
    f"Number of pairs: {sources.shape[0]}, Number of grid points: {grid_points_torch.shape[0]}"
)
# Calculate sensitivities for all pairs and grid points using batched computation
sensitivities = cw_sensitivity_batched(
    grid_points_torch, sources, detectors, mu_eff, batch_size=500
)
print(f"Sensitivities shape: {sensitivities.shape}")

# %%
from guti.svd import compute_svd_gpu

s = compute_svd_gpu(sensitivities)

# %%
from guti.svd import plot_svd

plot_svd(s)

# %%
from guti.data_utils import save_svd, Parameters

save_svd(
    s,
    "fnirs_analytical_cw",
    Parameters(
        num_sensors=noptodes,
        grid_resolution_mm=grid_spacing_mm,
        num_brain_grid_points=grid_points_torch.shape[0],
    ),
    default=False,
    subdir="grid_resolution_sweep",  # Organize baseline runs in their own subdirectory
)

# %%
# # check sensor positions
# from guti.viz import visualize_grid_and_sensors

# fig = visualize_grid_and_sensors(
#     grid_points_mm,
#     sensor_positions_mm,
# )


# %%
