# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

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
max_dist = 50  # mm

grid_points_mm = get_grid_positions(grid_spacing_mm)
sensor_positions_mm = get_sensor_positions(noptodes)
mu_a = 0.02  # cm^-1
mu_s_prime = 6.7  # cm^-1
mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))
mu_eff = mu_eff * 1e-1  # mm^-1

# Convert to torch tensors and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_points_torch = torch.from_numpy(grid_points_mm).float().to(device)
sensor_positions_torch = torch.from_numpy(sensor_positions_mm).float().to(device)

# %%
# Get all valid source-detector pairs
sources, detectors = get_valid_source_detector_pairs(sensor_positions_torch, max_dist)

sensitivities = cw_sensitivity_batched(
    grid_points_torch, sources, detectors, mu_eff, batch_size=500
)
print(f"Matrix shape: {sensitivities.shape}")

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
)

# %%
# # check sensor positions
# from guti.viz import visualize_grid_and_sensors

# fig = visualize_grid_and_sensors(
#     grid_points_mm,
#     sensor_positions_mm,
# )


# %%
