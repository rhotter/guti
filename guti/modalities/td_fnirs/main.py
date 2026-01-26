# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

from guti.core import get_grid_positions, get_sensor_positions

import numpy as np
import torch
from guti.modalities.td_fnirs.utils import (
    td_sensitivity_batched,
    get_valid_source_detector_pairs,
)

# %%
# Parameters
grid_spacing_mm = 6.0
noptodes = 800
max_dist = 50  # mm
time_gates_ns = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # nanoseconds

# Optical properties (typical brain tissue at ~800nm)
mu_a = 0.02  # mm^-1 (absorption)
mu_s_prime = 0.67  # mm^-1 (reduced scattering)
n_tissue = 1.4
c_vacuum = 299.792  # mm/ns
c = c_vacuum / n_tissue  # speed of light in tissue [mm/ns]
D = 1 / (3 * (mu_a + mu_s_prime))  # diffusion coefficient [mm]

# %%
# Setup geometry
grid_points_mm = get_grid_positions(grid_spacing_mm)
sensor_positions_mm = get_sensor_positions(noptodes)

# Convert to torch tensors and move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_points_torch = torch.from_numpy(grid_points_mm).float().to(device)
sensor_positions_torch = torch.from_numpy(sensor_positions_mm).float().to(device)

print(f"Grid points: {grid_points_torch.shape[0]}")
print(f"Sensors: {sensor_positions_torch.shape[0]}")

# %%
# Get valid source-detector pairs
sources, detectors = get_valid_source_detector_pairs(sensor_positions_torch, max_dist)
print(f"Valid S-D pairs: {sources.shape[0]}")

# %%
# Compute TD sensitivity for each time gate
all_sensitivities = []
for t_ns in time_gates_ns:
    print(f"\nComputing sensitivity for t = {t_ns} ns...")
    sensitivities = td_sensitivity_batched(
        pos=grid_points_torch,
        source_pos=sources,
        detector_pos=detectors,
        t=t_ns,
        D=D,
        mu_a=mu_a,
        c=c,
        batch_size=500,
    )
    all_sensitivities.append(sensitivities)
    print(f"  Shape: {sensitivities.shape}")

# %%
# Stack and reshape: (n_gates, n_pairs, n_points) -> (n_pairs * n_gates, n_points)
stacked = torch.stack(all_sensitivities, dim=0)
stacked = stacked.permute(1, 0, 2)  # (n_pairs, n_gates, n_points)
jacobian = stacked.reshape(-1, grid_points_torch.shape[0])
print(f"\nFinal Jacobian shape: {jacobian.shape}")

# %%
from guti.svd import compute_svd_gpu

s = compute_svd_gpu(jacobian)

# %%
from guti.svd import plot_svd

plot_svd(s)

# %%
from guti.data_utils import save_svd, Parameters

save_svd(
    s,
    "td_fnirs_analytical",
    Parameters(
        num_sensors=noptodes,
        grid_resolution_mm=grid_spacing_mm,
        num_brain_grid_points=grid_points_torch.shape[0],
    ),
)

# %%
# Alternative: use the modality class directly
from guti.modalities.td_fnirs import TDfNIRSAnalytical
from guti.parameters import Parameters

modality = TDfNIRSAnalytical(
    params=Parameters(
        num_sensors=800,
        grid_resolution_mm=6.0,
        max_dist=50.0,
    ),
    time_gates_ns=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
)
singular_values = modality.run()

# %%
