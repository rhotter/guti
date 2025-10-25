# WOO
# %%

import torch
import numpy as np

import matplotlib.pyplot as plt
from guti.data_utils import save_svd
from guti.modalities.us.utils import create_medium, create_sources, create_receivers, simulate_free_field_propagation

# %%

import argparse

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=4000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=4000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')

args = parser.parse_args()

n_sources = args.n_sources
n_sensors = args.n_sensors
temporal_sampling = args.temporal_sampling

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

center_frequency = 1.5e6
sources, source_mask = create_sources(domain, time_axis, freq_Hz=1e6, n_sources=n_sources, pad=0, inside=True)
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=center_frequency, n_sensors=n_sensors, pad=0)

print(f"Number of sources: {len(sources.positions)}")
print(f"Number of sensors: {len(sensors.positions)}")

# %%

source_positions = np.stack([np.array(x) for x in sources.positions]).T

sensor_positions = np.stack([np.array(x) for x in sensors.positions]).T

n_sources = source_positions.shape[0]

time_step = 1e-1 / center_frequency
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
source_signals = np.tile(source_signals, (n_sources, 1))

voxel_size = np.array(domain.dx)

print(f"Coinciding source and sensor positions: {torch.argwhere(torch.all(torch.tensor(source_positions).unsqueeze(0) == torch.tensor(sensor_positions).unsqueeze(1), dim=-1))}")

#%%

# device = "cuda"
device = "cpu"

temporal_sampling = 5

use_complex_ampitudes = False

pressure_field = simulate_free_field_propagation(
    torch.tensor(source_positions).to(device),
    torch.tensor(sensor_positions).to(device),
    torch.tensor(source_signals).to(device),
    time_step,
    center_frequency,
    torch.tensor(voxel_size).to(device),
    device=device,
    compute_time_series=not use_complex_ampitudes,
    temporal_sampling=temporal_sampling
) # [n_sensors, n_sources]

# %%

pressure_field = pressure_field.reshape(-1, n_sources)

# IF WE WANTED TO COMPUTE PHASES
if use_complex_ampitudes:
    matrix = torch.cat([pressure_field.real, pressure_field.imag], dim=0).float()
else:
    matrix = pressure_field.float()  # already [n_receivers, n_sources]

# %%

# maybe can use something like this to simulate Born's approximation (see https://ausargeo.com/deepwave/scalar_born)
# matrix = matrix * matrix

matrix = matrix.cuda()

smaller_matrix = matrix.T @ matrix

s = torch.linalg.svdvals(smaller_matrix)
s = torch.sqrt(s)
s = s.cpu().numpy()

# %%

plt.semilogy(s)

from guti.data_utils import Parameters

save_svd(s, 'us_analytical', params=Parameters(
    num_sensors=len(sensor_positions),
    grid_resolution_mm=None,
    num_brain_grid_points=len(source_positions),
    time_resolution=time_step,
    comment=None
))

# %%
