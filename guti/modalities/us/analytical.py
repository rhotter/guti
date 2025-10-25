# WOO
# %%

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from guti.data_utils import save_svd
from guti.modalities.us.utils import create_medium, create_sources, create_receivers, simulate_free_field_propagation
import time

# %%

import argparse

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=4000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=4000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
# parser.add_argument('--sensor_batch_size', type=int, default=256, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--sensor_batch_size', type=int, default=128, help='Batch size across sensors for Gram accumulation')

args = parser.parse_args()

n_sources = args.n_sources
n_sensors = args.n_sensors
temporal_sampling = args.temporal_sampling
sensor_batch_size = args.sensor_batch_size

print(f"n_sources: {n_sources}, n_sensors: {n_sensors}, temporal_sampling: {temporal_sampling}, sensor_batch_size: {sensor_batch_size}")

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

# center_frequency = 1.5e6
center_frequency = 0.3e6

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

# device for propagation (and potentially accumulation)
device = "cuda"
# device = "cpu"

temporal_sampling = 5

use_complex_ampitudes = False

# %%

print("Computing SVD (batched simulation + Gram accumulation)...")

num_sensors_total = sensor_positions.shape[0]
num_sources_total = n_sources
print(f"num_sensors: {num_sensors_total}, num_sources: {num_sources_total}, device: {device}")

# Pre-build constant tensors on propagation device to avoid repeated transfers
source_positions_t = torch.tensor(source_positions, device=device)
source_signals_t = torch.tensor(source_signals, device=device)
voxel_size_t = torch.tensor(voxel_size, device=device)

# Accumulate Gram matrix in batches over sensors to reduce memory and avoid storing full matrix
t0 = time.perf_counter()
G = torch.zeros((num_sources_total, num_sources_total), dtype=torch.float32, device=device)
for start in range(0, num_sensors_total, sensor_batch_size):
    print(f"Processing batch {start // sensor_batch_size + 1} of {num_sensors_total // sensor_batch_size + 1}")
    end = min(start + sensor_batch_size, num_sensors_total)
    receiver_positions_t = torch.tensor(sensor_positions[start:end], device=device)

    # Simulate for this sensor batch
    pf_chunk = simulate_free_field_propagation(
        source_positions_t,
        receiver_positions_t,
        source_signals_t,
        time_step,
        center_frequency,
        voxel_size_t,
        device=device,
        compute_time_series=not use_complex_ampitudes,
        temporal_sampling=temporal_sampling
    )

    # Convert to chunk matrix [rows, n_sources]
    if use_complex_ampitudes:
        # Complex propagator, split into real/imag parts along rows
        chunk_matrix = torch.cat([pf_chunk.real, pf_chunk.imag], dim=0).float()
    else:
        # Time series, flatten receiver/time dims
        chunk_matrix = pf_chunk.reshape(-1, num_sources_total).float()

    # Accumulate into Gram matrix on compute device
    chunk_matrix = chunk_matrix.to(device, non_blocking=True)
    G += chunk_matrix.T @ chunk_matrix

if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"gram_accumulate: {t1 - t0:.3f}s")

# breakpoint()

t0 = time.perf_counter()
G = G.cpu()
s = torch.linalg.svdvals(G)
if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"svd_total: {t1 - t0:.3f}s")

s = torch.sqrt(s)
s = s.cpu().numpy()
print("Done!")

# %%

plt.semilogy(s)
ax = plt.gca()
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value")
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True, min_n_ticks=10))
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=30))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=30))
log_formatter = mticker.LogFormatter(base=10.0, labelOnlyBase=False)
ax.yaxis.set_major_formatter(log_formatter)
ax.yaxis.set_minor_formatter(log_formatter)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()

from guti.data_utils import Parameters

save_svd(s, 'us_analytical', params=Parameters(
    num_sensors=len(sensor_positions),
    grid_resolution_mm=None,
    num_brain_grid_points=len(source_positions),
    time_resolution=time_step,
    comment=None
# ), subdir="us_free_field_analytical")
# ), subdir="us_free_field_analytical_n_sources_sweep")
), subdir="us_free_field_analytical_n_sources_sweep_300khz")

# %%
