"""
Simulation of ultrasound propagation in a free field, using the analytical fundamental solution (Green's function).
We treat the "independent variables" in ultrasound imaging as sources. This relies on the approximation that the intensity of the transmit pulse is the same at each point in the medium, which is related to the Born approximation.
"""

# %%

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from guti.data_utils import save_svd
from guti.modalities.us.utils import create_medium, create_sources_real, create_receivers_real, simulate_free_field_propagation, plot_medium
import time

import torch, torch.backends.cuda as cu
torch.set_float32_matmul_precision('high')  # allow TF32 on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
cu.preferred_linalg_library("magma")        # robust & fast dense LA

# %%

import argparse

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=4000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=1000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
parser.add_argument('--sensor_batch_size', type=int, default=256, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--center_frequency', type=float, default=0.05e6, help='Center frequency in Hz')
parser.add_argument('--accumulate_on_cpu', action='store_true', help='Accumulate Gram matrix on CPU instead of GPU')
parser.add_argument('--svd_device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to compute eigenvalues/SVD of Gram')

args = parser.parse_args()

n_sources = args.n_sources
n_sensors = args.n_sensors
temporal_sampling = args.temporal_sampling
sensor_batch_size = args.sensor_batch_size
svd_device = args.svd_device
center_frequency = args.center_frequency

print(f"n_sources: {n_sources}, n_sensors: {n_sensors}, temporal_sampling: {temporal_sampling}, sensor_batch_size: {sensor_batch_size}")

# We create a jwave medium object. This is mostly useful for non-free field simulations, but we use it here for convenience/consistency.
domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium(central_frequency=center_frequency, pad=30)

# Create the source and receiver positions in real space (meters).
source_positions = create_sources_real(domain, time_axis, freq_Hz=center_frequency, n_sources=n_sources, inside=True, pad=30)
sensor_positions = create_receivers_real(domain, time_axis, freq_Hz=center_frequency, n_sensors=n_sensors, pad=30)

n_sources = source_positions.shape[0]

# Continuous wave signals
time_step = 1e-1 / center_frequency
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
source_signals = np.tile(source_signals, (n_sources, 1))

nt = time_axis.shape[0]//temporal_sampling + 1
voxel_size = np.array(domain.dx)

#%%

# device for propagation (and potentially accumulation)
device = "cuda"

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

# Accumulate Gram or do TSQR in batches over sensors
t0 = time.perf_counter()
G = None
R_acc = None
gram_via_tiling = False
G = torch.zeros((num_sensors_total * nt, num_sources_total), dtype=torch.float32, device="cuda")
k = 10  # Number of chunks to accumulate on GPU before transferring to CPU
gpu_chunks = []
gpu_start_idx = 0
last_index = 0
print(f"num_sensors_total: {num_sensors_total}, nt: {nt}")
for start in range(0, num_sensors_total, sensor_batch_size):
    print(f"Processing batch {start // sensor_batch_size + 1} of {(num_sensors_total + sensor_batch_size - 1) // sensor_batch_size}")
    end = min(start + sensor_batch_size, num_sensors_total)
    print(f"start: {start}, end: {end}")
    receiver_positions_t = torch.tensor(sensor_positions[start:end], device=device)
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
    # Build chunk matrix [rows, n_sources]
    if use_complex_ampitudes:
        chunk_matrix = torch.cat([pf_chunk.real, pf_chunk.imag], dim=0).float()
    else:
        # ensure receivers/time are flattened to rows, sources are columns
        chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, num_sources_total).float()
    chunk_rows = chunk_matrix.shape[0]
    print(f"last_index: {last_index}, chunk_rows: {chunk_rows}")
    G[last_index:last_index+chunk_rows] = chunk_matrix
    last_index += chunk_rows

if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"gram_accumulate: {t1 - t0:.3f}s")


t0 = time.perf_counter()

# if G.shape[1] > G.shape[0]:
#     G = G @ G.T
# else:
#     G = G.T @ G

# s = torch.linalg.svdvals(G)
# s = torch.sqrt(s)

s = torch.linalg.svdvals(G)


s = s.cpu().numpy()
print("Done!")

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
from guti.core import get_bitrate, noise_floor_heuristic

from guti.data_utils import save_svd

save_svd(s, f'us_free_field_analytical_n_sources_sweep_{int(center_frequency/1e3)}khz', params=Parameters(
    num_sensors=len(sensor_positions),
    num_brain_grid_points=len(source_positions),
    time_resolution=time_step,
    vincent_trick=False
))

s_normalized = s / len(source_positions)**0.5

noise_level = noise_floor_heuristic(s_normalized, heuristic="power")

print(f"noise_level: {noise_level}")
print(f"bitrate: {get_bitrate(s_normalized, noise_level, time_resolution=1.0)}")

exit(0)

# @torch.no_grad()
# def bitrate_slq_torch(
#     apply_A, apply_AT, m, n,
#     noise_std_full_brain: float,
#     time_resolution: float = 1.0,
#     n_sources: int = 1,
#     n_detectors: int = 1,
#     s: int = 16, t: int = 40,
#     device: str = "cuda",
#     dtype = torch.float64,
# ):
#     ln2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))

#     # noise_var_eff = (noise_std_full_brain / (n_eff))**2
#     noise_var_eff = (noise_std_full_brain * (n_detectors**0.5) / (n_sources**0.5))**2
#     alpha = torch.tensor(1.0 / noise_var_eff, dtype=dtype, device=device)

#     use_left = (m <= n)
#     d = m if use_left else n

#     def apply_B(v):
#         return apply_A(apply_AT(v)) if use_left else apply_AT(apply_A(v))

#     est = torch.zeros((), dtype=dtype, device=device)

#     for _ in range(s):
#         # Rademacher probe
#         z = (torch.randint(0, 2, (d,), device=device, dtype=torch.int8) * 2 - 1).to(dtype)
#         norm_z = torch.linalg.vector_norm(z)
#         q = z / norm_z
#         q_prev = torch.zeros_like(q)

#         alphas = torch.zeros(t, dtype=dtype, device=device)
#         betas  = torch.zeros(t-1, dtype=dtype, device=device)
#         t_eff = t

#         for k in range(t):
#             w = apply_B(q)
#             if k > 0:
#                 w = w - betas[k-1] * q_prev
#             alpha_k = torch.dot(q, w)
#             w = w - alpha_k * q
#             alphas[k] = alpha_k
#             if k < t-1:
#                 beta_k = torch.linalg.vector_norm(w)
#                 betas[k] = beta_k
#                 if beta_k == 0:
#                     t_eff = k+1
#                     alphas = alphas[:t_eff]
#                     betas  = betas[:t_eff-1]
#                     break
#                 q_prev, q = q, (w / beta_k)

#         # Move tiny T to CPU for eigh (or use torch.linalg.eigh on GPU; both are fine)
#         T = torch.diag(alphas)
#         if t_eff > 1:
#             T += torch.diag(betas, 1) + torch.diag(betas, -1)
#         evals, evecs = torch.linalg.eigh(T)
#         weights = evecs[0, :]**2
#         quad = torch.dot(weights, torch.log1p(alpha * evals))
#         est += (norm_z**2) * quad

#     bits_per_sample = est / s / ln2
#     return (bits_per_sample / time_resolution).item()


# # Dense convenience wrapper
# @torch.no_grad()
# def bitrate_slq_dense_torch(A: torch.Tensor, noise_std_full_brain, time_resolution=1.0, n_sources=1, n_detectors=1, s=16, t=40):
#     m, n = A.shape
#     return bitrate_slq_torch(
#         apply_A=lambda x: (A @ x.to(A.dtype)).to(torch.float64),
#         apply_AT=lambda x: (A.T @ x.to(A.dtype)).to(torch.float64),
#         m=m, n=n,
#         noise_std_full_brain=noise_std_full_brain,
#         time_resolution=time_resolution,
#         n_sources=n_sources,
#         n_detectors=n_detectors,
#         s=s, t=t,
#         device=str(A.device),
#         dtype=torch.float64,
#     )


# # bitrate = bitrate_slq_dense_torch(G, noise_std_full_brain=1.0, time_resolution=time_step, n_detectors=n_sensors)
# bitrate = bitrate_slq_dense_torch(G, noise_std_full_brain=noise_level, time_resolution=1.0, n_sources=len(source_positions), n_detectors=len(sensor_positions))
# print(f"bitrate: {bitrate}")