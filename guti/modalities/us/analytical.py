# WOO
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


domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium(central_frequency=center_frequency, pad=30)


# sources, source_mask = create_sources(domain, time_axis, freq_Hz=center_frequency, n_sources=n_sources, inside=True, pad=30)
# sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=center_frequency, n_sensors=n_sensors, pad=30)

sources = create_sources_real(domain, time_axis, freq_Hz=center_frequency, n_sources=n_sources, inside=True, pad=30)
sensors = create_receivers_real(domain, time_axis, freq_Hz=center_frequency, n_sensors=n_sensors, pad=30)

# %%

# Plot medium, sources, and receivers before overwriting time_axis
# plot_medium(medium_original, source_mask, sources, time_axis, receivers_mask)

# source_positions = np.stack([np.array(x) for x in sources.positions]).T
# sensor_positions = np.stack([np.array(x) for x in sensors.positions]).T

# breakpoint()

source_positions = sources
sensor_positions = sensors

n_sources = source_positions.shape[0]

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
# G = torch.zeros((num_sensors_total * nt, num_sources_total), dtype=torch.float32, device="cpu")
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
    # if gram_via_tiling:
    #     # handled in tiling branch above
    #     pass
    # else:
    #     # Store chunk on GPU temporarily
    #     gpu_chunks.append(chunk_matrix)
        
    #     # Transfer to CPU every k chunks
    #     if len(gpu_chunks) >= k:
    #         stacked_chunks = torch.cat(gpu_chunks, dim=0).cpu()
    #         chunk_rows = stacked_chunks.shape[0]
    #         G[gpu_start_idx:gpu_start_idx+chunk_rows] = stacked_chunks
    #         gpu_start_idx += chunk_rows
    #         gpu_chunks = []
    chunk_rows = chunk_matrix.shape[0]
    print(f"last_index: {last_index}, chunk_rows: {chunk_rows}")
    G[last_index:last_index+chunk_rows] = chunk_matrix
    last_index += chunk_rows

# # Transfer any remaining chunks to CPU
# if gpu_chunks:
#     stacked_chunks = torch.cat(gpu_chunks, dim=0).cpu()
#     chunk_rows = stacked_chunks.shape[0]
#     G[gpu_start_idx:gpu_start_idx+chunk_rows] = stacked_chunks

if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"gram_accumulate: {t1 - t0:.3f}s")

# breakpoint()

t0 = time.perf_counter()

# qr_device = 'cuda' if (svd_device == 'cuda' and torch.cuda.is_available()) else 'cpu'
# R_final = None
# for start in range(0, num_sensors_total, sensor_batch_size):
#     print(f"Processing batch {start // sensor_batch_size + 1} of {(num_sensors_total + sensor_batch_size - 1) // sensor_batch_size}")
#     end = min(start + sensor_batch_size, num_sensors_total)
#     Ai = G[start*nt:end*nt].to(qr_device)
#     _, Ri = torch.linalg.qr(Ai, mode='r')
#     if R_final is None:
#         R_final = Ri
#     else:
#         _, R_final = torch.linalg.qr(torch.cat([R_final, Ri], dim=0), mode='r')
#     del Ri, Ai
#     if qr_device == 'cuda':
#         torch.cuda.empty_cache()
# print("Computing SVD...")
# s = torch.linalg.svdvals(R_final)
# torch.cuda.synchronize()

# t1 = time.perf_counter()
# print(f"eig_total: {t1 - t0:.3f}s")

# # %%

G = G @ G.T
# G = G.T @ G

s = torch.linalg.svdvals(G)
s = torch.sqrt(s)

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
))

s_normalized = s / len(source_positions)**0.5

noise_level = noise_floor_heuristic(s_normalized, heuristic="power")

print(f"noise_level: {noise_level}")
print(f"bitrate: {get_bitrate(s_normalized, noise_level, time_resolution=1.0, n_sources=len(source_positions), n_detectors=len(sensor_positions))}")

exit(0)

# %%

# import torch

# import math
# import torch

# # -----------------------
# # Streamed operator (CPU A ➜ GPU tiles)
# # -----------------------
# class StreamedOp:
#     """
#     Apply y = A @ X and Z = A.T @ Y with A on CPU, streaming row tiles to GPU.
#     - A_cpu: torch.Tensor on CPU with shape (m, n)
#     - device: 'cuda' or specific CUDA device
#     - tile_rows: number of rows per streamed tile; if None, auto-choose from free GPU memory
#     - compute_dtype: matmul dtype on GPU (fp32 usually fastest; TF32 used automatically on Ampere if allowed)
#     - acc_dtype: accumulation dtype for stability (fp64 recommended)
#     - pin: if True, keep a pinned copy of A (uses extra RAM but speeds H2D)
#     """
#     def __init__(self, A_cpu: torch.Tensor, device="cuda", tile_rows=None,
#                  compute_dtype=torch.float32, acc_dtype=torch.float64, pin=False, mem_frac=0.5):
#         assert A_cpu.device.type == "cpu"
#         self.A_cpu = A_cpu.contiguous()
#         self.m, self.n = self.A_cpu.shape
#         self.device = torch.device(device)
#         self.compute_dtype = compute_dtype
#         self.acc_dtype = acc_dtype
#         self.pin = pin
#         if pin:
#             # Pinned copy for faster H2D (uses extra RAM)
#             self.A_cpu = self.A_cpu.pin_memory()

#         if tile_rows is None and self.device.type == "cuda":
#             free_b, total_b = torch.cuda.mem_get_info(self.device)
#             # Very rough upper bound: keep (mem_frac * free) for A_tile
#             bytes_per = torch.tensor([], dtype=compute_dtype).element_size()
#             # Reserve room for X/Y and overhead; conservative choice:
#             max_rows = int((mem_frac * free_b) / max(self.n * bytes_per, 1))
#             tile_rows = max(512, min(self.m, max_rows))
#         self.tile_rows = int(tile_rows) if tile_rows is not None else 4096

#     @torch.no_grad()
#     def A_mv(self, X_dev: torch.Tensor) -> torch.Tensor:
#         """
#         Y = A @ X, X_dev shape (n, r) or (n,), returns Y on self.device with shape (m, r).
#         Always moves/casts X to self.device/compute_dtype to avoid device mismatch.
#         """
#         if X_dev.ndim == 1:
#             X_dev = X_dev[:, None]
#         # Move & cast; this is a no-op if already correct
#         Xc = X_dev.to(self.device, dtype=self.compute_dtype, non_blocking=True)
#         n, r = Xc.shape
#         assert n == self.n, f"X has {n} rows but A has n={self.n}"
#         Y = torch.empty((self.m, r), device=self.device, dtype=self.acc_dtype)
#         for i in range(0, self.m, self.tile_rows):
#             ib = slice(i, min(i + self.tile_rows, self.m))
#             A_blk = self.A_cpu[ib, :].to(self.device, dtype=self.compute_dtype, non_blocking=True)
#             # GEMM
#             Y_blk = (A_blk @ Xc).to(self.acc_dtype)
#             Y[ib, :] = Y_blk
#             del A_blk, Y_blk
#         return Y[:, 0] if Y.shape[1] == 1 else Y

#     @torch.no_grad()
#     def AT_mv(self, Y_dev: torch.Tensor) -> torch.Tensor:
#         """
#         Z = A.T @ Y, Y_dev shape (m, r) or (m,), returns Z on self.device with shape (n, r).
#         Always moves/casts Y to self.device/compute_dtype.
#         """
#         if Y_dev.ndim == 1:
#             Y_dev = Y_dev[:, None]
#         Yc = Y_dev.to(self.device, dtype=self.compute_dtype, non_blocking=True)
#         m, r = Yc.shape
#         assert m == self.m, f"Y has {m} rows but A has m={self.m}"
#         Z = torch.zeros((self.n, r), device=self.device, dtype=self.acc_dtype)
#         for i in range(0, self.m, self.tile_rows):
#             ib = slice(i, min(i + self.tile_rows, self.m))
#             A_blk = self.A_cpu[ib, :].to(self.device, dtype=self.compute_dtype, non_blocking=True)
#             Y_blk = Yc[ib, :]
#             Z = Z + (A_blk.T @ Y_blk).to(self.acc_dtype)
#             del A_blk
#         return Z[:, 0] if Z.shape[1] == 1 else Z

#     @torch.no_grad()
#     def B_mv(self, Q_dev: torch.Tensor, left=True) -> torch.Tensor:
#         """
#         Apply B to Q on the correct side, coercing Q to self.device for safety.
#         left=True:  B = A A^T, Q in R^{m×b}
#         left=False: B = A^T A, Q in R^{n×b}
#         """
#         Q_dev = Q_dev.to(self.device, dtype=self.acc_dtype, non_blocking=True)
#         if left:
#             # R^{m×b} -> R^{m×b}
#             return self.A_mv(self.AT_mv(Q_dev))
#         else:
#             # R^{n×b} -> R^{n×b}
#             return self.AT_mv(self.A_mv(Q_dev))


# # -----------------------
# # Batched SLQ (independent Lanczos per probe, vectorized)
# # -----------------------
# @torch.no_grad()
# def bitrate_slq_streamed_torch(
#     A_cpu: torch.Tensor,
#     noise_std_full_brain: float,
#     time_resolution: float = 1.0,
#     n_detectors: int | None = None,
#     s: int = 32,          # total number of Hutchinson probes
#     t: int = 40,          # Lanczos steps per probe
#     batch: int = 8,       # probes processed simultaneously (GEMM-friendly)
#     device: str = "cuda",
#     compute_dtype = torch.float32,
#     acc_dtype = torch.float64,
#     pin: bool = False,
#     use_left_if_smaller: bool = True,  # use B=AA^T if m<=n else B=A^T A
#     tile_rows: int | None = None,
# ):
#     """
#     Fast SLQ estimate of bits/sec with CPU-resident A (torch tensor), streamed to GPU tiles.
#     """
#     m, n = A_cpu.shape
#     left = (m <= n) if use_left_if_smaller else True
#     d = m if left else n

#     # α = 1 / noise_var_eff, noise_var_eff = (noise / sqrt(n_eff))^2
#     n_eff = n_detectors or (n if left else m)
#     noise_var_eff = (noise_std_full_brain / math.sqrt(n_eff))**2
#     alpha = torch.tensor(1.0 / noise_var_eff, dtype=acc_dtype, device=device)
#     ln2 = torch.log(torch.tensor(2.0, dtype=acc_dtype, device=device))

#     # Operator
#     op = StreamedOp(A_cpu, device=device, tile_rows=tile_rows,
#                     compute_dtype=compute_dtype, acc_dtype=acc_dtype, pin=pin)

#     # Probe loop in batches
#     est_sum = torch.zeros((), dtype=acc_dtype, device=device)
#     num_done = 0
#     while num_done < s:
#         b = min(batch, s - num_done)

#         # Rademacher probes in R^d
#         z = (torch.randint(0, 2, (d, b), device=device, dtype=torch.int8) * 2 - 1).to(acc_dtype)
#         normz2 = torch.sum(z*z, dim=0)  # (b,)
#         Q = (z / torch.sqrt(normz2)[None, :]).to(acc_dtype)  # columns normalized

#         # Store diagonal/off-diagonals of each probe's tridiagonal (alphas, betas)
#         alphas = torch.zeros((t, b), dtype=acc_dtype, device=device)
#         betas  = torch.zeros((t-1, b), dtype=acc_dtype, device=device)
#         Q_prev = torch.zeros_like(Q)

#         # Lanczos
#         for k in range(t):
#             W = op.B_mv(Q, left=left)                 # (d, b)
#             if k > 0:
#                 W = W - Q_prev * betas[k-1, :][None, :]
#             alpha_k = torch.sum(Q * W, dim=0)         # (b,)
#             W = W - Q * alpha_k[None, :]
#             alphas[k, :] = alpha_k
#             if k < t-1:
#                 beta_k = torch.linalg.vector_norm(W, dim=0)  # (b,)
#                 betas[k, :] = beta_k
#                 # Avoid divide-by-zero for any converged column
#                 mask = (beta_k > 0)
#                 Q_next = torch.zeros_like(Q)
#                 Q_next[:, mask] = (W[:, mask] / beta_k[None, mask])
#                 Q_prev, Q = Q, Q_next

#         # Tiny tridiagonals → eigenpairs → e1^T f(T) e1 per column
#         # We do them in a short loop (b is small).
#         batch_contrib = torch.zeros((), dtype=acc_dtype, device=device)
#         e1 = None
#         for j in range(b):
#             tj = t
#             # If a column hit beta=0 early, you *could* shrink tj; we keep tj for simplicity.
#             T = torch.zeros((tj, tj), dtype=acc_dtype, device=device)
#             T.fill_(0)
#             T.diagonal(0).copy_(alphas[:tj, j])
#             if tj > 1:
#                 T.diagonal(1).copy_(betas[:tj-1, j])
#                 T.diagonal(-1).copy_(betas[:tj-1, j])
#             evals, evecs = torch.linalg.eigh(T)
#             if e1 is None or e1.shape[0] != tj:
#                 e1 = torch.zeros((tj,), dtype=acc_dtype, device=device); e1[0] = 1.0
#             weights = (evecs[0, :])**2
#             quad = torch.dot(weights, torch.log1p(alpha * evals))
#             batch_contrib += normz2[j] * quad

#         est_sum += batch_contrib
#         num_done += b

#     bits_per_sample = est_sum / (s * ln2)
#     return float(bits_per_sample / time_resolution)

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

# # Example
# torch.set_float32_matmul_precision('high')  # enable TF32 on Ampere+ for fp32 matmuls

# noise_std = 1.0
# bps = bitrate_slq_streamed_torch(
#     G,
#     noise_std_full_brain=noise_std,
#     time_resolution=1.0,
#     n_detectors=None,   # or set this
#     s=32, t=40, batch=4096,
#     device="cuda",
#     compute_dtype=torch.float32,  # fastest; accumulates in float64
#     acc_dtype=torch.float64,
#     pin=False,          # set True if you can afford a pinned copy of A
#     tile_rows=None      # auto-choose from free GPU mem
# )
# print("bits/sec ~", bps)


# import os, math, torch

# # Use all CPU cores; let MKL/oneDNN do the heavy lifting
# torch.set_num_threads(os.cpu_count() or 8)

# @torch.no_grad()
# def bitrate_slq_cpu(
#     A_cpu: torch.Tensor,                 # (m,n) on CPU, contiguous
#     noise_std_full_brain: float,
#     time_resolution: float = 1.0,
#     n_detectors: int | None = None,
#     s: int = 16,                         # probes (↑ => lower MC noise)
#     t: int = 40,                         # Lanczos steps (↑ => lower bias)
#     batch: int = 32,                     # probes processed together (turns GEMV -> GEMM)
#     use_left_if_smaller: bool = True,    # use B = A A^T if m <= n else A^T A
#     dtype = torch.float64,               # keep Krylov scalars in fp64
# ):
#     assert A_cpu.device.type == "cpu"
#     A = A_cpu.contiguous()
#     m, n = A.shape
#     left = (m <= n) if use_left_if_smaller else True
#     d = m if left else n

#     n_eff = n_detectors or (n if left else m)
#     alpha = 1.0 / (noise_std_full_brain**2 / n_eff)
#     ln2 = math.log(2.0)

#     def B_mv(Q):                 # Q: (d,b) on CPU
#         if left:                 # B = A A^T
#             T = A.T @ Q          # (n,b)
#             return A @ T         # (m,b)
#         else:                    # B = A^T A
#             T = A @ Q            # (m,b)
#             return A.T @ T       # (n,b)

#     est = 0.0
#     done = 0
#     while done < s:
#         b = min(batch, s - done)
#         print(f"Processing probes {done+1}-{done+b}/{s}...")

#         # Rademacher probes, normalized
#         z = (torch.randint(0, 2, (d, b)) * 2 - 1).to(dtype)
#         norm2 = torch.sum(z*z, dim=0)                 # (b,)
#         Q = z / torch.sqrt(norm2)[None, :]            # (d,b)
#         Qm1 = torch.zeros_like(Q)

#         al = torch.zeros((t, b), dtype=dtype)         # diag blocks
#         be = torch.zeros((t-1, b), dtype=dtype)       # off-diag

#         # Lanczos iteration: build tridiagonal matrix T such that Q^T B Q ≈ T
#         # where B = A A^T (or A^T A). Each iteration extends the Krylov subspace.
#         for k in range(t):
#             W = B_mv(Q)                               # apply B to current basis vectors
#             if k > 0:
#                 W -= Qm1 * be[k-1][None, :]           # orthogonalize against previous basis
#             ak = torch.sum(Q * W, dim=0)              # diagonal entry: <Q, B Q>
#             W -= Q * ak[None, :]                      # orthogonalize against current basis
#             al[k] = ak                                # store diagonal
#             if k < t-1:
#                 bk = torch.linalg.vector_norm(W, dim=0)  # off-diagonal (coupling strength)
#                 be[k] = bk                            # store off-diagonal
#                 mask = bk > 0                         # avoid division by zero
#                 Qm1, Q = Q, torch.where(mask[None, :], W / bk[None, :], Q)  # normalize new basis

#         # after the Lanczos loop with arrays:
#         #   al: shape (t_max, b)
#         #   be: shape (t_max-1, b)   # betas (off-diagonals)

#         # Process each probe vector independently to compute its contribution to the log-determinant estimate
#         print(f"Processing probes {done+1}-{done+b}/{s}...")
#         for j in range(b):
#             # Build the tridiagonal matrix Tj from the Lanczos coefficients for this probe.
#             # The effective size tj is determined by how many Lanczos steps were numerically stable
#             # (i.e., how many off-diagonal entries are nonzero before breakdown).
#             nz = (be[:, j] != 0)
#             tj = 1 + int(nz.sum().item())  # tj ∈ [1, t_max]
#             Tj = torch.zeros((tj, tj), dtype=dtype)

#             # main diagonal: alpha coefficients from Lanczos
#             Tj.diagonal(0).copy_(al[:tj, j])

#             # off diagonals: beta coefficients (symmetric tridiagonal)
#             if tj > 1:
#                 off = be[:tj-1, j]
#                 Tj.diagonal( 1).copy_(off)
#                 Tj.diagonal(-1).copy_(off)

#             # Compute eigendecomposition of the tridiagonal matrix.
#             # The eigenvalues approximate the Ritz values of B in the Krylov subspace.
#             evals, evecs = torch.linalg.eigh(Tj)
            
#             # Weight by the squared first component of each eigenvector (Gauss quadrature weights)
#             # and accumulate the stochastic trace estimate: tr(log(I + alpha*B)) ≈ sum_i w_i * log(1 + alpha*lambda_i)
#             w1 = evecs[0, :]**2
#             est += norm2[j].item() * torch.dot(w1, torch.log1p(alpha * evals)).item()

#         done += b

#     bits_per_sample = est / (s * ln2)
#     return bits_per_sample / time_resolution

# bitrate = bitrate_slq_cpu(
#     G.T.float(), 
#     noise_std_full_brain=1.0, 
#     time_resolution=1.0, 
#     n_detectors=n_sensors,
#     s=16,
#     t=40,
#     batch=8,
#     use_left_if_smaller=True,
#     dtype=torch.float32
# )
# print(f"bitrate: {bitrate}")


# import math, os, torch
# torch.set_num_threads(os.cpu_count() or 192)

# @torch.no_grad()
# def bitrate_countsketch(
#     A_cpu: torch.Tensor,                 # (m,n) CPU contiguous, fp32 recommended
#     noise_std_full_brain: float,
#     time_resolution: float = 1.0,
#     n_detectors: int | None = None,
#     ell: int = 4096,                     # sketch dimension (try 2048–8192)
#     block_rows: int = 65536,             # row block for cache-friendly index_add_
#     seed: int = 0,
#     compute_dtype = torch.float32,
#     acc_dtype = torch.float64,
#     use_gpu_chol: bool = False,          # set True to run the tiny Cholesky on GPU
#     device: str = "cuda",
# ):
#     assert A_cpu.device.type == "cpu"
#     A = A_cpu.contiguous().to(compute_dtype, copy=False)
#     m, n = A.shape
#     n_eff = n_detectors or n
#     alpha = torch.tensor(1.0 / (noise_std_full_brain**2 / (n_eff)), dtype=acc_dtype)

#     # CountSketch: random bucket map h:[0..m-1]→[0..ell-1], and signs s∈{±1}^m
#     g = torch.Generator(device="cpu").manual_seed(seed)
#     h = torch.randint(0, ell, (m,), generator=g)                 # long
#     s = (torch.randint(0, 2, (m,), generator=g)*2 - 1).to(compute_dtype)  # ±1

#     # Build SA = S @ A in one pass over rows (CPU)
#     SA = torch.zeros((ell, n), dtype=compute_dtype)
#     for i in range(0, m, block_rows):
#         ib = slice(i, min(i+block_rows, m))
#         # SA[h[i]] += s[i] * A[i]
#         SA.index_add_(0, h[ib], s[ib].unsqueeze(1) * A[ib, :])
    
#     print("Built SA")

#     # Form small SPD: Sℓ = I + α (SA @ SA^T)  (on GPU for speed)
#     if use_gpu_chol:
#         SA = SA.to(device)
#     S_small = torch.eye(ell, dtype=acc_dtype, device=device)
#     # GEMM in fp32 then cast to fp64 is fast & accurate for this step
#     G_small = (SA @ SA.t()).to(acc_dtype)            # ℓ×ℓ
#     S_small += alpha * G_small

#     # Tiny Cholesky + logdet (CPU or GPU)
#     # if use_gpu_chol:
#     #     S_small = S_small.to(device)                 # copy only ℓ×ℓ once
#     L = torch.linalg.cholesky(S_small)
#     logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
#     bits_per_sample = logdet / math.log(2.0)
#     return float(bits_per_sample / time_resolution)


# # Call the bitrate_countsketch function with the Gram matrix G
# bitrate = bitrate_countsketch(
#     G,
#     noise_std_full_brain=1.0,
#     time_resolution=1.0,
#     n_detectors=n_sensors,
#     ell=65000,
#     block_rows=65536,
#     seed=0,
#     compute_dtype=torch.float32,
#     acc_dtype=torch.float64,
#     use_gpu_chol=True,
#     device="cuda"
# )
# print(f"Bitrate (CountSketch): {bitrate:.2f} bits/sec")
