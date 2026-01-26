"""
Simulation of ultrasound propagation in a free field, using the analytical fundamental solution (Green's function).
We treat the "independent variables" in ultrasound imaging as sources. This relies on the approximation that the intensity of the transmit pulse is the same at each point in the medium, which is related to the Born approximation.
"""

# %%

import torch
import math
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


@torch.no_grad()
def bitrate_slq_torch_gpu_chunked(
    A_cpu: np.ndarray | torch.Tensor,
    noise_std_full_brain: float,
    time_resolution: float = 1.0,
    n_detectors: int | None = None,
    s: int = 16,
    t: int = 40,
    batch: int = 256,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    device: str = "cuda",
    chunk_rows: int = 1024,
    normalize_scale: float = 1.0,
    verbose: bool = False,
):
    if isinstance(A_cpu, torch.Tensor):
        assert A_cpu.device.type == "cpu"
    m, n = A_cpu.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    n_eff = n_detectors if n_detectors is not None else 1
    alpha = torch.tensor(
        1.0 / (noise_std_full_brain**2 / n_eff),
        dtype=krylov_dtype,
        device=device,
    )
    ln2 = torch.tensor(math.log(2.0), dtype=krylov_dtype, device=device)

    Q = torch.empty((d, batch), dtype=krylov_dtype, device=device)
    Qm1 = torch.zeros_like(Q)
    al = torch.empty((t, batch), dtype=krylov_dtype, device=device)
    be = torch.empty((t - 1, batch), dtype=krylov_dtype, device=device)

    def get_chunk(start, end):
        if isinstance(A_cpu, torch.Tensor):
            chunk = A_cpu[start:end]
            return chunk.to(device=device, dtype=compute_dtype, non_blocking=False)
        return torch.as_tensor(A_cpu[start:end], dtype=compute_dtype, device=device)

    def B_mv(Qk):
        b = Qk.shape[1]
        if left:
            T32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
            for row_start in range(0, m, chunk_rows):
                row_end = min(row_start + chunk_rows, m)
                A_chunk = get_chunk(row_start, row_end)
                if normalize_scale != 1.0:
                    A_chunk = A_chunk * normalize_scale
                Q_chunk32 = Qk[row_start:row_end].to(compute_dtype)
                T32.addmm_(A_chunk.T, Q_chunk32)
            W32 = torch.empty((m, b), dtype=compute_dtype, device=device)
            for row_start in range(0, m, chunk_rows):
                row_end = min(row_start + chunk_rows, m)
                A_chunk = get_chunk(row_start, row_end)
                if normalize_scale != 1.0:
                    A_chunk = A_chunk * normalize_scale
                W32[row_start:row_end] = A_chunk @ T32
            return W32.to(krylov_dtype)
        Q32 = Qk.to(compute_dtype)
        W32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
        for row_start in range(0, m, chunk_rows):
            row_end = min(row_start + chunk_rows, m)
            A_chunk = get_chunk(row_start, row_end)
            if normalize_scale != 1.0:
                A_chunk = A_chunk * normalize_scale
            T_chunk = A_chunk @ Q32
            W32.addmm_(A_chunk.T, T_chunk)
        return W32.to(krylov_dtype)

    est = torch.zeros((), dtype=krylov_dtype, device=device)
    done = 0
    sqrt_d = math.sqrt(d)
    scale = torch.tensor(float(d), dtype=krylov_dtype, device=device)
    t0 = time.perf_counter()
    batch_idx = 0

    while done < s:
        b = min(batch, s - done)
        batch_idx += 1
        if verbose:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"[slq] batch {batch_idx}, probes {done}/{s}, elapsed {elapsed:.2f}s")
        Z = (torch.randint(0, 2, (d, b), device=device) * 2 - 1).to(krylov_dtype)
        Q[:, :b] = Z / sqrt_d
        Qm1[:, :b].zero_()

        for k in range(t):
            W = B_mv(Q[:, :b])
            if verbose and (k == 0 or (k + 1) % 10 == 0):
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                print(f"[slq] batch {batch_idx} lanczos {k + 1}/{t} elapsed {elapsed:.2f}s")
            if k > 0:
                W -= Qm1[:, :b] * be[k - 1, :b][None, :]
            ak = torch.sum(Q[:, :b] * W, dim=0)
            W -= Q[:, :b] * ak[None, :]
            al[k, :b] = ak
            if k < t - 1:
                bk = torch.linalg.vector_norm(W, dim=0)
                be[k, :b] = bk
                mask = bk > 1e-30
                Qm1[:, :b] = Q[:, :b]
                Q[:, :b] = torch.where(mask[None, :], W / bk[None, :], Q[:, :b])

        for j in range(b):
            tj = t
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device=device)
            Tj.diagonal(0).copy_(al[:tj, j])
            if tj > 1:
                off = be[:tj - 1, j]
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            est += scale * torch.dot(w1, torch.log1p(alpha * evals))

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))

# %%

import argparse

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=32000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=1000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
parser.add_argument('--sensor_batch_size', type=int, default=256, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--center_frequency', type=float, default=0.05e6, help='Center frequency in Hz')
parser.add_argument('--accumulate_on_cpu', action='store_true', help='Accumulate Gram matrix on CPU instead of GPU')
parser.add_argument('--svd_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to compute eigenvalues/SVD of Gram')
parser.add_argument(
    '--bitrate_method',
    type=str,
    default='slq',
    choices=['svd', 'slq', 'both'],
    help='Compute bitrate via SVD, SLQ, or both',
)
parser.add_argument('--slq_s', type=int, default=16, help='Number of SLQ probe vectors')
parser.add_argument('--slq_t', type=int, default=40, help='Lanczos steps for SLQ')
parser.add_argument('--slq_batch', type=int, default=64, help='SLQ batch size per iteration')
parser.add_argument('--slq_chunk_rows', type=int, default=1024, help='Row chunk size for SLQ matvecs')
parser.add_argument('--slq_verbose', action='store_true', help='Print SLQ progress logs')

args = parser.parse_args()

n_sources = args.n_sources
n_sensors = args.n_sensors
temporal_sampling = args.temporal_sampling
sensor_batch_size = args.sensor_batch_size
svd_device = args.svd_device
center_frequency = args.center_frequency
bitrate_method = args.bitrate_method
accumulate_on_cpu = args.accumulate_on_cpu

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
G_device = "cpu" if accumulate_on_cpu else device
G = torch.zeros((num_sensors_total * nt, num_sources_total), dtype=torch.float32, device=G_device)
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
    if accumulate_on_cpu:
        G[last_index:last_index + chunk_rows] = chunk_matrix.cpu()
    else:
        G[last_index:last_index + chunk_rows] = chunk_matrix
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

from guti.data_utils import Parameters
from guti.core import get_bitrate, noise_floor_heuristic
from guti.data_utils import save_svd

noise_level = None
s_normalized = None

if bitrate_method in {"svd", "both"}:
    G_svd = G if G.device.type == svd_device else G.to(svd_device)
    s = torch.linalg.svdvals(G_svd)
    s = s.cpu().numpy()
    print(f"First 10 singular values: {s[:10]}")
    print(f"Ratio of sums: {np.sum(s[:10])}")
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

    save_svd(s, f'us_free_field_analytical_frequency_sweep', params=Parameters(
        num_sensors=len(sensor_positions),
        num_brain_grid_points=len(source_positions),
        time_resolution=time_step,
        frequency_hz=center_frequency,
        vincent_trick=False
    ))

    s_normalized = s / (len(source_positions)**0.5 * len(sensor_positions)**0.5)
    noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=2000.0)
    print(f"noise_level: {noise_level}")
    print(f"bitrate: {get_bitrate(s_normalized, noise_level, time_resolution=1.0)}")

if bitrate_method in {"slq", "both"}:
    if noise_level is None:
        G_svd = G if G.device.type == svd_device else G.to(svd_device)
        s = torch.linalg.svdvals(G_svd)
        s = s.cpu().numpy()
        s_normalized = s / (len(source_positions)**0.5 * len(sensor_positions)**0.5)
        noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=2000.0)
    if torch.cuda.is_available():
        print("Computing bitrate using SLQ")
        G_cpu = G if G.device.type == "cpu" else G.cpu()
        bitrate_slq = bitrate_slq_torch_gpu_chunked(
            G_cpu,
            noise_std_full_brain=noise_level,
            time_resolution=1.0,
            s=args.slq_s,
            t=args.slq_t,
            batch=args.slq_batch,
            chunk_rows=args.slq_chunk_rows,
            normalize_scale=1.0 / math.sqrt(len(source_positions) * len(sensor_positions)),
            verbose=args.slq_verbose,
        )
        print(f"bitrate (SLQ GPU): {bitrate_slq}")
    else:
        print("CUDA unavailable; skipping SLQ bitrate approximation.")

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
