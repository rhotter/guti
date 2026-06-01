# experimental: uses newton-schulz iteration for SVD calculations (doesn't seem to converge?)
"""
Simulation of ultrasound propagation in a free field, using the analytical fundamental solution (Green's function).
We treat the "independent variables" in ultrasound imaging as sources. This relies on the approximation that the intensity of the transmit pulse is the same at each point in the medium, which is related to the Born approximation.
"""

# %%

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
parser.add_argument('--n_sources', type=int, default=32000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=1000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
parser.add_argument('--sensor_batch_size', type=int, default=256, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--center_frequency', type=float, default=0.05e6, help='Center frequency in Hz')
parser.add_argument('--svd_device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to compute eigenvalues/SVD of Gram')



def newton_schulz_svd(G_local, rank, num_iters=10):
    n = G_local.shape[1]
    device = G_local.device
    
    # Estimate spectral norm via power iteration
    v = torch.randn(n, device=device)
    v = v / v.norm()
    
    for _ in range(5):
        # G @ v: each rank computes local part, we only need ||G@v||^2
        Gv_local = G_local @ v
        
        # GTGv = G^T @ (G @ v)
        GTGv_local = G_local.T @ Gv_local
        GTGv = GTGv_local.clone()
        dist.all_reduce(GTGv, op=dist.ReduceOp.SUM)
        
        v = GTGv / GTGv.norm()
    
    # Estimate largest singular value: sqrt(v^T G^T G v)
    Gv_local = G_local @ v
    norm_sq_local = (Gv_local ** 2).sum()
    norm_sq = norm_sq_local.clone()
    dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM)
    alpha = norm_sq.sqrt()
    
    # Newton-Schulz
    Y = G_local / alpha
    
    for it in range(num_iters):
        YtY = Y.T @ Y
        dist.all_reduce(YtY, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            orthogonality_error = torch.norm(YtY - torch.eye(n, device=device))
            print(f"Iteration {it}: orthogonality error = {orthogonality_error.item():.6e}")
        
        Y = Y @ (1.5 * torch.eye(n, device=device) - 0.5 * YtY)
    
    # Y is now U_local for G/alpha, compute singular values
    Sigma_local = Y.T @ (G_local / alpha)
    dist.all_reduce(Sigma_local, op=dist.ReduceOp.SUM)
    
    # Sigma_local is now Y^T (G/alpha) = U^T (G/alpha) ≈ (Sigma/alpha) V^T
    # Column norms give singular values divided by alpha
    s = torch.linalg.norm(Sigma_local, dim=0) * alpha
    
    return s

def worker(rank, world_size, G):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:29500',
        rank=rank,
        world_size=world_size
    )
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        m, n = G.shape
        G = G.to(device)
    else:
        m = n = 0
    
    shape_tensor = torch.tensor([m, n], device=device)
    dist.broadcast(shape_tensor, src=0)
    m, n = shape_tensor.tolist()
    
    rows_per_gpu = m // world_size
    remainder = m % world_size
    local_rows = rows_per_gpu + (1 if rank < remainder else 0)
    start_row = rank * rows_per_gpu + min(rank, remainder)
    
    G_local = torch.empty((local_rows, n), device=device)
    
    if rank == 0:
        for r in range(world_size):
            r_rows = rows_per_gpu + (1 if r < remainder else 0)
            r_start = r * rows_per_gpu + min(r, remainder)
            chunk = G[r_start:r_start + r_rows]
            if r == 0:
                G_local.copy_(chunk)
            else:
                dist.send(chunk.contiguous(), dst=r)
    else:
        dist.recv(G_local, src=0)
    
    s = newton_schulz_svd(G_local, rank)
    
    if rank == 0:
        torch.save(s.cpu(), '/tmp/singular_values.pt')
    
    dist.destroy_process_group()

if __name__ == '__main__':
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
    
    G = G.share_memory_()  # Make it shared across processes
    world_size = 5
    mp.spawn(worker, args=(world_size, G), nprocs=world_size, join=True)
    
    s = torch.load('/tmp/singular_values.pt').numpy()
    print(f"First 10 singular values: {s[:10]}")
    print(f"Ratio of sums: {np.sum(s[:10])}")

    print("Done!")

    # plt.semilogy(s)
    # ax = plt.gca()
    # ax.set_xlabel("Singular value index")
    # ax.set_ylabel("Singular value")
    # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True, min_n_ticks=10))
    # ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    # ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=30))
    # ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=30))
    # log_formatter = mticker.LogFormatter(base=10.0, labelOnlyBase=False)
    # ax.yaxis.set_major_formatter(log_formatter)
    # ax.yaxis.set_minor_formatter(log_formatter)
    # ax.grid(True, which='both', linestyle='--', alpha=0.3)
    # plt.tight_layout()
    
    from guti.data_utils import Parameters
    from guti.core import get_bitrate, noise_floor_heuristic
    
    from guti.data_utils import save_svd
    time_step = 1e-1 / center_frequency
    
    save_svd(s, f'us_free_field_analytical_n_sources_sweep_{int(args.center_frequency/1e3)}khz', params=Parameters(
        num_sensors=len(sensor_positions),
        num_brain_grid_points=len(source_positions),
        time_resolution=time_step,
        vincent_trick=False
    ))

    s_normalized = s * len(sensor_positions)**0.5 / len(source_positions)**0.5
    
    noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=10.0)
    
    print(f"noise_level: {noise_level}")
    print(f"bitrate: {get_bitrate(s_normalized, noise_level, time_resolution=1.0)}")
    
    exit(0)