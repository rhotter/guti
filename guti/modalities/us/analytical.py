"""
Simulation of ultrasound propagation in a free field, using the analytical fundamental solution (Green's function).
We treat the "independent variables" in ultrasound imaging as sources. This relies on the approximation that the intensity of the transmit pulse is the same at each point in the medium, which is related to the Born approximation.
"""

# %%

import torch
import math
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from guti.data_utils import save_svd
from guti.modalities.us.utils import create_medium, create_sources_real, create_receivers_real, simulate_free_field_propagation, plot_medium
import time
from pathlib import Path

import torch, torch.backends.cuda as cu
import torch.cuda.comm as cuda_comm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
torch.set_float32_matmul_precision('high')  # allow TF32 on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
cu.preferred_linalg_library("magma")        # robust & fast dense LA


def build_source_signal(
    time_axis: np.ndarray,
    center_frequency: float,
    signal_type: str = "tone_burst",
    signal_cycles: float = 2.0,
    signal_window: str = "hann",
) -> np.ndarray:
    carrier = np.sin(2 * np.pi * time_axis * center_frequency)
    if signal_type == "cw":
        return carrier
    if signal_type != "tone_burst":
        raise ValueError(f"Unsupported signal_type={signal_type!r}")

    if time_axis.size == 0:
        return carrier
    if signal_cycles <= 0:
        raise ValueError("signal_cycles must be positive")

    if time_axis.size == 1:
        dt = 1.0 / (10.0 * center_frequency)
    else:
        dt = float(time_axis[1] - time_axis[0])
    active_duration = signal_cycles / center_frequency
    active_samples = max(1, min(time_axis.size, int(round(active_duration / dt))))

    envelope = np.ones(active_samples, dtype=np.float64)
    if signal_window == "hann":
        if active_samples > 1:
            envelope = np.hanning(active_samples)
    elif signal_window != "rect":
        raise ValueError(f"Unsupported signal_window={signal_window!r}")

    signal = np.zeros_like(carrier)
    signal[:active_samples] = carrier[:active_samples] * envelope
    return signal


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
    """Approximate the Shannon bitrate using Stochastic Lanczos Quadrature.

    The exact SVD path computes sum_i log2(1 + sigma_i^2 / noise^2). This
    routine estimates the equivalent trace-log expression
    Tr log(I + A A^T / noise^2) without forming the full spectrum. Random
    Rademacher probe vectors estimate the trace, and Lanczos turns each probe's
    quadratic form into a small tridiagonal eigendecomposition.
    """
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
    alpha_cpu = alpha.cpu()
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
            print(f"[slq] batch {batch_idx}, probes {done}/{s}, elapsed {elapsed:.2f}s", flush=True)
        Z = (torch.randint(0, 2, (d, b), device=device) * 2 - 1).to(krylov_dtype)
        Q[:, :b] = Z / sqrt_d
        Qm1[:, :b].zero_()

        for k in range(t):
            W = B_mv(Q[:, :b])
            if verbose and (k == 0 or (k + 1) % 10 == 0):
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                print(f"[slq] batch {batch_idx} lanczos {k + 1}/{t} elapsed {elapsed:.2f}s", flush=True)
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
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device="cpu")
            Tj.diagonal(0).copy_(al[:tj, j].cpu())
            if tj > 1:
                off = be[:tj - 1, j].cpu()
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            quad = torch.dot(w1, torch.log1p(alpha_cpu * evals))
            est += scale * quad.to(device=est.device)

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))


@torch.no_grad()
def bitrate_slq_torch_gpu_chunked_probe_parallel(
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
    device_ids: list[int] | None = None,
    chunk_rows: int = 1024,
    normalize_scale: float = 1.0,
    verbose: bool = False,
):
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise ValueError("device_ids must include at least one CUDA device")

    n_dev = len(device_ids)
    base = s // n_dev
    remainder = s % n_dev
    s_parts = [base + (1 if i < remainder else 0) for i in range(n_dev)]

    def worker(dev_id, s_local):
        if s_local == 0:
            return 0.0, 0
        torch.cuda.set_device(dev_id)
        bitrate_local = bitrate_slq_torch_gpu_chunked(
            A_cpu,
            noise_std_full_brain=noise_std_full_brain,
            time_resolution=time_resolution,
            n_detectors=n_detectors,
            s=s_local,
            t=t,
            batch=batch,
            use_left_if_smaller=use_left_if_smaller,
            compute_dtype=compute_dtype,
            krylov_dtype=krylov_dtype,
            device=f"cuda:{dev_id}",
            chunk_rows=chunk_rows,
            normalize_scale=normalize_scale,
            verbose=verbose,
        )
        return bitrate_local, s_local

    est_sum = 0.0
    s_sum = 0
    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        futures = []
        for dev_id, s_local in zip(device_ids, s_parts):
            futures.append(pool.submit(worker, dev_id, s_local))
        for fut in as_completed(futures):
            bitrate_local, s_local = fut.result()
            if s_local > 0:
                est_sum += bitrate_local * s_local
                s_sum += s_local

    if s_sum == 0:
        return 0.0
    return est_sum / s_sum


@torch.no_grad()
def bitrate_slq_torch_multi_gpu_sharded(
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
    device_ids: list[int] | None = None,
    normalize_scale: float = 1.0,
    verbose: bool = False,
):
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise ValueError("device_ids must include at least one CUDA device")

    if isinstance(A_cpu, torch.Tensor) and A_cpu.device.type != "cpu":
        A_cpu = A_cpu.cpu()

    m, n = A_cpu.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    n_eff = n_detectors if n_detectors is not None else 1
    alpha = torch.tensor(
        1.0 / (noise_std_full_brain**2 / n_eff),
        dtype=krylov_dtype,
        device=f"cuda:{device_ids[0]}",
    )
    alpha_cpu = alpha.cpu()
    ln2 = torch.tensor(math.log(2.0), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")

    # Shard rows of A across devices.
    row_splits = torch.linspace(0, m, len(device_ids) + 1, dtype=torch.int64).tolist()
    A_shards = []
    for i, dev in enumerate(device_ids):
        r0, r1 = row_splits[i], row_splits[i + 1]
        shard = torch.as_tensor(A_cpu[r0:r1], dtype=compute_dtype, device=f"cuda:{dev}")
        if normalize_scale != 1.0:
            shard = shard * normalize_scale
        A_shards.append(shard)

    Q_shards = None
    Qm1_shards = None
    if left:
        Q_shards = [
            torch.empty((A_shards[i].shape[0], batch), dtype=krylov_dtype, device=A_shards[i].device)
            for i in range(len(device_ids))
        ]
        Qm1_shards = [torch.zeros_like(Q_shards[i]) for i in range(len(device_ids))]
    else:
        Q = torch.empty((d, batch), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")
        Qm1 = torch.zeros_like(Q)

    al = torch.empty((t, batch), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")
    be = torch.empty((t - 1, batch), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")

    def reduce_sum(tensors):
        return cuda_comm.reduce_add(tensors)

    def broadcast(tensor):
        return cuda_comm.broadcast(tensor, devices=device_ids)

    def B_mv_left(Q_parts):
        # T = sum_g A_g^T Q_g
        T_parts = [A_shards[i].T @ Q_parts[i].to(compute_dtype) for i in range(len(device_ids))]
        T = reduce_sum(T_parts)
        T_parts_b = broadcast(T)
        W_parts = [A_shards[i] @ T_parts_b[i] for i in range(len(device_ids))]
        return W_parts

    def B_mv_right(Q_full):
        Q_parts = broadcast(Q_full)
        T_parts = [A_shards[i] @ Q_parts[i].to(compute_dtype) for i in range(len(device_ids))]
        W_parts = [A_shards[i].T @ T_parts[i] for i in range(len(device_ids))]
        W = reduce_sum(W_parts)
        return W

    est = torch.zeros((), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")
    done = 0
    sqrt_d = math.sqrt(d)
    scale = torch.tensor(float(d), dtype=krylov_dtype, device=f"cuda:{device_ids[0]}")
    t0 = time.perf_counter()
    batch_idx = 0

    while done < s:
        b = min(batch, s - done)
        batch_idx += 1
        if verbose:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"[slq-mgpu] batch {batch_idx}, probes {done}/{s}, elapsed {elapsed:.2f}s", flush=True)

        if left:
            for i in range(len(device_ids)):
                dev = A_shards[i].device
                z = (torch.randint(0, 2, (A_shards[i].shape[0], b), device=dev) * 2 - 1).to(krylov_dtype)
                Q_shards[i][:, :b] = z / sqrt_d
                Qm1_shards[i][:, :b].zero_()
        else:
            z = (torch.randint(0, 2, (d, b), device=Q.device) * 2 - 1).to(krylov_dtype)
            Q[:, :b] = z / sqrt_d
            Qm1[:, :b].zero_()

        for k in range(t):
            if left:
                W_parts = B_mv_left([Q_shards[i][:, :b] for i in range(len(device_ids))])
                if k > 0:
                    be_prev = be[k - 1, :b]
                    be_parts = broadcast(be_prev)
                    for i in range(len(device_ids)):
                        W_parts[i] -= Qm1_shards[i][:, :b] * be_parts[i][None, :]
                ak_parts = [
                    torch.sum(Q_shards[i][:, :b] * W_parts[i], dim=0)
                    for i in range(len(device_ids))
                ]
                ak = reduce_sum(ak_parts)
                ak_parts = broadcast(ak)
                for i in range(len(device_ids)):
                    W_parts[i] -= Q_shards[i][:, :b] * ak_parts[i][None, :]
                al[k, :b] = ak
                if k < t - 1:
                    bk_parts = [torch.sum(W_parts[i] ** 2, dim=0) for i in range(len(device_ids))]
                    bk2 = reduce_sum(bk_parts)
                    bk = torch.sqrt(bk2)
                    be[k, :b] = bk
                    bk_parts = broadcast(bk)
                    mask = bk > 1e-30
                    mask_parts = broadcast(mask)
                    for i in range(len(device_ids)):
                        Qm1_shards[i][:, :b] = Q_shards[i][:, :b]
                        Q_shards[i][:, :b] = torch.where(
                            mask_parts[i][None, :],
                            W_parts[i] / bk_parts[i][None, :],
                            Q_shards[i][:, :b],
                        )
            else:
                W = B_mv_right(Q[:, :b])
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

            if verbose and (k == 0 or (k + 1) % 10 == 0):
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                print(f"[slq-mgpu] batch {batch_idx} lanczos {k + 1}/{t} elapsed {elapsed:.2f}s", flush=True)

        for j in range(b):
            tj = t
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device="cpu")
            Tj.diagonal(0).copy_(al[:tj, j].cpu())
            if tj > 1:
                off = be[:tj - 1, j].cpu()
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            quad = torch.dot(w1, torch.log1p(alpha_cpu * evals))
            est += scale * quad.to(device=est.device)

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))


@torch.no_grad()
def estimate_spectral_norm_chunked(
    A_cpu: np.ndarray | torch.Tensor,
    normalize_scale: float = 1.0,
    n_iters: int = 20,
    device: str = "cuda",
    chunk_rows: int = 1024,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    if isinstance(A_cpu, torch.Tensor):
        assert A_cpu.device.type == "cpu"
    m, n = A_cpu.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

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

    v = torch.randn((d, 1), dtype=krylov_dtype, device=device)
    v = v / torch.linalg.vector_norm(v)
    for it in range(n_iters):
        if verbose and (it == 0 or (it + 1) % 5 == 0):
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"[noise-first] power iter {it + 1}/{n_iters}", flush=True)
        w = B_mv(v)
        w_norm = torch.linalg.vector_norm(w)
        v = w / (w_norm + 1e-30)
    rayleigh = torch.dot(v.squeeze(1), B_mv(v).squeeze(1))
    return float(torch.sqrt(rayleigh).item())


@torch.no_grad()
def estimate_frobenius_norm_sq_hutchinson(
    A_cpu: np.ndarray | torch.Tensor,
    normalize_scale: float = 1.0,
    n_probes: int = 16,
    device: str = "cuda",
    chunk_rows: int = 1024,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    if isinstance(A_cpu, torch.Tensor):
        assert A_cpu.device.type == "cpu"
    m, n = A_cpu.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

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

    acc = 0.0
    for i in range(n_probes):
        if verbose:
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"[noise-power] probe {i + 1}/{n_probes}", flush=True)
        z = (torch.randint(0, 2, (d, 1), device=device) * 2 - 1).to(krylov_dtype)
        acc += torch.dot(z.squeeze(1), B_mv(z).squeeze(1)).item()
    return acc / n_probes


@torch.no_grad()
def bitrate_slq_torch_gpu_streaming(
    compute_chunk_matrix,
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
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
    normalize_scale: float = 1.0,
    verbose: bool = False,
):
    """Streaming SLQ bitrate estimate without materializing the propagation matrix.

    This estimates the same trace-log bitrate as ``bitrate_slq_torch_gpu_chunked``,
    but builds only the matrix chunks needed for each Lanczos matvec. It is the
    path used for large analytical sweeps where storing the full sensor-time by
    source matrix would dominate memory.
    """
    m = num_sensors_total * nt
    n = num_sources_total
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    n_eff = n_detectors if n_detectors is not None else 1
    alpha = torch.tensor(
        1.0 / (noise_std_full_brain**2 / n_eff),
        dtype=krylov_dtype,
        device=device,
    )
    alpha_cpu = alpha.cpu()
    ln2 = torch.tensor(math.log(2.0), dtype=krylov_dtype, device=device)

    Q = torch.empty((d, batch), dtype=krylov_dtype, device=device)
    Qm1 = torch.zeros_like(Q)
    al = torch.empty((t, batch), dtype=krylov_dtype, device=device)
    be = torch.empty((t - 1, batch), dtype=krylov_dtype, device=device)

    def B_mv(Qk):
        b = Qk.shape[1]
        if left:
            T32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[slq-stream] A^T Q", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                Q_chunk32 = Qk[row_start:row_start + rows].to(compute_dtype)
                T32.addmm_(chunk_matrix.T, Q_chunk32)
                row_start += rows
            W32 = torch.empty((m, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[slq-stream] A T", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                W32[row_start:row_start + rows] = chunk_matrix @ T32
                row_start += rows
            return W32.to(krylov_dtype)
        Q32 = Qk.to(compute_dtype)
        W32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
        batch_iter = range(0, num_sensors_total, sensor_batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="[slq-stream] A^T A Q", leave=False)
        for start in batch_iter:
            end = min(start + sensor_batch_size, num_sensors_total)
            chunk_matrix = compute_chunk_matrix(start, end)
            if normalize_scale != 1.0:
                chunk_matrix = chunk_matrix * normalize_scale
            T_chunk = chunk_matrix @ Q32
            W32.addmm_(chunk_matrix.T, T_chunk)
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
            print(f"[slq-stream] batch {batch_idx}, probes {done}/{s}, elapsed {elapsed:.2f}s", flush=True)
        Z = (torch.randint(0, 2, (d, b), device=device) * 2 - 1).to(krylov_dtype)
        Q[:, :b] = Z / sqrt_d
        Qm1[:, :b].zero_()

        for k in range(t):
            W = B_mv(Q[:, :b])
            if verbose and (k == 0 or (k + 1) % 10 == 0):
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                print(f"[slq-stream] batch {batch_idx} lanczos {k + 1}/{t} elapsed {elapsed:.2f}s", flush=True)
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
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device="cpu")
            Tj.diagonal(0).copy_(al[:tj, j].cpu())
            if tj > 1:
                off = be[:tj - 1, j].cpu()
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            quad = torch.dot(w1, torch.log1p(alpha_cpu * evals))
            est += scale * quad.to(device=est.device)

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))


@torch.no_grad()
def bitrate_slq_torch_gpu_streaming_probe_parallel(
    make_compute_chunk_matrix,
    device_ids: list[int],
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
    noise_std_full_brain: float,
    time_resolution: float = 1.0,
    n_detectors: int | None = None,
    s: int = 16,
    t: int = 40,
    batch: int = 256,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    normalize_scale: float = 1.0,
    verbose: bool = False,
):
    if not device_ids:
        raise ValueError("device_ids must include at least one CUDA device")
    n_dev = len(device_ids)
    base = s // n_dev
    remainder = s % n_dev
    s_parts = [base + (1 if i < remainder else 0) for i in range(n_dev)]

    def worker(dev_id, s_local):
        if s_local == 0:
            return 0.0, 0
        torch.cuda.set_device(dev_id)
        compute_chunk_matrix = make_compute_chunk_matrix(dev_id)
        bitrate_local = bitrate_slq_torch_gpu_streaming(
            compute_chunk_matrix,
            num_sensors_total,
            num_sources_total,
            nt,
            sensor_batch_size,
            noise_std_full_brain=noise_std_full_brain,
            time_resolution=time_resolution,
            n_detectors=n_detectors,
            s=s_local,
            t=t,
            batch=batch,
            use_left_if_smaller=use_left_if_smaller,
            compute_dtype=compute_dtype,
            krylov_dtype=krylov_dtype,
            device=f"cuda:{dev_id}",
            normalize_scale=normalize_scale,
            verbose=verbose,
        )
        return bitrate_local, s_local

    est_sum = 0.0
    s_sum = 0
    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        futures = []
        for dev_id, s_local in zip(device_ids, s_parts):
            futures.append(pool.submit(worker, dev_id, s_local))
        for fut in as_completed(futures):
            bitrate_local, s_local = fut.result()
            if s_local > 0:
                est_sum += bitrate_local * s_local
                s_sum += s_local

    if s_sum == 0:
        return 0.0
    return est_sum / s_sum


@torch.no_grad()
def estimate_spectral_norm_streaming(
    compute_chunk_matrix,
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
    normalize_scale: float = 1.0,
    n_iters: int = 20,
    device: str = "cuda",
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    m = num_sensors_total * nt
    n = num_sources_total
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    def B_mv(Qk):
        b = Qk.shape[1]
        if left:
            T32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[noise-first] A^T Q", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                Q_chunk32 = Qk[row_start:row_start + rows].to(compute_dtype)
                T32.addmm_(chunk_matrix.T, Q_chunk32)
                row_start += rows
            W32 = torch.empty((m, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[noise-first] A T", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                W32[row_start:row_start + rows] = chunk_matrix @ T32
                row_start += rows
            return W32.to(krylov_dtype)
        Q32 = Qk.to(compute_dtype)
        W32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
        batch_iter = range(0, num_sensors_total, sensor_batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="[noise-first] A^T A Q", leave=False)
        for start in batch_iter:
            end = min(start + sensor_batch_size, num_sensors_total)
            chunk_matrix = compute_chunk_matrix(start, end)
            if normalize_scale != 1.0:
                chunk_matrix = chunk_matrix * normalize_scale
            T_chunk = chunk_matrix @ Q32
            W32.addmm_(chunk_matrix.T, T_chunk)
        return W32.to(krylov_dtype)

    v = torch.randn((d, 1), dtype=krylov_dtype, device=device)
    v = v / torch.linalg.vector_norm(v)
    for it in range(n_iters):
        if verbose and (it == 0 or (it + 1) % 5 == 0):
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"[noise-first] power iter {it + 1}/{n_iters}", flush=True)
        w = B_mv(v)
        w_norm = torch.linalg.vector_norm(w)
        v = w / (w_norm + 1e-30)
    rayleigh = torch.dot(v.squeeze(1), B_mv(v).squeeze(1))
    return float(torch.sqrt(rayleigh).item())


@torch.no_grad()
def estimate_frobenius_norm_sq_streaming(
    compute_chunk_matrix,
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
    normalize_scale: float = 1.0,
    n_probes: int = 16,
    device: str = "cuda",
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    m = num_sensors_total * nt
    n = num_sources_total
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    def B_mv(Qk):
        b = Qk.shape[1]
        if left:
            T32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[noise-power] A^T Q", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                Q_chunk32 = Qk[row_start:row_start + rows].to(compute_dtype)
                T32.addmm_(chunk_matrix.T, Q_chunk32)
                row_start += rows
            W32 = torch.empty((m, b), dtype=compute_dtype, device=device)
            row_start = 0
            batch_iter = range(0, num_sensors_total, sensor_batch_size)
            if verbose:
                batch_iter = tqdm(batch_iter, desc="[noise-power] A T", leave=False)
            for start in batch_iter:
                end = min(start + sensor_batch_size, num_sensors_total)
                chunk_matrix = compute_chunk_matrix(start, end)
                if normalize_scale != 1.0:
                    chunk_matrix = chunk_matrix * normalize_scale
                rows = chunk_matrix.shape[0]
                W32[row_start:row_start + rows] = chunk_matrix @ T32
                row_start += rows
            return W32.to(krylov_dtype)
        Q32 = Qk.to(compute_dtype)
        W32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
        batch_iter = range(0, num_sensors_total, sensor_batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="[noise-power] A^T A Q", leave=False)
        for start in batch_iter:
            end = min(start + sensor_batch_size, num_sensors_total)
            chunk_matrix = compute_chunk_matrix(start, end)
            if normalize_scale != 1.0:
                chunk_matrix = chunk_matrix * normalize_scale
            T_chunk = chunk_matrix @ Q32
            W32.addmm_(chunk_matrix.T, T_chunk)
        return W32.to(krylov_dtype)

    acc = 0.0
    for i in range(n_probes):
        if verbose:
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"[noise-power] probe {i + 1}/{n_probes}", flush=True)
        z = (torch.randint(0, 2, (d, 1), device=device) * 2 - 1).to(krylov_dtype)
        acc += torch.dot(z.squeeze(1), B_mv(z).squeeze(1)).item()
    return acc / n_probes


@torch.no_grad()
def estimate_spectral_norm_streaming_probe_parallel(
    make_compute_chunk_matrix,
    device_ids: list[int],
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
    normalize_scale: float = 1.0,
    n_iters: int = 20,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    if not device_ids:
        raise ValueError("device_ids must include at least one CUDA device")

    def worker(dev_id):
        torch.cuda.set_device(dev_id)
        compute_chunk_matrix = make_compute_chunk_matrix(dev_id)
        return estimate_spectral_norm_streaming(
            compute_chunk_matrix,
            num_sensors_total,
            num_sources_total,
            nt,
            sensor_batch_size,
            normalize_scale=normalize_scale,
            n_iters=n_iters,
            device=f"cuda:{dev_id}",
            use_left_if_smaller=use_left_if_smaller,
            compute_dtype=compute_dtype,
            krylov_dtype=krylov_dtype,
            verbose=verbose,
        )

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures = [pool.submit(worker, dev_id) for dev_id in device_ids]
        vals = [f.result() for f in as_completed(futures)]
    return max(vals)


@torch.no_grad()
def estimate_frobenius_norm_sq_streaming_probe_parallel(
    make_compute_chunk_matrix,
    device_ids: list[int],
    num_sensors_total: int,
    num_sources_total: int,
    nt: int,
    sensor_batch_size: int,
    normalize_scale: float = 1.0,
    n_probes: int = 16,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    verbose: bool = False,
):
    if not device_ids:
        raise ValueError("device_ids must include at least one CUDA device")

    n_dev = len(device_ids)
    base = n_probes // n_dev
    remainder = n_probes % n_dev
    probes_parts = [base + (1 if i < remainder else 0) for i in range(n_dev)]

    def worker(dev_id, n_local):
        if n_local == 0:
            return 0.0, 0
        torch.cuda.set_device(dev_id)
        compute_chunk_matrix = make_compute_chunk_matrix(dev_id)
        val = estimate_frobenius_norm_sq_streaming(
            compute_chunk_matrix,
            num_sensors_total,
            num_sources_total,
            nt,
            sensor_batch_size,
            normalize_scale=normalize_scale,
            n_probes=n_local,
            device=f"cuda:{dev_id}",
            use_left_if_smaller=use_left_if_smaller,
            compute_dtype=compute_dtype,
            krylov_dtype=krylov_dtype,
            verbose=verbose,
        )
        return val * n_local, n_local

    acc = 0.0
    total = 0
    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        futures = []
        for dev_id, n_local in zip(device_ids, probes_parts):
            futures.append(pool.submit(worker, dev_id, n_local))
        for fut in as_completed(futures):
            val, n_local = fut.result()
            if n_local > 0:
                acc += val
                total += n_local
    if total == 0:
        return 0.0
    return acc / total

# %%

import argparse


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _sanitize_json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_result_json(path_str: str | None, record: dict) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize_json_value(record), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _emit_result_json(record: dict) -> None:
    sanitized = _sanitize_json_value(record)
    payload = json.dumps(sanitized, sort_keys=True, allow_nan=False)
    print(f"RESULT_JSON: {payload}", flush=True)
    _write_result_json(args.result_json_path, sanitized)


def _sqrt_nonnegative_estimate(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    if value < 0.0:
        if value > -1e-12:
            value = 0.0
        else:
            raise ValueError(f"{label} must be non-negative, got {value}")
    return math.sqrt(value)

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=32000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=1000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
parser.add_argument('--sensor_batch_size', type=int, default=512, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--center_frequency', type=float, default=0.05e6, help='Center frequency in Hz')
parser.add_argument('--signal_type', type=str, default='tone_burst', choices=['cw', 'tone_burst'], help='Excitation waveform type')
parser.add_argument('--signal_cycles', type=float, default=2.0, help='Cycles in the emitted tone burst')
parser.add_argument('--signal_window', type=str, default='hann', choices=['rect', 'hann'], help='Envelope for tone-burst excitation')
parser.add_argument('--accumulate_on_cpu', action='store_true', help='Accumulate Gram matrix on CPU instead of GPU')
parser.add_argument('--svd_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to compute eigenvalues/SVD of Gram')
parser.add_argument(
    '--bitrate_method',
    type=str,
    default='slq',
    choices=['svd', 'slq', 'both'],
    help='Compute bitrate via SVD, SLQ, or both',
)
parser.add_argument('--slq_s', type=int, default=128, help='Number of SLQ probe vectors')
parser.add_argument('--slq_t', type=int, default=128, help='Lanczos steps for SLQ')
parser.add_argument('--slq_batch', type=int, default=128, help='SLQ batch size per iteration')
parser.add_argument('--slq_chunk_rows', type=int, default=65536, help='Row chunk size for SLQ matvecs')
parser.add_argument('--slq_verbose', action='store_true', default=True, help='Print SLQ progress logs')
parser.add_argument('--slq_multi_gpu', action='store_true', help='Use multi-GPU sharded SLQ')
parser.add_argument('--slq_devices', type=str, default='', help='Comma-separated CUDA device IDs for SLQ')
parser.add_argument('--slq_streaming', action='store_true', help='Stream SLQ matvecs without materializing G')
parser.add_argument('--slq_probe_parallel', action='store_true', help='Parallelize SLQ probes across GPUs (chunked mode)')
parser.add_argument('--noise_heuristic', type=str, default='power', choices=['power', 'first'], help='Noise heuristic to use when estimating without SVD')
parser.add_argument('--noise_snr', type=float, default=2000.0, help='SNR used by noise heuristic')
parser.add_argument('--noise_power_probes', type=int, default=16, help='Probes for power heuristic via Hutchinson')
parser.add_argument('--noise_iters', type=int, default=100, help='Power-iteration steps for spectral norm')
parser.add_argument('--noise_level', type=float, default=None, help='Override noise level (skip estimation)')
parser.add_argument('--noise_verbose', action='store_true', default=True, help='Print noise estimation logs')
parser.add_argument(
    '--disable_matrix_normalization',
    action='store_true',
    help='Use the raw propagation matrix instead of dividing by sqrt(n_sources * n_sensors).',
)
parser.add_argument(
    '--result_json_path',
    type=str,
    default=None,
    help='Optional path to save a strict JSON result record for this run.',
)

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
print(f"matrix_normalization: {'disabled' if args.disable_matrix_normalization else 'enabled'}")

# We create a jwave medium object. This is mostly useful for non-free field simulations, but we use it here for convenience/consistency.
domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium(central_frequency=center_frequency, pad=30)

# Create the source and receiver positions in real space (meters).
source_positions = create_sources_real(domain, time_axis, freq_Hz=center_frequency, n_sources=n_sources, inside=True, pad=30)
sensor_positions = create_receivers_real(domain, time_axis, freq_Hz=center_frequency, n_sensors=n_sensors, pad=30)

n_sources = source_positions.shape[0]

# Source waveform
time_step = 1e-1 / center_frequency
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signal = build_source_signal(
    time_axis,
    center_frequency,
    signal_type=args.signal_type,
    signal_cycles=args.signal_cycles,
    signal_window=args.signal_window,
)
source_signals = source_signal
source_signals = np.tile(source_signals, (n_sources, 1))

nt = math.ceil(time_axis.shape[0] / temporal_sampling)
voxel_size = np.array(domain.dx)
effective_time_resolution = time_step * temporal_sampling

#%%

# device for propagation (and potentially accumulation)
device = "cuda"

use_complex_ampitudes = False

if args.slq_streaming and bitrate_method in {"svd", "both"}:
    raise ValueError("--slq_streaming cannot be used with bitrate_method=svd or both")
if args.slq_streaming and args.slq_multi_gpu:
    raise ValueError("--slq_streaming currently supports single-GPU SLQ only")
if args.slq_streaming and args.slq_probe_parallel:
    pass

# %%

if not args.slq_streaming:
    print("Computing SVD (batched simulation + Gram accumulation)...")

num_sensors_total = sensor_positions.shape[0]
num_sources_total = n_sources
print(f"num_sensors: {num_sensors_total}, num_sources: {num_sources_total}, device: {device}")

# Pre-build constant tensors on propagation device to avoid repeated transfers
source_positions_t = torch.tensor(source_positions, device=device)
source_signals_t = torch.tensor(source_signals, device=device)
voxel_size_t = torch.tensor(voxel_size, device=device)

stream_device_ids = None
slq_device_ids = None
source_positions_t_list = None
source_signals_t_list = None
voxel_size_t_list = None
if args.slq_streaming and args.slq_probe_parallel:
    if args.slq_devices:
        stream_device_ids = [int(x) for x in args.slq_devices.split(",") if x.strip() != ""]
    else:
        stream_device_ids = list(range(torch.cuda.device_count()))
    slq_device_ids = stream_device_ids
    source_positions_t_list = [
        torch.tensor(source_positions, device=f"cuda:{dev}") for dev in stream_device_ids
    ]
    source_signals_t_list = [
        torch.tensor(source_signals, device=f"cuda:{dev}") for dev in stream_device_ids
    ]
    voxel_size_t_list = [
        torch.tensor(voxel_size, device=f"cuda:{dev}") for dev in stream_device_ids
    ]

def compute_chunk_matrix(start, end):
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
    if use_complex_ampitudes:
        return torch.cat([pf_chunk.real, pf_chunk.imag], dim=0).float()
    return pf_chunk.permute(0, 2, 1).reshape(-1, num_sources_total).float()


def make_compute_chunk_matrix_for_device(dev_id: int):
    if stream_device_ids is None or source_positions_t_list is None:
        raise ValueError("Streaming probe-parallel tensors are not initialized")
    idx = stream_device_ids.index(dev_id)
    source_positions_t_dev = source_positions_t_list[idx]
    source_signals_t_dev = source_signals_t_list[idx]
    voxel_size_t_dev = voxel_size_t_list[idx]

    def _compute(start, end):
        receiver_positions_t = torch.tensor(sensor_positions[start:end], device=f"cuda:{dev_id}")
        pf_chunk = simulate_free_field_propagation(
            source_positions_t_dev,
            receiver_positions_t,
            source_signals_t_dev,
            time_step,
            center_frequency,
            voxel_size_t_dev,
            device=f"cuda:{dev_id}",
            compute_time_series=not use_complex_ampitudes,
            temporal_sampling=temporal_sampling
        )
        if use_complex_ampitudes:
            return torch.cat([pf_chunk.real, pf_chunk.imag], dim=0).float()
        return pf_chunk.permute(0, 2, 1).reshape(-1, num_sources_total).float()

    return _compute

G = None
if not args.slq_streaming:
    # Accumulate Gram or do TSQR in batches over sensors
    t0 = time.perf_counter()
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
        chunk_matrix = compute_chunk_matrix(start, end)
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
from guti.core import get_bitrate, get_bitrate_channel_capacity, noise_floor_heuristic
from guti.noise_models import compute_noise_effective
from guti.data_utils import save_svd

noise_level = args.noise_level
s_normalized = None
bitrate_svd = None
bitrate_slq = None
saved_svd_path = None

matrix_normalization_scale = (
    1.0
    if args.disable_matrix_normalization
    else 1.0 / math.sqrt(len(source_positions) * len(sensor_positions))
)
if noise_level is None:
    raw_noise_level = compute_noise_effective(
        "us_analytical",
        n_sensors=len(sensor_positions),
        frequency_hz=center_frequency,
    )
    noise_level = raw_noise_level * matrix_normalization_scale
    print(f"noise_eff (raw matrix units): {raw_noise_level}")
    print(f"noise_level (analysis matrix units): {noise_level}")

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

    saved_svd_path = save_svd(s, f'us_free_field_analytical_frequency_sweep', params=Parameters(
        num_sensors=len(sensor_positions),
        num_brain_grid_points=len(source_positions),
        time_resolution=effective_time_resolution,
        frequency_hz=center_frequency,
        comment=f"signal_type={args.signal_type},signal_cycles={args.signal_cycles},signal_window={args.signal_window}",
        vincent_trick=False
    ))

    if args.disable_matrix_normalization:
        s_normalized = s
    else:
        s_normalized = s / (len(source_positions)**0.5 * len(sensor_positions)**0.5)
    if noise_level is None:
        noise_level = noise_floor_heuristic(
            s_normalized,
            heuristic=args.noise_heuristic,
            snr=args.noise_snr,
        )
    bitrate_svd = float(get_bitrate(s_normalized, noise_level, time_resolution=effective_time_resolution))
    print(f"noise_level: {noise_level}")
    print(f"bitrate: {bitrate_svd}")
    print(n_sensors)
    # print(f"bitrate: {get_bitrate_channel_capacity(s, args.noise_snr, nsensors_reference=n_sensors, n_sensors=n_sensors, time_resolution=1.0)}")

if bitrate_method in {"slq", "both"}:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping SLQ bitrate approximation.")
    else:
        if args.disable_matrix_normalization:
            normalize_scale = 1.0
        else:
            normalize_scale = 1.0 / math.sqrt(len(source_positions) * len(sensor_positions))
        if args.slq_streaming:
            if noise_level is None:
                if args.noise_heuristic == "first":
                    if args.slq_probe_parallel:
                        sigma_max = estimate_spectral_norm_streaming_probe_parallel(
                            make_compute_chunk_matrix_for_device,
                            stream_device_ids,
                            num_sensors_total,
                            num_sources_total,
                            nt,
                            sensor_batch_size,
                            normalize_scale=normalize_scale,
                            n_iters=args.noise_iters,
                            verbose=args.noise_verbose,
                        )
                    else:
                        sigma_max = estimate_spectral_norm_streaming(
                            compute_chunk_matrix,
                            num_sensors_total,
                            num_sources_total,
                            nt,
                            sensor_batch_size,
                            normalize_scale=normalize_scale,
                            n_iters=args.noise_iters,
                            device="cuda",
                            verbose=args.noise_verbose,
                        )
                    noise_level = sigma_max / args.noise_snr
                else:
                    if args.slq_probe_parallel:
                        frob_sq = estimate_frobenius_norm_sq_streaming_probe_parallel(
                            make_compute_chunk_matrix_for_device,
                            stream_device_ids,
                            num_sensors_total,
                            num_sources_total,
                            nt,
                            sensor_batch_size,
                            normalize_scale=normalize_scale,
                            n_probes=args.noise_power_probes,
                            verbose=args.noise_verbose,
                        )
                    else:
                        frob_sq = estimate_frobenius_norm_sq_streaming(
                            compute_chunk_matrix,
                            num_sensors_total,
                            num_sources_total,
                            nt,
                            sensor_batch_size,
                            normalize_scale=normalize_scale,
                            n_probes=args.noise_power_probes,
                            device="cuda",
                            verbose=args.noise_verbose,
                        )
                    noise_level = _sqrt_nonnegative_estimate(
                        frob_sq,
                        "streaming Frobenius norm estimate",
                    ) / args.noise_snr
                print(f"noise_level (estimated): {noise_level}")
            print("Computing bitrate using SLQ (streaming)")
            if args.slq_probe_parallel:
                if stream_device_ids is None:
                    raise ValueError("stream_device_ids not initialized for probe-parallel streaming")
                bitrate_slq = bitrate_slq_torch_gpu_streaming_probe_parallel(
                    make_compute_chunk_matrix_for_device,
                    stream_device_ids,
                    num_sensors_total,
                    num_sources_total,
                    nt,
                    sensor_batch_size,
                    noise_std_full_brain=noise_level,
                    time_resolution=effective_time_resolution,
                    s=args.slq_s,
                    t=args.slq_t,
                    batch=args.slq_batch,
                    normalize_scale=normalize_scale,
                    verbose=args.slq_verbose,
                )
            else:
                slq_device_ids = [torch.cuda.current_device()]
                bitrate_slq = bitrate_slq_torch_gpu_streaming(
                    compute_chunk_matrix,
                    num_sensors_total,
                    num_sources_total,
                    nt,
                    sensor_batch_size,
                    noise_std_full_brain=noise_level,
                    time_resolution=effective_time_resolution,
                    s=args.slq_s,
                    t=args.slq_t,
                    batch=args.slq_batch,
                    normalize_scale=normalize_scale,
                    verbose=args.slq_verbose,
                )
        else:
            G_cpu = G if G.device.type == "cpu" else G.cpu()
            if noise_level is None:
                if args.noise_heuristic == "first":
                    sigma_max = estimate_spectral_norm_chunked(
                        G_cpu,
                        normalize_scale=normalize_scale,
                        n_iters=args.noise_iters,
                        device="cuda",
                        chunk_rows=args.slq_chunk_rows,
                        verbose=args.noise_verbose,
                    )
                    noise_level = sigma_max / args.noise_snr
                else:
                    frob_sq = estimate_frobenius_norm_sq_hutchinson(
                        G_cpu,
                        normalize_scale=normalize_scale,
                        n_probes=args.noise_power_probes,
                        device="cuda",
                        chunk_rows=args.slq_chunk_rows,
                        verbose=args.noise_verbose,
                    )
                    noise_level = _sqrt_nonnegative_estimate(
                        frob_sq,
                        "chunked Frobenius norm estimate",
                    ) / args.noise_snr
                print(f"noise_level (estimated): {noise_level}")
            print("Computing bitrate using SLQ")
            if args.slq_probe_parallel:
                if args.slq_devices:
                    device_ids = [int(x) for x in args.slq_devices.split(",") if x.strip() != ""]
                else:
                    device_ids = list(range(torch.cuda.device_count()))
                slq_device_ids = device_ids
                bitrate_slq = bitrate_slq_torch_gpu_chunked_probe_parallel(
                    G_cpu,
                    noise_std_full_brain=noise_level,
                    time_resolution=effective_time_resolution,
                    s=args.slq_s,
                    t=args.slq_t,
                    batch=args.slq_batch,
                    device_ids=device_ids,
                    chunk_rows=args.slq_chunk_rows,
                    normalize_scale=normalize_scale,
                    verbose=args.slq_verbose,
                )
            elif args.slq_multi_gpu:
                if args.slq_devices:
                    device_ids = [int(x) for x in args.slq_devices.split(",") if x.strip() != ""]
                else:
                    device_ids = list(range(torch.cuda.device_count()))
                slq_device_ids = device_ids
                bitrate_slq = bitrate_slq_torch_multi_gpu_sharded(
                    G_cpu,
                    noise_std_full_brain=noise_level,
                    time_resolution=effective_time_resolution,
                    s=args.slq_s,
                    t=args.slq_t,
                    batch=args.slq_batch,
                    device_ids=device_ids,
                    normalize_scale=normalize_scale,
                    verbose=args.slq_verbose,
                )
            else:
                bitrate_slq = bitrate_slq_torch_gpu_chunked(
                    G_cpu,
                    noise_std_full_brain=noise_level,
                    time_resolution=effective_time_resolution,
                    s=args.slq_s,
                    t=args.slq_t,
                    batch=args.slq_batch,
                    chunk_rows=args.slq_chunk_rows,
                    normalize_scale=normalize_scale,
                    verbose=args.slq_verbose,
                )
                slq_device_ids = [torch.cuda.current_device()]
        print(f"bitrate (SLQ GPU): {bitrate_slq}")

result_record = {
    "status": "ok",
    "n_sources": int(len(source_positions)),
    "n_sensors": int(len(sensor_positions)),
    "frequency_hz": float(center_frequency),
    "frequency_khz": float(center_frequency / 1_000.0),
    "temporal_sampling": int(args.temporal_sampling),
    "time_step_seconds": float(time_step),
    "effective_time_resolution_seconds": float(effective_time_resolution),
    "sensor_batch_size": int(args.sensor_batch_size),
    "bitrate_method": bitrate_method,
    "bitrate": bitrate_slq if bitrate_slq is not None else bitrate_svd,
    "bitrate_slq": bitrate_slq,
    "bitrate_svd": bitrate_svd,
    "noise_level": noise_level,
    "matrix_normalization": "disabled" if args.disable_matrix_normalization else "enabled",
    "slq_streaming": bool(args.slq_streaming),
    "slq_probe_parallel": bool(args.slq_probe_parallel),
    "slq_multi_gpu": bool(args.slq_multi_gpu),
    "slq_device_ids": slq_device_ids if slq_device_ids is not None else stream_device_ids,
    "saved_svd_path": saved_svd_path,
}
_emit_result_json(result_record)

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
