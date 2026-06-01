"""Stochastic Lanczos Quadrature (SLQ) bitrate estimator.

This is a general, modality-agnostic alternative to the SVD bitrate pipeline.
Given a forward operator ``A`` (a :class:`guti.linop.ForwardOperator`) and a
detector noise level, it estimates the Gaussian-channel bitrate

    C = 1/(2 * time_resolution) * Σ_i log₂(1 + σ_i² / noise²)
      = 1/(2 * time_resolution) * (1/ln2) * tr ln(I + (1/noise²) A Aᵀ)

WITHOUT computing the SVD or ever materializing ``A``. The trace-log is
estimated by stochastic trace estimation (Hutchinson) with each quadratic form
evaluated via Lanczos quadrature. The only thing touched matrix-side is
``A @ x`` / ``Aᵀ @ x``, so a matrix-free :class:`ChunkedForwardOperator` works
just as well as a dense one.

This convention matches :func:`guti.core.get_bitrate` (equal/iid input power),
so on a small dense operator ``bitrate_slq`` agrees with
``get_bitrate(svdvals(A), noise)`` — see ``tests/test_slq.py``. (It is NOT the
water-filling capacity, which would require the spectrum SLQ avoids.)

Backends: works for NumPy and PyTorch operators. Matvecs and probes stay in the
operator's backend; the tiny per-probe tridiagonal eigendecomposition is done
in NumPy.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

from guti.linop import ForwardOperator, as_operator, transpose


def _scalar(x) -> float:
    """Extract a Python float from a (1,1) NumPy/torch result."""
    return float(x.reshape(()) if hasattr(x, "reshape") else x)


def _dot(a, b) -> float:
    return _scalar(transpose(a) @ b)


def _norm(v) -> float:
    return math.sqrt(max(_dot(v, v), 0.0))


def _lanczos(B_mv, z, num_steps: int):
    """Lanczos on ``B`` started at ``z`` (with full reorthogonalization).

    Returns (alphas, betas) as Python-float lists defining the symmetric
    tridiagonal ``T`` (alphas on the diagonal, betas on the off-diagonal).
    ``B_mv(v)`` applies the (implicit) SPD matrix ``B`` to a ``(d, 1)`` vector.
    """
    beta = _norm(z)
    if beta == 0.0:
        return [0.0], []
    q = z / beta
    Q = [q]
    alphas: list[float] = []
    betas: list[float] = []
    q_prev = None
    beta_prev = 0.0
    for k in range(num_steps):
        w = B_mv(q)
        if k > 0:
            w = w - beta_prev * q_prev
        a = _dot(q, w)
        alphas.append(a)
        w = w - a * q
        # Full reorthogonalization against all previous Lanczos vectors.
        for qi in Q:
            w = w - _dot(qi, w) * qi
        b = _norm(w)
        if b < 1e-12:
            break
        betas.append(b)
        q_prev = q
        beta_prev = b
        q = w / b
        Q.append(q)
    return alphas, betas


def _quadrature(alphas, betas, f) -> float:
    """Gauss quadrature weight·f(node) sum from a Lanczos tridiagonal."""
    t = len(alphas)
    T = np.zeros((t, t), dtype=np.float64)
    for i in range(t):
        T[i, i] = alphas[i]
    for i in range(len(betas)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    theta, Y = np.linalg.eigh(T)
    weights = Y[0, :] ** 2
    return float(np.sum(weights * f(theta)))


def bitrate_slq(
    A,
    noise_std: float,
    *,
    num_probes: int = 16,
    num_lanczos: int = 40,
    side: Literal["auto", "left", "right"] = "auto",
    probe: Literal["rademacher", "basis"] = "rademacher",
    time_resolution: float = 1.0,
    n_detectors: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate the equal-power bitrate of ``A`` via stochastic Lanczos quadrature.

    Parameters
    ----------
    A : ForwardOperator | array
        The forward operator (or a dense array, which is wrapped).
    noise_std : float
        Detector noise standard deviation.
    num_probes : int
        Number of stochastic probe vectors (Hutchinson samples). Ignored when
        ``probe="basis"`` (which uses all ``d`` basis vectors for an exact,
        deterministic trace — only practical for small ``d``, used in tests).
    num_lanczos : int
        Lanczos steps per probe.
    side : {"auto","left","right"}
        Whether to work with ``B = A Aᵀ`` ("left", ``d = m``) or ``B = Aᵀ A``
        ("right", ``d = n``). "auto" picks the smaller dimension.
    probe : {"rademacher","basis"}
        Probe distribution. "basis" gives an exact trace (deterministic) and is
        intended for testing on small operators.
    time_resolution : float
        Same meaning as in :func:`guti.core.get_bitrate`.
    n_detectors : int, optional
        If given, the noise variance is divided by ``n_detectors`` (matches the
        ultrasound SLQ convention of per-detector noise scaling).
    rng : numpy.random.Generator, optional
        Source of randomness for reproducible probes.

    Returns
    -------
    float
        Estimated bitrate in bits per sample-period.
    """
    op: ForwardOperator = as_operator(A)
    m, n = op.shape
    if side == "auto":
        left = m <= n
    else:
        left = side == "left"
    d = m if left else n

    noise_var = noise_std ** 2
    if n_detectors is not None:
        noise_var = noise_var / n_detectors

    def B_mv(v):
        # Apply B = A Aᵀ (left) or Aᵀ A (right) to a (d, 1) vector.
        if left:
            return op.matvec(op.rmatvec(v))
        return op.rmatvec(op.matvec(v))

    def f(theta):
        # f(λ) = ln(1 + λ / noise_var); clamp tiny negatives from roundoff.
        return np.log1p(np.clip(theta, 0.0, None) / noise_var)

    if probe == "basis":
        # Exact trace: sum of e_iᵀ f(B) e_i over the standard basis.
        total = 0.0
        for i in range(d):
            z = op.zeros((d, 1))
            z[i] = 1.0
            alphas, betas = _lanczos(B_mv, z, num_lanczos)
            total += _quadrature(alphas, betas, f)  # ||e_i||² = 1
        trace_est = total
    else:
        if rng is None:
            rng = np.random.default_rng()
        total = 0.0
        for _ in range(num_probes):
            z = op.rademacher((d, 1), rng)
            znorm2 = _dot(z, z)  # = d for Rademacher
            alphas, betas = _lanczos(B_mv, z, num_lanczos)
            total += znorm2 * _quadrature(alphas, betas, f)
        trace_est = total / num_probes

    return (0.5 / math.log(2.0)) * trace_est / time_resolution


# ===========================================================================
# Torch GPU SLQ (production, chunked / streaming / multi-GPU)
# ===========================================================================
#
# These are the heavy, torch-based estimators used for the large ultrasound
# sweeps (and any future large modality). They were originally written in
# guti/modalities/us/analytical.py and are kept here, the canonical SLQ home,
# so they are importable without pulling in jax/jwave (which us.utils requires).
# guti.modalities.us.analytical re-exports them for backward compatibility.
#
# They operate matrix-free via a row-chunk builder (compute_chunk_matrix) or a
# pre-built CPU matrix, mirroring the ChunkedForwardOperator concept above but
# specialized for GPU throughput. Requires torch; tqdm is optional at import.
# ---------------------------------------------------------------------------

import time as _time  # noqa: F401  (block below references `time`)
import time
try:
    import torch
    import torch.cuda.comm as cuda_comm
except Exception:  # pragma: no cover - torch is required to call these
    torch = None
    cuda_comm = None
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *a, **k):
        return x
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    The exact SVD path computes sum_i log2(1 + sigma_i^2 / noise^2) under unit
    input power per source mode. This routine estimates the equivalent trace-log
    expression
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
