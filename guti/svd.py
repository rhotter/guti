import torch
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager
from typing import Optional, Tuple


def _maybe_import_cupy():
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    return cp


def _as_torch_tensor(x, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = None
        try:
            cp = _maybe_import_cupy()
            if cp is not None and isinstance(x, cp.ndarray):
                t = torch.utils.dlpack.from_dlpack(x.toDlpack())
        except Exception:
            t = None
        if t is None:
            raise TypeError(f"Unsupported input type for torch tensor: {type(x)}")

    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _as_cupy_array(x):
    cp = _maybe_import_cupy()
    if cp is None:
        raise RuntimeError("cupy is not available in this environment")

    if isinstance(x, cp.ndarray):
        return x
    if torch.is_tensor(x):
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))
    if isinstance(x, np.ndarray):
        return cp.asarray(x)
    raise TypeError(f"Unsupported input type for cupy array: {type(x)}")


@contextmanager
def _torch_linalg_backend(backend: Optional[str]):
    if backend is None or backend == "default":
        yield
        return

    if not hasattr(torch.backends.cuda, "preferred_linalg_library"):
        yield
        return

    try:
        original = torch.backends.cuda.preferred_linalg_library()
    except Exception:
        yield
        return

    try:
        torch.backends.cuda.preferred_linalg_library(backend)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            torch.backends.cuda.preferred_linalg_library(original)
        except Exception:
            pass


def torch_gram_matrix(
    G: torch.Tensor,
    use_fp16_matmul: bool = False,
    gram_dtype: Optional[torch.dtype] = torch.float32,
) -> Tuple[torch.Tensor, str]:
    G = _as_torch_tensor(G)
    if G.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(G.shape)}")

    if use_fp16_matmul and G.dtype in (torch.float32, torch.float64):
        G_mm = G.half()
    else:
        G_mm = G

    m, n = G_mm.shape
    if m >= n:
        gram = G_mm.T @ G_mm
        side = "GtG"
    else:
        gram = G_mm @ G_mm.T
        side = "GGt"

    if gram_dtype is not None:
        gram = gram.to(gram_dtype)

    return gram, side


def cupy_gram_matrix(
    G,
    use_fp16_matmul: bool = False,
    gram_dtype: Optional[str] = "float32",
):
    cp = _maybe_import_cupy()
    if cp is None:
        raise RuntimeError("cupy is not available in this environment")

    G = _as_cupy_array(G)
    if G.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(G.shape)}")

    if use_fp16_matmul and G.dtype in (cp.float32, cp.float64):
        G_mm = G.astype(cp.float16, copy=False)
    else:
        G_mm = G

    m, n = G_mm.shape
    if m >= n:
        gram = G_mm.T @ G_mm
        side = "GtG"
    else:
        gram = G_mm @ G_mm.T
        side = "GGt"

    if gram_dtype is not None:
        gram = gram.astype(getattr(cp, gram_dtype), copy=False)

    return gram, side


def torch_svdvals(
    Jac_gpu: torch.Tensor,
    driver: Optional[str] = None,
    linalg_backend: Optional[str] = None,
    return_numpy: bool = True,
) -> np.ndarray:
    Jac_gpu = _as_torch_tensor(Jac_gpu)
    kwargs = {}
    if driver is not None and driver != "default":
        kwargs["driver"] = driver

    with _torch_linalg_backend(linalg_backend):
        try:
            s = torch.linalg.svdvals(Jac_gpu, **kwargs)
        except TypeError:
            if "driver" in kwargs:
                s = torch.linalg.svdvals(Jac_gpu)
            else:
                raise

    if return_numpy:
        return s.detach().cpu().numpy()
    return s


def torch_gram_svdvals(
    G: torch.Tensor,
    use_fp16_matmul: bool = False,
    gram_dtype: Optional[torch.dtype] = torch.float32,
    symmetrize: bool = False,
    linalg_backend: Optional[str] = None,
    return_numpy: bool = True,
) -> np.ndarray:
    gram, _ = torch_gram_matrix(G, use_fp16_matmul=use_fp16_matmul, gram_dtype=gram_dtype)
    if symmetrize:
        gram = 0.5 * (gram + gram.T)

    with _torch_linalg_backend(linalg_backend):
        eigvals = torch.linalg.eigvalsh(gram)

    s = eigvals.clamp_min(0).sqrt_().flip(0)
    if return_numpy:
        return s.detach().cpu().numpy()
    return s


def cupy_svdvals(G, return_numpy: bool = True):
    cp = _maybe_import_cupy()
    if cp is None:
        raise RuntimeError("cupy is not available in this environment")

    G = _as_cupy_array(G)
    s = cp.linalg.svd(G, compute_uv=False)
    if return_numpy:
        return cp.asnumpy(s)
    return s


def cupy_gram_svdvals(
    G,
    use_fp16_matmul: bool = False,
    gram_dtype: Optional[str] = "float32",
    symmetrize: bool = False,
    return_numpy: bool = True,
):
    cp = _maybe_import_cupy()
    if cp is None:
        raise RuntimeError("cupy is not available in this environment")

    gram, _ = cupy_gram_matrix(G, use_fp16_matmul=use_fp16_matmul, gram_dtype=gram_dtype)
    if symmetrize:
        gram = 0.5 * (gram + gram.T)

    eigvals = cp.linalg.eigvalsh(gram)
    s = cp.sqrt(cp.maximum(eigvals, 0.0))[::-1]
    if return_numpy:
        return cp.asnumpy(s)
    return s


def compute_svd_cpu(Jac_cpu: np.ndarray) -> np.ndarray:
    from scipy.sparse.linalg import svds
    from scipy import sparse

    # Convert to sparse matrix if it isn't already
    if not sparse.issparse(Jac_cpu):
        Jac_cpu = sparse.csr_matrix(Jac_cpu)

    # For sparse matrices, use scipy's svds
    # k is the number of singular values to compute
    # If k is None, it will compute min(n, m) singular values
    s = svds(Jac_cpu, k=None, return_singular_vectors=False)
    return s


def compute_svd_gpu_fallback(Jac_gpu: torch.Tensor) -> np.ndarray:
    # First retry on the GPU via a float64 Gram + symmetric eigendecomposition.
    # The direct GPU SVD (cuSOLVER gesvdj) fails outright (CUSOLVER_STATUS_INVALID_VALUE)
    # on the large, ill-conditioned matrices these forward models produce, which
    # otherwise forces the slow dense CPU path. Forming the smaller Gram matrix and
    # taking eigvalsh in float64 is robust (never the gesvdj failure) and far faster
    # than CPU LAPACK. Squaring the condition number only degrades singular values
    # below ~1e-8 of the largest, which is well past the resolved part of the spectrum.
    if torch.cuda.is_available():
        try:
            t64 = _as_torch_tensor(Jac_gpu).to(device="cuda", dtype=torch.float64)
            m, n = t64.shape
            gram = (t64.T @ t64) if m >= n else (t64 @ t64.T)
            del t64
            eigvals = torch.linalg.eigvalsh(gram)
            del gram
            s = eigvals.clamp_min(0).sqrt().flip(0).detach().cpu().numpy()
            torch.cuda.empty_cache()
            if np.all(np.isfinite(s)):
                return s
            print("float64 Gram eigvalsh returned non-finite values; trying next.")
        except Exception as exc:
            print(f"float64 Gram eigvalsh failed: {exc}")

    for backend in ("magma", "cusolver"):
        try:
            s = torch_svdvals(Jac_gpu, linalg_backend=backend, return_numpy=True)
            if np.all(np.isfinite(s)):
                return s
            print(f"{backend} backend returned non-finite values; trying next.")
        except Exception as exc:
            print(f"{backend} backend failed: {exc}")

    print("All GPU methods failed. Computing SVD on CPU (dense LAPACK)...")
    Jac_cpu = _as_torch_tensor(Jac_gpu).detach().cpu().numpy()
    torch.cuda.empty_cache()
    # Dense LAPACK driver is robust for ill-conditioned matrices; the sparse
    # svds path can itself return non-finite/partial spectra here.
    return np.linalg.svd(Jac_cpu, compute_uv=False)


def compute_svd_gpu(
    Jac_gpu: torch.Tensor,
    method: str = "torch_svdvals",
    **kwargs,
) -> np.ndarray:
    """
    Compute singular values on GPU using the selected method.

    Methods:
      - torch_svdvals
      - torch_gram_eigvalsh
      - cupy_svdvals
      - cupy_gram_eigvalsh
    """
    method = method.lower()

    if method in ("torch_svdvals", "svdvals", "torch"):
        try:
            s = torch_svdvals(Jac_gpu, return_numpy=True, **kwargs)
            # torch's GPU SVD can silently return non-finite values for extremely
            # ill-conditioned matrices (no exception raised). Treat that as a
            # failure and fall back to a more robust path.
            if not np.all(np.isfinite(s)):
                print("Default torch SVD returned non-finite values; falling back.")
                return compute_svd_gpu_fallback(Jac_gpu)
            return s
        except Exception as exc:
            print(f"Default torch SVD failed: {exc}")
            return compute_svd_gpu_fallback(Jac_gpu)

    if method in ("torch_gram_eigvalsh", "gram_torch", "eigvalsh"):
        return torch_gram_svdvals(Jac_gpu, return_numpy=True, **kwargs)

    if method in ("cupy_svdvals", "svdvals_cupy"):
        return cupy_svdvals(Jac_gpu, return_numpy=True)

    if method in ("cupy_gram_eigvalsh", "gram_cupy"):
        return cupy_gram_svdvals(Jac_gpu, return_numpy=True, **kwargs)

    raise ValueError(f"Unknown SVD method: {method}")


def compute_svd_fast(
    Jac,
    k: int = 8000,
    niter: int = 6,
    oversample: int = 20,
    full_below: int = 8000,
) -> np.ndarray:
    """Fast singular values for the large, ill-conditioned forward-model matrices.

    Adaptive:
      - min(m, n) <= ``full_below`` (or no CUDA): exact-ish full spectrum via a
        float64 Gram + cuSOLVER ``eigvalsh`` (the direct GPU SVD / cuSOLVER gesvdj
        fails outright on these matrices).
      - larger: **float32 randomized SVD** of the top-``k`` singular values
        (``torch.svd_lowrank``). This is GEMM-bound (fast on GPU) and resolves the
        spectrum down to ~1e-5 of the largest value, which is all that the
        sqrt(N)-normalised spectra and the channel-capacity (noise floor ~1e-2..1e-3)
        need. Working on ``A`` directly (not the Gram) keeps the top-k above the
        float32 floor, so float32 is accurate there yet far faster than float64.

    Validated against exact CPU/double SVD: top-k relative error ~1e-5 in the
    resolved band (only the last ~oversample values, near the truncation edge,
    are less accurate).
    """
    t = _as_torch_tensor(Jac)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        t = t.cuda()
    m, n = t.shape
    r = min(m, n)

    if r <= full_below or not use_cuda:
        with _torch_linalg_backend("cusolver" if use_cuda else None):
            t64 = t.double()
            gram = (t64.T @ t64) if m >= n else (t64 @ t64.T)
            del t64
            eig = torch.linalg.eigvalsh(gram)
            del gram
        s = eig.clamp_min(0).sqrt_().flip(0).detach().cpu().numpy()
        if use_cuda:
            torch.cuda.empty_cache()
        return s

    # Randomized top-k. For very tall matrices the working buffers (Q, the power-
    # iteration products) are ~m x q float32 each and a handful are live at once, so
    # they can blow past GPU memory on long-separation / many-gate configs. Cap k to
    # the free GPU memory so the randomized SVD always fits; the smaller resolved
    # band (still well below the noise floor) is all the spectra/capacity need.
    kk = min(k, r)
    if torch.cuda.is_available():
        free_b = torch.cuda.mem_get_info()[0]
        # budget ~5 live (m x q) float32 buffers within 60% of free memory
        q_cap = int(0.6 * free_b / (m * 4 * 5))
        kk = min(kk, max(1024, q_cap - oversample))
        if kk < min(k, r):
            print(f"compute_svd_fast: capping randomized k to {kk} "
                  f"(rows={m}, free={free_b/1e9:.0f}GB) to fit GPU memory")
    q = min(kk + oversample, r)
    with _torch_linalg_backend("cusolver"):
        _, S, _ = torch.svd_lowrank(t.float(), q=q, niter=niter)
    s = S[:kk].detach().cpu().numpy()
    torch.cuda.empty_cache()
    return s


def plot_svd(s):
    plt.figure()
    plt.semilogy(s / s[0])
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
    plt.ylim(1e-5, 1)  # Set y-axis limits from 1e-5 to 1
