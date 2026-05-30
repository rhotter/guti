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
    for backend in ("magma", "cusolver"):
        try:
            return torch_svdvals(Jac_gpu, linalg_backend=backend, return_numpy=True)
        except Exception as exc:
            print(f"{backend} backend failed: {exc}")

    print("All GPU methods failed. Computing SVD on CPU...")
    Jac_cpu = _as_torch_tensor(Jac_gpu).detach().cpu().numpy()
    torch.cuda.empty_cache()
    return compute_svd_cpu(Jac_cpu)


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
            return torch_svdvals(Jac_gpu, return_numpy=True, **kwargs)
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


def plot_svd(s):
    plt.figure()
    plt.semilogy(s / s[0])
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
    plt.ylim(1e-5, 1)  # Set y-axis limits from 1e-5 to 1
