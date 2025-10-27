import torch
import numpy as np
import matplotlib.pyplot as plt


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
    # Try with different backend
    try:
        # Set preferred backend to magma if available
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library("magma")
        s = torch.linalg.svdvals(Jac_gpu)
        s = s.cpu().numpy()
        torch.backends.cuda.preferred_linalg_library(original_backend)
        return s
    except Exception as e2:
        print(f"Magma backend failed: {e2}")
        
        # Try with cusolver backend
        try:
            torch.backends.cuda.preferred_linalg_library("cusolver")
            s = torch.linalg.svdvals(Jac_gpu)
            s = s.cpu().numpy()
            torch.backends.cuda.preferred_linalg_library(original_backend)
            return s
        except Exception as e3:
            print(f"Cusolver backend failed: {e3}")
            
            # Final fallback: use CPU computation
            print("All GPU methods failed. Computing SVD on CPU...")
            Jac_gpu = None  # Free GPU memory
            torch.cuda.empty_cache()
            return compute_svd_cpu(Jac_gpu.cpu().numpy())

def compute_svd_gpu(Jac_gpu: torch.Tensor) -> np.ndarray:
    if not isinstance(Jac_gpu, torch.Tensor):
        Jac_gpu = torch.from_numpy(Jac_gpu)
    
    try:
        s = torch.linalg.svdvals(Jac_gpu)
        s = s.cpu().numpy()
        return s
    except Exception as e:
        print(f"Default SVD failed: {e}")
        return compute_svd_gpu_fallback(Jac_gpu)


def plot_svd(s):
    plt.figure()
    plt.semilogy(s / s[0])
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
    plt.ylim(1e-5, 1)  # Set y-axis limits from 1e-5 to 1
