#!/usr/bin/env python3
import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import torch

from guti import svd as svd_utils


def parse_shape(shape: Optional[str], m: Optional[int], n: Optional[int]) -> Tuple[int, int]:
    if shape:
        cleaned = shape.lower().replace("x", ",").replace(" ", "")
        parts = [p for p in cleaned.split(",") if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid shape string: {shape}")
        return int(parts[0]), int(parts[1])
    if m is None or n is None:
        raise ValueError("Provide --shape or both --m and --n")
    return m, n


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def maybe_sync_torch(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maybe_sync_cupy():
    cp = svd_utils._maybe_import_cupy()
    if cp is None:
        return
    cp.cuda.Stream.null.synchronize()


def clear_caches():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cp = svd_utils._maybe_import_cupy()
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def load_matrix(path: str, device: torch.device, dtype: torch.dtype, key: Optional[str] = None) -> torch.Tensor:
    if path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if not key:
                raise ValueError("Loaded a dict. Provide --key to select the tensor.")
            data = data[key]
        if not torch.is_tensor(data):
            raise ValueError(f"Expected tensor in {path}, got {type(data)}")
        return data.to(device=device, dtype=dtype)

    if path.endswith(".npy"):
        arr = np.load(path)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    if path.endswith(".npz"):
        if not key:
            raise ValueError("Provide --key for .npz files")
        arr = np.load(path)[key]
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    raise ValueError(f"Unsupported file type: {path}")


def run_torch_svdvals(
    G: torch.Tensor,
    backend: Optional[str],
    driver: Optional[str],
) -> Tuple[np.ndarray, dict]:
    maybe_sync_torch(G.device)
    start = time.perf_counter()
    s = svd_utils.torch_svdvals(G, driver=driver, linalg_backend=backend, return_numpy=True)
    maybe_sync_torch(G.device)
    elapsed = time.perf_counter() - start
    return s, {"total_s": elapsed}


def run_torch_gram(
    G: torch.Tensor,
    backend: Optional[str],
    use_fp16_matmul: bool,
    gram_dtype: torch.dtype,
    symmetrize: bool,
) -> Tuple[np.ndarray, dict]:
    maybe_sync_torch(G.device)
    start = time.perf_counter()
    gram, side = svd_utils.torch_gram_matrix(G, use_fp16_matmul=use_fp16_matmul, gram_dtype=gram_dtype)
    maybe_sync_torch(G.device)
    t_matmul = time.perf_counter() - start

    with svd_utils._torch_linalg_backend(backend):
        maybe_sync_torch(G.device)
        start = time.perf_counter()
        if symmetrize:
            gram = 0.5 * (gram + gram.T)
        eigvals = torch.linalg.eigvalsh(gram)
        maybe_sync_torch(G.device)
        t_eig = time.perf_counter() - start

    s = eigvals.clamp_min(0).sqrt_().flip(0).cpu().numpy()
    return s, {"total_s": t_matmul + t_eig, "matmul_s": t_matmul, "eig_s": t_eig, "gram_side": side}


def run_cupy_svdvals(G: torch.Tensor) -> Tuple[np.ndarray, dict]:
    maybe_sync_cupy()
    start = time.perf_counter()
    s = svd_utils.cupy_svdvals(G, return_numpy=True)
    maybe_sync_cupy()
    elapsed = time.perf_counter() - start
    return s, {"total_s": elapsed}


def run_cupy_gram(
    G: torch.Tensor,
    use_fp16_matmul: bool,
    gram_dtype: str,
    symmetrize: bool,
) -> Tuple[np.ndarray, dict]:
    maybe_sync_cupy()
    start = time.perf_counter()
    gram, side = svd_utils.cupy_gram_matrix(G, use_fp16_matmul=use_fp16_matmul, gram_dtype=gram_dtype)
    maybe_sync_cupy()
    t_matmul = time.perf_counter() - start

    maybe_sync_cupy()
    start = time.perf_counter()
    if symmetrize:
        gram = 0.5 * (gram + gram.T)
    cp = svd_utils._maybe_import_cupy()
    eigvals = cp.linalg.eigvalsh(gram)
    s = cp.sqrt(cp.maximum(eigvals, 0.0))[::-1]
    maybe_sync_cupy()
    t_eig = time.perf_counter() - start

    return cp.asnumpy(s), {"total_s": t_matmul + t_eig, "matmul_s": t_matmul, "eig_s": t_eig, "gram_side": side}


def build_runs(args) -> List[dict]:
    runs = []
    methods = set(args.methods or [])
    try_all = args.try_all

    backends = args.backends or []
    drivers = args.drivers or []

    if try_all and not backends:
        backends = ["default", "cusolver", "magma"]
    if try_all and not drivers:
        drivers = ["default", "gesvd", "gesvdj", "gesvda"]

    if try_all or "torch_gram_eigvalsh" in methods:
        for backend in backends or ["default"]:
            runs.append({
                "name": "torch_gram_eigvalsh",
                "backend": backend,
            })

    if try_all or "torch_svdvals" in methods:
        for backend in backends or ["default"]:
            for driver in drivers or ["default"]:
                runs.append({
                    "name": "torch_svdvals",
                    "backend": backend,
                    "driver": driver,
                })

    if try_all or "cupy_gram_eigvalsh" in methods:
        runs.append({"name": "cupy_gram_eigvalsh"})

    if try_all or "cupy_svdvals" in methods:
        runs.append({"name": "cupy_svdvals"})

    if not runs:
        runs.append({"name": "torch_svdvals", "backend": "default", "driver": "default"})

    return runs


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU singular value methods (all singular values).")
    parser.add_argument("--shape", type=str, default=None, help="Matrix shape like 20000x30000")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--matrix-path", type=str, default=None)
    parser.add_argument("--key", type=str, default=None, help="Key for .npz or dict .pt files")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--try-all", action="store_true", help="Run all torch/cupy methods and variants")
    parser.add_argument("--methods", nargs="*", default=[], help="Methods to run")
    parser.add_argument("--backends", nargs="*", default=[], help="Torch backends: default, cusolver, magma")
    parser.add_argument("--drivers", nargs="*", default=[], help="Torch SVD drivers: default, gesvd, gesvdj, gesvda")

    parser.add_argument("--fp16-matmul", action="store_true", help="Use fp16 matmul for Gram methods")
    parser.add_argument("--gram-dtype", type=str, default="float32", help="dtype for Gram matrix")
    parser.add_argument("--symmetrize-gram", action="store_true", help="Force Gram to be symmetric")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 matmul in torch")
    parser.add_argument("--matmul-precision", type=str, default=None, help="torch.set_float32_matmul_precision")
    parser.add_argument("--no-clear-cache", action="store_true", help="Do not clear CUDA memory pools between runs")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    m, n = parse_shape(args.shape, args.m, args.n)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    if args.matrix_path:
        G = load_matrix(args.matrix_path, device=device, dtype=dtype, key=args.key)
    else:
        torch.manual_seed(args.seed)
        G = torch.randn((m, n), device=device, dtype=dtype)

    if not G.is_contiguous():
        G = G.contiguous()

    print(f"Matrix: {tuple(G.shape)}, dtype={G.dtype}, device={G.device}")
    print(f"Size (GB): {G.numel() * G.element_size() / (1024 ** 3):.2f}")

    runs = build_runs(args)

    rows = []
    for run in runs:
        name = run["name"]
        backend = run.get("backend")
        driver = run.get("driver")
        label = name
        if backend:
            label += f" [backend={backend}]"
        if driver:
            label += f" [driver={driver}]"

        try:
            if name == "torch_svdvals":
                s, stats = run_torch_svdvals(G, backend, driver)
            elif name == "torch_gram_eigvalsh":
                gram_dtype = dtype_from_string(args.gram_dtype)
                s, stats = run_torch_gram(G, backend, args.fp16_matmul, gram_dtype, args.symmetrize_gram)
            elif name == "cupy_svdvals":
                s, stats = run_cupy_svdvals(G)
            elif name == "cupy_gram_eigvalsh":
                s, stats = run_cupy_gram(G, args.fp16_matmul, args.gram_dtype, args.symmetrize_gram)
            else:
                raise ValueError(f"Unknown method: {name}")

            rows.append({
                "label": label,
                "status": "ok",
                "total_s": stats.get("total_s"),
                "matmul_s": stats.get("matmul_s"),
                "eig_s": stats.get("eig_s"),
                "gram_side": stats.get("gram_side"),
                "sv_len": len(s),
            })
        except Exception as exc:
            rows.append({
                "label": label,
                "status": f"fail: {exc}",
                "total_s": None,
                "matmul_s": None,
                "eig_s": None,
                "gram_side": None,
                "sv_len": None,
            })

        if not args.no_clear_cache:
            clear_caches()

    print("\nResults:")
    header = f"{'method':46} {'status':20} {'total_s':>10} {'matmul_s':>10} {'eig_s':>10} {'gram':>6} {'sv_len':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        total = f"{row['total_s']:.3f}" if row["total_s"] is not None else "-"
        matmul = f"{row['matmul_s']:.3f}" if row["matmul_s"] is not None else "-"
        eig = f"{row['eig_s']:.3f}" if row["eig_s"] is not None else "-"
        gram = row["gram_side"] or "-"
        sv_len = str(row["sv_len"]) if row["sv_len"] is not None else "-"
        print(f"{row['label'][:46]:46} {row['status'][:20]:20} {total:>10} {matmul:>10} {eig:>10} {gram:>6} {sv_len:>7}")


if __name__ == "__main__":
    main()
