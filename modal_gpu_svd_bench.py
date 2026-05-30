"""
Modal benchmark for GPU singular value methods (all singular values).
Runs torch svdvals with drivers/backends and Gram+eigvalsh via torch and cupy.
"""

import argparse
import os
from typing import Optional, List, Dict

import modal


image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.02-py3",
        add_python=None,
    )
    .pip_install("cupy-cuda12x", "scipy")
)

app = modal.App("gpu-svd-benchmark", image=image)


@app.function(gpu="A100", timeout=7200)
def benchmark_gpu_svd(
    m: int,
    n: int,
    dtype: str = "float32",
    try_all: bool = True,
    methods: Optional[List[str]] = None,
    backends: Optional[List[str]] = None,
    drivers: Optional[List[str]] = None,
    fp16_matmul: bool = True,
    gram_dtype: str = "float32",
    symmetrize_gram: bool = False,
    allow_tf32: bool = True,
    matmul_precision: Optional[str] = "high",
    seed: int = 0,
    clear_cache: bool = True,
):
    import time
    import torch
    import numpy as np
    from contextlib import contextmanager

    # Inline SVD utils to avoid mount issues
    def _maybe_import_cupy():
        try:
            import cupy as cp
            return cp
        except Exception:
            return None

    @contextmanager
    def _torch_linalg_backend(backend):
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

    def torch_gram_matrix(G, use_fp16_matmul=False, gram_dtype=torch.float32):
        G_mm = G.half() if use_fp16_matmul and G.dtype in (torch.float32, torch.float64) else G
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

    def cupy_gram_matrix(G, use_fp16_matmul=False, gram_dtype="float32"):
        cp = _maybe_import_cupy()
        if torch.is_tensor(G):
            G = cp.fromDlpack(torch.utils.dlpack.to_dlpack(G))
        G_mm = G.astype(cp.float16, copy=False) if use_fp16_matmul and G.dtype in (cp.float32, cp.float64) else G
        m, n = G_mm.shape
        if m >= n:
            gram = G_mm.T @ G_mm
            side = "GtG"
        else:
            gram = G_mm @ G_mm.T
            side = "GGt"
        if gram_dtype:
            gram = gram.astype(getattr(cp, gram_dtype), copy=False)
        return gram, side

    def torch_svdvals(G, driver=None, linalg_backend=None, return_numpy=True):
        kwargs = {}
        if driver and driver != "default":
            kwargs["driver"] = driver
        with _torch_linalg_backend(linalg_backend):
            try:
                s = torch.linalg.svdvals(G, **kwargs)
            except TypeError:
                s = torch.linalg.svdvals(G)
        return s.detach().cpu().numpy() if return_numpy else s

    def cupy_svdvals(G, return_numpy=True):
        cp = _maybe_import_cupy()
        if torch.is_tensor(G):
            G = cp.fromDlpack(torch.utils.dlpack.to_dlpack(G))
        s = cp.linalg.svd(G, compute_uv=False)
        return cp.asnumpy(s) if return_numpy else s

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
        cp = _maybe_import_cupy()
        if cp is None:
            return
        cp.cuda.Stream.null.synchronize()

    def clear_caches():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cp = _maybe_import_cupy()
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def run_torch_svdvals(G: torch.Tensor, backend: Optional[str], driver: Optional[str]):
        maybe_sync_torch(G.device)
        start = time.perf_counter()
        s = torch_svdvals(G, driver=driver, linalg_backend=backend, return_numpy=True)
        maybe_sync_torch(G.device)
        elapsed = time.perf_counter() - start
        return s, {"total_s": elapsed}

    def run_torch_gram(G: torch.Tensor, backend: Optional[str]):
        gram_t = dtype_from_string(gram_dtype)
        maybe_sync_torch(G.device)
        start = time.perf_counter()
        gram, side = torch_gram_matrix(
            G, use_fp16_matmul=fp16_matmul, gram_dtype=gram_t
        )
        maybe_sync_torch(G.device)
        t_matmul = time.perf_counter() - start

        with _torch_linalg_backend(backend):
            maybe_sync_torch(G.device)
            start = time.perf_counter()
            if symmetrize_gram:
                gram = 0.5 * (gram + gram.T)
            eigvals = torch.linalg.eigvalsh(gram)
            maybe_sync_torch(G.device)
            t_eig = time.perf_counter() - start

        s = eigvals.clamp_min(0).sqrt_().flip(0).cpu().numpy()
        return s, {"total_s": t_matmul + t_eig, "matmul_s": t_matmul, "eig_s": t_eig, "gram_side": side}

    def run_cupy_svdvals(G: torch.Tensor):
        maybe_sync_cupy()
        start = time.perf_counter()
        s = cupy_svdvals(G, return_numpy=True)
        maybe_sync_cupy()
        elapsed = time.perf_counter() - start
        return s, {"total_s": elapsed}

    def run_cupy_gram(G: torch.Tensor):
        maybe_sync_cupy()
        start = time.perf_counter()
        gram, side = cupy_gram_matrix(
            G, use_fp16_matmul=fp16_matmul, gram_dtype=gram_dtype
        )
        maybe_sync_cupy()
        t_matmul = time.perf_counter() - start

        maybe_sync_cupy()
        start = time.perf_counter()
        if symmetrize_gram:
            gram = 0.5 * (gram + gram.T)
        cp = _maybe_import_cupy()
        eigvals = cp.linalg.eigvalsh(gram)
        s = cp.sqrt(cp.maximum(eigvals, 0.0))[::-1]
        maybe_sync_cupy()
        t_eig = time.perf_counter() - start
        return cp.asnumpy(s), {"total_s": t_matmul + t_eig, "matmul_s": t_matmul, "eig_s": t_eig, "gram_side": side}

    def build_runs() -> List[Dict]:
        runs = []
        methods_set = set([m.lower() for m in (methods or [])])
        bks = backends or []
        drs = drivers or []

        if try_all and not bks:
            bks = ["default", "cusolver", "magma"]
        if try_all and not drs:
            drs = ["default", "gesvd", "gesvdj", "gesvda"]

        if try_all or "torch_gram_eigvalsh" in methods_set:
            for backend in bks or ["default"]:
                runs.append({"name": "torch_gram_eigvalsh", "backend": backend})

        if try_all or "torch_svdvals" in methods_set:
            for backend in bks or ["default"]:
                for driver in drs or ["default"]:
                    runs.append({"name": "torch_svdvals", "backend": backend, "driver": driver})

        if try_all or "cupy_gram_eigvalsh" in methods_set:
            runs.append({"name": "cupy_gram_eigvalsh"})

        if try_all or "cupy_svdvals" in methods_set:
            runs.append({"name": "cupy_svdvals"})

        if not runs:
            runs.append({"name": "torch_svdvals", "backend": "default", "driver": "default"})
        return runs

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Ensure GPU torch is installed in the image.")

    torch.manual_seed(seed)
    G = torch.randn((m, n), device=device, dtype=dtype_from_string(dtype))
    if not G.is_contiguous():
        G = G.contiguous()

    runs = build_runs()
    results = []

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
                s, stats = run_torch_gram(G, backend)
            elif name == "cupy_svdvals":
                s, stats = run_cupy_svdvals(G)
            elif name == "cupy_gram_eigvalsh":
                s, stats = run_cupy_gram(G)
            else:
                raise ValueError(f"Unknown method: {name}")

            results.append({
                "label": label,
                "status": "ok",
                "total_s": stats.get("total_s"),
                "matmul_s": stats.get("matmul_s"),
                "eig_s": stats.get("eig_s"),
                "gram_side": stats.get("gram_side"),
                "sv_len": len(s),
            })
        except Exception as exc:
            results.append({
                "label": label,
                "status": f"fail: {exc}",
                "total_s": None,
                "matmul_s": None,
                "eig_s": None,
                "gram_side": None,
                "sv_len": None,
            })

        if clear_cache:
            clear_caches()

    return {
        "shape": (m, n),
        "dtype": dtype,
        "results": results,
    }


def _print_results(result: dict):
    print(f"Matrix: {result['shape']}, dtype={result['dtype']}")
    rows = result["results"]
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


@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description="Modal GPU SVD benchmark")
    parser.add_argument("--shape", type=str, default=None, help="Matrix shape like 20000x30000")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--try-all", action="store_true")
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--backends", nargs="*", default=[])
    parser.add_argument("--drivers", nargs="*", default=[])
    parser.add_argument("--fp16-matmul", action="store_true")
    parser.add_argument("--gram-dtype", type=str, default="float32")
    parser.add_argument("--symmetrize-gram", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--matmul-precision", type=str, default="high")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clear-cache", action="store_true")
    args, _ = parser.parse_known_args()

    def env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    def env_list(name: str) -> List[str]:
        raw = os.getenv(name, "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    # Allow env overrides since modal run doesn't pass CLI args to local_entrypoint.
    if args.shape is None:
        args.shape = os.getenv("MODAL_BENCH_SHAPE")
    if args.m is None:
        env_m = os.getenv("MODAL_BENCH_M")
        args.m = int(env_m) if env_m else None
    if args.n is None:
        env_n = os.getenv("MODAL_BENCH_N")
        args.n = int(env_n) if env_n else None

    if not args.try_all:
        args.try_all = env_flag("MODAL_BENCH_TRY_ALL", default=False)
    if not args.methods:
        args.methods = env_list("MODAL_BENCH_METHODS")
    if not args.backends:
        args.backends = env_list("MODAL_BENCH_BACKENDS")
    if not args.drivers:
        args.drivers = env_list("MODAL_BENCH_DRIVERS")

    if not args.fp16_matmul:
        args.fp16_matmul = env_flag("MODAL_BENCH_FP16_MATMUL", default=False)
    if not args.symmetrize_gram:
        args.symmetrize_gram = env_flag("MODAL_BENCH_SYMMETRIZE_GRAM", default=False)
    if not args.allow_tf32:
        args.allow_tf32 = env_flag("MODAL_BENCH_ALLOW_TF32", default=False)

    if args.gram_dtype == "float32":
        args.gram_dtype = os.getenv("MODAL_BENCH_GRAM_DTYPE", args.gram_dtype)
    if args.matmul_precision == "high":
        args.matmul_precision = os.getenv("MODAL_BENCH_MATMUL_PRECISION", args.matmul_precision)
    if args.seed == 0:
        env_seed = os.getenv("MODAL_BENCH_SEED")
        if env_seed:
            args.seed = int(env_seed)
    if args.no_clear_cache is False:
        args.no_clear_cache = env_flag("MODAL_BENCH_NO_CLEAR_CACHE", default=False)

    if args.shape:
        cleaned = args.shape.lower().replace("x", ",").replace(" ", "")
        parts = [p for p in cleaned.split(",") if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid shape: {args.shape}")
        m, n = int(parts[0]), int(parts[1])
    else:
        if args.m is None or args.n is None:
            raise ValueError("Provide --shape or both --m and --n")
        m, n = args.m, args.n

    result = benchmark_gpu_svd.remote(
        m=m,
        n=n,
        dtype=args.dtype,
        try_all=args.try_all,
        methods=args.methods,
        backends=args.backends,
        drivers=args.drivers,
        fp16_matmul=args.fp16_matmul,
        gram_dtype=args.gram_dtype,
        symmetrize_gram=args.symmetrize_gram,
        allow_tf32=args.allow_tf32,
        matmul_precision=args.matmul_precision,
        seed=args.seed,
        clear_cache=not args.no_clear_cache,
    )

    _print_results(result)
