# modal_svd.py
# Multi-GPU SVD for large matrices that don't fit on a single GPU
#
# Key insight: Randomized SVD only needs matrix-vector products (A @ v and A^T @ v),
# not the full matrix in memory. This allows streaming computation for arbitrarily large matrices.
#
# Two approaches implemented:
# 1. multi_gpu_randomized_svd: Keeps matrix chunks in GPU memory (fast, limited by total GPU memory)
# 2. streaming_randomized_svd: Streams chunks from disk/generates on-fly (slower, unlimited matrix size)
import modal

# Use RAPIDS image which has dask-cuda and cupy pre-installed
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12",
        add_python=None,
    )
    .pip_install("scipy")
)

app = modal.App("multi-gpu-svd", image=image)


@app.function(gpu="T4:2", timeout=1800)
def dask_multi_gpu_svd(m, n, k=50, n_oversamples=10, n_iter=2, dtype="float32"):
    """
    Multi-GPU randomized SVD using dask-cuda for distributed task scheduling.

    This implementation uses dask delayed to schedule GPU tasks across workers,
    but avoids storing large matrices in the task graph by regenerating chunks
    with deterministic seeds.
    """
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client, wait
    import cupy as cp
    import numpy as np

    cluster = LocalCUDACluster()
    client = Client(cluster)

    n_gpus = len(client.scheduler_info()['workers'])
    print(f"Dask cluster with {n_gpus} GPU workers")

    l = k + n_oversamples
    np_dtype = np.float32 if dtype == "float32" else np.float64
    matrix_size_gb = (m * n * (4 if dtype == "float32" else 8)) / (1024**3)
    print(f"Matrix size: {m} x {n} = {matrix_size_gb:.2f} GB")

    # Divide rows among GPUs
    chunk_rows = m // n_gpus
    chunks = [(i * chunk_rows, (i + 1) * chunk_rows if i < n_gpus - 1 else m)
              for i in range(n_gpus)]
    print(f"Row chunks: {chunks}")

    # Random projection
    np.random.seed(42)
    omega = np.random.randn(n, l).astype(np_dtype)

    def compute_Y_chunk_on_gpu(start, end, omega, matrix_seed, n_cols, gpu_id):
        """Generate matrix chunk and compute A_chunk @ omega on GPU."""
        import cupy as cp
        cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        with cp.cuda.Device(gpu_id):
            cp.random.seed(matrix_seed + start)
            A_chunk = cp.random.randn(end - start, n_cols, dtype=cp_dtype)
            omega_gpu = cp.asarray(omega)
            Y_chunk = A_chunk @ omega_gpu
            return cp.asnumpy(Y_chunk)

    def compute_AtY_chunk_on_gpu(start, end, Y_slice, matrix_seed, n_cols, gpu_id):
        """Regenerate matrix chunk and compute A_chunk.T @ Y on GPU."""
        import cupy as cp
        cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        with cp.cuda.Device(gpu_id):
            cp.random.seed(matrix_seed + start)
            A_chunk = cp.random.randn(end - start, n_cols, dtype=cp_dtype)
            Y_gpu = cp.asarray(Y_slice)
            result = A_chunk.T @ Y_gpu
            return cp.asnumpy(result)

    def compute_AX_chunk_on_gpu(start, end, X, matrix_seed, n_cols, gpu_id):
        """Regenerate matrix chunk and compute A_chunk @ X on GPU."""
        import cupy as cp
        cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        with cp.cuda.Device(gpu_id):
            cp.random.seed(matrix_seed + start)
            A_chunk = cp.random.randn(end - start, n_cols, dtype=cp_dtype)
            X_gpu = cp.asarray(X)
            result = A_chunk @ X_gpu
            return cp.asnumpy(result)

    def compute_QtA_chunk_on_gpu(start, end, Q_slice, matrix_seed, n_cols, gpu_id):
        """Regenerate matrix chunk and compute Q_slice.T @ A_chunk on GPU."""
        import cupy as cp
        cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        with cp.cuda.Device(gpu_id):
            cp.random.seed(matrix_seed + start)
            A_chunk = cp.random.randn(end - start, n_cols, dtype=cp_dtype)
            Q_gpu = cp.asarray(Q_slice)
            result = Q_gpu.T @ A_chunk
            return cp.asnumpy(result)

    matrix_seed = 12345

    # Step 1: Y = A @ omega
    print("Computing Y = A @ omega...")
    futures = [
        client.submit(compute_Y_chunk_on_gpu, start, end, omega, matrix_seed, n, gpu_id)
        for gpu_id, (start, end) in enumerate(chunks)
    ]
    wait(futures)
    Y = np.vstack([f.result() for f in futures])

    # Step 2: Power iterations
    for iteration in range(n_iter):
        print(f"Power iteration {iteration + 1}/{n_iter}")

        # A^T @ Y
        futures = [
            client.submit(compute_AtY_chunk_on_gpu, start, end, Y[start:end], matrix_seed, n, gpu_id)
            for gpu_id, (start, end) in enumerate(chunks)
        ]
        wait(futures)
        AtY = sum(f.result() for f in futures)

        # A @ AtY
        futures = [
            client.submit(compute_AX_chunk_on_gpu, start, end, AtY, matrix_seed, n, gpu_id)
            for gpu_id, (start, end) in enumerate(chunks)
        ]
        wait(futures)
        Y = np.vstack([f.result() for f in futures])

    # Step 3: QR
    print("Computing QR...")
    Q, _ = np.linalg.qr(Y)

    # Step 4: B = Q^T @ A
    print("Computing B = Q^T @ A...")
    futures = [
        client.submit(compute_QtA_chunk_on_gpu, start, end, Q[start:end], matrix_seed, n, gpu_id)
        for gpu_id, (start, end) in enumerate(chunks)
    ]
    wait(futures)
    B = sum(f.result() for f in futures)

    # Step 5: SVD of B
    print(f"Computing SVD of B ({B.shape})...")
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    print(f"Done! Top 10 singular values: {S[:10]}")

    client.close()
    cluster.close()

    return U[:, :k], S[:k], Vt[:k, :]


@app.function(gpu="T4:2", timeout=1800)
def multi_gpu_randomized_svd(m, n, k=50, n_oversamples=10, n_iter=2, dtype="float32"):
    """
    Multi-GPU randomized SVD - keeps matrix chunks in GPU memory.

    Fast but limited by total GPU memory. Each GPU holds its portion of the matrix.
    """
    import cupy as cp
    import numpy as np

    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Using {n_gpus} GPUs (in-memory mode)")

    l = k + n_oversamples
    np_dtype = np.float32 if dtype == "float32" else np.float64

    matrix_size_gb = (m * n * (4 if dtype == "float32" else 8)) / (1024**3)
    print(f"Matrix size: {m} x {n} = {matrix_size_gb:.2f} GB")

    rows_per_gpu = m // n_gpus
    gpu_row_ranges = [(i * rows_per_gpu, (i + 1) * rows_per_gpu if i < n_gpus - 1 else m)
                      for i in range(n_gpus)]
    print(f"Row distribution: {gpu_row_ranges}")

    np.random.seed(42)
    omega = np.random.randn(n, l).astype(np_dtype)

    # Generate and store chunks on GPUs
    Y_chunks = []
    for gpu_id, (start, end) in enumerate(gpu_row_ranges):
        with cp.cuda.Device(gpu_id):
            cp.random.seed(12345 + start)
            A_chunk = cp.random.randn(end - start, n, dtype=cp.float32 if dtype == "float32" else cp.float64)
            omega_gpu = cp.asarray(omega)
            Y_chunk = A_chunk @ omega_gpu
            Y_chunks.append((gpu_id, cp.asnumpy(Y_chunk), A_chunk))

    Y = np.vstack([chunk for _, chunk, _ in Y_chunks])

    # Power iterations
    for iteration in range(n_iter):
        print(f"Power iteration {iteration + 1}/{n_iter}")

        AtY_chunks = []
        for gpu_id, (start, end) in enumerate(gpu_row_ranges):
            with cp.cuda.Device(gpu_id):
                _, _, A_chunk = Y_chunks[gpu_id]
                Y_chunk_gpu = cp.asarray(Y[start:end])
                AtY_chunk = A_chunk.T @ Y_chunk_gpu
                AtY_chunks.append(cp.asnumpy(AtY_chunk))

        AtY = sum(AtY_chunks)

        new_Y_chunks = []
        for gpu_id, (start, end) in enumerate(gpu_row_ranges):
            with cp.cuda.Device(gpu_id):
                _, _, A_chunk = Y_chunks[gpu_id]
                AtY_gpu = cp.asarray(AtY)
                Y_chunk = A_chunk @ AtY_gpu
                new_Y_chunks.append(cp.asnumpy(Y_chunk))

        Y = np.vstack(new_Y_chunks)

    print("Computing QR decomposition...")
    Q, _ = np.linalg.qr(Y)

    print("Computing B = Q^T @ A...")
    B_chunks = []
    for gpu_id, (start, end) in enumerate(gpu_row_ranges):
        with cp.cuda.Device(gpu_id):
            _, _, A_chunk = Y_chunks[gpu_id]
            Q_chunk_gpu = cp.asarray(Q[start:end])
            B_chunk = Q_chunk_gpu.T @ A_chunk
            B_chunks.append(cp.asnumpy(B_chunk))

    B = sum(B_chunks)

    print(f"Computing SVD of B ({B.shape[0]} x {B.shape[1]})...")
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    print(f"Done! Top 10 singular values: {S[:10]}")
    return U[:, :k], S[:k], Vt[:k, :]


@app.function(gpu="T4:2", timeout=3600)
def streaming_randomized_svd(m, n, k=50, n_oversamples=10, n_iter=2, dtype="float32", chunk_rows=50000):
    """
    Streaming randomized SVD for matrices much larger than GPU memory.

    Regenerates matrix chunks on-the-fly, never keeping full matrix in memory.
    Can handle arbitrarily large matrices (200GB+).

    Memory: O(chunk_rows * n) per GPU + O(m * (k+oversamples)) for Y in CPU RAM.
    """
    import cupy as cp
    import numpy as np

    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Using {n_gpus} GPUs (streaming mode)")

    l = k + n_oversamples
    np_dtype = np.float32 if dtype == "float32" else np.float64

    matrix_size_gb = (m * n * (4 if dtype == "float32" else 8)) / (1024**3)
    chunk_size_gb = (chunk_rows * n * (4 if dtype == "float32" else 8)) / (1024**3)
    y_size_gb = (m * l * (4 if dtype == "float32" else 8)) / (1024**3)

    print(f"Matrix size: {m} x {n} = {matrix_size_gb:.2f} GB")
    print(f"Chunk size: {chunk_rows} rows = {chunk_size_gb:.2f} GB")
    print(f"Y matrix: {m} x {l} = {y_size_gb:.2f} GB (CPU)")

    n_chunks = (m + chunk_rows - 1) // chunk_rows
    print(f"Processing {n_chunks} chunks")

    np.random.seed(42)
    omega = np.random.randn(n, l).astype(np_dtype)

    def generate_chunk(start_row, n_rows, seed):
        np.random.seed(seed + start_row)
        return np.random.randn(n_rows, n).astype(np_dtype)

    def process_matmul(right_matrix, desc):
        results = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_rows
            end = min(start + chunk_rows, m)
            gpu_id = chunk_idx % n_gpus

            with cp.cuda.Device(gpu_id):
                chunk_np = generate_chunk(start, end - start, 12345)
                chunk_gpu = cp.asarray(chunk_np)
                right_gpu = cp.asarray(right_matrix)
                result_chunk = chunk_gpu @ right_gpu
                results.append(cp.asnumpy(result_chunk))
                del chunk_gpu, right_gpu
                cp.get_default_memory_pool().free_all_blocks()

            if (chunk_idx + 1) % max(1, n_chunks // 5) == 0:
                print(f"  {desc}: {chunk_idx + 1}/{n_chunks}")

        return np.vstack(results)

    def process_transpose_matmul(Y_matrix, desc):
        result = np.zeros((n, Y_matrix.shape[1]), dtype=np_dtype)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_rows
            end = min(start + chunk_rows, m)
            gpu_id = chunk_idx % n_gpus

            with cp.cuda.Device(gpu_id):
                chunk_np = generate_chunk(start, end - start, 12345)
                chunk_gpu = cp.asarray(chunk_np)
                Y_chunk_gpu = cp.asarray(Y_matrix[start:end])
                contribution = chunk_gpu.T @ Y_chunk_gpu
                result += cp.asnumpy(contribution)
                del chunk_gpu, Y_chunk_gpu
                cp.get_default_memory_pool().free_all_blocks()

            if (chunk_idx + 1) % max(1, n_chunks // 5) == 0:
                print(f"  {desc}: {chunk_idx + 1}/{n_chunks}")

        return result

    print("Computing Y = A @ omega...")
    Y = process_matmul(omega, "Y")

    for iteration in range(n_iter):
        print(f"Power iteration {iteration + 1}/{n_iter}")
        AtY = process_transpose_matmul(Y, "A^T @ Y")
        Y = process_matmul(AtY, "A @ AtY")

    print(f"Computing QR ({Y.shape})...")
    Q, _ = np.linalg.qr(Y)

    print("Computing B = Q^T @ A...")
    B = np.zeros((l, n), dtype=np_dtype)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_rows
        end = min(start + chunk_rows, m)
        gpu_id = chunk_idx % n_gpus

        with cp.cuda.Device(gpu_id):
            chunk_np = generate_chunk(start, end - start, 12345)
            chunk_gpu = cp.asarray(chunk_np)
            Q_chunk_gpu = cp.asarray(Q[start:end])
            contribution = Q_chunk_gpu.T @ chunk_gpu
            B += cp.asnumpy(contribution)
            del chunk_gpu, Q_chunk_gpu
            cp.get_default_memory_pool().free_all_blocks()

        if (chunk_idx + 1) % max(1, n_chunks // 5) == 0:
            print(f"  B: {chunk_idx + 1}/{n_chunks}")

    print(f"Computing SVD of B ({B.shape})...")
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    print(f"Done! Top 10 singular values: {S[:10]}")
    return U[:, :k], S[:k], Vt[:k, :]


@app.function(gpu="T4:2", timeout=600)
def test_memory_usage():
    """Test GPU memory and basic operations."""
    import cupy as cp
    import numpy as np

    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Number of GPUs: {n_gpus}")

    for i in range(n_gpus):
        with cp.cuda.Device(i):
            total = cp.cuda.runtime.getDeviceProperties(i)['totalGlobalMem']
            print(f"GPU {i}: {total / 1024**3:.2f} GB")

    test_size_gb = 3.0
    n_elements = int(test_size_gb * 1024**3 / 4)
    n_rows = n_elements // 10000
    n_cols = 10000

    print(f"\nTest: {test_size_gb} GB per GPU ({n_rows} x {n_cols})")

    arrays = []
    for i in range(n_gpus):
        with cp.cuda.Device(i):
            arr = cp.random.randn(n_rows, n_cols, dtype=cp.float32)
            arrays.append(arr)
            used = cp.get_default_memory_pool().used_bytes()
            print(f"GPU {i}: {used / 1024**3:.2f} GB allocated")

    omega = cp.random.randn(n_cols, 100, dtype=cp.float32)
    results = []
    for i, arr in enumerate(arrays):
        with cp.cuda.Device(i):
            result = arr @ cp.asarray(omega)
            results.append(cp.asnumpy(result))

    combined = np.vstack(results)
    print(f"Matmul result: {combined.shape}")
    return "Memory test passed!"


@app.local_entrypoint()
def main():
    import numpy as np
    import time

    print("=== GPU Memory Test ===")
    print(test_memory_usage.remote())

    # Test GPU version with 1 power iteration (faster, less precise)
    print("\n=== Test 1: Dask Multi-GPU SVD (3.7 GB, 1 power iter) ===")
    m, n, k = 100000, 10000, 50
    print(f"Matrix: {m} x {n} = {m * n * 4 / 1024**3:.2f} GB")
    start = time.time()
    U, S, Vt = dask_multi_gpu_svd.remote(m, n, k=k, n_iter=1)
    elapsed = time.time() - start
    print(f"Result: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    print(f"Top 5 singular values: {S[:5]}")
    print(f"Time: {elapsed:.1f}s")

    # Test GPU version with 2 power iterations (slower, more precise)
    print("\n=== Test 2: Dask Multi-GPU SVD (3.7 GB, 2 power iter) ===")
    start = time.time()
    U, S, Vt = dask_multi_gpu_svd.remote(m, n, k=k, n_iter=2)
    elapsed = time.time() - start
    print(f"Result: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    print(f"Top 5 singular values: {S[:5]}")
    print(f"Time: {elapsed:.1f}s")

    # Larger matrix
    print("\n=== Test 3: Dask Multi-GPU SVD (13.4 GB, 1 power iter) ===")
    m, n, k = 300000, 12000, 50
    print(f"Matrix: {m} x {n} = {m * n * 4 / 1024**3:.2f} GB")
    start = time.time()
    U, S, Vt = dask_multi_gpu_svd.remote(m, n, k=k, n_iter=1)
    elapsed = time.time() - start
    print(f"Result: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    print(f"Top 5 singular values: {S[:5]}")
    print(f"Time: {elapsed:.1f}s")

    # Streaming for very large matrices
    print("\n=== Test 4: Streaming SVD (46.6 GB, 1 power iter) ===")
    m, n, k = 500000, 25000, 50
    print(f"Matrix: {m} x {n} = {m * n * 4 / 1024**3:.2f} GB")
    start = time.time()
    U, S, Vt = streaming_randomized_svd.remote(m, n, k=k, chunk_rows=30000, n_iter=1)
    elapsed = time.time() - start
    print(f"Result: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    print(f"Top 5 singular values: {S[:5]}")
    print(f"Time: {elapsed:.1f}s")
