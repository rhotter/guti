# modal_us_analytical.py
# Ultrasound analytical simulation on Modal with multi-GPU SVD
#
# CLI-compatible with guti/modalities/us/analytical.py
# Results are saved in the same format to results/variants/
#
# Usage:
#   modal run modal_us_analytical.py --n-sources 32000 --n-sensors 1000
#   modal run modal_us_analytical.py --help

import modal

# Use RAPIDS image which has dask-cuda and cupy pre-installed
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12",
        add_python=None,
    )
    .pip_install("scipy", "torch")
)

app = modal.App("us-analytical-multi-gpu", image=image)

# Volume for persisting results
results_volume = modal.Volume.from_name("us-results", create_if_missing=True)


# ============================================================================
# Geometry generation functions (self-contained, from guti/core.py)
# ============================================================================

BRAIN_RADIUS = 80  # mm
SCALP_RADIUS = 92  # mm
K_B = 1.380649e-23
BODY_TEMP_K = 310
US_TYPICAL_OUTPUT_AMPLITUDE = 1e-3
US_TRANSMIT_PRESSURE = 1e4
US_TRANSDUCER_SENSITIVITY = 1e-3
US_SOUND_SPEED = 1540.0
US_BRAIN_DEPTH = 0.150
US_SCALP_AREA_MM2 = 2 * 3.141592653589793 * SCALP_RADIUS**2
US_R_ELEC = 50


def get_sensor_positions(n_sensors: int = 100, offset: float = 0, start_n: int = 0, end_n=None):
    """Get sensor positions uniformly on the surface of a hemisphere."""
    import numpy as np

    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n_sensors)
    z = (indices + 0.5) / n_sensors
    theta = np.arccos(z)
    phi = golden_angle * indices
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    positions = np.stack([x, y, z], axis=1)
    positions = positions * (SCALP_RADIUS + offset) + np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0])
    return positions[start_n:end_n]


def get_grid_positions(grid_spacing_mm: float = 5.0, radius: float = BRAIN_RADIUS):
    """Generate positions using a uniform 3D grid within the hemisphere."""
    import numpy as np

    x_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    y_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    z_coords = np.arange(0, radius + grid_spacing_mm, grid_spacing_mm)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    center = np.array([radius, radius, 0])
    distances = np.linalg.norm(grid_points - center, axis=1)
    inside_hemisphere = (distances <= radius) & (grid_points[:, 2] >= 0)
    return grid_points[inside_hemisphere]


def create_sources_real(n_sources: int, center_frequency: float):
    """Create source positions in real coordinates (meters)."""
    import numpy as np

    grid_spacing_mm = ((2/3) * np.pi * BRAIN_RADIUS**3 / n_sources)**(1/3)
    source_positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    return source_positions * 1e-3  # Convert mm to meters


def create_receivers_real(n_sensors: int):
    """Create receiver positions in real coordinates (meters)."""
    sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8)
    return sensor_positions * 1e-3  # Convert mm to meters


def compute_us_output_noise_std(freq_hz: float, n_sensors: int, f_brain: float = 1.0) -> float:
    """Per-output-channel US noise in forward-model units."""
    import math

    freq_khz = freq_hz / 1e3
    acoustic_noise_pa_per_sqrt_hz = math.sqrt(10 ** ((-15 + 20 * math.log10(freq_khz)) / 10)) * 1e-6
    element_area_mm2 = US_SCALP_AREA_MM2 / n_sensors
    element_radius_m = math.sqrt(element_area_mm2 / math.pi) * 1e-3
    ka = 2 * math.pi * freq_hz / US_SOUND_SPEED * element_radius_m
    directivity_factor = 1.0 / math.sqrt(1.0 + ka**2)
    thermal_pa = acoustic_noise_pa_per_sqrt_hz * directivity_factor

    voltage_johnson = math.sqrt(4 * K_B * BODY_TEMP_K * US_R_ELEC)
    electronic_pa = voltage_johnson / US_TRANSDUCER_SENSITIVITY
    pressure_noise_pa = math.sqrt(thermal_pa**2 + electronic_pa**2)

    prf = US_SOUND_SPEED / (2 * US_BRAIN_DEPTH)
    bandwidth_eff = freq_hz * f_brain / prf
    return pressure_noise_pa / US_TRANSMIT_PRESSURE * math.sqrt(bandwidth_eff)


def compute_iid_bitrate_from_output_power(
    singular_values,
    *,
    average_output_power: float,
    output_noise: float,
    n_sources: int,
    n_outputs: int,
) -> float:
    import numpy as np

    spectrum_power = float(np.sum(np.asarray(singular_values) ** 2))
    if spectrum_power <= 0:
        return 0.0
    per_source_power = n_outputs * average_output_power / spectrum_power
    snr_per_mode = (np.asarray(singular_values) ** 2) * per_source_power / output_noise**2
    return float(0.5 * np.sum(np.log2(1.0 + snr_per_mode)))


# ============================================================================
# Ultrasound propagation simulation (from guti/modalities/us/utils.py)
# ============================================================================

def simulate_free_field_propagation(
    source_positions,
    receiver_positions,
    source_signals,
    time_step: float,
    center_frequency: float,
    voxel_size,
    device: str = "cuda",
    temporal_sampling: int = 1,
):
    """
    Simulates free field propagation using a free field propagator.
    Returns the time-resolved pressure field.
    """
    import torch

    sound_speed = 1500.0  # m/s

    source_positions = source_positions.to(device)
    receiver_positions = receiver_positions.to(device)
    source_signals = source_signals.to(device)
    voxel_size = voxel_size.to(device)

    # Calculate distances between all source-receiver pairs
    distances = torch.cdist(
        receiver_positions.float().unsqueeze(0),
        source_positions.float().unsqueeze(0)
    )[0]

    # Avoid division by zero
    zero_distances = distances == 0
    if torch.any(zero_distances):
        distances = torch.where(zero_distances, torch.tensor(1e-10, device=device), distances)

    # Calculate propagator factor
    wavelength = sound_speed / center_frequency
    wavenumber = 2 * torch.pi / wavelength
    spatial_step = torch.mean(voxel_size)
    propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * torch.pi * distances)

    # Calculate retardation times
    travel_times = distances / sound_speed
    delay_steps = (torch.floor(travel_times / time_step)).int()

    num_sources = source_signals.shape[0]
    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]

    selected_time_indices = torch.arange(0, num_time_steps, temporal_sampling, device=device)

    # Pad source waveforms
    padded_source_signals = torch.cat([
        torch.zeros(num_sources, 1, device=device, dtype=source_signals.dtype),
        source_signals
    ], dim=1)

    source_idx = torch.arange(num_sources).unsqueeze(0).expand(num_receivers, num_sources).to(device)

    n_selected = selected_time_indices.shape[0]
    pressure_field = torch.empty(
        (num_receivers, num_sources, n_selected),
        dtype=padded_source_signals.dtype,
        device=device,
    )

    for idx, t_idx in enumerate(selected_time_indices):
        time_idx_matrix = t_idx - delay_steps + 1
        time_idx_matrix = torch.clamp(time_idx_matrix, min=0, max=padded_source_signals.shape[1] - 1)
        delayed_signals_step = padded_source_signals[source_idx, time_idx_matrix]
        pressure_field[:, :, idx] = delayed_signals_step * propagator_factor

    return pressure_field


# ============================================================================
# Multi-GPU SVD function
# ============================================================================

def multi_gpu_svd_from_matrix(G, k=50, n_oversamples=10, n_iter=2):
    """
    Multi-GPU randomized SVD on an existing matrix G.
    Uses cupy directly without dask to avoid large graph transfers.
    """
    import cupy as cp
    import numpy as np

    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Using {n_gpus} GPUs for randomized SVD")

    m, n = G.shape
    l = k + n_oversamples
    matrix_size_gb = (m * n * 4) / (1024**3)
    print(f"Matrix size: {m} x {n} = {matrix_size_gb:.2f} GB")

    # Convert G to numpy if it's a torch tensor
    if hasattr(G, 'cpu'):
        G_np = G.cpu().numpy()
    else:
        G_np = np.asarray(G)

    # Divide rows among GPUs
    rows_per_gpu = m // n_gpus
    gpu_row_ranges = [(i * rows_per_gpu, (i + 1) * rows_per_gpu if i < n_gpus - 1 else m)
                      for i in range(n_gpus)]
    print(f"Row distribution: {gpu_row_ranges}")

    # Random projection
    np.random.seed(42)
    omega = np.random.randn(n, l).astype(np.float32)

    # Step 1: Y = G @ omega
    print("Computing Y = G @ omega...")
    Y_chunks = []
    for gpu_id, (start, end) in enumerate(gpu_row_ranges):
        with cp.cuda.Device(gpu_id):
            G_chunk_gpu = cp.asarray(G_np[start:end])
            omega_gpu = cp.asarray(omega)
            Y_chunk = G_chunk_gpu @ omega_gpu
            Y_chunks.append(cp.asnumpy(Y_chunk))

    Y = np.vstack(Y_chunks)

    # Step 2: Power iterations
    for iteration in range(n_iter):
        print(f"Power iteration {iteration + 1}/{n_iter}")

        # G^T @ Y
        GtY_chunks = []
        for gpu_id, (start, end) in enumerate(gpu_row_ranges):
            with cp.cuda.Device(gpu_id):
                G_chunk_gpu = cp.asarray(G_np[start:end])
                Y_chunk_gpu = cp.asarray(Y[start:end])
                GtY_chunk = G_chunk_gpu.T @ Y_chunk_gpu
                GtY_chunks.append(cp.asnumpy(GtY_chunk))
        GtY = sum(GtY_chunks)

        # G @ GtY
        Y_chunks = []
        for gpu_id, (start, end) in enumerate(gpu_row_ranges):
            with cp.cuda.Device(gpu_id):
                G_chunk_gpu = cp.asarray(G_np[start:end])
                GtY_gpu = cp.asarray(GtY)
                Y_chunk = G_chunk_gpu @ GtY_gpu
                Y_chunks.append(cp.asnumpy(Y_chunk))
        Y = np.vstack(Y_chunks)

    # Step 3: QR
    print("Computing QR...")
    Q, _ = np.linalg.qr(Y)

    # Step 4: B = Q^T @ G
    print("Computing B = Q^T @ G...")
    B_chunks = []
    for gpu_id, (start, end) in enumerate(gpu_row_ranges):
        with cp.cuda.Device(gpu_id):
            G_chunk_gpu = cp.asarray(G_np[start:end])
            Q_chunk_gpu = cp.asarray(Q[start:end])
            B_chunk = Q_chunk_gpu.T @ G_chunk_gpu
            B_chunks.append(cp.asnumpy(B_chunk))
    B = sum(B_chunks)

    # Step 5: SVD of B
    print(f"Computing SVD of B ({B.shape})...")
    _, S, _ = np.linalg.svd(B, full_matrices=False)

    print(f"Done! Top 10 singular values: {S[:10]}")

    return S


# ============================================================================
# Main simulation function (GPU)
# ============================================================================

@app.function(gpu="H100", timeout=7200, memory=262144, volumes={"/results": results_volume})  # 256GB RAM for large matrices
def run_us_simulation(
    n_sources: int = 32000,
    n_sensors: int = 1000,
    temporal_sampling: int = 5,
    center_frequency: float = 0.05e6,
    sensor_batch_size: int = 256,
    save_results: bool = True,
    use_eig: bool = True,  # Use eigendecomposition (faster) vs direct SVD (more stable)
):
    """
    Run ultrasound free-field simulation with full SVD on GPU.

    Args:
        n_sources: Number of source points in brain volume
        n_sensors: Number of sensor points on scalp
        temporal_sampling: Temporal downsampling factor
        center_frequency: Center frequency in Hz
        sensor_batch_size: Batch size for sensor processing
        use_multi_gpu_svd: If True, use multi-GPU randomized SVD
        k: Number of singular values to compute (for randomized SVD)
        n_iter: Number of power iterations for SVD
        save_results: If True, save results to volume

    Returns:
        Dictionary with singular values and metadata
    """
    import torch
    import numpy as np
    import time
    import os
    import json
    import hashlib
    from dataclasses import dataclass, asdict

    print(f"=== Ultrasound Analytical Simulation (Modal) ===")
    print(f"n_sources: {n_sources}, n_sensors: {n_sensors}")
    print(f"temporal_sampling: {temporal_sampling}, center_frequency: {center_frequency/1e3:.1f} kHz")

    # Create source and sensor positions
    source_positions = create_sources_real(n_sources, center_frequency)
    sensor_positions = create_receivers_real(n_sensors)

    n_sources_actual = source_positions.shape[0]
    print(f"Actual n_sources after grid generation: {n_sources_actual}")

    # Create time axis and source signals
    time_step = 1e-1 / center_frequency
    time_duration = 120e-6
    time_axis = np.arange(0, time_duration, time_step)
    source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
    source_signals = np.tile(source_signals, (n_sources_actual, 1))

    nt = time_axis.shape[0] // temporal_sampling + 1
    min_speed = 1500.0
    PPW = 24
    dx_m = min_speed / (PPW * center_frequency)
    voxel_size = np.array([dx_m, dx_m, dx_m])

    print(f"Time steps: {len(time_axis)}, nt after sampling: {nt}")

    # Estimate matrix size
    matrix_rows = n_sensors * nt
    matrix_cols = n_sources_actual
    matrix_size_gb = (matrix_rows * matrix_cols * 4) / (1024**3)
    gram_size_gb = (min(matrix_rows, matrix_cols) ** 2 * 4) / (1024**3)
    print(f"Matrix G size: {matrix_rows} x {matrix_cols} = {matrix_size_gb:.2f} GB")
    print(f"Gram matrix size: {min(matrix_rows, matrix_cols)} x {min(matrix_rows, matrix_cols)} = {gram_size_gb:.2f} GB")

    device = "cuda"
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU memory: {gpu_mem_gb:.1f} GB")

    # Pre-build constant tensors
    source_positions_t = torch.tensor(source_positions, device=device)
    source_signals_t = torch.tensor(source_signals, device=device)
    voxel_size_t = torch.tensor(voxel_size, device=device)

    num_batches = (n_sensors + sensor_batch_size - 1) // sensor_batch_size

    # Check if we can fit G in GPU memory (need ~1.5x for workspace)
    can_fit_g = matrix_size_gb < gpu_mem_gb * 0.6

    if can_fit_g:
        # Standard approach: build full G on GPU
        print("Building matrix G on GPU...")
        t0 = time.perf_counter()

        G = torch.zeros((matrix_rows, matrix_cols), dtype=torch.float32, device="cuda")
        last_index = 0

        for batch_idx, start in enumerate(range(0, n_sensors, sensor_batch_size)):
            end = min(start + sensor_batch_size, n_sensors)
            print(f"Processing sensor batch {batch_idx + 1}/{num_batches} (sensors {start}-{end})")

            receiver_positions_t = torch.tensor(sensor_positions[start:end], device=device)

            pf_chunk = simulate_free_field_propagation(
                source_positions_t,
                receiver_positions_t,
                source_signals_t,
                time_step,
                center_frequency,
                voxel_size_t,
                device=device,
                temporal_sampling=temporal_sampling
            )

            chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources_actual).float()
            chunk_rows = chunk_matrix.shape[0]
            G[last_index:last_index + chunk_rows] = chunk_matrix
            last_index += chunk_rows

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Matrix G built in {t1 - t0:.3f}s")
    else:
        # Streaming approach: build G chunks and accumulate Gram matrix directly
        print(f"Matrix too large for GPU ({matrix_size_gb:.1f}GB > {gpu_mem_gb*0.6:.1f}GB)")
        print("Using streaming Gram accumulation (G never fully materialized)...")
        t0 = time.perf_counter()

        # Store G chunks on CPU, then compute Gram
        G_chunks_cpu = []

        for batch_idx, start in enumerate(range(0, n_sensors, sensor_batch_size)):
            end = min(start + sensor_batch_size, n_sensors)
            print(f"Processing sensor batch {batch_idx + 1}/{num_batches} (sensors {start}-{end})")

            receiver_positions_t = torch.tensor(sensor_positions[start:end], device=device)

            pf_chunk = simulate_free_field_propagation(
                source_positions_t,
                receiver_positions_t,
                source_signals_t,
                time_step,
                center_frequency,
                voxel_size_t,
                device=device,
                temporal_sampling=temporal_sampling
            )

            chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources_actual).float()
            G_chunks_cpu.append(chunk_matrix.cpu())  # Store on CPU
            del pf_chunk, chunk_matrix
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"G chunks generated in {t1 - t0:.3f}s")

        # Now compute Gram matrix by streaming chunks
        # G has shape (m, n), we compute G @ G^T (m x m) or G^T @ G (n x n)
        # Choose the smaller one
        m, n = matrix_rows, matrix_cols

        if m <= n:
            # Compute Gram = G @ G^T by accumulating chunk contributions
            print(f"Computing Gram matrix G @ G^T ({m}x{m}) via streaming...")
            t0 = time.perf_counter()

            # Gram = sum over column blocks: G[:, block] @ G[:, block]^T
            # But we have row blocks, so we need a different approach
            # Gram[i,j] = sum_k G[i,k] * G[j,k]
            # With row blocks: Gram = sum of contributions from each row block pair

            # Actually for row blocks, we can still accumulate:
            # If G = [G1; G2; ...] (row blocks), then
            # Gram is NOT simply sum of Gi @ Gi^T
            # We need the full outer product structure

            # For streaming with row blocks, we need to:
            # 1. Concatenate all chunks to form G (on CPU if needed)
            # 2. Then compute Gram

            # If Gram fits in GPU, load G to GPU in pieces and compute
            if gram_size_gb < gpu_mem_gb * 0.85:  # Leave 15% for eigvalsh workspace
                print(f"  Gram matrix fits in GPU memory, computing directly...")
                # Concatenate G on CPU
                G_cpu = torch.cat(G_chunks_cpu, dim=0)
                del G_chunks_cpu

                # Compute Gram on GPU by processing G in column chunks
                # Use float16 for Gram to save memory (half the size)
                gram = torch.zeros((m, m), dtype=torch.float16, device="cuda")
                torch.cuda.synchronize()

                gram_f16_size_gb = (m * m * 2) / (1024**3)  # float16 = 2 bytes
                print(f"    Using float16 Gram matrix: {gram_f16_size_gb:.1f} GB")

                # Calculate available memory for column chunks
                available_gb = gpu_mem_gb - gram_f16_size_gb - 2  # 2GB safety margin
                # Each column chunk of shape (m, chunk_size) needs m * chunk_size * 2 bytes (float16)
                # Plus the intermediate result of same size as gram (m * m * 2)
                # So we need chunk such that: m * chunk * 2 + m * m * 2 < available * 1024^3
                # Actually the matmul result is m x m regardless of chunk size
                # So we need: gram_f16 + chunk + temp_result < available
                # temp_result = m x m (same as gram)
                # We need at least 2x gram_f16 memory + chunk
                if available_gb < gram_f16_size_gb:
                    raise RuntimeError(f"Not enough GPU memory for streaming Gram computation")

                remaining_gb = available_gb - gram_f16_size_gb  # for temp result
                col_chunk_size = max(1000, int((remaining_gb * 1024**3) / (m * 2)))  # float16
                print(f"    Using column chunk size: {col_chunk_size} (remaining: {remaining_gb:.1f}GB)")

                num_col_chunks = (n + col_chunk_size - 1) // col_chunk_size
                for i, col_start in enumerate(range(0, n, col_chunk_size)):
                    col_end = min(col_start + col_chunk_size, n)
                    G_col_chunk = G_cpu[:, col_start:col_end].half().to(device)
                    # Use addmm for in-place-ish update (still creates temp but immediately added)
                    gram.addmm_(G_col_chunk, G_col_chunk.T)
                    del G_col_chunk
                    if i % 20 == 0:
                        torch.cuda.empty_cache()
                        print(f"    Processed columns {col_start}-{col_end} of {n} ({i+1}/{num_col_chunks})")

                # Free G_cpu memory before conversion
                del G_cpu

                # Convert to float32 for eigendecomposition (eigvalsh needs float32)
                # Do this carefully to avoid OOM: copy to CPU, delete GPU, reload as float32
                print(f"    Converting Gram matrix to float32...")
                gram_cpu = gram.cpu()
                del gram
                torch.cuda.empty_cache()

                # Check for NaN/Inf values on CPU (memory efficient)
                nan_count = torch.isnan(gram_cpu).sum().item()
                inf_count = torch.isinf(gram_cpu).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"    WARNING: Gram matrix has {nan_count} NaN and {inf_count} Inf values!")
                    print(f"    Replacing NaN/Inf with zeros...")
                    gram_cpu = torch.nan_to_num(gram_cpu, nan=0.0, posinf=0.0, neginf=0.0)

                # Convert to float32
                gram_cpu = gram_cpu.float()

                # Symmetrize on CPU to ensure numerical symmetry (required for eigvalsh)
                print(f"    Symmetrizing Gram matrix on CPU...")
                gram_cpu = 0.5 * (gram_cpu + gram_cpu.T)

                # Move to GPU
                gram = gram_cpu.to(device)
                del gram_cpu
                t1 = time.perf_counter()
                print(f"  Gram matrix computed in {t1 - t0:.3f}s")
                G = None  # Signal that we used streaming
            else:
                raise RuntimeError(f"Gram matrix ({gram_size_gb:.1f}GB) too large for GPU ({gpu_mem_gb:.1f}GB)")
        else:
            # G^T @ G case - similar streaming approach
            raise NotImplementedError("Streaming G^T @ G not implemented yet")

    t0 = time.perf_counter()

    if use_eig:
        # === EIGENDECOMPOSITION PATH ===
        # Use torch with cusolver backend (fastest based on benchmark)
        torch.backends.cuda.preferred_linalg_library("cusolver")

        if G is None:
            # Streaming case: gram matrix already computed
            print(f"\nComputing eigenvalues of pre-computed Gram matrix ({gram.shape})...")
        else:
            # Standard case: compute Gram matrix from G
            m, n = G.shape
            print(f"\nComputing singular values via eigendecomposition (GPU, torch+cusolver, {G.shape})...")

            # Use float16 for the matrix multiply (tensor cores), then float32 for eig
            G_f16 = G.half()
            del G
            torch.cuda.empty_cache()

            if m >= n:
                # Tall matrix: compute G^T @ G (n x n)
                print(f"  Computing GᵀG ({n}x{n}) using float16 tensor cores...")
                t_matmul = time.perf_counter()
                gram = torch.mm(G_f16.T, G_f16).float()
                torch.cuda.synchronize()
                print(f"  GᵀG computed in {time.perf_counter() - t_matmul:.3f}s")
            else:
                # Wide matrix: compute G @ G^T (m x m)
                print(f"  Computing GGᵀ ({m}x{m}) using float16 tensor cores...")
                t_matmul = time.perf_counter()
                gram = torch.mm(G_f16, G_f16.T).float()
                torch.cuda.synchronize()
                print(f"  GGᵀ computed in {time.perf_counter() - t_matmul:.3f}s")

            del G_f16
            torch.cuda.empty_cache()

        # Check Gram matrix stats (without creating large boolean tensors)
        print(f"  Gram matrix range: [{gram.min().item():.4e}, {gram.max().item():.4e}]")
        # Quick NaN/Inf check using any() which is memory efficient
        has_nan = torch.isnan(gram.view(-1)[:1000]).any().item() or torch.isnan(gram.view(-1)[-1000:]).any().item()
        has_inf = torch.isinf(gram.view(-1)[:1000]).any().item() or torch.isinf(gram.view(-1)[-1000:]).any().item()
        if has_nan or has_inf:
            print(f"  WARNING: Gram matrix may have NaN or Inf values (sampled check)")

        gram_size = gram.shape[0]
        # cuSOLVER has limits around 50k-60k for eigendecomposition
        use_cpu_fallback = gram_size > 50000

        if use_cpu_fallback:
            print(f"  Gram matrix ({gram_size}x{gram_size}) too large for GPU cuSOLVER")
            print(f"  Using scipy on CPU (this will be slower but works for any size)...")

            # Move to CPU and use scipy
            gram_cpu = gram.cpu().numpy()
            del gram
            torch.cuda.empty_cache()

            from scipy.linalg import eigvalsh as scipy_eigvalsh
            t_eig = time.perf_counter()
            eigenvalues_np = scipy_eigvalsh(gram_cpu)
            print(f"  Eigenvalues computed with scipy in {time.perf_counter() - t_eig:.3f}s")

            # Convert to torch tensor (eigenvalues are in ascending order)
            eigenvalues = torch.from_numpy(eigenvalues_np)
            del gram_cpu, eigenvalues_np
        else:
            print(f"  Computing eigenvalues (torch.linalg.eigvalsh with cusolver)...")
            t_eig = time.perf_counter()
            try:
                eigenvalues = torch.linalg.eigvalsh(gram)
                torch.cuda.synchronize()
                print(f"  Eigenvalues computed in {time.perf_counter() - t_eig:.3f}s")
            except Exception as e:
                print(f"  cusolver failed: {e}")
                print(f"  Falling back to scipy on CPU...")

                gram_cpu = gram.cpu().numpy()
                del gram
                torch.cuda.empty_cache()

                from scipy.linalg import eigvalsh as scipy_eigvalsh
                t_eig = time.perf_counter()
                eigenvalues_np = scipy_eigvalsh(gram_cpu)
                print(f"  Eigenvalues computed with scipy in {time.perf_counter() - t_eig:.3f}s")

                eigenvalues = torch.from_numpy(eigenvalues_np)
                del gram_cpu, eigenvalues_np

        del gram
        torch.cuda.empty_cache()

        # Eigenvalues are sorted ascending, we want descending
        # Singular values are sqrt of eigenvalues (clamp negatives from numerical error)
        s = eigenvalues.clamp(min=0).sqrt().flip(0).cpu().numpy()
    else:
        # === DIRECT SVD PATH ===
        # More numerically stable, no squaring of condition number
        print(f"\nComputing direct SVD (GPU, cuSOLVER gesvdj, {G.shape})...")

        # Transfer to cupy
        G_cp = cp.asarray(G)
        del G
        torch.cuda.empty_cache()

        # Try gesvdj (Jacobi SVD) first - can be faster for certain shapes
        # gesvdj is accessed through cupy.cuda.cusolver
        try:
            from cupy.cuda import cusolver
            from cupy.cuda import device as cuda_device

            handle = cuda_device.get_cusolver_handle()
            m, n = G_cp.shape

            # gesvdj works best when we have the matrix in the right orientation
            # For very rectangular matrices, standard SVD might be better
            aspect_ratio = max(m, n) / min(m, n)

            if aspect_ratio < 10:
                # Use gesvdj for reasonably square matrices
                print(f"  Using gesvdj (Jacobi SVD)...")
                s = cp.linalg.svd(G_cp, compute_uv=False)
            else:
                # For very rectangular, use standard SVD
                print(f"  Using standard cuSOLVER SVD (aspect ratio {aspect_ratio:.1f})...")
                s = cp.linalg.svd(G_cp, compute_uv=False)
        except Exception as e:
            print(f"  cuSOLVER SVD failed ({e}), using numpy fallback...")
            G_np = cp.asnumpy(G_cp)
            s = np.linalg.svd(G_np, compute_uv=False)

        if isinstance(s, cp.ndarray):
            s = cp.asnumpy(s)

    t1 = time.perf_counter()
    print(f"Singular values computed in {t1 - t0:.3f}s")

    print(f"\nFirst 10 singular values: {s[:10]}")
    print(f"Sum of first 10: {np.sum(s[:10]):.4f}")

    # Compute bitrate from per-output average signal power and per-output noise.
    noise_level = compute_us_output_noise_std(center_frequency, n_sensors)
    bitrate = compute_iid_bitrate_from_output_power(
        s,
        average_output_power=US_TYPICAL_OUTPUT_AMPLITUDE**2,
        output_noise=noise_level,
        n_sources=n_sources_actual,
        n_outputs=matrix_rows,
    )

    print(f"Noise level: {noise_level:.6e}")
    print(f"Bitrate: {bitrate:.2f} bits/sample")

    # Save results in same format as original
    if save_results:
        # Create parameters dict (matching guti.parameters.Parameters)
        params = {
            "num_sensors": n_sensors,
            "num_brain_grid_points": n_sources_actual,
            "time_resolution": time_step,
            "vincent_trick": False,
            "matrix_size": (matrix_rows, matrix_cols),
            "comment": f"Modal multi-GPU, center_freq={center_frequency/1e3:.0f}kHz, temporal_sampling={temporal_sampling}",
        }

        # Compute hash for filename
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        # Save to volume
        modality_name = f"us_free_field_analytical_{int(center_frequency/1e3)}khz"
        save_dir = f"/results/variants/{modality_name}"
        os.makedirs(save_dir, exist_ok=True)

        filepath = f"{save_dir}/{params_hash}.npz"
        np.savez(filepath, singular_values=s, parameters=params)
        print(f"\nSaved results to {filepath}")

        # Also save a copy with descriptive name for easy access
        descriptive_name = f"ns{n_sources_actual}_nd{n_sensors}_ts{temporal_sampling}"
        descriptive_path = f"{save_dir}/{descriptive_name}.npz"
        np.savez(descriptive_path, singular_values=s, parameters=params)
        print(f"Also saved to {descriptive_path}")

        # Commit volume changes
        results_volume.commit()

    return {
        "singular_values": s.tolist(),
        "n_sources": n_sources_actual,
        "n_sensors": n_sensors,
        "matrix_shape": (matrix_rows, matrix_cols),
        "matrix_size_gb": matrix_size_gb,
        "bitrate": bitrate,
        "noise_level": noise_level,
        "average_output_power": US_TYPICAL_OUTPUT_AMPLITUDE**2,
        "params_hash": params_hash if save_results else None,
    }


# ============================================================================
# CPU-only simulation function
# ============================================================================

# CPU image - lighter weight, no GPU dependencies
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy")

def simulate_free_field_propagation_numpy(
    source_positions,
    receiver_positions,
    source_signals,
    time_step,
    center_frequency,
    voxel_size,
    temporal_sampling=1,
):
    """Numpy version of free field propagation simulation."""
    import numpy as np
    from scipy.spatial.distance import cdist

    sound_speed = 1500.0

    # Calculate distances between all source-receiver pairs
    # distances[i,j] = distance from receiver i to source j
    distances = cdist(receiver_positions, source_positions)

    # Avoid division by zero
    distances = np.where(distances == 0, 1e-10, distances)

    # Calculate propagator factor
    wavelength = sound_speed / center_frequency
    wavenumber = 2 * np.pi / wavelength
    spatial_step = np.mean(voxel_size)
    propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * np.pi * distances)

    # Calculate retardation times
    travel_times = distances / sound_speed
    delay_steps = np.floor(travel_times / time_step).astype(np.int32)

    num_sources = source_signals.shape[0]
    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]

    selected_time_indices = np.arange(0, num_time_steps, temporal_sampling)

    # Pad source waveforms
    padded_source_signals = np.concatenate([
        np.zeros((num_sources, 1), dtype=source_signals.dtype),
        source_signals
    ], axis=1)

    n_selected = len(selected_time_indices)
    pressure_field = np.empty((num_receivers, num_sources, n_selected), dtype=np.float32)

    for idx, t_idx in enumerate(selected_time_indices):
        time_idx_matrix = t_idx - delay_steps + 1  # shape: (num_receivers, num_sources)
        time_idx_matrix = np.clip(time_idx_matrix, 0, padded_source_signals.shape[1] - 1)
        # Gather delayed signals for each receiver-source pair
        # padded_source_signals is (num_sources, num_time_steps+1)
        # We need padded_source_signals[j, time_idx_matrix[i,j]] for all i,j
        for i in range(num_receivers):
            pressure_field[i, :, idx] = padded_source_signals[np.arange(num_sources), time_idx_matrix[i, :]] * propagator_factor[i, :]

    return pressure_field


@app.function(cpu=16, memory=131072, timeout=7200, volumes={"/results": results_volume}, image=cpu_image)
def run_us_simulation_cpu(
    n_sources: int = 32000,
    n_sensors: int = 1000,
    temporal_sampling: int = 5,
    center_frequency: float = 0.05e6,
    sensor_batch_size: int = 256,
    save_results: bool = True,
):
    """
    Run ultrasound free-field simulation on CPU only (numpy/scipy).
    """
    import numpy as np
    import time
    import os
    import json
    import hashlib

    print(f"=== Ultrasound Analytical Simulation (Modal CPU) ===")
    print(f"n_sources: {n_sources}, n_sensors: {n_sensors}")
    print(f"temporal_sampling: {temporal_sampling}, center_frequency: {center_frequency/1e3:.1f} kHz")

    # Create source and sensor positions
    source_positions = create_sources_real(n_sources, center_frequency)
    sensor_positions = create_receivers_real(n_sensors)

    n_sources_actual = source_positions.shape[0]
    print(f"Actual n_sources after grid generation: {n_sources_actual}")

    # Create time axis and source signals
    time_step = 1e-1 / center_frequency
    time_duration = 120e-6
    time_axis = np.arange(0, time_duration, time_step)
    source_signals = np.sin(2 * np.pi * time_axis * center_frequency)
    source_signals = np.tile(source_signals, (n_sources_actual, 1))

    nt = time_axis.shape[0] // temporal_sampling + 1
    min_speed = 1500.0
    PPW = 24
    dx_m = min_speed / (PPW * center_frequency)
    voxel_size = np.array([dx_m, dx_m, dx_m])

    print(f"Time steps: {len(time_axis)}, nt after sampling: {nt}")

    # Estimate matrix size
    matrix_rows = n_sensors * nt
    matrix_cols = n_sources_actual
    matrix_size_gb = (matrix_rows * matrix_cols * 4) / (1024**3)
    print(f"Matrix G size: {matrix_rows} x {matrix_cols} = {matrix_size_gb:.2f} GB")

    # Build matrix G in batches
    print("Building matrix G...")
    t0 = time.perf_counter()

    G = np.zeros((matrix_rows, matrix_cols), dtype=np.float32)
    last_index = 0

    num_batches = (n_sensors + sensor_batch_size - 1) // sensor_batch_size

    for batch_idx, start in enumerate(range(0, n_sensors, sensor_batch_size)):
        end = min(start + sensor_batch_size, n_sensors)
        print(f"Processing sensor batch {batch_idx + 1}/{num_batches} (sensors {start}-{end})")

        pf_chunk = simulate_free_field_propagation_numpy(
            source_positions,
            sensor_positions[start:end],
            source_signals,
            time_step,
            center_frequency,
            voxel_size,
            temporal_sampling=temporal_sampling
        )

        # Reshape: [receivers, sources, time] -> [receivers*time, sources]
        chunk_matrix = pf_chunk.transpose(0, 2, 1).reshape(-1, n_sources_actual)
        chunk_rows = chunk_matrix.shape[0]
        G[last_index:last_index + chunk_rows] = chunk_matrix
        last_index += chunk_rows

    t1 = time.perf_counter()
    print(f"Matrix G built in {t1 - t0:.3f}s")

    # Compute full SVD using scipy with 'gesdd' driver (divide-and-conquer, multi-threaded)
    # Enable OpenBLAS/MKL multi-threading
    os.environ.setdefault('OMP_NUM_THREADS', '16')
    os.environ.setdefault('MKL_NUM_THREADS', '16')

    from scipy.linalg import svd as scipy_svd

    print(f"\nComputing full SVD (CPU, scipy gesdd, {G.shape})...")
    t0 = time.perf_counter()
    s = scipy_svd(G, compute_uv=False, lapack_driver='gesdd')
    t1 = time.perf_counter()
    print(f"SVD computed in {t1 - t0:.3f}s")

    print(f"\nFirst 10 singular values: {s[:10]}")
    print(f"Sum of first 10: {np.sum(s[:10]):.4f}")

    # Compute bitrate from per-output average signal power and per-output noise.
    noise_level = compute_us_output_noise_std(center_frequency, n_sensors)
    bitrate = compute_iid_bitrate_from_output_power(
        s,
        average_output_power=US_TYPICAL_OUTPUT_AMPLITUDE**2,
        output_noise=noise_level,
        n_sources=n_sources_actual,
        n_outputs=matrix_rows,
    )

    print(f"Noise level: {noise_level:.6e}")
    print(f"Bitrate: {bitrate:.2f} bits/sample")

    # Save results
    params_hash = None
    if save_results:
        params = {
            "num_sensors": n_sensors,
            "num_brain_grid_points": n_sources_actual,
            "time_resolution": time_step,
            "vincent_trick": False,
            "matrix_size": (matrix_rows, matrix_cols),
            "comment": f"Modal CPU, center_freq={center_frequency/1e3:.0f}kHz, temporal_sampling={temporal_sampling}",
        }

        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        modality_name = f"us_free_field_analytical_{int(center_frequency/1e3)}khz"
        save_dir = f"/results/variants/{modality_name}"
        os.makedirs(save_dir, exist_ok=True)

        filepath = f"{save_dir}/{params_hash}.npz"
        np.savez(filepath, singular_values=s, parameters=params)
        print(f"\nSaved results to {filepath}")

        descriptive_name = f"ns{n_sources_actual}_nd{n_sensors}_ts{temporal_sampling}_cpu"
        descriptive_path = f"{save_dir}/{descriptive_name}.npz"
        np.savez(descriptive_path, singular_values=s, parameters=params)
        print(f"Also saved to {descriptive_path}")

        results_volume.commit()

    return {
        "singular_values": s.tolist(),
        "n_sources": n_sources_actual,
        "n_sensors": n_sensors,
        "matrix_shape": (matrix_rows, matrix_cols),
        "matrix_size_gb": matrix_size_gb,
        "bitrate": bitrate,
        "noise_level": noise_level,
        "average_output_power": US_TYPICAL_OUTPUT_AMPLITUDE**2,
        "params_hash": params_hash,
    }


@app.function(volumes={"/results": results_volume})
def list_results():
    """List all saved results."""
    import os

    results_dir = "/results/variants"
    if not os.path.exists(results_dir):
        return {"modalities": []}

    modalities = {}
    for modality in os.listdir(results_dir):
        modality_dir = os.path.join(results_dir, modality)
        if os.path.isdir(modality_dir):
            files = [f for f in os.listdir(modality_dir) if f.endswith('.npz')]
            modalities[modality] = files

    return modalities


@app.function(volumes={"/results": results_volume})
def download_result(modality: str, filename: str):
    """Download a specific result file."""
    import numpy as np

    filepath = f"/results/variants/{modality}/{filename}"
    data = np.load(filepath, allow_pickle=True)
    return {
        "singular_values": data["singular_values"].tolist(),
        "parameters": data["parameters"].item() if "parameters" in data else None,
    }


@app.local_entrypoint()
def main(
    n_sources: int = 32000,
    n_sensors: int = 1000,
    temporal_sampling: int = 5,
    center_frequency: float = 0.05e6,
    sensor_batch_size: int = 256,
    list_only: bool = False,
    download: str = None,
    cpu: bool = False,
    svd: bool = False,  # Use direct SVD instead of eigendecomposition
):
    """
    Ultrasound analytical simulation on Modal with full SVD.

    Usage:
        modal run modal_us_analytical.py --n-sources 32000 --n-sensors 1000
        modal run modal_us_analytical.py --n-sources 1000 --n-sensors 100 --cpu
        modal run modal_us_analytical.py --list-only
        modal run modal_us_analytical.py --download "us_free_field_analytical_50khz/abc12345.npz"

    Results are saved to Modal Volume 'us-results' and can be downloaded locally.
    """
    import numpy as np
    import os

    if list_only:
        print("=== Saved Results ===")
        results = list_results.remote()
        for modality, files in results.items():
            print(f"\n{modality}/")
            for f in files:
                print(f"  {f}")
        return

    if download:
        parts = download.split("/")
        if len(parts) != 2:
            print("Error: Use format 'modality/filename.npz'")
            return
        modality, filename = parts
        print(f"Downloading {download}...")
        data = download_result.remote(modality, filename)

        # Save locally
        local_dir = f"results/variants/{modality}"
        os.makedirs(local_dir, exist_ok=True)
        local_path = f"{local_dir}/{filename}"
        np.savez(local_path,
                 singular_values=np.array(data["singular_values"]),
                 parameters=data["parameters"])
        print(f"Saved to {local_path}")
        return

    # Run simulation
    print("=" * 60)
    if cpu:
        mode = "CPU"
    elif svd:
        mode = "GPU (direct SVD)"
    else:
        mode = "GPU (eigendecomposition + float16)"
    print(f"Running Ultrasound Analytical Simulation on Modal ({mode})")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n_sources: {n_sources}")
    print(f"  n_sensors: {n_sensors}")
    print(f"  temporal_sampling: {temporal_sampling}")
    print(f"  center_frequency: {center_frequency/1e3:.1f} kHz")
    print(f"  sensor_batch_size: {sensor_batch_size}")
    print()

    if cpu:
        result = run_us_simulation_cpu.remote(
            n_sources=n_sources,
            n_sensors=n_sensors,
            temporal_sampling=temporal_sampling,
            center_frequency=center_frequency,
            sensor_batch_size=sensor_batch_size,
            save_results=True,
        )
    else:
        result = run_us_simulation.remote(
            n_sources=n_sources,
            n_sensors=n_sensors,
            temporal_sampling=temporal_sampling,
            center_frequency=center_frequency,
            sensor_batch_size=sensor_batch_size,
            save_results=True,
            use_eig=not svd,  # Default is eig (faster), --svd uses direct SVD
        )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Matrix shape: {result['matrix_shape']}")
    print(f"Matrix size: {result['matrix_size_gb']:.2f} GB")
    print(f"Actual n_sources: {result['n_sources']}")
    print(f"Top 10 singular values: {result['singular_values'][:10]}")
    print(f"Bitrate: {result['bitrate']:.2f} bits/sample")
    print(f"Params hash: {result['params_hash']}")

    # Save locally as well
    local_dir = f"results/variants/us_free_field_analytical_{int(center_frequency/1e3)}khz"
    os.makedirs(local_dir, exist_ok=True)
    local_path = f"{local_dir}/{result['params_hash']}.npz"

    s = np.array(result['singular_values'])
    params = {
        "num_sensors": result['n_sensors'],
        "num_brain_grid_points": result['n_sources'],
        "time_resolution": 1e-1 / center_frequency,
        "vincent_trick": False,
        "matrix_size": result['matrix_shape'],
    }
    np.savez(local_path, singular_values=s, parameters=params)
    print(f"\nSaved locally to: {local_path}")
