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
# Main simulation function
# ============================================================================

@app.function(gpu="H200:8", timeout=3600, memory=32768, volumes={"/results": results_volume})
def run_us_simulation(
    n_sources: int = 32000,
    n_sensors: int = 1000,
    temporal_sampling: int = 5,
    center_frequency: float = 0.05e6,
    sensor_batch_size: int = 256,
    use_multi_gpu_svd: bool = True,
    k: int | None = None,
    n_iter: int = 1,
    save_results: bool = True,
):
    """
    Run ultrasound free-field simulation with multi-GPU SVD.

    Args:
        n_sources: Number of source points in brain volume
        n_sensors: Number of sensor points on scalp
        temporal_sampling: Temporal downsampling factor
        center_frequency: Center frequency in Hz
        sensor_batch_size: Batch size for sensor processing
        use_multi_gpu_svd: If True, use multi-GPU randomized SVD
        k: Number of singular values to compute. If None, computes all (rank of matrix)
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
    print(f"Matrix G size: {matrix_rows} x {matrix_cols} = {matrix_size_gb:.2f} GB")

    device = "cuda"

    # Pre-build constant tensors
    source_positions_t = torch.tensor(source_positions, device=device)
    source_signals_t = torch.tensor(source_signals, device=device)
    voxel_size_t = torch.tensor(voxel_size, device=device)

    # Build matrix G in batches
    print("Building matrix G...")
    t0 = time.perf_counter()

    G = torch.zeros((matrix_rows, matrix_cols), dtype=torch.float32, device="cuda")
    last_index = 0

    num_batches = (n_sensors + sensor_batch_size - 1) // sensor_batch_size

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

        # Reshape: [receivers, sources, time] -> [receivers*time, sources]
        chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources_actual).float()
        chunk_rows = chunk_matrix.shape[0]
        G[last_index:last_index + chunk_rows] = chunk_matrix
        last_index += chunk_rows

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Matrix G built in {t1 - t0:.3f}s")

    # Compute SVD
    print("\nComputing SVD...")
    t0 = time.perf_counter()

    if use_multi_gpu_svd and matrix_size_gb > 0.5:
        # Use multi-GPU randomized SVD
        # If k is None, compute all singular values (rank of matrix)
        k_actual = k if k is not None else min(matrix_rows, matrix_cols)
        print(f"Using multi-GPU randomized SVD with k={k_actual}...")
        s = multi_gpu_svd_from_matrix(G, k=k_actual, n_oversamples=10, n_iter=n_iter)
    else:
        # For smaller matrices, use direct torch SVD
        print("Using direct torch SVD...")
        s = torch.linalg.svdvals(G)
        s = s.cpu().numpy()

    t1 = time.perf_counter()
    print(f"SVD computed in {t1 - t0:.3f}s")

    print(f"\nFirst 10 singular values: {s[:10]}")
    print(f"Sum of first 10: {np.sum(s[:10]):.4f}")

    # Compute bitrate
    s_normalized = s / (n_sources_actual**0.5 * n_sensors**0.5)
    total_power = np.sum(np.abs(s_normalized) ** 2)
    noise_level = np.sqrt(total_power) / 2000.0  # SNR=2000
    bitrate = 0.5 * np.sum(np.log2(1 + (s_normalized / noise_level)**2))

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
        "params_hash": params_hash if save_results else None,
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
    k: int | None = None,
    n_iter: int = 1,
    list_only: bool = False,
    download: str = None,
):
    """
    Ultrasound analytical simulation on Modal with multi-GPU SVD.

    Usage:
        modal run modal_us_analytical.py --n-sources 32000 --n-sensors 1000
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
    print("Running Ultrasound Analytical Simulation on Modal")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n_sources: {n_sources}")
    print(f"  n_sensors: {n_sensors}")
    print(f"  temporal_sampling: {temporal_sampling}")
    print(f"  center_frequency: {center_frequency/1e3:.1f} kHz")
    print(f"  sensor_batch_size: {sensor_batch_size}")
    print(f"  k (SVD rank): {k if k is not None else 'all (full rank)'}")
    print(f"  n_iter (power iterations): {n_iter}")
    print()

    result = run_us_simulation.remote(
        n_sources=n_sources,
        n_sensors=n_sensors,
        temporal_sampling=temporal_sampling,
        center_frequency=center_frequency,
        sensor_batch_size=sensor_batch_size,
        use_multi_gpu_svd=True,
        k=k,
        n_iter=n_iter,
        save_results=True,
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
