"""
Verify that Modal remote execution produces the same results as local execution.
"""

import modal
import numpy as np

# Use the same image as modal_us_analytical.py
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12",
        add_python=None,
    )
    .pip_install("scipy", "torch")
)

app = modal.App("verify-us-analytical", image=image)

# Test parameters
N_SOURCES = 1000
N_SENSORS = 100
TEMPORAL_SAMPLING = 5
CENTER_FREQUENCY = 0.05e6
SENSOR_BATCH_SIZE = 32

BRAIN_RADIUS = 80
SCALP_RADIUS = 92


@app.function(gpu="T4", timeout=300)
def run_modal_simulation():
    """Run the simulation on Modal and return singular values."""
    import torch
    import numpy as np

    def get_sensor_positions(n_sensors=100, offset=0):
        golden_angle = np.pi * (3 - np.sqrt(5))
        indices = np.arange(n_sensors)
        z = (indices + 0.5) / n_sensors
        theta = np.arccos(z)
        phi = golden_angle * indices
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        positions = np.stack([x, y, z], axis=1)
        positions = positions * (SCALP_RADIUS + offset) + np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0])
        return positions

    def get_grid_positions(grid_spacing_mm=5.0, radius=BRAIN_RADIUS):
        x_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
        y_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
        z_coords = np.arange(0, radius + grid_spacing_mm, grid_spacing_mm)
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        center = np.array([radius, radius, 0])
        distances = np.linalg.norm(grid_points - center, axis=1)
        inside_hemisphere = (distances <= radius) & (grid_points[:, 2] >= 0)
        return grid_points[inside_hemisphere]

    def simulate_free_field_propagation(
        source_positions, receiver_positions, source_signals,
        time_step, center_frequency, voxel_size,
        device="cuda", temporal_sampling=1,
    ):
        sound_speed = 1500.0
        source_positions = source_positions.to(device)
        receiver_positions = receiver_positions.to(device)
        source_signals = source_signals.to(device)
        voxel_size = voxel_size.to(device)

        distances = torch.cdist(
            receiver_positions.float().unsqueeze(0),
            source_positions.float().unsqueeze(0)
        )[0]

        zero_distances = distances == 0
        if torch.any(zero_distances):
            distances = torch.where(zero_distances, torch.tensor(1e-10, device=device), distances)

        wavelength = sound_speed / center_frequency
        wavenumber = 2 * torch.pi / wavelength
        spatial_step = torch.mean(voxel_size)
        propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * torch.pi * distances)

        travel_times = distances / sound_speed
        delay_steps = (torch.floor(travel_times / time_step)).int()

        num_sources = source_signals.shape[0]
        num_receivers = receiver_positions.shape[0]
        num_time_steps = source_signals.shape[1]

        selected_time_indices = torch.arange(0, num_time_steps, temporal_sampling, device=device)

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

    # Generate positions
    grid_spacing_mm = ((2/3) * np.pi * BRAIN_RADIUS**3 / N_SOURCES)**(1/3)
    source_positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm) * 1e-3
    sensor_positions = get_sensor_positions(n_sensors=N_SENSORS, offset=8) * 1e-3

    n_sources = source_positions.shape[0]

    # Time parameters
    time_step = 1e-1 / CENTER_FREQUENCY
    time_duration = 120e-6
    time_axis = np.arange(0, time_duration, time_step)
    source_signals = np.sin(2 * np.pi * time_axis * CENTER_FREQUENCY)
    source_signals = np.tile(source_signals, (n_sources, 1))

    nt = time_axis.shape[0] // TEMPORAL_SAMPLING + 1

    min_speed = 1500.0
    PPW = 24
    dx_m = min_speed / (PPW * CENTER_FREQUENCY)
    voxel_size = np.array([dx_m, dx_m, dx_m])

    device = "cuda"

    # Build matrix G
    source_positions_t = torch.tensor(source_positions, device=device)
    source_signals_t = torch.tensor(source_signals, device=device)
    voxel_size_t = torch.tensor(voxel_size, device=device)

    matrix_rows = N_SENSORS * nt
    G = torch.zeros((matrix_rows, n_sources), dtype=torch.float32, device=device)
    last_index = 0

    for start in range(0, N_SENSORS, SENSOR_BATCH_SIZE):
        end = min(start + SENSOR_BATCH_SIZE, N_SENSORS)
        receiver_positions_t = torch.tensor(sensor_positions[start:end], device=device)

        pf_chunk = simulate_free_field_propagation(
            source_positions_t, receiver_positions_t, source_signals_t,
            time_step, CENTER_FREQUENCY, voxel_size_t,
            device=device, temporal_sampling=TEMPORAL_SAMPLING
        )

        chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources).float()
        chunk_rows = chunk_matrix.shape[0]
        G[last_index:last_index + chunk_rows] = chunk_matrix
        last_index += chunk_rows

    # Compute SVD
    s = torch.linalg.svdvals(G)
    s = s.cpu().numpy()

    return {
        "singular_values": s.tolist(),
        "n_sources": n_sources,
        "n_sensors": N_SENSORS,
        "matrix_shape": list(G.shape),
    }


@app.local_entrypoint()
def main():
    import numpy as np

    print("=" * 60)
    print("Verifying Modal Remote Execution vs Local")
    print("=" * 60)

    # Run on Modal
    print("\nRunning simulation on Modal...")
    result = run_modal_simulation.remote()

    s_modal = np.array(result["singular_values"])
    print(f"Modal: n_sources={result['n_sources']}, matrix_shape={result['matrix_shape']}")
    print(f"Modal first 10 singular values: {s_modal[:10]}")

    # Load local results
    print("\nLoading local results from comparison script...")
    try:
        data = np.load('/tmp/us_comparison_data.npz')
        s_local = data['s_local']
        print(f"Local first 10 singular values: {s_local[:10]}")

        # Compare
        print("\n" + "=" * 60)
        print("COMPARISON: Modal Remote vs Local")
        print("=" * 60)

        min_len = min(len(s_local), len(s_modal))
        abs_diff = np.abs(s_local[:min_len] - s_modal[:min_len])
        rel_diff = abs_diff / (np.abs(s_local[:min_len]) + 1e-10)

        print(f"\n  First 10 singular values:")
        print(f"  {'Index':<8} {'Local':<15} {'Modal Remote':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
        print(f"  {'-'*68}")
        for i in range(min(10, min_len)):
            print(f"  {i:<8} {s_local[i]:<15.6e} {s_modal[i]:<15.6e} {abs_diff[i]:<15.6e} {rel_diff[i]:<15.6e}")

        print(f"\n  Summary statistics for all {min_len} singular values:")
        print(f"    Max absolute difference: {abs_diff.max():.6e}")
        print(f"    Mean absolute difference: {abs_diff.mean():.6e}")
        print(f"    Max relative difference: {rel_diff.max():.6e}")
        print(f"    Mean relative difference: {rel_diff.mean():.6e}")

        TOLERANCE = 1e-4  # Allow slightly more tolerance for remote execution
        if rel_diff.max() < TOLERANCE:
            print(f"\n  ✓ PASS: Modal remote matches local within tolerance ({TOLERANCE})")
        else:
            print(f"\n  ✗ FAIL: Modal remote differs from local by more than tolerance ({TOLERANCE})")

    except FileNotFoundError:
        print("  WARNING: Local comparison data not found. Run compare_us_implementations.py first.")
        print(f"\n  Modal-only results - First 20 singular values:")
        for i in range(min(20, len(s_modal))):
            print(f"    {i}: {s_modal[i]:.6e}")
