"""
Compare the local and Modal implementations of ultrasound analytical simulation.
Verifies that both produce the same singular value spectrum.
"""

import numpy as np
import torch

# Test parameters - small for quick comparison
N_SOURCES = 1000
N_SENSORS = 100
TEMPORAL_SAMPLING = 5
CENTER_FREQUENCY = 0.05e6  # 50 kHz
SENSOR_BATCH_SIZE = 32

print("=" * 60)
print("Comparing Local vs Modal US Analytical Implementations")
print("=" * 60)
print(f"Parameters: n_sources={N_SOURCES}, n_sensors={N_SENSORS}")
print(f"            temporal_sampling={TEMPORAL_SAMPLING}, center_freq={CENTER_FREQUENCY/1e3:.0f}kHz")
print()

# ============================================================================
# LOCAL IMPLEMENTATION (using guti package)
# ============================================================================
print("Running LOCAL implementation...")

from guti.modalities.us.utils import (
    create_medium,
    create_sources_real,
    create_receivers_real,
    simulate_free_field_propagation
)

# Create medium (for domain parameters)
domain, medium, time_axis_jwave, brain_mask, skull_mask, scalp_mask = create_medium(
    central_frequency=CENTER_FREQUENCY, pad=30
)

# Create positions
source_positions_local = create_sources_real(
    domain, time_axis_jwave, freq_Hz=CENTER_FREQUENCY,
    n_sources=N_SOURCES, inside=True, pad=30
)
sensor_positions_local = create_receivers_real(
    domain, time_axis_jwave, freq_Hz=CENTER_FREQUENCY,
    n_sensors=N_SENSORS, pad=30
)

n_sources_actual = source_positions_local.shape[0]
print(f"  Actual n_sources after grid generation: {n_sources_actual}")
print(f"  Source positions shape: {source_positions_local.shape}")
print(f"  Sensor positions shape: {sensor_positions_local.shape}")

# Time axis and signals
time_step = 1e-1 / CENTER_FREQUENCY
time_duration = 120e-6
time_axis = np.arange(0, time_duration, time_step)
source_signals = np.sin(2 * np.pi * time_axis * CENTER_FREQUENCY)
source_signals = np.tile(source_signals, (n_sources_actual, 1))

nt = time_axis.shape[0] // TEMPORAL_SAMPLING + 1
voxel_size = np.array(domain.dx)

print(f"  Time steps: {len(time_axis)}, nt after sampling: {nt}")
print(f"  Voxel size: {voxel_size}")

# Build matrix G
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

source_positions_t = torch.tensor(source_positions_local, device=device)
source_signals_t = torch.tensor(source_signals, device=device)
voxel_size_t = torch.tensor(voxel_size, device=device)

matrix_rows = N_SENSORS * nt
G_local = torch.zeros((matrix_rows, n_sources_actual), dtype=torch.float32, device=device)
last_index = 0

for start in range(0, N_SENSORS, SENSOR_BATCH_SIZE):
    end = min(start + SENSOR_BATCH_SIZE, N_SENSORS)
    receiver_positions_t = torch.tensor(sensor_positions_local[start:end], device=device)

    pf_chunk = simulate_free_field_propagation(
        source_positions_t,
        receiver_positions_t,
        source_signals_t,
        time_step,
        CENTER_FREQUENCY,
        voxel_size_t,
        device=device,
        compute_time_series=True,
        temporal_sampling=TEMPORAL_SAMPLING
    )

    chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources_actual).float()
    chunk_rows = chunk_matrix.shape[0]
    G_local[last_index:last_index + chunk_rows] = chunk_matrix
    last_index += chunk_rows

# Compute SVD
s_local = torch.linalg.svdvals(G_local)
s_local = s_local.cpu().numpy()

print(f"  Matrix G shape: {G_local.shape}")
print(f"  First 10 singular values: {s_local[:10]}")

# Save positions for Modal comparison
np.savez(
    '/tmp/us_comparison_data.npz',
    source_positions=source_positions_local,
    sensor_positions=sensor_positions_local,
    voxel_size=voxel_size,
    time_step=time_step,
    time_duration=time_duration,
    center_frequency=CENTER_FREQUENCY,
    temporal_sampling=TEMPORAL_SAMPLING,
    s_local=s_local,
    G_local=G_local.cpu().numpy()
)
print("  Saved comparison data to /tmp/us_comparison_data.npz")

# ============================================================================
# MODAL IMPLEMENTATION (self-contained functions)
# ============================================================================
print("\nRunning MODAL implementation (local simulation of Modal code)...")

# Replicate the Modal geometry functions
BRAIN_RADIUS = 80  # mm
SCALP_RADIUS = 92  # mm

def get_sensor_positions_modal(n_sensors=100, offset=0, start_n=0, end_n=None):
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

def get_grid_positions_modal(grid_spacing_mm=5.0, radius=BRAIN_RADIUS):
    x_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    y_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    z_coords = np.arange(0, radius + grid_spacing_mm, grid_spacing_mm)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    center = np.array([radius, radius, 0])
    distances = np.linalg.norm(grid_points - center, axis=1)
    inside_hemisphere = (distances <= radius) & (grid_points[:, 2] >= 0)
    return grid_points[inside_hemisphere]

def simulate_free_field_propagation_modal(
    source_positions,
    receiver_positions,
    source_signals,
    time_step,
    center_frequency,
    voxel_size,
    device="cuda",
    temporal_sampling=1,
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

# Generate positions using Modal functions
grid_spacing_mm = ((2/3) * np.pi * BRAIN_RADIUS**3 / N_SOURCES)**(1/3)
source_positions_modal = get_grid_positions_modal(grid_spacing_mm=grid_spacing_mm) * 1e-3  # mm to m
sensor_positions_modal = get_sensor_positions_modal(n_sensors=N_SENSORS, offset=8) * 1e-3  # mm to m

n_sources_modal = source_positions_modal.shape[0]
print(f"  Actual n_sources after grid generation: {n_sources_modal}")
print(f"  Source positions shape: {source_positions_modal.shape}")
print(f"  Sensor positions shape: {sensor_positions_modal.shape}")

# Use same time parameters
time_axis_modal = np.arange(0, time_duration, time_step)
source_signals_modal = np.sin(2 * np.pi * time_axis_modal * CENTER_FREQUENCY)
source_signals_modal = np.tile(source_signals_modal, (n_sources_modal, 1))

# Compute voxel size same way as Modal
min_speed = 1500.0
PPW = 24
dx_m = min_speed / (PPW * CENTER_FREQUENCY)
voxel_size_modal = np.array([dx_m, dx_m, dx_m])

nt_modal = time_axis_modal.shape[0] // TEMPORAL_SAMPLING + 1
print(f"  Time steps: {len(time_axis_modal)}, nt after sampling: {nt_modal}")
print(f"  Voxel size: {voxel_size_modal}")

# Build matrix G
source_positions_t_modal = torch.tensor(source_positions_modal, device=device)
source_signals_t_modal = torch.tensor(source_signals_modal, device=device)
voxel_size_t_modal = torch.tensor(voxel_size_modal, device=device)

matrix_rows_modal = N_SENSORS * nt_modal
G_modal = torch.zeros((matrix_rows_modal, n_sources_modal), dtype=torch.float32, device=device)
last_index = 0

for start in range(0, N_SENSORS, SENSOR_BATCH_SIZE):
    end = min(start + SENSOR_BATCH_SIZE, N_SENSORS)
    receiver_positions_t = torch.tensor(sensor_positions_modal[start:end], device=device)

    pf_chunk = simulate_free_field_propagation_modal(
        source_positions_t_modal,
        receiver_positions_t,
        source_signals_t_modal,
        time_step,
        CENTER_FREQUENCY,
        voxel_size_t_modal,
        device=device,
        temporal_sampling=TEMPORAL_SAMPLING
    )

    chunk_matrix = pf_chunk.permute(0, 2, 1).reshape(-1, n_sources_modal).float()
    chunk_rows = chunk_matrix.shape[0]
    G_modal[last_index:last_index + chunk_rows] = chunk_matrix
    last_index += chunk_rows

# Compute SVD
s_modal = torch.linalg.svdvals(G_modal)
s_modal = s_modal.cpu().numpy()

print(f"  Matrix G shape: {G_modal.shape}")
print(f"  First 10 singular values: {s_modal[:10]}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

# Check if positions match
print("\nPosition comparison:")
print(f"  Local sources: {source_positions_local.shape}, Modal sources: {source_positions_modal.shape}")
print(f"  Local sensors: {sensor_positions_local.shape}, Modal sensors: {sensor_positions_modal.shape}")

if source_positions_local.shape == source_positions_modal.shape:
    source_diff = np.abs(source_positions_local - source_positions_modal).max()
    print(f"  Max source position difference: {source_diff:.6e} m")
else:
    print(f"  WARNING: Different number of sources!")

if sensor_positions_local.shape == sensor_positions_modal.shape:
    sensor_diff = np.abs(sensor_positions_local - sensor_positions_modal).max()
    print(f"  Max sensor position difference: {sensor_diff:.6e} m")
else:
    print(f"  WARNING: Different number of sensors!")

# Compare voxel sizes
print(f"\nVoxel size comparison:")
print(f"  Local:  {voxel_size}")
print(f"  Modal:  {voxel_size_modal}")
print(f"  Max difference: {np.abs(voxel_size - voxel_size_modal).max():.6e}")

# Compare singular values
print(f"\nSingular value comparison:")
min_len = min(len(s_local), len(s_modal))
s_local_cmp = s_local[:min_len]
s_modal_cmp = s_modal[:min_len]

abs_diff = np.abs(s_local_cmp - s_modal_cmp)
rel_diff = abs_diff / (np.abs(s_local_cmp) + 1e-10)

print(f"  Number of singular values: Local={len(s_local)}, Modal={len(s_modal)}")
print(f"\n  First 10 singular values:")
print(f"  {'Index':<8} {'Local':<15} {'Modal':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
print(f"  {'-'*68}")
for i in range(min(10, min_len)):
    print(f"  {i:<8} {s_local_cmp[i]:<15.6e} {s_modal_cmp[i]:<15.6e} {abs_diff[i]:<15.6e} {rel_diff[i]:<15.6e}")

print(f"\n  Summary statistics for all {min_len} singular values:")
print(f"    Max absolute difference: {abs_diff.max():.6e}")
print(f"    Mean absolute difference: {abs_diff.mean():.6e}")
print(f"    Max relative difference: {rel_diff.max():.6e}")
print(f"    Mean relative difference: {rel_diff.mean():.6e}")

# Determine if they match
TOLERANCE = 1e-5
if rel_diff.max() < TOLERANCE:
    print(f"\n  ✓ PASS: Singular values match within tolerance ({TOLERANCE})")
else:
    print(f"\n  ✗ FAIL: Singular values differ by more than tolerance ({TOLERANCE})")
    print(f"    This may be due to differences in position generation between local and Modal implementations.")
