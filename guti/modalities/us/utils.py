from guti.core import get_voxel_mask, get_sensor_positions, get_grid_positions, get_sensor_positions
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch
import jax
import jwave
from jwave.geometry import Medium, TimeAxis


def create_medium(central_frequency: float = 0.25e6, pad: int = None):
    min_speed_of_sound = 1500.
    PPW = 24
    # Simulation parameters
    dx_m = min_speed_of_sound / (PPW * central_frequency)
    print(f"dx_m: {dx_m}")
    dx = (dx_m, dx_m, dx_m)

    tissues_map = get_voxel_mask(dx_m * 1e3, offset=8) #0.5mm resolution
    if pad is not None:
        if pad > 0:
            # Positive pad: add padding around the domain
            tissues_map = np.pad(tissues_map, ((pad, pad), (pad, pad), (pad, 0)), mode='constant', constant_values=0)
        elif pad < 0:
            # Negative pad: crop into the domain
            tissues_map = tissues_map[-pad:pad, -pad:pad, -pad:]
    N = tuple(tissues_map.shape)
    print(f"Domain size: {N}")
    domain = jwave.geometry.Domain(N, dx)

    # Set sound speed values based on tissue type
    # 0: outside (1500 m/s)
    # 1: brain (1525 m/s)
    # 2: skull (2400 m/s)
    # 3: scalp (1530 m/s)
    speed = jnp.ones_like(tissues_map, dtype=jnp.float32) * 1500.
    speed = jnp.where(tissues_map == 1, 1525., speed)
    speed = jnp.where(tissues_map == 2, 2400., speed)
    speed = jnp.where(tissues_map == 3, 1530., speed)
    sound_speed = jwave.FourierSeries(speed, domain)
    # Create density map with same shell mask
    # Use typical densities: ~1000 kg/m³ for water, ~2000 kg/m³ for the skull
    density = jnp.where(tissues_map == 0, 1000., 1000.)
    density = jnp.where(tissues_map == 1, 1000., 1000.)
    density = jnp.where(tissues_map == 2, 2000., 1000.)
    density = jnp.where(tissues_map == 3, 1000., 1000.)
    density_field = jwave.FourierSeries(density, domain)

    pml_size = 7
    # Pad the domain by the PML size to ensure proper absorption at boundaries
    domain = jwave.geometry.Domain(N, dx)

    extent = N[0] * dx[0]
    time_duration = extent / min_speed_of_sound
    print(f"time_duration: {time_duration}")

    # Update the tissue masks to match the padded domain
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density_field, pml_size=pml_size)
    time_axis = TimeAxis.from_medium(medium, cfl=0.15, t_end=time_duration)
    # medium = jwave.geometry.Medium(domain=domain, sound_speed=sound_speed, density=density_field, pml_size=pml_size)
    # time_axis = jwave.geometry.TimeAxis.from_medium(medium, cfl=0.15, t_end=50e-06)
    # # time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=5e-06)

    brain_mask = tissues_map == 1
    skull_mask = tissues_map == 2
    scalp_mask = tissues_map == 3

    return domain, medium, time_axis, brain_mask, skull_mask, scalp_mask

def create_sources_real(domain, time_axis, freq_Hz=0.25e6, inside: bool = False, n_sources: int = 400, pad: int = 0):
    """
    Create sources and source mask.
    """
    import jwave
    N = domain.N
    dx = domain.dx

    from guti.core import BRAIN_RADIUS

    grid_spacing_mm = ((2/3) * np.pi * BRAIN_RADIUS**3 / n_sources)**(1/3)

    # Get spiral source positions in world coordinates
    if not inside:
        source_positions = get_sensor_positions(n_sensors=n_sources, offset=8)
    else:
        source_positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    print("Got source positions")
    # Convert to voxel indices
    source_positions_voxels = jnp.floor(source_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = source_positions[:, 0], source_positions[:, 1], source_positions[:, 2]
    return np.stack([x_real, y_real, z_real], axis=1)*1e-3

def create_receivers_real(domain, time_axis, freq_Hz=0.25e6, n_sensors: int = 200, start_n: int = 0, end_n: int | None = None, spiral: bool = True, pad: int = 0):
    N = domain.N
    dx = domain.dx
    # Get spiral sensor positions in world coordinates
    if spiral:
        sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8, start_n=start_n, end_n=end_n)
    else:
        sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8, start_n=start_n, end_n=end_n)
    print("Got sensor positions")
    # Convert to voxel indices
    sensor_positions_voxels = jnp.floor(sensor_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2]
    return np.stack([x_real, y_real, z_real], axis=1)*1e-3


# Create sources and receivers used for non-free field simulations

def create_sources(domain, time_axis, freq_Hz=0.25e6, inside: bool = False, n_sources: int = 400, pad: int = 0):
    """
    Create sources and source mask.
    """
    import jwave
    N = domain.N
    dx = domain.dx

    from guti.core import BRAIN_RADIUS

    grid_spacing_mm = ((2/3) * np.pi * BRAIN_RADIUS**3 / n_sources)**(1/3)

    # Get spiral source positions in world coordinates
    if not inside:
        source_positions = get_sensor_positions(n_sensors=n_sources, offset=8)
    else:
        source_positions = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    print("Got source positions")
    # Convert to voxel indices
    source_positions_voxels = jnp.floor(source_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = source_positions[:, 0], source_positions[:, 1], source_positions[:, 2]
    x, y, z = source_positions_voxels[:, 0], source_positions_voxels[:, 1], source_positions_voxels[:, 2]
    pad_x = pad_y = pad_z = pad
    # Filter positions within the padded volume
    valid_indices = (
        (x_real >= -pad_x * dx[0] * 1e3) & (x_real < N[0] * dx[0] * 1e3 + pad_x * dx[0] * 1e3) &
        (y_real >= -pad_y * dx[1] * 1e3) & (y_real < N[1] * dx[1] * 1e3 + pad_y * dx[1] * 1e3) &
        (z_real >= -pad_z * dx[2] * 1e3) & (z_real < N[2] * dx[2] * 1e3 + pad_z * dx[2] * 1e3)
    )
    x += pad_x; y += pad_y; z += pad_z
    x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]
    N_sources = x.shape[0]

    # Create source signals
    signal = jnp.sin(2 * jnp.pi * freq_Hz * time_axis.to_array())
    T = 1 / freq_Hz
    max_signal_index = int(T / time_axis.dt)
    signal = signal.at[max_signal_index * 2 :].set(0)
    signals = jnp.stack([signal] * N_sources)

    # Instantiate sources
    sources = jwave.geometry.Sources(positions=(x, y, z), signals=signals, dt=time_axis.dt, domain=domain)

    # Create source mask
    source_mask = jnp.full((1,) + N, False)
    source_mask = source_mask.at[:, x, y, z].set(True)

    return sources, source_mask


def create_receivers(domain, time_axis, freq_Hz=0.25e6, n_sensors: int = 200, start_n: int = 0, end_n: int | None = None, spiral: bool = True, pad: int = 0):
    N = domain.N
    dx = domain.dx
    # Get spiral sensor positions in world coordinates
    if spiral:
        sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8, start_n=start_n, end_n=end_n)
    else:
        sensor_positions = get_sensor_positions(n_sensors=n_sensors, offset=8, start_n=start_n, end_n=end_n)
    print("Got sensor positions")
    # Convert to voxel indices
    sensor_positions_voxels = jnp.floor(sensor_positions / (jnp.array(dx) * 1e3)).astype(jnp.int32)
    x_real, y_real, z_real = sensor_positions[:, 0], sensor_positions[:, 1], sensor_positions[:, 2]

    x, y, z = sensor_positions_voxels[:, 0], sensor_positions_voxels[:, 1], sensor_positions_voxels[:, 2]
    pad_x = pad_y = pad_z = pad
    # Filter positions within the padded volume
    valid_indices = (
        (x_real >= -pad_x * dx[0] * 1e3) & (x_real < N[0] * dx[0] * 1e3 + pad_x * dx[0] * 1e3) &
        (y_real >= -pad_y * dx[1] * 1e3) & (y_real < N[1] * dx[1] * 1e3 + pad_y * dx[1] * 1e3) &
        (z_real >= -pad_z * dx[2] * 1e3) & (z_real < N[2] * dx[2] * 1e3 + pad_z * dx[2] * 1e3)
    )
    x += pad_x; y += pad_y; z += pad_z
    x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]

    # Create receiver mask
    receivers_mask = jnp.full(N, False)
    receivers_mask = receivers_mask.at[x, y, z].set(True)
    receiver_positions = jnp.argwhere(receivers_mask)

    # Instantiate sensors
    sensors = jwave.geometry.Sensors(positions=tuple(receiver_positions.T.tolist()))
    sensors_all = jwave.geometry.Sensors(positions=tuple(jnp.argwhere(jnp.ones(N)).T.tolist()))
    return sensors, sensors_all, receivers_mask

def plot_medium(medium, sources, sensors, time_axis):
    N = medium.domain.N
    dx = medium.domain.dx
    
    # Plot the speed of sound map with sources and receivers overlaid
    plt.figure(figsize=(10, 8))
    plt.imshow(medium.sound_speed.on_grid[N[0]//2, :, :, 0].T, cmap='viridis', origin='lower')
    plt.colorbar(label='Speed of Sound (m/s)')
    
    # Convert world coordinates (mm) to voxel indices for plotting
    if sources is not None and len(sources) > 0:
        # sources are in mm, convert to voxel indices
        sources_voxels = sources / (np.array(dx) * 1e3)
        plt.scatter(sources_voxels[:, 1], sources_voxels[:, 2], c='red', s=10, alpha=0.5, label='Sources', edgecolors='black', linewidths=0.5)
    
    if sensors is not None and len(sensors) > 0:
        # sensors are in mm, convert to voxel indices
        sensors_voxels = sensors / (np.array(dx) * 1e3)
        plt.scatter(sensors_voxels[:, 1], sensors_voxels[:, 2], c='blue', s=10, alpha=0.5, label='Receivers', edgecolors='black', linewidths=0.5)
    
    plt.title('Speed of Sound Distribution with Sources and Receivers')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.legend()
    plt.savefig('speed_of_sound.png')

    # Plot the source locations
    plt.figure(figsize=(10, 8))
    plt.imshow(np.ones((N[1], N[2])), cmap='binary', origin='lower')
    # Overlay source positions as scatter plot
    if sources is not None and len(sources) > 0:
        sources_voxels = sources / (np.array(dx) * 1e3)
        plt.scatter(sources_voxels[:, 1], sources_voxels[:, 2], c='red', s=10, alpha=0.5, label='Sources')
    plt.title('Source Locations')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.colorbar(label='Source Present')
    plt.legend()
    plt.savefig('source_mask.png')

    # Plot the receivers locations
    plt.figure(figsize=(10, 8))
    plt.imshow(np.ones((N[1], N[2])), cmap='binary', origin='lower')
    # Overlay sensor positions as scatter plot
    if sensors is not None and len(sensors) > 0:
        sensors_voxels = sensors / (np.array(dx) * 1e3)
        plt.scatter(sensors_voxels[:, 1], sensors_voxels[:, 2], c='blue', s=10, alpha=0.5, label='Receivers')
    plt.title('Receivers Locations')
    plt.xlabel('y (grid points)')
    plt.ylabel('z (grid points)')
    plt.colorbar(label='Receivers Present')
    plt.legend()
    plt.savefig('receivers_mask.png')

    # Plot the signal used for sources
    # Generate a sample signal based on time_axis and center frequency
    signal = np.sin(2 * np.pi * time_axis.to_array() * 0.25e6)  # Default center frequency
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis.to_array() * 1e6, signal)  # Convert time to microseconds
    plt.title('Source Signal')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('source_signal.png')



# Define function that converts the waveforms at the receiver sensors, to the final sensor output. In this case, we define the final sensor output to be the arrival time of the waveform for each sensor.

def find_arrival_time(signal2, sources):
  signal = sources.signals[0]
  # Cross-correlate with signal to get arrival times

  correlation = jnp.correlate(signal2, signal, mode='full')[-len(signal):]

  max_idx = jnp.argmax(correlation)
  correlation = correlation / (jnp.max(correlation) + 1e-8)  # Normalize for numerical stability

  # Use softmax-based differentiable argmax
  # Temperature parameter controls sharpness of the softmax
  temperature = 1e-2
  softmax_weights = jax.nn.softmax(correlation / temperature, axis=0)

  # Compute weighted sum of time indices
  time_axis_array = time_axis.to_array()
  arrival_times = jnp.sum(softmax_weights * time_axis_array, axis=0)

  return arrival_times


def simulate_free_field_propagation(
    source_positions: torch.Tensor,
    receiver_positions: torch.Tensor, 
    source_signals: torch.Tensor,
    time_step: float,
    center_frequency: float,
    voxel_size: torch.Tensor,
    device: str = "cpu",
    compute_time_series: bool = False,
    temporal_sampling: int = 1,
) -> torch.Tensor:
    """
    Simulates free field propagation using a free field propagator.
    
    Args:
        source_positions: Source positions in voxel coordinates [n_sources, 3]
        receiver_positions: Receiver positions in voxel coordinates [n_receivers, 3]
        source_signals: Source waveforms [n_sources, n_time_steps]
        time_step: Time step in seconds
        center_frequency: Base frequency in Hz
        voxel_size: Voxel size in meters [3]
        device: Device to run computation on
        compute_time_series: If True, returns the time-resolved pressure field
            sampled every ``temporal_sampling`` steps with shape
            [n_receivers, n_sources, ⌈n_time_steps/temporal_sampling⌉].
            If False (default), returns only the spatial propagator matrix
            [n_receivers, n_sources].
        temporal_sampling: Positive integer stride that determines the temporal
            subsampling factor applied when ``compute_time_series`` is True. A
            value of 1 (default) keeps the original resolution, whereas larger
            values reduce memory usage by computing only every *temporal_sampling*-th
            time sample.
        
    Returns:
        Tensor of shape depending on ``compute_time_series`` (see above).
    """
    # Free field medium properties (water-like)
    sound_speed = 1500.0  # Speed of sound in m/s
    
    # Move tensors to device
    source_positions = source_positions.to(device)
    receiver_positions = receiver_positions.to(device)
    source_signals = source_signals.to(device)
    voxel_size = voxel_size.to(device)
    
    # Calculate distances between all source-receiver pairs
    distances = torch.cdist(
        receiver_positions.float().unsqueeze(0), 
        source_positions.float().unsqueeze(0)
    )[0]

    # Check for zero distances which would cause division by zero
    zero_distances = distances == 0
    if torch.any(zero_distances):
        print("Found zero distances which would cause division by zero")
        # Replace zeros with small epsilon to avoid division by zero
        distances = torch.where(zero_distances, torch.tensor(1e-10, device=device), distances)
    
    # Calculate expected pressure based on point source formula from acoustic wave theory
    # Combines point source pressure equation p = (4πR)^(-1) * m''(t-R/c) (see Pierce's acoustics formula 4.3.8)
    # with K-wave's pressure-to-density source conversion
    # The mass source acceleration m'' is related to pressure through the wavenumber k
    # Final formula: P = 1/(4πR) * (2π/λ) * peak_pressure * dx^2
    # where R is distance, λ is wavelength, dx is voxel size
    
    # Calculate propagator factor (spatial Green's function component)
    wavelength = sound_speed / center_frequency
    wavenumber = 2 * torch.pi / wavelength
    spatial_step = torch.mean(voxel_size)
    propagator_factor = (2 * wavenumber * spatial_step**2) / (4 * torch.pi * distances)
    
    # If only the propagator matrix is required, return early
    if not compute_time_series:
        return propagator_factor * torch.exp(-1j * wavenumber * distances)
    
    # Calculate retardation times and time steps (only needed when computing full time series)
    travel_times = distances / sound_speed
    delay_steps = (torch.floor(travel_times / time_step)).int()
    
    num_sources = source_signals.shape[0]
    num_receivers = receiver_positions.shape[0]
    num_time_steps = source_signals.shape[1]
    
    # Determine which time indices will be computed based on temporal sampling
    if temporal_sampling < 1:
        raise ValueError("temporal_sampling must be a positive integer >= 1")

    selected_time_indices = torch.arange(0, num_time_steps, temporal_sampling, device=device)
    
    # Pad source waveforms with zeros at the beginning
    padded_source_signals = torch.cat([
        torch.zeros(num_sources, 1, device=device, dtype=source_signals.dtype), 
        source_signals
    ], dim=1)
    
    # Create source indices for broadcasting
    source_idx = torch.arange(num_sources).unsqueeze(0).expand(num_receivers, num_sources).to(device)
    
    # ------------------------------------------------------------------
    # Serial computation across the selected time steps to save memory
    # ------------------------------------------------------------------
    n_selected = selected_time_indices.shape[0]
    pressure_field = torch.empty(
        (num_receivers, num_sources, n_selected),
        dtype=padded_source_signals.dtype,
        device=device,
    )

    for idx, t_idx in enumerate(selected_time_indices):
        # Compute delayed time indices for this specific time step
        time_idx_matrix = t_idx - delay_steps + 1
        time_idx_matrix = torch.clamp(time_idx_matrix, min=0, max=padded_source_signals.shape[1] - 1)

        # Gather delayed signals for the current time step
        delayed_signals_step = padded_source_signals[source_idx, time_idx_matrix]

        # Apply propagator factor and store in output tensor
        pressure_field[:, :, idx] = delayed_signals_step * propagator_factor

    # Return the serially computed pressure field
    return pressure_field
