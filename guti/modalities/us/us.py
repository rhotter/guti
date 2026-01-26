#%%
%load_ext autoreload
%autoreload 2

# %%
# ---- JAX memory behaviour ---------------------------------------------
import os
import math
import torch
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   # ⬅ no 75 % grab
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'   # optional, 30 %
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # safer allocator
# os.environ['CUDA_VISIBLE_DEVICES'] = ''                 # ← CPU-only fallback
# # ------------------------------------------------------------------------

torch.set_num_threads(os.cpu_count() or 192)

from jax import jit

from jwave import FourierSeries, FiniteDifferences
from jwave.acoustics.time_varying import simulate_wave_propagation, TimeWavePropagationSettings
from jwave.geometry import Medium
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import grad, value_and_grad
import jax
from jax import vmap

import jax

from scipy.sparse.linalg import LinearOperator, svds

from guti.core import get_bitrate, noise_floor_heuristic
from guti.modalities.us.utils import create_medium, create_sources, create_receivers, plot_medium, find_arrival_time
import scipy.sparse


# %%

@torch.no_grad()
def bitrate_slq_torch_gpu(
    A: torch.Tensor,
    noise_std_full_brain: float,
    time_resolution: float = 1.0,
    n_detectors: int | None = None,
    s: int = 16,
    t: int = 40,
    batch: int = 256,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
):
    assert A.device.type == "cuda"
    A = A.contiguous()
    m, n = A.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    n_eff = n_detectors if n_detectors is not None else 1
    alpha = torch.tensor(
        1.0 / (noise_std_full_brain**2 / n_eff),
        dtype=krylov_dtype,
        device=A.device,
    )
    ln2 = torch.tensor(math.log(2.0), dtype=krylov_dtype, device=A.device)

    A32 = A.to(compute_dtype, copy=False)

    if left:
        T32 = torch.empty((n, batch), dtype=compute_dtype, device=A.device)
        W32 = torch.empty((m, batch), dtype=compute_dtype, device=A.device)
    else:
        T32 = torch.empty((m, batch), dtype=compute_dtype, device=A.device)
        W32 = torch.empty((n, batch), dtype=compute_dtype, device=A.device)

    Q = torch.empty((d, batch), dtype=krylov_dtype, device=A.device)
    Qm1 = torch.zeros_like(Q)
    al = torch.empty((t, batch), dtype=krylov_dtype, device=A.device)
    be = torch.empty((t - 1, batch), dtype=krylov_dtype, device=A.device)

    def B_mv(Qk):
        Q32 = Qk.to(compute_dtype)
        if left:
            torch.matmul(A32.T, Q32, out=T32)
            torch.matmul(A32, T32, out=W32)
        else:
            torch.matmul(A32, Q32, out=T32)
            torch.matmul(A32.T, T32, out=W32)
        return W32.to(krylov_dtype)

    est = torch.zeros((), dtype=krylov_dtype, device=A.device)
    done = 0
    sqrt_d = math.sqrt(d)
    scale = torch.tensor(float(d), dtype=krylov_dtype, device=A.device)

    while done < s:
        b = min(batch, s - done)

        Z = (torch.randint(0, 2, (d, b), device=A.device) * 2 - 1).to(krylov_dtype)
        Q[:, :b] = Z / sqrt_d
        Qm1[:, :b].zero_()

        for k in range(t):
            W = B_mv(Q[:, :b])
            if k > 0:
                W -= Qm1[:, :b] * be[k - 1, :b][None, :]
            ak = torch.sum(Q[:, :b] * W, dim=0)
            W -= Q[:, :b] * ak[None, :]
            al[k, :b] = ak
            if k < t - 1:
                bk = torch.linalg.vector_norm(W, dim=0)
                be[k, :b] = bk
                mask = bk > 1e-30
                Qm1[:, :b] = Q[:, :b]
                Q[:, :b] = torch.where(mask[None, :], W / bk[None, :], Q[:, :b])

        for j in range(b):
            tj = t
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device=A.device)
            Tj.diagonal(0).copy_(al[:tj, j])
            if tj > 1:
                off = be[:tj - 1, j]
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            est += scale * torch.dot(w1, torch.log1p(alpha * evals))

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))


@torch.no_grad()
def bitrate_slq_torch_gpu_chunked(
    A_cpu: np.ndarray | torch.Tensor,
    noise_std_full_brain: float,
    time_resolution: float = 1.0,
    n_detectors: int | None = None,
    s: int = 16,
    t: int = 40,
    batch: int = 256,
    use_left_if_smaller: bool = True,
    compute_dtype=torch.float32,
    krylov_dtype=torch.float64,
    device: str = "cuda",
    chunk_rows: int = 1024,
    normalize_scale: float = 1.0,
):
    if isinstance(A_cpu, torch.Tensor):
        assert A_cpu.device.type == "cpu"
    m, n = A_cpu.shape
    left = (m <= n) if use_left_if_smaller else True
    d = m if left else n

    n_eff = n_detectors if n_detectors is not None else 1
    alpha = torch.tensor(
        1.0 / (noise_std_full_brain**2 / n_eff),
        dtype=krylov_dtype,
        device=device,
    )
    ln2 = torch.tensor(math.log(2.0), dtype=krylov_dtype, device=device)

    Q = torch.empty((d, batch), dtype=krylov_dtype, device=device)
    Qm1 = torch.zeros_like(Q)
    al = torch.empty((t, batch), dtype=krylov_dtype, device=device)
    be = torch.empty((t - 1, batch), dtype=krylov_dtype, device=device)

    def get_chunk(start, end):
        if isinstance(A_cpu, torch.Tensor):
            chunk = A_cpu[start:end]
            return chunk.to(device=device, dtype=compute_dtype, non_blocking=False)
        return torch.as_tensor(A_cpu[start:end], dtype=compute_dtype, device=device)

    def B_mv(Qk):
        b = Qk.shape[1]
        if left:
            T32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
            for row_start in range(0, m, chunk_rows):
                row_end = min(row_start + chunk_rows, m)
                A_chunk = get_chunk(row_start, row_end)
                if normalize_scale != 1.0:
                    A_chunk = A_chunk * normalize_scale
                Q_chunk32 = Qk[row_start:row_end].to(compute_dtype)
                T32.addmm_(A_chunk.T, Q_chunk32)
            W32 = torch.empty((m, b), dtype=compute_dtype, device=device)
            for row_start in range(0, m, chunk_rows):
                row_end = min(row_start + chunk_rows, m)
                A_chunk = get_chunk(row_start, row_end)
                if normalize_scale != 1.0:
                    A_chunk = A_chunk * normalize_scale
                W32[row_start:row_end] = A_chunk @ T32
            return W32.to(krylov_dtype)
        Q32 = Qk.to(compute_dtype)
        W32 = torch.zeros((n, b), dtype=compute_dtype, device=device)
        for row_start in range(0, m, chunk_rows):
            row_end = min(row_start + chunk_rows, m)
            A_chunk = get_chunk(row_start, row_end)
            if normalize_scale != 1.0:
                A_chunk = A_chunk * normalize_scale
            T_chunk = A_chunk @ Q32
            W32.addmm_(A_chunk.T, T_chunk)
        return W32.to(krylov_dtype)

    est = torch.zeros((), dtype=krylov_dtype, device=device)
    done = 0
    sqrt_d = math.sqrt(d)
    scale = torch.tensor(float(d), dtype=krylov_dtype, device=device)

    while done < s:
        b = min(batch, s - done)
        Z = (torch.randint(0, 2, (d, b), device=device) * 2 - 1).to(krylov_dtype)
        Q[:, :b] = Z / sqrt_d
        Qm1[:, :b].zero_()

        for k in range(t):
            W = B_mv(Q[:, :b])
            if k > 0:
                W -= Qm1[:, :b] * be[k - 1, :b][None, :]
            ak = torch.sum(Q[:, :b] * W, dim=0)
            W -= Q[:, :b] * ak[None, :]
            al[k, :b] = ak
            if k < t - 1:
                bk = torch.linalg.vector_norm(W, dim=0)
                be[k, :b] = bk
                mask = bk > 1e-30
                Qm1[:, :b] = Q[:, :b]
                Q[:, :b] = torch.where(mask[None, :], W / bk[None, :], Q[:, :b])

        for j in range(b):
            tj = t
            Tj = torch.zeros((tj, tj), dtype=krylov_dtype, device=device)
            Tj.diagonal(0).copy_(al[:tj, j])
            if tj > 1:
                off = be[:tj - 1, j]
                Tj.diagonal(1).copy_(off)
                Tj.diagonal(-1).copy_(off)
            evals, evecs = torch.linalg.eigh(Tj)
            w1 = evecs[0, :] ** 2
            est += scale * torch.dot(w1, torch.log1p(alpha * evals))

        done += b

    bits_per_sample = est / (s * ln2)
    return float(bits_per_sample / (2.0 * time_resolution))

#NOTE: There's a bug in jwave, where the gradients are not computed correctly when using FiniteDifferences. Therefore, we use the FourierSeries class instead.

"""Check if JAX is using CUDA."""
platforms = jax.devices()
is_cuda = any('cuda' in str(device).lower() for device in platforms)
print(f"JAX is using CUDA: {is_cuda}")

print("Creating medium")

domain, medium_original, time_axis, brain_mask, skull_mask, scalp_mask = create_medium()

sources, source_mask = create_sources(domain, time_axis, freq_Hz=0.1666e6)
sensors, sensors_all, receivers_mask = create_receivers(domain, time_axis, freq_Hz=0.1666e6)

find_arrival_time_vectorized = vmap(lambda signal2: find_arrival_time(signal2, sources))

print("Creating solver functions")
# Compile and create the solver functions
@jit
def solver_all(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors_all)

@jit
def solver_receiver(medium, sources):
  return simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)

plot_medium(medium_original, source_mask, sources, time_axis, receivers_mask)

pml_size = medium_original.pml_size

#%%

# # Plot of the forward simulation for sanity check
# N = domain.N
# pressure_0 = solver_all(medium_original, sources).reshape(-1, N[0], N[1], N[2], 1)
# pressure_0_numpy = np.array(pressure_0)

# # Plot a slice at a specific time step (e.g., middle of the simulation)
# time_step = pressure_0_numpy.shape[0] - 1  # Last time step
# plt.figure(figsize=(10, 8))
# plt.imshow(pressure_0_numpy[time_step, N[0]//2, pml_size:-pml_size, pml_size:-pml_size, 0].T, cmap='seismic')
# plt.colorbar(label='Pressure')
# plt.title(f'Pressure field at time step {time_step}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

#%%

# Function mapping speed of sound field, sources, and sensors to pressure field

settings = TimeWavePropagationSettings(checkpoint=True)
# Jax function to map a speed of sound map to the corresponding pressure field
def output_field(s, sources, sensors):
    sound_speed2 = FourierSeries(s, domain)
    # Recreate the sound_speed and medium with the new speed
    density_field = jax.lax.stop_gradient(medium_original.density)
    medium = Medium(domain=domain, sound_speed=sound_speed2, density=density_field, pml_size=pml_size)
    # Get pressure field
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors, settings=settings)
    # pressure = simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors)
    return pressure

#%%

# Subsampling the voxels in the speed of sound field to create the contrast sources (the inputs to the imaging forward model)

speed = medium_original.sound_speed.on_grid[...,0]

contrast_sources_mask = jnp.full(speed.shape, False)

# Set the contrast sources mask to be every nth voxel inside the brain mask
n = 10  # Take every 10th voxel
brain_indices = jnp.argwhere(brain_mask)
selected_indices = brain_indices[::n]  # Take every nth index

# Create the contrast sources mask
contrast_sources_mask = contrast_sources_mask.at[tuple(selected_indices.T)].set(True)

# Print the number of contrast source points
num_contrast_points = jnp.sum(contrast_sources_mask)
print(f"Number of contrast source points: {num_contrast_points}")

n_inputs = len(selected_indices)


#%%

print("Computing Jacobian")
nt = time_axis.Nt

n_sensors = len(sensors.positions[0])

combined_jacobian = np.zeros((int(n_sensors * nt), int(n_inputs)), dtype=np.float32)

print(f"Jacobian shape: {combined_jacobian.shape}")

n_outputs_filled = 0

for i in range(20):

  print(f"Computing Jacobian for time shift {i}")

  def receiver_output(speed_contrast_sources):
      speed_of_sound = speed.at[contrast_sources_mask].set(speed_contrast_sources)
      pressure = output_field(speed_of_sound, sources, sensors)

      pressure_downsampled = pressure[i::20,:,0].flatten()
      
      return pressure_downsampled

  speed_contrast_sources = speed[contrast_sources_mask]


  jacobian = jax.jacrev(receiver_output)(speed_contrast_sources)

  combined_jacobian[n_outputs_filled:n_outputs_filled+jacobian.shape[0], :] = jacobian

  n_outputs_filled += jacobian.shape[0]



# %%

# Compute the singular value spectrum.
u, s, vh = np.linalg.svd(np.array(combined_jacobian))

# Compute bitrate from the SVD spectrum (reference computation).
s_normalized = s / math.sqrt(n_inputs * n_sensors)
noise_level = noise_floor_heuristic(s_normalized, heuristic="power", snr=2000.0)
bitrate_exact = get_bitrate(s_normalized, noise_level, time_resolution=1.0)
print(f"noise_level: {noise_level}")
print(f"bitrate (exact, SVD): {bitrate_exact}")

# Approximate bitrate on GPU via Lanczos SLQ without forming Gram matrices.
if torch.cuda.is_available():
    bitrate_slq = bitrate_slq_torch_gpu_chunked(
        combined_jacobian,
        noise_std_full_brain=noise_level,
        time_resolution=1.0,
        s=16,
        t=40,
        batch=64,
        chunk_rows=1024,
        normalize_scale=1.0 / math.sqrt(n_inputs * n_sensors),
    )
    print(f"bitrate (SLQ GPU): {bitrate_slq}")
else:
    print("CUDA unavailable; skipping SLQ bitrate approximation.")

# Plot singular value spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(s)
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum for amplitude+arrival time based US imaging')
plt.show()

#%%

# Save results

from guti.data_utils import save_svd

save_svd(s, 'us')

np.save('combined_jacobian.npy', combined_jacobian)

#%%

# # Example of computing gradients of an objective function with respect to the speed of sound map.
# # We need to perturb the speed of sound map so that the gradients are not zero.


# target = jnp.array(np.array(solver_receiver(medium_original, sources)))

# # Plot the target waveform at a specific point
# plt.figure(figsize=(10, 6))
# plt.plot(target[:,100,0])
# plt.title('Target waveform at point (100,100)')
# plt.xlabel('Time step')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()


# # Define a function that returns a scalar value from the pressure field
# # @jit
# def objective(s):

#     pressure = output_field(s, sources, sensors)

#     diff = (pressure - target)

#     # Compute mean squared error
#     return jnp.sum(diff**2)

# speed2 = jnp.array(np.copy(medium_original.sound_speed.on_grid[...,0].at[brain_mask].set(1650.)))

# # Plot the speed of sound map
# plt.figure(figsize=(10, 8))
# N = domain.N
# plt.imshow(speed2[N[0]//2, :, :], cmap='viridis')
# plt.colorbar(label='Speed of Sound (m/s)')
# plt.title('Speed of Sound Distribution')
# plt.xlabel('x (grid points)')
# plt.ylabel('y (grid points)')
# plt.show()

# # sound_speed2 = FourierSeries(speed2, domain)

# # Test the gradient computation with current speed
# obj_value, gradient = value_and_grad(objective)(speed2)

# print(f"Objective value: {obj_value}")
# print(f"Gradient shape: {gradient.shape}")
# print(f"Gradient max: {gradient.max()}")
# print(f"Gradient min: {gradient.min()}")

# # Compute gradient

# # Plot the gradient
# plt.figure(figsize=(10, 8))
# plt.imshow(gradient[N[0]//2, pml_size:-pml_size, pml_size:-pml_size].T, cmap='seismic')
# plt.colorbar(label='Gradient')
# plt.title('Gradient of objective with respect to sound speed')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# %%
