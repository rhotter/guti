
"""
Adapted from https://github.com/rhotter/eeg-resolution/blob/main/utils.py
Based on https://ieeexplore.ieee.org/document/7782724

Here, we use a 4-layer head model to compute the EEG transfer function from the inner-most layer to the scalp.

Example usage:
    radii = [7.9, 8.0, 8.6, 9.1]
    conductivities = [1, 5, 1 / 15, 1]
    l_max = 100
    H = compute_eeg_transfer_function(conductivities, radii, l_max)
"""
# %%
import os
import time

# Configure JAX backend (e.g., Apple Metal) before importing jax
# Opt-in via environment: export GUTI_JAX_BACKEND=metal | cpu
from jax import config as jax_config
# Prefer 64-bit for stability where supported
jax_config.update("jax_enable_x64", True)
_guti_backend = os.environ.get("GUTI_JAX_BACKEND")
if _guti_backend in ("metal", "cpu"):
    jax_config.update("jax_platform_name", "metal" if _guti_backend == "metal" else "cpu")
del _guti_backend

import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from guti.core import BRAIN_RADIUS, CSF_RADIUS, SKULL_RADIUS, SCALP_RADIUS, get_grid_positions
from guti.data_utils import save_svd
from guti.parameters import Parameters


def compute_eeg_transfer_function(conductivities: list[float], radii: list[float] = [BRAIN_RADIUS, SKULL_RADIUS, SCALP_RADIUS], l_max: int = 100):
    """
    Compute the transfer function from the inner-most layer (brain) to outer-most layer (scalp), as in Theorem 1 of the paper.

    Parameters:
    - conductivities (list of float): List of electrical conductivities for each layer, starting from the inner-most to the outer-most.
    - radii (list of float): List of radii for each spherical layer, starting from the inner-most to the outer-most.
    - l_max (int): The maximum degree of spherical harmonics to compute.

    Returns:
    - numpy.ndarray: A vector of transfer function values for each degree up to l_max.
    """
    H = jnp.zeros(l_max + 1)
    for l in range(l_max + 1):
        H = H.at[l].set(compute_eeg_transfer_function_l(conductivities, radii, l))
    return H


def compute_gamma(i, zetta_ip1, l, conductivities):
    """
    Compute the gamma value for layer i using the zetta value of the next layer.

    Parameters:
    - i (int): Index of the current layer.
    - zetta_ip1 (float): Zetta value of the next layer.
    - l (int): The degree of spherical harmonics.
    - conductivities (list of float): List of electrical conductivities for each layer.

    Returns:
    - float: Computed gamma value for the current layer.
    """
    sigma = conductivities[i] / conductivities[i + 1]
    return (sigma - zetta_ip1) / ((l + 1) * sigma + zetta_ip1)


def compute_zetta(i, gamma_i, l, radii):
    """
    Compute the zetta value for layer i using the gamma value of the current layer.

    Parameters:
    - i (int): Index of the current layer.
    - gamma_i (float): Gamma value of the current layer.
    - l (int): The degree of spherical harmonics.
    - radii (list of float): List of radii for each spherical layer.

    Returns:
    - float: Computed zetta value for the current layer.
    """
    r_ratio = radii[i - 1] / radii[i]
    num = l * r_ratio**l - (l + 1) * gamma_i * 1 / r_ratio ** (l + 1)
    denom = r_ratio**l + gamma_i * (1 / r_ratio ** (l + 1))
    return num / denom


def compute_A_B(i, A_im1, B_im1, gamma_i, l, radii):
    """
    Compute the A and B values for layer i using the A and B values of the previous layer and the gamma value of the current layer.

    Parameters:
    - i (int): Index of the current layer.
    - A_im1 (float): A value of the previous layer.
    - B_im1 (float): B value of the previous layer.
    - gamma_i (float): Gamma value of the current layer.
    - l (int): The degree of spherical harmonics.
    - radii (list of float): List of radii for each spherical layer.

    Returns:
    - tuple: Computed A and B values for the current layer.
    """
    r_ratio = radii[i - 1] / radii[i]
    num = A_im1 + B_im1
    denom = (r_ratio) ** l + gamma_i * (1 / r_ratio ** (l + 1))
    A = num / denom
    B = gamma_i * A
    return A, B


def compute_gamma_zetta_A_B(conductivities, radii, l):
    """
    Compute all gamma, zetta, A, and B values for the l'th spherical harmonic.

    Parameters:
    - conductivities (list of float): List of electrical conductivities for each layer, starting from the inner-most to the outer-most.
    - radii (list of float): List of radii for each spherical layer, starting from the inner-most to the outer-most.
    - l (int): The degree of spherical harmonics to compute.

    Returns:
    - tuple: Four arrays (gamma, zetta, A, B) containing values for each layer.
    """
    N = len(radii)

    gamma = jnp.zeros(N)
    zetta = jnp.zeros(N)
    A = jnp.zeros(N)
    B = jnp.zeros(N)

    # first compute gamma and zetta
    for i in reversed(range(N)):
        if i == N - 1:
            gamma = gamma.at[i].set(l / (l + 1))
        else:
            gamma = gamma.at[i].set(compute_gamma(i, zetta[i + 1], l, conductivities))
        zetta = zetta.at[i].set(compute_zetta(i, gamma[i], l, radii))

    A = A.at[0].set(1)
    B = B.at[0].set(1)
    for i in range(1, N):
        A_i, B_i = compute_A_B(i, A[i - 1], B[i - 1], gamma[i], l, radii)
        A = A.at[i].set(A_i)
        B = B.at[i].set(B_i)

    return gamma, zetta, A, B


def compute_eeg_transfer_function_l(conductivities, radii, l):
    """
    Compute the EEG transfer function value for the l'th spherical harmonic.

    Parameters:
    - conductivities (list of float): List of electrical conductivities for each layer, starting from the inner-most to the outer-most.
    - radii (list of float): List of radii for each spherical layer, starting from the inner-most to the outer-most.
    - l (int): The degree of spherical harmonics to compute.

    Returns:
    - float: The EEG transfer function value for the l'th spherical harmonic.
    """
    gamma, zetta, A, B = compute_gamma_zetta_A_B(conductivities, radii, l)
    
    H_l = (A[-1] + B[-1]) / (A[0] + B[0])
    return H_l


def compute_spherical_harmonic(l: int, m: int, positions: jnp.ndarray) -> jnp.ndarray:
    # Convert positions to JAX array and normalize to unit sphere
    pos = jnp.asarray(positions)
    r = jnp.linalg.norm(pos, axis=1)
    x = pos[:, 0] / r
    y = pos[:, 1] / r
    z = pos[:, 2] / r

    # Compute theta (polar angle [0, pi]) and phi (azimuthal angle [0, 2pi])
    theta = jnp.arccos(jnp.clip(z, -1.0, 1.0))  # polar angle
    phi = jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)  # azimuthal angle

    # Compute spherical harmonics using JAX SciPy in a robust way
    # Ensure degree/order are arrays matching the sampling points
    m_arr = jnp.full(theta.shape, m, dtype=jnp.int32)
    l_arr = jnp.full(theta.shape, l, dtype=jnp.int32)
    # sph_harm signature: (m, n, theta[azimuth], phi[polar])
    Y_lm = sph_harm(m_arr, l_arr, phi, theta)
    return Y_lm


def compute_dipole_field(
    dipole_position: jnp.ndarray, dipole_moment: jnp.ndarray, positions: jnp.ndarray, center_position: jnp.ndarray, 
    radii: list[float], conductivities: list[float], l_max: int = 100) -> jnp.ndarray:
    """equation 7 from the paper"""
    rz = jnp.linalg.norm(dipole_position - center_position)

    position_rs = jnp.linalg.norm(positions - center_position, axis=1)
    # cos(theta) between (dipole_position - center_position) and (position - center_position) for each position
    v = dipole_position - center_position
    w = positions - center_position
    eps = 1e-12
    cos_thetas = jnp.sum(v * w, axis=1) / (jnp.maximum(rz, eps) * jnp.maximum(position_rs, eps))
    cos_thetas = jnp.clip(cos_thetas, -1.0, 1.0)
    
    def compute_alpha(l):
        gamma, zetta, A, B = compute_gamma_zetta_A_B(conductivities, radii, l)
        # Use a decaying factor for interior sources: (rz / r_brain)^(l+1)
        # This avoids overflow for large l and small rz.
        alpha = (1 / (gamma[0] + 1e-12)) * (rz / (radii[0] + 1e-12)) ** (l + 1)
        return alpha
    
    def legendre_polynomial(l, x):
        # Legendre polynomials of the first kind via recurrence:
        # P0(x) = 1, P1(x) = x,
        # Pn(x) = ((2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n
        if l == 0:
            return jnp.ones_like(x)
        if l == 1:
            return x
        P_nm2 = jnp.ones_like(x)  # P0
        P_nm1 = x                  # P1
        for n in range(2, l + 1):
            P_n = ((2 * n - 1) * x * P_nm1 - (n - 1) * P_nm2) / n
            P_nm2, P_nm1 = P_nm1, P_n
        return P_nm1
    
    values = jnp.zeros(len(positions))
    for l in range(l_max):
        alpha = compute_alpha(l)
        contribution = dipole_moment / (4 * jnp.pi * conductivities[0] * (jnp.maximum(rz, eps) ** 2)) * (
            alpha * (position_rs / radii[0]) ** l + (jnp.maximum(rz, eps) / jnp.maximum(position_rs, eps)) ** (l+1)
        ) * l * legendre_polynomial(l, cos_thetas)
        values = values + contribution

    return jnp.nan_to_num(values)


def _fibonacci_sphere(n):
    """~equal-solid-angle directions on S^2 (unit sphere)."""
    i = jnp.arange(n) + 0.5
    phi = jnp.pi * (3.0 - jnp.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * i / n
    rho = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    theta = jnp.arccos(jnp.clip(z, -1.0, 1.0))  # polar
    az = (i * phi) % (2 * jnp.pi)               # azimuth
    x = rho * jnp.cos(az)
    y = rho * jnp.sin(az)
    dirs = jnp.stack([x, y, z], axis=1)        # shape (n, 3)
    return dirs, theta, az  # (dirs on unit sphere), theta, phi

def _lm_list(l_max):
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]

def _design_matrix(theta, phi, l_max):
    N = theta.shape[0]
    lms = _lm_list(l_max)
    L = len(lms)
    A = jnp.zeros((N, L), dtype=jnp.complex64)
    for k, (l, m) in enumerate(lms):
        m_arr = jnp.full(theta.shape, m, dtype=jnp.int32)
        l_arr = jnp.full(theta.shape, l, dtype=jnp.int32)
        A = A.at[:, k].set(sph_harm(m_arr, l_arr, phi, theta))
    return A

def spherical_harmonic_decomposition(
    f, l_max: int = 50, r: float = 1.0, n_samples: int | None = None,
    method: str = "auto", lam: float | None = None, return_grid: bool = False
):
    """
    Decompose a scalar function on the sphere into spherical harmonics.

    Args:
      f: callable f(positions) -> values, positions shape (N,3) at radius r.
      l_max: maximum degree.
      r: sampling radius (only affects where f is evaluated; harmonics live on S^2).
      n_samples: number of sampling points; default ~ 6 * (#coeffs).
      method: "auto" | "quadrature" | "ls"
        - "quadrature": c_{lm} ≈ (4π/N) Σ f_i Y_{lm}^*(θ_i,φ_i)   (fast, needs good coverage).
        - "ls": weighted least squares with tiny Tikhonov λ (robust when N not >> (#coeffs)).
      lam: optional Tikhonov λ for "ls"; default picks a tiny stabilized value.
      return_grid: if True, also return dict with positions, theta, phi, f_samples.

    Returns:
      coeffs: complex array shape ((l_max+1)^2,), in order given by _lm_list(l_max).
      meta: dict with keys 'lms' and optionally 'grid' if return_grid=True.
    """
    L = (l_max + 1) ** 2
    if n_samples is None:
        n_samples = int(6 * L)  # decent oversampling for stability

    dirs, theta, phi = _fibonacci_sphere(n_samples)  # unit sphere
    positions = r * dirs
    f_vals = jnp.asarray(f(positions))  # shape (N,)

    # choose method
    if method == "auto":
        method = "quadrature" if n_samples >= 4 * L else "ls"

    if method == "quadrature":
        # equal-solid-angle weights: each point ≈ 4π/N
        w = (4.0 * jnp.pi) / n_samples
        A = _design_matrix(theta, phi, l_max)  # (N,L)
        # c = sum_i w f_i conj(Y_{i,:})
        coeffs = (jnp.conj(A) * f_vals[:, None]).sum(axis=0) * w

    elif method == "ls":
        # weighted least squares with W ≈ I (equal-solid-angle). You could
        # swap in spherical Voronoi areas here if you have them.
        A = _design_matrix(theta, phi, l_max)  # (N,L)
        Aw = A  # equal weights
        fw = f_vals
        AtA = jnp.conj(Aw).T @ Aw
        Atf = jnp.conj(Aw).T @ fw
        if lam is None:
            lam = 1e-6 * (jnp.trace(AtA).real / L)
        coeffs = jnp.linalg.solve(AtA + lam * jnp.eye(L, dtype=AtA.dtype), Atf)
    else:
        raise ValueError("method must be 'auto', 'quadrature', or 'ls'")

    meta = {"lms": _lm_list(l_max)}
    if return_grid:
        meta["grid"] = {
            "positions": positions,
            "theta": theta,
            "phi": phi,
            "f_samples": f_vals,
        }
    return coeffs, meta

def reconstruct_on(theta, phi, coeffs, l_max):
    A = _design_matrix(theta, phi, l_max)
    return jnp.real(A @ coeffs)  # if you expect a real field


def compute_outer_harmonics_for_dipoles(dipole_positions: jnp.ndarray, dipole_moments: jnp.ndarray, center_position: jnp.ndarray, radii: list[float], conductivities: list[float], l_max: int = 100, n_samples_for_decomposition: int | None = None):
    def field_function(positions: jnp.ndarray):
        # Shift sampling positions to absolute coordinates centered at the head center
        positions_abs = positions + center_position
        results = jnp.zeros(len(positions_abs))
        for dipole_position, dipole_moment in zip(dipole_positions, dipole_moments):
            results = results + jnp.nan_to_num(
                compute_dipole_field(
                    dipole_position,
                    dipole_moment,
                    positions_abs,
                    center_position,
                    radii,
                    conductivities,
                    l_max,
                )
            )
        return results
    # Prefer stable quadrature over LS to avoid ill-conditioned solves during autodiff
    L = (l_max + 1) ** 2
    n_samples = n_samples_for_decomposition if n_samples_for_decomposition is not None else int(6 * L)
    n_samples = int(jnp.maximum(n_samples, 4 * L))
    inner_harmonics, meta = spherical_harmonic_decomposition(
        field_function,
        l_max,
        BRAIN_RADIUS,
        n_samples=n_samples,
        method="quadrature",
    )
    inner_harmonics = jnp.nan_to_num(inner_harmonics)
    transfer_function = compute_eeg_transfer_function(conductivities, radii, l_max)  # shape (l_max+1,)
    transfer_function = jnp.nan_to_num(transfer_function)
    # Expand transfer function across m for each l
    lms = meta["lms"]
    weights = jnp.array([transfer_function[l] for (l, m) in lms])
    outer_harmonics = inner_harmonics * weights
    return outer_harmonics


def harmonics_to_phi_theta_grid(harmonics: jnp.ndarray, num_phi: int = 100, num_theta: int = 100):
    """
    Take harmonics as in the spherical harmonic decomposition, generate a phi-theta grid, and evaluate the function on the grid.
    """
    # Infer l_max from coefficient vector length
    L = harmonics.shape[0]
    l_max = int(jnp.sqrt(L).astype(int)) - 1
    # Create grid (avoid exact poles and 2π to keep sph_harm stable)
    phi_step = (2.0 * jnp.pi) / num_phi
    theta_step = jnp.pi / num_theta
    phi = jnp.arange(num_phi) * phi_step + 0.5 * phi_step
    theta = jnp.arange(num_theta) * theta_step + 0.5 * theta_step
    phi_grid, theta_grid = jnp.meshgrid(phi, theta)  # shapes (num_theta, num_phi)
    # Flatten for evaluation
    phi_flat = phi_grid.reshape(-1)
    theta_flat = theta_grid.reshape(-1)
    # Accumulate reconstruction
    result_flat = jnp.zeros_like(phi_flat, dtype=jnp.complex64)
    lms = _lm_list(l_max)
    for idx, (l, m) in enumerate(lms):
        m_arr = jnp.full(theta_flat.shape, m, dtype=jnp.int32)
        l_arr = jnp.full(theta_flat.shape, l, dtype=jnp.int32)
        Y_lm = sph_harm(m_arr, l_arr, phi_flat, theta_flat)
        result_flat = result_flat + harmonics[idx] * Y_lm
    result = result_flat.reshape(theta_grid.shape)
    return jnp.real(jnp.nan_to_num(result))


# Jacobians w.r.t. dipole moments
def jacobian_outer_harmonics_wrt_dipole_moments(
    dipole_positions: jnp.ndarray,
    dipole_moments: jnp.ndarray,
    center_position: jnp.ndarray,
    radii: list[float],
    conductivities: list[float],
    l_max: int = 100,
    n_samples_for_decomposition: int | None = None,
):
    """Jacobian d Re[outer_harmonics] / d dipole_moments.

    Returns array of shape ((l_max+1)^2, num_dipoles).
    """
    def f(dmom: jnp.ndarray) -> jnp.ndarray:
        outer = compute_outer_harmonics_for_dipoles(
            dipole_positions,
            dmom,
            center_position,
            radii,
            conductivities,
            l_max=l_max,
            n_samples_for_decomposition=n_samples_for_decomposition,
        )
        return jnp.real(outer)

    return jax.jacrev(f)(dipole_moments)


def jacobian_grid_wrt_dipole_moments(
    dipole_positions: jnp.ndarray,
    dipole_moments: jnp.ndarray,
    center_position: jnp.ndarray,
    radii: list[float],
    conductivities: list[float],
    l_max: int = 100,
    n_samples_for_decomposition: int | None = None,
    num_phi: int = 100,
    num_theta: int = 100,
):
    """Jacobian d scalp_grid / d dipole_moments.

    Returns array of shape (num_theta*num_phi, num_dipoles), grid flattened row-major.
    """
    def g(dmom: jnp.ndarray) -> jnp.ndarray:
        outer = compute_outer_harmonics_for_dipoles(
            dipole_positions,
            dmom,
            center_position,
            radii,
            conductivities,
            l_max=l_max,
            n_samples_for_decomposition=n_samples_for_decomposition,
        )
        grid = harmonics_to_phi_theta_grid(outer, num_phi=num_phi, num_theta=num_theta)
        vec = grid.reshape(-1)
        return jnp.nan_to_num(vec)

    return jax.jacrev(g)(dipole_moments)

def compute_svd(dipole_grid_spacing: float = 20., l_max: int = 40, n_samples_for_decomposition: int | None = None, num_phi: int = 50, num_theta: int = 50):
    if n_samples_for_decomposition is None:
        n_samples_for_decomposition = num_phi * num_theta
    dipole_positions = get_grid_positions(dipole_grid_spacing)
    dipole_moments = jnp.array([1.])
    center_position = jnp.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.])
    radii = jnp.array([BRAIN_RADIUS, CSF_RADIUS, SKULL_RADIUS, SCALP_RADIUS])
    conductivities = [1, 5, 1 / 15, 1]
    start_time = time.time()
    # harmonics = compute_outer_harmonics_for_dipoles(dipole_positions, dipole_moments, center_position, radii, conductivities, l_max=40, n_samples_for_decomposition=1000)
    jac = jacobian_grid_wrt_dipole_moments(dipole_positions, dipole_moments, center_position, radii, conductivities, l_max=l_max, n_samples_for_decomposition=n_samples_for_decomposition, num_phi=num_phi, num_theta=num_theta)
    u, s, vh = jnp.linalg.svd(jac, full_matrices=False)
    save_svd(s, 'eeg_analytical', Parameters(
        num_brain_grid_points=len(dipole_positions),
        source_spacing_mm=dipole_grid_spacing,
        num_sensors=num_phi * num_theta,
    ))
    print(f"Time taken: {time.time() - start_time} seconds")
    print(jac)
    print(jnp.max(jnp.abs(jac)))

# %%
if __name__ == "__main__":
    compute_svd()

# %%
