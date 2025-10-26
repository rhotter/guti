# adapted from https://github.com/rhotter/eeg-resolution/blob/main/utils.py
"""
Here, we use a 4-layer head model to compute the EEG transfer function from the inner-most layer to the scalp.

Example usage:
    radii = [7.9, 8.0, 8.6, 9.1]
    conductivities = [1, 5, 1 / 15, 1]
    l_max = 100
    H = compute_eeg_transfer_function(conductivities, radii, l_max)
"""
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from guti.core import BRAIN_RADIUS, SKULL_RADIUS, SCALP_RADIUS


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
    H = jnp.zeros(l_max)
    for l in range(l_max):
        H[l] = compute_eeg_transfer_function_l(conductivities, radii, l)
    return H


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
    N = len(radii)

    def compute_gamma(i, zetta_ip1):
        """
        Compute the gamma value for layer i using the zetta value of the next layer.

        Parameters:
        - i (int): Index of the current layer.
        - zetta_ip1 (float): Zetta value of the next layer.

        Returns:
        - float: Computed gamma value for the current layer.
        """
        sigma = conductivities[i] / conductivities[i + 1]
        return (sigma - zetta_ip1) / ((l + 1) * sigma + zetta_ip1)

    def compute_zetta(i, gamma_i):
        """
        Compute the zetta value for layer i using the gamma value of the current layer.

        Parameters:
        - i (int): Index of the current layer.
        - gamma_i (float): Gamma value of the current layer.

        Returns:
        - float: Computed zetta value for the current layer.
        """
        r_ratio = radii[i - 1] / radii[i]
        num = l * r_ratio**l - (l + 1) * gamma_i * 1 / r_ratio ** (l + 1)
        denom = r_ratio**l + gamma_i * (1 / r_ratio ** (l + 1))
        return num / denom

    def compute_A_B(i, A_im1, B_im1, gamma_i):
        """
        Compute the A and B values for layer i using the A and B values of the previous layer and the gamma value of the current layer.

        Parameters:
        - i (int): Index of the current layer.
        - A_im1 (float): A value of the previous layer.
        - B_im1 (float): B value of the previous layer.
        - gamma_i (float): Gamma value of the current layer.

        Returns:
        - tuple: Computed A and B values for the current layer.
        """
        r_ratio = radii[i - 1] / radii[i]
        num = A_im1 + B_im1
        denom = (r_ratio) ** l + gamma_i * (1 / r_ratio ** (l + 1))
        A = num / denom
        B = gamma_i * A
        return A, B

    gamma = jnp.zeros(N)
    zetta = jnp.zeros(N)
    A = jnp.zeros(N)
    B = jnp.zeros(N)

    # first compute gamma and zetta
    for i in reversed(range(N)):
        if i == N - 1:
            gamma[i] = l / (l + 1)
        else:
            gamma[i] = compute_gamma(i, zetta[i + 1])
        zetta[i] = compute_zetta(i, gamma[i])

    A[0] = 1
    B[0] = 1
    for i in range(1, N):
        A[i], B[i] = compute_A_B(i, A[i - 1], B[i - 1], gamma[i])

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

    # Compute spherical harmonics using JAX SciPy
    # sph_harm signature: (m, n, theta[azimuth], phi[polar])
    Y_lm = sph_harm(m, l, phi, theta)
    return Y_lm
