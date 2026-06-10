"""MEG forward model for a spherical conductor (Sarvas formula).

This is the canonical, in-process MEG lead-field implementation used across the
project. The magnetic field outside a spherically symmetric conductor depends
only on the source geometry (the skull/scalp conductivities drop out), so the
forward model is a closed-form expression — no BEM/OpenMEEG run required.

Two sensor standoffs are modelled by the ``offset_mm`` argument:
  - OPM   ~7 mm from the scalp
  - SQUID ~25 mm from the scalp
"""

import numpy as np

from guti.core import get_sensor_positions, get_grid_positions, BRAIN_RADIUS

SPHERE_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

OPM_OFFSET_MM = 7.0
SQUID_OFFSET_MM = 25.0


def sarvas_formula(r, r0, center=SPHERE_CENTER):
    """Sarvas lead-field matrix M for a spherical conductor, B = M @ q.

    Parameters
    ----------
    r : array-like, shape (3,)
        Sensor position in head coordinates (mm).
    r0 : array-like, shape (3,)
        Dipole position in head coordinates (mm).
    center : array-like, shape (3,)
        Sphere center in head coordinates (mm).

    Returns
    -------
    M : ndarray, shape (3, 3)
        Lead-field matrix mapping a dipole moment to the field at the sensor.
    """
    mu0 = 4 * np.pi * 1e-7  # vacuum permeability

    # Sarvas assumes the sphere center is the origin; convert mm -> m.
    r = (np.asarray(r) - center) * 1e-3
    r0 = (np.asarray(r0) - center) * 1e-3

    a_vec = r - r0
    a = np.linalg.norm(a_vec)
    r_norm = np.linalg.norm(r)

    if a < 1e-12 or r_norm < 1e-12:
        return np.zeros((3, 3))

    F = a * (a * r_norm + r_norm**2 - np.dot(r0, r))
    if abs(F) < 1e-20:
        return np.zeros((3, 3))

    a_dot_r = np.dot(a_vec, r)
    nabla_F = (
        (a**2 / r_norm + a_dot_r / a + 2 * a + 2 * r_norm) * r
        - (a + 2 * r_norm + a_dot_r / a) * r0
    )

    # Cross-product matrix for r0 (so that r0_cross @ q = r0 × q)
    r0_cross = np.array(
        [
            [0.0, -r0[2], r0[1]],
            [r0[2], 0.0, -r0[0]],
            [-r0[1], r0[0], 0.0],
        ]
    )
    # Sarvas (1987) eq. 5: B = (mu0/4pi) [F*(Q x r0) - (Q x r0 . nabla_F)*r] / F^2
    # In matrix form: M = (mu0/4pi) [-F*r0_cross - outer(r, r0 x nabla_F)] / F^2
    M = (mu0 / (4 * np.pi)) * (-F * r0_cross - np.outer(r, np.cross(r0, nabla_F))) / (F**2)
    return M


def compute_forward_matrix(n_sensors, grid_spacing_mm, offset_mm):
    """Assemble the MEG forward matrix A (3*n_sensors, 3*n_sources).

    Each sensor measures the 3 field components; each source contributes a
    3-vector dipole moment, so A is built from 3x3 Sarvas blocks.
    """
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    A = np.zeros((3 * n_sensors, 3 * n_sources))
    for i, sensor in enumerate(sensors):
        for j, source in enumerate(sources):
            A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = sarvas_formula(sensor, source)
    return A


def compute_svd(n_sensors, grid_spacing_mm, offset_mm):
    """Singular values of the MEG forward matrix (descending)."""
    A = compute_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
    return np.linalg.svd(A, full_matrices=False, compute_uv=False)
