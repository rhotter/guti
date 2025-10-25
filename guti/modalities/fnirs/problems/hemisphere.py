import numpy as np
from medium import Medium
from sensor_geometry import SensorGeometry

from guti.core import SCALP_RADIUS

HEAD_RADIUS_MM = SCALP_RADIUS
VOXEL_SIZE_MM = 1.0  # Target voxel size in millimeters


def hemisphere_medium(
    is_3d=False,
):
    """
    Builds a hemisphere medium in either 2D or 3D

    Parameters
    ----------
    is_3d : bool, optional
        Whether to create a 3D (True) or 2D (False) medium. Default is False.

    Returns
    -------
    medium : Medium
        The medium (2D or 3D)
    """
    # Compute voxels_per_dim to achieve 1mm voxel size
    # We need enough voxels to cover the head diameter (2 * HEAD_RADIUS_MM)
    voxels_per_dim = int(2 * HEAD_RADIUS_MM / VOXEL_SIZE_MM)
    # Round up to nearest multiple of 2 for better discretization
    voxels_per_dim = (voxels_per_dim + 1) & ~1

    if is_3d:
        nz = voxels_per_dim // 2
    else:
        nz = 1
    ny = voxels_per_dim // 2
    nx = voxels_per_dim
    medium = Medium(
        (nz, ny, nx),
    )

    # head
    medium.add_ball((nz // 2, ny, nx // 2), HEAD_RADIUS_MM, 1)

    # set optical properties
    g = 0.9  # anisotropy factor
    mua0 = 0.02  # background absorption [1/mm]
    mus0 = 0.67 / (1 - g)  # background scattering [1/mm]
    refr_index = 1.4  # refractive index

    medium.optical_properties = np.array([[0, 0, 1, 1], [mua0, mus0, g, refr_index]])

    return medium


def hemisphere_2d_sensors(noptodes: int, medium: Medium):
    """
    Builds a 2D sensor geometry for the hemisphere medium.

    Parameters
    ----------
    noptodes : int
        The number of optodes.
    medium : Medium
        The 2D medium.

    Returns
    -------
    sensors : SensorGeometry
        The 2D sensor geometry for the hemisphere medium.
    """
    det_pos = np.zeros((noptodes, 2))
    src_pos = np.zeros((noptodes, 2))

    # Create an array of i values
    i_values = np.arange(noptodes)

    # Calculate phi for detectors and sources
    phi_sensors = i_values / noptodes * np.pi / 2 + np.pi / 4
    phi_sources = (i_values + 0.5) / noptodes * np.pi / 2 + np.pi / 4

    # Calculate sensor and source positions
    scaling_factor = 1
    center_point = np.array([0, medium.nx // 2])

    det_pos = (
        scaling_factor
        * HEAD_RADIUS_MM
        * np.vstack((np.sin(phi_sensors), np.cos(phi_sensors))).T
        + center_point
    )
    src_pos = (
        HEAD_RADIUS_MM * np.vstack((np.sin(phi_sources), np.cos(phi_sources))).T
        + center_point
    )

    # add a column of zeros to cast as 3D
    src_pos = np.hstack((medium.nz // 2 * np.ones((noptodes, 1)), src_pos))
    det_pos = np.hstack((medium.nz // 2 * np.ones((noptodes, 1)), det_pos))

    # Calculate source directions (pointing inward)
    center_point_3d = np.array([medium.nz // 2, 0, medium.nx // 2])
    src_dirs = center_point_3d - src_pos
    src_dirs = np.hstack((np.zeros((noptodes, 1)), src_dirs[:, 1:]))
    src_dirs = src_dirs / np.linalg.norm(src_dirs, axis=1)[:, None]

    sensors = SensorGeometry(src_pos, det_pos, src_dirs)
    return sensors


def hemisphere_3d_sensors(noptodes: int, medium: Medium):
    """
    Builds a 3D sensor geometry for the hemisphere medium with sensors distributed
    across the entire hemisphere surface using a Fibonacci sphere algorithm.

    Parameters
    ----------
    noptodes : int
        The number of optodes.
    medium : Medium
        The 3D medium.

    Returns
    -------
    sensors : SensorGeometry
        The 3D sensor geometry for the hemisphere medium.
    """
    det_pos = np.zeros((noptodes, 3))
    src_pos = np.zeros((noptodes, 3))

    # Fibonacci sphere algorithm for uniform distribution
    phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians
    i = np.arange(noptodes)
    y = 1 - (i / (noptodes - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    # Only keep points in upper hemisphere (y >= 0)
    y = np.abs(y)  # reflect negative y values to positive

    theta = phi * i  # golden angle increment
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    # Scale to head radius and shift to center
    center_point = np.array([medium.nz // 2, medium.ny, medium.nx // 2])

    # Set detector positions
    det_pos[:, 0] = HEAD_RADIUS_MM * y + center_point[0]  # z coordinate
    det_pos[:, 1] = HEAD_RADIUS_MM * x + center_point[1]  # y coordinate
    det_pos[:, 2] = HEAD_RADIUS_MM * z + center_point[2]  # x coordinate

    src_pos[:, 0] = HEAD_RADIUS_MM * y + center_point[0]
    src_pos[:, 1] = HEAD_RADIUS_MM * x + center_point[1]
    src_pos[:, 2] = HEAD_RADIUS_MM * z + center_point[2]

    # Calculate source directions (pointing inward)
    src_dirs = center_point - src_pos
    src_dirs = src_dirs / np.linalg.norm(src_dirs, axis=1)[:, None]

    sensors = SensorGeometry(src_pos, det_pos, src_dirs)
    return sensors


# For backward compatibility
def hemisphere_2d_medium(*args, **kwargs):
    """Backward compatibility wrapper for 2D medium"""
    return hemisphere_medium(*args, is_3d=False, **kwargs)
