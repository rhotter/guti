import numpy as np

from guti.core import SCALP_RADIUS, N_SENSORS_DEFAULT
from guti.core import get_sensor_positions, get_voxel_mask
from guti.modalities.fnirs.medium import Medium
from guti.modalities.fnirs.sensor_geometry import SensorGeometry

HEAD_RADIUS_MM = SCALP_RADIUS
VOXEL_SIZE_MM = 1.0  # Target voxel size in millimeters


def hemisphere_3d(
    noptodes: int = N_SENSORS_DEFAULT, voxel_size_mm: float = VOXEL_SIZE_MM
):
    if noptodes % 2 != 0:
        raise ValueError("noptodes must be a multiple of 2")
    sensor_pos = get_sensor_positions(noptodes)
    sensor_pos /= voxel_size_mm
    mask = get_voxel_mask(voxel_size_mm)
    brain_mask = mask == 1

    # set optical properties
    g = 0.9  # anisotropy factor
    mua0 = 0.02  # background absorption [1/mm]
    mus0 = 0.67 / (1 - g)  # background scattering [1/mm]
    refr_index = 1.4  # refractive index

    optical_properties = np.array([[0, 0, 1, 1], [mua0, mus0, g, refr_index]])

    medium = Medium(mask.shape)
    mask[mask == 2] = 1
    mask[mask == 3] = 1
    medium.volume = mask
    medium.optical_properties = optical_properties

    source_pos = sensor_pos[::2]
    det_pos = sensor_pos[1::2]
    (nx, ny, nz) = medium.shape
    center_point = np.array([nx // 2, ny // 2, 0])
    src_dirs = center_point - source_pos
    src_dirs = src_dirs / np.linalg.norm(src_dirs, axis=1)[:, None]
    sensors = SensorGeometry(source_pos, det_pos, src_dirs)
    return medium, sensors, brain_mask
