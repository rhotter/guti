from medium import Medium
from sensor_geometry import SensorGeometry
import pmcx
import numpy as np


class Solver:
    def __init__(
        self, medium: Medium, sensors: SensorGeometry, tstart=0, tend=1e-8, tstep=5e-10
    ):
        self.medium = medium
        self.sensors = sensors
        self.tstart = tstart
        self.tend = tend
        self.tstep = tstep
        self.nt = int((tend - tstart) / tstep)

    def forward(
        self,
        src_idx,
        nphoton=1e7,
        random_seed=1,
        voxel_size_mm=1.0,
        max_src_det_distance_mm=None,
    ):
        """
        Implements the forward monte carlo solver.

        Parameters:
        -----------
        src_idx : int
            Index of the source to use
        nphoton : float
            Number of photons to simulate
        random_seed : int
            Random seed for simulation
        voxel_size_mm : float
            Size of each voxel in mm
        max_src_det_distance_mm : float or None
            If provided, only use detectors within this distance from the source
        """
        # Get source position
        src_pos = self.sensors.src_pos[src_idx]

        # Filter detectors by distance if max_src_det_distance_mm is provided
        valid_dets_mask = None
        if max_src_det_distance_mm is not None:
            # Calculate distances from source to each detector
            distances = np.linalg.norm(self.sensors.det_pos - src_pos, axis=1)
            # Create a mask for detectors within the max distance
            valid_dets_mask = distances <= max_src_det_distance_mm
            # Filter detector positions and radii
            det_pos_filtered = self.sensors.det_pos[valid_dets_mask]
            det_radii_filtered = self.sensors.det_radii[valid_dets_mask]
            # need to append the radii column to the det_pos
            det_pos = np.hstack((det_pos_filtered, det_radii_filtered))
        else:
            # Use all detectors
            # need to append the radii column to the det_pos
            det_pos = np.hstack((self.sensors.det_pos, self.sensors.det_radii))

        volume = self.medium.volume

        config = {
            "seed": random_seed,
            "nphoton": nphoton,
            "vol": volume,
            "tstart": self.tstart,
            "tend": self.tend,
            "tstep": self.tstep,
            "srcpos": src_pos,
            "srcdir": self.sensors.src_dirs[src_idx],
            "prop": self.medium.optical_properties,
            "detpos": det_pos,
            "replaydet": -1,
            "issavedet": 1,
            "issrcfrom0": 1,
            "issaveseed": 1,
            "unitinmm": voxel_size_mm,
            "maxdetphoton": nphoton,
        }

        result = pmcx.mcxlab(config)

        # Store the detector mask in the result for use in jacobian function
        result["valid_dets_mask"] = valid_dets_mask
        result["total_detectors"] = len(self.sensors.det_radii)

        return result, config


def jacobian(forward_result, cfg):
    # one must define cfg['seed'] using the returned seeds
    cfg["seed"] = forward_result["seeds"]

    # one must define cfg['detphotons'] using the returned detp data
    cfg["detphotons"] = forward_result["detp"]["data"]

    # tell mcx to output absorption (μ_a) Jacobian
    cfg["outputtype"] = "jacobian"

    result = pmcx.mcxlab(cfg)

    J = result["flux"]  # Jacobian of shape (nz, ny, nx, nt, ndetectors)

    # If detectors were filtered during forward, expand J to include all detectors
    if (
        "valid_dets_mask" in forward_result
        and forward_result["valid_dets_mask"] is not None
    ):
        valid_dets_mask = forward_result["valid_dets_mask"]
        total_detectors = forward_result["total_detectors"]

        # Create a full Jacobian matrix with zeros for filtered out detectors
        J_full = np.zeros((*J.shape[:-1], total_detectors), dtype=J.dtype)

        # Fill in values for the detectors that were used
        valid_indices = np.where(valid_dets_mask)[0]
        for i, idx in enumerate(valid_indices):
            if i < J.shape[-1]:  # Make sure we don't go out of bounds
                J_full[..., idx] = J[..., i]

        J = J_full

    # Flip sign of jacobian (since dphi = -J @ dmua)
    J = -J
    return J
