"""
fNIRS Analytical modality implementation using continuous-wave sensitivity.

This modality computes the Jacobian matrix for functional near-infrared spectroscopy
using an analytical closed-form solution based on the diffusion approximation.

Uses equation 14.8 from Bigio & Fantini "Quantitative Biomedical Optics"
"""

import numpy as np
import torch
from typing import Optional

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import get_sensor_positions, get_grid_positions
from guti.modalities.cw_fnirs.utils import (
    cw_sensitivity_batched,
    get_valid_source_detector_pairs,
)

class CWfNIRS(ImagingModality):
    @property
    def name(self) -> str:
        return "cw_fnirs"

    def _get_default_modality_params(self) -> Parameters:
        """Return default parameters for fNIRS analytical modality."""
        return Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
            max_dist=50.0,
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Converged (asymptotic-capacity) configuration for CW fNIRS.

        Derived from the SVD-spectrum convergence of the saved sweeps in
        results/variants/cw_fnirs (reproduce with
        scripts/analyze_svd_convergence.py). fNIRS is diffusion-limited, so the
        spectrum saturates early:

          * num_sensors: the capacity proxy changes only ~0.6% from 800->1024
            (and ~0.8% 1024->2048) with the effective rank flat, at
            grid=6 mm / max_dist=40 mm. Knee ~512-600.
          * grid_resolution_mm: the voxel grid reaches its continuum limit by
            ~2.5 mm; 3.0 mm is already within ~1% (num_sensors=256 sweep).
          * max_dist: the spectrum is unchanged (<1%) beyond 30-40 mm; 50-70 mm
            add nothing (num_sensors=800 / grid=6 mm sweep).

        The previous estimate (1600, 2.0 mm, 50 mm) sat well past these knees,
        i.e. correct but needlessly expensive.
        """
        return Parameters(
            num_sensors=1024,
            grid_resolution_mm=2.5,
            max_dist=40.0,
        )

    def __init__(self, params: Optional[Parameters] = None):
        """
        Initialize fNIRS analytical modality.

        Parameters
        ----------
        params : Parameters, optional
            Standard parameters (num_sensors, grid_resolution_mm, etc.)
        max_dist : float, default=50.0
            Maximum source-detector distance (mm) for valid pairs
        """
        super().__init__(params)

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_geometry(self) -> None:
        """
        Setup hemisphere optodes and volumetric grid points.

        Creates:
        - Optodes on hemisphere surface (using Fibonacci spiral)
        - Grid points within brain volume
        - Filters valid source-detector pairs by distance
        """
        # Get optode positions on hemisphere
        self.sensor_positions = get_sensor_positions(self.params.num_sensors)

        # Get volumetric grid points within brain
        self.grid_points = get_grid_positions(self.params.grid_resolution_mm)

        self.params.num_brain_grid_points = len(self.grid_points)

    def compute_forward_model(self) -> np.ndarray:
        """
        Compute analytical sensitivity matrix (Jacobian).

        Uses batched computation to handle large numbers of source-detector pairs
        without exceeding GPU memory.

        Returns
        -------
        np.ndarray
            Sensitivity matrix of shape (n_valid_pairs, n_grid_points)
            where each element J[i,j] is the voxel-integrated transfer
            function from absorption change at grid point j to measurement i.
            The underlying analytical sensitivity is a density sampled in
            mm coordinates; multiplying by voxel volume makes the returned
            transfer function have units of mm^-1.
        """
        # Tissue optical properties (brain at ~800 nm, Jacques 2013 PMB)
        mu_a = 0.013  # Absorption coefficient [mm^-1]
        mu_s_prime = 1.14  # Reduced scattering coefficient [mm^-1]
        mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))  # [mm^-1]

        # Convert to torch tensors and move to GPU
        grid_points_torch = torch.from_numpy(self.grid_points).float().to(self.device)
        sensor_positions_torch = (
            torch.from_numpy(self.sensor_positions).float().to(self.device)
        )

        # Get valid source-detector pairs (within max_dist)
        valid_sources, valid_detectors = get_valid_source_detector_pairs(
            sensor_positions_torch, self.params.max_dist
        )

        # Compute sensitivities
        sensitivities = cw_sensitivity_batched(
            pos=grid_points_torch,
            source_pos=valid_sources,
            detector_pos=valid_detectors,
            mu_eff=mu_eff,
        )
        voxel_volume_mm3 = float(self.params.grid_resolution_mm) ** 3
        self.params.forward_model_convention = "voxel_integrated_transfer"
        self.params.forward_model_units = "mm^-1"
        self.params.voxel_volume_mm3 = voxel_volume_mm3

        return sensitivities * voxel_volume_mm3


if __name__ == "__main__":
    modality = CWfNIRS()
    singular_values = modality.run()
