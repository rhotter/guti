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
from guti.modalities.fnirs_analytical.utils import (
    cw_sensitivity_batched,
    get_valid_source_detector_pairs,
)


class FNIRSAnalytical(ImagingModality):
    """
    Functional near-infrared spectroscopy using analytical continuous-wave model.

    This modality places optodes (sensors) on a hemisphere surface and computes
    sensitivity to absorption changes in a volumetric grid of sources using
    the diffusion approximation.

    Parameters
    ----------
    num_sensors : int, default=800
        Number of optodes on the hemisphere surface
    grid_resolution_mm : float, default=6.0
        Spacing between grid points in the brain volume
    max_dist : float, default=50.0
        Maximum source-detector distance (mm) for valid pairs
    """

    def modality_name(self) -> str:
        return "fnirs_analytical_cw"

    def _get_default_modality_params(self) -> Parameters:
        """Return default parameters for fNIRS analytical modality."""
        return Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
        )

    def __init__(self, params: Optional[Parameters] = None, max_dist: float = 50.0):
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

        # Store experimental parameters
        self.max_dist = max_dist

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

        # Update actual grid point count
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
            where each element J[i,j] is the sensitivity of measurement i
            to absorption changes at grid point j.
        """
        # Compute effective attenuation coefficient from tissue constants
        # Physics constants (tissue optical properties)
        mu_a = 0.02  # Absorption coefficient (cm^-1)
        mu_s_prime = 6.7  # Reduced scattering coefficient (cm^-1)
        mu_eff = np.sqrt(3 * mu_a * (mu_s_prime + mu_a))
        mu_eff = mu_eff * 1e-1  # Convert cm^-1 to mm^-1

        # Convert to torch tensors and move to GPU
        grid_points_torch = torch.from_numpy(self.grid_points).float().to(self.device)
        sensor_positions_torch = (
            torch.from_numpy(self.sensor_positions).float().to(self.device)
        )

        # Get valid source-detector pairs (within max_dist)
        valid_sources, valid_detectors = get_valid_source_detector_pairs(
            sensor_positions_torch, self.max_dist
        )

        # Compute sensitivities
        sensitivities = cw_sensitivity_batched(
            pos=grid_points_torch,
            source_pos=valid_sources,
            detector_pos=valid_detectors,
            mu_eff=mu_eff,
        )

        return sensitivities


if __name__ == "__main__":
    # Example: Run single configuration
    params = Parameters(num_sensors=800, grid_resolution_mm=6.0)
    modality = FNIRSAnalytical(params)
    singular_values = modality.run()

    # Example: Run parameter sweep
    # run_fnirs_analytical_sweep(
    #     num_sensors_list=[400, 800],
    #     grid_spacing_list=[6.0, 8.0]
    # )
