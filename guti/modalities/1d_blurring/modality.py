"""
1D Gaussian Blurring Modality

Models a 1D convolution operator that blurs a signal with a Gaussian kernel.
This serves as a simple test case for understanding how spatial resolution
affects the singular value spectrum.
"""

import numpy as np
from typing import Optional
from guti.base_modality import ImagingModality
from guti.parameters import Parameters


class Blur1D(ImagingModality):
    def __init__(self, params: Optional[Parameters] = None):
        """
        Relevant params:
        - num_brain_grid_points: number of input grid points
        - num_sensors: number of output grid points
        """
        super().__init__(params)
        self.L = 0.1 # in meters
        self.sigma = 0.01 # in meters

    def modality_name(self) -> str:
        return "1d_blurring"

    def setup_geometry(self) -> None:
        self.x_in = np.linspace(0, self.L, self.params.num_brain_grid_points)
        self.x_out = np.linspace(0, self.L, self.params.num_sensors)

        # For compatibility with base class expectations
        self.sources = self.x_in.reshape(-1, 1)  # Shape (N, 1)
        self.sensors = self.x_out.reshape(-1, 1)  # Shape (N, 1)

    def compute_forward_model(self) -> np.ndarray:
        """
        Compute the Gaussian convolution matrix.

        Returns
        -------
        np.ndarray
            Convolution matrix K of shape (num_voxels, num_voxels)
            where K[i,j] = exp(-((x_out[i] - x_in[j])^2) / (2*sigma^2))
            normalized by dx / (sigma * sqrt(2*pi))
        """
        dx = self.L / self.params.num_brain_grid_points

        # Create convolution kernel matrix
        K = np.zeros((self.params.num_sensors, self.params.num_brain_grid_points))
        for i in range(self.params.num_sensors):
            for j in range(self.params.num_brain_grid_points):
                # Gaussian kernel
                K[i, j] = np.exp(-((self.x_out[i] - self.x_in[j]) ** 2) / (2 * self.sigma**2))

        # Normalize the kernel
        K *= dx / (self.sigma * np.sqrt(2 * np.pi))

        return K
    
    def _get_default_modality_params(self) -> Parameters:
        return Parameters(num_brain_grid_points=128, num_sensors=128)


if __name__ == "__main__":
    # Run with default parameters
    modality = Blur1D()
    singular_values = modality.run()
