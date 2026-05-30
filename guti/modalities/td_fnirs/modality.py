"""
Time-domain fNIRS Analytical modality implementation.

This modality computes the Jacobian matrix for time-domain functional near-infrared
spectroscopy using an analytical solution based on the diffusion approximation.

Extends the continuous-wave (CW) formulation in fnirs_analytical to include
discrete time gates, where each gate provides different depth sensitivity.
"""

import numpy as np
import torch
from typing import Optional, List

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import get_sensor_positions, get_grid_positions
from guti.modalities.td_fnirs.utils import (
    td_sensitivity_batched,
    get_valid_source_detector_pairs,
)


class TDfNIRSAnalytical(ImagingModality):
    @property
    def name(self) -> str:
        return "td_fnirs_analytical"

    def _get_default_modality_params(self) -> Parameters:
        """Return default parameters for TD-fNIRS analytical modality."""
        return Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
            max_dist=50.0,
            n_time_gates=6,
        )

    def __init__(
        self,
        params: Optional[Parameters] = None,
    ):
        """
        Initialize TD-fNIRS analytical modality.

        Parameters
        ----------
        params : Parameters, optional
            Standard parameters (num_sensors, grid_resolution_mm, max_dist, n_time_gates)
        """
        super().__init__(params)

        # Generate time gates based on n_time_gates parameter
        # Evenly spaced from 0.5 to 3.0 ns
        n_gates = self.params.n_time_gates or 6
        self.time_gates_ns = list(np.linspace(0.5, 3.0, n_gates))

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optical properties (brain at ~800 nm, Jacques 2013 PMB)
        # Units: mm and ns
        self.mu_a = 0.013  # Absorption coefficient [mm^-1]
        self.mu_s_prime = 1.14  # Reduced scattering coefficient [mm^-1]
        self.n_tissue = 1.4  # Refractive index of tissue
        self.c_vacuum = 299.792  # Speed of light in vacuum [mm/ns]
        self.c = self.c_vacuum / self.n_tissue  # Speed of light in tissue [mm/ns]
        self.D = 1 / (3 * (self.mu_a + self.mu_s_prime))  # Diffusion coefficient [mm]

    def setup_geometry(self) -> None:
        """
        Setup hemisphere optodes and volumetric grid points.

        Creates:
        - Optodes on hemisphere surface (using Fibonacci spiral)
        - Grid points within brain volume
        """
        # Get optode positions on hemisphere
        self.sensor_positions = get_sensor_positions(self.params.num_sensors)

        # Get volumetric grid points within brain
        self.grid_points = get_grid_positions(self.params.grid_resolution_mm)

        self.params.num_brain_grid_points = len(self.grid_points)

    def compute_forward_model(self) -> np.ndarray:
        """
        Compute analytical time-domain sensitivity matrix (Jacobian).

        For each time gate, computes the sensitivity of detected photons at that
        time to absorption changes at each grid point. The sensitivity is computed
        via the adjoint formulation using the time-domain Green's function.

        Returns
        -------
        np.ndarray
            Sensitivity matrix of shape (n_valid_pairs * n_time_gates, n_grid_points)
            where each element J[i,j] is the voxel-integrated transfer function
            from absorption change at grid point j to measurement i. The
            underlying analytical sensitivity is a density sampled in mm
            coordinates; multiplying by voxel volume makes the returned
            transfer function have units of mm^-1.

            The rows are organized as:
            [pair_0_gate_0, pair_0_gate_1, ..., pair_0_gate_N,
             pair_1_gate_0, pair_1_gate_1, ..., pair_1_gate_N, ...]
        """
        # Convert to torch tensors and move to GPU
        grid_points_torch = torch.from_numpy(self.grid_points).float().to(self.device)
        sensor_positions_torch = (
            torch.from_numpy(self.sensor_positions).float().to(self.device)
        )

        # Get valid source-detector pairs and their outward normals on the hemisphere.
        (
            valid_sources,
            valid_source_normals,
            valid_detectors,
            valid_detector_normals,
        ) = get_valid_source_detector_pairs(sensor_positions_torch, self.params.max_dist)

        n_pairs = valid_sources.shape[0]
        n_points = grid_points_torch.shape[0]
        n_gates = len(self.time_gates_ns)

        print(f"  Valid S-D pairs: {n_pairs}")
        print(f"  Grid points: {n_points}")
        print(f"  Time gates: {self.time_gates_ns} ns")
        print(f"  Output matrix shape: ({n_pairs * n_gates}, {n_points})")

        # Compute sensitivities for each time gate (semi-infinite medium).
        all_sensitivities = []
        for t_ns in self.time_gates_ns:
            sensitivities = td_sensitivity_batched(
                pos=grid_points_torch,
                source_pos=valid_sources,
                source_normal=valid_source_normals,
                detector_pos=valid_detectors,
                detector_normal=valid_detector_normals,
                t=t_ns,
                D=self.D,
                mu_a=self.mu_a,
                c=self.c,
                mu_s_prime=self.mu_s_prime,
            )
            all_sensitivities.append(sensitivities)

        # Stack: shape becomes (n_gates, n_pairs, n_points)
        stacked = torch.stack(all_sensitivities, dim=0)

        # Reshape to (n_pairs * n_gates, n_points)
        # We want rows ordered as [pair0_t0, pair0_t1, ..., pair1_t0, pair1_t1, ...]
        # So transpose to (n_pairs, n_gates, n_points) then reshape
        stacked = stacked.permute(1, 0, 2)  # (n_pairs, n_gates, n_points)
        result = stacked.reshape(n_pairs * n_gates, n_points)
        voxel_volume_mm3 = float(self.params.grid_resolution_mm) ** 3
        self.params.forward_model_convention = "voxel_integrated_transfer"
        self.params.forward_model_units = "mm^-1"
        self.params.voxel_volume_mm3 = voxel_volume_mm3

        return (result * voxel_volume_mm3).cpu().numpy()


if __name__ == "__main__":
    modality = TDfNIRSAnalytical()
    singular_values = modality.run()
