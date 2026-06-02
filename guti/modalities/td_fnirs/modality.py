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
        # NOTE: the incoming merge renamed this to "td_fnirs", but the rest of the
        # pipeline (variant store, canonical npz, meta plot, noise model, existing
        # variants) all use "td_fnirs_analytical". Kept here for consistency; revisit
        # as a full rename during merge resolution if "td_fnirs" is preferred.
        return "td_fnirs_analytical"

    @property
    def noise_model_name(self) -> str:
        return "td_fnirs_analytical"

    def _get_default_modality_params(self) -> Parameters:
        """Return default parameters for TD-fNIRS analytical modality."""
        return Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
            max_dist=50.0,
            n_time_gates=6,
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Asymptotic-bitrate configuration for time-domain fNIRS.

        Same diffusion-limited spatial plateau as CW fNIRS (dense scalp
        sampling + sub-blur-scale voxels). Time gates add depth information
        with diminishing returns under photon starvation; n_time_gates=12
        sits past the practical knee for the default 0.5-3.0 ns window. Values
        are a reasonable scaled-up estimate pending a dedicated sweep.
        """
        return Parameters(
            num_sensors=1600,
            grid_resolution_mm=2.0,
            max_dist=50.0,
            n_time_gates=12,
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
        Setup scalp optodes and volumetric grid points.

        Creates:
        - Optodes on the MIDA scalp surface when available
        - Grid points within brain volume
        """
        # Get optode positions on scalp.
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

        # Get valid source-detector pairs and outward normals from the head center.
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

        # Compute sensitivities gate-by-gate, writing each directly into a
        # preallocated (n_pairs, n_gates, n_points) buffer. This avoids holding the
        # per-gate list AND a torch.stack copy at once (which doubled/tripled peak
        # GPU memory and crashed large fine-grid x many-gate matrices). Rows end up
        # ordered [pair0_t0, pair0_t1, ..., pair1_t0, ...] after the final reshape.
        voxel_volume_mm3 = float(self.params.grid_resolution_mm) ** 3
        result = torch.empty(
            (n_pairs, n_gates, n_points), dtype=torch.float32, device=self.device
        )
        for gi, t_ns in enumerate(self.time_gates_ns):
            result[:, gi, :] = td_sensitivity_batched(
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = result.reshape(n_pairs * n_gates, n_points)
        result *= voxel_volume_mm3  # in-place: voxel-integrated transfer (mm^-1)
        self.params.forward_model_convention = "voxel_integrated_transfer"
        self.params.forward_model_units = "mm^-1"
        self.params.voxel_volume_mm3 = voxel_volume_mm3

        return result.cpu().numpy()

    def compute_svd(self) -> np.ndarray:
        """Singular values via the fast adaptive solver.

        The TD forward matrices at fine grids are large and extremely
        ill-conditioned, where the direct GPU SVD (cuSOLVER gesvdj) fails and a
        full eigendecomposition is slow. compute_svd_fast uses an exact full SVD
        for small matrices and a float32 randomized top-k SVD for large ones
        (resolved down to ~1e-5 of the largest singular value, which is all the
        sqrt(N) spectra and capacity need).
        """
        from guti.svd import compute_svd_fast

        return compute_svd_fast(self.jacobian)


if __name__ == "__main__":
    modality = TDfNIRSAnalytical()
    singular_values = modality.run()
