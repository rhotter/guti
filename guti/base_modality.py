"""
Base class for all imaging modalities in GUTI.

Provides a unified interface for:
- Geometry setup (sources, sensors, tissue)
- Forward model computation (Jacobian/sensitivity matrix)
- SVD analysis
- Results storage with parameter tracking
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import time
from guti.parameters import Parameters


class ImagingModality(ABC):
    """
    Abstract base class for all imaging modalities.

    All modalities follow the same workflow:
    1. Setup geometry (sources, sensors, tissue)
    2. Compute forward model (Jacobian matrix)
    3. Perform SVD analysis
    4. Save results with parameter tracking

    Subclasses must implement:
    - modality_name(): Return string identifier
    - setup_geometry(): Define sources, sensors, tissue geometry
    - compute_forward_model(): Return Jacobian matrix (n_measurements, n_sources)
    - _get_default_modality_params(): Return default parameter dictionary

    Example usage:
        modality = FNIRSAnalytical(Parameters(num_sensors=400, grid_resolution_mm=8.0))
        singular_values = modality.run()
    """

    def __init__(self, params: Optional[Parameters] = None):
        """
        Initialize imaging modality with parameters.

        Parameters
        ----------
        params : Parameters, optional
            Parameters object. If None, uses defaults from _get_default_modality_params().
        """
        # Use provided params or defaults
        self.params = (
            params if params is not None else self._get_default_modality_params()
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return modality identifier string.

        Returns
        -------
        str
            Modality name (e.g., 'eeg', 'meg', 'fnirs_analytical_cw')
            Used for results storage and identification.
        """
        pass

    @abstractmethod
    def setup_geometry(self) -> None:
        """
        Define sources, sensors, and tissue geometry.

        Should populate at minimum:
        - self.sources: Array of source positions (n_sources, 3)
        - self.sensors: Array of sensor positions (n_sensors, 3)

        Optionally:
        - self.geometry: Modality-specific geometry object

        May update self.modality_params with computed values
        (e.g., actual num_brain_grid_points after filtering).
        """
        pass

    @abstractmethod
    def compute_forward_model(self) -> Any:
        """
        Compute the Jacobian/sensitivity matrix.

        Returns
        -------
        Any
            Jacobian matrix of shape (n_measurements, n_sources)
            where each element J[i,j] represents the sensitivity of
            measurement i to source j.
            Can be a NumPy array, PyTorch tensor, JAX array, or any other array-like object.
        """
        pass

    def _get_default_modality_params(self) -> Parameters:
        """
        Return default parameter dictionary for this modality.


        Standard fields will be automatically extracted for save_svd().
        Physics-specific fields are only used internally.

        Returns
        -------
        Parameters
            Default parameters. User-provided params will override these.

        Example
        -------
        return Parameters(
            # Standard Parameters fields
            num_sensors=800,
            grid_resolution_mm=6.0,
        )
        """
        return Parameters()

    def run(self, save_results: bool = True, default_run: bool = False) -> np.ndarray:
        """
        Execute complete pipeline: geometry → forward model → SVD → save.

        Parameters
        ----------
        save_results : bool, default=True
            Whether to save SVD results to disk with parameter tracking.
        default_run : bool, default=False
            Whether to save as default run configuration.

        Returns
        -------
        np.ndarray
            Singular values from SVD analysis.
        """
        t0 = time.perf_counter()
        print(f"[{self.name}] Setting up geometry...")
        self.setup_geometry()

        print(f"[{self.name}] Computing forward model...")
        start_time = time.perf_counter()
        self.jacobian = self.compute_forward_model()
        forward_time = time.perf_counter() - start_time
        print(f"[{self.name}] Forward model computed in {forward_time:.2f} seconds")

        self.params.matrix_size = tuple(self.jacobian.shape)
        print(f"[{self.name}] Jacobian shape: {self.jacobian.shape}")

        # Validate that the Jacobian is non-empty
        if self.jacobian.shape[0] == 0 or self.jacobian.shape[1] == 0:
            raise ValueError(
                f"Forward model produced empty Jacobian with shape {self.jacobian.shape}. "
                f"Current parameters: {self.params}"
            )
        print(f"[{self.name}] Computing SVD...")
        start_time = time.perf_counter()
        singular_values = self.compute_svd()
        svd_time = time.perf_counter() - start_time
        print(f"[{self.name}] SVD computed in {svd_time:.2f} seconds")

        if save_results:
            print(f"[{self.name}] Saving results...")
            self.save_results(singular_values, default_run=default_run)

        print(f"[{self.name}] Completed in {time.perf_counter() - t0:.2f} seconds")
        return singular_values

    def compute_svd(self) -> np.ndarray:
        """
        Perform SVD analysis with automatic GPU/CPU fallback.

        Returns
        -------
        np.ndarray
            Singular values in descending order.
        """
        from guti.svd import compute_svd_gpu, compute_svd_cpu

        return compute_svd_gpu(self.jacobian)

    def save_results(self, singular_values: np.ndarray, default_run: bool = False) -> None:
        """
        Save SVD results with parameter tracking.

        Parameters
        ----------
        singular_values : np.ndarray
            Singular values from SVD analysis.
        """
        from guti.data_utils import save_svd

        save_svd(singular_values, self.name, self.params, default_run=default_run)

    def __repr__(self) -> str:
        """String representation showing modality name and parameters."""
        return f"{self.__class__.__name__}({self.params})"
