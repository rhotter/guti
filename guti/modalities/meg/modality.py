"""MEG imaging modality (Sarvas spherical forward model).

Wraps the canonical in-process lead-field in :mod:`guti.modalities.meg.meg`.
The sensor standoff selects the variant: ``sensor_offset_mm`` near 7 mm is an
OPM array, near 25 mm is a SQUID array, and ``name`` reports the matching
canonical id (``meg_opm`` / ``meg_squid``) so results and noise models line up.
"""

from typing import Optional

import numpy as np

from guti.base_modality import ImagingModality
from guti.capacity import (
    DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
    DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
    DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
    noise_normalized_singular_values,
    sensor_noise_normalized_singular_values,
)
from guti.parameters import Parameters
from guti.core import get_sensor_positions, get_grid_positions
from guti.modalities.meg.johnson_noise import compute_meg_johnson_noise_covariance
from guti.modalities.meg.meg import (
    compute_forward_matrix,
    OPM_OFFSET_MM,
    SQUID_OFFSET_MM,
)
from guti.noise_models import compute_output_noise_std, get_noise_model

# Offsets at or below this (mm) are treated as OPM, above as SQUID.
_OPM_SQUID_THRESHOLD_MM = 0.5 * (OPM_OFFSET_MM + SQUID_OFFSET_MM)


class MEGModality(ImagingModality):
    @property
    def name(self) -> str:
        offset = self.params.sensor_offset_mm
        if offset is None or offset <= _OPM_SQUID_THRESHOLD_MM:
            return "meg_opm"
        return "meg_squid"

    def _get_default_modality_params(self) -> Parameters:
        return Parameters(
            num_sensors=1000,
            source_spacing_mm=5.0,
            sensor_offset_mm=OPM_OFFSET_MM,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.7,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
            noise_covariance_model="meg_johnson_radial",
            noise_correlation_kernel="meg_johnson",
            noise_distance_metric="finite_volume_head",
            noise_voxel_resolution_mm=4.0,
            noise_solver="finite_volume",
            noise_absolute_scale=False,
            noise_sensor_components="radial",
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Asymptotic-bitrate configuration for MEG (OPM standoff).

        The external magnetic field carries a limited number of spatial degrees
        of freedom, so adding sensors past a few thousand and refining the
        source grid below ~3 mm changes the bitrate by only a few percent.
        OPM standoff (7 mm) is used because it dominates SQUID (25 mm) at every
        sensor count. Values track the top of the MEG scaling sweeps.
        """
        return Parameters(
            num_sensors=4000,
            source_spacing_mm=3.0,
            sensor_offset_mm=OPM_OFFSET_MM,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.7,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
            noise_covariance_model="meg_johnson_radial",
            noise_correlation_kernel="meg_johnson",
            noise_distance_metric="finite_volume_head",
            noise_voxel_resolution_mm=4.0,
            noise_solver="finite_volume",
            noise_absolute_scale=False,
            noise_sensor_components="radial",
        )

    def setup_geometry(self) -> None:
        self.sensors = get_sensor_positions(
            self.params.num_sensors, offset=self.params.sensor_offset_mm
        )
        self.sources = get_grid_positions(grid_spacing_mm=self.params.source_spacing_mm)
        self.params.num_brain_grid_points = len(self.sources)

    def compute_forward_model(self) -> np.ndarray:
        return compute_forward_matrix(
            n_sensors=self.params.num_sensors,
            grid_spacing_mm=self.params.source_spacing_mm,
            offset_mm=self.params.sensor_offset_mm,
        )

    def _noise_normalized_extra_arrays(self) -> dict[str, np.ndarray]:
        """Return Johnson-covariance-whitened singular values for saved MEG runs."""

        model = self.params.noise_covariance_model or "meg_johnson_radial"
        if model not in {"meg_johnson_radial", "meg_johnson_xyz"}:
            raise ValueError(
                "MEGModality supports noise_covariance_model="
                "'meg_johnson_radial' or 'meg_johnson_xyz'"
            )
        sensor_components = "xyz" if model == "meg_johnson_xyz" else "radial"
        detector_noise = compute_output_noise_std(
            self.name,
            n_sensors=int(self.params.num_sensors),
        )
        noise_std = None if self.params.noise_absolute_scale else detector_noise
        bandwidth_hz = get_noise_model(self.name).reference_bandwidth_hz
        covariance, metadata, scale_info = compute_meg_johnson_noise_covariance(
            self.sensors,
            noise_std=noise_std,
            sensor_components=sensor_components,
            voxel_resolution_mm=float(self.params.noise_voxel_resolution_mm or 4.0),
            bandwidth_hz=bandwidth_hz,
            solver=self.params.noise_solver or "finite_volume",
            return_metadata=True,
        )
        if sensor_components == "radial":
            s_noise_normalized = sensor_noise_normalized_singular_values(
                self.jacobian,
                sensor_noise_covariance=covariance,
                outputs_per_sensor=3,
            )
        else:
            s_noise_normalized = noise_normalized_singular_values(
                self.jacobian,
                output_noise_covariance=covariance,
            )
        return {
            "noise_normalized_singular_values": s_noise_normalized,
            "noise_covariance_model": np.array(model),
            "noise_detector_std_t": np.array(detector_noise, dtype=np.float64),
            "noise_absolute_scale": np.array(
                bool(self.params.noise_absolute_scale),
                dtype=bool,
            ),
            "johnson_voxel_resolution_mm": np.array(
                float(self.params.noise_voxel_resolution_mm or 4.0),
                dtype=np.float64,
            ),
            "johnson_solver": np.array(self.params.noise_solver or "finite_volume"),
            "johnson_sensor_components": np.array(sensor_components),
            "johnson_n_voxels": np.array(metadata.n_voxels, dtype=np.int64),
            "johnson_bandwidth_hz": np.array(bandwidth_hz, dtype=np.float64),
            "johnson_raw_body_noise_median_T_per_sqrtHz": np.array(
                scale_info["raw_body_noise_median_T_per_sqrtHz"],
                dtype=np.float64,
            ),
            "johnson_regularization_jitter_T2": np.array(
                scale_info["regularization_jitter_T2"],
                dtype=np.float64,
            ),
        }

    def save_results(
        self,
        singular_values,
        default_run: bool = False,
        extra_arrays: Optional[dict] = None,
    ) -> None:
        arrays = {} if extra_arrays is None else dict(extra_arrays)
        if singular_values is not None and hasattr(self, "jacobian"):
            arrays.update(self._noise_normalized_extra_arrays())
        super().save_results(
            singular_values,
            default_run=default_run,
            extra_arrays=arrays,
        )


if __name__ == "__main__":
    modality = MEGModality()
    singular_values = modality.run()
