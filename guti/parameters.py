from dataclasses import dataclass, asdict
from typing import Optional, Dict
import json
import hashlib


@dataclass
class Parameters:
    """
    Parameters structure for SVD analysis.

    Attributes
    ----------
    num_sensors : int, optional
        Number of sensors used in the measurement
    grid_resolution_mm : float, optional
        Resolution of the computational grid (mesh resolution)
    source_spacing_mm : float, optional
        Spacing between dipole sources (for EEG/MEG grid-based method)
    n_radial_lines : int, optional
        Number of radial lines from center to brain surface (for radial dipole method)
    n_dipoles_per_line : int, optional
        Number of dipoles along each radial line (for radial dipole method)
    sensor_offset_mm : float, optional
        Distance of sensors from scalp surface (for MEG: 5mm=OPMs, 20mm=SQUIDs)
    num_brain_grid_points : int, optional
        Number of grid points in the brain model
    time_resolution : float, optional
        Temporal resolution of the measurement
    comment : str, optional
        Additional comment or description
    noise_full_brain : float, optional
        Noise level for the full brain
    """

    num_sensors: Optional[int] = None
    grid_resolution_mm: Optional[float] = None
    source_spacing_mm: Optional[float] = None
    n_radial_lines: Optional[int] = None
    n_dipoles_per_line: Optional[int] = None
    sensor_offset_mm: Optional[float] = None
    num_brain_grid_points: Optional[int] = None
    time_resolution: Optional[float] = None
    comment: Optional[str] = None
    noise_full_brain: Optional[float] = None
    noise_covariance_model: Optional[str] = None
    noise_correlation_length_mm: Optional[float] = None
    noise_correlation_kernel: Optional[str] = None
    noise_distance_metric: Optional[str] = None
    noise_voxel_resolution_mm: Optional[float] = None
    noise_solver: Optional[str] = None
    noise_absolute_scale: Optional[bool] = None
    noise_sensor_components: Optional[str] = None
    matrix_size: Optional[tuple[int, int]] = None
    vincent_trick: Optional[bool] = None
    frequency_hz: Optional[float] = None

    # for 1d blurring
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None

    # for fnirs
    max_dist: Optional[float] = None

    # for us (free-field ultrasound)
    temporal_sampling: Optional[int] = None

    # bitrate pipeline selection: "svd" (spectrum) or "slq" (matrix-free trace est.)
    bitrate_method: Optional[str] = None
    slq_num_probes: Optional[int] = None
    slq_num_lanczos: Optional[int] = None

    # output frequency spectrum for bitrate/capacity calculations
    output_spectrum_type: Optional[str] = None
    output_spectrum_beta: Optional[float] = None
    output_spectrum_min_freq_hz: Optional[float] = None
    output_spectrum_max_freq_hz: Optional[float] = None
    output_spectrum_bin_width_hz: Optional[float] = None

    # optional noise frequency spectrum for bitrate/capacity calculations
    noise_spectrum_type: Optional[str] = None
    noise_spectrum_beta: Optional[float] = None
    noise_spectrum_min_freq_hz: Optional[float] = None
    noise_spectrum_max_freq_hz: Optional[float] = None
    noise_spectrum_bin_width_hz: Optional[float] = None

    # for td_fnirs
    n_time_gates: Optional[int] = None
    forward_model_convention: Optional[str] = None
    forward_model_units: Optional[str] = None
    voxel_volume_mm3: Optional[float] = None

    # for reconstructed image modalities such as fMRI
    psf_fwhm_mm: Optional[float] = None
    bold_contrast: Optional[float] = None
    bold_snr: Optional[float] = None
    tsnr: Optional[float] = None
    hrf_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "Parameters":
        """Create Parameters object from dictionary."""
        # Build kwargs from all valid dataclass fields present in data
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                kwargs[field_name] = data[field_name]

        return cls(**kwargs)

    def __str__(self) -> str:
        fields = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                fields.append(f"{field}={value!r}")
        return f"Parameters({', '.join(fields)})" if fields else "Parameters()"
    
    def get_hash(self) -> str:
        params_dict = asdict(self)
        # Sort keys for consistent hashing
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict:
        """Return dictionary of only non-None fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}
