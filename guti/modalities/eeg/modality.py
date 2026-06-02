"""EEG imaging modality (OpenMEEG BEM forward model)."""

from __future__ import annotations

from pathlib import Path

from guti.base_modality import ImagingModality
from guti.capacity import (
    DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
    DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
    DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
)
from guti.parameters import Parameters
from guti.core import create_eeg_bem_model, get_grid_positions
from guti.modalities.eeg.compute_eeg_leadfield import (
    compute_eeg_leadfield_from_bem_dir,
)

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "_openmeeg_model"


class EEGModality(ImagingModality):
    @property
    def name(self) -> str:
        return "eeg"

    @property
    def noise_model_name(self) -> str:
        return "eeg_openmeeg"

    def _get_default_modality_params(self) -> Parameters:
        return Parameters(
            num_sensors=256,
            source_spacing_mm=5.0,
            grid_resolution_mm=20.0,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.5,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Asymptotic-bitrate configuration for EEG.

        The skull (~1/30 the conductivity of brain/scalp) low-pass filters the
        potential, so EEG resolves only a modest number of spatial modes:
        capacity saturates by a few hundred electrodes and a few-mm source
        grid. ``grid_resolution_mm`` is the BEM mesh resolution (smaller = more
        accurate but much heavier in OpenMEEG); 15 mm keeps assembly tractable.
        Values are a reasonable scaled-up estimate pending a dedicated sweep.
        """
        return Parameters(
            num_sensors=512,
            source_spacing_mm=3.0,
            grid_resolution_mm=15.0,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.5,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        )

    def setup_geometry(self) -> None:
        # Geometry is materialized as BEM files in compute_forward_model;
        # record the grid size here for parameter tracking.
        self.sources = get_grid_positions(grid_spacing_mm=self.params.source_spacing_mm)
        self.params.num_brain_grid_points = len(self.sources)

    def compute_forward_model(self) -> np.ndarray:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        create_eeg_bem_model(
            source_spacing_mm=self.params.source_spacing_mm or 5.0,
            n_sensors=self.params.num_sensors or 256,
            grid_resolution=self.params.grid_resolution_mm or 20.0,
            output_dir=str(_MODEL_DIR),
        )

        return compute_eeg_leadfield_from_bem_dir(_MODEL_DIR)


if __name__ == "__main__":
    modality = EEGModality()
    singular_values = modality.run()
