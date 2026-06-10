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

    def _get_default_modality_params(self) -> Parameters:
        return Parameters(
            num_sensors=256,
            source_spacing_mm=5.0,
            grid_resolution_mm=20.0,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.4,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Converged (asymptotic-capacity) configuration for EEG.

        Read off the SVD-spectrum convergence sweeps saved in
        results/variants/eeg_openmeeg and plotted in
        guti/modalities/eeg/results/ (reproduce with
        guti/modalities/eeg/plot_eeg_sweeps.py). The skull (~1/30 the
        conductivity of brain/scalp) low-pass filters the potential, so EEG
        resolves only a modest number of spatial modes; the normalized spectrum
        stops moving once each axis is refined past:

          * num_sensors = 2048: spectrum flat past ~1024-2048 electrodes
            (grid=8 mm, n_radial_lines=409 sweep).
          * grid_resolution_mm = 5.0: this is the BEM *mesh* resolution. The
            spectrum keeps resolving more modes as the mesh is refined down to
            ~5-6 mm and is converged there; the previous 15 mm estimate sat far
            short of that (it discarded roughly half the resolvable modes).
          * n_radial_lines = 409 (x n_dipoles_per_line = 5, three orthogonal
            orientations per location): the radial-line source layout the sweeps
            used; the spectrum is converged by ~274-409 lines.

        This uses the radial-line source layout the convergence study was run
        with, so compute_forward_model builds the forward model the same way.
        OpenMEEG assembly at a 5 mm mesh is heavy: this is the asymptote, not a
        quick default run.
        """
        return Parameters(
            num_sensors=2048,
            grid_resolution_mm=5.0,
            n_radial_lines=409,
            n_dipoles_per_line=5,
            output_spectrum_type="power_law",
            output_spectrum_beta=1.4,
            output_spectrum_min_freq_hz=DEFAULT_NEURAL_SPECTRUM_MIN_FREQ_HZ,
            output_spectrum_max_freq_hz=DEFAULT_NEURAL_SPECTRUM_MAX_FREQ_HZ,
            output_spectrum_bin_width_hz=DEFAULT_OUTPUT_POWER_LAW_BIN_WIDTH_HZ,
        )

    def setup_geometry(self) -> None:
        # Geometry is materialized as BEM files in compute_forward_model;
        # record the source-location count here for parameter tracking.
        if self.params.n_radial_lines is not None:
            # Radial-line dipole layout (the layout the EEG convergence sweeps
            # used): n_radial_lines x n_dipoles_per_line source locations.
            n_per_line = self.params.n_dipoles_per_line or 5
            self.sources = None
            self.params.num_brain_grid_points = (
                int(self.params.n_radial_lines) * int(n_per_line)
            )
        else:
            self.sources = get_grid_positions(grid_spacing_mm=self.params.source_spacing_mm)
            self.params.num_brain_grid_points = len(self.sources)

    def compute_forward_model(self) -> np.ndarray:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        create_eeg_bem_model(
            source_spacing_mm=self.params.source_spacing_mm or 5.0,
            n_sensors=self.params.num_sensors or 256,
            grid_resolution=self.params.grid_resolution_mm or 20.0,
            # When n_radial_lines is set, create_eeg_bem_model uses the radial
            # dipole layout and ignores source_spacing_mm (None otherwise ->
            # grid layout, unchanged default behaviour).
            n_radial_lines=self.params.n_radial_lines,
            n_dipoles_per_line=self.params.n_dipoles_per_line,
            output_dir=str(_MODEL_DIR),
        )

        return compute_eeg_leadfield_from_bem_dir(_MODEL_DIR)


if __name__ == "__main__":
    modality = EEGModality()
    singular_values = modality.run()
