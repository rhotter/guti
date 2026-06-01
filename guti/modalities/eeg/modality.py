"""EEG imaging modality (OpenMEEG BEM forward model).

Unlike MEG, the EEG forward problem depends on the skull/scalp conductivities,
so there is no simple closed form for the 3-layer head — the lead field is
assembled with OpenMEEG via a boundary-element pipeline.

``compute_forward_model``:
  1. writes the 3-layer BEM model + source/sensor geometry (guti.core), then
  2. shells out to ``compute_eeg_leadfield.sh <model_dir> <out_dir>``
     (OpenMEEG om_assemble / om_minverser / om_gain), then
  3. loads the resulting lead field (n_sensors x 3*n_grid_points) from the
     OpenMEEG ``.mat`` (HDF5) output.

Requires OpenMEEG (``om_*`` on PATH) and h5py; both are used lazily so the
class can be imported/inspected without them. This path has not been executed
in the current environment (no OpenMEEG installed).
"""

from pathlib import Path

import numpy as np

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import create_eeg_bem_model, get_grid_positions

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "_openmeeg_model"
_OUT_DIR = _HERE / "_openmeeg_out"
_LEADFIELD_SH = _HERE / "compute_eeg_leadfield.sh"


class EEGModality(ImagingModality):
    @property
    def name(self) -> str:
        return "eeg_openmeeg"

    def _get_default_modality_params(self) -> Parameters:
        return Parameters(
            num_sensors=256,
            source_spacing_mm=5.0,
            grid_resolution_mm=20.0,
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
        )

    def setup_geometry(self) -> None:
        # Geometry is materialized as BEM files in compute_forward_model;
        # record the grid size here for parameter tracking.
        self.sources = get_grid_positions(grid_spacing_mm=self.params.source_spacing_mm)
        self.params.num_brain_grid_points = len(self.sources)

    def compute_forward_model(self) -> np.ndarray:
        import subprocess
        import h5py

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _OUT_DIR.mkdir(parents=True, exist_ok=True)

        create_eeg_bem_model(
            source_spacing_mm=self.params.source_spacing_mm or 5.0,
            n_sensors=self.params.num_sensors or 256,
            grid_resolution=self.params.grid_resolution_mm or 20.0,
            output_dir=str(_MODEL_DIR),
        )

        result = subprocess.run(
            ["bash", str(_LEADFIELD_SH), str(_MODEL_DIR), str(_OUT_DIR)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "OpenMEEG EEG leadfield computation failed (is OpenMEEG on PATH?):\n"
                f"{result.stdout}\n{result.stderr}"
            )

        leadfield_path = _OUT_DIR / "eeg_leadfield.mat"
        if not leadfield_path.exists():
            raise FileNotFoundError(f"Leadfield not found at {leadfield_path}")
        # OpenMEEG writes a MATLAB v7.3 (HDF5) file with the gain under 'linop'.
        with h5py.File(leadfield_path, "r") as f:
            leadfield = np.array(f["linop"])
        return leadfield


if __name__ == "__main__":
    modality = EEGModality()
    singular_values = modality.run()
