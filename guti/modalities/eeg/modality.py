"""EEG imaging modality (OpenMEEG BEM forward model).

Unlike MEG, the EEG forward problem depends on the skull/scalp conductivities,
so there is no simple closed form for the 3-layer head — the lead field is
assembled with OpenMEEG via a boundary-element pipeline.

``compute_forward_model``:
  1. writes the 3-layer BEM model + source/sensor geometry (guti.core), then
  2. assembles the lead field in-process with OpenMEEG's Python bindings
     (HeadMat → invert → DipSourceMat / Head2EEGMat → GainEEG), the same
     pipeline the ``om_assemble``/``om_minverser``/``om_gain`` CLI tools run,
     and returns the gain as a dense array (n_sensors x 3*n_grid_points).

Requires the ``openmeeg`` Python package; it is imported lazily so the class
can be imported/inspected without it. This path has not been executed in the
current environment (OpenMEEG not installed).
"""

from pathlib import Path

import numpy as np

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import create_eeg_bem_model, get_grid_positions

_HERE = Path(__file__).parent
_MODEL_DIR = _HERE / "_openmeeg_model"


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
        try:
            import openmeeg as om
        except ImportError as e:
            raise ImportError(
                "EEG forward model needs the OpenMEEG Python bindings; "
                "install with `pip install openmeeg` (or `conda install -c conda-forge openmeeg`)."
            ) from e

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        create_eeg_bem_model(
            source_spacing_mm=self.params.source_spacing_mm or 5.0,
            n_sensors=self.params.num_sensors or 256,
            grid_resolution=self.params.grid_resolution_mm or 20.0,
            output_dir=str(_MODEL_DIR),
        )

        # In-process equivalent of the om_assemble / om_minverser / om_gain CLI:
        geom = om.read_geometry(
            str(_MODEL_DIR / "sphere_head.geom"),
            str(_MODEL_DIR / "sphere_head.cond"),
        )
        dipoles = om.Matrix(str(_MODEL_DIR / "dipole_locations.txt"))
        electrodes = om.Sensors(str(_MODEL_DIR / "sensor_locations.txt"), geom)

        hm = om.HeadMat(geom)
        hm.invert()  # in place; replaces om_minverser (GainEEG wants HM^{-1})
        dsm = om.DipSourceMat(geom, dipoles, "Brain")  # -DSM (dipoles live in the Brain domain)
        h2em = om.Head2EEGMat(geom, electrodes)  # -H2EM
        gain = om.GainEEG(hm, dsm, h2em)  # -EEG

        # gain is (n_sensors x 3*n_grid_points); copy out of the OpenMEEG buffer.
        return np.array(gain.array(), dtype=np.float64)


if __name__ == "__main__":
    modality = EEGModality()
    singular_values = modality.run()
