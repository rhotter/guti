"""MEG imaging modality (Sarvas spherical forward model).

Wraps the canonical in-process lead-field in :mod:`guti.modalities.meg.meg`.
The sensor standoff selects the variant: ``sensor_offset_mm`` near 7 mm is an
OPM array, near 25 mm is a SQUID array, and ``name`` reports the matching
canonical id (``meg_opm`` / ``meg_squid``) so results and noise models line up.
"""

from typing import Optional

import numpy as np

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import get_sensor_positions, get_grid_positions
from guti.modalities.meg.meg import (
    compute_forward_matrix,
    OPM_OFFSET_MM,
    SQUID_OFFSET_MM,
)

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


if __name__ == "__main__":
    modality = MEGModality()
    singular_values = modality.run()
