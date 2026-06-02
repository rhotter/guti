"""Python port of compute_eeg_leadfield.sh using the OpenMEEG Python API.

Replaces the om_assemble / om_minverser / om_gain CLI pipeline with direct
bindings so the sweep runs without OpenMEEG's CLI binaries installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmeeg as om


def compute_eeg_leadfield(
    geom_file: str | Path,
    cond_file: str | Path,
    dipole_file: str | Path,
    sensor_file: str | Path,
    domain_name: str = "Brain",
) -> np.ndarray:
    """Compute the EEG leadfield matrix from BEM input files.

    Mirrors, step for step, the OpenMEEG CLI pipeline:
        om_assemble -HM        → HeadMat(geom)
        om_minverser           → SymMatrix.invert()
        om_assemble -DSM       → DipSourceMat(geom, dipoles, domain)
        om_assemble -H2EM      → Head2EEGMat(geom, sensors)
        om_gain     -EEG       → GainEEG(hm_inv, dsm, h2em)

    Returns the gain matrix as a ``(n_sensors, n_dipoles)`` numpy array.
    """
    geom = om.read_geometry(str(geom_file), str(cond_file))

    sensors = om.Sensors()
    sensors.load(str(sensor_file))

    # OpenMEEG's Matrix ctor requires Fortran-order arrays.
    dipoles = np.asfortranarray(np.loadtxt(str(dipole_file)))
    dip_mat = om.Matrix(dipoles)

    hm = om.HeadMat(geom)
    hm.invert()  # in place → HM becomes HM^{-1}
    dsm = om.DipSourceMat(geom, dip_mat, domain_name)
    h2em = om.Head2EEGMat(geom, sensors)
    gain = om.GainEEG(hm, dsm, h2em)

    return np.asarray(gain.array())
