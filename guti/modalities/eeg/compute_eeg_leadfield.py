"""OpenMEEG EEG leadfield assembly through the Python API."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _openmeeg():
    try:
        import openmeeg as om
    except ImportError as exc:
        raise ImportError(
            "EEG OpenMEEG leadfield assembly needs the `openmeeg` Python "
            "package; install with `pip install openmeeg` or conda-forge."
        ) from exc
    return om


def compute_eeg_leadfield(
    geom_file: str | Path,
    cond_file: str | Path,
    dipole_file: str | Path,
    sensor_file: str | Path,
    domain_name: str = "Brain",
) -> np.ndarray:
    """Compute the EEG leadfield matrix from BEM input files.

    This is the single active OpenMEEG assembly path for EEG in this repo. It
    performs the same operations as the historical CLI pipeline, but keeps the
    workflow in-process:
        HeadMat → invert → DipSourceMat / Head2EEGMat → GainEEG

    Returns the gain matrix as a ``(n_sensors, n_dipoles)`` numpy array.
    """
    om = _openmeeg()
    geom = om.read_geometry(str(geom_file), str(cond_file))

    sensors = om.Sensors()
    sensors.load(str(sensor_file))

    # OpenMEEG's Matrix ctor requires Fortran-order arrays.
    dipoles = np.atleast_2d(np.loadtxt(str(dipole_file), dtype=np.float64))
    dipoles = np.asfortranarray(dipoles)
    dip_mat = om.Matrix(dipoles)

    hm = om.HeadMat(geom)
    hm.invert()
    dsm = om.DipSourceMat(geom, dip_mat, domain_name)
    h2em = om.Head2EEGMat(geom, sensors)
    gain = om.GainEEG(hm, dsm, h2em)

    return np.asarray(gain.array(), dtype=np.float64)


def compute_eeg_leadfield_from_bem_dir(
    bem_dir: str | Path,
    *,
    domain_name: str = "Brain",
) -> np.ndarray:
    """Compute an EEG leadfield from a directory written by create_eeg_bem_model."""
    root = Path(bem_dir)
    return compute_eeg_leadfield(
        geom_file=root / "sphere_head.geom",
        cond_file=root / "sphere_head.cond",
        dipole_file=root / "dipole_locations.txt",
        sensor_file=root / "sensor_locations.txt",
        domain_name=domain_name,
    )
