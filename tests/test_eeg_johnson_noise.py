import numpy as np

from guti.core import get_sensor_positions
from guti.modalities.eeg.scalp_resistance import surface_impedance_kernel_matrix
from guti.noise_models import (
    compute_johnson_noise_covariance,
    covariance_to_correlation_matrix,
)


def test_spherical_eeg_surface_impedance_kernel_is_symmetric_psd():
    sensors = get_sensor_positions(12)
    impedance = surface_impedance_kernel_matrix(
        sensors,
        electrode_area_cm2=1.0,
        lmax=96,
    )

    np.testing.assert_allclose(impedance, impedance.T, rtol=1e-12, atol=1e-12)
    assert impedance.shape == (12, 12)
    assert np.all(np.diag(impedance) > 0.0)

    eigenvalues = np.linalg.eigvalsh(impedance)
    assert eigenvalues[0] > -1e-9 * eigenvalues[-1]


def test_spherical_eeg_johnson_covariance_can_match_detector_diagonal():
    sensors = get_sensor_positions(10)
    impedance = surface_impedance_kernel_matrix(
        sensors,
        electrode_area_cm2=1.0,
        lmax=96,
    )
    detector_noise = 3e-8

    covariance = compute_johnson_noise_covariance(
        impedance,
        bandwidth_hz=100.0,
        noise_std=detector_noise,
    )
    correlation = covariance_to_correlation_matrix(covariance)

    np.testing.assert_allclose(
        np.diag(covariance),
        detector_noise**2,
        rtol=1e-12,
    )
    np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=1e-12)
    assert np.count_nonzero(np.abs(correlation - np.eye(10)) > 1e-6) > 0
