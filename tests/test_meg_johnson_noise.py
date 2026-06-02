import numpy as np

from guti.core import get_sensor_positions
from guti.modalities.meg.johnson_noise import (
    compute_meg_johnson_noise_covariance,
    compute_meg_johnson_noise_spectral_density,
    match_covariance_diagonal,
    sensor_channel_orientations,
    voxelized_layered_hemisphere,
    voxelized_layered_hemisphere_grid,
)
from guti.modalities.meg.meg import OPM_OFFSET_MM
from guti.noise_models import covariance_to_correlation_matrix


def test_meg_johnson_radial_covariance_is_symmetric_psd():
    sensors = get_sensor_positions(4, offset=OPM_OFFSET_MM)

    spectral_density, metadata = compute_meg_johnson_noise_spectral_density(
        sensors,
        sensor_components="radial",
        voxel_resolution_mm=35.0,
        solver="finite_volume",
        return_metadata=True,
    )

    assert metadata.n_channels == 4
    assert metadata.n_voxels > 0
    assert metadata.solver == "finite_volume"
    assert spectral_density.shape == (4, 4)
    np.testing.assert_allclose(spectral_density, spectral_density.T, rtol=1e-12)
    assert np.all(np.diag(spectral_density) > 0.0)
    assert np.linalg.eigvalsh(spectral_density)[0] >= -1e-28

    correlation = covariance_to_correlation_matrix(spectral_density)
    np.testing.assert_allclose(np.diag(correlation), np.ones(4), rtol=1e-12)


def test_meg_johnson_xyz_channels_match_forward_row_grouping():
    sensors = get_sensor_positions(2, offset=OPM_OFFSET_MM)
    channel_sensor_indices, orientations = sensor_channel_orientations(
        sensors,
        sensor_components="xyz",
    )

    np.testing.assert_array_equal(channel_sensor_indices, np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_allclose(orientations[:3], np.eye(3), rtol=1e-12)

    spectral_density = compute_meg_johnson_noise_spectral_density(
        sensors,
        sensor_components="xyz",
        voxel_resolution_mm=40.0,
        solver="finite_volume",
    )
    assert spectral_density.shape == (6, 6)
    assert np.all(np.diag(spectral_density) > 0.0)


def test_finite_volume_projection_does_not_increase_dissipation():
    sensors = get_sensor_positions(3, offset=OPM_OFFSET_MM)
    finite_volume = compute_meg_johnson_noise_spectral_density(
        sensors,
        sensor_components="radial",
        voxel_resolution_mm=35.0,
        solver="finite_volume",
    )
    vector_potential = compute_meg_johnson_noise_spectral_density(
        sensors,
        sensor_components="radial",
        voxel_resolution_mm=35.0,
        solver="vector_potential",
    )

    assert np.all(np.diag(finite_volume) <= np.diag(vector_potential) * (1.0 + 1e-12))


def test_match_covariance_diagonal_preserves_correlation_structure():
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])

    matched = match_covariance_diagonal(covariance, np.array([0.2, 0.5]))

    np.testing.assert_allclose(np.diag(matched), np.array([0.04, 0.25]))
    original_corr = covariance_to_correlation_matrix(covariance)
    matched_corr = covariance_to_correlation_matrix(matched)
    np.testing.assert_allclose(matched_corr, original_corr, rtol=1e-12)


def test_meg_johnson_covariance_can_match_detector_diagonal():
    sensors = get_sensor_positions(5, offset=OPM_OFFSET_MM)

    covariance, metadata, scale_info = compute_meg_johnson_noise_covariance(
        sensors,
        noise_std=2.0e-15,
        sensor_components="radial",
        voxel_resolution_mm=35.0,
        solver="finite_volume",
        return_metadata=True,
    )

    assert metadata.n_channels == 5
    assert scale_info["scale_mode"] == "matched_detector_diagonal"
    np.testing.assert_allclose(
        np.diag(covariance),
        np.full(5, (2.0e-15) ** 2) + scale_info["regularization_jitter_T2"],
        rtol=1e-12,
    )
    assert np.linalg.eigvalsh(covariance)[0] > 0.0


def test_voxelized_layered_hemisphere_assigns_positive_conductivity():
    points_m, sigma, voxel_volume_m3 = voxelized_layered_hemisphere(
        voxel_resolution_mm=40.0
    )

    assert points_m.shape[1] == 3
    assert points_m.shape[0] == sigma.shape[0]
    assert points_m.shape[0] > 0
    assert voxel_volume_m3 > 0.0
    assert np.all(sigma > 0.0)

    grid = voxelized_layered_hemisphere_grid(voxel_resolution_mm=40.0)
    np.testing.assert_allclose(grid.points_m, points_m)
    np.testing.assert_allclose(grid.sigma_s_per_m, sigma)
    assert grid.indices_ijk.shape == (points_m.shape[0], 3)
