import unittest

import numpy as np
import torch

from compute_information_maps import (
    DEFAULT_SCALING,
    capacity_attribution_scalar,
    capacity_attribution_vector3,
    compute_meg_forward_matrix,
    empirical_noise_for_matrix,
    mode_bits_from_singular_values,
    normalize_to_first_in_brain_value,
    posterior_info_scalar,
    posterior_info_vector3,
)
from export_svd_json import compute_bitrate
from guti.capacity import get_bitrate
from guti.core import get_sensor_positions
from guti.noise_models import (
    capacity_forward_gain_scale,
    compute_detector_noise_std,
    compute_total_input_power,
)
from guti.parameters import Parameters
from guti.modalities.cw_fnirs.utils import (
    get_valid_source_detector_pairs as get_cw_pairs,
)
from guti.modalities.td_fnirs.utils import (
    get_valid_source_detector_pairs as get_td_pairs,
)
from recompute_meg_variants import sarvas_formula


class InformationMapMathTests(unittest.TestCase):
    def test_default_depth_map_scaling_is_physical(self):
        self.assertEqual(DEFAULT_SCALING, "physical")

    def test_scalar_posterior_matches_direct_covariance(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(8, 12)) * 0.1
        noise = 0.3

        posterior = np.linalg.inv(np.eye(A.shape[1]) + (A.T @ A) / noise**2)
        posterior_var, info = posterior_info_scalar(A, noise, chunk_cols=5)

        np.testing.assert_allclose(posterior_var, np.diag(posterior), atol=1e-12)
        np.testing.assert_allclose(info, -0.5 * np.log2(np.diag(posterior)), atol=1e-12)

    def test_scalar_posterior_matches_direct_covariance_when_measurements_exceed_sources(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(12, 5)) * 0.1
        noise = 0.3

        posterior = np.linalg.inv(np.eye(A.shape[1]) + (A.T @ A) / noise**2)
        posterior_var, info = posterior_info_scalar(A, noise, chunk_cols=3)

        np.testing.assert_allclose(posterior_var, np.diag(posterior), atol=1e-12)
        np.testing.assert_allclose(info, -0.5 * np.log2(np.diag(posterior)), atol=1e-12)

    def test_vector3_posterior_matches_direct_covariance_blocks(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(8, 12)) * 0.1
        noise = 0.3

        posterior = np.linalg.inv(np.eye(A.shape[1]) + (A.T @ A) / noise**2)
        posterior_det, info = posterior_info_vector3(A, noise, n_voxels=4)

        expected_det = []
        for i in range(4):
            block = posterior[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
            expected_det.append(np.linalg.det(block))
        expected_det = np.array(expected_det)

        np.testing.assert_allclose(posterior_det, expected_det, atol=1e-12)
        np.testing.assert_allclose(info, -0.5 * np.log2(expected_det), atol=1e-12)

    def test_scalar_capacity_attribution_sums_to_svd_capacity(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(9, 5)) * 0.1
        noise = 0.2

        attribution, singular_values, mode_bits = capacity_attribution_scalar(A, noise)
        expected_bits = mode_bits_from_singular_values(np.linalg.svd(A, compute_uv=False), noise)

        np.testing.assert_allclose(np.sum(attribution), np.sum(expected_bits), atol=1e-12)
        np.testing.assert_allclose(mode_bits, expected_bits, atol=1e-12)

    def test_vector3_capacity_attribution_sums_to_svd_capacity(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(7, 12)) * 0.1
        noise = 0.2

        attribution, singular_values, mode_bits = capacity_attribution_vector3(
            A,
            noise,
            n_voxels=4,
        )
        expected_bits = mode_bits_from_singular_values(np.linalg.svd(A, compute_uv=False), noise)

        np.testing.assert_allclose(np.sum(attribution), np.sum(expected_bits), atol=1e-12)
        np.testing.assert_allclose(mode_bits, expected_bits, atol=1e-12)

    def test_capacity_attribution_keeps_tiny_si_scale_modes(self):
        rng = np.random.default_rng(6)
        A = rng.normal(size=(5, 9)) * 1e-13
        noise = 2e-14

        attribution, _, mode_bits = capacity_attribution_vector3(
            A,
            noise,
            n_voxels=3,
        )
        expected_bits = mode_bits_from_singular_values(np.linalg.svd(A, compute_uv=False), noise)

        np.testing.assert_allclose(np.sum(attribution), np.sum(expected_bits), rtol=1e-10)
        np.testing.assert_allclose(mode_bits, expected_bits, rtol=1e-10)

    def test_empirical_noise_preserves_ratios_under_global_gain(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 10))

        noise_a, meta_a = empirical_noise_for_matrix(A, "meg_opm", n_sensors=200)
        noise_b, meta_b = empirical_noise_for_matrix(37.0 * A, "meg_opm", n_sensors=200)

        self.assertAlmostEqual(meta_a["empirical_snr"], meta_b["empirical_snr"])
        np.testing.assert_allclose(A / noise_a, (37.0 * A) / noise_b, atol=1e-12)

    def test_physical_bitrate_preserves_forward_gain_unlike_empirical_mode(self):
        s = np.array([0.12, 0.03, 0.01])
        physical = compute_bitrate(
            s,
            "meg_opm",
            n_sensors=1000,
            tier="today",
            time_resolution=0.01,
            noise_mode="physical_detector_floor",
        )
        physical_scaled = compute_bitrate(
            10.0 * s,
            "meg_opm",
            n_sensors=1000,
            tier="today",
            time_resolution=0.01,
            noise_mode="physical_detector_floor",
        )
        empirical = compute_bitrate(
            s,
            "meg_opm",
            n_sensors=1000,
            tier="today",
            time_resolution=0.01,
            noise_mode="empirical_observed_snr",
        )
        empirical_scaled = compute_bitrate(
            10.0 * s,
            "meg_opm",
            n_sensors=1000,
            tier="today",
            time_resolution=0.01,
            noise_mode="empirical_observed_snr",
        )

        self.assertGreater(physical_scaled, physical)
        self.assertAlmostEqual(empirical_scaled, empirical)

    def test_fnirs_physical_bitrate_uses_voxel_integrated_transfer_function(self):
        s_integrated = np.array([6.0e-2, 2.0e-2, 1.0e-2])
        params = Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
            num_brain_grid_points=3,
            matrix_size=(3, 3),
        )
        noise = compute_detector_noise_std(
            "cw_fnirs",
            n_sensors=800,
            tier="today",
        )
        total_input_power = compute_total_input_power(
            "cw_fnirs",
            n_sources=3,
        )

        actual = compute_bitrate(
            s_integrated,
            "cw_fnirs",
            n_sensors=800,
            tier="today",
            time_resolution=1.0,
            params=params,
            noise_mode="physical_detector_floor",
        )
        expected = get_bitrate(
            s_integrated,
            n_sources=3,
            total_input_power=total_input_power,
            noise=noise,
            time_resolution=1.0,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        self.assertEqual(
            capacity_forward_gain_scale("cw_fnirs", params=params),
            1.0,
        )

    def test_normalize_to_first_in_brain_value_uses_first_positive_depth(self):
        x = np.array([-1.0, 1.0, 3.0, 5.0])
        y = np.array([10.0, 2.0, 1.0, 0.5])

        normalized, reference = normalize_to_first_in_brain_value(x, y)

        self.assertEqual(reference, 2.0)
        np.testing.assert_allclose(normalized, [5.0, 1.0, 0.5, 0.25])

    def test_vectorized_meg_forward_matches_sarvas_blocks(self):
        n_sensors = 2
        offset_mm = 5.0
        A, sources = compute_meg_forward_matrix(n_sensors, 80.0, offset_mm)
        sensors = get_sensor_positions(n_sensors, offset=offset_mm)

        for sensor_idx, sensor in enumerate(sensors):
            for source_idx, source in enumerate(sources):
                rows = slice(3 * sensor_idx, 3 * sensor_idx + 3)
                cols = slice(3 * source_idx, 3 * source_idx + 3)
                np.testing.assert_allclose(
                    A[rows, cols],
                    sarvas_formula(sensor, source),
                    atol=1e-24,
                )

    def test_fnirs_helpers_return_unique_source_detector_pairs(self):
        sensors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        cw_sources, cw_detectors = get_cw_pairs(sensors, max_dist=1.5)
        self.assertEqual(cw_sources.shape[0], 2)
        self.assertEqual(cw_detectors.shape[0], 2)

        td_sources, _, td_detectors, _ = get_td_pairs(
            sensors,
            max_dist=1.5,
            head_center=torch.tensor([0.0, 0.0, -1.0]),
        )
        self.assertEqual(td_sources.shape[0], 2)
        self.assertEqual(td_detectors.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
