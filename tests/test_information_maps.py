import unittest

import numpy as np
import torch

from compute_information_maps import (
    compute_meg_forward_matrix,
    empirical_noise_for_matrix,
    normalize_to_first_in_brain_value,
    posterior_info_scalar,
    posterior_info_vector3,
)
from guti.core import get_sensor_positions
from guti.modalities.fnirs_analytical.utils import (
    get_valid_source_detector_pairs as get_cw_pairs,
)
from guti.modalities.td_fnirs.utils import (
    get_valid_source_detector_pairs as get_td_pairs,
)
from recompute_meg_variants import sarvas_formula


class InformationMapMathTests(unittest.TestCase):
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

    def test_empirical_noise_preserves_ratios_under_global_gain(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 10))

        noise_a, meta_a = empirical_noise_for_matrix(A, "meg_opm", n_sensors=200)
        noise_b, meta_b = empirical_noise_for_matrix(37.0 * A, "meg_opm", n_sensors=200)

        self.assertAlmostEqual(meta_a["empirical_snr"], meta_b["empirical_snr"])
        np.testing.assert_allclose(A / noise_a, (37.0 * A) / noise_b, atol=1e-12)

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
