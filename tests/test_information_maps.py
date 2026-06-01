import unittest

import numpy as np

from compute_information_maps import (
    empirical_noise_for_matrix,
    posterior_info_scalar,
    posterior_info_vector3,
)
from export_svd_json import compute_bitrate
from guti.core import get_bitrate
from guti.noise_models import (
    capacity_forward_gain_scale,
    compute_noise_empirical,
)
from guti.parameters import Parameters


class InformationMapMathTests(unittest.TestCase):
    def test_scalar_posterior_matches_direct_covariance(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(8, 12)) * 0.1
        noise = 0.3

        posterior = np.linalg.inv(np.eye(A.shape[1]) + (A.T @ A) / noise**2)
        posterior_var, info = posterior_info_scalar(A, noise, chunk_cols=5)

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

    def test_fnirs_bitrate_uses_voxel_integrated_transfer_function(self):
        s_integrated = np.array([6.0e-2, 2.0e-2, 1.0e-2])
        params = Parameters(num_sensors=800, grid_resolution_mm=6.0)
        noise = compute_noise_empirical(
            s_integrated,
            "cw_fnirs",
            n_sensors=800,
            tier="today",
        )

        actual = compute_bitrate(
            s_integrated,
            "cw_fnirs",
            n_sensors=800,
            tier="today",
            time_resolution=1.0,
            params=params,
        )
        expected = get_bitrate(
            s_integrated,
            noise,
            time_resolution=1.0,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        self.assertEqual(
            capacity_forward_gain_scale("cw_fnirs", params=params),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
