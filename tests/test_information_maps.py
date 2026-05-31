import unittest

import numpy as np

from compute_information_maps import (
    posterior_info_scalar,
    posterior_info_vector3,
    scale_matrix_for_output_power,
)
from export_svd_json import compute_bitrate
from guti.capacity import get_bitrate_from_average_output_power
from guti.noise_models import (
    capacity_forward_gain_scale,
    compute_average_output_power,
    compute_output_noise_std,
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

    def test_output_power_scaling_preserves_ratios_under_global_gain(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 10))

        A_a, noise_a, meta_a = scale_matrix_for_output_power(A, "meg_opm", n_sensors=200)
        A_b, noise_b, meta_b = scale_matrix_for_output_power(37.0 * A, "meg_opm", n_sensors=200)

        self.assertAlmostEqual(meta_a["output_snr"], meta_b["output_snr"])
        np.testing.assert_allclose(A_a / noise_a, A_b / noise_b, atol=1e-12)

    def test_output_power_scaling_matches_svd_total_input_power_path(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(6, 10))
        s = np.linalg.svd(A, compute_uv=False)

        _, matrix_noise, meta = scale_matrix_for_output_power(
            A,
            "meg_opm",
            n_sensors=A.shape[0],
        )
        svd_bitrate = get_bitrate_from_average_output_power(
            s,
            average_output_power=compute_average_output_power("meg_opm"),
            noise=compute_output_noise_std("meg_opm", n_sensors=A.shape[0]),
            n_sources=A.shape[1],
            n_outputs=A.shape[0],
        )
        matrix_bitrate = get_bitrate_from_average_output_power(
            s,
            average_output_power=compute_average_output_power("meg_opm"),
            noise=matrix_noise,
            n_sources=A.shape[1],
            n_outputs=A.shape[0],
        )

        np.testing.assert_allclose(meta["total_input_power"], meta["per_source_input_power"] * A.shape[1])
        np.testing.assert_allclose(matrix_bitrate, svd_bitrate, rtol=1e-12)

    def test_fnirs_bitrate_uses_voxel_integrated_transfer_function(self):
        s_integrated = np.array([6.0e-2, 2.0e-2, 1.0e-2])
        params = Parameters(
            num_sensors=800,
            grid_resolution_mm=6.0,
            num_brain_grid_points=4,
            matrix_size=(3, 4),
        )

        actual = compute_bitrate(
            s_integrated,
            "fnirs_analytical_cw",
            n_sensors=800,
            tier="today",
            time_resolution=1.0,
            params=params,
        )
        expected = get_bitrate_from_average_output_power(
            s_integrated,
            average_output_power=compute_average_output_power("fnirs_analytical_cw"),
            noise=compute_output_noise_std(
                "fnirs_analytical_cw",
                n_sensors=800,
                tier="today",
            ),
            n_sources=4,
            n_outputs=3,
            time_resolution=1.0,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        self.assertEqual(
            capacity_forward_gain_scale("fnirs_analytical_cw", params=params),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
