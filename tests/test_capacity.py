import unittest

import numpy as np

from guti.capacity import (
    get_bitrate,
    get_bitrate_from_average_output_power,
    get_capacity,
    get_capacity_from_average_output_power,
    total_input_power_from_average_output_power,
    water_filling_power_allocation,
)


class CapacityPowerTests(unittest.TestCase):
    def test_get_bitrate_rejects_positional_power(self):
        with self.assertRaises(TypeError):
            get_bitrate(np.array([1.0]), 0.1)

    def test_get_bitrate_spreads_total_power_over_sources(self):
        s = np.array([2.0, 0.5, 0.25])
        n_sources = 12
        per_source_power = 0.7
        noise = 0.1
        time_resolution = 0.01

        actual = get_bitrate(
            s,
            total_input_power=n_sources * per_source_power,
            n_sources=n_sources,
            noise=noise,
            time_resolution=time_resolution,
        )
        expected = (1.0 / (2.0 * time_resolution)) * np.sum(
            np.log2(1.0 + (s**2) * per_source_power / noise**2)
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_white_noise_covariance_matches_scalar_noise_for_matrix(self):
        A = np.array(
            [
                [1.0, 2.0, 0.5],
                [-0.5, 1.5, 2.0],
                [0.25, -0.75, 1.25],
            ]
        )
        total_input_power = 2.5
        n_sources = A.shape[1]
        noise = 0.2
        covariance = (noise**2) * np.eye(A.shape[0])

        scalar_bitrate = get_bitrate(
            A,
            total_input_power=total_input_power,
            n_sources=n_sources,
            noise=noise,
        )
        covariance_bitrate = get_bitrate(
            A,
            total_input_power=total_input_power,
            n_sources=n_sources,
            output_noise_covariance=covariance,
        )
        scalar_capacity = get_capacity(
            A,
            total_input_power=total_input_power,
            noise=noise,
        )
        covariance_capacity = get_capacity(
            A,
            total_input_power=total_input_power,
            output_noise_covariance=covariance,
        )

        np.testing.assert_allclose(covariance_bitrate, scalar_bitrate, rtol=1e-12)
        np.testing.assert_allclose(covariance_capacity, scalar_capacity, rtol=1e-12)

    def test_correlated_noise_bitrate_matches_noise_weighted_gramian(self):
        A = np.array(
            [
                [1.0, 0.5],
                [0.2, 1.5],
                [1.2, -0.3],
            ]
        )
        covariance = np.array(
            [
                [0.5, 0.1, 0.05],
                [0.1, 1.2, 0.2],
                [0.05, 0.2, 0.8],
            ]
        )
        total_input_power = 1.6
        n_sources = A.shape[1]
        time_resolution = 0.25

        actual = get_bitrate(
            A,
            total_input_power=total_input_power,
            n_sources=n_sources,
            output_noise_covariance=covariance,
            time_resolution=time_resolution,
        )
        gains_squared = np.linalg.eigvalsh(A.T @ np.linalg.solve(covariance, A))
        gains_squared = np.clip(gains_squared, 0.0, None)
        expected = np.sum(
            np.log2(1.0 + gains_squared * total_input_power / n_sources)
        ) / (2.0 * time_resolution)

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_correlated_noise_capacity_uses_water_filling_after_whitening(self):
        A = np.eye(2)
        covariance = np.diag([0.25, 4.0])
        total_input_power = 5.0

        actual = get_capacity(
            A,
            total_input_power=total_input_power,
            output_noise_covariance=covariance,
        )

        gains = np.array([2.0, 0.5])
        allocation = water_filling_power_allocation(
            gains,
            total_input_power=total_input_power,
            noise=1.0,
        )
        expected = 0.5 * np.sum(np.log2(1.0 + (gains**2) * allocation))

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(allocation, np.array([4.375, 0.625]), rtol=1e-12)

    def test_output_noise_covariance_requires_channel_matrix(self):
        with self.assertRaises(ValueError):
            get_bitrate(
                np.array([2.0, 1.0]),
                total_input_power=1.0,
                n_sources=2,
                output_noise_covariance=np.eye(2),
            )

    def test_scalar_noise_and_covariance_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            get_capacity(
                np.eye(2),
                total_input_power=1.0,
                noise=0.1,
                output_noise_covariance=np.eye(2),
            )

    def test_estimates_total_input_power_from_average_output_power(self):
        A = np.array(
            [
                [1.0, 2.0, 0.5],
                [-0.5, 1.5, 2.0],
            ]
        )
        n_outputs, n_sources = A.shape
        per_source_power = 0.7
        expected_total_input_power = n_sources * per_source_power
        average_output_power = per_source_power * np.sum(A**2) / n_outputs
        singular_values = np.linalg.svd(A, compute_uv=False)

        actual = total_input_power_from_average_output_power(
            singular_values,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
        )

        np.testing.assert_allclose(actual, expected_total_input_power, rtol=1e-12)

    def test_get_bitrate_from_average_output_power_matches_explicit_power(self):
        A = np.array(
            [
                [1.0, 2.0, 0.5],
                [-0.5, 1.5, 2.0],
            ]
        )
        n_outputs, n_sources = A.shape
        per_source_power = 0.7
        noise = 0.2
        time_resolution = 0.01
        average_output_power = per_source_power * np.sum(A**2) / n_outputs
        singular_values = np.linalg.svd(A, compute_uv=False)

        actual = get_bitrate_from_average_output_power(
            singular_values,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=noise,
            time_resolution=time_resolution,
        )
        expected = get_bitrate(
            singular_values,
            total_input_power=n_sources * per_source_power,
            n_sources=n_sources,
            noise=noise,
            time_resolution=time_resolution,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_average_output_power_workflow_handles_consistent_global_scaling(self):
        A = np.array(
            [
                [0.25, 1.0, -0.5, 2.0],
                [1.5, -0.25, 0.75, 0.5],
                [0.0, 0.5, 1.0, -1.0],
            ]
        )
        n_outputs, n_sources = A.shape
        per_source_power = 1.3
        raw_noise = 0.2
        scale = np.sqrt(n_sources * n_outputs)
        average_output_power = per_source_power * np.sum(A**2) / n_outputs
        singular_values = np.linalg.svd(A, compute_uv=False)

        actual = get_bitrate_from_average_output_power(
            singular_values / scale,
            average_output_power=average_output_power / scale**2,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=raw_noise / scale,
        )
        expected = get_bitrate(
            singular_values,
            total_input_power=n_sources * per_source_power,
            n_sources=n_sources,
            noise=raw_noise,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_average_output_power_workflow_accepts_correlated_noise_covariance(self):
        A = np.array(
            [
                [0.25, 1.0, -0.5],
                [1.5, -0.25, 0.75],
                [0.0, 0.5, 1.0],
            ]
        )
        covariance = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.2, 1.4, -0.15],
                [0.1, -0.15, 0.6],
            ]
        )
        n_outputs, n_sources = A.shape
        per_source_power = 0.8
        average_output_power = per_source_power * np.sum(A**2) / n_outputs

        actual = get_bitrate_from_average_output_power(
            A,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            output_noise_covariance=covariance,
        )
        expected = get_bitrate(
            A,
            total_input_power=n_sources * per_source_power,
            n_sources=n_sources,
            output_noise_covariance=covariance,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

        capacity_actual = get_capacity_from_average_output_power(
            A,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            output_noise_covariance=covariance,
        )
        capacity_expected = get_capacity(
            A,
            total_input_power=n_sources * per_source_power,
            output_noise_covariance=covariance,
        )

        np.testing.assert_allclose(capacity_actual, capacity_expected, rtol=1e-12)

    def test_water_filling_uses_total_input_power_constraint(self):
        s = np.array([4.0, 2.0, 0.5])
        total_input_power = 3.0
        noise = 0.5

        allocation = water_filling_power_allocation(
            s,
            total_input_power=total_input_power,
            noise=noise,
        )

        np.testing.assert_allclose(np.sum(allocation), total_input_power, rtol=1e-12)
        self.assertGreater(allocation[0], allocation[1])
        self.assertGreaterEqual(allocation[1], allocation[2])

    def test_capacity_from_average_output_power_matches_explicit_power(self):
        A = np.array(
            [
                [1.0, 0.0, 0.5],
                [0.2, 1.5, 0.0],
            ]
        )
        n_outputs, n_sources = A.shape
        per_source_power = 0.4
        noise = 0.3
        s = np.linalg.svd(A, compute_uv=False)
        average_output_power = per_source_power * np.sum(A**2) / n_outputs

        actual = get_capacity_from_average_output_power(
            s,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=noise,
        )
        expected = get_capacity(
            s,
            total_input_power=n_sources * per_source_power,
            noise=noise,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
