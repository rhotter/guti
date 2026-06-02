import unittest

import numpy as np

from guti.capacity import (
    get_bitrate,
    get_bitrate_temporal_filter,
    get_bitrate_from_average_output_power,
    get_capacity,
    get_capacity_temporal_filter,
    get_capacity_from_average_output_power,
    power_law_frequency_spectrum,
    sensor_noise_normalized_singular_values,
    resolve_total_input_power,
    total_input_power_from_average_output_power,
    total_input_power_from_input_amplitude,
    water_filling_power_allocation,
)
from guti.noise_models import (
    K_B,
    compute_output_noise_covariance,
    compute_johnson_noise_covariance,
    compute_sensor_noise_covariance,
    estimate_effective_correlation_length_mm,
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

    def test_distance_covariance_uses_sensor_variances_and_kernel(self):
        positions = np.array(
            [
                [172.0, 80.0, 0.0],
                [80.0, 172.0, 0.0],
                [80.0, 80.0, 92.0],
            ]
        )
        noise_std = np.array([0.1, 0.2, 0.4])

        covariance = compute_sensor_noise_covariance(
            positions,
            noise_std,
            correlation_length_mm=5.0,
            kernel="exponential",
        )

        np.testing.assert_allclose(np.diag(covariance), noise_std**2, rtol=1e-12)
        self.assertGreater(covariance[0, 1], 0.0)
        self.assertLess(covariance[0, 1], noise_std[0] * noise_std[1])
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)

    def test_output_covariance_expands_rows_grouped_by_sensor(self):
        positions = np.array(
            [
                [172.0, 80.0, 0.0],
                [80.0, 172.0, 0.0],
            ]
        )
        covariance = compute_output_noise_covariance(
            positions,
            0.5,
            correlation_length_mm=5.0,
            outputs_per_sensor=3,
        )

        self.assertEqual(covariance.shape, (6, 6))
        np.testing.assert_allclose(np.diag(covariance), np.full(6, 0.25))
        np.testing.assert_allclose(covariance[0:3, 0:3], 0.25 * np.eye(3))
        np.testing.assert_allclose(covariance[0, 4], 0.0, atol=1e-15)
        self.assertGreater(covariance[0, 3], 0.0)

    def test_johnson_covariance_scales_transfer_resistance(self):
        resistance = np.array([[5_000.0, 700.0], [700.0, 4_000.0]])
        temperature_k = 310.0
        bandwidth_hz = 100.0

        covariance = compute_johnson_noise_covariance(
            resistance,
            bandwidth_hz=bandwidth_hz,
            temperature_k=temperature_k,
        )

        expected = 4.0 * K_B * temperature_k * bandwidth_hz * resistance
        np.testing.assert_allclose(covariance, expected, rtol=1e-12)

    def test_johnson_covariance_applies_montage_and_series_resistance(self):
        terminal_resistance = np.array(
            [
                [3.0, 0.4, 0.2],
                [0.4, 4.0, 0.1],
                [0.2, 0.1, 5.0],
            ]
        )
        montage = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        contact_resistance = np.array([10.0, 20.0, 30.0])
        bandwidth_hz = 25.0
        temperature_k = 300.0

        covariance = compute_johnson_noise_covariance(
            terminal_resistance,
            bandwidth_hz=bandwidth_hz,
            temperature_k=temperature_k,
            series_resistance_ohm=contact_resistance,
            montage_matrix=montage,
        )

        channel_resistance = montage @ (
            terminal_resistance + np.diag(contact_resistance)
        ) @ montage.T
        expected = 4.0 * K_B * temperature_k * bandwidth_hz * channel_resistance
        np.testing.assert_allclose(covariance, expected, rtol=1e-12)

    def test_johnson_covariance_can_match_existing_noise_std(self):
        resistance = np.array([[9.0, 3.0], [3.0, 4.0]])
        noise_std = np.array([0.1, 0.2])

        covariance = compute_johnson_noise_covariance(
            resistance,
            bandwidth_hz=10.0,
            noise_std=noise_std,
        )

        expected_corr = 3.0 / np.sqrt(9.0 * 4.0)
        np.testing.assert_allclose(np.diag(covariance), noise_std**2, rtol=1e-12)
        np.testing.assert_allclose(
            covariance[0, 1],
            expected_corr * noise_std[0] * noise_std[1],
            rtol=1e-12,
        )

    def test_johnson_covariance_integrates_frequency_dependent_impedance(self):
        frequencies_hz = np.array([0.0, 10.0, 20.0])
        impedance = np.array(
            [
                [[2.0, 0.2], [0.2, 3.0]],
                [[4.0, 0.4], [0.4, 5.0]],
                [[6.0, 0.6], [0.6, 7.0]],
            ]
        )
        temperature_k = 300.0

        covariance = compute_johnson_noise_covariance(
            impedance,
            frequencies_hz=frequencies_hz,
            temperature_k=temperature_k,
        )

        _trapezoid = getattr(np, "trapezoid", None) or np.trapz
        expected = 4.0 * K_B * temperature_k * _trapezoid(
            impedance,
            frequencies_hz,
            axis=0,
        )
        np.testing.assert_allclose(covariance, expected, rtol=1e-12)

    def test_estimates_effective_correlation_length(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
            ]
        )
        length_mm = 12.0
        distances = np.abs(positions[:, 0, None] - positions[None, :, 0])
        covariance = np.exp(-distances / length_mm)

        fit = estimate_effective_correlation_length_mm(
            positions,
            covariance,
            kernel="exponential",
            distance_metric="euclidean",
        )

        np.testing.assert_allclose(fit.length_mm, length_mm, rtol=1e-12)
        self.assertEqual(fit.n_pairs, 6)

    def test_sensor_whitening_matches_full_output_covariance(self):
        A = np.array(
            [
                [1.0, 0.2],
                [0.1, 1.2],
                [0.4, -0.3],
                [0.7, 0.5],
            ]
        )
        sensor_covariance = np.array([[0.25, 0.05], [0.05, 0.64]])
        output_covariance = np.kron(sensor_covariance, np.eye(2))

        actual = sensor_noise_normalized_singular_values(
            A,
            sensor_noise_covariance=sensor_covariance,
            outputs_per_sensor=2,
        )
        expected_bitrate = get_bitrate(
            A,
            n_sources=A.shape[1],
            total_input_power=1.0,
            output_noise_covariance=output_covariance,
        )
        actual_bitrate = get_bitrate(
            actual,
            n_sources=A.shape[1],
            total_input_power=1.0,
            noise=1.0,
        )

        np.testing.assert_allclose(actual_bitrate, expected_bitrate, rtol=1e-12)

    def test_water_filling_ignores_numerical_null_modes(self):
        gains = np.array([1.0e10, 9.0e9, 1.0e-39])
        total_input_power = 3.0e-16

        allocation = water_filling_power_allocation(
            gains,
            total_input_power=total_input_power,
            noise=1.0,
        )

        floors = 1.0 / (gains[:2] ** 2)
        water_level = (total_input_power + np.sum(floors)) / 2.0
        expected = np.array([water_level - floors[0], water_level - floors[1], 0.0])

        np.testing.assert_allclose(allocation, expected, rtol=1e-12)
        np.testing.assert_allclose(np.sum(allocation), total_input_power, rtol=1e-12)

    def test_water_filling_handles_underflowed_noise_floor(self):
        gains = np.array([1.0, 1.0e-200])
        total_input_power = 1.0e-6

        allocation = water_filling_power_allocation(
            gains,
            total_input_power=total_input_power,
            noise=1.0,
        )

        np.testing.assert_allclose(
            allocation,
            np.array([total_input_power, 0.0]),
            rtol=1e-12,
            atol=0.0,
        )

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

    def test_estimates_total_input_power_from_input_amplitude(self):
        actual = total_input_power_from_input_amplitude(0.25, n_sources=8)
        np.testing.assert_allclose(actual, 8 * 0.25**2, rtol=1e-12)

    def test_resolve_total_input_power_accepts_exactly_one_convention(self):
        with self.assertRaises(ValueError):
            resolve_total_input_power(np.array([1.0]), n_sources=1)
        with self.assertRaises(ValueError):
            resolve_total_input_power(
                np.array([1.0]),
                n_sources=1,
                total_input_power=1.0,
                input_amplitude=0.5,
            )

        actual = resolve_total_input_power(
            np.array([2.0]),
            n_sources=3,
            input_power_per_source=0.75,
        )
        np.testing.assert_allclose(actual, 2.25, rtol=1e-12)

    def test_input_amplitude_workflow_matches_effective_noise_workflow(self):
        s = np.array([2.0, 0.5, 0.25])
        input_amplitude = 0.01
        output_noise = 2e-4
        n_sources = 12
        time_resolution = 0.02

        actual = get_bitrate(
            s,
            n_sources=n_sources,
            input_amplitude=input_amplitude,
            noise=output_noise,
            time_resolution=time_resolution,
        )
        effective_noise = output_noise / input_amplitude
        expected = np.sum(np.log2(1.0 + (s / effective_noise) ** 2)) / (
            2.0 * time_resolution
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_direct_average_output_power_argument_matches_wrapper(self):
        A = np.array(
            [
                [1.0, 2.0, 0.5],
                [-0.5, 1.5, 2.0],
            ]
        )
        n_outputs, n_sources = A.shape
        average_output_power = 1.1
        noise = 0.2

        actual = get_bitrate(
            A,
            n_sources=n_sources,
            n_outputs=n_outputs,
            average_output_power=average_output_power,
            noise=noise,
        )
        expected = get_bitrate_from_average_output_power(
            A,
            average_output_power=average_output_power,
            n_sources=n_sources,
            n_outputs=n_outputs,
            noise=noise,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

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

    def test_temporal_filter_input_power_matches_effective_noise_workflow(self):
        s = np.array([2.0, 0.5])
        freqs = np.array([0.0, 0.25, 0.5])
        H = np.array([1.0, 0.5, 0.25])
        input_amplitude = 0.03
        output_noise = 0.2

        actual = get_bitrate_temporal_filter(
            s,
            freqs,
            H,
            n_sources=4,
            input_amplitude=input_amplitude,
            noise=output_noise,
        )
        gains = np.outer(s, H).ravel() / (output_noise / input_amplitude)
        expected = 0.25 * np.sum(np.log2(1.0 + gains**2))

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_temporal_capacity_dominates_bitrate_on_same_budget(self):
        # Water-filling (get_capacity_temporal_filter) must give >= the
        # equal-power bitrate (get_bitrate_temporal_filter) on an identical
        # total-input-power budget, and both must be ~invariant to df.
        rng = np.random.default_rng(0)
        s = np.sort(rng.uniform(0.1, 2.0, 6))[::-1]
        kw = dict(n_sources=6, total_input_power=4.0, noise=0.25)

        prev_br = prev_cap = None
        for n_freq in (40, 80, 160):
            freqs = np.linspace(0.0, 1.0, n_freq)
            H = np.exp(-freqs / 0.3)
            H = H / H.max()
            br = get_bitrate_temporal_filter(s, freqs, H, **kw)
            cap = get_capacity_temporal_filter(s, freqs, H, **kw)
            self.assertGreaterEqual(cap, br - 1e-9)
            if prev_br is not None:
                # both stable as resolution increases (within a few percent)
                self.assertLess(abs(br - prev_br) / prev_br, 0.05)
                self.assertLess(abs(cap - prev_cap) / prev_cap, 0.05)
            prev_br, prev_cap = br, cap

    def test_temporal_capacity_zero_power_is_zero(self):
        s = np.array([2.0, 0.5])
        freqs = np.array([0.0, 0.25, 0.5])
        H = np.array([1.0, 0.5, 0.25])
        val = get_capacity_temporal_filter(
            s, freqs, H, n_sources=4, total_input_power=0.0, noise=0.2
        )
        self.assertEqual(val, 0.0)

    def test_output_frequency_spectrum_splits_total_output_power(self):
        s = np.array([2.0, 0.5])
        average_output_power = 1.2
        n_sources = 4
        n_outputs = 3
        noise = 0.3
        output_spectrum = np.array([1.0, 3.0])
        bandwidth_hz = 8.0
        time_resolution = 1.0 / bandwidth_hz
        df = 2.0

        actual = get_bitrate(
            s,
            n_sources=n_sources,
            n_outputs=n_outputs,
            average_output_power=average_output_power,
            noise=noise,
            time_resolution=time_resolution,
            output_frequency_spectrum=output_spectrum,
            output_frequency_bin_width=df,
        )

        weights = output_spectrum / np.sum(output_spectrum)
        bin_noise = noise * np.sqrt(df / bandwidth_hz)
        expected = sum(
            get_bitrate(
                s,
                n_sources=n_sources,
                n_outputs=n_outputs,
                average_output_power=average_output_power * weight,
                noise=bin_noise,
                time_resolution=1.0 / df,
            )
            for weight in weights
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_noise_frequency_spectrum_with_scalar_noise_scales_per_bin_noise(self):
        s = np.array([1.5, 0.25])
        total_input_power = 2.0
        noise = 0.4
        output_spectrum = np.array([1.0, 1.0])
        noise_spectrum = np.array([1.0, 3.0])
        bandwidth_hz = 6.0
        time_resolution = 1.0 / bandwidth_hz
        df = 1.5

        actual = get_bitrate(
            s,
            n_sources=2,
            total_input_power=total_input_power,
            noise=noise,
            time_resolution=time_resolution,
            output_frequency_spectrum=output_spectrum,
            output_frequency_bin_width=df,
            noise_frequency_spectrum=noise_spectrum,
            noise_frequency_bin_width=df,
        )

        output_weights = output_spectrum / np.sum(output_spectrum)
        bin_noise_base = noise * np.sqrt(df / bandwidth_hz)
        expected = sum(
            get_bitrate(
                s,
                n_sources=2,
                total_input_power=total_input_power * output_weight,
                noise=bin_noise_base * noise_level_scale,
                time_resolution=1.0 / df,
            )
            for output_weight, noise_level_scale in zip(
                output_weights,
                noise_spectrum,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_absolute_noise_frequency_spectrum_can_replace_scalar_noise(self):
        s = np.array([1.25, 0.75])
        total_input_power = 3.0
        noise_spectrum = np.array([0.2, 0.5])
        df = 1.0

        actual = get_bitrate(
            s,
            n_sources=2,
            total_input_power=total_input_power,
            noise_frequency_spectrum=noise_spectrum,
            noise_frequency_bin_width=df,
        )

        expected = sum(
            get_bitrate(
                s,
                n_sources=2,
                total_input_power=0.5 * total_input_power,
                noise=noise_level,
                time_resolution=1.0 / df,
            )
            for noise_level in noise_spectrum
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_power_law_frequency_spectrum_matches_explicit_spectrum(self):
        s = np.array([2.0, 1.0, 0.25])
        kwargs = dict(
            beta=1.0,
            min_freq_hz=1.0,
            max_freq_hz=4.0,
            freq_bin_width_hz=1.0,
        )
        explicit = power_law_frequency_spectrum(**kwargs)

        actual = get_bitrate(
            s,
            n_sources=3,
            total_input_power=2.5,
            noise=0.2,
            output_power_law_beta=kwargs["beta"],
            output_power_law_min_freq_hz=kwargs["min_freq_hz"],
            output_power_law_max_freq_hz=kwargs["max_freq_hz"],
            output_power_law_bin_width_hz=kwargs["freq_bin_width_hz"],
        )
        expected = get_bitrate(
            s,
            n_sources=3,
            total_input_power=2.5,
            noise=0.2,
            output_frequency_spectrum=explicit,
            output_frequency_bin_width=kwargs["freq_bin_width_hz"],
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_capacity_frequency_spectrum_sums_per_bin_capacity(self):
        s = np.array([2.0, 0.5])
        total_input_power = 4.0
        output_spectrum = np.array([2.0, 1.0])
        noise_spectrum = np.array([0.25, 0.5])
        df = 2.0

        actual = get_capacity(
            s,
            n_sources=2,
            total_input_power=total_input_power,
            output_frequency_spectrum=output_spectrum,
            output_frequency_bin_width=df,
            noise_frequency_spectrum=noise_spectrum,
            noise_frequency_bin_width=df,
        )

        output_weights = output_spectrum / np.sum(output_spectrum)
        expected = sum(
            get_capacity(
                s,
                total_input_power=total_input_power * output_weight,
                noise=noise_level,
                time_resolution=1.0 / df,
            )
            for output_weight, noise_level in zip(output_weights, noise_spectrum)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_capacity_frequency_spectrum_scales_scalar_noise_to_bin_bandwidth(self):
        s = np.array([1.8, 0.6])
        total_input_power = 3.5
        output_spectrum = np.array([1.0, 2.0])
        noise = 0.45
        bandwidth_hz = 12.0
        time_resolution = 1.0 / bandwidth_hz
        df = 3.0

        actual = get_capacity(
            s,
            n_sources=2,
            total_input_power=total_input_power,
            noise=noise,
            time_resolution=time_resolution,
            output_frequency_spectrum=output_spectrum,
            output_frequency_bin_width=df,
        )

        output_weights = output_spectrum / np.sum(output_spectrum)
        bin_noise = noise * np.sqrt(df / bandwidth_hz)
        expected = sum(
            get_capacity(
                s,
                total_input_power=total_input_power * output_weight,
                noise=bin_noise,
                time_resolution=1.0 / df,
            )
            for output_weight in output_weights
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
