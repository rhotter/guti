import unittest

import numpy as np

from guti.capacity import get_bitrate
from guti.modalities.eeg.calibration import (
    anchored_mode_snr,
    anchored_eeg_capacity,
    anchored_eeg_bitrate,
    exclude_boundary_voxels,
    load_eeg_leadfield,
)
from guti.noise_models import get_noise_model


def _top_dynamic_range(A):
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[min(9, len(s) - 1)]


class TestEEGCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, cls.pos = load_eeg_leadfield()
        model = get_noise_model("eeg_openmeeg")
        cls.snr_today = model.anchor_snr_today
        cls.snr_fund = model.anchor_snr_fundamental

    def test_anchor_snr_values_present(self):
        self.assertEqual(self.snr_today, 1.0)
        self.assertEqual(self.snr_fund, 3.4)

    def test_boundary_exclusion_keeps_spectrum_clean(self):
        """Boundary exclusion should leave the cached spectrum artifact-free.

        Some cached lead fields are already generated with a surface margin; older
        all-grid caches had large boundary blow-ups that this step removed.
        """
        raw = _top_dynamic_range(self.A)
        A_clean, _ = exclude_boundary_voxels(self.A, self.pos, margin_mm=4.0)
        clean = _top_dynamic_range(A_clean)
        self.assertLess(clean, 10.0)
        self.assertLessEqual(clean, raw * 1.01)

    def test_ref_depth_is_a_smooth_knob(self):
        """With artifacts excluded, anchor depth is a smooth, monotonic modeling
        knob (deeper reference → weaker peak → larger inferred signal → more bits),
        not the pathological ~4x jump the raw field shows when a shallow anchor grabs
        a boundary blow-up. Bounded well under that here."""
        vals = [
            anchored_eeg_bitrate(
                self.snr_today, ref_depth_mm=d, leadfield=(self.A, self.pos)
            )
            for d in (15.0, 20.0, 25.0, 30.0)
        ]
        self.assertEqual(vals, sorted(vals))  # monotonic increasing
        spread = (max(vals) - min(vals)) / np.mean(vals)
        self.assertLess(spread, 0.45)

    def test_fundamental_exceeds_today(self):
        lf = (self.A, self.pos)
        self.assertGreater(
            anchored_eeg_bitrate(self.snr_fund, leadfield=lf),
            anchored_eeg_bitrate(self.snr_today, leadfield=lf),
        )

    def test_bitrate_in_expected_band(self):
        """Anchored EEG is well above the raw-gain ~1k estimate."""
        bits = anchored_eeg_bitrate(self.snr_today, leadfield=(self.A, self.pos))
        self.assertGreater(bits, 10_000.0)
        self.assertLess(bits, 60_000.0)

    def test_capacity_exceeds_equal_power_bitrate(self):
        lf = (self.A, self.pos)
        self.assertGreater(
            anchored_eeg_capacity(self.snr_today, leadfield=lf),
            anchored_eeg_bitrate(self.snr_today, leadfield=lf),
        )

    def test_temporal_spectrum_scales_full_band_eeg_noise_per_bin(self):
        """EEG full-band Johnson noise should be scaled to each freq bin."""
        output_spectrum = np.array([1.0, 3.0])
        bandwidth_hz = 8.0
        time_resolution = 1.0 / bandwidth_hz
        df = 2.0
        lf = (self.A, self.pos)

        actual = anchored_eeg_bitrate(
            self.snr_today,
            time_resolution=time_resolution,
            leadfield=lf,
            spectrum_kwargs={
                "output_frequency_spectrum": output_spectrum,
                "output_frequency_bin_width": df,
            },
        )

        A_clean, pos_clean = exclude_boundary_voxels(self.A, self.pos)
        snr_modes = anchored_mode_snr(A_clean, pos_clean, self.snr_today)
        output_weights = output_spectrum / np.sum(output_spectrum)
        bin_noise = np.sqrt(df / bandwidth_hz)
        expected = sum(
            get_bitrate(
                snr_modes,
                n_sources=len(snr_modes),
                total_input_power=float(len(snr_modes)) * output_weight,
                noise=bin_noise,
                time_resolution=1.0 / df,
            )
            for output_weight in output_weights
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
