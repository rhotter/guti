import unittest

import numpy as np

from guti.modalities.eeg.calibration import (
    DEFAULT_EFFECTIVE_CHANNELS,
    anchored_eeg_bitrate,
    anchored_mode_snr,
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
        cls.clean, _ = exclude_boundary_voxels(cls.A, cls.pos, 4.0)
        model = get_noise_model("eeg_openmeeg")
        cls.snr_today = model.anchor_snr_today
        cls.snr_fund = model.anchor_snr_fundamental

    def test_anchor_snr_values_present(self):
        self.assertEqual(self.snr_today, 1.0)
        self.assertEqual(self.snr_fund, 3.4)

    def test_boundary_exclusion_removes_artifacts(self):
        """The raw spectrum is dominated by BEM boundary blow-ups; excluding the
        4 mm margin collapses the top dynamic range from ~thousands to single digits."""
        self.assertGreater(_top_dynamic_range(self.A), 1000.0)
        self.assertLess(_top_dynamic_range(self.clean), 10.0)

    def test_per_mode_snr_is_physical(self):
        """The whole point of the fix: per-mode SNR must be physical. The best mode
        sits at snr_ref·√N_eff (a few), NOT the ~200 the old deep-voxel anchor gave."""
        m = anchored_mode_snr(self.clean, self.snr_today)
        self.assertAlmostEqual(m[0], self.snr_today * np.sqrt(DEFAULT_EFFECTIVE_CHANNELS), places=6)
        self.assertLess(m[0], 10.0)            # not the old ~218
        self.assertGreater(int((m > 1).sum()), 3)   # but more than a couple usable modes
        self.assertLess(int((m > 1).sum()), 60)     # and not ~all 256

    def test_array_gain_monotonic(self):
        """More effective channels → more gain → more bits (the array-gain knob)."""
        lf = (self.A, self.pos)
        vals = [
            anchored_eeg_bitrate(self.snr_today, effective_channels=n, leadfield=lf)
            for n in (16, 25, 32, 40)
        ]
        self.assertEqual(vals, sorted(vals))

    def test_fundamental_exceeds_today(self):
        lf = (self.A, self.pos)
        self.assertGreater(
            anchored_eeg_bitrate(self.snr_fund, leadfield=lf),
            anchored_eeg_bitrate(self.snr_today, leadfield=lf),
        )

    def test_bitrate_in_expected_band(self):
        """EEG is a low-spatial-resolution modality: ~10-40 DOF × 100 Hz × modest
        SNR → O(1-8k) bits/s, in line with the original physical estimate."""
        today = anchored_eeg_bitrate(self.snr_today, leadfield=(self.A, self.pos))
        fund = anchored_eeg_bitrate(self.snr_fund, leadfield=(self.A, self.pos))
        self.assertGreater(today, 1_000.0)
        self.assertLess(today, 4_000.0)
        self.assertGreater(fund, 3_000.0)
        self.assertLess(fund, 9_000.0)


if __name__ == "__main__":
    unittest.main()
