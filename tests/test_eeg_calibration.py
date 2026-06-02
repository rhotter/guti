import unittest

import numpy as np

from guti.modalities.eeg.calibration import (
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

    def test_boundary_exclusion_removes_artifacts(self):
        """The raw spectrum is dominated by BEM boundary blow-ups; excluding the
        4 mm margin collapses the top dynamic range from ~thousands to single digits."""
        raw = _top_dynamic_range(self.A)
        A_clean, _ = exclude_boundary_voxels(self.A, self.pos, margin_mm=4.0)
        clean = _top_dynamic_range(A_clean)
        self.assertGreater(raw, 1000.0)
        self.assertLess(clean, 10.0)

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
        """Anchored EEG is the same order as guti2 (~80k), not the raw-gain ~1k."""
        bits = anchored_eeg_bitrate(self.snr_today, leadfield=(self.A, self.pos))
        self.assertGreater(bits, 25_000.0)
        self.assertLess(bits, 60_000.0)


if __name__ == "__main__":
    unittest.main()
