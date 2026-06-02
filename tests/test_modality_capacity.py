import csv
import unittest
from pathlib import Path

from guti.modality_capacity import compute_bitrate_capacity
from scripts.plot_modality_convergence import load_variant

SUMMARY = Path("results/modality_correlated_noise_summary/summary.csv")

# summary label -> (noise_model, source_orientations). EEG is excluded (its converged
# row uses the anchored calibration, while the charts use each variant's own spectrum
# by design); US is excluded (its summary row uses a separate cone-slice model, while
# the charts use the RBC λ³ sensor sweep).
PARITY = {
    "MEG OPM": ("meg_opm", 3),
    "MEG SQUID": ("meg_squid", 3),
    "fNIRS CW": ("cw_fnirs", 1),
}


class TestModalityCapacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {r["modality"]: r for r in csv.DictReader(open(SUMMARY))}

    def test_readme_parity(self):
        """The shared function reproduces the README summary table within <1%."""
        for label, (nm, so) in PARITY.items():
            row = self.rows[label]
            s, params, snn, _ = load_variant(row["source_path"])
            out = compute_bitrate_capacity(
                s, params, noise_model=nm, source_orientations=so,
                s_noise_normalized=snn,
            )
            for got_key, ref_key in (
                ("bitrate_bits_per_s", "bitrate_bits_per_s"),
                ("channel_capacity_bits_per_s", "capacity_bits_per_s"),
            ):
                got = out[got_key]
                ref = float(row[ref_key])
                self.assertAlmostEqual(
                    got / ref, 1.0, delta=0.01,
                    msg=f"{label} {got_key}: {got:,.0f} vs README {ref:,.0f}",
                )

    def test_capacity_at_least_bitrate(self):
        for label, (nm, so) in PARITY.items():
            row = self.rows[label]
            s, params, snn, _ = load_variant(row["source_path"])
            out = compute_bitrate_capacity(
                s, params, noise_model=nm, source_orientations=so,
                s_noise_normalized=snn,
            )
            self.assertGreaterEqual(
                out["channel_capacity_bits_per_s"],
                out["bitrate_bits_per_s"] * (1 - 1e-9),
                msg=f"{label}: capacity < bitrate",
            )


if __name__ == "__main__":
    unittest.main()
