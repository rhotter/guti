#!/usr/bin/env python3
"""Generate the per-modality noise-levels table for the README.

Columns (units inline in each cell, since they differ per modality):
    Modality | Current noise level | Noise limit | Output amplitude

All values are taken at each modality's reference sensor count and reference
bandwidth, straight from ``guti.noise_models.NOISE_MODELS``:
    * current noise level = ``today_best_noise``
    * noise limit         = ``physical_floor_noise``
    * output amplitude    = ``typical_signal_amplitude`` (the empirically
      observed signal, in the same measurement units as the noise)

Usage:
    python scripts/make_noise_levels_table.py            # print table
    python scripts/make_noise_levels_table.py --write    # inject into README
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guti.noise_models import NOISE_MODELS  # noqa: E402

README_PATH = REPO_ROOT / "README.md"
TABLE_BEGIN = "<!-- BEGIN GENERATED MODALITY NOISE LEVELS -->"
TABLE_END = "<!-- END GENERATED MODALITY NOISE LEVELS -->"

# Display name, value-formatting style, and base unit per modality, in order.
# Each entry's first field is a tuple of candidate keys (the TD-fNIRS model has
# been registered under both "td_fnirs" and the older "td_fnirs_analytical").
#   style "si"    -> SI-prefixed physical unit (e.g. 50 fT, 92.5 nV)
#   style "ratio" -> dimensionless, shown as % / ppm / ppb
ROWS = [
    (("eeg",), "EEG", "si", "V"),
    (("meg_opm",), "MEG (OPM)", "si", "T"),
    (("meg_squid",), "MEG (SQUID)", "si", "T"),
    (("cw_fnirs",), "CW-fNIRS", "ratio", "ΔI/I"),
    (("td_fnirs", "td_fnirs_analytical"), "TD-fNIRS", "ratio", "ΔI/I"),
    (("us_analytical",), "Ultrasound", "ratio", "pressure ratio"),
    (("fmri_bold",), "fMRI (BOLD)", "ratio", "fractional BOLD"),
]

# Engineering SI prefixes keyed by power-of-ten exponent (steps of 3).
_SI_PREFIX = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m",
              0: "", 3: "k", 6: "M", 9: "G"}


def resolve_model(candidates):
    """Return the first registered NoiseModel among candidate keys."""
    for key in candidates:
        if key in NOISE_MODELS:
            return NOISE_MODELS[key]
    raise KeyError(f"No noise model registered for any of {candidates}")


def _sig3(value: float) -> str:
    """3 significant figures as a plain decimal (no scientific notation)."""
    if value == 0:
        return "0"
    rounded = round(value, -int(math.floor(math.log10(abs(value)))) + 2)
    return f"{rounded:f}".rstrip("0").rstrip(".")


def _si_format(value: float, unit: str) -> str:
    """Format with the nearest SI prefix, e.g. ``50 fT``, ``92.5 nV``."""
    if value == 0:
        return f"0 {unit}"
    exp = math.floor(math.log10(abs(value)))
    pexp = min(9, max(-15, 3 * math.floor(exp / 3)))
    mantissa = value / 10**pexp
    return f"{_sig3(mantissa)} {_SI_PREFIX[pexp]}{unit}"


def _ratio_unit(reference: float) -> tuple[str, float]:
    """Pick a single % / ppm / ppb unit (label, scale) for a row from its
    reference (output) amplitude, so every cell in the row shares it."""
    if abs(reference) >= 1e-2:
        return "%", 1e2
    if abs(reference) >= 1e-6:
        return "ppm", 1e6
    return "ppb", 1e9


def _ratio_format(value: float, label: str, scale: float) -> str:
    """Format a dimensionless ratio in a fixed unit, e.g. ``1000 ppm``, ``1.25%``."""
    scaled = _sig3(value * scale)
    return f"{scaled}%" if label == "%" else f"{scaled} {label}"


def build_table() -> str:
    header = (
        "| Modality | Current noise level | Noise limit | Output amplitude |\n"
        "| --- | ---: | ---: | ---: |"
    )
    lines = [header]
    for candidates, name, style, unit in ROWS:
        m = resolve_model(candidates)
        values = (
            m.today_best_noise,
            m.physical_floor_noise,
            m.typical_signal_amplitude,
        )
        if style == "si":
            cells = [_si_format(v, unit) for v in values]
        elif style == "ratio":
            label, scale = _ratio_unit(m.typical_signal_amplitude)
            cells = [_ratio_format(v, label, scale) for v in values]
        else:
            raise ValueError(f"Unknown format style {style!r}")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def write_into_readme(table: str) -> None:
    text = README_PATH.read_text()
    if TABLE_BEGIN not in text or TABLE_END not in text:
        raise SystemExit(
            f"Markers not found in {README_PATH}. Add:\n{TABLE_BEGIN}\n{TABLE_END}"
        )
    pre = text.split(TABLE_BEGIN)[0]
    post = text.split(TABLE_END)[1]
    block = (
        f"{TABLE_BEGIN}\n"
        "### Noise Levels\n\n"
        "Per-modality detector noise (per sensor, at the reference sensor count and "
        "bandwidth) versus the typical observed output signal, in matching units. "
        "Generated by `scripts/make_noise_levels_table.py`.\n\n"
        f"{table}\n\n"
        "Dimensionless quantities (shown as ppm / %): CW- and TD-fNIRS are "
        "intensity ratios ΔI/I, Ultrasound is a pressure-amplitude ratio, and "
        "fMRI is fractional BOLD signal.\n\n"
        f"{TABLE_END}"
    )
    README_PATH.write_text(pre + block + post)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Inject the table into README.md between the generation markers",
    )
    args = parser.parse_args()

    table = build_table()
    if args.write:
        write_into_readme(table)
        print(f"Wrote noise-levels table into {README_PATH}")
    else:
        print(table)


if __name__ == "__main__":
    main()
