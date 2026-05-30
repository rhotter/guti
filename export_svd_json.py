"""
Export SVD variant data to JSON files for the interactive web component.

For each modality, writes:
  web/public/data/{modality}.json

Schema per file:
{
  "modality": "meg_opm",
  "label": "MEG OPM",
  "sweep_params": ["num_sensors", "source_spacing_mm"],
  "variants": [
    {
      "num_sensors": 500,
      "source_spacing_mm": 5.0,
      "n_singular_values": 300,
      "singular_values": [0.123, ...],   // downsampled to ≤300 points for web
      "sv_indices": [1, 2, ...],         // corresponding indices (1-based)
      "bitrate_today": 1450.3,
      "bitrate_fundamental": 23456.7,
      "first_sv": 0.123,
      "snr_empirical_today": 53.8
    },
    ...
  ]
}
"""

import os, json, math
import numpy as np

from guti.data_utils import list_svd_variants, load_svd_variant
from guti.parameters import Parameters
from guti.core import get_bitrate
from guti.noise_models import (
    capacity_forward_gain_scale,
    compute_noise_empirical,
    compute_noise_effective,
    compute_empirical_snr,
    get_noise_model,
    scale_singular_values_for_capacity,
)

OUT_DIR = "web/public/data"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SV_POINTS = 300   # max singular values to embed per variant

MODALITIES = {
    "meg_opm":            "MEG OPM",
    "meg_squid":          "MEG SQUID",
    "eeg_openmeeg":       "EEG (OpenMEEG)",
    "fnirs_analytical_cw": "fNIRS CW",
}

# time resolution per modality (seconds)
TIME_RESOLUTION = {
    "meg_opm":            0.01,   # 100 Hz
    "meg_squid":          0.01,
    "eeg_openmeeg":       0.01,
    "fnirs_analytical_cw": 1.0,  # 1 Hz hemodynamic
}


def downsample(arr, n):
    """Logarithmically downsample array to n points."""
    if len(arr) <= n:
        idx = np.arange(len(arr))
    else:
        idx = np.unique(np.round(
            np.logspace(0, np.log10(len(arr) - 1), n)
        ).astype(int))
        idx = np.clip(idx, 0, len(arr) - 1)
    return idx.tolist(), arr[idx].tolist()


def compute_bitrate(
    s,
    modality,
    n_sensors,
    freq=None,
    tier="today",
    time_resolution=0.01,
    params=None,
):
    model = get_noise_model(modality)
    s_capacity = scale_singular_values_for_capacity(
        s,
        modality,
        params=params,
        voxel_size_mm=getattr(params, "grid_resolution_mm", None),
    )
    if model.typical_signal_amplitude > 0.0:
        noise = compute_noise_empirical(s_capacity, modality, n_sensors=n_sensors,
                                        frequency_hz=freq, tier=tier)
    else:
        noise = compute_noise_effective(modality, n_sensors=n_sensors,
                                        frequency_hz=freq, tier=tier)
    return float(get_bitrate(s_capacity, noise, time_resolution=time_resolution))


def export_modality(modality, label):
    print(f"\n=== {label} ({modality}) ===")
    tr = TIME_RESOLUTION.get(modality, 1.0)
    model = get_noise_model(modality)

    # Collect all variants
    all_variants = list_svd_variants(modality)
    if not all_variants:
        print("  No variants found, skipping.")
        return

    records = []
    for hash_key, v in all_variants.items():
        s = v["s"]
        params = v["params"]
        n_sensors = params.num_sensors
        freq = getattr(params, "frequency_hz", None)
        capacity_sv_scale = capacity_forward_gain_scale(
            modality,
            params=params,
            voxel_size_mm=getattr(params, "grid_resolution_mm", None),
        )

        # Downsample singular values for web
        idx, sv_vals = downsample(s, MAX_SV_POINTS)
        sv_indices = [i + 1 for i in idx]  # 1-based

        # Bitrates
        try:
            br_today = compute_bitrate(s, modality, n_sensors, freq, "today", tr, params)
        except Exception as e:
            br_today = None
            print(f"  bitrate today failed: {e}")

        try:
            br_fund = compute_bitrate(s, modality, n_sensors, freq, "fundamental", tr, params)
        except Exception as e:
            br_fund = None

        # Empirical SNR
        try:
            snr_emp = float(compute_empirical_snr(modality, n_sensors=n_sensors, frequency_hz=freq))
        except Exception:
            snr_emp = None

        rec = {
            "hash": hash_key,
            "num_sensors": params.num_sensors,
            "source_spacing_mm": params.source_spacing_mm,
            "sensor_offset_mm": getattr(params, "sensor_offset_mm", None),
            "frequency_hz": freq,
            "n_singular_values": len(s),
            "sv_indices": sv_indices,
            "singular_values": [float(x) for x in sv_vals],
            "first_sv": float(s[0]),
            "capacity_singular_value_scale": capacity_sv_scale,
            "bitrate_today": br_today,
            "bitrate_fundamental": br_fund,
            "snr_empirical_today": snr_emp,
        }
        records.append(rec)
        br_str = f"{br_today:.0f}" if br_today is not None else "N/A"
        print(f"  {hash_key}: N={n_sensors}, sp={params.source_spacing_mm}mm → br_today={br_str} b/s")

    # Determine which parameters were actually swept
    sweep_params = []
    for key in ["num_sensors", "source_spacing_mm", "frequency_hz"]:
        vals = set(r[key] for r in records if r[key] is not None)
        if len(vals) > 1:
            sweep_params.append(key)

    out = {
        "modality": modality,
        "label": label,
        "sweep_params": sweep_params,
        "noise_label_today": f"{model.today_best_noise:.2e} {model.measurement_units}",
        "noise_label_fundamental": f"{model.physical_floor_noise:.2e} {model.measurement_units}",
        "typical_signal": model.typical_signal_amplitude,
        "variants": records,
    }

    path = os.path.join(OUT_DIR, f"{modality}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {path} ({len(records)} variants)")


if __name__ == "__main__":
    for modality, label in MODALITIES.items():
        export_modality(modality, label)
    print("\nDone.")
