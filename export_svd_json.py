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
      "bitrate_today": 1450.3,              // physical detector-floor mode
      "bitrate_fundamental": 23456.7,       // physical detector-floor mode
      "bitrate_physical_today": 1450.3,
      "bitrate_physical_fundamental": 23456.7,
      "bitrate_empirical_today": 123.4,
      "bitrate_empirical_fundamental": 456.7,
      "first_sv": 0.123,
      "capacity_singular_value_scale": 1.0,
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
from guti.core import get_bitrate, get_bitrate_temporal_filter
from guti.hrf import get_canonical_hrf_spectrum
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
DEFAULT_BITRATE_MODE = "physical_detector_floor"
BITRATE_MODES = {
    "physical_detector_floor": (
        "Physical detector floor",
        "Uses detector_noise/source_amplitude and preserves raw forward gain. "
        "Voxel-density modalities are saved as voxel-integrated transfer functions.",
    ),
    "empirical_observed_snr": (
        "Empirical observed SNR",
        "Uses Frobenius(SVD)/observed_SNR and normalizes away raw forward gain.",
    ),
}

MODALITIES = {
    "meg_opm":            "MEG OPM",
    "meg_squid":          "MEG SQUID",
    "eeg_openmeeg":       "EEG (OpenMEEG)",
    "fnirs_analytical_cw": "fNIRS CW",
    "fmri_bold":          "fMRI BOLD",
}

# time resolution per modality (seconds)
TIME_RESOLUTION = {
    "meg_opm":            0.01,   # 100 Hz
    "meg_squid":          0.01,
    "eeg_openmeeg":       0.01,
    "fnirs_analytical_cw": 1.0,  # 1 Hz hemodynamic
    "fmri_bold":           2.0,  # TR = 2 s; HRF handled explicitly below
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


def _noise_kwargs(params, freq):
    return {
        "frequency_hz": freq,
        "voxel_size_mm": getattr(params, "grid_resolution_mm", None),
        "tr_s": getattr(params, "time_resolution", None),
        "bold_contrast": getattr(params, "bold_contrast", None),
        "bold_snr": getattr(params, "bold_snr", None),
    }


def compute_bitrate(
    s,
    modality,
    n_sensors,
    freq=None,
    tier="today",
    time_resolution=0.01,
    params=None,
    noise_mode=DEFAULT_BITRATE_MODE,
):
    model = get_noise_model(modality)
    kwargs = _noise_kwargs(params, freq)
    s_capacity = scale_singular_values_for_capacity(
        s,
        modality,
        params=params,
        voxel_size_mm=kwargs["voxel_size_mm"],
    )

    if noise_mode == "empirical_observed_snr":
        if model.typical_signal_amplitude <= 0.0:
            return None
        noise = compute_noise_empirical(
            s_capacity,
            modality,
            n_sensors=n_sensors,
            tier=tier,
            **kwargs,
        )
        return float(get_bitrate(s_capacity, noise, time_resolution=time_resolution))

    if noise_mode != "physical_detector_floor":
        raise ValueError(f"Unknown noise_mode {noise_mode!r}")

    if modality == "fmri_bold":
        voxel_size_mm = getattr(params, "grid_resolution_mm", None)
        tr_s = getattr(params, "time_resolution", None) or time_resolution
        noise = compute_noise_effective(
            modality,
            n_sensors=n_sensors,
            tier=tier,
            voxel_size_mm=voxel_size_mm,
            tr_s=tr_s,
            bold_contrast=kwargs["bold_contrast"],
            bold_snr=kwargs["bold_snr"],
        )
        freqs, H = get_canonical_hrf_spectrum(
            f_max=0.5 / tr_s,
            df=0.002,
            hrf_type=getattr(params, "hrf_type", None) or "spm",
            tr=0.01,
        )
        return float(get_bitrate_temporal_filter(s, noise, freqs, H))

    noise = compute_noise_effective(
        modality,
        n_sensors=n_sensors,
        tier=tier,
        **kwargs,
    )
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

        # Bitrates.  The compatibility fields bitrate_today/fundamental are
        # the physical detector-floor mode used by the blog's first-principles
        # capacity claim.  Empirical observed-SNR values are exported separately
        # for diagnostic comparisons.
        try:
            br_physical_today = compute_bitrate(
                s, modality, n_sensors, freq, "today", tr, params,
                noise_mode="physical_detector_floor",
            )
        except Exception as e:
            br_physical_today = None
            print(f"  physical bitrate today failed: {e}")

        try:
            br_physical_fund = compute_bitrate(
                s, modality, n_sensors, freq, "fundamental", tr, params,
                noise_mode="physical_detector_floor",
            )
        except Exception as e:
            br_physical_fund = None
            print(f"  physical bitrate fundamental failed: {e}")

        try:
            br_emp_today = compute_bitrate(
                s, modality, n_sensors, freq, "today", tr, params,
                noise_mode="empirical_observed_snr",
            )
        except Exception as e:
            br_emp_today = None
            print(f"  empirical bitrate today failed: {e}")

        try:
            br_emp_fund = compute_bitrate(
                s, modality, n_sensors, freq, "fundamental", tr, params,
                noise_mode="empirical_observed_snr",
            )
        except Exception:
            br_emp_fund = None

        # Empirical SNR
        try:
            snr_emp = float(
                compute_empirical_snr(
                    modality,
                    n_sensors=n_sensors,
                    frequency_hz=freq,
                    tier="today",
                    voxel_size_mm=getattr(params, "grid_resolution_mm", None),
                    tr_s=getattr(params, "time_resolution", None),
                    bold_contrast=getattr(params, "bold_contrast", None),
                    bold_snr=getattr(params, "bold_snr", None),
                )
            )
        except Exception:
            snr_emp = None

        rec = {
            "hash": hash_key,
            "num_sensors": params.num_sensors,
            "source_spacing_mm": params.source_spacing_mm,
            "grid_resolution_mm": getattr(params, "grid_resolution_mm", None),
            "psf_fwhm_mm": getattr(params, "psf_fwhm_mm", None),
            "bold_contrast": getattr(params, "bold_contrast", None),
            "bold_snr": getattr(params, "bold_snr", None),
            "sensor_offset_mm": getattr(params, "sensor_offset_mm", None),
            "frequency_hz": freq,
            "n_singular_values": len(s),
            "sv_indices": sv_indices,
            "singular_values": [float(x) for x in sv_vals],
            "first_sv": float(s[0]),
            "capacity_singular_value_scale": capacity_sv_scale,
            "bitrate_today": br_physical_today,
            "bitrate_fundamental": br_physical_fund,
            "bitrate_physical_today": br_physical_today,
            "bitrate_physical_fundamental": br_physical_fund,
            "bitrate_empirical_today": br_emp_today,
            "bitrate_empirical_fundamental": br_emp_fund,
            "snr_empirical_today": snr_emp,
        }
        records.append(rec)
        br_str = f"{br_physical_today:.0f}" if br_physical_today is not None else "N/A"
        emp_str = f"{br_emp_today:.0f}" if br_emp_today is not None else "N/A"
        print(
            f"  {hash_key}: N={n_sensors}, sp={params.source_spacing_mm}mm "
            f"→ physical_today={br_str} b/s, empirical_today={emp_str} b/s"
        )

    # Determine which parameters were actually swept
    sweep_params = []
    for key in ["num_sensors", "source_spacing_mm", "grid_resolution_mm", "psf_fwhm_mm", "bold_snr", "frequency_hz"]:
        vals = set(r[key] for r in records if r[key] is not None)
        if len(vals) > 1:
            sweep_params.append(key)

    out = {
        "modality": modality,
        "label": label,
        "sweep_params": sweep_params,
        "default_bitrate_mode": DEFAULT_BITRATE_MODE,
        "bitrate_modes": {
            key: {"label": label, "description": description}
            for key, (label, description) in BITRATE_MODES.items()
        },
        "noise_label_today": f"{model.today_best_noise:.2e} {model.measurement_units}",
        "noise_label_fundamental": f"{model.physical_floor_noise:.2e} {model.measurement_units}",
        "source_amplitude": model.source_amplitude,
        "source_amplitude_units": model.source_amplitude_units,
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
