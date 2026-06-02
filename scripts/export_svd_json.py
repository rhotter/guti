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
from guti.hrf import get_modality_bitrate, is_hemodynamic
from guti.noise_models import (
    capacity_forward_gain_scale,
    compute_detector_noise_std,
    compute_noise_empirical,
    compute_total_input_power,
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
        "Uses detector output noise with total input power set by source_amplitude² "
        "per source channel. Voxel-density modalities are saved as voxel-integrated "
        "transfer functions.",
    ),
    "empirical_observed_snr": (
        "Empirical observed SNR",
        "Uses Frobenius(SVD)/observed_SNR and normalizes away raw forward gain.",
    ),
    "empirical_anchored": (
        "Empirically anchored",
        "EEG only. Excludes near-boundary BEM artifact voxels, then pins the best "
        "spatial mode to the literature single-channel SNR × the √N_eff array gain "
        "(N_eff ≈ 32 independent channels under correlated EEG noise); the lead-field "
        "spectrum shape spreads it across the rest. Only the shape is used, not the "
        "unreliable absolute BEM gain. Uses the cached 256-channel lead field, so it "
        "is a single anchored estimate rather than a per-layout sweep.",
    ),
}

# Per-modality default mode. EEG's absolute BEM gain is unreliable, so it defaults
# to the empirically anchored estimate; everything else keeps the physical floor.
DEFAULT_BITRATE_MODE_BY_MODALITY = {
    "eeg_openmeeg": "empirical_anchored",
}


def default_bitrate_mode_for(modality):
    return DEFAULT_BITRATE_MODE_BY_MODALITY.get(modality, DEFAULT_BITRATE_MODE)

MODALITIES = {
    "meg_opm":            "MEG OPM",
    "meg_squid":          "MEG SQUID",
    "eeg_openmeeg":       "EEG (OpenMEEG)",
    "cw_fnirs": "fNIRS CW",
    "td_fnirs": "fNIRS TD",
    "fmri_bold":          "fMRI BOLD",
    "us_free_field_analytical_frequency_sweep": "Ultrasound",
}

# time resolution per modality (seconds)
TIME_RESOLUTION = {
    "meg_opm":            0.01,   # 100 Hz
    "meg_squid":          0.01,
    "eeg_openmeeg":       0.01,
    "cw_fnirs": 1.0,  # 1 Hz hemodynamic
    "td_fnirs": 1.0,  # 1 Hz hemodynamic
    "fmri_bold":           2.0,  # TR = 2 s; HRF handled explicitly below
    "us_free_field_analytical_frequency_sweep": 1.0,
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


def _infer_n_sources_for_bitrate(modality, params, s_capacity):
    if params is None:
        return int(len(s_capacity))
    if getattr(params, "matrix_size", None) is not None:
        return int(params.matrix_size[1])
    if modality.startswith("meg_") and params.source_spacing_mm is not None:
        from guti.core import get_grid_positions

        return 3 * len(get_grid_positions(grid_spacing_mm=params.source_spacing_mm))
    if modality.startswith("eeg_") and params.num_brain_grid_points is not None:
        return 3 * int(params.num_brain_grid_points)
    if params.num_brain_grid_points is not None:
        return int(params.num_brain_grid_points)

    # Physical and empirical bitrate modes use a per-source input power, so the
    # actual source count cancels out.  Keep spectrum-only callers usable by
    # falling back to one unit-power source per singular value.
    return int(len(s_capacity))


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

    if noise_mode == "empirical_anchored":
        # Anchored capacity ignores the saved (artifact-contaminated) spectrum and
        # recomputes from the raw lead field with boundary voxels excluded.
        snr_ref = (
            model.anchor_snr_today if tier == "today"
            else model.anchor_snr_fundamental
        )
        if snr_ref <= 0.0:
            return None
        if modality == "eeg_openmeeg":
            from guti.modalities.eeg.calibration import anchored_eeg_bitrate

            return anchored_eeg_bitrate(snr_ref, time_resolution=time_resolution)
        return None  # anchoring is only defined for EEG today

    kwargs = _noise_kwargs(params, freq)
    s_capacity = scale_singular_values_for_capacity(
        s,
        modality,
        params=params,
        voxel_size_mm=kwargs["voxel_size_mm"],
    )
    n_sources = _infer_n_sources_for_bitrate(modality, params, s_capacity)

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
        return float(
            get_modality_bitrate(
                s_capacity,
                modality,
                n_sources=n_sources,
                total_input_power=float(n_sources),
                noise=noise,
                time_resolution=time_resolution,
                hrf_type=getattr(params, "hrf_type", None),
            )
        )

    if noise_mode != "physical_detector_floor":
        raise ValueError(f"Unknown noise_mode {noise_mode!r}")

    if modality == "fmri_bold" and kwargs["tr_s"] is None:
        kwargs = {**kwargs, "tr_s": time_resolution}

    detector_noise = compute_detector_noise_std(
        modality,
        n_sensors=n_sensors,
        tier=tier,
        **kwargs,
    )
    total_input_power = compute_total_input_power(
        modality,
        n_sources=n_sources,
        bold_contrast=kwargs["bold_contrast"],
    )

    # Hemodynamic modalities set their HRF bandwidth from the BOLD/optical TR
    # (params override, else the modality default); other modalities use a flat
    # per-sample time_resolution scaling inside get_modality_bitrate.
    tr_for_bitrate = time_resolution
    if is_hemodynamic(modality):
        tr_for_bitrate = kwargs["tr_s"] or time_resolution

    return float(
        get_modality_bitrate(
            s_capacity,
            modality,
            n_sources=n_sources,
            total_input_power=total_input_power,
            noise=detector_noise,
            time_resolution=tr_for_bitrate,
            hrf_type=getattr(params, "hrf_type", None),
        )
    )


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

        # Empirically anchored (EEG only; None elsewhere).
        try:
            br_anchored_today = compute_bitrate(
                s, modality, n_sensors, freq, "today", tr, params,
                noise_mode="empirical_anchored",
            )
            br_anchored_fund = compute_bitrate(
                s, modality, n_sensors, freq, "fundamental", tr, params,
                noise_mode="empirical_anchored",
            )
        except Exception as e:
            br_anchored_today = br_anchored_fund = None
            print(f"  anchored bitrate failed: {e}")

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
            "bitrate_anchored_today": br_anchored_today,
            "bitrate_anchored_fundamental": br_anchored_fund,
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
        "default_bitrate_mode": default_bitrate_mode_for(modality),
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
