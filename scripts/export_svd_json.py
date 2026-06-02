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
from guti.modality_capacity import compute_bitrate_capacity
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
        "EEG only. Excludes near-boundary BEM artifact voxels, then anchors the "
        "source amplitude so a canonical 20 mm-deep cortical source hits the "
        "literature single-channel SNR; the SVD shape spreads it across modes. "
        "Uses the cached 256-channel lead field, so it is a single anchored "
        "estimate rather than a per-layout sweep.",
    ),
}

# Per-modality default mode. EEG's absolute BEM gain is unreliable, so it defaults
# to the empirically anchored estimate; everything else keeps the physical floor.
DEFAULT_BITRATE_MODE_BY_MODALITY = {
    "eeg": "empirical_anchored",
}


def default_bitrate_mode_for(modality):
    return DEFAULT_BITRATE_MODE_BY_MODALITY.get(modality, DEFAULT_BITRATE_MODE)

MODALITIES = {
    "meg_opm":            "MEG OPM",
    "meg_squid":          "MEG SQUID",
    "eeg":       "EEG",
    "cw_fnirs": "fNIRS CW",
    "td_fnirs": "fNIRS TD",
    "fmri_bold":          "fMRI BOLD",
    "us_free_field_analytical_frequency_sweep": "Ultrasound",
}

# Maps the export modality key to (noise_model, source_orientations) for the shared
# README bitrate/capacity algorithm (guti.modality_capacity). fMRI is omitted (the
# README excludes it — no convergence SVD files).
README_ALGO_MODALITY = {
    "meg_opm": ("meg_opm", 3),
    "meg_squid": ("meg_squid", 3),
    "eeg_openmeeg": ("eeg_openmeeg", 3),
    "cw_fnirs": ("cw_fnirs", 1),
    "td_fnirs": ("td_fnirs_analytical", 1),
    "us_free_field_analytical_frequency_sweep": ("us_analytical", 1),
}

# time resolution per modality (seconds)
TIME_RESOLUTION = {
    "meg_opm":            0.01,   # 100 Hz
    "meg_squid":          0.01,
    "eeg":       0.01,
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


def _optional_np_scalar(data, name):
    if name not in data.files:
        return None
    value = data[name]
    try:
        value = value.item()
    except ValueError:
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_saved_noise_normalized_singular_values(modality, hash_key):
    path = os.path.join("results", "variants", modality, f"{hash_key}.npz")
    if not os.path.exists(path):
        return None, {}
    data = np.load(path, allow_pickle=True)
    if "noise_normalized_singular_values" not in data.files:
        return None, {}
    metadata = {
        "noise_covariance_model": _optional_np_scalar(data, "noise_covariance_model"),
        "noise_detector_std_t": _optional_np_scalar(data, "noise_detector_std_t"),
        "noise_absolute_scale": _optional_np_scalar(data, "noise_absolute_scale"),
        "johnson_voxel_resolution_mm": _optional_np_scalar(
            data,
            "johnson_voxel_resolution_mm",
        ),
        "johnson_solver": _optional_np_scalar(data, "johnson_solver"),
        "johnson_sensor_components": _optional_np_scalar(
            data,
            "johnson_sensor_components",
        ),
        "johnson_n_voxels": _optional_np_scalar(data, "johnson_n_voxels"),
        "johnson_raw_body_noise_median_T_per_sqrtHz": _optional_np_scalar(
            data,
            "johnson_raw_body_noise_median_T_per_sqrtHz",
        ),
    }
    return np.asarray(data["noise_normalized_singular_values"], dtype=float), metadata


def compute_bitrate(
    s,
    modality,
    n_sensors,
    freq=None,
    tier="today",
    time_resolution=0.01,
    params=None,
    noise_mode=DEFAULT_BITRATE_MODE,
    s_noise_normalized=None,
    noise_normalized_detector_std=None,
    noise_absolute_scale=False,
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
        if modality == "eeg":
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

    if s_noise_normalized is not None:
        capacity_scale = capacity_forward_gain_scale(
            modality,
            params=params,
            voxel_size_mm=kwargs["voxel_size_mm"],
        )
        noise_scale = 1.0
        if not noise_absolute_scale and noise_normalized_detector_std is not None:
            noise_scale = float(noise_normalized_detector_std) / detector_noise
        return float(
            get_modality_bitrate(
                np.asarray(s_noise_normalized, dtype=float)
                * capacity_scale
                * noise_scale,
                modality,
                n_sources=n_sources,
                total_input_power=total_input_power,
                noise=1.0,
                time_resolution=tr_for_bitrate,
                hrf_type=getattr(params, "hrf_type", None),
            )
        )

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
        # US: only the 50 kHz proxy frequency is calibrated for the λ³→2 MHz
        # extrapolation, so the chart sweeps sensors at 50 kHz (frequency is not a
        # physical scaling axis here).
        if (
            modality == "us_free_field_analytical_frequency_sweep"
            and params.frequency_hz not in (50000, 50000.0)
        ):
            continue
        n_sensors = params.num_sensors
        freq = getattr(params, "frequency_hz", None)
        s_noise_normalized, noise_metadata = _load_saved_noise_normalized_singular_values(
            modality,
            hash_key,
        )
        noise_model_type = (
            "spatial_covariance" if s_noise_normalized is not None else "scalar_iid"
        )
        capacity_sv_scale = capacity_forward_gain_scale(
            modality,
            params=params,
            voxel_size_mm=getattr(params, "grid_resolution_mm", None),
        )

        # Downsample singular values for web
        idx, sv_vals = downsample(s, MAX_SV_POINTS)
        sv_indices = [i + 1 for i in idx]  # 1-based

        # Bitrate + water-filled capacity from the shared README algorithm
        # (guti.modality_capacity), per variant, for the today and fundamental
        # detector-noise tiers. This is the single source of truth shared with
        # the README modality summary table.
        br_today = br_fund = cap_today = cap_fund = snr_out = None
        nm_so = README_ALGO_MODALITY.get(modality)
        if nm_so is not None:
            nm, source_orientations = nm_so
            for tier, set_rate, set_cap in (
                ("today", "today", "today"),
                ("fundamental", "fundamental", "fundamental"),
            ):
                try:
                    out_bc = compute_bitrate_capacity(
                        s, params, noise_model=nm,
                        source_orientations=source_orientations,
                        s_noise_normalized=s_noise_normalized, tier=tier,
                    )
                    if tier == "today":
                        br_today = out_bc["bitrate_bits_per_s"]
                        cap_today = out_bc["channel_capacity_bits_per_s"]
                        snr_out = out_bc["output_snr"]
                    else:
                        br_fund = out_bc["bitrate_bits_per_s"]
                        cap_fund = out_bc["channel_capacity_bits_per_s"]
                except Exception as e:
                    print(f"  {modality} {tier} bitrate/capacity failed: {e}")

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
            "bitrate_today": br_today,
            "bitrate_fundamental": br_fund,
            "capacity_today": cap_today,
            "capacity_fundamental": cap_fund,
            "output_snr": snr_out,
            "noise_model_type": noise_model_type,
            **{
                key: value
                for key, value in noise_metadata.items()
                if value is not None
            },
        }
        records.append(rec)
        br_str = f"{br_today:.0f}" if br_today is not None else "N/A"
        cap_str = f"{cap_today:.0f}" if cap_today is not None else "N/A"
        print(
            f"  {hash_key}: N={n_sensors}, sp={params.source_spacing_mm}mm "
            f"→ bitrate={br_str} b/s, capacity={cap_str} b/s"
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
