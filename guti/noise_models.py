from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from guti.core import get_bitrate_channel_capacity, noise_floor_from_total_snr


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K_B = 1.380649e-23  # Boltzmann constant, J/K
BODY_TEMP_K = 310  # body temperature in kelvin
H_PLANCK = 6.62607015e-34  # Planck constant, J·s
C_LIGHT = 2.998e8  # speed of light, m/s


@dataclass(frozen=True)
class NoiseModel:
    """
    Modality-level noise model for capacity estimates.

    Noise values are stored in SI measurement units at the reference bandwidth
    and reference sensor count.  ``compute_detector_noise_std`` scales them
    to arbitrary (N, bandwidth) combinations.
    """

    canonical_name: str
    noise_source: str
    measurement_units: str
    reference_sensor_count: int
    sensor_count_noise_exponent: float
    reference_bandwidth_hz: float

    # Noise std per sensor in *measurement_units* at reference conditions
    today_best_noise: float
    physical_floor_noise: float

    # Typical physiological source amplitude (sets the signal level for physics-based SNR)
    source_amplitude: float
    source_amplitude_units: str

    # Empirically observed signal amplitude in *measurement_units*.
    # Used to compute SNR from actual measurements rather than from an idealised
    # single-dipole source amplitude.  SNR_empirical = typical_signal / detector_noise.
    typical_signal_amplitude: float = 0.0
    typical_signal_notes: str = ""

    # Legacy field kept for backward compatibility with get_bitrate_channel_capacity
    reference_total_snr: float = 100.0

    notes: str = ""


# ---------------------------------------------------------------------------
# Per-modality noise models
# ---------------------------------------------------------------------------

# EEG: Johnson noise  V_n = sqrt(4 k_B T R Δf)
# R_ref = 5 kΩ at N_ref = 256 electrodes, BW_ref = 100 Hz
# noise = sqrt(4 * 1.38e-23 * 310 * 5000 * 100) ≈ 9.25e-8 V
_EEG_NOISE_TODAY = math.sqrt(4 * K_B * BODY_TEMP_K * 5_000 * 100)  # ~9.25e-8 V

# MEG SQUID: 5 fT/√Hz today, 0.1 fT/√Hz fundamental, BW_ref = 100 Hz
_MEG_SQUID_NOISE_TODAY = 5e-15 * math.sqrt(100)  # 5e-14 T
_MEG_SQUID_NOISE_FUND = 0.1e-15 * math.sqrt(100)  # 1e-15 T

# MEG OPM: 15 fT/√Hz today, 0.5 fT/√Hz fundamental (spin projection noise, 1 cm³ SERF cell).
# The body thermal floor (0.1 fT/√Hz) is NOT the binding limit for wearable OPMs:
# spin projection noise δB ∝ 1/√(N_atoms·T) gives ~0.5 fT/√Hz for a 1 cm³ cell (~10¹⁴ atoms).
# Reaching body thermal would require ~40 cm³ cells — incompatible with wearable use.
# SQUIDs ARE body-thermal-limited (quantum limit ~0.0001 fT/√Hz is far below body thermal).
_MEG_OPM_NOISE_TODAY = 15e-15 * math.sqrt(100)  # 1.5e-13 T
_MEG_OPM_NOISE_FUND = 0.5e-15 * math.sqrt(100)  # 5e-15 T  (spin projection, 1 cm³ cell)

# fNIRS CW shot noise (dimensionless relative intensity noise)
# Source power 5 mW (today), λ = 830 nm, OD ≈ 4
# Photon energy = hc/λ = 2.394e-19 J
# Φ_total = P / E_photon * 10^(-OD)
# Per detector at N_ref = 800: Φ_det = Φ_total / N_ref
# σ = 1/√(Φ_det / BW)  (dimensionless relative intensity noise)
_FNIRS_PHOTON_ENERGY = H_PLANCK * C_LIGHT / 830e-9  # J per photon at 830 nm
_FNIRS_SOURCE_POWER_TODAY = 5e-3  # W (typical commercial fNIRS)
_FNIRS_OD = 4  # optical density ~4 at 30 mm separation

# ANSI Z136.1 skin MPE at 830nm for CW exposure >10s:
#   MPE = 0.2 × C_A  W/cm²,  C_A = 10^(0.002(λ-700)) = 10^0.26 ≈ 1.82
#   MPE ≈ 364 mW/cm²
# With 3.5 mm ANSI limiting aperture (area = π(0.175)² ≈ 0.096 cm²):
#   P_max = 364 × 0.096 ≈ 35 mW
_FNIRS_ANSI_CA = 10 ** (0.002 * (830 - 700))  # ~1.82
_FNIRS_ANSI_MPE = 0.2 * _FNIRS_ANSI_CA  # W/cm²  (~0.364)
_FNIRS_ANSI_APERTURE_CM2 = math.pi * 0.175**2  # 3.5 mm diameter → cm²
_FNIRS_SOURCE_POWER_FUND = _FNIRS_ANSI_MPE * _FNIRS_ANSI_APERTURE_CM2  # ~35 mW

_FNIRS_DETECTED_RATE_TODAY = (
    _FNIRS_SOURCE_POWER_TODAY / _FNIRS_PHOTON_ENERGY * 10 ** (-_FNIRS_OD)
)
_FNIRS_DETECTED_RATE_FUND = (
    _FNIRS_SOURCE_POWER_FUND / _FNIRS_PHOTON_ENERGY * 10 ** (-_FNIRS_OD)
)

_FNIRS_CW_BW_REF = 10.0  # Hz — hemodynamic bandwidth
_FNIRS_CW_N_REF = 800
_FNIRS_CW_NOISE_TODAY = 1.0 / math.sqrt(
    _FNIRS_DETECTED_RATE_TODAY / _FNIRS_CW_N_REF / _FNIRS_CW_BW_REF
)
_FNIRS_CW_NOISE_FUND = 1.0 / math.sqrt(
    _FNIRS_DETECTED_RATE_FUND / _FNIRS_CW_N_REF / _FNIRS_CW_BW_REF
)

# TD-fNIRS: same physics but time gating retains ~1% of photons
_TD_FNIRS_GATING_FACTOR = 0.01
_TD_FNIRS_N_REF = 400
_TD_FNIRS_NOISE_TODAY = 1.0 / math.sqrt(
    _FNIRS_DETECTED_RATE_TODAY * _TD_FNIRS_GATING_FACTOR / _TD_FNIRS_N_REF / _FNIRS_CW_BW_REF
)
_TD_FNIRS_NOISE_FUND = 1.0 / math.sqrt(
    _FNIRS_DETECTED_RATE_FUND * _TD_FNIRS_GATING_FACTOR / _TD_FNIRS_N_REF / _FNIRS_CW_BW_REF
)

# Ultrasound: noise in forward-model units (dimensionless pressure ratio)
# The forward model maps scatterer reflectivity (dimensionless) to received
# pressure ratio (dimensionless).  Two noise sources:
#
# 1. Acoustic thermal noise (Mellen 1952):
#    NL = -15 + 20·log₁₀(f_kHz)  [dB re 1 µPa²/Hz]
#    At 50 kHz: NL ≈ 19 dB → S_p ≈ 7.9e-11 Pa²/Hz → p_th ≈ 8.9 µPa/√Hz
#    This is thermal pressure fluctuations in the medium itself.
#
# 2. Electronic Johnson noise:
#    V_noise = √(4 k_B T R), R = 50 Ω → 0.93 nV/√Hz
#    Referred to pressure via transducer sensitivity S_rx = 1 mV/Pa:
#    p_elec ≈ 0.93 µPa/√Hz
#
# Referred to forward-model units: noise_fwd = p_total / P_tx
#   where P_tx ≈ 10 kPa (transmit pressure at brain depth)
#
# The noise model is *frequency-aware*:
#   - Acoustic thermal noise (Mellen 1952) scales as f² in PSD (f in amplitude)
#   - Aperture directivity: for ka > 1 the element spatially filters isotropic
#     thermal noise, reducing effective noise by ~1/ka in amplitude.
#     ka = 2πf·a/c, where a = element radius = √(scalp_area / (N·π)).
#   - Pulse averaging: PRF = c/(2D) pulse-echoes per second, all seeing the
#     same brain state.  Averaging N_avg = PRF/f_brain reduces noise by √N_avg.
#     The effective noise bandwidth is BW_pulse * f_brain / PRF.
_US_DEFAULT_CENTER_FREQ = 50e3  # Hz  (used when frequency not specified)
_US_TRANSMIT_PRESSURE = 1e4  # Pa  (10 kPa at brain depth after skull)
_US_TRANSDUCER_SENSITIVITY = 1e-3  # V/Pa
_US_SOUND_SPEED = 1540.0  # m/s in soft tissue
_US_BRAIN_DEPTH = 0.150  # m  (max imaging depth, sets PRF)
_US_DEFAULT_F_BRAIN = 1.0  # Hz  (brain-state temporal bandwidth)

# Scalp hemisphere area for element sizing: 2π × R² with R = 92 mm
_US_SCALP_AREA_MM2 = 2 * math.pi * 92**2  # ~53,200 mm²
_US_DEFAULT_N_SENSORS = 6000

# Electronic Johnson noise referred to pressure (frequency-independent)
_US_R_ELEC = 50  # Ω
_US_V_JOHNSON = math.sqrt(4 * K_B * BODY_TEMP_K * _US_R_ELEC)  # V/√Hz
_US_P_ELECTRONIC = _US_V_JOHNSON / _US_TRANSDUCER_SENSITIVITY  # Pa/√Hz


def _us_prf(depth_m: float = _US_BRAIN_DEPTH) -> float:
    """Max pulse repetition frequency limited by round-trip time."""
    return _US_SOUND_SPEED / (2 * depth_m)


def _us_acoustic_thermal_noise(freq_hz: float) -> float:
    """Mellen (1952) acoustic thermal noise spectral density in Pa/sqrt(Hz)."""
    freq_khz = freq_hz / 1e3
    nl_db = -15 + 20 * math.log10(freq_khz)  # dB re 1 µPa²/Hz
    return math.sqrt(10 ** (nl_db / 10)) * 1e-6  # Pa/√Hz


def _us_element_ka(freq_hz: float, n_sensors: int) -> float:
    """Compute ka for a circular piston element tiling the scalp hemisphere."""
    area_mm2 = _US_SCALP_AREA_MM2 / n_sensors
    radius_m = math.sqrt(area_mm2 / math.pi) * 1e-3  # mm → m
    k = 2 * math.pi * freq_hz / _US_SOUND_SPEED
    return k * radius_m


def _us_directivity_factor(ka: float) -> float:
    """Noise reduction factor from aperture directivity.

    For a circular piston receiving isotropic noise:
      ka << 1  →  omnidirectional, factor = 1 (no reduction)
      ka >> 1  →  directional, factor ≈ 1/ka

    We use a smooth interpolation: factor = 1 / sqrt(1 + ka²).
    """
    return 1.0 / math.sqrt(1.0 + ka**2)


def _us_total_noise_fwd(
    freq_hz: float,
    n_sensors: int | None = None,
    f_brain: float = _US_DEFAULT_F_BRAIN,
) -> float:
    """Total US noise in forward-model units per brain-state sample.

    Combines acoustic thermal (frequency-dependent, aperture-filtered) and
    electronic Johnson (frequency-independent) noise in RSS, then refers to
    fwd-model units by dividing by transmit pressure.

    The noise bandwidth accounts for pulse averaging: each brain-state sample
    averages N_avg = PRF/f_brain pulse-echoes, so the effective bandwidth is
    BW_pulse * f_brain / PRF.  This ensures consistency with the capacity
    formula C = (1/2T) Σ log₂(1 + (σ/noise)²) where T = 1/f_brain.
    """
    p_thermal = _us_acoustic_thermal_noise(freq_hz)

    # Aperture directivity reduces thermal noise (but not electronic noise)
    n = n_sensors if n_sensors is not None else _US_DEFAULT_N_SENSORS
    ka = _us_element_ka(freq_hz, n)
    dir_factor = _us_directivity_factor(ka)
    p_thermal_eff = p_thermal * dir_factor

    p_total = math.sqrt(p_thermal_eff**2 + _US_P_ELECTRONIC**2)

    # Effective bandwidth after pulse averaging:
    # BW_pulse = freq_hz (100% fractional BW)
    # PRF = c/(2D) ≈ 5133 Hz
    # N_avg = PRF / f_brain
    # BW_eff = BW_pulse / N_avg = BW_pulse * f_brain / PRF
    prf = _us_prf()
    bw_pulse = freq_hz
    bw_eff = bw_pulse * f_brain / prf

    return p_total / _US_TRANSMIT_PRESSURE * math.sqrt(bw_eff)


# Default noise at 50 kHz (for the NoiseModel entry)
_US_NOISE_FWD_50K = _us_total_noise_fwd(_US_DEFAULT_CENTER_FREQ, _US_DEFAULT_N_SENSORS)

# Neural current dipole amplitude (same for EEG, MEG)
_NEURAL_DIPOLE = 10e-9  # 10 nA·m

# EEG forward model uses mm coordinates (OpenMEEG BEM), so the dipole
# "unit" in the lead field is A·mm, not A·m.
# 10 nA·m = 10e-9 A·m × 1000 mm/m = 10e-6 A·mm
_EEG_DIPOLE_MM = _NEURAL_DIPOLE * 1e3  # 10e-6 A·mm

# Typical brain acoustic reflectivity  ΔZ/Z ≈ 1%
_US_REFLECTIVITY = 0.01  # dimensionless

# fNIRS absorption change
_FNIRS_DELTA_MUA = 0.002  # mm⁻¹

# fMRI reconstructed-BOLD model.
#
# We model the measurement as fractional BOLD signal.  A typical task-evoked
# BOLD contrast is ~1%, and temporal SNR for whole-brain 3 mm voxels at 3T is
# often O(50-100).  The "fundamental" tier here should be read as a high-quality
# physiological-noise-limited reference, not as a thermodynamic MRI limit.
_FMRI_REF_VOXEL_SIZE_MM = 3.0
_FMRI_REF_TR_S = 2.0
_FMRI_BOLD_CONTRAST = 0.01
_FMRI_TODAY_TSNR = 80.0
_FMRI_HIGH_QUALITY_TSNR = 200.0
_FMRI_TODAY_BOLD_SNR = _FMRI_BOLD_CONTRAST * _FMRI_TODAY_TSNR
_FMRI_HIGH_QUALITY_BOLD_SNR = _FMRI_BOLD_CONTRAST * _FMRI_HIGH_QUALITY_TSNR
_FMRI_TODAY_REL_NOISE = _FMRI_BOLD_CONTRAST / _FMRI_TODAY_BOLD_SNR
_FMRI_HIGH_QUALITY_REL_NOISE = _FMRI_BOLD_CONTRAST / _FMRI_HIGH_QUALITY_BOLD_SNR


def _fmri_relative_noise(
    voxel_size_mm: float | None = None,
    tr_s: float | None = None,
    bold_contrast: float | None = None,
    bold_snr: float | None = None,
    tier: str = "today",
) -> float:
    """Relative BOLD noise for a voxel and TR.

    If ``bold_snr`` is supplied, it directly specifies response SNR:
    ``bold_contrast / noise``.  Otherwise we use a tSNR-derived fallback that
    scales with voxel volume and sqrt(TR).
    """
    contrast = bold_contrast if bold_contrast is not None else _FMRI_BOLD_CONTRAST
    if bold_snr is not None:
        return contrast / bold_snr

    if tier == "fundamental":
        return _FMRI_HIGH_QUALITY_REL_NOISE

    voxel = voxel_size_mm if voxel_size_mm is not None else _FMRI_REF_VOXEL_SIZE_MM
    tr = tr_s if tr_s is not None else _FMRI_REF_TR_S

    thermal_ref = math.sqrt(
        max(_FMRI_TODAY_REL_NOISE**2 - _FMRI_HIGH_QUALITY_REL_NOISE**2, 0.0)
    )
    volume_factor = (_FMRI_REF_VOXEL_SIZE_MM / voxel) ** 3
    tr_factor = math.sqrt(_FMRI_REF_TR_S / tr)
    thermal = thermal_ref * volume_factor * tr_factor
    return math.sqrt(thermal**2 + _FMRI_HIGH_QUALITY_REL_NOISE**2)


NOISE_MODELS = {
    "eeg_openmeeg": NoiseModel(
        canonical_name="eeg_openmeeg",
        noise_source="Johnson noise at the electrode-contact / front-end",
        measurement_units="V",
        reference_sensor_count=256,
        sensor_count_noise_exponent=0.5,
        reference_bandwidth_hz=100.0,
        today_best_noise=_EEG_NOISE_TODAY,
        physical_floor_noise=_EEG_NOISE_TODAY,  # Johnson IS fundamental
        source_amplitude=_EEG_DIPOLE_MM,
        source_amplitude_units="A·mm (OpenMEEG BEM uses mm coordinates)",
        typical_signal_amplitude=5e-6,  # 5 µV: typical evoked potential amplitude at scalp
        typical_signal_notes="5 µV: midpoint of 1–10 µV range for evoked responses (ERPs, SSEPs). "
                              "Spontaneous alpha/beta can be 20–100 µV but those are bulk rhythms, "
                              "not single-source events.",
        reference_total_snr=100.0,
        notes=(
            "R=5kΩ at 256 electrodes.  If electrode area shrinks as 1/N, "
            "contact resistance grows like N → Johnson noise ∝ √N.  "
            "source_amplitude is in A·mm (not A·m) because the OpenMEEG "
            "lead field uses mm geometry."
        ),
    ),
    "meg_opm": NoiseModel(
        canonical_name="meg_opm",
        noise_source="Atomic projection / photon-shot noise in OPMs",
        measurement_units="T",
        reference_sensor_count=1000,
        sensor_count_noise_exponent=0.0,  # OPMs measure field directly
        reference_bandwidth_hz=100.0,
        today_best_noise=_MEG_OPM_NOISE_TODAY,
        physical_floor_noise=_MEG_OPM_NOISE_FUND,
        source_amplitude=_NEURAL_DIPOLE,
        source_amplitude_units="A·m",
        typical_signal_amplitude=100e-15,  # 100 fT: typical evoked MEG response
        typical_signal_notes="100 fT: midpoint of 50–200 fT range for evoked MEG responses. "
                              "Spontaneous alpha/mu rhythms can reach 500–1000 fT but "
                              "evoked single-trial responses are 50–200 fT.",
        reference_total_snr=100.0,
        notes=(
            "OPMs measure field directly — noise is intrinsic to the vapor cell "
            "and does not scale with sensor count.  "
            "Fundamental is spin projection noise for a 1 cm³ SERF cell (~0.5 fT/√Hz), "
            "NOT body thermal (0.1 fT/√Hz): body thermal requires ~40 cm³ cells, "
            "incompatible with wearable use.  SQUIDs are body-thermal-limited; "
            "OPMs are spin-projection-limited at realistic cell sizes."
        ),
    ),
    "meg_squid": NoiseModel(
        canonical_name="meg_squid",
        noise_source="SQUID flux noise mapped to field noise",
        measurement_units="T",
        reference_sensor_count=1000,
        sensor_count_noise_exponent=1.0,
        reference_bandwidth_hz=100.0,
        today_best_noise=_MEG_SQUID_NOISE_TODAY,
        physical_floor_noise=_MEG_SQUID_NOISE_FUND,
        source_amplitude=_NEURAL_DIPOLE,
        source_amplitude_units="A·m",
        typical_signal_amplitude=100e-15,  # 100 fT: typical evoked MEG response
        typical_signal_notes="100 fT: same as OPM — same brain physics, same signal levels. "
                              "SQUIDs sit ~20 mm from scalp vs ~6 mm for OPMs, so absolute "
                              "field amplitudes are somewhat lower, but 100 fT is a good midpoint.",
        reference_total_snr=100.0,
        notes=(
            "Fixed helmet coverage, loop area ∝ 1/N.  Area-independent flux "
            "noise → field noise ∝ N."
        ),
    ),
    "fnirs_analytical_cw": NoiseModel(
        canonical_name="fnirs_analytical_cw",
        noise_source="Shot noise (photon counting)",
        measurement_units="dimensionless (ΔI/I)",
        reference_sensor_count=_FNIRS_CW_N_REF,
        sensor_count_noise_exponent=0.5,
        reference_bandwidth_hz=_FNIRS_CW_BW_REF,
        today_best_noise=_FNIRS_CW_NOISE_TODAY,
        physical_floor_noise=_FNIRS_CW_NOISE_FUND,
        source_amplitude=_FNIRS_DELTA_MUA,
        source_amplitude_units="mm⁻¹ (Δμ_a)",
        typical_signal_amplitude=1e-3,  # 0.1% = 1000 ppm ΔI/I: typical hemodynamic response
        typical_signal_notes="1000 ppm (0.1% ΔI/I): typical hemodynamic response amplitude "
                              "at 30 mm separation. Range is 500–5000 ppm depending on task "
                              "and channel geometry.",
        reference_total_snr=100.0,
        notes=(
            "Today: 5 mW source.  Fundamental: ANSI Z136.1 max ~35 mW "
            "(364 mW/cm² × 3.5 mm aperture) at 830 nm, OD≈4.  "
            "Detector area ∝ 1/N → shot noise ∝ √N."
        ),
    ),
    "td_fnirs_analytical": NoiseModel(
        canonical_name="td_fnirs_analytical",
        noise_source="Shot noise with time-gated photon starvation",
        measurement_units="dimensionless (ΔI/I)",
        reference_sensor_count=_TD_FNIRS_N_REF,
        sensor_count_noise_exponent=0.5,
        reference_bandwidth_hz=_FNIRS_CW_BW_REF,
        today_best_noise=_TD_FNIRS_NOISE_TODAY,
        physical_floor_noise=_TD_FNIRS_NOISE_FUND,
        source_amplitude=_FNIRS_DELTA_MUA,
        source_amplitude_units="mm⁻¹ (Δμ_a)",
        typical_signal_amplitude=1e-3,  # same hemodynamic signal as CW
        typical_signal_notes="Same 1000 ppm hemodynamic response as CW fNIRS — TD measures "
                              "the same signal but with time-resolved photon distributions.",
        reference_total_snr=40.0,
        notes=(
            "Same as CW but time gating retains ~1% of photons.  "
            "Fundamental: ANSI max power ~35 mW at 830 nm."
        ),
    ),
    "fmri_bold": NoiseModel(
        canonical_name="fmri_bold",
        noise_source="Thermal/reconstruction noise plus physiological BOLD fluctuations",
        measurement_units="fractional BOLD signal",
        reference_sensor_count=40_000,
        sensor_count_noise_exponent=0.0,
        reference_bandwidth_hz=1.0 / _FMRI_REF_TR_S,
        today_best_noise=_FMRI_TODAY_REL_NOISE,
        physical_floor_noise=_FMRI_HIGH_QUALITY_REL_NOISE,
        source_amplitude=_FMRI_BOLD_CONTRAST,
        source_amplitude_units="fractional BOLD contrast",
        typical_signal_amplitude=0.0,
        reference_total_snr=_FMRI_TODAY_BOLD_SNR,
        notes=(
            "Reconstructed-BOLD model.  The capacity parameter is BOLD response "
            "SNR, not raw time-series tSNR.  The default fallback derives "
            "response SNR=0.8 from 1% BOLD contrast and tSNR=80 for 3 mm voxels "
            "at TR=2 s; high-quality fallback response SNR=2.0.  This is not a "
            "Bloch-equation scanner simulator."
        ),
    ),
    "us_analytical": NoiseModel(
        canonical_name="us_analytical",
        noise_source="Acoustic thermal (Mellen) + electronic Johnson noise",
        measurement_units="dimensionless (pressure amplitude ratio)",
        reference_sensor_count=6000,
        sensor_count_noise_exponent=0.0,
        reference_bandwidth_hz=_US_DEFAULT_CENTER_FREQ,  # BW = center freq
        today_best_noise=_US_NOISE_FWD_50K,
        physical_floor_noise=_US_NOISE_FWD_50K,  # thermal IS fundamental
        source_amplitude=_US_REFLECTIVITY,
        source_amplitude_units="dimensionless (ΔZ/Z reflectivity)",
        typical_signal_amplitude=1e-3,  # ~0.1% reflectivity change; in fwd-model units noise_eff is already dimensionless
        typical_signal_notes="0.1% acoustic reflectivity change (ΔZ/Z ≈ 0.001). US fwd model is already in "
                              "dimensionless units so noise is already noise_eff; typical_signal here is "
                              "the dimensionless reflectivity contrast expected from brain tissue.",
        reference_total_snr=2000.0,
        notes=(
            "Frequency-aware: use frequency_hz kwarg in compute_noise_effective. "
            "Acoustic thermal noise (Mellen 1952) scales as f²; dominates "
            "electronic Johnson (~0.93 µPa/√Hz) above ~10 kHz.  "
            "Defaults to 50 kHz.  At 2 MHz noise is ~195× higher."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Modality name resolution
# ---------------------------------------------------------------------------

def canonicalize_modality_name(modality_name: str) -> str:
    if modality_name in NOISE_MODELS:
        return modality_name

    # Order matters: check td_fnirs before fnirs
    if modality_name.startswith("td_fnirs_analytical"):
        return "td_fnirs_analytical"

    if modality_name.startswith("us_free_field_analytical"):
        return "us_analytical"

    if modality_name.startswith("fnirs_analytical"):
        return "fnirs_analytical_cw"

    if modality_name.startswith("fmri"):
        return "fmri_bold"

    raise KeyError(f"No noise model registered for modality '{modality_name}'")


def get_noise_model(modality_name: str) -> NoiseModel:
    return NOISE_MODELS[canonicalize_modality_name(modality_name)]


# ---------------------------------------------------------------------------
# Physics-based noise computation
# ---------------------------------------------------------------------------

def compute_detector_noise_std(
    modality_name: str,
    n_sensors: int | None = None,
    bandwidth_hz: float | None = None,
    tier: str = "today",
    frequency_hz: float | None = None,
    voxel_size_mm: float | None = None,
    tr_s: float | None = None,
    bold_contrast: float | None = None,
    bold_snr: float | None = None,
) -> float:
    """
    Noise std per sensor in SI measurement units, scaled from reference
    conditions to the given (n_sensors, bandwidth_hz).

    Parameters
    ----------
    tier : "today" or "fundamental"
    frequency_hz : float, optional
        Center frequency — only used for ultrasound, where acoustic thermal
        noise scales as f² (Mellen 1952).  Defaults to 50 kHz.
    """
    canon = canonicalize_modality_name(modality_name)
    model = NOISE_MODELS[canon]

    n = n_sensors if n_sensors is not None else model.reference_sensor_count

    # Ultrasound: recompute noise from scratch (frequency + aperture aware)
    if canon == "us_analytical":
        freq = frequency_hz if frequency_hz is not None else _US_DEFAULT_CENTER_FREQ
        return _us_total_noise_fwd(freq, n_sensors=n)

    if canon == "fmri_bold":
        return _fmri_relative_noise(
            voxel_size_mm=voxel_size_mm,
            tr_s=tr_s,
            bold_contrast=bold_contrast,
            bold_snr=bold_snr,
            tier=tier,
        )

    base_noise = (
        model.today_best_noise if tier == "today" else model.physical_floor_noise
    )

    bw = bandwidth_hz if bandwidth_hz is not None else model.reference_bandwidth_hz

    # Scale from reference conditions
    bw_factor = math.sqrt(bw / model.reference_bandwidth_hz)
    n_factor = (n / model.reference_sensor_count) ** model.sensor_count_noise_exponent

    return base_noise * bw_factor * n_factor


def compute_noise_effective(
    modality_name: str,
    n_sensors: int | None = None,
    bandwidth_hz: float | None = None,
    tier: str = "today",
    frequency_hz: float | None = None,
    voxel_size_mm: float | None = None,
    tr_s: float | None = None,
    bold_contrast: float | None = None,
    bold_snr: float | None = None,
) -> float:
    """
    Effective noise in forward-model units: detector_noise / source_amplitude.

    This is the value to pass directly to ``get_bitrate(s, noise, ...)``,
    where *s* are the raw (un-normalised) singular values of the forward model.
    The ratio ``s_i / noise_effective`` is then dimensionless.

    Parameters
    ----------
    frequency_hz : float, optional
        Center frequency — only used for ultrasound (Mellen acoustic thermal
        noise scales as f²).  Defaults to 50 kHz.
    """
    canon = canonicalize_modality_name(modality_name)
    model = NOISE_MODELS[canon]
    detector_noise = compute_detector_noise_std(
        modality_name, n_sensors=n_sensors, bandwidth_hz=bandwidth_hz,
        tier=tier, frequency_hz=frequency_hz, voxel_size_mm=voxel_size_mm, tr_s=tr_s,
        bold_contrast=bold_contrast, bold_snr=bold_snr,
    )
    source_amplitude = (
        bold_contrast
        if canon == "fmri_bold" and bold_contrast is not None
        else model.source_amplitude
    )
    return detector_noise / source_amplitude


def capacity_forward_gain_scale(
    modality_name: str,
    params=None,
    voxel_size_mm: float | None = None,
) -> float:
    """Scale raw saved singular values into capacity forward-model units.

    Saved fNIRS transfer matrices are voxel-integrated before SVD, so no hidden
    voxel-volume correction is applied in the SNR/capacity path.  This helper is
    kept as a single hook for future modalities that may need a convention
    conversion at export time.
    """
    return 1.0


def scale_singular_values_for_capacity(
    s: np.ndarray,
    modality_name: str,
    params=None,
    voxel_size_mm: float | None = None,
) -> np.ndarray:
    """Apply modality-specific gain scaling before a capacity calculation."""
    return np.asarray(s) * capacity_forward_gain_scale(
        modality_name,
        params=params,
        voxel_size_mm=voxel_size_mm,
    )


def compute_empirical_snr(
    modality_name: str,
    n_sensors: int | None = None,
    bandwidth_hz: float | None = None,
    tier: str = "today",
    frequency_hz: float | None = None,
    voxel_size_mm: float | None = None,
    tr_s: float | None = None,
    bold_contrast: float | None = None,
    bold_snr: float | None = None,
) -> float:
    """
    SNR derived from empirically observed signal amplitudes.

    Returns typical_signal_amplitude / detector_noise, giving an SNR
    grounded in what instruments actually measure rather than in an
    idealised single-dipole source amplitude.
    """
    canon = canonicalize_modality_name(modality_name)
    model = NOISE_MODELS[canon]
    if model.typical_signal_amplitude == 0.0:
        raise ValueError(f"No typical_signal_amplitude set for '{modality_name}'")
    detector_noise = compute_detector_noise_std(
        modality_name, n_sensors=n_sensors, bandwidth_hz=bandwidth_hz,
        tier=tier, frequency_hz=frequency_hz, voxel_size_mm=voxel_size_mm, tr_s=tr_s,
        bold_contrast=bold_contrast, bold_snr=bold_snr,
    )
    return model.typical_signal_amplitude / detector_noise


def compute_noise_empirical(
    s: np.ndarray,
    modality_name: str,
    n_sensors: int | None = None,
    bandwidth_hz: float | None = None,
    tier: str = "today",
    frequency_hz: float | None = None,
    voxel_size_mm: float | None = None,
    tr_s: float | None = None,
    bold_contrast: float | None = None,
    bold_snr: float | None = None,
) -> float:
    """
    Noise floor in SVD units anchored to empirically observed signal amplitudes.

    Uses noise_floor_from_total_snr(s, SNR_empirical) so that the total
    output SNR equals SNR_empirical = typical_signal / detector_noise.
    This is useful as an observed-SNR diagnostic, but it normalizes away the
    absolute gain of the forward model.  For first-principles detector-floor
    capacity estimates, use compute_noise_effective(...), which preserves raw
    SVD gain through detector_noise / source_amplitude.
    """
    snr = compute_empirical_snr(
        modality_name, n_sensors=n_sensors, bandwidth_hz=bandwidth_hz,
        tier=tier, frequency_hz=frequency_hz, voxel_size_mm=voxel_size_mm, tr_s=tr_s,
        bold_contrast=bold_contrast, bold_snr=bold_snr,
    )
    return noise_floor_from_total_snr(s, snr)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_effective_total_snr(
    modality_name: str,
    n_sensors: int | None = None,
    reference_total_snr: float | None = None,
) -> float:
    model = get_noise_model(modality_name)
    snr = (
        model.reference_total_snr
        if reference_total_snr is None
        else reference_total_snr
    )

    if n_sensors is None:
        return snr

    return snr * (
        model.reference_sensor_count / n_sensors
    ) ** model.sensor_count_noise_exponent


def get_effective_noise_floor(
    s,
    modality_name: str,
    n_sensors: int | None = None,
    reference_total_snr: float | None = None,
) -> float:
    total_snr = get_effective_total_snr(
        modality_name=modality_name,
        n_sensors=n_sensors,
        reference_total_snr=reference_total_snr,
    )
    return noise_floor_from_total_snr(s, total_snr)


def get_bitrate_channel_capacity_for_modality(
    s,
    modality_name: str,
    n_sensors: int | None = None,
    time_resolution: float = 1.0,
    reference_total_snr: float | None = None,
) -> float:
    model = get_noise_model(modality_name)
    snr = (
        model.reference_total_snr
        if reference_total_snr is None
        else reference_total_snr
    )
    return get_bitrate_channel_capacity(
        s=s,
        snr_at_reference_nsensors=snr,
        nsensors_reference=model.reference_sensor_count,
        n_sensors=n_sensors,
        time_resolution=time_resolution,
        sensor_count_snr_exponent=model.sensor_count_noise_exponent,
    )
