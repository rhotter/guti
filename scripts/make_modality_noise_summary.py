#!/usr/bin/env python3
"""Write a compact modality noise/capacity summary table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-guti")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from scripts.plot_modality_convergence import (  # noqa: E402
    SWEEP_SPECS,
    load_records,
    noise_plot_tag,
)
from guti.capacity import (  # noqa: E402
    default_output_frequency_spectrum_kwargs,
    get_bitrate,
    get_capacity,
    total_input_power_from_average_output_power,
)
from guti.noise_models import scale_singular_values_for_capacity  # noqa: E402
from guti.noise_models import get_noise_model  # noqa: E402
from guti.modalities.eeg.calibration import (  # noqa: E402
    anchored_eeg_bitrate,
    anchored_eeg_capacity,
)
from guti.parameters import Parameters  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("results/modality_correlated_noise_summary")
US_RBC_FREQ_HZ = 2_000_000.0
US_RBC_REFERENCE_SVD_FREQ_HZ = 50_000.0
US_RBC_RATE_BANDWIDTH_HZ = 1.0
US_RBC_EXTERNAL_PRESSURE_PA = 1_000_000.0
US_RBC_SKULL_TRANSMISSION_2MHZ = 0.03
US_RBC_BSC_10MHZ_CM_INV_SR_INV = 3e-5
US_RBC_CEREBRAL_BLOOD_VOLUME = 0.03
US_RBC_VOXEL_VOLUME_MM3 = 24.0
US_RBC_RANGE_M = 0.10
US_SOUND_SPEED_M_S = 1540.0
US_BRAIN_DEPTH_M = 0.150
US_BODY_TEMP_K = 310.0
US_K_B = 1.380649e-23
US_TISSUE_DENSITY_KG_M3 = 1000.0
US_ELECTRONIC_RESISTANCE_OHM = 50.0
US_RX_SENSITIVITY_V_PA = 1e-3
US_SCALP_AREA_MM2 = 2.0 * math.pi * 92.0**2
US_N_SENSORS = 6000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"missing numeric field {key!r}")
    return float(value)


def _int(row: dict[str, Any], key: str) -> int:
    return int(_float(row, key))


def _is_kernel(
    kernel: str,
    *,
    covariance_model: str | None = None,
) -> Callable[[dict[str, Any]], bool]:
    def predicate(row: dict[str, Any]) -> bool:
        if row["noise_model_type"] != "spatial_covariance":
            return False
        if str(row.get("noise_correlation_kernel") or "").lower() != kernel:
            return False
        if covariance_model is None:
            return True
        return str(row.get("noise_covariance_model") or "").lower() == covariance_model

    return predicate


def _is_johnson_covariance(row: dict[str, Any]) -> bool:
    if row["noise_model_type"] != "spatial_covariance":
        return False
    johnson_names = {"spherical_johnson", "johnson_volume"}
    kernel = str(row.get("noise_correlation_kernel") or "").lower()
    covariance_model = str(row.get("noise_covariance_model") or "").lower()
    return kernel in johnson_names or covariance_model in johnson_names


def _is_scalar(row: dict[str, Any]) -> bool:
    return row["noise_model_type"] == "scalar_iid" and not row.get(
        "noise_correlation_kernel"
    )


def select_converged_row(
    rows: list[dict[str, Any]],
    *,
    modality: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["modality"] == modality and predicate(row)
    ]
    if not candidates:
        return None

    max_voxels = max(_int(row, "n_voxels") for row in candidates)
    candidates = [row for row in candidates if _int(row, "n_voxels") == max_voxels]
    max_sensors = max(_int(row, "n_sensors") for row in candidates)
    candidates = [row for row in candidates if _int(row, "n_sensors") == max_sensors]
    return max(candidates, key=lambda row: (float(row.get("mtime") or 0.0), row["path"]))


def load_all_records() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for spec in SWEEP_SPECS:
        spec_records, spec_errors = load_records(spec)
        records.extend(spec_records)
        errors.extend(spec_errors)
    return records, errors


def _fmt_number(value: float, *, sigfigs: int = 3) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if 1e-3 <= abs_value < 1e4:
        return f"{value:.{sigfigs}g}"
    return f"{value:.{sigfigs}e}"


def _fmt_rate(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:.3g}"


def _raw_csv_row(
    row: dict[str, Any],
    label: str,
    note: str,
    *,
    output_units: str,
    output_scale: float,
    frequency_spectrum_model: str,
) -> dict[str, Any]:
    return {
        "modality": label,
        "bandwidth_hz": _float(row, "bandwidth_hz"),
        "frequency_spectrum_model": frequency_spectrum_model,
        "output_units": output_units,
        "output_amplitude": _float(row, "output_amplitude") * output_scale,
        "output_noise": _float(row, "output_noise") * output_scale,
        "snr": _float(row, "output_snr"),
        "bitrate_bits_per_s": _float(row, "bitrate_bits_per_s"),
        "capacity_bits_per_s": _float(row, "channel_capacity_bits_per_s"),
        "n_voxels": _int(row, "n_voxels"),
        "n_sensors": _int(row, "n_sensors"),
        "noise_model_type": row["noise_model_type"],
        "noise_correlation_kernel": row.get("noise_correlation_kernel"),
        "noise_covariance_model": row.get("noise_covariance_model"),
        "noise_plot_tag": noise_plot_tag(row),
        "source_path": row["path"],
        "note": note,
    }


def _us_rbc_noise_pressure_pa() -> tuple[float, float]:
    prf_hz = US_SOUND_SPEED_M_S / (2.0 * US_BRAIN_DEPTH_M)
    noise_bandwidth_hz = US_RBC_FREQ_HZ * US_RBC_RATE_BANDWIDTH_HZ / prf_hz
    element_area_m2 = US_SCALP_AREA_MM2 * 1e-6 / US_N_SENSORS
    wavelength_m = US_SOUND_SPEED_M_S / US_RBC_FREQ_HZ
    mode_count = element_area_m2 * 2.0 * math.pi / wavelength_m**2
    acoustic_power_w = mode_count * US_K_B * US_BODY_TEMP_K * noise_bandwidth_hz
    acoustic_pressure_pa = math.sqrt(
        (acoustic_power_w / element_area_m2)
        * US_TISSUE_DENSITY_KG_M3
        * US_SOUND_SPEED_M_S
    )
    electronic_pressure_pa = (
        math.sqrt(
            4.0
            * US_K_B
            * US_BODY_TEMP_K
            * US_ELECTRONIC_RESISTANCE_OHM
            * noise_bandwidth_hz
        )
        / US_RX_SENSITIVITY_V_PA
    )
    return (
        math.sqrt(acoustic_pressure_pa**2 + electronic_pressure_pa**2),
        noise_bandwidth_hz,
    )


def _us_rbc_output_amplitude() -> tuple[float, dict[str, float]]:
    freq_mhz = US_RBC_FREQ_HZ / 1e6
    skull_transmission = US_RBC_SKULL_TRANSMISSION_2MHZ
    bsc_blood_cm = US_RBC_BSC_10MHZ_CM_INV_SR_INV * (freq_mhz / 10.0) ** 4
    bsc_brain_m = (
        bsc_blood_cm
        * US_RBC_CEREBRAL_BLOOD_VOLUME
        * 100.0
    )
    voxel_volume_m3 = US_RBC_VOXEL_VOLUME_MM3 * 1e-9
    amplitude = (
        skull_transmission**2
        * math.sqrt(bsc_brain_m * voxel_volume_m3)
        / US_RBC_RANGE_M
    )
    return amplitude, {
        "skull_transmission": skull_transmission,
        "bsc_blood_cm_inv_sr_inv": bsc_blood_cm,
        "bsc_brain_m_inv_sr_inv": bsc_brain_m,
    }


def _us_rbc_lambda3_row(row: dict[str, Any]) -> dict[str, Any]:
    spectrum_path = Path(str(row["path"]))
    with np.load(spectrum_path, allow_pickle=True) as data:
        singular_values = np.asarray(data["singular_values"], dtype=np.float64)
        params = Parameters.from_dict(data["parameters"].item())

    spec = next(spec for spec in SWEEP_SPECS if spec.name == "us_analytical_50khz")
    spectrum = scale_singular_values_for_capacity(
        singular_values,
        spec.noise_model,
        params=params,
        voxel_size_mm=params.grid_resolution_mm,
    )
    output_amplitude_ratio, signal_metadata = _us_rbc_output_amplitude()
    output_amplitude_pa = output_amplitude_ratio * US_RBC_EXTERNAL_PRESSURE_PA
    noise_pressure_pa, noise_bandwidth_hz = _us_rbc_noise_pressure_pa()
    output_noise_ratio = noise_pressure_pa / US_RBC_EXTERNAL_PRESSURE_PA
    total_input_power = total_input_power_from_average_output_power(
        spectrum,
        average_output_power=output_amplitude_pa**2,
        n_sources=_int(row, "n_sources"),
        n_outputs=_int(row, "n_outputs"),
    )
    lambda3_scale = (US_RBC_FREQ_HZ / US_RBC_REFERENCE_SVD_FREQ_HZ) ** 3
    bitrate = get_bitrate(
        spectrum,
        n_sources=_int(row, "n_sources"),
        total_input_power=total_input_power,
        noise=noise_pressure_pa,
        time_resolution=1.0 / US_RBC_RATE_BANDWIDTH_HZ,
    ) * lambda3_scale
    capacity = get_capacity(
        spectrum[spectrum > 0],
        n_sources=_int(row, "n_sources"),
        total_input_power=total_input_power,
        noise=noise_pressure_pa,
        time_resolution=1.0 / US_RBC_RATE_BANDWIDTH_HZ,
    ) * lambda3_scale
    return {
        "modality": "US 2 MHz RBC",
        "bandwidth_hz": US_RBC_RATE_BANDWIDTH_HZ,
        "frequency_spectrum_model": "none; 1 Hz brain-state band",
        "output_units": "mPa",
        "output_amplitude": output_amplitude_pa * 1e3,
        "output_noise": noise_pressure_pa * 1e3,
        "snr": output_amplitude_pa / noise_pressure_pa,
        "bitrate_bits_per_s": bitrate,
        "capacity_bits_per_s": capacity,
        "n_voxels": _int(row, "n_voxels"),
        "n_sensors": _int(row, "n_sensors"),
        "noise_model_type": "scalar_iid",
        "noise_correlation_kernel": None,
        "noise_covariance_model": None,
        "noise_plot_tag": None,
        "source_path": str(spectrum_path),
        "note": (
            "Uses the previous 50 kHz US SVD spectrum, 2 MHz RBC backscatter "
            "output/noise, 1 Hz bitrate bandwidth, and lambda^3 spatial scaling "
            f"({lambda3_scale:g}x). Noise bandwidth is {noise_bandwidth_hz:.6g} Hz; "
            f"skull transmission is {signal_metadata['skull_transmission']:.6g}; "
            f"absolute pressures are signal={output_amplitude_pa:.6g} Pa and "
            f"noise={noise_pressure_pa:.6g} Pa; "
            f"pressure ratios are signal={output_amplitude_ratio:.6g} and "
            f"noise={output_noise_ratio:.6g} relative to {US_RBC_EXTERNAL_PRESSURE_PA:.6g} Pa."
        ),
    }


def _eeg_anchored_row(row: dict[str, Any]) -> dict[str, Any]:
    model = get_noise_model("eeg_openmeeg")
    output_noise = _float(row, "output_noise")
    output_amplitude = float(model.typical_signal_amplitude)
    output_snr = output_amplitude / output_noise
    bandwidth_hz = _float(row, "bandwidth_hz")
    time_resolution = 1.0 / bandwidth_hz
    spectrum_kwargs = default_output_frequency_spectrum_kwargs("eeg_openmeeg")
    return {
        "modality": "EEG OpenMEEG",
        "bandwidth_hz": bandwidth_hz,
        "frequency_spectrum_model": "power law beta=1.5, 1-100 Hz",
        "output_units": "uV",
        "output_amplitude": output_amplitude * 1e6,
        "output_noise": output_noise * 1e6,
        "snr": output_snr,
        "bitrate_bits_per_s": anchored_eeg_bitrate(
            output_snr,
            time_resolution=time_resolution,
            spectrum_kwargs=spectrum_kwargs,
        ),
        "capacity_bits_per_s": anchored_eeg_capacity(
            output_snr,
            time_resolution=time_resolution,
            spectrum_kwargs=spectrum_kwargs,
        ),
        "n_voxels": _int(row, "n_voxels"),
        "n_sensors": _int(row, "n_sensors"),
        "noise_model_type": row["noise_model_type"],
        "noise_correlation_kernel": row.get("noise_correlation_kernel"),
        "noise_covariance_model": row.get("noise_covariance_model"),
        "noise_plot_tag": noise_plot_tag(row),
        "source_path": row["path"],
        "note": (
            "EEG bitrate/capacity use the empirically anchored lead-field "
            "calibration from guti/modalities/eeg/calibration.py with the "
            "default EEG output power-law spectrum beta=1.5; output amplitude "
            "reports the typical EEG signal amplitude from guti/noise_models.py, "
            "and bitrate/capacity use the displayed output SNR; the saved "
            "Johnson covariance row supplies the detector-noise scale."
        ),
    }


def build_summary(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    selections = [
        {
            "modality": "eeg_openmeeg",
            "label": "EEG OpenMEEG",
            "output_units": "uV",
            "output_scale": 1e6,
            "frequency_spectrum_model": "power law beta=1.5, 1-100 Hz",
            "preferred": _is_johnson_covariance,
            "fallback": None,
            "note": "EEG empirically anchored lead-field calibration with Johnson detector noise.",
            "transform": _eeg_anchored_row,
        },
        {
            "modality": "meg_opm",
            "label": "MEG OPM",
            "output_units": "fT",
            "output_scale": 1e15,
            "frequency_spectrum_model": "power law beta=1.0, 1-100 Hz",
            "preferred": _is_kernel(
                "spherical_johnson",
                covariance_model="spherical_johnson",
            ),
            "fallback": _is_johnson_covariance,
            "note": (
                "MEG uses the upstream EEG spherical Johnson covariance correlation "
                "with the MEG OPM scalar detector-noise diagonal."
            ),
            "fallback_note": (
                "Requested MEG spherical Johnson covariance is not available; this row "
                "uses the latest saved legacy Johnson covariance."
            ),
        },
        {
            "modality": "meg_squid",
            "label": "MEG SQUID",
            "output_units": "fT",
            "output_scale": 1e15,
            "frequency_spectrum_model": "power law beta=1.0, 1-100 Hz",
            "preferred": _is_kernel(
                "spherical_johnson",
                covariance_model="spherical_johnson",
            ),
            "fallback": _is_johnson_covariance,
            "note": (
                "MEG uses the upstream EEG spherical Johnson covariance correlation "
                "with the MEG SQUID scalar detector-noise diagonal."
            ),
            "fallback_note": (
                "Requested MEG spherical Johnson covariance is not available; this row "
                "uses the latest saved legacy Johnson covariance."
            ),
        },
        {
            "modality": "cw_fnirs",
            "label": "fNIRS CW",
            "output_units": "1e-3 rel.",
            "output_scale": 1e3,
            "frequency_spectrum_model": "none; scalar 10 Hz band",
            "preferred": _is_scalar,
            "fallback": None,
            "note": "No saved spatial-covariance fNIRS sweep; scalar detector noise row.",
        },
        {
            "modality": "us_analytical_50khz",
            "label": "US 2 MHz RBC",
            "output_units": "mPa",
            "output_scale": 1e3,
            "frequency_spectrum_model": "none; 1 Hz brain-state band",
            "preferred": _is_scalar,
            "fallback": None,
            "note": (
                "US uses the previous 50 kHz SVD spectrum with 2 MHz RBC "
                "output/noise and lambda^3 spatial scaling."
            ),
            "transform": _us_rbc_lambda3_row,
        },
    ]

    summary_rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for selection in selections:
        row = select_converged_row(
            records,
            modality=selection["modality"],
            predicate=selection["preferred"],
        )
        used_fallback = False
        if row is None and selection["fallback"] is not None:
            row = select_converged_row(
                records,
                modality=selection["modality"],
                predicate=selection["fallback"],
            )
            used_fallback = True
        if row is None:
            notes.append(f"- {selection['label']}: no matching row found.")
            continue
        note = str(selection["fallback_note"] if used_fallback else selection["note"])
        if used_fallback:
            notes.append(f"- {selection['label']}: {note}")
        transform = selection.get("transform")
        if transform is not None:
            summary_rows.append(transform(row))
        else:
            summary_rows.append(
                _raw_csv_row(
                    row,
                    str(selection["label"]),
                    note,
                    output_units=str(selection["output_units"]),
                    output_scale=float(selection["output_scale"]),
                    frequency_spectrum_model=str(selection["frequency_spectrum_model"]),
                )
            )

    return summary_rows, notes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "modality",
        "bandwidth_hz",
        "frequency_spectrum_model",
        "output_units",
        "output_amplitude",
        "output_noise",
        "snr",
        "bitrate_bits_per_s",
        "capacity_bits_per_s",
        "n_voxels",
        "n_sensors",
        "noise_model_type",
        "noise_correlation_kernel",
        "noise_covariance_model",
        "noise_plot_tag",
        "source_path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    notes: list[str],
    *,
    load_errors: list[dict[str, str]],
) -> None:
    meg_johnson_rows = [
        row
        for row in rows
        if row["modality"] in {"MEG OPM", "MEG SQUID"}
        and _is_johnson_covariance(row)
    ]
    if len(meg_johnson_rows) == 2:
        meg_note = (
            "- MEG OPM and MEG SQUID use the same EEG layered spherical Johnson "
            "covariance correlation, with each modality's scalar field-noise "
            "diagonal."
        )
        meg_noise_basis = (
            "Johnson covariance correlation reused from EEG; diagonal set by "
            "the MEG scalar detector noise."
        )
    else:
        meg_note = (
            "- MEG Johnson covariance spectra are not present for every MEG row; "
            "missing rows fall back to the newest saved exponential correlated-noise proxy."
        )
        meg_noise_basis = (
            "Distance-kernel correlated proxy where Johnson covariance rows are missing."
        )

    lines = [
        "# Modality Noise and Capacity Summary",
        "",
        "Rows select the largest available voxel count, then the largest available",
        "sensor count within that voxel count. Bitrate and capacity are recomputed",
        "from saved SVD spectra, except EEG, which uses the merged empirically",
        "anchored lead-field calibration. Neural rows use the output temporal",
        "power-spectrum workflow; EEG uses beta=1.5 over 1--100 Hz by default.",
        "",
        "| Modality | BW Hz | Freq spectrum model | Output unit | Output amp | Output noise | SNR | Bit-rate | Capacity |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {modality} | {bw} | {freq_model} | {unit} | {amp} | {noise} | {snr} | {bitrate} | {capacity} |".format(
                modality=row["modality"],
                bw=_fmt_number(float(row["bandwidth_hz"])),
                freq_model=row["frequency_spectrum_model"],
                unit=row["output_units"],
                amp=_fmt_number(float(row["output_amplitude"])),
                noise=_fmt_number(float(row["output_noise"])),
                snr=_fmt_number(float(row["snr"])),
                bitrate=_fmt_rate(float(row["bitrate_bits_per_s"])),
                capacity=_fmt_rate(float(row["capacity_bits_per_s"])),
            )
        )

    lines.extend(
        [
            "",
            "## Selection Notes",
            "",
            "- EEG uses the empirically anchored EEG calibration from the merged EEG fix, with output power-law beta=1.5 over 1--100 Hz; output amplitude reports the typical EEG signal amplitude, and the saved Johnson covariance row supplies the detector-noise scale.",
            meg_note + " MEG uses the default neural output power-law beta=1.0 over 1--100 Hz.",
            "- fNIRS does not currently have a saved spatial covariance sweep, so its row uses the scalar detector-noise spectrum.",
            "- US uses the previous 50 kHz SVD spectrum, 2 MHz RBC output/noise, `1 Hz` bitrate bandwidth, and lambda-cubed spatial scaling.",
            "- No fMRI convergence SVD files are present under `results/variants`, so fMRI is not included.",
        ]
    )
    if notes:
        lines.extend(["", "Additional missing-data notes:", *notes])

    lines.extend(
        [
            "",
            "## Calculation",
            "",
            "The table reports the per-output signal amplitude and detector noise from",
            "`guti/noise_models.py`, then uses the selected saved SVD spectrum to compute",
            "bitrate and water-filled capacity.",
            "Output amplitude/noise are displayed in modality-specific scaled units",
            "to keep the numbers readable; SNR and rates are computed before display",
            "scaling.",
            "For rows with an output temporal spectrum, total signal power is",
            "distributed across frequency bins and the per-bin bitrates/capacities",
            "are summed.",
            "",
            "## Output Amplitude and Noise References",
            "",
            "| Modality | Output amplitude basis | Output noise basis |",
            "| --- | --- | --- |",
            "| EEG OpenMEEG | `5 uV` typical evoked EEG signal amplitude from `guti/noise_models.py`; displayed SNR is this typical amplitude divided by detector noise, and bitrate/capacity use that same SNR through the anchored EEG mode-gain calculation with output power-law beta=1.5 over 1--100 Hz. | Johnson-Nyquist electrode/front-end noise with `R=5 kOhm`, `T=310 K`, `B=100 Hz`; Johnson covariance uses layered spherical EEG impedance for correlation. |",
            f"| MEG OPM | `100 fT` typical evoked MEG field amplitude | `15 fT/sqrt(Hz)` OPM field noise integrated over `B=100 Hz`; {meg_noise_basis} |",
            f"| MEG SQUID | `100 fT` typical evoked MEG field amplitude | `5 fT/sqrt(Hz)` SQUID field noise integrated over `B=100 Hz`; {meg_noise_basis} |",
            "| fNIRS CW | `0.001` relative-intensity hemodynamic response (`1000 ppm`) | Photon shot noise from `P=5 mW`, `lambda=830 nm`, `OD=4`, divided over channels and bandwidth. |",
            "| US 2 MHz RBC | RBC volume-backscatter echo pressure from `1 MPa` external pressure, `T_skull=0.03`, `CBV=3%`, `V=24 mm^3`, and `r=10 cm`; displayed in mPa. | 2 MHz acoustic/electronic receiver noise, displayed in mPa; pulse-averaged noise bandwidth is distinct from the `1 Hz` bitrate bandwidth. |",
            "",
            "Output amplitude is the square root of the average per-output signal power:",
            "",
            "$$",
            "A_{out} = \\sqrt{P_{out}} = a_{typical}.",
            "$$",
            "",
            "The displayed SNR is the scalar per-output amplitude ratio:",
            "",
            "$$",
            "\\mathrm{SNR}_{out} = \\frac{A_{out}}{\\sigma_{out}}.",
            "$$",
            "",
            "For scalar detector-noise models, the default scaling is:",
            "",
            "$$",
            "\\sigma_{out}(N,B) = \\sigma_{ref}\\sqrt{\\frac{B}{B_{ref}}}",
            "\\left(\\frac{N}{N_{ref}}\\right)^\\alpha.",
            "$$",
            "",
            "For covariance-whitened spectra, bitrate/capacity use the singular values",
            "of the whitened operator:",
            "",
            "$$",
            "\\tilde{s}_i = \\mathrm{svd}\\left(K_N^{-1/2}H\\right)_i.",
            "$$",
            "",
            "Equal-input-power bitrate is:",
            "",
            "$$",
            "R = \\frac{1}{2T}\\sum_i \\log_2\\left(1 + \\tilde{s}_i^2",
            "\\frac{P_{in,total}}{n_{sources}}\\right), \\qquad T=1/B.",
            "$$",
            "",
            "Capacity uses the same whitened gains but water-fills the total input power",
            "across modes.",
            "",
            "## Modality Noise Formulas",
            "",
            "### EEG",
            "",
            "The scalar diagonal is Johnson-Nyquist electrode/front-end noise:",
            "",
            "$$",
            "\\sigma_{EEG,ref}=\\sqrt{4k_B T R B_{ref}},",
            "\\quad R=5\\,\\mathrm{k}\\Omega,\\quad B_{ref}=100\\,\\mathrm{Hz}.",
            "$$",
            "",
            "The Johnson covariance run normalizes the layered spherical impedance",
            "matrix to a correlation matrix and then restores the scalar diagonal:",
            "",
            "$$",
            "K_J = 4k_BTB_J\\operatorname{Re}_H Z,\\quad",
            "C_J = D_J^{-1/2}K_JD_J^{-1/2},\\quad",
            "K_N = D_\\sigma C_J D_\\sigma.",
            "$$",
            "",
            "Here `Z` is built from `guti/modalities/eeg/scalp_resistance.py`,",
            "`B_J=100 Hz`, electrode area is `1 cm^2`, and `lmax=2000`.",
            "",
            "For bitrate/capacity, EEG uses the merged empirically anchored",
            "calibration instead of trusting the absolute OpenMEEG/BEM gain. The",
            "cached lead-field shape is cleaned with a boundary margin and scaled so",
            "a reference source has the displayed single-channel amplitude SNR:",
            "",
            "$$",
            "g_i = \\mathrm{SNR}_{ref}\\frac{\\sigma_i(A_{clean})}{p_{ref}},",
            "\\qquad \\mathrm{SNR}_{ref}=A_{out}/\\sigma_{out}.",
            "$$",
            "",
            "The table's EEG bitrate is the equal-power sum over `g_i`; EEG capacity",
            "water-fills the same anchored mode gains. In this summary those gains",
            "are evaluated through the output temporal-spectrum workflow with:",
            "",
            "$$",
            "S_{out}(f) \\propto f^{-1.5},\\qquad 1\\le f\\le 100\\,\\mathrm{Hz}.",
            "$$",
            "",
            "### MEG",
            "",
            "The current scalar MEG detector noises are field sensitivities integrated",
            "over bandwidth:",
            "",
            "$$",
            "\\sigma_{OPM}=15\\,\\mathrm{fT}/\\sqrt{\\mathrm{Hz}}\\sqrt{B},\\qquad",
            "\\sigma_{SQUID}=5\\,\\mathrm{fT}/\\sqrt{\\mathrm{Hz}}\\sqrt{B}.",
            "$$",
            "",
            "For this table, MEG uses the same Johnson covariance correlation as",
            "EEG, but restores the diagonal with the MEG scalar detector noise:",
            "",
            "$$",
            "K_{N,MEG}=D_{\\sigma,MEG} C_J D_{\\sigma,MEG},\\qquad",
            "K_{out}=K_{N,MEG}\\otimes I_3.",
            "$$",
            "",
            "The `I_3` expansion matches the 3 magnetic-field components saved per",
            "MEG sensor in the Sarvas sweep.",
            "",
            "### fNIRS CW",
            "",
            "CW fNIRS noise is photon shot noise on relative intensity:",
            "",
            "$$",
            "E_\\gamma = \\frac{hc}{\\lambda},\\quad",
            "\\Phi = \\frac{P}{E_\\gamma}10^{-OD},\\quad",
            "\\sigma_{fNIRS}=\\sqrt{\\frac{NB}{\\Phi}}.",
            "$$",
            "",
            "The implemented defaults are `P=5 mW`, `lambda=830 nm`, `OD=4`,",
            "`N_ref=800`, and `B_ref=10 Hz`.",
            "",
            "### Ultrasound",
            "",
            "The ultrasound row keeps the previous 50 kHz SVD spectrum but replaces",
            "the signal and noise with a 2 MHz RBC backscatter estimate. The table",
            "uses `B_rate = f_brain = 1 Hz` for bitrate. Receiver noise uses the",
            "pulse-averaged noise bandwidth:",
            "",
            "$$",
            "B_{noise}=f_0\\frac{f_{brain}}{PRF},\\qquad PRF=\\frac{c}{2D}.",
            "$$",
            "",
            "For the table, `f_0=2 MHz`, so `B_noise=389.6 Hz` while",
            "`B_rate=1 Hz`.",
            "",
            "RBC output amplitude is estimated from volume backscatter. The",
            "dimensionless pressure transfer ratio is:",
            "",
            "$$",
            "a_{US}=T_{skull}^2\\frac{\\sqrt{\\eta V}}{r},\\qquad",
            "\\eta=CBV\\cdot BSC_{blood},\\qquad",
            "BSC_{blood}(f)=BSC_{10MHz}\\left(\\frac{f}{10MHz}\\right)^4.",
            "$$",
            "",
            "The pressure reported in the table is:",
            "",
            "$$",
            "p_{echo}=P_{external}a_{US}.",
            "$$",
            "",
            "The assumptions are `T_skull=0.03`, `BSC_10MHz=3e-5 cm^-1 sr^-1`,",
            "`CBV=3%`, `V=24 mm^3`, and `r=10 cm`.",
            "",
            "The physical interpretation is: the backscatter coefficient gives a",
            "differential scattered intensity fraction per unit volume and steradian.",
            "For an order-of-magnitude per-voxel echo, the intensity ratio from one",
            "voxel scales like `eta V / r^2`; pressure amplitude is the square root",
            "of intensity, so the received/transmitted pressure ratio scales like",
            "`sqrt(eta V) / r`. The `T_skull^2` factor applies one skull pass on",
            "transmit and one on receive.",
            "",
            "Numerically:",
            "",
            "$$",
            "BSC_{blood}(2MHz)=3\\times10^{-5}\\left(\\frac{2}{10}\\right)^4",
            "=4.8\\times10^{-8}\\,\\mathrm{cm}^{-1}\\mathrm{sr}^{-1}.",
            "$$",
            "",
            "$$",
            "\\eta=0.03\\,BSC_{blood}=1.44\\times10^{-9}\\,\\mathrm{cm}^{-1}",
            "\\mathrm{sr}^{-1}=1.44\\times10^{-7}\\,\\mathrm{m}^{-1}",
            "\\mathrm{sr}^{-1}.",
            "$$",
            "",
            "$$",
            "\\frac{\\sqrt{\\eta V}}{r}",
            "=\\frac{\\sqrt{(1.44\\times10^{-7})(24\\times10^{-9})}}{0.10}",
            "=5.88\\times10^{-7}.",
            "$$",
            "",
            "$$",
            "a_{US}=0.03^2\\times5.88\\times10^{-7}=5.29\\times10^{-10}.",
            "$$",
            "",
            "For `P_external=1 MPa`, this gives:",
            "",
            "$$",
            "p_{echo}=10^6\\,\\mathrm{Pa}\\times5.29\\times10^{-10}",
            "=5.29\\times10^{-4}\\,\\mathrm{Pa}=0.529\\,\\mathrm{mPa}.",
            "$$",
            "",
            "This is intentionally a simple backscatter estimate. It does not include",
            "array focusing gain, coherent summation across multiple resolution cells,",
            "or a detailed RBC form-factor model; it is the per-voxel pressure echo",
            "implied by the assumed volume backscatter coefficient.",
            "",
            "Acoustic thermal modal power and electronic Johnson pressure noise are",
            "combined in root-sum-square:",
            "",
            "$$",
            "P_{n,ac}=A_{elem}\\frac{2\\pi}{\\lambda^2}k_BTB_{noise},\\quad",
            "p_{ac}=\\sqrt{\\frac{P_{n,ac}}{A_{elem}}\\rho c},",
            "$$",
            "",
            "$$",
            "p_{elec}=\\frac{\\sqrt{4k_BTRB_{noise}}}{S_{rx}},\\quad",
            "p_{n,US}=\\sqrt{p_{ac}^2+p_{elec}^2}.",
            "$$",
            "",
            "The equivalent normalized noise used internally is",
            "`p_n,US / P_external`; reporting pressures or ratios gives the same",
            "SNR when both signal and noise use the same convention.",
            "",
            "Finally, bitrate and capacity from the 50 kHz SVD row are scaled by",
            "the assumed spatial-mode growth:",
            "",
            "$$",
            "\\left(\\frac{\\lambda_{50kHz}}{\\lambda_{2MHz}}\\right)^3",
            "=\\left(\\frac{2MHz}{50kHz}\\right)^3=64000.",
            "$$",
            "",
            "## References",
            "",
            "- Code implementation: `guti/noise_models.py`, `guti/capacity.py`, and `scripts/plot_modality_convergence.py`.",
            "- Johnson-Nyquist noise: `4 k_B T R B`, used for EEG and US electronics.",
            "- Shot-noise model: Poisson photon counting, `sigma = 1/sqrt(n_photons)`.",
            "- Acoustic thermal mode-count model: modal thermal power `k_B T B` per accepted acoustic mode.",
            "",
            "## Data",
            "",
            "- [summary.csv](summary.csv)",
            "- [summary.json](summary.json)",
        ]
    )
    if load_errors:
        lines.append(f"- Load errors: {len(load_errors)} records could not be read.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records, errors = load_all_records()
    rows, notes = build_summary(records)
    write_csv(args.outdir / "summary.csv", rows)
    (args.outdir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.outdir / "README.md", rows, notes, load_errors=errors)
    print(f"Wrote {len(rows)} rows to {args.outdir}")
    if errors:
        print(f"Skipped {len(errors)} load errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
