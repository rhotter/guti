"""
Hemodynamic Response Functions (HRFs) and their power spectral densities.

This module plots standard HRF models used in neuroimaging to characterize
the temporal bandwidth of the hemodynamic signal — the key bottleneck for
temporal resolution in fMRI and fNIRS bitrate calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm.first_level import (
    spm_hrf,
    glover_hrf,
    FirstLevelModel,
)
from nilearn.datasets import fetch_localizer_first_level
from nilearn.image import get_data
import pandas as pd


def compute_psd(signal, dt):
    """Compute one-sided power spectral density via FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(signal)
    psd = (np.abs(fft_vals) ** 2) * (dt / n)
    return freqs, psd


def get_canonical_hrf_spectrum(
    f_max: float = 2.0,
    df: float = 0.01,
    hrf_type: str = "spm",
    tr: float = 0.01,
):
    """Return (freqs, |H(f)|) for a canonical HRF, peak-normalized to 1.

    The HRF is sampled at period ``tr`` and zero-padded so that the rfft
    frequency resolution is ``df``. Output is truncated to ``[0, f_max]``.
    """
    from nilearn.glm.first_level import spm_hrf, glover_hrf

    if hrf_type == "spm":
        hrf = spm_hrf(tr, oversampling=1)
    elif hrf_type == "glover":
        hrf = glover_hrf(tr, oversampling=1)
    else:
        raise ValueError(f"Unknown hrf_type: {hrf_type}")

    n_samples = int(round(1.0 / (df * tr)))
    if n_samples < len(hrf):
        # df is too coarse to resolve the HRF — bump n_samples up.
        n_samples = len(hrf)
    padded = np.zeros(n_samples)
    padded[: len(hrf)] = hrf

    freqs = np.fft.rfftfreq(n_samples, d=tr)
    H_mag = np.abs(np.fft.rfft(padded))

    mask = freqs <= f_max
    freqs = freqs[mask]
    H_mag = H_mag[mask]
    H_mag = H_mag / H_mag.max()
    return freqs, H_mag


def get_empirical_hrf():
    """Extract an empirical HRF from real fMRI data via FIR deconvolution."""
    from nilearn.maskers import NiftiSpheresMasker

    data = fetch_localizer_first_level()
    fmri_img = data.epi_img
    t_r = data.t_r  # 2.4s
    events = pd.read_table(data.events)

    # Event-triggered average from auditory cortex ROI
    # MNI coordinates for left primary auditory cortex
    masker = NiftiSpheresMasker(
        seeds=[(-56, -22, 8)], radius=8,
        detrend=True, standardize=True, t_r=t_r,
    )
    timeseries = masker.fit_transform(fmri_img).flatten()

    # Get onsets for auditory events
    audio_events = events[events.trial_type.str.contains("audio")]
    onset_trs = (audio_events.onset.values / t_r).astype(int)

    # Epoch-average around onsets
    window_scans = 13  # ~31s post-stimulus
    epochs = []
    for onset_tr in onset_trs:
        if onset_tr + window_scans <= len(timeseries):
            epoch = timeseries[onset_tr : onset_tr + window_scans]
            epoch = epoch - epoch[0]  # baseline correct
            epochs.append(epoch)

    empirical_hrf = np.mean(epochs, axis=0)
    times = np.arange(window_scans) * t_r
    return times, empirical_hrf


def main():
    tr = 0.01  # 10 ms sampling for smooth curves

    hrfs = {
        "SPM canonical (double gamma)": spm_hrf(tr, oversampling=1),
        "Glover canonical": glover_hrf(tr, oversampling=1),
    }

    # Get empirical HRF
    print("Fetching fMRI data and estimating empirical HRF...")
    emp_times, emp_hrf = get_empirical_hrf()

    # Interpolate empirical HRF to same fine time grid for PSD comparison
    from scipy.interpolate import interp1d
    t_fine = np.arange(0, emp_times[-1], tr)
    emp_interp = interp1d(emp_times, emp_hrf, kind="cubic", fill_value=0, bounds_error=False)
    emp_hrf_fine = emp_interp(t_fine)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    emp_color = colors[2]  # green

    # --- HRF time courses ---
    ax = axes[0]
    for name, hrf in hrfs.items():
        t = np.arange(len(hrf)) * tr
        ax.plot(t, hrf / np.max(np.abs(hrf)), label=name, linewidth=1.5)
    # Plot empirical (normalized to peak=1)
    ax.plot(emp_times, emp_hrf / np.max(np.abs(emp_hrf)), "o-",
            color=emp_color, label="Empirical (FIR, localizer)", linewidth=1.5, markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Standard Hemodynamic Response Functions")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim(0, 32)

    # --- PSD (normalized) with 90%/99% power cutoff ---
    ax = axes[1]

    all_psds = {}
    for name, hrf in hrfs.items():
        freqs, psd = compute_psd(hrf, tr)
        all_psds[name] = (freqs, psd)

    # Empirical PSD from interpolated HRF
    emp_freqs, emp_psd = compute_psd(emp_hrf_fine, tr)
    all_psds["Empirical (FIR, localizer)"] = (emp_freqs, emp_psd)

    plot_colors = [colors[0], colors[1], emp_color]
    for idx, (name, (freqs, psd)) in enumerate(all_psds.items()):
        psd_normalized = psd / psd.max()
        mask = freqs <= 3.0
        c = plot_colors[idx]
        ax.semilogy(freqs[mask], psd_normalized[mask], label=name, linewidth=1.5, color=c)

        # 90% and 99% cumulative power cutoffs
        cumulative = np.cumsum(psd)
        x_offsets = [0.3 + idx * 0.5, 0.3 + idx * 0.5]
        for j, (pct, y_mult) in enumerate([(0.90, 10), (0.99, 5)]):
            ci = np.searchsorted(cumulative, pct * cumulative[-1])
            cf = freqs[ci]
            cp = psd_normalized[ci]
            ax.annotate(
                f"{int(pct*100)}%: {cf:.2f} Hz",
                xy=(cf, cp),
                xytext=(cf + x_offsets[j], cp * y_mult),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
                fontsize=9, color=c, fontweight="bold",
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title("PSD of Hemodynamic Response Functions")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(1e-4, 1.5)

    plt.tight_layout()
    plt.savefig("results/hrf_and_psd.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved to results/hrf_and_psd.png")


if __name__ == "__main__":
    main()
