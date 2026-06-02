"""Regenerate the cross-modality meta spectrum plot (spectrum.png).

Standalone, headless version of the first plotting cell in results.ipynb: loads
each modality's canonical converged SVD from results/<name>_svd_spectrum.npz and
overlays the s0-normalized spectra (sigma / sigma_0).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from guti.data_utils import load_all_svds

try:
    from labellines import labelLines
    _HAVE_LABELLINES = True
except Exception:
    _HAVE_LABELLINES = False

FILENAMES_TO_LABELS = {
    "eeg_openmeeg": "EEG",
    "fnirs_analytical_cw": "fNIRS (CW)",
    "td_fnirs_analytical": "fNIRS (TD)",
    "us_analytical": "Ultrasound (40 kHz)",
    "meg_opm": "MEG (OPM)",
    "meg_squid": "MEG (SQUID)",
}

all_svds = load_all_svds()

fig, ax = plt.subplots(figsize=(9, 6))
for modality_name, (s, params) in sorted(all_svds.items()):
    # Only plot the canonical labeled modalities. Skips stale/legacy duplicates
    # (e.g. an old `cw_fnirs` default alongside the current `fnirs_analytical_cw`).
    if modality_name not in FILENAMES_TO_LABELS:
        print(f"[skip] {modality_name}: not in canonical modality set")
        continue
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)) or s[0] <= 0:
        print(f"[skip] {modality_name}: non-finite spectrum")
        continue
    ax.plot(
        np.arange(1, len(s) + 1),
        s / s[0],
        label=FILENAMES_TO_LABELS.get(modality_name, modality_name),
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Singular Value Index")
ax.set_ylabel("Singular Value (normalized to $\\sigma_0$)")
ax.set_title("Spectrum of Imaging Modalities")
ax.set_ylim(1e-4, 2)
ax.grid(True, alpha=0.3)
if _HAVE_LABELLINES:
    labelLines(ax.get_lines(), zorder=2.5)
else:
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("spectrum.png", dpi=150)
print("saved spectrum.png")
for m, (s, p) in sorted(all_svds.items()):
    n = None if p is None else getattr(p, "num_sensors", None)
    g = None if p is None else getattr(p, "grid_resolution_mm", None)
    print(f"  {m:24s} n_sv={len(s):6d}  num_sensors={n}  grid_mm={g}")
