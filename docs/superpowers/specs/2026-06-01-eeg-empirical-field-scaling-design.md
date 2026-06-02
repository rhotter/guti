# EEG empirical field scaling — design

**Date:** 2026-06-01
**Status:** approved
**Scope:** EEG only

## Problem

gut-of-imaging's EEG bitrate (`physical_detector_floor` mode) trusts the absolute
OpenMEEG/BEM voltage gain (V per A·mm). That absolute calibration is unreliable:

- The lead field's per-voxel peak response spans a **735,000× dynamic range**. A
  handful of source voxels at the brain surface (r = 80–82 mm, brain radius 80) hit
  a peak of **18.2** versus a median of **1.5e-4** at the same depth — numerical
  BEM blow-ups at the brain/skull boundary, not physics.
- The top singular value (75.6) is **~6000× the 10th** (0.012): the whole spectrum
  is dominated by those few artifact voxels.
- Result: the top SVD mode sits at amplitude SNR ≈ 3 and only ~7 of 256 modes clear
  SNR 1 → ~1 kbit/s. The sibling repo `~/dev/guti2` reports ~80 kbit/s, but that
  turns out to be inflated by its over-smooth analytical sphere; the corrected
  anchored estimate here lands near the original ~1–9 kbit/s (see correction below).

MEG does not have this problem: the Sarvas field gives a trustworthy absolute
T/(A·m), so MEG is left untouched.

## Approach

Replace the EEG capacity computation with an **empirically anchored** path,
computed from the raw lead field at export time:

1. **Exclude near-boundary voxels** before the SVD, removing the BEM blow-ups.
2. **Pin the best spatial mode** to `snr_ref · √N_eff` — the literature
   single-channel SNR times the realistic array gain — and scale the other modes
   by the spectrum shape `σ_i / σ_1`.

Anchoring uses only the *shape* of the lead field (the robust part of the
multilayer model), not its absolute scale.

### Why pin the *top mode*, not a deep reference (correction)

The first version pinned a 20 mm-deep reference voxel to single-channel SNR ≈ 1 and
let the SVD spread it. Because the BEM lead field has a steep depth gradient, the
much-better-coupled superficial modes blew up to amplitude SNR ≈ 218 (power
≈ 47,000 : 1) — impossible for a single EEG snapshot and ~14× over the √N
coherent-combination ceiling. That gave a spuriously high ~31 kbit/s.

Two compounding errors drove it: (a) anchoring the *weakest* (deep) source so
everything shallower inflated, and (b) scalar/IID sensor noise letting the model
claim the full √256 coherent gain even though EEG noise is heavily spatially
correlated. The corrected anchor pins the **best** mode (caps from the top) and uses
the **√N_eff** array gain, with `N_eff ≈ 32` the effective independent-channel count
under correlated EEG noise (EEG's spatial DOF, ~20–40, not the electrode count):

    snr_mode_i = snr_ref · √N_eff · (σ_i / σ_1)

## Components

### 1. `guti/modalities/eeg/calibration.py` (new)

- `load_eeg_leadfield() -> (A, pos)` — raw `A` (n_sensors × 3·N) from the cached
  `guti/modalities/leadfields/eeg/eeg_leadfield.mat` (h5py, v7.3) and the matching
  `core.get_grid_positions(5.0)` source grid. Falls back to
  `eeg.modality.build_eeg_gain` when a leadfield for a different layout is needed.
- `exclude_boundary_voxels(A, pos, margin_mm=4.0) -> (A_clean, pos_clean)` — drop
  sources with `r > BRAIN_RADIUS − margin_mm` (centre = (80, 80, 0)). Default
  4 mm (matches guti2's grid offset); removes all artifacts (`s1/s10` falls from
  ~6000 to ~4). `margin_mm` is a parameter with a 4 mm default.
- `anchored_mode_snr(A_clean, snr_ref, effective_channels=32) -> np.ndarray` —
  `σ = svd(A_clean)`; pins the best mode to `snr_ref · √effective_channels` and
  returns per-mode **amplitude SNR** `snr_ref · √effective_channels · σ / σ_1`. The
  absolute lead-field scale is not used. `effective_channels` (N_eff) is the
  realistic array gain — EEG's spatial DOF under correlated noise, default 32.
- `anchored_eeg_bitrate(snr_ref, margin_mm=4.0, effective_channels=32,
  time_resolution=0.01) -> float` — `get_bitrate(snr_modes, noise=1.0,
  time_resolution)`, reusing `core.get_bitrate` (prefactor 1/(2·tr) = 50, the same
  convention as the other modes).

### 2. `guti/noise_models.py`

Add two fields to `NoiseModel` (defaults `0.0` ⇒ anchoring unused):
`anchor_snr_today`, `anchor_snr_fundamental`.

Set EEG: `anchor_snr_today = 1.0`, `anchor_snr_fundamental = 3.4`
(20 nA·m dipole vs 30 nV/√Hz amplifier → SNR≈1; vs 9 nV/√Hz Johnson → SNR≈3.4).
This is the first time EEG `today ≠ fundamental`.

### 3. `export_svd_json.py`

- Add bitrate mode `empirical_anchored`. For EEG it calls
  `anchored_eeg_bitrate(snr_ref=model.anchor_snr_{today,fundamental})`.
- Make `empirical_anchored` the **default** for EEG (`physical_detector_floor` and
  `empirical_observed_snr` remain available as alternates). Other modalities keep
  `physical_detector_floor`.
- EEG JSON gains `bitrate_anchored_today` / `bitrate_anchored_fundamental` and the
  provenance fields `anchor_snr_today`, `anchor_snr_fundamental`, `margin_mm`,
  `ref_depth_mm`.
- The anchored recompute needs the raw lead field; it uses the cached 256-sensor
  `.mat` (and `build_eeg_gain` for other layouts when available). Variants without
  an available raw lead field are logged and skipped for the anchored mode — never
  silently filled.

## Expected results (margin 4 mm, N_eff = 32, prefactor 50)

- EEG SOTA (snr_ref = 1.0): **≈ 2.0 kbit/s** (top-mode SNR ≈ 5.7, ~10 useful modes)
- EEG fundamental (snr_ref = 3.4): **≈ 6.1 kbit/s** (top-mode SNR ≈ 19, ~30 modes)
- This is in line with the original `physical_detector_floor` estimate (~1–9 kbit/s)
  and with EEG being a low-spatial-resolution modality (~10–40 spatial DOF × 100 Hz
  × modest SNR). guti2's 80 kbit/s is the outlier — its over-smooth analytical
  sphere lets the same anchoring overcount.
- Robust to N_eff ∈ [16, 40]: ~1.4–2.3 kbit/s (today), ~4.6–6.7 kbit/s (fund).

## Testing

`tests/test_eeg_calibration.py`:

1. **Artifact removal (spectrum conditioning)** — excluding the 4 mm margin drops
   the top dynamic range `s1/s10` from > 1000 to < 10.
2. **Per-mode SNR is physical** — the best mode equals `snr_ref · √N_eff` (a few,
   < 10), not the ~218 the old deep-voxel anchor produced; useful modes (SNR > 1)
   number a handful, not ~all 256.
3. **Array gain is monotonic** — more effective channels → more bits.
4. **Monotonicity** — fundamental (3.4) > today (1.0).
5. **Sanity band** — today ∈ [1, 4] kbit/s, fundamental ∈ [3, 9] kbit/s, i.e. the
   low-spatial-resolution regime, in line with the original physical estimate.

## Out of scope

MEG/fNIRS/imaging anchoring; changing noise-floor *values* (e.g. SQUID 5→3 fT/√Hz);
regenerating the full web JSON (requires an OpenMEEG run — code path is wired, data
regen is a follow-up).
