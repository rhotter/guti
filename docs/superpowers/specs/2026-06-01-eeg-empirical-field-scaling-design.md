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
  SNR 1 → ~1 kbit/s, ~80× below the sibling repo `~/dev/guti2` (80 kbit/s) which
  anchors EEG to a literature single-channel SNR instead of trusting the BEM gain.

MEG does not have this problem: the Sarvas field gives a trustworthy absolute
T/(A·m), so MEG is left untouched.

## Approach

Replace the EEG capacity computation with a guti2-style **empirically anchored**
path, computed from the raw lead field at export time:

1. **Exclude near-boundary voxels** before the SVD, removing the BEM blow-ups.
2. **Anchor** the source amplitude so a canonical, fixed-depth cortical source
   produces a literature single-channel amplitude SNR. The SVD then spreads that
   budget across spatial modes (top modes reach SNR ≫ the single-channel value).

Anchoring uses only the *shape* of the lead field (the robust part of the
multilayer model), not its absolute scale.

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
- `anchored_mode_snr(A_clean, pos_clean, snr_ref, ref_depth_mm=20.0) -> np.ndarray`
  — reference voxel = grid point nearest `ref_depth_mm` below the scalp
  (r = 92 − ref_depth; fixed depth ⇒ grid-invariant, per guti2);
  `peak_ref = max_sensor |A_clean[:, ref_col]|`; returns per-mode **amplitude SNR**
  `snr_ref · σ / peak_ref` where `σ = svd(A_clean)`.
- `anchored_eeg_bitrate(snr_ref, margin_mm=4.0, ref_depth_mm=20.0,
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

## Expected results (margin 4 mm, prefactor 50)

- EEG SOTA (snr_ref = 1.0): **≈ 31 kbit/s**
- EEG fundamental (snr_ref = 3.4): **≈ 58 kbit/s**
- Same order as guti2 (80 / 123 kbit/s); the residual ~2× is guti2's 1/f×198
  temporal factor vs our flat prefactor-50 plus grid/sensor-count differences.

## Testing

`tests/test_eeg_calibration.py`:

1. **Artifact removal (spectrum conditioning)** — excluding the 4 mm margin drops
   the top dynamic range `s1/s10` from > 1000 to < 10. This is exclusion's real
   job: a well-conditioned spectrum, hence physical per-mode SNRs.
2. **Anchor depth is a smooth knob, not a pathological jump** — with artifacts
   excluded, `anchored_eeg_bitrate` is monotonic in `ref_depth_mm` and its swing
   over {15,20,25,30} mm is bounded (< 45%), versus the ~4× jump the raw field
   shows when a shallow anchor grabs a boundary blow-up. (`margin_mm` and
   `ref_depth_mm` are legitimate modeling knobs, fixed at 4/20 mm by default;
   the test guards against *instability*, not against this expected dependence.)
3. **Monotonicity** — fundamental (3.4) > today (1.0).
4. **Sanity band** — anchored EEG ∈ [25, 60] kbit/s, i.e. same order as guti2 and
   not the raw-gain ~1 kbit/s.

## Out of scope

MEG/fNIRS/imaging anchoring; changing noise-floor *values* (e.g. SQUID 5→3 fT/√Hz);
regenerating the full web JSON (requires an OpenMEEG run — code path is wired, data
regen is a follow-up).
