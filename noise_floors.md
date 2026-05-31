# Physics-Based Noise Floors

## What changed

Previously, channel capacity was computed using **arbitrary SNR values** (100 for most modalities, 2000 for ultrasound). The noise floor was derived from the SVD spectrum itself via `noise = sqrt(sum(s_i^2)) / total_snr`, making it circular --- the noise depended on the forward model rather than on independent detector physics.

We replaced this with **physics-based noise floors** that derive from:
1. A detector noise model (Johnson noise, shot noise, acoustic thermal noise, etc.) expressed in measurement units (V, T, Pa, etc.)
2. A physiological source amplitude that sets the signal level (e.g., 10 nA·m for a cortical current dipole)

The effective noise, `noise_eff = detector_noise / source_amplitude`, has the same units as the forward model's singular values, making `s_i / noise_eff` dimensionless.

### Bug fixes included

- **2× factor in `get_bitrate_channel_capacity`**: The `1/T` prefactor was missing the factor of 2 from Shannon's formula. Now `1/(2T)`, consistent with `get_bitrate()`.
- **MEG OPM sensor count exponent**: Changed from 0.5 to 0.0. OPMs measure field directly via atomic vapor; noise is intrinsic to the cell and does not scale with sensor count (unlike SQUIDs where coil area shrinks as 1/N).
- **EEG source amplitude units**: OpenMEEG BEM uses mm geometry with S/m conductivity, so the lead field's dipole "unit" is A·mm, not A·m. The source amplitude is 10 nA·m × 1000 mm/m = 10 µA·mm.

---

## Noise physics by modality

### EEG: Johnson (thermal) noise

The fundamental noise in EEG is **Johnson-Nyquist noise** from the electrode-scalp contact impedance:

```
V_n = √(4 k_B T R Δf)
```

- `R` = electrode-scalp contact impedance (1–10 kΩ wet, 10–100 kΩ dry)
- `T` = 310 K (body temperature)
- At R = 5 kΩ, BW = 100 Hz: **V_n = 93 nV**

**Why is this irreducible?** Johnson noise is a direct consequence of the fluctuation-dissipation theorem. The same thermal equilibrium that gives the resistance its dissipative character produces voltage fluctuations. You cannot go below √(4 k_B T R) without cooling the scalp (impractical) or reducing R to zero (impossible for a finite-impedance contact).

**Scaling with electrode count:** If N electrodes tile a fixed scalp area, each electrode area ∝ 1/N, so contact resistance R ∝ N (resistivity/area). Johnson noise ∝ √R ∝ √N. The signal (lead field gain at a point) does not change with electrode size, so per-sensor SNR degrades as 1/√N. But more sensors provide more spatial measurements, so total information can still increase. Exponent = 0.5.

**Amplifier noise:** Modern EEG amplifiers achieve ~8 nV/√Hz input-referred noise, comparable to Johnson noise at 5 kΩ (9.25 nV/√Hz). The system is nearly Johnson-limited with good electrodes.

**Body thermal voltage:** Thermal currents in brain tissue produce voltage noise, but the effective tissue resistance between scalp electrodes (~tens to hundreds of Ω) is far below the contact impedance (kΩ), so this contribution is negligible.

### MEG SQUID: flux noise

A SQUID transduces magnetic flux to voltage. The noise comes from Johnson noise in the Josephson junction shunt resistors (at 4 K operating temperature):

```
S_Φ ~ 16 k_B T_SQUID L / R_shunt   [Wb²/Hz]
```

The SQUID measures flux, not field. A large pickup coil (area A) captures flux Φ = B · A. Since the intrinsic flux noise is independent of pickup area, the equivalent field noise is:

```
B_noise = √S_Φ / A_eff
```

Typical modern SQUIDs: **3–5 fT/√Hz** with ~20 mm diameter pickup coils.

**Scaling with sensor count:** In a fixed-size MEG helmet, if you double the channel count, each pickup coil area halves. Flux noise stays constant → field noise doubles. Noise ∝ N. Exponent = 1.0.

### MEG OPM: atomic spin projection noise

OPMs use alkali vapor (Rb/K) in the spin-exchange relaxation-free (SERF) regime. Two quantum noise sources:

1. **Spin projection noise** — quantum uncertainty in measuring collective atomic spin:
   ```
   δB_SPN = ℏ / (g_F μ_B √(2F)) × √(Γ / (N_atoms · T_meas))
   ```
2. **Photon shot noise** — quantum fluctuation in the probe beam: ∝ 1/√Φ_probe

Current commercial OPMs: **10–15 fT/√Hz**. Theoretical SERF limit for 1 cm³ cell: ~0.5 fT/√Hz. The gap is from technical noise (laser intensity, field gradients, temperature).

**Scaling with sensor count:** Each OPM has its own independent vapor cell and laser. Adding sensors does not divide a shared resource. Noise is fixed per sensor. Exponent = 0.0.

**Fundamental limit — spin projection noise, not body thermal:** The body thermal floor (0.1 fT/√Hz) is NOT the binding limit for wearable OPMs. The spin projection noise scales as 1/√(N_atoms · T_meas); for a 1 cm³ SERF cell (~10¹⁴ atoms) this gives **~0.5 fT/√Hz** — above body thermal. Reaching 0.1 fT/√Hz would require ~40 cm³ cells, incompatible with scalp-contact use. SQUIDs, by contrast, ARE body-thermal-limited: their quantum noise floor (~0.0001 fT/√Hz) is a million times below body thermal, so body thermal is the actual binding constraint.

### MEG fundamental floor: body thermal magnetic noise

The conducting human head (brain σ ≈ 0.3 S/m) at 310 K produces thermal magnetic fluctuations. Measured by Körber et al. (2019) with ultra-sensitive SQUIDs:

```
B_body ≈ 0.055–0.08 fT/√Hz
```

This is the irreducible thermodynamic floor for any magnetic measurement of the brain. We approximate it as **0.1 fT/√Hz** in the model.

### fNIRS: photon shot noise

fNIRS measures relative intensity changes ΔI/I. The fundamental noise is shot noise from photon counting:

```
σ = 1/√N_photons   (dimensionless)
```

The photon budget:
- Source power: 5 mW (today) or 35 mW (ANSI Z136.1 max at 830 nm, see below)
- Photon energy at 830 nm: 2.39×10⁻¹⁹ J → photon rate ≈ 2.1×10¹⁶ /s per mW
- Attenuation at 30 mm separation: OD ≈ 4 (factor 10⁻⁴)
- Detected rate: ~2.1×10¹² /s (at 5 mW, total across all detectors)
- Per detector at N=800: ~2.6×10⁹ /s
- Integration time (BW=10 Hz): 0.1 s → N_photons ≈ 2.6×10⁸
- Shot noise: σ ≈ 6.2×10⁻⁵ (dimensionless)

**ANSI power limit (fundamental):** ANSI Z136.1 skin MPE at 830 nm for CW exposure >10 s:
```
MPE = 0.2 × C_A  W/cm²,   C_A = 10^(0.002(λ-700)) = 1.82 at 830 nm
MPE ≈ 364 mW/cm²
```
With 3.5 mm ANSI limiting aperture (area 0.096 cm²): **P_max ≈ 35 mW** (7× today).
Shot noise at ANSI max: **σ ≈ 2.3×10⁻⁵** (√7 ≈ 2.65× better).

**Scaling with sensor count:** If total detector area is fixed, each detector gets area ∝ 1/N → photon count ∝ 1/N → shot noise ∝ √N. Exponent = 0.5. However, the forward model singular values also scale as ~√N (more source-detector pairs → more rows in the Jacobian), so per-channel SNR stays approximately constant. Capacity increases with N only through additional spatial channels, with diminishing returns.

### Ultrasound: acoustic thermal modal power + electronic Johnson noise

Two independent receiver noise sources are combined in RSS.

**1. Acoustic thermal modal power:** For one scalp tile, the accepted thermal acoustic power is
```
A_elem = A_head / N
lambda = c / f
P_n = A_elem * (2*pi / lambda^2) * k_B * T * Delta_f
p_thermal = sqrt((P_n / A_elem) * rho * c)
```

This is the thermodynamic acoustic floor written as accepted half-space modes. It replaces the previous Mellen-plus-directivity calculation: do not multiply by a separate aperture directivity factor, because the accepted-mode count is already the spatial-mode normalization. After converting power back to pressure, `A_elem` cancels, so this ideal detector pressure floor is independent of `N` for fixed hemisphere coverage.

**2. Electronic Johnson noise:** `V = sqrt(4 k_B T R)`, `R = 50 ohm` gives 0.93 nV/sqrt(Hz). Referred to pressure via transducer sensitivity `S_rx = 1 mV/Pa`: **p_electronic ≈ 0.93 microPa/sqrt(Hz)**.

Referred to forward-model units:
```
p_total = sqrt(p_thermal^2 + (p_electronic_density * sqrt(BW_eff))^2)
noise_fwd = p_total / P_tx
noise_eff = noise_fwd / (Delta Z/Z)
```

with `P_tx = 10 kPa` and `Delta Z/Z = 0.01`.

**Pulse averaging:** US sends `PRF = c/(2D) ≈ 5,133` pulse-echoes per second at `D = 150 mm`. With brain states changing at `f_brain = 1 Hz`, each brain-state sample averages about 5,133 pulses. The effective bandwidth is `BW_eff = f_center * f_brain / PRF` (9.74 Hz for 50 kHz, 390 Hz for 2 MHz).

The resulting code-generated floors are the same for N=1500 and N=6000 because the modal-power area factor cancels in pressure:

| Frequency | BW_eff | p_thermal RMS | p_electronic RMS | noise_eff (per brain sample) |
|-----------|--------|---------------|------------------|------------------------------|
| 50 kHz | 9.74 Hz | 20.6 microPa | 2.89 microPa | 2.08×10⁻⁷ |
| 500 kHz | 97.4 Hz | 0.652 mPa | 9.13 microPa | 6.52×10⁻⁶ |
| 2 MHz | 390 Hz | 5.22 mPa | 18.3 microPa | 5.22×10⁻⁵ |
| 5 MHz | 974 Hz | 20.6 mPa | 28.9 microPa | 2.06×10⁻⁴ |

With pulse averaging, the effective acoustic thermal pressure grows approximately as `f^(3/2)` because the mode density contributes `f^2` and `BW_eff` contributes another factor of `f` under a square root.

**Skull attenuation** remains the dominant challenge at clinical frequencies: ~15 dB/cm/MHz in bone, giving ~42 dB round trip at 2 MHz through 7 mm skull. This reduces the singular values by 125×. No SVD simulations currently exist above 70 kHz.

**Scaling with sensor count:** Under this modal-power receiver model, thermal pressure noise is independent of element area after the accepted power is converted to pressure. Sensor count changes capacity through the forward matrix, not through a separate detector-noise exponent. Exponent = 0.0.

---

## Noise summary table

All values at reference bandwidth and sensor count.

| Modality | Dominant noise | Today | Fundamental | BW_ref | N_ref | Source amplitude |
|----------|---------------|-------|-------------|--------|-------|-----------------|
| **EEG** | Johnson (R=5 kΩ) | 93 nV | 93 nV | 100 Hz | 256 | 10 µA·mm (= 10 nA·m) |
| **MEG SQUID** | SQUID flux noise | 50 fT (5 fT/√Hz) | 1 fT (0.1 fT/√Hz) | 100 Hz | 1000 | 10 nA·m |
| **MEG OPM** | Atomic spin (spin projection) | 150 fT (15 fT/√Hz) | 5 fT (0.5 fT/√Hz, 1 cm³ cell) | 100 Hz | 1000 | 10 nA·m |
| **fNIRS CW** | Photon shot noise | 62 ppm (5 mW) | 23 ppm (35 mW ANSI) | 10 Hz | 800 | 0.002 mm⁻¹ (Δμ_a) |
| **TD-fNIRS** | Shot + gating | 440 ppm (5 mW) | 165 ppm (35 mW ANSI) | 10 Hz | 400 | 0.002 mm⁻¹ (Δμ_a) |
| **Ultrasound** | Acoustic thermal modal power | 2.08e-9 pressure ratio at 50 kHz | same | 9.74 Hz effective @ 50 kHz | 6000 | 0.01 (ΔZ/Z) |

### Sensor-count noise scaling

| Modality | Exponent | Physical reason |
|----------|----------|----------------|
| EEG | 0.5 | Contact R ∝ N (fixed scalp) → noise ∝ √N |
| MEG SQUID | 1.0 | Coil area ∝ 1/N, flux noise fixed → field noise ∝ N |
| MEG OPM | 0.0 | Each vapor cell is independent |
| fNIRS | 0.5 | Detector area ∝ 1/N → photons ∝ 1/N → shot noise ∝ √N |
| Ultrasound | 0.0 | Accepted modal power ∝ element area, but pressure conversion divides by element area, so the detector pressure floor is N-independent for fixed hemisphere coverage |

---

## Bitrate estimates

Using the largest available sensor configuration per modality. Temporal sampling: 100 Hz for EEG/MEG, 1 Hz for fNIRS and ultrasound.

### Today's technology

| Modality | N | f_samp | noise_eff | SNR₁ | Ch > noise | **bits/s** |
|----------|---|--------|-----------|------|------------|-----------|
| **EEG** | 256 | 100 Hz | 9.25e-3 | 3 | 7 | **1,025** |
| **MEG SQUID** | 1000 | 100 Hz | 5.00e-6 | 1,907 | 221 | **90,208** |
| **MEG OPM** | 1000 | 100 Hz | 1.50e-5 | 1,133 | 491 | **178,741** |
| **fNIRS CW** | 800 | 1 Hz | 3.09e-2 | 0.01 | 0 | **0** |
| **US 50 kHz** | 1500 | 1 Hz | 2.08e-7 | 4,752,941 | 15,044 | **210,984** |
| **US 2 MHz** | 1500 | 1 Hz | 5.22e-5 | 18,971 | 13,819 | **96,224** |
| **US 2 MHz + skull** | 1500 | 1 Hz | 5.22e-5 | 151 | 5,612 | **11,476** |

### Fundamental physics limits

| Modality | N | f_samp | noise_eff | SNR₁ | Ch > noise | **bits/s** | Floor |
|----------|---|--------|-----------|------|------------|-----------|-------|
| **EEG** | 256 | 100 Hz | 9.25e-3 | 3 | 7 | **1,025** | Johnson @ body temp |
| **MEG SQUID** | 1000 | 100 Hz | 1.00e-7 | 95,335 | 494 | **288,700** | Body thermal B-field |
| **MEG OPM** | 1000 | 100 Hz | 5.00e-7 | 33,992 | ~900 | **~200,000** | Spin projection (1 cm³ SERF cell) |
| **fNIRS CW** | 800 | 1 Hz | 1.17e-2 | 0.03 | 0 | **0** | Shot noise @ ANSI max |
| **US 50 kHz** | 1500 | 1 Hz | 2.08e-7 | 4,752,941 | 15,044 | **210,984** | Acoustic thermal |
| **US 2 MHz** | 1500 | 1 Hz | 5.22e-5 | 18,971 | 13,819 | **96,224** | Acoustic thermal |
| **US 2 MHz + skull** | 1500 | 1 Hz | 5.22e-5 | 151 | 5,612 | **11,476** | Acoustic thermal + skull |

US 2 MHz rows use the 50 kHz SVD spectrum (no 2 MHz simulation exists). The "+ skull" row applies 42 dB round-trip attenuation (15 dB/cm/MHz × 2 MHz × 7 mm × 2). Noise uses the modal thermal-power formula and pulse-averaged bandwidth; no separate aperture-directivity multiplier is applied.

---

## Key observations

**EEG** — Today = fundamental. Johnson noise at body temperature is the thermodynamic floor. The 1,025 bits/s is for a 10 nA·m dipole (single cortical column). Larger coherent patches (ERPs, alpha) produce effective dipoles of 1,000–10,000 nA·m, which would give much higher capacity.

**MEG** — Huge headroom between today and fundamental. SQUIDs (5 fT/√Hz) are 50× above the body thermal floor (0.1 fT/√Hz) — body thermal is their true fundamental. OPMs (15 fT/√Hz) are 30× above their actual fundamental: spin projection noise (~0.5 fT/√Hz for a 1 cm³ SERF cell), NOT body thermal. Body thermal (0.1 fT/√Hz) would require ~40 cm³ cells — not wearable. At the spin-projection limit, OPMs retain an advantage from closer scalp proximity and zero sensor-count noise scaling.

**fNIRS** — Zero bits at both tiers. The 7× power increase from ANSI max (35 mW vs 5 mW) only improves shot noise by 2.65×. The fundamental bottleneck is attenuation: OD ≈ 4 at 30 mm means only 1 in 10,000 photons reach the detector. σ₁ = 3.5×10⁻⁴ is 33× below the fundamental noise floor. Reaching SNR₁ = 1 would require ~4.8 kW optical power (138,000× ANSI limit). fNIRS is fundamentally incapable of per-voxel imaging at depth; it measures bulk hemodynamic changes over large volumes.

**Ultrasound** — Today = fundamental (acoustic thermal noise is irreducible at body temperature). At 50 kHz (free-field, no skull): **211k bits/s** with SNR₁ ≈ 4.8M. At 2 MHz (clinical frequency), the modal thermal floor is much higher: `noise_eff` is 250× the 50 kHz value after pulse averaging, giving **96k bits/s** free-field. With skull attenuation (~42 dB round trip at 2 MHz), the same SVD estimate falls to **11.5k bits/s**. That is still above the EEG row in this table, but below the MEG rows.

---

## How to use

```python
from guti.noise_models import compute_noise_effective, compute_detector_noise_std
from guti.core import get_bitrate

# Get noise in measurement units (e.g., Tesla for MEG)
noise_T = compute_detector_noise_std("meg_opm", n_sensors=500, tier="today")

# Get noise in forward-model units (pass directly to get_bitrate with raw s)
noise_eff = compute_noise_effective("meg_opm", n_sensors=500, tier="today")
bitrate = get_bitrate(s_raw, noise_eff, time_resolution=0.01)  # 100 Hz

# Compare today vs fundamental limit
br_today = get_bitrate(s, compute_noise_effective("meg_squid", tier="today"), 0.01)
br_fund  = get_bitrate(s, compute_noise_effective("meg_squid", tier="fundamental"), 0.01)

# Ultrasound at different frequencies (modal thermal power scales through lambda)
noise_50k = compute_noise_effective("us_analytical", n_sensors=1500, frequency_hz=50e3)
noise_2M  = compute_noise_effective("us_analytical", n_sensors=1500, frequency_hz=2e6)
```

## Open questions

1. **Source amplitude choice**: 10 nA·m is a single cortical column. It penalizes EEG and fNIRS, which detect distributed activity. A multi-scale comparison varying source extent would be more informative.
2. **Ultrasound receive normalization**: The current model assumes each receiver tile accepts half-space acoustic modes and does not add a separate directivity correction. This should be revisited only if the forward model explicitly includes a matched receive aperture response.
3. **fNIRS forward model units**: The zero-bitrate result should be validated by verifying the Jacobian has the expected units (mm, mapping mm⁻¹ absorption to dimensionless ΔI/I).
