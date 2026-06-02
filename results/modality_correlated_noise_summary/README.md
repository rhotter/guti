# Modality Noise and Capacity Summary

Rows select the largest available voxel count, then the largest available
sensor count within that voxel count. Bitrate and capacity are computed
from saved SVD spectra or SLQ spectral estimates. EEG uses the merged
empirically anchored lead-field calibration when the cached lead field
is available; otherwise it falls back to the saved Johnson covariance
spectrum row. Neural rows use the output temporal power-spectrum workflow;
EEG uses beta=1.4 over 1--100 Hz
by default.

| Modality | BW Hz | Freq spectrum model | Covariance computation | Output unit | Output amp | Output noise | SNR | Bit-rate | Capacity |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| EEG | 100 | power law beta=1.4, 1-100 Hz | Layered spherical Johnson impedance covariance | uV | 5 | 0.156 | 32.1 | 49.8k | 55.4k |
| MEG OPM | 100 | power law beta=1.7, 1-100 Hz | EEG spherical Johnson correlation + OPM scalar diagonal | fT | 100 | 50 | 2 | 10.3k | 60.9k |
| MEG SQUID | 100 | power law beta=1.7, 1-100 Hz | EEG spherical Johnson correlation + SQUID scalar diagonal | fT | 100 | 10 | 10 | 17k | 63k |
| fNIRS CW | 10 | none; scalar 10 Hz band | Scalar IID diagonal from photon shot noise | 1e-3 rel. | 1 | 0.0619 | 16.2 | 49.3k | 61.3k |
| US 2 MHz RBC cone slice 1% CBV | 1 | none; 1 Hz brain-state band; 2 MHz range slice; 1% CBV variability; lambda^3 spatial scaling | Scalar IID 5 mPa pressure noise with 1% CBV cone-scaled RBC operator | mPa | 0.723 | 5 | 0.145 | 70.8M | 70.8M |

## Selection Notes

- EEG uses the empirically anchored EEG calibration from the merged EEG fix, with output power-law beta=1.4 over 1--100 Hz; output amplitude reports the typical EEG signal amplitude, and the saved Johnson covariance row supplies the detector-noise scale.
- MEG OPM and MEG SQUID use the same EEG layered spherical Johnson covariance correlation, with each modality's scalar field-noise diagonal. MEG uses the default neural output power-law beta=1.7 over 1--100 Hz.
- fNIRS does not currently have a saved spatial covariance sweep, so its row uses the scalar detector-noise spectrum.
- US uses the 128k-source 50 kHz analytical cone-scaled RBC SLQ result, `1 Hz` bitrate bandwidth, a 2 MHz range-slice source-power correction, a `0.01` voxel-to-sensor amplitude factor for 1% CBV variability, and lambda-cubed spatial scaling from 50 kHz to 2 MHz. The capacity column repeats the lambda-cubed-scaled equal-power SLQ bitrate as a lower-bound proxy because water-filled capacity was not recomputed from a full spectrum.
- No fMRI convergence SVD files are present under `results/variants`, so fMRI is not included.

## Calculation

The table reports the per-output signal amplitude and detector noise from
`guti/noise_models.py` or modality-specific result metadata, then uses
the selected saved SVD/SLQ spectrum estimate to compute bitrate and capacity.
Output amplitude/noise are displayed in modality-specific scaled units
to keep the numbers readable; SNR and rates are computed before display
scaling.
For rows with an output temporal spectrum, total signal power is
distributed across frequency bins and the per-bin bitrates/capacities
are summed. Scalar detector-noise values are interpreted as integrated
over the row bandwidth; each frequency bin uses
`sigma_bin = sigma_out sqrt(delta_f / B)`.

## Output Amplitude and Noise References

| Modality | Output amplitude basis | Output noise basis |
| --- | --- | --- |
| EEG | `5 uV` typical evoked EEG signal amplitude from `guti/noise_models.py`; displayed SNR is this typical amplitude divided by detector noise. When the cached empirical lead field is available, bitrate/capacity use that SNR through the anchored EEG mode-gain calculation with output power-law beta=1.4 over 1--100 Hz; otherwise they use the selected saved Johnson covariance spectrum row. | Johnson-Nyquist electrode/front-end noise with `R=5 kOhm`, `T=310 K`, `B=100 Hz`; Johnson covariance uses layered spherical EEG impedance for correlation. |
| MEG OPM | `100 fT` typical evoked MEG field amplitude | `5 fT/sqrt(Hz)` OPM field noise integrated over `B=100 Hz`; Johnson covariance correlation reused from EEG; diagonal set by the MEG scalar detector noise. |
| MEG SQUID | `100 fT` typical evoked MEG field amplitude | `1 fT/sqrt(Hz)` SQUID field noise integrated over `B=100 Hz`; Johnson covariance correlation reused from EEG; diagonal set by the MEG scalar detector noise. |
| fNIRS CW | `0.001` relative-intensity hemodynamic response (`1000 ppm`) | Photon shot noise from `P=5 mW`, `lambda=830 nm`, `OD=4`, divided over channels and bandwidth. |
| US 2 MHz RBC cone slice 1% CBV | RMS output pressure from the 128k-source cone-scaled RBC operator, corrected to one two-cycle 2 MHz axial range slice and multiplied by `0.01` for 1% CBV variability; pair amplitudes use `1 MPa` external pressure, baseline `CBV=3%`, `V=24 mm^3`, `T_skull=0.5` inside the 15 deg +x cone, and `T_skull=0.1` elsewhere. | Fixed scalar pressure noise `5 mPa`, displayed in mPa. |

Output amplitude is the square root of the average per-output signal power:

$$
A_{out} = \sqrt{P_{out}} = a_{typical}.
$$

The displayed SNR is the scalar per-output amplitude ratio:

$$
\mathrm{SNR}_{out} = \frac{A_{out}}{\sigma_{out}}.
$$

For scalar detector-noise models, the default scaling is:

$$
\sigma_{out}(N,B) = \sigma_{ref}\sqrt{\frac{B}{B_{ref}}}
\left(\frac{N}{N_{ref}}\right)^\alpha,\qquad
\sigma_{bin}=\sigma_{out}\sqrt{\frac{\Delta f}{B}}.
$$

For covariance-whitened spectra, bitrate/capacity use the singular values
of the whitened operator:

$$
\tilde{s}_i = \mathrm{svd}\left(K_N^{-1/2}H\right)_i.
$$

Equal-input-power bitrate is:

$$
R = \frac{1}{2T}\sum_i \log_2\left(1 + \tilde{s}_i^2
\frac{P_{in,total}}{n_{sources}}\right), \qquad T=1/B.
$$

Capacity uses the same whitened gains but water-fills the total input power
across modes. The US row is SLQ-only, so its capacity cell is the
equal-power bitrate lower bound rather than a water-filled calculation.

## Modality Noise Formulas

### EEG

The scalar diagonal is Johnson-Nyquist electrode/front-end noise:

$$
\sigma_{EEG,ref}=\sqrt{4k_B T R B_{ref}},
\quad R=5\,\mathrm{k}\Omega,\quad B_{ref}=100\,\mathrm{Hz}.
$$

The Johnson covariance run normalizes the layered spherical impedance
matrix to a correlation matrix and then restores the scalar diagonal:

$$
K_J = 4k_BTB_J\operatorname{Re}_H Z,\quad
C_J = D_J^{-1/2}K_JD_J^{-1/2},\quad
K_N = D_\sigma C_J D_\sigma.
$$

Here `Z` is built from `guti/modalities/eeg/scalp_resistance.py`,
`B_J=100 Hz`, electrode area is `1 cm^2`, and `lmax=2000`.

When the cached lead field is available, EEG uses the merged
empirically anchored calibration instead of trusting the absolute
OpenMEEG/BEM gain. The cached lead-field shape is cleaned with a
boundary margin and scaled so a reference source has the displayed
single-channel amplitude SNR:

$$
g_i = \mathrm{SNR}_{ref}\frac{\sigma_i(A_{clean})}{p_{ref}},
\qquad \mathrm{SNR}_{ref}=A_{out}/\sigma_{out}.
$$

In anchored mode, the table's EEG bitrate is the equal-power sum over
`g_i`; EEG capacity water-fills the same anchored mode gains. If the
cache is unavailable, the generated row falls back to the selected
saved Johnson covariance spectrum's bitrate/capacity. In this summary
the EEG frequency weighting is:

$$
S_{out}(f) \propto f^{-1.4},\qquad 1\le f\le 100\,\mathrm{Hz}.
$$

The displayed EEG noise is the Johnson noise integrated over the full
`100 Hz` band. During the frequency-bin sum, that full-band noise is
converted to each bin as `sigma_bin = sigma_out sqrt(delta_f / 100 Hz)`.

### MEG

The current scalar MEG detector noises are field sensitivities integrated
over bandwidth:

$$
\sigma_{OPM}=5\,\mathrm{fT}/\sqrt{\mathrm{Hz}}\sqrt{B},\qquad
\sigma_{SQUID}=1\,\mathrm{fT}/\sqrt{\mathrm{Hz}}\sqrt{B}.
$$

For this table, MEG uses the same Johnson covariance correlation as
EEG, but restores the diagonal with the MEG scalar detector noise:

$$
K_{N,MEG}=D_{\sigma,MEG} C_J D_{\sigma,MEG},\qquad
K_{out}=K_{N,MEG}\otimes I_3.
$$

The `I_3` expansion matches the 3 magnetic-field components saved per
MEG sensor in the Sarvas sweep.

### fNIRS CW

CW fNIRS noise is photon shot noise on relative intensity:

$$
E_\gamma = \frac{hc}{\lambda},\quad
\Phi = \frac{P}{E_\gamma}10^{-OD},\quad
\sigma_{fNIRS}=\sqrt{\frac{NB}{\Phi}}.
$$

The implemented defaults are `P=5 mW`, `lambda=830 nm`, `OD=4`,
`N_ref=800`, and `B_ref=10 Hz`.

### Ultrasound

The ultrasound row uses the 128k-source analytical 50 kHz free-field
operator, but every source-sensor pair is scaled into 2 MHz RBC
backscatter pressure units with cone-dependent skull transmission.
The table uses `B_rate = f_brain = 1 Hz` for bitrate and `5 mPa`
fixed scalar receiver noise.

#### RBC Backscatter Model

RBC pair amplitude is estimated from volume backscatter. The
dimensionless pressure transfer ratio is:

$$
a_{US}=T_{source}T_{sensor}\frac{\sqrt{\eta V}}{r},\qquad
\eta=CBV\cdot BSC_{blood},\qquad
BSC_{blood}(f)=BSC_{10MHz}\left(\frac{f}{10MHz}\right)^4.
$$

The per-pair pressure contribution is:

$$
p_{echo}=P_{external}a_{US}.
$$

Equivalently, the pressure contribution from one voxel to one sensor is:

$$
p_{ij}=P_{external}\,T_iT_j\frac{\sqrt{CBV\,BSC_{blood}(f)\,V_{voxel}}}{r_{ij}}.
$$

The assumptions are `BSC_10MHz=3e-5 cm^-1 sr^-1`, `CBV=3%`,
`V=24 mm^3`, and `r=10 cm` for the reference pair pressures.
For the cone run, `T_skull=0.5` for sources and sensors inside a
`15 deg` cone around the positive x-axis and `T_skull=0.1` elsewhere.

The physical interpretation is: the backscatter coefficient gives a
differential scattered intensity fraction per unit volume and steradian.
For an order-of-magnitude per-voxel echo, the intensity ratio from one
voxel scales like `eta V / r^2`; pressure amplitude is the square root
of intensity, so the received/transmitted pressure ratio scales like
`sqrt(eta V) / r`. The two transmission factors apply one skull pass
on transmit and one on receive.

The streamed 50 kHz free-field operator already has a `1/r_ij`
distance dependence. The RBC run therefore converts each streamed
chunk to pressure units by multiplying the raw operator by a common
RBC pressure gain and by per-source/per-sensor skull transmissions:

$$
G^{RBC}_{ij}(t)=G^{raw}_{ij}(t)\,g_{RBC}\,T_iT_j.
$$

For this discretization, the metadata reports:

$$
g_{RBC}=214464\,\mathrm{Pa},\qquad
N_{source,in}=2379,\qquad
N_{sensor,in}=102.
$$

The common gain is a conversion from this discretized 50 kHz raw
free-field basis to the 2 MHz RBC pair-pressure model; the physical
pair amplitudes are the `p_ij` values below.

Numerically:

$$
BSC_{blood}(2MHz)=3\times10^{-5}\left(\frac{2}{10}\right)^4
=4.8\times10^{-8}\,\mathrm{cm}^{-1}\mathrm{sr}^{-1}.
$$

$$
\eta=0.03\,BSC_{blood}=1.44\times10^{-9}\,\mathrm{cm}^{-1}
\mathrm{sr}^{-1}=1.44\times10^{-7}\,\mathrm{m}^{-1}
\mathrm{sr}^{-1}.
$$

$$
\frac{\sqrt{\eta V}}{r}
=\frac{\sqrt{(1.44\times10^{-7})(24\times10^{-9})}}{0.10}
=5.88\times10^{-7}.
$$

$$
a_{US}=T_{source}T_{sensor}\times5.88\times10^{-7}.
$$

For `P_external=1 MPa`, this gives the following baseline 10 cm
reference pressures before the 1% CBV-variability factor:

$$
p_{out,out}=5.88\,\mathrm{mPa},\quad
p_{in,out}=29.4\,\mathrm{mPa},\quad
p_{in,in}=147\,\mathrm{mPa}.
$$

The variable component used in the table is `0.01` times these
pair pressures.

This is intentionally a simple backscatter estimate. It does not include
array focusing gain, coherent summation across multiple resolution cells,
or a detailed RBC form-factor model; it is the per-voxel pressure echo
implied by the assumed volume backscatter coefficient.

The full-volume unit-variance output RMS is estimated from the SLQ
Frobenius metadata as:

$$
A_{all}=\sqrt{\|G\|_F^2/n_{outputs}}.
$$

For the selected `6000`-sensor row, the stored SLQ metadata gives:

$$
\|G\|_F^2=372369,\qquad
n_{outputs}=366000.
$$

$$
A_{all}
=\sqrt{372369/366000}
=1.00866\,\mathrm{Pa}.
$$

Only one axial range slice is used for the table's source-power
normalization. For a two-cycle 2 MHz pulse:

$$
f_{slice}=\frac{c(2/f_0)/2}{D}
=\frac{1540(2/2MHz)/2}{0.15}=0.00513.
$$

The displayed US output amplitude is
`A_all sqrt(f_slice) * 0.01`, and the SLQ bitrate is recomputed
with `f_slice * 0.01^2` as the per-source input variance.
Numerically:

$$
A_{table}
=1.00866\sqrt{0.00513333}
\times 0.01
=0.00072268\,\mathrm{Pa}
=0.72268\,\mathrm{mPa}.
$$

The resulting 50 kHz SLQ bitrate is then extrapolated to 2 MHz
by the assumed spatial-mode growth:

$$
R_{2MHz}=R_{50kHz}\left(\frac{2MHz}{50kHz}\right)^3
=64000\,R_{50kHz}.
$$

Exact water-filled capacity was not recomputed for
this SLQ-only run, so the capacity column repeats the equal-power
bitrate after the same lambda-cubed scaling as a lower-bound proxy.

## References

- Code implementation: `guti/noise_models.py`, `guti/capacity.py`, and `scripts/plot_modality_convergence.py`.
- Johnson-Nyquist noise: `4 k_B T R B`, used for EEG.
- Shot-noise model: Poisson photon counting, `sigma = 1/sqrt(n_photons)`.
- Acoustic thermal mode-count model: modal thermal power `k_B T B` per accepted acoustic mode.

## Data

- [summary.csv](summary.csv)
- [summary.json](summary.json)
