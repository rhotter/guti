# Output-Power Capacity Workflow

Capacity calculations now use one input convention:

1. Per-output-channel average signal power: `P_out_avg`
2. Per-output-channel noise standard deviation: `noise`
3. Source and output counts: `n_sources`, `n_outputs`

The total input power is derived from the singular-value spectrum:

```text
P_in,total = n_sources * n_outputs * P_out_avg / sum_i s_i^2
```

Then:

- `get_bitrate(...)` spreads `P_in,total` uniformly over sources.
- `get_capacity(...)` water-fills the same `P_in,total` over SVD modes.
- `get_bitrate_from_average_output_power(...)` and
  `get_capacity_from_average_output_power(...)` are the preferred wrappers.

## Noise Model Helpers

Use `guti.noise_models` to get the two output-channel quantities:

```python
from guti.capacity import get_bitrate_from_average_output_power
from guti.noise_models import compute_average_output_power, compute_output_noise_std

average_output_power = compute_average_output_power("meg_opm")
noise = compute_output_noise_std("meg_opm", n_sensors=500, tier="today")

bitrate = get_bitrate_from_average_output_power(
    s,
    average_output_power=average_output_power,
    noise=noise,
    n_sources=n_sources,
    n_outputs=n_outputs,
    time_resolution=0.01,
)
```

`compute_average_output_power()` is currently based on the typical observed
output amplitude configured for each modality. `compute_output_noise_std()`
returns detector/output noise in the same measurement units, scaled by sensor
count, bandwidth, tier, and ultrasound frequency where applicable.

## Modality Notes

- EEG uses Johnson noise at the electrode/front-end and a typical scalp signal
  amplitude of 5 microvolts.
- MEG OPM and SQUID use field noise in Tesla and a typical output signal
  amplitude of 100 fT.
- fNIRS uses shot-noise-limited relative intensity noise and a typical
  hemodynamic output amplitude of 1000 ppm.
- Ultrasound uses acoustic thermal plus electronic Johnson noise in the
  forward-model pressure-ratio units and a typical output amplitude of 0.001.

The older source-amplitude-derived effective noise, total-SNR helpers, and
SNR-based channel-capacity wrappers have been removed.
