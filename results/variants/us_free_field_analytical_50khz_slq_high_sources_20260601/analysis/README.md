# Ultrasound 50 kHz Streaming SLQ Convergence

This analysis extends the exact 50 kHz SVD convergence sweep with JSON-only streaming SLQ bitrate estimates at higher source counts.

SLQ estimates the equal-power trace-log bitrate. It does not save singular spectra and this script does not estimate the water-filled channel capacity.

## Bitrate Parameters

These rows use the equal-input-power bitrate formula `sum log2(1 + sigma_i^2 P_source / noise^2) / (2T)`.

- Input power convention: `average_output_power`
- Average output signal amplitude(s): 0.001
- Noise multiplier(s): 1
- Bitrate time resolution(s): 2e-06 s

The current completed high-source rows used the very high-SNR default (`1e-3` output amplitude with the modeled acoustic/electronic noise floor) and `T=2e-6 s`. That convention makes weak high-index modes count strongly and can delay apparent source-count convergence.

## Inputs

- Exact SVD rows: 15
- SLQ rows: 11
- Sensor counts: 1000, 3000, 6000
- Largest SLQ realized source count: 195147

## Plots

- [bitrate_vs_sources_exact_plus_slq.png](bitrate_vs_sources_exact_plus_slq.png)
- [relative_bitrate_vs_sources_exact_plus_slq.png](relative_bitrate_vs_sources_exact_plus_slq.png)
- [bitrate_vs_sensors_largest_slq_sources.png](bitrate_vs_sensors_largest_slq_sources.png)

## Largest-Source Sensor Sweep

| sensors | realized sources | SLQ bitrate bit/s |
| ---: | ---: | ---: |
| 3000 | 195147 | 1.47443e+12 |
| 6000 | 195147 | 1.63663e+12 |

## Data

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
