# Ultrasound 50 kHz Convergence Check

This directory contains the local analysis for a Modal sweep of the analytical ultrasound forward model at 50 kHz.

## Sweep

Command:

```bash
/Users/nudge/.pyenv/versions/venv/bin/python run_modal_us_analytical_sweep.py --runner sdk --frequencies-khz 50 --source-counts 4000,8000,16000,32000 --sensor-counts 1000,3000,6000 --temporal-sampling 1 --sensor-batch-size 128 --bitrate-method svd --svd-method gram --stream-gram --save-gram-matrix --modal-output-dir results/variants/us_free_field_analytical_50khz_convergence_20260531 --log-dir logs/us_50khz_convergence_20260531 --jobs 1
```

- Completed NPZ results analyzed: 12
- Sensor counts: 1000, 3000, 6000
- Realized source counts: 4237, 8385, 16601, 32940

## Plots

- [bitrate_vs_sources.png](bitrate_vs_sources.png)
- [capacity_vs_sources.png](capacity_vs_sources.png)
- [relative_bitrate_vs_sources.png](relative_bitrate_vs_sources.png)
- [relative_capacity_vs_sources.png](relative_capacity_vs_sources.png)
- [bitrate_vs_sensors_largest_sources.png](bitrate_vs_sensors_largest_sources.png)
- [capacity_vs_sensors_largest_sources.png](capacity_vs_sensors_largest_sources.png)

## Canonical Result

The canonical result was updated at `results/us_analytical_svd_spectrum.npz` from `results/variants/us_free_field_analytical_50khz_convergence_20260531/012_50khz_32000src_6000sensors__e9dcf4ea.npz`.

| quantity | value |
| --- | ---: |
| sensors | 6000 |
| realized source points | 32940 |
| matrix shape | 366000 x 32940 |
| singular values | 32940 |
| first singular value | 0.023958 |
| rank > 1% first SV | 21963 |
| bitrate | 2.97379e+11 bit/s |
| water-filled channel capacity | 2.97379e+11 bit/s |
| Modal Gram path | `/modal_results/us_analytical_grams/012_50khz_32000src_6000sensors_gram.npy` |
| Modal Gram size | 4340174528 bytes |

Sensor-scaling plots use the largest common realized source count, `32940`, when available.

| sensors | sources | bitrate bit/s | channel capacity bit/s |
| ---: | ---: | ---: | ---: |
| 1000 | 32940 | 2.31846e+11 | 2.32034e+11 |
| 3000 | 32940 | 2.82769e+11 | 2.82769e+11 |
| 6000 | 32940 | 2.97379e+11 | 2.97379e+11 |

## Data

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
