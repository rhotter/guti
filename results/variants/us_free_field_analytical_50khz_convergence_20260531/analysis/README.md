# Ultrasound 50 kHz Convergence Check

This directory contains the local analysis for a Modal sweep of the analytical ultrasound forward model at 50 kHz.

## Sweep

Command:

```bash
Initial sweep: /Users/nudge/.pyenv/versions/venv/bin/python run_modal_us_analytical_sweep.py --runner sdk --frequencies-khz 50 --source-counts 4000,8000,16000,32000 --sensor-counts 1000,3000,6000 --temporal-sampling 1 --sensor-batch-size 128 --bitrate-method svd --svd-method gram --stream-gram --save-gram-matrix --modal-output-dir results/variants/us_free_field_analytical_50khz_convergence_20260531 --log-dir logs/us_50khz_convergence_20260531 --jobs 1
Extension: /Users/nudge/.pyenv/versions/venv/bin/python run_modal_us_analytical_sweep.py --runner sdk --frequencies-khz 50 --source-counts 44000 --sensor-counts 1000,3000,6000 --temporal-sampling 1 --sensor-batch-size 128 --bitrate-method svd --svd-method gram --stream-gram --modal-output-dir results/variants/us_free_field_analytical_50khz_convergence_20260531 --log-dir logs/us_50khz_convergence_20260531 --jobs 1
```

- Completed NPZ results analyzed: 15
- Sensor counts: 1000, 3000, 6000
- Realized source counts: 4237, 8385, 16601, 32940, 45177
- Input power convention: average_output_power
- Source power normalization: none

## Plots

- [bitrate_vs_sources.png](bitrate_vs_sources.png)
- [capacity_vs_sources.png](capacity_vs_sources.png)
- [relative_bitrate_vs_sources.png](relative_bitrate_vs_sources.png)
- [relative_capacity_vs_sources.png](relative_capacity_vs_sources.png)
- [bitrate_vs_sensors_largest_sources.png](bitrate_vs_sensors_largest_sources.png)
- [capacity_vs_sensors_largest_sources.png](capacity_vs_sensors_largest_sources.png)

## Canonical Result

The canonical result was updated at `results/us_analytical_svd_spectrum.npz` from `results/variants/us_free_field_analytical_50khz_convergence_20260531/002_50khz_44000src_6000sensors__5e7ae3d4.npz`.

| quantity | value |
| --- | ---: |
| sensors | 6000 |
| realized source points | 45177 |
| matrix shape | 366000 x 45177 |
| singular values | 45177 |
| first singular value | 0.0204732 |
| rank > 1% first SV | 23582 |
| source amplitude scale | 1 |
| total input power | 175956 |
| bitrate | 3.98789e+11 bit/s |
| water-filled channel capacity | 3.98789e+11 bit/s |
| Modal Gram path | `None` |
| Modal Gram size | None bytes |

Sensor-scaling plots use the largest common realized source count, `45177`, when available.

| sensors | sources | bitrate bit/s | channel capacity bit/s |
| ---: | ---: | ---: | ---: |
| 1000 | 45177 | 2.78946e+11 | 2.80185e+11 |
| 3000 | 45177 | 3.73348e+11 | 3.73349e+11 |
| 6000 | 45177 | 3.98789e+11 | 3.98789e+11 |

## Data

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
