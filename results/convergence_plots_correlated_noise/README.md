# Modality SVD Convergence Plots

Metrics are recomputed from saved singular-value spectra using the average-output-power workflow.
When a sweep file contains `noise_normalized_singular_values`, metrics use the spatially correlated noise model saved with that file.
`time_resolution_s` is set to `1 / bandwidth_hz` for each modality.

## Coverage

| modality | rows | voxel counts | sensor counts |
| --- | ---: | --- | --- |
| eeg_openmeeg | 63 | 18, 43, 148, 302, 984, 1842, 4347, 7347, 14355 | 32, 64, 128, 256, 512, 1024, 2048 |

## Plots

- [eeg_openmeeg/bitrate_vs_n_voxels_correlated_noise.png](eeg_openmeeg/bitrate_vs_n_voxels_correlated_noise.png)
- [eeg_openmeeg/capacity_vs_n_voxels_correlated_noise.png](eeg_openmeeg/capacity_vs_n_voxels_correlated_noise.png)
- [eeg_openmeeg/bitrate_vs_n_sensors_correlated_noise.png](eeg_openmeeg/bitrate_vs_n_sensors_correlated_noise.png)
- [eeg_openmeeg/capacity_vs_n_sensors_correlated_noise.png](eeg_openmeeg/capacity_vs_n_sensors_correlated_noise.png)

## Data

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
- [load_errors.json](load_errors.json)
