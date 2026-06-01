# Modality SVD Convergence Plots

Metrics are recomputed from saved singular-value spectra using the average-output-power workflow.
When a sweep file contains `noise_normalized_singular_values`, metrics use the spatially correlated noise model saved with that file.
`time_resolution_s` is set to `1 / bandwidth_hz` for each modality.

## Coverage

| modality | rows | voxel counts | sensor counts |
| --- | ---: | --- | --- |
| eeg_openmeeg | 117 | 18, 43, 148, 302, 984, 1842, 4347, 7347, 14355 | 32, 64, 128, 256, 512, 1024, 2048, 10000 |
| meg_opm | 67 | 50, 153, 360, 1153, 8937 | 50, 100, 200, 500, 600, 700, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000 |
| meg_squid | 67 | 50, 153, 360, 1153, 8937 | 50, 100, 200, 500, 600, 700, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000 |
| us_analytical_50khz | 15 | 4237, 8385, 16601, 32940, 45177 | 1000, 3000, 6000 |

8 rows use explicit spectrum estimates; see `spectrum_estimate_method` and reference fields in the metrics.

## Plots

- [eeg_openmeeg/bitrate_vs_n_voxels.png](eeg_openmeeg/bitrate_vs_n_voxels.png)
- [eeg_openmeeg/capacity_vs_n_voxels.png](eeg_openmeeg/capacity_vs_n_voxels.png)
- [eeg_openmeeg/bitrate_vs_n_sensors.png](eeg_openmeeg/bitrate_vs_n_sensors.png)
- [eeg_openmeeg/capacity_vs_n_sensors.png](eeg_openmeeg/capacity_vs_n_sensors.png)
- [eeg_openmeeg/bitrate_vs_n_voxels_correlated_noise.png](eeg_openmeeg/bitrate_vs_n_voxels_correlated_noise.png)
- [eeg_openmeeg/capacity_vs_n_voxels_correlated_noise.png](eeg_openmeeg/capacity_vs_n_voxels_correlated_noise.png)
- [eeg_openmeeg/bitrate_vs_n_sensors_correlated_noise.png](eeg_openmeeg/bitrate_vs_n_sensors_correlated_noise.png)
- [eeg_openmeeg/capacity_vs_n_sensors_correlated_noise.png](eeg_openmeeg/capacity_vs_n_sensors_correlated_noise.png)
- [meg_opm/bitrate_vs_n_voxels.png](meg_opm/bitrate_vs_n_voxels.png)
- [meg_opm/capacity_vs_n_voxels.png](meg_opm/capacity_vs_n_voxels.png)
- [meg_opm/bitrate_vs_n_sensors.png](meg_opm/bitrate_vs_n_sensors.png)
- [meg_opm/capacity_vs_n_sensors.png](meg_opm/capacity_vs_n_sensors.png)
- [meg_opm/bitrate_vs_n_voxels_correlated_noise.png](meg_opm/bitrate_vs_n_voxels_correlated_noise.png)
- [meg_opm/capacity_vs_n_voxels_correlated_noise.png](meg_opm/capacity_vs_n_voxels_correlated_noise.png)
- [meg_opm/bitrate_vs_n_sensors_correlated_noise.png](meg_opm/bitrate_vs_n_sensors_correlated_noise.png)
- [meg_opm/capacity_vs_n_sensors_correlated_noise.png](meg_opm/capacity_vs_n_sensors_correlated_noise.png)
- [meg_squid/bitrate_vs_n_voxels.png](meg_squid/bitrate_vs_n_voxels.png)
- [meg_squid/capacity_vs_n_voxels.png](meg_squid/capacity_vs_n_voxels.png)
- [meg_squid/bitrate_vs_n_sensors.png](meg_squid/bitrate_vs_n_sensors.png)
- [meg_squid/capacity_vs_n_sensors.png](meg_squid/capacity_vs_n_sensors.png)
- [meg_squid/bitrate_vs_n_voxels_correlated_noise.png](meg_squid/bitrate_vs_n_voxels_correlated_noise.png)
- [meg_squid/capacity_vs_n_voxels_correlated_noise.png](meg_squid/capacity_vs_n_voxels_correlated_noise.png)
- [meg_squid/bitrate_vs_n_sensors_correlated_noise.png](meg_squid/bitrate_vs_n_sensors_correlated_noise.png)
- [meg_squid/capacity_vs_n_sensors_correlated_noise.png](meg_squid/capacity_vs_n_sensors_correlated_noise.png)
- [us_analytical_50khz/bitrate_vs_n_voxels.png](us_analytical_50khz/bitrate_vs_n_voxels.png)
- [us_analytical_50khz/capacity_vs_n_voxels.png](us_analytical_50khz/capacity_vs_n_voxels.png)
- [us_analytical_50khz/bitrate_vs_n_sensors.png](us_analytical_50khz/bitrate_vs_n_sensors.png)
- [us_analytical_50khz/capacity_vs_n_sensors.png](us_analytical_50khz/capacity_vs_n_sensors.png)

## Data

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
- [load_errors.json](load_errors.json)
