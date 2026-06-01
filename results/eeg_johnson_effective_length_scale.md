# EEG Johnson Noise Effective Length Scale

This is a reproducible estimate of the scalar distance-kernel length
that best approximates the Johnson covariance from the GUTI layered
spherical EEG head model.

## Configuration

- Sensors: 256 Fibonacci scalp electrodes
- Electrode patch area: 1.0 cm^2 circular cap
- Spherical-harmonic truncation: l <= 2000
- Temperature: 310 K
- Bandwidth: 100.0 Hz
- Fit objective: least squares in log-correlation over positive
  off-diagonal sensor pairs.

## Estimates

| Model | max offdiag corr | min offdiag corr | exp L (mm) | exp log RMSE | gaussian L (mm) | gaussian log RMSE |
|---|---:|---:|---:|---:|---:|---:|
| volume only | 0.4463 | -0.0395 | 18.688 | 0.506 | 27.090 | 0.664 |
| volume + 500 ohm independent series | 0.1402 | -0.01241 | 14.304 | 0.565 | 23.948 | 1.221 |
| volume + 1000 ohm independent series | 0.08318 | -0.007361 | 12.936 | 0.674 | 22.848 | 1.487 |
| volume + 5000 ohm independent series | 0.01955 | -0.00173 | 10.224 | 1.085 | 20.446 | 2.237 |

## Interpretation

For the pure volume-conductor Johnson term, the effective length is
about 19 mm for the exponential kernel and 27 mm for the Gaussian
kernel. If the existing Gaussian distance model is kept as a
fallback approximation, 27 mm is the closest match under this fit.

Adding independent series/contact resistance mostly increases the
diagonal variance and suppresses off-diagonal correlations. With a
5 kOhm independent series term, the nearest-neighbor correlation is
only about 0.02, so a single length-scale kernel is a poor physical
description; the direct Johnson covariance should be preferred.
