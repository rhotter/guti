# 128k-source ultrasound RBC cone sweep

This run recomputed the three 50 kHz analytical ultrasound sensor counts with a 2 MHz RBC pressure-amplitude model and cone-dependent skull transmission.

## Model

- Requested sources: `128000`; realized source grid: `130399` points.
- Bitrate estimator: streaming SLQ, `s=8`, `t=32`, equal unit input power per source.
- Bitrate bandwidth convention: `1 Hz` via `bitrate_time_resolution=1.0 s`.
- Output noise: `5 mPa = 0.005 Pa`, scalar iid.
- Matrix normalization: disabled; the operator is scaled into output-pressure units.
- Cone: half angle `15 deg` around the positive x-axis after subtracting the head center.
- Skull pressure transmission: `0.5` inside the cone, `0.1` outside.

The pair scaling replaces the raw free-field amplitude with:

```text
p_ij(t) = G_delay_ij(t) * P_external * sqrt(eta * V) / r_ij * T_source_i * T_sensor_j
eta = CBV * BSC_10MHz * (2 MHz / 10 MHz)^4
```

Equivalently, because the raw free-field operator already has a `1/r_ij` factor, each streamed chunk is scaled by a common RBC pressure gain, source transmission, and sensor/output transmission before Gram accumulation and SLQ.

Reference pair pressures at `r=0.10 m`:

- outside/outside: `0.00587878 Pa` (`5.87878 mPa`)
- inside/outside: `0.0293939 Pa` (`29.3939 mPa`)
- inside/inside: `0.146969 Pa` (`146.969 mPa`)

## Outputs

- [metrics.csv](metrics.csv)
- [metrics.json](metrics.json)
- [bitrate_vs_sensors_rbc_cone.png](bitrate_vs_sensors_rbc_cone.png)

## Results

| sensors | realized sources | n_outputs | bitrate bit/s | Gram side | Gram shape | Gram size | Gram Modal path |
|---:|---:|---:|---:|---|---:|---:|---|
| 1000 | 130399 | 61000 | 311992 | G G^T | 61000x61000 | 14884000128 | `/modal_results/us_analytical_grams/001_50khz_128000src_1000sensors_gram.npy` |
| 3000 | 130399 | 183000 | 705968 | G^T G | 130399x130399 | 68015596932 | `/modal_results/us_analytical_grams/001_50khz_128000src_3000sensors_gram.npy` |
| 6000 | 130399 | 366000 | 769052 | G^T G | 130399x130399 | 68015596932 | `/modal_results/us_analytical_grams/002_50khz_128000src_6000sensors_gram.npy` |

## Timing

| sensors | Gram save seconds | SLQ seconds | note |
|---:|---:|---:|---|
| 1000 | 38.733 | 151.840 |  |
| 3000 | 244.094 | 234.618 |  |
| 6000 | 268.491 | 558.151 | Gram saved in the interrupted save+SLQ run; bitrate came from a follow-up bitrate-only run. |
