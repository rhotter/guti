"""
Simulation of ultrasound propagation in a free field, using the analytical fundamental solution (Green's function).
We treat the "independent variables" in ultrasound imaging as sources. This relies on the approximation that the intensity of the transmit pulse is the same at each point in the medium, which is related to the Born approximation.
"""

# %%

import torch
import math
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from guti.data_utils import save_svd
from guti.modalities.us.utils import (
    build_source_signal,
    free_field_voxel_size,
    make_free_field_chunk_fn,
    create_free_field_sources as create_free_field_sources_real,
    create_free_field_receivers as create_free_field_receivers_real,
)
from guti.noise_models import (
    DEFAULT_NOISE_CORRELATION_KERNEL,
    DEFAULT_NOISE_CORRELATION_LENGTH_MM,
)
import time
from pathlib import Path

import torch, torch.backends.cuda as cu
import torch.cuda.comm as cuda_comm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
torch.set_float32_matmul_precision('high')  # allow TF32 on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True


def set_torch_linalg_backend(backend: str) -> str:
    """Set the preferred CUDA linalg backend if this PyTorch build supports it."""
    if backend == "default":
        cu.preferred_linalg_library(None)
        return "default"
    try:
        cu.preferred_linalg_library(backend)
        return backend
    except RuntimeError as exc:
        print(
            f"WARNING: could not set torch CUDA linalg backend to {backend!r}: {exc}. "
            "Using default backend."
        )
        try:
            cu.preferred_linalg_library(None)
        except RuntimeError:
            pass
        return "default"


# Free-field geometry / waveform / voxel helpers and the receiver-batch chunk
# function are imported from guti.modalities.us.utils (the single shared
# definition used by USModality too) — see the imports above.


# ---------------------------------------------------------------------------
# SLQ estimators live in guti.slq (canonical, jax-free home); re-exported here
# for backward compatibility with existing callers and the Modal sweep driver.
# ---------------------------------------------------------------------------
from guti.slq import (
    bitrate_slq_torch_gpu_chunked,
    bitrate_slq_torch_gpu_chunked_probe_parallel,
    bitrate_slq_torch_multi_gpu_sharded,
    estimate_spectral_norm_chunked,
    estimate_frobenius_norm_sq_hutchinson,
    bitrate_slq_torch_gpu_streaming,
    bitrate_slq_torch_gpu_streaming_probe_parallel,
    estimate_spectral_norm_streaming,
    estimate_frobenius_norm_sq_streaming,
    compute_frobenius_norm_sq_streaming_exact,
    estimate_spectral_norm_streaming_probe_parallel,
    estimate_frobenius_norm_sq_streaming_probe_parallel,
)


# %%

import argparse


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _sanitize_json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_result_json(path_str: str | None, record: dict) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize_json_value(record), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _emit_result_json(record: dict, result_json_path: str | None) -> None:
    sanitized = _sanitize_json_value(record)
    payload = json.dumps(sanitized, sort_keys=True, allow_nan=False)
    print(f"RESULT_JSON: {payload}", flush=True)
    _write_result_json(result_json_path, sanitized)


def _sqrt_nonnegative_estimate(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    if value < 0.0:
        if value > -1e-12:
            value = 0.0
        else:
            raise ValueError(f"{label} must be non-negative, got {value}")
    return math.sqrt(value)

parser = argparse.ArgumentParser(description='Ultrasound simulation parameters')
parser.add_argument('--n_sources', type=int, default=32000, help='Number of source points')
parser.add_argument('--n_sensors', type=int, default=1000, help='Number of sensor points') 
parser.add_argument('--temporal_sampling', type=int, default=5, help='Temporal sampling rate')
parser.add_argument('--sensor_batch_size', type=int, default=512, help='Batch size across sensors for Gram accumulation')
parser.add_argument('--center_frequency', type=float, default=0.05e6, help='Center frequency in Hz')
parser.add_argument('--signal_type', type=str, default='tone_burst', choices=['cw', 'tone_burst'], help='Excitation waveform type')
parser.add_argument('--signal_cycles', type=float, default=2.0, help='Cycles in the emitted tone burst')
parser.add_argument('--signal_window', type=str, default='hann', choices=['rect', 'hann'], help='Envelope for tone-burst excitation')
parser.add_argument(
    '--source_power_normalization',
    type=str,
    default='none',
    choices=['none', 'fixed_total'],
    help='Scale per-source waveform amplitude. fixed_total uses 1/sqrt(n_sources).',
)
parser.add_argument(
    '--input_power_convention',
    type=str,
    default='average_output_power',
    choices=['average_output_power', 'fixed_total_source_power'],
    help='Input power convention for bitrate/capacity estimates.',
)
parser.add_argument('--accumulate_on_cpu', action='store_true', help='Accumulate Gram matrix on CPU instead of GPU')
parser.add_argument('--svd_device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to compute eigenvalues/SVD of Gram')
parser.add_argument(
    '--linalg_backend',
    type=str,
    default=os.environ.get("GUTI_TORCH_LINALG_BACKEND", "magma"),
    choices=['magma', 'cusolver', 'default'],
    help='Preferred torch CUDA linear algebra backend for SVD/eigendecomposition.',
)
parser.add_argument(
    '--svd_method',
    type=str,
    default='direct',
    choices=['direct', 'gram'],
    help='SVD method for bitrate_method=svd. direct uses torch.linalg.svdvals; gram uses eigvalsh of the smaller Gram matrix.',
)
parser.add_argument(
    '--stream_gram',
    action='store_true',
    help='For --svd_method=gram, accumulate G^T G from sensor chunks without materializing G.',
)
parser.add_argument(
    '--save_gram_matrix',
    action='store_true',
    help='Save the full Gram matrix used for SVD. Use with --svd_method=gram.',
)
parser.add_argument(
    '--gram_output_path',
    type=str,
    default=None,
    help='Optional .npy path for the saved Gram matrix. Defaults next to the saved SVD result.',
)
parser.add_argument(
    '--bitrate_method',
    type=str,
    default='slq',
    choices=['svd', 'slq', 'both'],
    help='Compute bitrate via SVD, SLQ, or both',
)
parser.add_argument('--slq_s', type=int, default=128, help='Number of SLQ probe vectors')
parser.add_argument('--slq_t', type=int, default=128, help='Lanczos steps for SLQ')
parser.add_argument('--slq_batch', type=int, default=128, help='SLQ batch size per iteration')
parser.add_argument('--slq_chunk_rows', type=int, default=65536, help='Row chunk size for SLQ matvecs')
parser.add_argument('--slq_verbose', action='store_true', default=False, help='Print SLQ progress logs')
parser.add_argument('--slq_multi_gpu', action='store_true', help='Use multi-GPU sharded SLQ')
parser.add_argument('--slq_devices', type=str, default='', help='Comma-separated CUDA device IDs for SLQ')
parser.add_argument('--slq_streaming', action='store_true', help='Stream SLQ matvecs without materializing G')
parser.add_argument('--slq_probe_parallel', action='store_true', help='Parallelize SLQ probes across GPUs (chunked mode)')
parser.add_argument('--noise_level', type=float, default=None, help='Override noise level (skip estimation)')
parser.add_argument(
    '--noise_multiplier',
    type=float,
    default=1.0,
    help='Multiply the modeled output-noise std before matrix normalization.',
)
parser.add_argument(
    '--noise_correlation_length_mm',
    type=float,
    default=DEFAULT_NOISE_CORRELATION_LENGTH_MM,
    help='Scalp noise-correlation length in mm for covariance-aware capacity. Default: 5.',
)
parser.add_argument(
    '--noise_correlation_kernel',
    type=str,
    default=DEFAULT_NOISE_CORRELATION_KERNEL,
    choices=['gaussian', 'exponential'],
    help='Spatial noise-correlation kernel. Default: gaussian.',
)
parser.add_argument(
    '--average_output_signal_amplitude',
    type=float,
    default=None,
    help='Override the typical per-output signal amplitude used by average_output_power.',
)
parser.add_argument(
    '--bitrate_time_resolution',
    type=float,
    default=None,
    help='Override the time resolution in seconds used in bitrate/capacity formulas.',
)
parser.add_argument(
    '--disable_matrix_normalization',
    action='store_true',
    help='Use the raw propagation matrix instead of dividing by sqrt(n_sources * n_sensors).',
)
parser.add_argument(
    '--result_json_path',
    type=str,
    default=None,
    help='Optional path to save a strict JSON result record for this run.',
)



def main() -> None:
    args = parser.parse_args()
    if args.noise_multiplier <= 0.0:
        raise ValueError("--noise_multiplier must be positive")
    if args.noise_correlation_length_mm <= 0.0:
        raise ValueError("--noise_correlation_length_mm must be positive")
    if args.average_output_signal_amplitude is not None and args.average_output_signal_amplitude < 0.0:
        raise ValueError("--average_output_signal_amplitude must be non-negative")
    if args.bitrate_time_resolution is not None and args.bitrate_time_resolution <= 0.0:
        raise ValueError("--bitrate_time_resolution must be positive")

    n_sources = args.n_sources
    n_sensors = args.n_sensors
    temporal_sampling = args.temporal_sampling
    sensor_batch_size = args.sensor_batch_size
    svd_device = args.svd_device
    center_frequency = args.center_frequency
    bitrate_method = args.bitrate_method
    accumulate_on_cpu = args.accumulate_on_cpu
    active_linalg_backend = set_torch_linalg_backend(args.linalg_backend)

    print(f"n_sources: {n_sources}, n_sensors: {n_sensors}, temporal_sampling: {temporal_sampling}, sensor_batch_size: {sensor_batch_size}")
    print(f"matrix_normalization: {'disabled' if args.disable_matrix_normalization else 'enabled'}")
    print(f"torch CUDA linalg backend: {active_linalg_backend}")

    # Create the source and receiver positions in real space (meters).
    source_positions = create_free_field_sources_real(n_sources)
    sensor_positions = create_free_field_receivers_real(n_sensors)

    n_sources = source_positions.shape[0]

    # Source waveform
    time_step = 1e-1 / center_frequency
    time_duration = 120e-6
    time_axis = np.arange(0, time_duration, time_step)
    source_signal = build_source_signal(
        time_axis,
        center_frequency,
        signal_type=args.signal_type,
        signal_cycles=args.signal_cycles,
        signal_window=args.signal_window,
    )
    if args.source_power_normalization == "fixed_total":
        source_amplitude_scale = 1.0 / math.sqrt(float(n_sources))
    else:
        source_amplitude_scale = 1.0
    print(f"source_power_normalization: {args.source_power_normalization}")
    print(f"source_amplitude_scale: {source_amplitude_scale}")
    source_signals = source_signal * source_amplitude_scale
    source_signals = np.tile(source_signals, (n_sources, 1))

    nt = math.ceil(time_axis.shape[0] / temporal_sampling)
    voxel_size = free_field_voxel_size(center_frequency)
    effective_time_resolution = time_step * temporal_sampling
    if args.bitrate_time_resolution is not None:
        effective_time_resolution = args.bitrate_time_resolution
    print(f"effective_time_resolution: {effective_time_resolution}")

    #%%

    # device for propagation (and potentially accumulation)
    device = "cuda"

    use_complex_ampitudes = False

    if args.slq_streaming and bitrate_method == "both":
        raise ValueError("--slq_streaming currently supports bitrate_method=slq only")
    if args.slq_streaming and bitrate_method == "svd":
        raise ValueError("--slq_streaming cannot be used with bitrate_method=svd")
    if args.slq_streaming and args.slq_multi_gpu:
        raise ValueError("--slq_streaming currently supports single-GPU SLQ only")
    if args.slq_streaming and args.slq_probe_parallel:
        pass

    # %%

    gram_for_svd = None
    gram_side_for_svd = None
    gram_output_path = None

    if not args.slq_streaming:
        print("Computing SVD (batched simulation + Gram accumulation)...")

    num_sensors_total = sensor_positions.shape[0]
    num_sources_total = n_sources
    print(f"num_sensors: {num_sensors_total}, num_sources: {num_sources_total}, device: {device}")

    # Pre-build constant tensors on propagation device to avoid repeated transfers
    source_positions_t = torch.tensor(source_positions, device=device)
    source_signals_t = torch.tensor(source_signals, device=device)
    voxel_size_t = torch.tensor(voxel_size, device=device)

    stream_device_ids = None
    slq_device_ids = None
    source_positions_t_list = None
    source_signals_t_list = None
    voxel_size_t_list = None
    if args.slq_streaming and args.slq_probe_parallel:
        if args.slq_devices:
            stream_device_ids = [int(x) for x in args.slq_devices.split(",") if x.strip() != ""]
        else:
            stream_device_ids = list(range(torch.cuda.device_count()))
        slq_device_ids = stream_device_ids
        source_positions_t_list = [
            torch.tensor(source_positions, device=f"cuda:{dev}") for dev in stream_device_ids
        ]
        source_signals_t_list = [
            torch.tensor(source_signals, device=f"cuda:{dev}") for dev in stream_device_ids
        ]
        voxel_size_t_list = [
            torch.tensor(voxel_size, device=f"cuda:{dev}") for dev in stream_device_ids
        ]

    # Single-device receiver-batch chunk fn (shared with USModality via utils).
    compute_chunk_matrix = make_free_field_chunk_fn(
        source_positions_t,
        sensor_positions,
        source_signals_t,
        time_step=time_step,
        center_frequency=center_frequency,
        voxel_size_t=voxel_size_t,
        temporal_sampling=temporal_sampling,
        device=device,
        num_sources=num_sources_total,
        use_complex_amplitudes=use_complex_ampitudes,
    )


    def make_compute_chunk_matrix_for_device(dev_id: int):
        if stream_device_ids is None or source_positions_t_list is None:
            raise ValueError("Streaming probe-parallel tensors are not initialized")
        idx = stream_device_ids.index(dev_id)
        return make_free_field_chunk_fn(
            source_positions_t_list[idx],
            sensor_positions,
            source_signals_t_list[idx],
            time_step=time_step,
            center_frequency=center_frequency,
            voxel_size_t=voxel_size_t_list[idx],
            temporal_sampling=temporal_sampling,
            device=f"cuda:{dev_id}",
            num_sources=num_sources_total,
            use_complex_amplitudes=use_complex_ampitudes,
        )

    G = None
    if not args.slq_streaming:
        # Accumulate matrix rows in batches over sensors. For large source/sensor
        # convergence sweeps, the full G matrix may not fit in memory, but G^T G can
        # still fit. In that case --stream_gram avoids materializing G.
        t0 = time.perf_counter()
        print(f"num_sensors_total: {num_sensors_total}, nt: {nt}")

        if args.stream_gram:
            if args.svd_method != "gram":
                raise ValueError("--stream_gram requires --svd_method=gram")
            n_outputs_total = num_sensors_total * nt
            if n_outputs_total < num_sources_total:
                raise ValueError(
                    "--stream_gram currently accumulates G^T G and requires "
                    f"n_outputs >= n_sources, got {n_outputs_total} < {num_sources_total}. "
                    "Reduce temporal_sampling, increase sensors, or reduce sources."
                )
            gram_device = "cpu" if accumulate_on_cpu else device
            print(
                f"Streaming G^T G accumulation on {gram_device}; "
                f"Gram shape=({num_sources_total}, {num_sources_total})"
            )
            gram_for_svd = torch.zeros(
                (num_sources_total, num_sources_total),
                dtype=torch.float32,
                device=gram_device,
            )
            for start in range(0, num_sensors_total, sensor_batch_size):
                print(f"Processing batch {start // sensor_batch_size + 1} of {(num_sensors_total + sensor_batch_size - 1) // sensor_batch_size}")
                end = min(start + sensor_batch_size, num_sensors_total)
                print(f"start: {start}, end: {end}")
                chunk_matrix = compute_chunk_matrix(start, end)
                if accumulate_on_cpu:
                    chunk_matrix = chunk_matrix.cpu()
                gram_for_svd.addmm_(chunk_matrix.T, chunk_matrix)
                del chunk_matrix
                if device == "cuda":
                    torch.cuda.empty_cache()
            gram_for_svd = 0.5 * (gram_for_svd + gram_for_svd.T)
            gram_side_for_svd = "G^T G"
        else:
            G_device = "cpu" if accumulate_on_cpu else device
            G = torch.zeros((num_sensors_total * nt, num_sources_total), dtype=torch.float32, device=G_device)
            last_index = 0
            for start in range(0, num_sensors_total, sensor_batch_size):
                print(f"Processing batch {start // sensor_batch_size + 1} of {(num_sensors_total + sensor_batch_size - 1) // sensor_batch_size}")
                end = min(start + sensor_batch_size, num_sensors_total)
                print(f"start: {start}, end: {end}")
                chunk_matrix = compute_chunk_matrix(start, end)
                chunk_rows = chunk_matrix.shape[0]
                print(f"last_index: {last_index}, chunk_rows: {chunk_rows}")
                if accumulate_on_cpu:
                    G[last_index:last_index + chunk_rows] = chunk_matrix.cpu()
                else:
                    G[last_index:last_index + chunk_rows] = chunk_matrix
                last_index += chunk_rows

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"gram_accumulate: {t1 - t0:.3f}s")


    t0 = time.perf_counter()

    # if G.shape[1] > G.shape[0]:
    #     G = G @ G.T
    # else:
    #     G = G.T @ G

    # s = torch.linalg.svdvals(G)
    # s = torch.sqrt(s)

    from guti.data_utils import Parameters
    from guti.capacity import (
        get_bitrate,
        sensor_noise_normalized_singular_values,
        total_input_power_from_average_output_power,
    )
    from guti.noise_models import (
        DEFAULT_NOISE_CORRELATION_KERNEL,
        DEFAULT_NOISE_CORRELATION_LENGTH_MM,
        compute_average_output_power,
        compute_input_amplitude,
        compute_output_noise_std,
        compute_sensor_noise_covariance,
    )
    from guti.data_utils import save_svd

    noise_level = args.noise_level
    raw_noise_level = None
    average_output_signal_amplitude = (
        args.average_output_signal_amplitude
        if args.average_output_signal_amplitude is not None
        else math.sqrt(compute_average_output_power("us_analytical"))
    )
    s_normalized = None
    s_noise_normalized = None
    bitrate_svd = None
    bitrate_slq = None
    saved_svd_path = None
    total_input_power = None
    physical_total_source_power = compute_input_amplitude("us_analytical") ** 2
    slq_frobenius_norm_sq = None
    slq_logdet_alpha = None

    matrix_normalization_scale = (
        1.0
        if args.disable_matrix_normalization
        else 1.0 / math.sqrt(len(source_positions) * len(sensor_positions))
    )
    average_output_power = average_output_signal_amplitude**2 * matrix_normalization_scale**2
    if noise_level is None:
        raw_noise_level = compute_output_noise_std(
            "us_analytical",
            n_sensors=len(sensor_positions),
            frequency_hz=center_frequency,
        ) * args.noise_multiplier
        noise_level = raw_noise_level * matrix_normalization_scale
        print(f"output_noise (raw matrix units): {raw_noise_level}")
        print(f"noise_level (analysis matrix units): {noise_level}")
    else:
        print(f"noise_level override (analysis matrix units): {noise_level}")
    print(f"average_output_signal_amplitude (raw matrix units): {average_output_signal_amplitude}")
    print(f"average_output_power (analysis matrix units): {average_output_power}")
    print(f"noise_multiplier: {args.noise_multiplier}")

    if bitrate_method in {"svd", "both"}:
        if G is None and gram_for_svd is None:
            raise ValueError("No materialized matrix or streamed Gram is available for SVD")
        G_svd = None if G is None else (G if G.device.type == svd_device else G.to(svd_device))
        if args.svd_method == "gram":
            if gram_for_svd is not None:
                gram = gram_for_svd
                if gram.device.type != svd_device:
                    gram = gram.to(svd_device)
                gram_side = gram_side_for_svd or "streamed Gram"
            else:
                m, n = G_svd.shape
                print(f"Computing singular values via {args.svd_method} method for matrix {m} x {n}")
                if m >= n:
                    gram = G_svd.T @ G_svd
                    gram_side = "G^T G"
                else:
                    gram = G_svd @ G_svd.T
                    gram_side = "G G^T"
                gram = 0.5 * (gram + gram.T)
            gram_side_for_svd = gram_side
            print(f"Computing eigvalsh({gram_side}) with shape {tuple(gram.shape)}")
            eigvals = torch.linalg.eigvalsh(gram)
            s = eigvals.clamp_min(0).sqrt().flip(0)
            if args.save_gram_matrix:
                gram_output_path = args.gram_output_path
                if gram_output_path is None:
                    gram_output_path = "pending"
                print(f"Deferring Gram save until SVD path is known: {gram_output_path}")
            if gram_for_svd is None and not args.save_gram_matrix:
                del gram
            del eigvals
            if svd_device == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"Computing singular values via direct SVD for matrix {tuple(G_svd.shape)}")
            s = torch.linalg.svdvals(G_svd)
        s = s.cpu().numpy()
        print(f"First 10 singular values: {s[:10]}")
        print(f"Ratio of sums: {np.sum(s[:10])}")
        print("Done!")

        if G is not None:
            print("Computing noise-normalized singular values with spatial noise covariance")
            noise_std_for_covariance = noise_level / matrix_normalization_scale
            sensor_noise_covariance = compute_sensor_noise_covariance(
                sensor_positions * 1e3,
                noise_std_for_covariance,
                correlation_length_mm=float(args.noise_correlation_length_mm),
                kernel=args.noise_correlation_kernel,
            )
            G_cpu = G.detach().cpu().numpy() if isinstance(G, torch.Tensor) else np.asarray(G)
            s_noise_normalized = sensor_noise_normalized_singular_values(
                G_cpu,
                sensor_noise_covariance=sensor_noise_covariance,
                outputs_per_sensor=nt,
            )
            print(f"First 10 noise-normalized singular values: {s_noise_normalized[:10]}")
        else:
            print(
                "Skipping spatial noise covariance spectrum: stream_gram does not "
                "materialize the sensor-time matrix"
            )

        plt.semilogy(s)
        ax = plt.gca()
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True, min_n_ticks=10))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=30))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=30))
        log_formatter = mticker.LogFormatter(base=10.0, labelOnlyBase=False)
        ax.yaxis.set_major_formatter(log_formatter)
        ax.yaxis.set_minor_formatter(log_formatter)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()

        saved_svd_path = save_svd(s, f'us_free_field_analytical_frequency_sweep', params=Parameters(
            num_sensors=len(sensor_positions),
            num_brain_grid_points=len(source_positions),
            time_resolution=effective_time_resolution,
            frequency_hz=center_frequency,
            matrix_size=(num_sensors_total * nt, len(source_positions)),
            noise_correlation_length_mm=float(args.noise_correlation_length_mm),
            noise_correlation_kernel=args.noise_correlation_kernel,
            noise_distance_metric="geodesic",
            comment=(
                f"signal_type={args.signal_type},signal_cycles={args.signal_cycles},"
                f"signal_window={args.signal_window},"
                f"source_power_normalization={args.source_power_normalization},"
                f"source_amplitude_scale={source_amplitude_scale},"
                f"input_power_convention={args.input_power_convention},"
                f"matrix_normalization={'disabled' if args.disable_matrix_normalization else 'enabled'}"
            ),
            vincent_trick=False
        ), extra_arrays=(
            {
                "noise_normalized_singular_values": s_noise_normalized,
                "noise_correlation_length_mm": np.array(
                    args.noise_correlation_length_mm,
                    dtype=np.float64,
                ),
                "noise_correlation_kernel": np.array(args.noise_correlation_kernel),
            }
            if s_noise_normalized is not None
            else {
                "noise_correlation_length_mm": np.array(
                    args.noise_correlation_length_mm,
                    dtype=np.float64,
                ),
                "noise_correlation_kernel": np.array(args.noise_correlation_kernel),
            }
        ))
        if args.save_gram_matrix:
            if args.svd_method != "gram":
                raise ValueError("--save_gram_matrix requires --svd_method=gram")
            if gram_for_svd is not None:
                gram_to_save = gram_for_svd
            else:
                gram_to_save = gram
            if args.gram_output_path is None:
                gram_output_path = str(saved_svd_path).replace(".npz", "_gram.npy")
            else:
                gram_output_path = args.gram_output_path
            print(f"Saving Gram matrix to {gram_output_path}")
            Path(gram_output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(gram_output_path, gram_to_save.detach().cpu().numpy())
            print(f"Saved Gram matrix to {gram_output_path}")

        if args.disable_matrix_normalization:
            s_normalized = s
        else:
            s_normalized = s / (len(source_positions)**0.5 * len(sensor_positions)**0.5)
        if args.input_power_convention == "fixed_total_source_power":
            total_input_power = physical_total_source_power / (source_amplitude_scale**2)
        else:
            total_input_power = total_input_power_from_average_output_power(
                s_normalized,
                average_output_power=average_output_power,
                n_sources=len(source_positions),
                n_outputs=num_sensors_total * nt,
            )
        bitrate_svd = float(
            get_bitrate(
                s_noise_normalized if s_noise_normalized is not None else s_normalized,
                n_sources=len(source_positions),
                total_input_power=total_input_power,
                noise=1.0 if s_noise_normalized is not None else noise_level,
                time_resolution=effective_time_resolution,
            )
        )
        print(f"noise_level: {noise_level}")
        print(f"input_power_convention: {args.input_power_convention}")
        print(f"total_input_power: {total_input_power}")
        print(f"bitrate: {bitrate_svd}")
        print(n_sensors)
        # Channel capacity uses the same output-power/noise workflow as bitrate.

    if bitrate_method in {"slq", "both"}:
        if not args.slq_streaming:
            raise NotImplementedError("Use --slq_streaming for source sweeps that avoid materializing G.")
        print("Computing streaming SLQ bitrate estimate...")
        t0 = time.perf_counter()
        slq_frobenius_norm_sq = compute_frobenius_norm_sq_streaming_exact(
            compute_chunk_matrix,
            num_sensors_total=num_sensors_total,
            sensor_batch_size=sensor_batch_size,
            normalize_scale=matrix_normalization_scale,
            device=device,
            verbose=args.slq_verbose,
        )
        if args.input_power_convention == "fixed_total_source_power":
            total_input_power = physical_total_source_power / (source_amplitude_scale**2)
        else:
            total_input_power = total_input_power_from_average_output_power(
                np.asarray([math.sqrt(slq_frobenius_norm_sq)], dtype=np.float64),
                average_output_power=average_output_power,
                n_sources=len(source_positions),
                n_outputs=num_sensors_total * nt,
            )
        input_power_per_source = total_input_power / len(source_positions)
        slq_logdet_alpha = input_power_per_source / (noise_level**2)
        print(f"slq_frobenius_norm_sq: {slq_frobenius_norm_sq}")
        print(f"input_power_convention: {args.input_power_convention}")
        print(f"total_input_power: {total_input_power}")
        print(f"slq_logdet_alpha: {slq_logdet_alpha}")
        if args.slq_probe_parallel:
            bitrate_slq = bitrate_slq_torch_gpu_streaming_probe_parallel(
                make_compute_chunk_matrix_for_device,
                device_ids=slq_device_ids or [0],
                num_sensors_total=num_sensors_total,
                num_sources_total=num_sources_total,
                nt=nt,
                sensor_batch_size=sensor_batch_size,
                noise_std_full_brain=noise_level,
                time_resolution=effective_time_resolution,
                logdet_alpha=slq_logdet_alpha,
                s=args.slq_s,
                t=args.slq_t,
                batch=args.slq_batch,
                normalize_scale=matrix_normalization_scale,
                verbose=args.slq_verbose,
            )
        else:
            bitrate_slq = bitrate_slq_torch_gpu_streaming(
                compute_chunk_matrix,
                num_sensors_total=num_sensors_total,
                num_sources_total=num_sources_total,
                nt=nt,
                sensor_batch_size=sensor_batch_size,
                noise_std_full_brain=noise_level,
                time_resolution=effective_time_resolution,
                logdet_alpha=slq_logdet_alpha,
                s=args.slq_s,
                t=args.slq_t,
                batch=args.slq_batch,
                normalize_scale=matrix_normalization_scale,
                verbose=args.slq_verbose,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"slq_elapsed: {time.perf_counter() - t0:.3f}s")
        print(f"bitrate_slq: {bitrate_slq}")

    result_record = {
        "status": "ok",
        "n_sources": int(len(source_positions)),
        "n_sensors": int(len(sensor_positions)),
        "frequency_hz": float(center_frequency),
        "frequency_khz": float(center_frequency / 1_000.0),
        "temporal_sampling": int(args.temporal_sampling),
        "time_step_seconds": float(time_step),
        "effective_time_resolution_seconds": float(effective_time_resolution),
        "sensor_batch_size": int(args.sensor_batch_size),
        "bitrate_method": bitrate_method,
        "svd_method": args.svd_method,
        "matrix_size": [int(num_sensors_total * nt), int(len(source_positions))],
        "gram_output_path": gram_output_path,
        "gram_side": gram_side_for_svd,
        "bitrate": bitrate_slq if bitrate_slq is not None else bitrate_svd,
        "bitrate_slq": bitrate_slq,
        "bitrate_svd": bitrate_svd,
        "noise_level": noise_level,
        "raw_noise_level": float(raw_noise_level) if raw_noise_level is not None else None,
        "noise_multiplier": float(args.noise_multiplier),
        "noise_model_type": (
            "spatial_covariance" if s_noise_normalized is not None else "scalar_iid"
        ),
        "noise_correlation_length_mm": float(args.noise_correlation_length_mm),
        "noise_correlation_kernel": args.noise_correlation_kernel,
        "noise_normalized_first_singular_value": (
            float(s_noise_normalized[0]) if s_noise_normalized is not None else None
        ),
        "average_output_signal_amplitude": float(average_output_signal_amplitude),
        "average_output_power": float(average_output_power),
        "matrix_normalization": "disabled" if args.disable_matrix_normalization else "enabled",
        "source_power_normalization": args.source_power_normalization,
        "source_amplitude_scale": float(source_amplitude_scale),
        "input_power_convention": args.input_power_convention,
        "physical_total_source_power": float(physical_total_source_power),
        "total_input_power": float(total_input_power) if total_input_power is not None else None,
        "slq_frobenius_norm_sq": (
            float(slq_frobenius_norm_sq) if slq_frobenius_norm_sq is not None else None
        ),
        "slq_logdet_alpha": float(slq_logdet_alpha) if slq_logdet_alpha is not None else None,
        "slq_streaming": bool(args.slq_streaming),
        "slq_probe_parallel": bool(args.slq_probe_parallel),
        "slq_multi_gpu": bool(args.slq_multi_gpu),
        "slq_device_ids": slq_device_ids if slq_device_ids is not None else stream_device_ids,
        "saved_svd_path": saved_svd_path,
    }
    _emit_result_json(result_record, args.result_json_path)

    exit(0)

# @torch.no_grad()
# def bitrate_slq_torch(
#     apply_A, apply_AT, m, n,
#     noise_std_full_brain: float,
#     time_resolution: float = 1.0,
#     n_sources: int = 1,
#     n_detectors: int = 1,
#     s: int = 16, t: int = 40,
#     device: str = "cuda",
#     dtype = torch.float64,
# ):
#     ln2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))

#     # noise_var_eff = (noise_std_full_brain / (n_eff))**2
#     noise_var_eff = (noise_std_full_brain * (n_detectors**0.5) / (n_sources**0.5))**2
#     alpha = torch.tensor(1.0 / noise_var_eff, dtype=dtype, device=device)

#     use_left = (m <= n)
#     d = m if use_left else n

#     def apply_B(v):
#         return apply_A(apply_AT(v)) if use_left else apply_AT(apply_A(v))

#     est = torch.zeros((), dtype=dtype, device=device)

#     for _ in range(s):
#         # Rademacher probe
#         z = (torch.randint(0, 2, (d,), device=device, dtype=torch.int8) * 2 - 1).to(dtype)
#         norm_z = torch.linalg.vector_norm(z)
#         q = z / norm_z
#         q_prev = torch.zeros_like(q)

#         alphas = torch.zeros(t, dtype=dtype, device=device)
#         betas  = torch.zeros(t-1, dtype=dtype, device=device)
#         t_eff = t

#         for k in range(t):
#             w = apply_B(q)
#             if k > 0:
#                 w = w - betas[k-1] * q_prev
#             alpha_k = torch.dot(q, w)
#             w = w - alpha_k * q
#             alphas[k] = alpha_k
#             if k < t-1:
#                 beta_k = torch.linalg.vector_norm(w)
#                 betas[k] = beta_k
#                 if beta_k == 0:
#                     t_eff = k+1
#                     alphas = alphas[:t_eff]
#                     betas  = betas[:t_eff-1]
#                     break
#                 q_prev, q = q, (w / beta_k)

#         # Move tiny T to CPU for eigh (or use torch.linalg.eigh on GPU; both are fine)
#         T = torch.diag(alphas)
#         if t_eff > 1:
#             T += torch.diag(betas, 1) + torch.diag(betas, -1)
#         evals, evecs = torch.linalg.eigh(T)
#         weights = evecs[0, :]**2
#         quad = torch.dot(weights, torch.log1p(alpha * evals))
#         est += (norm_z**2) * quad

#     bits_per_sample = est / s / ln2
#     return (bits_per_sample / time_resolution).item()


# # Dense convenience wrapper
# @torch.no_grad()
# def bitrate_slq_dense_torch(A: torch.Tensor, noise_std_full_brain, time_resolution=1.0, n_sources=1, n_detectors=1, s=16, t=40):
#     m, n = A.shape
#     return bitrate_slq_torch(
#         apply_A=lambda x: (A @ x.to(A.dtype)).to(torch.float64),
#         apply_AT=lambda x: (A.T @ x.to(A.dtype)).to(torch.float64),
#         m=m, n=n,
#         noise_std_full_brain=noise_std_full_brain,
#         time_resolution=time_resolution,
#         n_sources=n_sources,
#         n_detectors=n_detectors,
#         s=s, t=t,
#         device=str(A.device),
#         dtype=torch.float64,
#     )


# # bitrate = bitrate_slq_dense_torch(G, noise_std_full_brain=1.0, time_resolution=time_step, n_detectors=n_sensors)
# bitrate = bitrate_slq_dense_torch(G, noise_std_full_brain=noise_level, time_resolution=1.0, n_sources=len(source_positions), n_detectors=len(sensor_positions))
# print(f"bitrate: {bitrate}")


if __name__ == "__main__":
    main()
