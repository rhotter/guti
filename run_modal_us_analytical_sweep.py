#!/usr/bin/env python3
"""Run modal_us_analytical.py over a source/sensor/frequency sweep.

This sweep targets the SVD bitrate path in ``guti.modalities.us.analytical``.
Any unknown flags are forwarded directly to the underlying analytical script,
so branch-specific options can be passed without changing this driver.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_FREQUENCIES_KHZ = [50, 100, 150, 250]


def linspace_int(start: int, stop: int, num_points: int) -> list[int]:
    """Return an inclusive integer linspace."""
    if num_points < 1:
        raise ValueError("num_points must be at least 1")
    if num_points == 1:
        return [start]

    step = (stop - start) / (num_points - 1)
    return [round(start + i * step) for i in range(num_points)]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Modal ultrasound analytical runs over sources, sensors, and "
            "frequencies using this branch's approximate bitrate path."
        )
    )
    parser.add_argument(
        "--source-counts",
        type=parse_int_csv,
        default=linspace_int(10_000, 200_000, 5),
        help="Comma-separated source counts. Default: 10000,57500,105000,152500,200000",
    )
    parser.add_argument(
        "--sensor-counts",
        type=parse_int_csv,
        default=linspace_int(1_000, 15_000, 5),
        help="Comma-separated sensor counts. Default: 1000,4500,8000,11500,15000",
    )
    parser.add_argument(
        "--frequencies-khz",
        type=parse_int_csv,
        default=DEFAULT_FREQUENCIES_KHZ,
        help="Comma-separated center frequencies in kHz. Default: 50,100,150,250",
    )
    parser.add_argument(
        "--temporal-sampling",
        type=int,
        default=5,
        help="Temporal downsampling factor passed to analytical.py.",
    )
    parser.add_argument(
        "--sensor-batch-size",
        type=int,
        default=256,
        help="Sensor batch size passed to analytical.py.",
    )
    parser.add_argument(
        "--bitrate-method",
        choices=["svd", "slq", "both"],
        default="svd",
        help=(
            "Bitrate path to use in guti.modalities.us.analytical. "
            "Default: svd"
        ),
    )
    parser.add_argument(
        "--svd-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Optional --svd_device value forwarded to analytical.py.",
    )
    parser.add_argument(
        "--accumulate-on-cpu",
        action="store_true",
        help="Pass --accumulate_on_cpu through to analytical.py.",
    )
    parser.add_argument(
        "--modal-binary",
        default="modal",
        help="Modal CLI executable to use.",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "cli", "sdk"],
        default="auto",
        help=(
            "Execution backend for launching Modal work. "
            "'sdk' uses the local Modal Python package, 'cli' shells out to "
            "'modal run', and 'auto' prefers the SDK when available."
        ),
    )
    parser.add_argument(
        "--modal-output-dir",
        default=None,
        help="Optional output directory passed to modal_us_analytical.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent Modal runs to launch. Default: 1",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/us_analytical_sweep",
        help="Directory for per-run logs when executing the sweep.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one run fails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only execute the first N runs after expansion. Useful for spot checks.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index of the first expanded run to execute. Default: 1",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="1-based index of the last expanded run to execute, inclusive.",
    )
    return parser


def make_run_label(frequency_khz: int, n_sources: int, n_sensors: int) -> str:
    return f"{frequency_khz}khz_{n_sources}src_{n_sensors}sensors"


def maybe_add_auto_slq_flags(
    bitrate_method: str,
    passthrough_args: list[str],
) -> tuple[list[str], list[str]]:
    effective_args = list(passthrough_args)
    added_flags: list[str] = []

    has_streaming_flag = "--slq_streaming" in effective_args
    has_multi_gpu_flag = "--slq_multi_gpu" in effective_args
    if bitrate_method == "slq" and not has_streaming_flag and not has_multi_gpu_flag:
        effective_args.append("--slq_streaming")
        added_flags.append("--slq_streaming")

    modal_gpu_count = int(os.environ.get("MODAL_GPU_COUNT", "1"))
    has_parallel_flag = any(
        flag in effective_args for flag in ("--slq_probe_parallel", "--slq_multi_gpu")
    )
    if (
        bitrate_method in {"slq", "both"}
        and modal_gpu_count > 1
        and not has_parallel_flag
    ):
        effective_args.append("--slq_probe_parallel")
        added_flags.append("--slq_probe_parallel")
    return effective_args, added_flags


def build_analytical_args(
    n_sources: int,
    n_sensors: int,
    frequency_khz: int,
    temporal_sampling: int,
    sensor_batch_size: int,
    bitrate_method: str,
    svd_device: str | None,
    accumulate_on_cpu: bool,
    passthrough_args: list[str],
) -> list[str]:
    analytical_args = [
        "--n_sources",
        str(n_sources),
        "--n_sensors",
        str(n_sensors),
        "--center_frequency",
        str(frequency_khz * 1_000),
        "--temporal_sampling",
        str(temporal_sampling),
        "--sensor_batch_size",
        str(sensor_batch_size),
        "--bitrate_method",
        bitrate_method,
    ]
    if svd_device is not None:
        analytical_args.extend(["--svd_device", svd_device])
    if accumulate_on_cpu:
        analytical_args.append("--accumulate_on_cpu")
    analytical_args.extend(passthrough_args)
    return analytical_args


def build_command(
    modal_binary: str,
    modal_script: Path,
    analytical_args: list[str],
    modal_output_dir: str | None,
) -> list[str]:
    command = [modal_binary, "run", str(modal_script)]
    if modal_output_dir is not None:
        command.extend(["--modal-output-dir", modal_output_dir])

    command.extend(analytical_args)
    return command


def try_import_modal_sdk():
    try:
        import modal
        import modal_us_analytical

        return modal, modal_us_analytical, None
    except ImportError as exc:
        return None, None, exc


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def append_summary_record(log_dir: Path, record: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary_path = log_dir / "runs.jsonl"
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_sdk_job_artifacts(output_dir: Path, output: dict) -> tuple[Path, Path | None]:
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / output["result_json_name"]
    json_path.write_bytes(output["result_json_payload"])

    npz_path: Path | None = None
    if output.get("npz_payload") is not None:
        npz_name = output.get("npz_download_name") or output.get("npz_name")
        if npz_name:
            npz_path = output_dir / npz_name
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.write_bytes(output["npz_payload"])
    return json_path, npz_path


def redownload_sdk_job_jsons(output_dir: Path, outputs: list[dict]) -> Path:
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "all_job_results.jsonl"
    sorted_outputs = sorted(
        outputs,
        key=lambda item: (
            item.get("index") is None,
            item.get("index") if item.get("index") is not None else 0,
        ),
    )
    with aggregate_path.open("w", encoding="utf-8") as aggregate:
        for output in sorted_outputs:
            json_path = json_dir / output["result_json_name"]
            json_path.write_bytes(output["result_json_payload"])
            aggregate.write(
                json.dumps(output["result_record"], sort_keys=True, allow_nan=False) + "\n"
            )
    return aggregate_path


def run_with_modal_sdk(
    run_specs: list[dict],
    jobs: int,
    modal_output_dir: str | None,
    log_dir_arg: str,
    continue_on_error: bool,
) -> int:
    modal, modal_module, import_error = try_import_modal_sdk()
    if import_error is not None:
        raise RuntimeError(
            "Modal SDK runner requested but the local 'modal' package is unavailable. "
            "Activate the right environment first or use --runner cli."
        ) from import_error

    output_dir = Path(modal_output_dir or "results_modal")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir_arg)
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_by_index = {spec["index"]: spec for spec in run_specs}
    downloaded_outputs: list[dict] = []

    failure_code: int | None = None
    total_batches = (len(run_specs) + jobs - 1) // jobs
    with modal.enable_output():
        with modal_module.app.run():
            for batch_index, batch in enumerate(batched(run_specs, jobs), start=1):
                print(f"\nSubmitting SDK batch {batch_index}/{total_batches} ({len(batch)} runs)")
                stop_after_batch = False
                try:
                    for output in modal_module.run_us_analytical.map(
                        batch,
                        order_outputs=False,
                    ):
                        downloaded_outputs.append(output)
                        json_path, npz_path = write_sdk_job_artifacts(output_dir, output)
                        spec = spec_by_index.get(output["index"])
                        record = {
                            "runner": "sdk",
                            "status": output["status"],
                            "index": output["index"],
                            "total": output["total"],
                            "frequency_khz": output["frequency_khz"],
                            "n_sources": output["n_sources"],
                            "n_sensors": output["n_sensors"],
                            "analytical_args": spec["analytical_args"] if spec is not None else None,
                            "command": spec["command"] if spec is not None else None,
                            "returncode": output.get("returncode"),
                            "bitrate": output.get("bitrate"),
                            "json_downloaded": str(json_path),
                            "npz_downloaded": str(npz_path) if npz_path is not None else None,
                            "has_payload": npz_path is not None,
                        }
                        append_summary_record(log_dir, record)
                        bitrate_text = (
                            f" bitrate={output['bitrate']}"
                            if output.get("bitrate") is not None
                            else ""
                        )
                        npz_text = (
                            f" npz={npz_path}"
                            if npz_path is not None
                            else " no npz result found"
                        )
                        print(
                            f"[{output['status']}] {output['index']}/{output['total']} "
                            f"freq={output['frequency_khz']}kHz "
                            f"sources={output['n_sources']} sensors={output['n_sensors']}"
                            f"{bitrate_text} json={json_path}{npz_text}"
                        )
                        if output["status"] != "ok" and failure_code is None:
                            failure_code = output.get("returncode") or 1
                            if not continue_on_error:
                                stop_after_batch = True
                except Exception as exc:
                    append_summary_record(
                        log_dir,
                        {
                            "runner": "sdk",
                            "status": "failed",
                            "index": None,
                            "total": len(run_specs),
                            "frequency_khz": None,
                            "n_sources": None,
                            "n_sensors": None,
                            "analytical_args": None,
                            "command": None,
                            "returncode": None,
                            "bitrate": None,
                            "json_downloaded": None,
                            "npz_downloaded": None,
                            "has_payload": False,
                            "exception": str(exc),
                            "batch_index": batch_index,
                        },
                    )
                    aggregate_path = redownload_sdk_job_jsons(output_dir, downloaded_outputs)
                    print(f"Re-downloaded {len(downloaded_outputs)} job JSONs to {output_dir / 'json'}")
                    print(f"Wrote aggregate SDK results to {aggregate_path}")
                    raise
                if stop_after_batch:
                    aggregate_path = redownload_sdk_job_jsons(output_dir, downloaded_outputs)
                    print(f"Re-downloaded {len(downloaded_outputs)} job JSONs to {output_dir / 'json'}")
                    print(f"Wrote aggregate SDK results to {aggregate_path}")
                    return failure_code or 1

    aggregate_path = redownload_sdk_job_jsons(output_dir, downloaded_outputs)
    print(f"Re-downloaded {len(downloaded_outputs)} job JSONs to {output_dir / 'json'}")
    print(f"Wrote aggregate SDK results to {aggregate_path}")

    if failure_code is not None and not continue_on_error:
        return failure_code
    return 0


def run_with_cli(
    run_specs: list[dict],
    jobs: int,
    log_dir_arg: str,
    continue_on_error: bool,
) -> int:
    if jobs == 1:
        log_dir = Path(log_dir_arg)
        log_dir.mkdir(parents=True, exist_ok=True)
        for spec in run_specs:
            result = subprocess.run(spec["command"], check=False)
            append_summary_record(
                log_dir,
                {
                    "runner": "cli",
                    "status": "ok" if result.returncode == 0 else "failed",
                    "index": spec["index"],
                    "total": spec["total"],
                    "frequency_khz": spec["frequency_khz"],
                    "n_sources": spec["n_sources"],
                    "n_sensors": spec["n_sensors"],
                    "analytical_args": spec["analytical_args"],
                    "command": spec["command"],
                    "returncode": result.returncode,
                },
            )
            if result.returncode == 0:
                continue
            print(f"Run failed with exit code {result.returncode}", file=sys.stderr)
            if not continue_on_error:
                return result.returncode
        return 0

    log_dir = Path(log_dir_arg)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending_specs = iter(run_specs)
    active_futures = {}
    failure_code: int | None = None
    continue_submitting = True

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        while len(active_futures) < jobs:
            try:
                spec = next(pending_specs)
            except StopIteration:
                break
            future = executor.submit(execute_command, **spec, log_dir=log_dir)
            active_futures[future] = spec

        while active_futures:
            future = next(as_completed(active_futures))
            result = future.result()
            del active_futures[future]
            append_summary_record(
                log_dir,
                {
                    "runner": "cli",
                    "status": "ok" if result["returncode"] == 0 else "failed",
                    "index": result["index"],
                    "total": result["total"],
                    "frequency_khz": result["frequency_khz"],
                    "n_sources": result["n_sources"],
                    "n_sensors": result["n_sensors"],
                    "command": result["command"],
                    "returncode": result["returncode"],
                    "elapsed_seconds": round(result["elapsed"], 3),
                    "log_path": str(result["log_path"]),
                },
            )
            print_run_result(result)

            if result["returncode"] != 0 and failure_code is None:
                failure_code = result["returncode"]
                if not continue_on_error:
                    continue_submitting = False

            if continue_submitting:
                try:
                    spec = next(pending_specs)
                except StopIteration:
                    spec = None
                if spec is not None:
                    new_future = executor.submit(execute_command, **spec, log_dir=log_dir)
                    active_futures[new_future] = spec

    if failure_code is not None and not continue_on_error:
        return failure_code
    return 0


def execute_command(
    index: int,
    total: int,
    frequency_khz: int,
    n_sources: int,
    n_sensors: int,
    command: list[str],
    log_dir: Path,
) -> dict:
    start = time.perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    label = make_run_label(frequency_khz, n_sources, n_sensors)
    log_path = log_dir / f"{index:03d}_{label}.log"
    log_text = (
        f"run={index}/{total}\n"
        f"label={label}\n"
        f"command={shlex.join(command)}\n"
        f"returncode={result.returncode}\n"
        f"elapsed_seconds={elapsed:.2f}\n"
        "\n[stdout]\n"
        f"{result.stdout}"
        "\n[stderr]\n"
        f"{result.stderr}"
    )
    log_path.write_text(log_text)

    return {
        "index": index,
        "total": total,
        "frequency_khz": frequency_khz,
        "n_sources": n_sources,
        "n_sensors": n_sensors,
        "command": command,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "log_path": log_path,
    }


def print_run_banner(index: int, total: int, frequency_khz: int, n_sources: int, n_sensors: int) -> None:
    print(
        f"\n[{index}/{total}] "
        f"freq={frequency_khz}kHz sources={n_sources} sensors={n_sensors}"
    )


def print_run_result(result: dict) -> None:
    status = "ok" if result["returncode"] == 0 else "failed"
    print(
        f"[{status}] {result['index']}/{result['total']} "
        f"freq={result['frequency_khz']}kHz "
        f"sources={result['n_sources']} sensors={result['n_sensors']} "
        f"in {result['elapsed']:.1f}s "
        f"log={result['log_path']}"
    )


def main() -> int:
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args()
    effective_passthrough_args, auto_added_flags = maybe_add_auto_slq_flags(
        args.bitrate_method,
        passthrough_args,
    )

    modal_script = Path(__file__).resolve().with_name("modal_us_analytical.py")
    source_counts = args.source_counts
    sensor_counts = args.sensor_counts
    frequencies_khz = args.frequencies_khz

    runs: list[tuple[int, int, int]] = []
    for frequency_khz in frequencies_khz:
        for n_sources in source_counts:
            for n_sensors in sensor_counts:
                runs.append((frequency_khz, n_sources, n_sensors))

    if args.start_index < 1:
        parser.error("--start-index must be at least 1")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index")

    start_offset = args.start_index - 1
    end_offset = args.end_index
    runs = runs[start_offset:end_offset]

    if args.limit is not None:
        runs = runs[: args.limit]

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    print("Modal script:", modal_script)
    print("Source counts:", source_counts)
    print("Sensor counts:", sensor_counts)
    print("Frequencies (kHz):", frequencies_khz)
    print("Bitrate method:", args.bitrate_method)
    print("Parallel jobs:", args.jobs)
    if args.start_index != 1 or args.end_index is not None:
        selected_end = start_offset + len(runs)
        print(f"Run index range: {args.start_index}-{selected_end}")
    if effective_passthrough_args:
        print("Forwarded args:", shlex.join(effective_passthrough_args))
    for flag in auto_added_flags:
        print(f"Auto-added flag: {flag}")
    print("Total runs:", len(runs))

    run_specs: list[dict] = []
    for index, (frequency_khz, n_sources, n_sensors) in enumerate(runs, start=1):
        analytical_args = build_analytical_args(
            n_sources=n_sources,
            n_sensors=n_sensors,
            frequency_khz=frequency_khz,
            temporal_sampling=args.temporal_sampling,
            sensor_batch_size=args.sensor_batch_size,
            bitrate_method=args.bitrate_method,
            svd_device=args.svd_device,
            accumulate_on_cpu=args.accumulate_on_cpu,
            passthrough_args=effective_passthrough_args,
        )
        command = build_command(
            modal_binary=args.modal_binary,
            modal_script=modal_script,
            analytical_args=analytical_args,
            modal_output_dir=args.modal_output_dir,
        )
        run_specs.append(
            {
                "index": index,
                "total": len(runs),
                "frequency_khz": frequency_khz,
                "n_sources": n_sources,
                "n_sensors": n_sensors,
                "analytical_args": analytical_args,
                "command": command,
            }
        )

    runner = args.runner
    if runner == "auto":
        modal, modal_module, import_error = try_import_modal_sdk()
        if import_error is None:
            runner = "sdk"
        else:
            runner = "cli"
    print("Runner:", runner)

    for spec in run_specs:
        print_run_banner(
            spec["index"],
            spec["total"],
            spec["frequency_khz"],
            spec["n_sources"],
            spec["n_sensors"],
        )
        command = spec["command"]
        print(shlex.join(command))

    if args.dry_run:
        return 0

    if runner == "sdk":
        return run_with_modal_sdk(
            run_specs=run_specs,
            jobs=args.jobs,
            modal_output_dir=args.modal_output_dir,
            log_dir_arg=args.log_dir,
            continue_on_error=args.continue_on_error,
        )

    return run_with_cli(
        run_specs=run_specs,
        jobs=args.jobs,
        log_dir_arg=args.log_dir,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
