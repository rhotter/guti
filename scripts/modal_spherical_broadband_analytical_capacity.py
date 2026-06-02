import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "guti-spherical-broadband-analytical-capacity"
MOUNT_PATH = "/root/guti"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "scipy",
        "matplotlib",
    )
    .add_local_dir(".", remote_path=MOUNT_PATH)
)

app = modal.App(APP_NAME)


def _load_analytical_module():
    if MOUNT_PATH not in sys.path:
        sys.path.insert(0, MOUNT_PATH)
    import plot_spherical_broadband_analytical_capacity as analytical

    return analytical


def _build_args_namespace(settings: dict[str, Any]):
    analytical = _load_analytical_module()
    args = analytical.build_parser().parse_args([])
    for key, value in settings.items():
        setattr(args, key, value)
    return analytical, args


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@app.function(
    image=image,
    cpu=int(os.environ.get("MODAL_CPU", "8")),
    memory=int(os.environ.get("MODAL_MEMORY", "32768")),
    timeout=int(os.environ.get("MODAL_TIMEOUT", "60")) * 60,
    single_use_containers=True,
)
def estimate_spherical_broadband_point(job: dict[str, Any]) -> dict[str, float | int]:
    analytical, args = _build_args_namespace(job["settings"])
    return analytical.estimate_frequency_result(
        float(job["frequency_khz"]),
        args,
        progress_prefix=None,
    )


@app.local_entrypoint()
def main(*cli_args):
    parser = argparse.ArgumentParser(
        description=(
            "Run the spherical broadband analytical bitrate/capacity sweep on "
            "Modal CPU workers and aggregate the resulting plots locally."
        )
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Maximum number of Modal CPU jobs to submit at once. Default: 8",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frequency jobs without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one frequency fails.",
    )
    passthrough_args = list(cli_args)
    if not passthrough_args:
        env_args = os.environ.get("MODAL_ARGS")
        if env_args:
            passthrough_args = shlex.split(env_args)
    meta_args, passthrough = parser.parse_known_args(passthrough_args)

    analytical = _load_analytical_module()
    target_args = analytical.build_parser().parse_args(passthrough)
    frequencies_khz = analytical.resolve_frequencies(target_args)

    if meta_args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    settings = vars(target_args).copy()
    settings["frequencies_khz"] = None

    print("Frequencies (kHz):", [float(freq) for freq in frequencies_khz])
    print("Parallel Modal CPU jobs:", meta_args.jobs)
    print("Output directory:", target_args.outdir)

    if meta_args.dry_run:
        for idx, freq_khz in enumerate(frequencies_khz, start=1):
            print(f"[{idx}/{len(frequencies_khz)}] freq={freq_khz:.1f} kHz")
        return

    results: list[dict[str, float | int]] = []
    failure = False
    total_batches = (len(frequencies_khz) + meta_args.jobs - 1) // meta_args.jobs

    with modal.enable_output():
        for batch_index, batch in enumerate(
            batched([float(freq) for freq in frequencies_khz], meta_args.jobs),
            start=1,
        ):
            print(
                f"\nSubmitting batch {batch_index}/{total_batches} "
                f"({len(batch)} frequencies)"
            )
            jobs = [
                {
                    "frequency_khz": freq_khz,
                    "settings": settings,
                }
                for freq_khz in batch
            ]
            outputs = list(
                estimate_spherical_broadband_point.map(
                    jobs,
                    order_outputs=True,
                    return_exceptions=True,
                    wrap_returned_exceptions=False,
                )
            )
            for freq_khz, output in zip(batch, outputs):
                if isinstance(output, Exception):
                    failure = True
                    print(f"[failed] freq={freq_khz:.1f} kHz exception={output}")
                    if not meta_args.continue_on_error:
                        raise SystemExit(1)
                    continue

                results.append(output)
                print(
                    f"[ok] freq={freq_khz:.1f} kHz "
                    f"bitrate={output['bitrate']:.3f} "
                    f"channel_capacity={output['channel_capacity']:.3f} "
                    f"mode_scaled_capacity={output['mode_scaled_capacity']:.3f} "
                    f"mode_count={output['mode_count']} "
                    f"ell_max={output['ell_max']} "
                    f"n_omega={output['n_omega']} "
                    f"n_r={output['n_r']} "
                    f"converged={bool(output.get('converged', 0))} "
                    f"rounds={output.get('rounds', 1)}"
                )

    if not results:
        raise SystemExit(1)

    outdir = Path(target_args.outdir)
    analytical.write_output_artifacts(results, outdir)
    print(f"\nWrote {outdir / 'bitrate_vs_frequency.png'}")
    print(f"Wrote {outdir / 'bitrate_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'channel_capacity_vs_frequency.png'}")
    print(f"Wrote {outdir / 'channel_capacity_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_scaled_capacity_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_scaled_capacity_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_count_vs_frequency.png'}")
    print(f"Wrote {outdir / 'mode_count_over_f3_vs_frequency.png'}")
    print(f"Wrote {outdir / 'average_mode_snr_vs_frequency.png'}")
    print(f"Wrote {outdir / 'top_singular_value_vs_frequency.png'}")
    print(f"Wrote {outdir / 'spherical_broadband_capacity_vs_frequency.npz'}")

    if failure and not meta_args.continue_on_error:
        raise SystemExit(1)
