import argparse
import json
import os
import shlex
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

import modal


APP_NAME = "guti-us-analytical"
MOUNT_PATH = "/root/guti"
RESULT_JSON_PREFIX = "RESULT_JSON:"
MODAL_RESULT_JSON_PREFIX = "MODAL_RESULT_JSON:"


def _ignore_modal_mount(path: Path) -> bool:
    ignored_parts = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "results",
        "logs",
        "plots",
        "dist",
    }
    if any(part in ignored_parts for part in path.parts):
        return True
    return path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}


def _gpu_config():
    gpu_type = os.environ.get("MODAL_GPU_TYPE", "H100")
    gpu_count = int(os.environ.get("MODAL_GPU_COUNT", "1"))
    if gpu_count <= 1:
        return gpu_type
    return f"{gpu_type}:{gpu_count}"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "jax",
        "jaxlib",
        "jaxdf==0.2.8",
        "jwave==0.2.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(".", remote_path=MOUNT_PATH, ignore=_ignore_modal_mount)
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name("us-results", create_if_missing=True)


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return value


def _metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_sources", type=int, default=None)
    parser.add_argument("--n_sensors", type=int, default=None)
    parser.add_argument("--center_frequency", type=float, default=None)
    return parser


def _extract_metadata_from_args(args: list[str]) -> dict[str, Any]:
    parsed, _ = _metadata_parser().parse_known_args(args)
    frequency_hz = parsed.center_frequency
    frequency_khz = None if frequency_hz is None else int(round(frequency_hz / 1_000.0))
    return {
        "frequency_khz": frequency_khz,
        "n_sources": parsed.n_sources,
        "n_sensors": parsed.n_sensors,
    }


def _make_job_label(index: Optional[int], frequency_khz: Optional[int], n_sources: Optional[int], n_sensors: Optional[int]) -> str:
    prefix = f"{index:03d}_" if isinstance(index, int) else ""
    if frequency_khz is None or n_sources is None or n_sensors is None:
        return f"{prefix}us_analytical"
    return f"{prefix}{frequency_khz}khz_{n_sources}src_{n_sensors}sensors"


def _normalize_job_spec(job_or_args: Any) -> dict[str, Any]:
    if isinstance(job_or_args, dict):
        spec = dict(job_or_args)
        analytical_args = list(spec.get("analytical_args") or spec.get("args") or [])
        spec["analytical_args"] = analytical_args
        metadata = _extract_metadata_from_args(analytical_args)
        spec.setdefault("frequency_khz", metadata["frequency_khz"])
        spec.setdefault("n_sources", metadata["n_sources"])
        spec.setdefault("n_sensors", metadata["n_sensors"])
        return spec

    analytical_args = list(job_or_args)
    metadata = _extract_metadata_from_args(analytical_args)
    return {
        "index": None,
        "total": None,
        "frequency_khz": metadata["frequency_khz"],
        "n_sources": metadata["n_sources"],
        "n_sensors": metadata["n_sensors"],
        "analytical_args": analytical_args,
    }


def _with_result_output_args(
    analytical_args: list[str],
    result_json_path: Path,
    gram_output_path: Path | None,
) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    save_gram_matrix = False
    has_gram_output_path = False
    for idx, arg in enumerate(analytical_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--result_json_path":
            skip_next = True
            continue
        if arg.startswith("--result_json_path="):
            continue
        if arg == "--save_gram_matrix":
            save_gram_matrix = True
        if arg == "--gram_output_path":
            has_gram_output_path = True
        if arg.startswith("--gram_output_path="):
            has_gram_output_path = True
        sanitized.append(arg)
    sanitized.extend(["--result_json_path", str(result_json_path)])
    if save_gram_matrix and not has_gram_output_path and gram_output_path is not None:
        sanitized.extend(["--gram_output_path", str(gram_output_path)])
    return sanitized


def _requests_gram_save(analytical_args: list[str]) -> bool:
    return any(
        arg == "--save_gram_matrix" or arg.startswith("--save_gram_matrix=")
        for arg in analytical_args
    )


def _snapshot_npzs(results_dir: Path) -> dict[str, int]:
    return {
        str(path.relative_to(results_dir)): path.stat().st_mtime_ns
        for path in results_dir.rglob("*.npz")
    }


def _find_changed_npzs(results_dir: Path, before: dict[str, int]) -> list[Path]:
    changed: list[Path] = []
    for path in results_dir.rglob("*.npz"):
        rel = str(path.relative_to(results_dir))
        mtime_ns = path.stat().st_mtime_ns
        if before.get(rel) != mtime_ns:
            changed.append(path)
    changed.sort(key=lambda p: p.stat().st_mtime_ns)
    return changed


def _load_json_if_present(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_prefixed_json(log_text: str, prefix: str) -> Optional[dict[str, Any]]:
    matches = [line for line in log_text.splitlines() if line.startswith(prefix)]
    if not matches:
        return None
    payload = matches[-1][len(prefix) :].strip()
    if not payload:
        return None
    return json.loads(payload)


def _tail_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_and_stream(cmd: list[str], cwd: str, env: dict[str, str]) -> tuple[int, str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            print(line, end="")
            captured.append(line)
    return process.wait(), "".join(captured)


def _build_remote_result(job_spec: dict[str, Any]) -> dict[str, Any]:
    label = _make_job_label(
        job_spec.get("index"),
        job_spec.get("frequency_khz"),
        job_spec.get("n_sources"),
        job_spec.get("n_sensors"),
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="us_analytical_job_"))
    result_json_path = temp_dir / f"{label}.json"
    result_json_name = f"{label}.json"
    gram_output_path = Path("/modal_results/us_analytical_grams") / f"{label}_gram.npy"
    npz_download_name: Optional[str] = None
    npz_payload: Optional[bytes] = None

    cmd = ["python", "-m", "guti.modalities.us.analytical"] + _with_result_output_args(
        job_spec["analytical_args"],
        result_json_path,
        gram_output_path,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    results_dir = Path(MOUNT_PATH) / "results"
    before = _snapshot_npzs(results_dir)
    returncode, combined_output = _run_and_stream(cmd, cwd=MOUNT_PATH, env=env)
    changed_npzs = _find_changed_npzs(results_dir, before)
    analytical_record = _load_json_if_present(result_json_path)
    if analytical_record is None:
        analytical_record = _extract_prefixed_json(combined_output, RESULT_JSON_PREFIX)

    gram_size_bytes = None
    if _requests_gram_save(job_spec["analytical_args"]) and gram_output_path.exists():
        gram_size_bytes = gram_output_path.stat().st_size
        artifact_volume.commit()

    latest_npz = changed_npzs[-1] if changed_npzs else None
    if latest_npz is not None:
        npz_download_name = f"{label}__{latest_npz.name}"
        npz_payload = latest_npz.read_bytes()

    result_record = dict(analytical_record or {})
    result_record.update(
        {
            "status": "ok" if returncode == 0 else "failed",
            "index": job_spec.get("index"),
            "total": job_spec.get("total"),
            "label": label,
            "frequency_khz": job_spec.get("frequency_khz"),
            "n_sources": job_spec.get("n_sources"),
            "n_sensors": job_spec.get("n_sensors"),
            "analytical_args": job_spec["analytical_args"],
            "returncode": returncode,
            "modal_gpu_type": os.environ.get("MODAL_GPU_TYPE", "H100"),
            "modal_gpu_count": int(os.environ.get("MODAL_GPU_COUNT", "1")),
            "changed_npz_relpaths": [str(path.relative_to(results_dir)) for path in changed_npzs],
            "modal_gram_output_path": str(gram_output_path) if gram_size_bytes is not None else None,
            "modal_gram_size_bytes": gram_size_bytes,
            "output_npz_name": latest_npz.name if latest_npz is not None else None,
            "output_npz_relpath": (
                str(latest_npz.relative_to(results_dir)) if latest_npz is not None else None
            ),
        }
    )
    if returncode != 0:
        result_record.setdefault(
            "exception",
            f"analytical subprocess exited with status {returncode}",
        )
        result_record["combined_output_tail"] = _tail_text(combined_output)

    sanitized_record = _sanitize_json_value(result_record)
    result_json_path.write_text(
        json.dumps(sanitized_record, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{MODAL_RESULT_JSON_PREFIX} "
        f"{json.dumps(sanitized_record, sort_keys=True, allow_nan=False)}",
        flush=True,
    )

    return {
        "status": sanitized_record["status"],
        "index": sanitized_record.get("index"),
        "total": sanitized_record.get("total"),
        "frequency_khz": sanitized_record.get("frequency_khz"),
        "n_sources": sanitized_record.get("n_sources"),
        "n_sensors": sanitized_record.get("n_sensors"),
        "bitrate": sanitized_record.get("bitrate"),
        "returncode": returncode,
        "result_record": sanitized_record,
        "result_json_name": result_json_name,
        "result_json_payload": result_json_path.read_bytes(),
        "npz_name": latest_npz.name if latest_npz is not None else None,
        "npz_download_name": npz_download_name,
        "npz_payload": npz_payload,
    }


def _failure_result(job_spec: dict[str, Any], exc: Exception) -> dict[str, Any]:
    label = _make_job_label(
        job_spec.get("index"),
        job_spec.get("frequency_khz"),
        job_spec.get("n_sources"),
        job_spec.get("n_sensors"),
    )
    result_record = _sanitize_json_value(
        {
            "status": "failed",
            "index": job_spec.get("index"),
            "total": job_spec.get("total"),
            "label": label,
            "frequency_khz": job_spec.get("frequency_khz"),
            "n_sources": job_spec.get("n_sources"),
            "n_sensors": job_spec.get("n_sensors"),
            "analytical_args": job_spec.get("analytical_args"),
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "modal_gpu_type": os.environ.get("MODAL_GPU_TYPE", "H100"),
            "modal_gpu_count": int(os.environ.get("MODAL_GPU_COUNT", "1")),
        }
    )
    payload = (json.dumps(result_record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    print(
        f"{MODAL_RESULT_JSON_PREFIX} "
        f"{json.dumps(result_record, sort_keys=True, allow_nan=False)}",
        flush=True,
    )
    return {
        "status": "failed",
        "index": result_record.get("index"),
        "total": result_record.get("total"),
        "frequency_khz": result_record.get("frequency_khz"),
        "n_sources": result_record.get("n_sources"),
        "n_sensors": result_record.get("n_sensors"),
        "bitrate": None,
        "returncode": 1,
        "result_record": result_record,
        "result_json_name": f"{label}.json",
        "result_json_payload": payload,
        "npz_name": None,
        "npz_download_name": None,
        "npz_payload": None,
    }


@app.function(
    image=image,
    gpu=_gpu_config(),
    cpu=int(os.environ.get("MODAL_CPU", "8")),
    memory=int(os.environ.get("MODAL_MEMORY", "32768")),
    timeout=int(os.environ.get("MODAL_TIMEOUT", "60")) * 60,
    single_use_containers=True,
    volumes={"/modal_results": artifact_volume},
)
def run_us_analytical(job_or_args: Any) -> dict[str, Any]:
    job_spec = _normalize_job_spec(job_or_args)
    try:
        return _build_remote_result(job_spec)
    except Exception as exc:
        return _failure_result(job_spec, exc)


def _write_local_artifacts(result: dict[str, Any], output_dir: Path) -> tuple[Path, Optional[Path]]:
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / result["result_json_name"]
    json_path.write_bytes(result["result_json_payload"])

    npz_path: Optional[Path] = None
    if result.get("npz_payload") is not None:
        npz_name = result.get("npz_download_name") or result.get("npz_name")
        if npz_name:
            npz_path = output_dir / npz_name
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.write_bytes(result["npz_payload"])
    return json_path, npz_path


@app.local_entrypoint()
def main(*cli_args):
    parser = argparse.ArgumentParser(
        description="Run guti/modalities/us/analytical.py on Modal with the same CLI."
    )
    parser.add_argument(
        "--modal-output-dir",
        default="results_modal",
        help="Where to save downloaded per-job JSON and any changed npz output",
    )
    passthrough_args = list(cli_args)
    if not passthrough_args:
        env_args = os.environ.get("MODAL_ARGS")
        if env_args:
            passthrough_args = shlex.split(env_args)
    args, passthrough = parser.parse_known_args(passthrough_args)

    result = run_us_analytical.remote(passthrough)
    output_dir = Path(args.modal_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = _write_local_artifacts(result, output_dir)
    print(f"Downloaded {json_path}")
    if npz_path is not None:
        print(f"Downloaded {npz_path}")
    if result["status"] != "ok":
        raise SystemExit(result.get("returncode") or 1)
