"""Run any GUTI modality on Modal.

This is the general-purpose remote runner for the unified
``run_modality.py <modality> [--param value ...]`` entrypoint. It mounts the
repo into a Modal container, selects a per-modality resource envelope, executes
the local modality runner remotely, and downloads newly-created ``results/*.npz``
files into ``results_modal/`` by default.

Examples:
    modal run scripts/modal_run.py blur_1d --num_brain_grid_points 256
    modal run scripts/modal_run.py cw_fnirs --num_sensors 400
    modal run scripts/modal_run.py td_fnirs --scaled-up --no-save
    modal run scripts/modal_run.py eeg --num_sensors 256
    modal run scripts/modal_run.py meg_squid --scaled-up
    modal run scripts/modal_run.py us --scaled-up

Flags after the modality name are passed through to ``run_modality.py``.

Resource overrides:
    MODAL_GPU_TYPE, MODAL_GPU_COUNT, MODAL_CPU, MODAL_MEMORY, MODAL_TIMEOUT
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

APP_NAME = "guti-modality"
MOUNT_PATH = "/root/guti"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModalConfig:
    """Compute envelope for a remote modality run."""

    image: str = "base"
    gpu: str | None = None
    cpu: float = 4.0
    memory: int = 16_384
    timeout_min: int = 60


@dataclass(frozen=True)
class RunTarget:
    """Resolved modality name plus optional arguments to prepend."""

    run_modality_name: str
    config_key: str
    prepended_args: tuple[str, ...] = ()


DEFAULT_CONFIG = ModalConfig()

MODAL_CONFIGS: dict[str, ModalConfig] = {
    "blur_1d": ModalConfig(image="base", gpu=None, cpu=2, memory=4_096, timeout_min=30),
    "cw_fnirs": ModalConfig(image="base", gpu="T4", cpu=4, memory=16_384, timeout_min=60),
    "td_fnirs": ModalConfig(image="base", gpu="T4", cpu=4, memory=16_384, timeout_min=60),
    "eeg": ModalConfig(image="openmeeg", gpu=None, cpu=8, memory=16_384, timeout_min=60),
    "meg": ModalConfig(image="base", gpu=None, cpu=8, memory=16_384, timeout_min=60),
    "us": ModalConfig(image="base", gpu="H100", cpu=8, memory=65_536, timeout_min=120),
}

MODALITY_ALIASES: dict[str, RunTarget] = {
    "1d_blurring": RunTarget("blur_1d", "blur_1d"),
    "meg_opm": RunTarget("meg", "meg", ("--sensor_offset_mm", "7.0")),
    "meg_squid": RunTarget("meg", "meg", ("--sensor_offset_mm", "25.0")),
    "td_fnirs_analytical": RunTarget("td_fnirs", "td_fnirs"),
    "us_analytical": RunTarget("us", "us"),
}

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
        "results_modal",
        "logs",
        "plots",
        "dist",
    }
    if any(part in ignored_parts for part in path.parts):
        return True
    return path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}


def _with_repo(img: modal.Image) -> modal.Image:
    return img.add_local_dir(str(REPO_ROOT), remote_path=MOUNT_PATH, ignore=_ignore_modal_mount)


base_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "h5py",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

openmeeg_image = _with_repo(
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("openmeeg", channels=["conda-forge"])
    .pip_install("numpy", "scipy", "matplotlib", "tqdm", "h5py", "torch")
)

app = modal.App(APP_NAME)


def _available_modality_dirs() -> set[str]:
    modalities_dir = REPO_ROOT / "guti" / "modalities"
    if not modalities_dir.is_dir():
        return set()
    return {
        path.name
        for path in modalities_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "modality.py").exists()
    }


def _has_param(args: list[str], key: str) -> bool:
    flag = f"--{key}"
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _merge_prepended_args(prepended: tuple[str, ...], passthrough: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(prepended):
        item = prepended[index]
        if item.startswith("--") and index + 1 < len(prepended):
            key = item[2:]
            value = prepended[index + 1]
            if not _has_param(passthrough, key):
                merged.extend([item, value])
            index += 2
            continue
        merged.append(item)
        index += 1
    return merged + passthrough


def resolve_target(modality_name: str, passthrough: list[str]) -> tuple[RunTarget, list[str]]:
    """Resolve user-facing modality aliases to ``run_modality.py`` arguments."""
    if modality_name in MODALITY_ALIASES:
        target = MODALITY_ALIASES[modality_name]
        return target, _merge_prepended_args(target.prepended_args, passthrough)

    if modality_name in MODAL_CONFIGS:
        return RunTarget(modality_name, modality_name), passthrough

    if modality_name in _available_modality_dirs():
        return RunTarget(modality_name, modality_name), passthrough

    supported = sorted(_available_modality_dirs() | set(MODALITY_ALIASES))
    raise SystemExit(
        f"Unknown modality {modality_name!r}. Supported names: {', '.join(supported)}"
    )


def _resolve_config(config_key: str) -> ModalConfig:
    cfg = MODAL_CONFIGS.get(config_key, DEFAULT_CONFIG)

    gpu_type = os.environ.get("MODAL_GPU_TYPE")
    if gpu_type is not None:
        normalized_gpu_type = gpu_type.strip()
        if normalized_gpu_type.lower() in {"", "none", "cpu", "false", "0"}:
            gpu: str | None = None
        else:
            count = int(os.environ.get("MODAL_GPU_COUNT", "1"))
            gpu = normalized_gpu_type if count <= 1 else f"{normalized_gpu_type}:{count}"
        cfg = ModalConfig(cfg.image, gpu, cfg.cpu, cfg.memory, cfg.timeout_min)

    if "MODAL_CPU" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, float(os.environ["MODAL_CPU"]), cfg.memory, cfg.timeout_min)
    if "MODAL_MEMORY" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, cfg.cpu, int(os.environ["MODAL_MEMORY"]), cfg.timeout_min)
    if "MODAL_TIMEOUT" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, cfg.cpu, cfg.memory, int(os.environ["MODAL_TIMEOUT"]))
    return cfg


def _snapshot_npzs(results_dir: Path) -> dict[str, int]:
    if not results_dir.is_dir():
        return {}
    return {
        str(path.relative_to(results_dir)): path.stat().st_mtime_ns
        for path in results_dir.rglob("*.npz")
    }


def _changed_npzs(results_dir: Path, before: dict[str, int]) -> list[Path]:
    changed: list[Path] = []
    for path in results_dir.rglob("*.npz"):
        rel = str(path.relative_to(results_dir))
        if before.get(rel) != path.stat().st_mtime_ns:
            changed.append(path)
    changed.sort(key=lambda path: path.stat().st_mtime_ns)
    return changed


def _stream(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            print(line, end="")
    return process.wait()


def _run_modality(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    results_dir = Path(MOUNT_PATH) / "results"
    before = _snapshot_npzs(results_dir)

    cmd = ["python", "run_modality.py", modality_name, *passthrough_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[modal_run] $ {shlex.join(cmd)}", flush=True)
    returncode = _stream(cmd, cwd=MOUNT_PATH, env=env)

    npzs: dict[str, bytes] = {}
    for path in _changed_npzs(results_dir, before):
        npzs[str(path.relative_to(results_dir))] = path.read_bytes()

    return {"modality": modality_name, "returncode": returncode, "npz_files": npzs}


# One function variant per (image, gpu, cpu, memory, timeout) combination that
# any modality in MODAL_CONFIGS uses.  Modal 1.x removed Function.with_options()
# so resources must be fixed at definition time.
@app.function(image=base_image, gpu=None, cpu=2, memory=4_096, timeout=30 * 60)
def run_base_small(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)

@app.function(image=base_image, gpu="T4", cpu=4, memory=16_384, timeout=60 * 60)
def run_base_t4(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)

@app.function(image=base_image, gpu=None, cpu=8, memory=16_384, timeout=60 * 60)
def run_base_cpu8(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)

@app.function(image=base_image, gpu="H100", cpu=8, memory=65_536, timeout=120 * 60)
def run_base_h100(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)

@app.function(image=openmeeg_image, gpu=None, cpu=8, memory=16_384, timeout=60 * 60)
def run_openmeeg_cpu8(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)


@app.function(image=base_image, gpu=None, cpu=8, memory=32_768, timeout=240 * 60)
def run_meg_sweep(sweep_args: list[str]) -> dict[str, Any]:
    """Run run_meg_sarvas_sweep.py in a Modal container and return all new NPZs."""
    results_dir = Path(MOUNT_PATH) / "results"
    before = _snapshot_npzs(results_dir)
    cmd = ["python", "scripts/run_meg_sarvas_sweep.py"] + sweep_args
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[modal_run] $ {shlex.join(cmd)}", flush=True)
    returncode = _stream(cmd, cwd=MOUNT_PATH, env=env)
    npzs: dict[str, bytes] = {}
    for path in _changed_npzs(results_dir, before):
        npzs[str(path.relative_to(results_dir))] = path.read_bytes()
    return {"returncode": returncode, "npz_files": npzs}


@app.local_entrypoint()
def meg_sweep(*cli_args: str) -> None:
    """Launch run_meg_sarvas_sweep.py on Modal and download results.

    Example:
        modal run scripts/modal_run.py::meg_sweep -- \\
            --modalities meg_opm,meg_squid \\
            --sensor-counts 50,100,200,500,700,1000,1500,2000,3000 \\
            --source-spacing-mm 5,10,15,20,30
    """
    sweep_args = list(cli_args)
    print(f"[meg_sweep] launching with args: {sweep_args}", flush=True)
    result = run_meg_sweep.remote(sweep_args)
    output_dir = Path("results_modal")
    npz_files: dict[str, bytes] = result.get("npz_files", {})
    for rel, payload in npz_files.items():
        destination = output_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"[meg_sweep] downloaded {destination}")
    print(f"[meg_sweep] {len(npz_files)} files downloaded")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])

# Map (image, gpu, cpu, memory, timeout_min) -> function
_FUNCTION_MAP: dict[tuple, Any] = {
    ("base", None, 2, 4_096, 30): run_base_small,
    ("base", "T4", 4, 16_384, 60): run_base_t4,
    ("base", None, 8, 16_384, 60): run_base_cpu8,
    ("base", "H100", 8, 65_536, 120): run_base_h100,
    ("openmeeg", None, 8, 16_384, 60): run_openmeeg_cpu8,
}

def _resolve_function(cfg: ModalConfig) -> Any:
    key = (cfg.image, cfg.gpu, int(cfg.cpu), cfg.memory, cfg.timeout_min)
    fn = _FUNCTION_MAP.get(key)
    if fn is None:
        raise SystemExit(
            f"No Modal function defined for config {cfg}. "
            f"Add a matching @app.function entry to modal_run.py."
        )
    return fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run any GUTI modality through run_modality.py on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--modal-output-dir",
        default=os.environ.get("MODAL_OUTPUT_DIR", "results_modal"),
        help="Where to download changed remote results/*.npz files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the remote command without launching Modal work.",
    )
    parser.add_argument(
        "--list-modalities",
        action="store_true",
        help="List supported local modality names and aliases.",
    )
    parser.add_argument("modality_name", nargs="?")
    return parser


@app.local_entrypoint()
def main(*cli_args: str) -> None:
    args = list(cli_args)
    if not args:
        env_args = os.environ.get("MODAL_ARGS")
        if env_args:
            args = shlex.split(env_args)

    parser = _build_parser()
    meta_args, passthrough = parser.parse_known_args(args)

    supported = sorted(_available_modality_dirs() | set(MODALITY_ALIASES))
    if meta_args.list_modalities:
        print("\n".join(supported))
        return

    if meta_args.modality_name is None:
        parser.error(
            "missing modality_name; use --list-modalities to see supported names"
        )

    target, remote_args = resolve_target(meta_args.modality_name, passthrough)
    cfg = _resolve_config(target.config_key)
    command = ["python", "run_modality.py", target.run_modality_name, *remote_args]

    print(
        f"[modal_run] {meta_args.modality_name} -> {target.run_modality_name}: "
        f"image={cfg.image} gpu={cfg.gpu or 'none'} cpu={cfg.cpu} "
        f"memory={cfg.memory}MB timeout={cfg.timeout_min}min",
        flush=True,
    )
    print(f"[modal_run] remote command: {shlex.join(command)}", flush=True)

    if meta_args.dry_run:
        return

    fn = _resolve_function(cfg)
    result = fn.remote(target.run_modality_name, remote_args)

    output_dir = Path(meta_args.modal_output_dir)
    npz_files: dict[str, bytes] = result.get("npz_files", {})
    for rel, payload in npz_files.items():
        destination = output_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"[modal_run] downloaded {destination}")

    if not npz_files:
        print("[modal_run] no new .npz files were produced")

    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
