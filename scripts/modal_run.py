"""Run any GUTI modality on Modal.

This is the general-purpose remote runner: it executes the unified
``run_modality.py <modality> [--param value ...]`` entrypoint inside a Modal
container and downloads the result ``.npz`` files back to the local machine.

Per-modality compute is declared in ``MODAL_CONFIGS`` below (image, GPU, CPU,
memory, timeout). The launcher reads that table locally and applies the
resource envelope to the matching Modal function via
``Function.with_options(...)``, so most modalities share one torch/jax image but
each gets its own GPU/CPU/memory -- ultrasound on a big GPU, the analytic CPU
modalities on a few cores, EEG on a dedicated OpenMEEG image, etc.

Usage:
    modal run scripts/modal_run.py us --scaled-up
    modal run scripts/modal_run.py cw_fnirs --num_sensors 400
    modal run scripts/modal_run.py eeg_openmeeg --num_sensors 256
    modal run scripts/modal_run.py blur_1d --num_brain_grid_points 256

Any flags after the modality name are passed straight through to
``run_modality.py`` (e.g. ``--scaled-up``, ``--no-save``, ``--<param> <value>``).

Resource overrides (handy for one-off tweaks without editing the table):
    MODAL_GPU_TYPE, MODAL_GPU_COUNT, MODAL_CPU, MODAL_MEMORY, MODAL_TIMEOUT
"""

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import modal

APP_NAME = "guti-modality"
MOUNT_PATH = "/root/guti"


# --------------------------------------------------------------------------- #
# Per-modality compute configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModalConfig:
    """Compute envelope for one modality's remote run.

    image: which container image to run in -- "base" (torch/jax/jwave) or
        "openmeeg" (adds the OpenMEEG om_* CLI binaries for the EEG BEM path).
    gpu: Modal GPU spec (e.g. "H100", "A100", "T4", "T4:2") or None for CPU-only.
    cpu: number of physical cores to request.
    memory: RAM in MB.
    timeout_min: hard wall-clock limit in minutes.
    """

    image: str = "base"
    gpu: Optional[str] = None
    cpu: float = 4.0
    memory: int = 16_384
    timeout_min: int = 60


# Default for any modality not listed below (light, CPU-only, base image).
DEFAULT_CONFIG = ModalConfig()

MODAL_CONFIGS: dict[str, ModalConfig] = {
    # Ultrasound: matrix-free SLQ over large torch operators -> big GPU + RAM.
    "us": ModalConfig(image="base", gpu="H100", cpu=8, memory=65_536, timeout_min=120),
    # fNIRS forward models run on torch; a modest GPU speeds the SVD/matmuls.
    "cw_fnirs": ModalConfig(image="base", gpu="T4", cpu=4, memory=16_384, timeout_min=60),
    "td_fnirs": ModalConfig(image="base", gpu="T4", cpu=4, memory=16_384, timeout_min=60),
    # MEG (analytic Sarvas) and blur_1d are pure-numpy / CPU.
    "meg": ModalConfig(image="base", gpu=None, cpu=8, memory=16_384, timeout_min=60),
    "blur_1d": ModalConfig(image="base", gpu=None, cpu=2, memory=4_096, timeout_min=30),
    # EEG shells out to OpenMEEG (om_assemble / om_minverser / om_gain); those
    # binaries live in the dedicated "openmeeg" image. CPU-only.
    "eeg_openmeeg": ModalConfig(image="openmeeg", gpu=None, cpu=8, memory=16_384, timeout_min=60),
}


def _resolve_config(modality_name: str) -> ModalConfig:
    """Look up a modality's config, then apply any env-var overrides."""
    cfg = MODAL_CONFIGS.get(modality_name, DEFAULT_CONFIG)
    gpu_type = os.environ.get("MODAL_GPU_TYPE")
    if gpu_type is not None:
        count = int(os.environ.get("MODAL_GPU_COUNT", "1"))
        cfg = ModalConfig(
            image=cfg.image,
            gpu=gpu_type if count <= 1 else f"{gpu_type}:{count}",
            cpu=cfg.cpu,
            memory=cfg.memory,
            timeout_min=cfg.timeout_min,
        )
    if "MODAL_CPU" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, float(os.environ["MODAL_CPU"]), cfg.memory, cfg.timeout_min)
    if "MODAL_MEMORY" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, cfg.cpu, int(os.environ["MODAL_MEMORY"]), cfg.timeout_min)
    if "MODAL_TIMEOUT" in os.environ:
        cfg = ModalConfig(cfg.image, cfg.gpu, cfg.cpu, cfg.memory, int(os.environ["MODAL_TIMEOUT"]))
    return cfg


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
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
    return img.add_local_dir(".", remote_path=MOUNT_PATH, ignore=_ignore_modal_mount)


# Shared image covering every torch/jax/jwave modality (us, fnirs, meg, blur_1d).
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
        "jax",
        "jaxlib",
        "jaxdf==0.2.8",
        "jwave==0.2.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# EEG image: OpenMEEG (om_* CLI binaries) from conda-forge via micromamba, plus
# the Python deps the EEG path needs (h5py to read the .mat lead field, torch
# for the SVD). conda-forge puts the om_* binaries on PATH in the env.
openmeeg_image = _with_repo(
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("openmeeg", channels=["conda-forge"])
    .pip_install("numpy", "scipy", "matplotlib", "tqdm", "h5py", "torch")
)

IMAGES = {"base": base_image, "openmeeg": openmeeg_image}

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Remote execution
# --------------------------------------------------------------------------- #
def _snapshot_npzs(results_dir: Path) -> dict[str, int]:
    if not results_dir.is_dir():
        return {}
    return {
        str(p.relative_to(results_dir)): p.stat().st_mtime_ns
        for p in results_dir.rglob("*.npz")
    }


def _changed_npzs(results_dir: Path, before: dict[str, int]) -> list[Path]:
    changed = []
    for p in results_dir.rglob("*.npz"):
        rel = str(p.relative_to(results_dir))
        if before.get(rel) != p.stat().st_mtime_ns:
            changed.append(p)
    changed.sort(key=lambda p: p.stat().st_mtime_ns)
    return changed


def _stream(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    with proc.stdout:
        for line in proc.stdout:
            print(line, end="")
    return proc.wait()


def _run_modality(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    """Run ``run_modality.py <modality> <args>`` and return any new npz files.

    Shared body for the per-image Modal functions below.
    """
    results_dir = Path(MOUNT_PATH) / "results"
    before = _snapshot_npzs(results_dir)

    cmd = ["python", "run_modality.py", modality_name, *passthrough_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[modal_run] $ {' '.join(cmd)}", flush=True)
    returncode = _stream(cmd, cwd=MOUNT_PATH, env=env)

    npzs: dict[str, bytes] = {}
    for p in _changed_npzs(results_dir, before):
        npzs[str(p.relative_to(results_dir))] = p.read_bytes()

    return {"modality": modality_name, "returncode": returncode, "npz_files": npzs}


# One function per image (image is fixed at decoration time; gpu/cpu/memory/
# timeout are supplied per call by the local entrypoint via with_options()).
@app.function(image=base_image, timeout=60 * 60)
def run_base(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)


@app.function(image=openmeeg_image, timeout=60 * 60)
def run_openmeeg(modality_name: str, passthrough_args: list[str]) -> dict[str, Any]:
    return _run_modality(modality_name, passthrough_args)


_FUNCTIONS = {"base": run_base, "openmeeg": run_openmeeg}


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(*cli_args):
    """``modal run scripts/modal_run.py <modality> [--param value ...]``."""
    args = list(cli_args)
    if not args:
        env_args = os.environ.get("MODAL_ARGS")
        if env_args:
            args = shlex.split(env_args)
    if not args:
        raise SystemExit(
            "Usage: modal run scripts/modal_run.py <modality> [--param value ...]\n"
            f"Configured modalities: {', '.join(sorted(MODAL_CONFIGS))}"
        )

    modality_name, *passthrough = args
    cfg = _resolve_config(modality_name)
    print(
        f"[modal_run] {modality_name}: image={cfg.image} gpu={cfg.gpu or 'none'} "
        f"cpu={cfg.cpu} memory={cfg.memory}MB timeout={cfg.timeout_min}min",
        flush=True,
    )

    fn = _FUNCTIONS[cfg.image].with_options(
        gpu=cfg.gpu,
        cpu=cfg.cpu,
        memory=cfg.memory,
        timeout=cfg.timeout_min * 60,
    )
    result = fn.remote(modality_name, passthrough)

    output_dir = Path(os.environ.get("MODAL_OUTPUT_DIR", "results_modal"))
    npz_files: dict[str, bytes] = result.get("npz_files", {})
    for rel, payload in npz_files.items():
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        print(f"[modal_run] downloaded {dest}")
    if not npz_files:
        print("[modal_run] (no new .npz files were produced)")

    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
