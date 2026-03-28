import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import modal


APP_NAME = "guti-us-analytical"
MOUNT_PATH = "/root/guti"


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
    .add_local_dir(".", remote_path=MOUNT_PATH)
)

app = modal.App(APP_NAME)


def _snapshot_npzs(results_dir: Path) -> dict[str, int]:
    return {
        str(path.relative_to(results_dir)): path.stat().st_mtime_ns
        for path in results_dir.rglob("*.npz")
    }


def _find_changed_npz(results_dir: Path, before: dict[str, int]) -> Optional[Path]:
    changed: list[Path] = []
    for path in results_dir.rglob("*.npz"):
        rel = str(path.relative_to(results_dir))
        mtime_ns = path.stat().st_mtime_ns
        if before.get(rel) != mtime_ns:
            changed.append(path)
    if not changed:
        return None
    return max(changed, key=lambda p: p.stat().st_mtime_ns)


@app.function(
    image=image,
    gpu=_gpu_config(),
    cpu=int(os.environ.get("MODAL_CPU", "8")),
    memory=int(os.environ.get("MODAL_MEMORY", "32768")),
    timeout=int(os.environ.get("MODAL_TIMEOUT", "60")) * 60,
    single_use_containers=True,
)
def run_us_analytical(args: List[str]) -> Tuple[str, Optional[bytes]]:
    cmd = ["python", "-m", "guti.modalities.us.analytical"] + args
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    results_dir = Path(MOUNT_PATH) / "results"
    before = _snapshot_npzs(results_dir)
    subprocess.run(cmd, check=True, cwd=MOUNT_PATH, env=env)
    changed = _find_changed_npz(results_dir, before)
    if changed is None:
        return "", None
    return changed.name, changed.read_bytes()


@app.local_entrypoint()
def main(*cli_args):
    parser = argparse.ArgumentParser(
        description="Run guti/modalities/us/analytical.py on Modal with the same CLI."
    )
    parser.add_argument("--modal-output-dir", default="results_modal", help="Where to save downloaded npz")
    passthrough_args = list(cli_args)
    if not passthrough_args:
        env_args = os.environ.get("MODAL_ARGS")
        if env_args:
            passthrough_args = shlex.split(env_args)
    args, passthrough = parser.parse_known_args(passthrough_args)

    filename, payload = run_us_analytical.remote(passthrough)
    if payload is None:
        print("No .npz result found under results/")
        return
    output_dir = Path(args.modal_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    print(f"Downloaded {output_path}")
