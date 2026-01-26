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


def _find_latest_npz(results_dir: Path) -> Optional[Path]:
    candidates = list(results_dir.rglob("*.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@app.function(
    image=image,
    gpu=_gpu_config(),
    cpu=int(os.environ.get("MODAL_CPU", "8")),
    memory=int(os.environ.get("MODAL_MEMORY", "32768")),
    timeout=int(os.environ.get("MODAL_TIMEOUT", "60")) * 60,
)
def run_us_analytical(args: List[str]) -> Tuple[str, Optional[bytes]]:
    cmd = ["python", "-m", "guti.modalities.us.analytical"] + args
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MOUNT_PATH}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, cwd=MOUNT_PATH, env=env)
    latest = _find_latest_npz(Path(MOUNT_PATH) / "results")
    if latest is None:
        return "", None
    return latest.name, latest.read_bytes()


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
    output_path.write_bytes(payload)
    print(f"Downloaded {output_path}")
