"""Two-GPU Modal runner for Section 4.3; runs seeds sequentially and saves results."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

import modal


ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "cs336-a5-grpo"
GPU = os.environ.get("CS336_GRPO_GPU", "B200:2")
SECRET_NAME = os.environ.get("CS336_WANDB_SECRET")
app = modal.App("cs336-a5-grpo")
storage = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["gpu"])
    .workdir("/root")
    .env({"HF_HOME": "/artifacts/huggingface", "PYTHONPATH": "/root",
          "VLLM_CACHE_ROOT": "/artifacts/vllm-cache"})
    .add_local_dir(str(ROOT / "cs336_alignment"), "/root/cs336_alignment")
    .add_local_dir(str(ROOT / "data/gsm8k"), "/root/data/gsm8k")
)


@app.function(
    image=image, gpu=GPU, cpu=8, memory=32768, timeout=8 * 3600,
    max_containers=1, volumes={"/artifacts": storage},
    secrets=[modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else [],
)
def train(run_name: str, seeds: list[int], arguments: list[str]) -> list[str]:
    destination = Path("/artifacts/runs") / run_name
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "experiment.json").write_text(json.dumps({
        "seeds": seeds, "arguments": arguments, "gpu": GPU,
    }, indent=2) + "\n")
    try:
        # A subprocess per seed releases CUDA memory and NCCL state between runs.
        for seed in seeds:
            command = [
                sys.executable, "-u", "-m", "cs336_alignment.train_grpo", *arguments,
                "--seed", str(seed), "--output-dir", str(destination / f"seed-{seed}"),
            ]
            print(shlex.join(command), flush=True)
            try:
                subprocess.run(command, cwd="/root", check=True)
            finally:
                storage.commit()
    finally:
        storage.commit()
    # Download logs/generations automatically; large checkpoints and W&B files
    # remain on the Volume and can be retrieved explicitly.
    return sorted(
        str(path.relative_to(Path("/artifacts")))
        for path in destination.rglob("*")
        if path.is_file() and not {"checkpoints", "wandb"}.intersection(path.relative_to(destination).parts)
    )


@app.local_entrypoint()
def main(
    seeds: str = "0",
    num_rollout_steps: int = 200,
    run_name: str = "",
    output_dir: str = "outputs/grpo",
    wandb_mode: str = "offline",
    extra_args: str = "",
):
    """Use --extra-args for the training module's remaining command-line options."""
    from cs336_alignment.train_grpo import make_parser, validate_args

    selected = [int(seed.strip()) for seed in seeds.split(",")]
    if not selected or min(selected) < 0 or len(set(selected)) != len(selected):
        raise ValueError("seeds must be distinct nonnegative integers")
    arguments = ["--num-rollout-steps", str(num_rollout_steps), "--wandb-mode", wandb_mode, *shlex.split(extra_args)]
    parsed = make_parser().parse_args(arguments)
    validate_args(parsed)
    if parsed.dry_run:
        raise ValueError("Run --dry-run locally with python -m cs336_alignment.train_grpo")
    if parsed.wandb_mode == "online" and not SECRET_NAME:
        raise ValueError("For online W&B, set CS336_WANDB_SECRET to a Modal secret containing WANDB_API_KEY")
    if not GPU.endswith(":2"):
        raise ValueError("CS336_GRPO_GPU must request two GPUs, e.g. B200:2")
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_name):
        raise ValueError("run-name must contain 1–100 letters/digits/underscores/hyphens")
    local = Path(output_dir) / run_name
    if local.exists():
        raise ValueError(f"Local output directory already exists: {local}")
    print(f"Run {run_name}: {GPU}, seeds {selected}, {parsed.num_rollout_steps} steps per seed", flush=True)
    print(f"Results: Modal Volume {VOLUME_NAME}/runs/{run_name}", flush=True)
    files = train.remote(run_name, selected, arguments)
    for filename in files:
        relative = Path(filename).relative_to(Path("runs") / run_name)
        target = local / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as sink:
            for chunk in storage.read_file(filename):
                sink.write(chunk)
    print(f"Downloaded logs and generations to {local.resolve()}", flush=True)
    print("Model checkpoints and offline W&B files remain in the Volume.", flush=True)


