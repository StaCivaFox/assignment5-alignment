"""Single-GPU Modal launcher for Section 3.4; see README.md for commands."""

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import uuid

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "cs336-a5-prompting"
app = modal.App("cs336-a5-prompting")
storage = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extras=["gpu"])
    .workdir("/root")
    .env({"HF_HOME": "/artifacts/huggingface", "PYTHONPATH": "/root"})
    .add_local_dir(str(REPO_ROOT / "cs336_alignment"), "/root/cs336_alignment")
    .add_local_dir(str(REPO_ROOT / "data/gsm8k"), "/root/data/gsm8k")
)


@app.function(image=image, gpu="B200", timeout=3600, max_containers=1, volumes={"/artifacts": storage})
def evaluate(run_name: str, limit: int, seed: int, batch_size: int, prompts: str) -> str:
    output_dir = Path("/artifacts/runs") / run_name
    command = [
        sys.executable, "-u", "-m", "cs336_alignment.prompting_baselines",
        "--output-dir", str(output_dir), "--limit", str(limit),
        "--seed", str(seed), "--batch-size", str(batch_size),
        "--prompts", *prompts.split(","),
    ]
    try:
        subprocess.run(command, cwd="/root", check=True)
    finally:
        # Persist raw generations even if grading or a later prompt fails.
        storage.commit()
    return f"runs/{run_name}"


@app.local_entrypoint()
def main(
    limit: int = 0,
    seed: int = 0,
    batch_size: int = 64,
    prompts: str = "question_only,r1_zero,r1_zero_three_shot",
    run_name: str = "",
    output_dir: str = "outputs/prompting_baselines",
):
    """Evaluate all prompts, persist to a Volume, and download the result files."""
    allowed = {"question_only", "r1_zero", "r1_zero_three_shot"}
    selected = [name.strip() for name in prompts.split(",")]
    if not selected or any(name not in allowed for name in selected) or len(set(selected)) != len(selected):
        raise ValueError(f"Choose distinct prompts from {sorted(allowed)}")
    if limit < 0 or batch_size < 1:
        raise ValueError("limit must be nonnegative and batch-size must be positive")
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_name):
        raise ValueError("run-name must be 1–100 letters/digits/underscores/hyphens, starting with a letter/digit")
    local_dir = Path(output_dir) / run_name
    if local_dir.exists():
        raise ValueError(f"Local destination already exists: {local_dir}")
    print(f"Run: {run_name}; results volume: {VOLUME_NAME}/runs/{run_name}", flush=True)
    print("Launching one B200. First use builds the image and downloads the model.", flush=True)
    remote_dir = evaluate.remote(run_name, limit, seed, batch_size, ",".join(selected))
    local_dir.mkdir(parents=True)
    filenames = ["config.json", "prompt_preview.json", "summary.json", "summary.csv", "review.md", "annotations.csv"]
    filenames.extend(f"{name}.jsonl" for name in selected)
    for filename in filenames:
        with (local_dir / filename).open("wb") as sink:
            for chunk in storage.read_file(f"{remote_dir}/{filename}"):
                sink.write(chunk)
    print(f"Downloaded results to {local_dir.resolve()}", flush=True)
    print(f"Open {local_dir / 'review.md'} for the comparison and sampled generations.", flush=True)
