"""Run a selected unit test on Modal CPU without installing ML packages locally.

uv run --no-project --with modal==1.4.2 modal run scripts/tests_modal.py
"""

from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("cs336-a5-unit-tests")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("numpy==2.2.6", "pytest==8.3.5", "transformers==5.7.0", "einops==0.8.1")
    .workdir("/workspace")
    .env({
        "PYTHONPATH": "/workspace",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "HF_HUB_OFFLINE": "1",
    })
    .add_local_dir(str(ROOT / "cs336_alignment"), "/workspace/cs336_alignment")
    .add_local_dir(str(ROOT / "tests"), "/workspace/tests")
    .add_local_file(str(ROOT / "pyproject.toml"), "/workspace/pyproject.toml")
)


@app.function(image=image, cpu=1, memory=2048, timeout=300, max_containers=1)
def run_tests(target: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-v", "--tb=short"],
        cwd="/workspace",
        check=False,
    ).returncode


@app.local_entrypoint()
def main(target: str = "tests/test_grpo.py::test_tokenize_prompt_and_output"):
    raise SystemExit(run_tests.remote(target))
