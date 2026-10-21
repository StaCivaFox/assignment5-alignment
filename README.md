# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Run unit tests on Modal CPU

The tokenization test uses three short strings and an in-memory toy tokenizer;
it does not load the OLMo model or GSM8K. The lightweight runner below uses one
CPU and 2 GiB of requested memory on Modal, with a five-minute test timeout.
There is no GPU charge; Modal CPU/memory and image-building charges may apply.
It installs CPU-only PyTorch and a small set of test dependencies remotely.

From the repository root, authenticate once if needed, then run the test:

```sh
uv run --no-project --with modal==1.4.2 modal setup
uv run --no-project --with modal==1.4.2 modal run scripts/tests_modal.py
```

Keep `--no-project`: it prevents `uv` from installing this repository's ML
dependencies on your local machine just to launch the remote job. Only the
Modal SDK is needed locally. The first run builds the remote image; subsequent
runs reuse it and upload your latest code. Pytest output appears in your terminal,
and a failing test produces a nonzero local exit status.

Choose another GRPO test, or run the whole GRPO test file, with `--target`:

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/tests_modal.py --target tests/test_grpo.py::test_get_response_log_probs
uv run --no-project --with modal==1.4.2 modal run scripts/tests_modal.py --target tests/test_grpo.py
```

This runner includes the local test fixtures and snapshots and disables Hugging
Face downloads. It is intended for CPU unit tests in `test_grpo.py`, not GPU
training or evaluation; other test files may need additional dependencies.

## Section 3.4: prompting baselines on Modal

The evaluator compares `question_only`, `r1_zero`, and
`r1_zero_three_shot` using `allenai/OLMo-2-0425-1B` on the included
1,319-example GSM8K test set. It uses temperature 1, top-p 1, a 512-token
generation limit, and one sampled response per question. The R1 prompts stop
at `</answer>` and retain the closing tag for grading. The supplied prompt
templates and reward functions are used without changing their contents or
relaxing their grading rules.

The dedicated launcher in `scripts/prompting_baselines_modal.py` uses one
B200, a one-hour timeout, and a persistent Modal Volume named
`cs336-a5-prompting` for model downloads and results. It works with a personal
Modal account; no SUNet ID or W&B secret is required. The course's shared
`modal_utils.py` is not used by this launcher.

### 1. Check the inputs locally

From the repository root, this requires only Python 3.12 and does not start a
GPU or download a model:

```sh
python3 -m cs336_alignment.prompting_baselines --dry-run --limit 3 --output-dir outputs/prompting-preview
```

Read `outputs/prompting-preview/prompt_preview.json` to see the three rendered
prompts. Every run needs a new or empty output directory.

### 2. Authenticate with Modal

Use the same Modal SDK version as `uv.lock`. `uv run --no-project --with`
installs only the launcher dependency locally; the full locked GPU environment
is built remotely from `pyproject.toml` and `uv.lock`.

```sh
uv run --no-project --with modal==1.4.2 modal setup
```

You can skip authentication if this machine is already authenticated with the
intended Modal workspace. See [Modal's setup guide](https://modal.com/docs/guide/getting-started).

### 3. Run a small GPU pilot

This command starts a paid single-B200 job and evaluates the first 12 questions
under all three prompts:

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/prompting_baselines_modal.py --limit 12 --seed 0
```

The first run builds the image and downloads the model. Subsequent runs reuse
the image and Hugging Face cache. On success, the launcher downloads all result
files into `outputs/prompting_baselines/<run-name>/` and prints that path.
Check that `config.json` says `complete`, each prompt has 12 records, and the
three category counts add up to 12. A small pilot can have few or no correct
responses; it is a plumbing check, not your final accuracy estimate.

### 4. Run the complete experiment

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/prompting_baselines_modal.py --seed 0
```

Omitting `--limit` evaluates all 1,319 questions for each prompt: 3,957
generations in total. The defaults match Section 3.4. This section does not
require the four-seed RL training runs described later in the handout.
For an optional repeat, change `--seed` and compare the results; each invocation
creates a separate run directory. Sampling seeds are recorded, but exact
reproduction also depends on the model revision, library versions, hardware,
and batching.

The launcher also accepts `--batch-size 64`, a comma-separated `--prompts`
selection, and an optional unique `--run-name`. For example, select only the
few-shot condition with `--prompts r1_zero_three_shot`.

### 5. Inspect the experiment

| File | What to inspect |
|---|---|
| `summary.csv` / `summary.json` | Accuracy, format rate, all three category counts, length-limit rate, and mean generated tokens per prompt |
| `<prompt>.jsonl` | Every question, reference answer/rationale, exact rendered prompt, unmodified response, rewards, finish reason, and generated-token count |
| `review.md` | Comparison table and up to ten reproducibly sampled examples per prompt/category |
| `annotations.csv` | The same review examples with blank human-correct, parsing-failure, and notes columns |
| `config.json` | Model, seed, dataset checksum, exact templates, generation settings, package versions, elapsed time, and completion status |

For Section 3.4(a), report the three category counts for each prompt. Inspect
at least ten formatted-but-incorrect responses and ten format failures per
prompt when available. Record how many were mathematically correct but rejected
by the grader, reporting both the count and number inspected. If a category has
fewer than ten examples, inspect all of them and state that sample size.
Keep the original grader metrics separate from your human annotations.

For Section 3.4(b), use actual examples to describe the model's behavior under
each prompt. Look for direct answers, continued questions or conversations,
reasoning, tag adherence, and truncation. The script deliberately leaves these
interpretations and human correctness judgments for you to make.

The review samples are balanced by category, not representative of the whole
dataset. Do not average their human-correct labels as an overall accuracy.
Raw responses are never cleaned up before grading, so parsing failures remain
visible as required by the assignment. `length_limit_rate` counts responses
whose vLLM finish reason is `length`.

### Recover results after a disconnect or failure

The launcher prints a run name before submission. Completed files remain in
the Modal Volume even if local downloading is interrupted. Use that name in
the following commands (replace `RUN_NAME`):

```sh
uv run --no-project --with modal==1.4.2 modal volume ls cs336-a5-prompting runs/RUN_NAME
uv run --no-project --with modal==1.4.2 modal volume get cs336-a5-prompting runs/RUN_NAME outputs/recovered-RUN_NAME
```

A normal exception commits partial results, including the raw generation that
caused a grader error. A forcibly terminated container may lose its most recent
uncommitted writes. A failed run is marked in `config.json`; any existing summary
may cover only the prompts that finished. No automatic retry or resume is
performed. Start a new run rather than mixing partial runs or overwriting files.
See [Modal Volumes](https://modal.com/docs/guide/volumes) for retrieval details.

### Run on your own GPU instead

Install the GPU extra and invoke the evaluator directly:

```sh
uv sync --frozen --extra gpu
uv run python -m cs336_alignment.prompting_baselines --output-dir outputs/local-baseline --seed 0
```

The evaluator uses vLLM's offline interface with one GPU and BF16 weights.
It disables model-specific generation defaults so its explicit sampling
configuration determines the experiment. It does not need a separate vLLM
server or any GRPO implementation.

CPU checks for this evaluator can be run without installing PyTorch or vLLM:

```sh
uv run --no-project --with 'math-verify[antlr4-13-2]==0.7.0' --with latex2sympy2-extended==1.10.1 --with pylatexenc==2.10 python -m unittest discover -s tests -p test_prompting_baselines.py
```

## Section 4.3: standard on-policy GRPO

`cs336_alignment/train_grpo.py` runs the experiment using your GRPO helpers.
`scripts/train_grpo_modal.py` provides two GPUs on a personal Modal account;
no SUNet ID is needed. The defaults match the handout:

| Setting | Default |
|---|---|
| Model / prompt | `allenai/OLMo-2-0425-1B` / zero-shot `r1_zero` |
| Training / validation examples | First 6,400 / first 1,024 GSM8K examples |
| Updates / learning rate | 200 / 1e-5 |
| Responses per update / group size | 256 / 8 (32 questions) |
| Microbatches per update | 32, each containing one complete group of 8 |
| Optimizer | AdamW, betas=(0.9, 0.95), weight_decay=0 |
| Gradient clipping | Norm 1.0 |
| Sampling | Temperature 1, top-p 1, up to 512 new tokens |
| Validation | Before training, every 10 updates, and after the final update |
| Training rollout logs | First update, every 40 updates, and the final update |

Each iteration synchronizes policy weights into vLLM, samples a fresh batch,
and makes exactly one optimizer update with gradient accumulation. Validation
synchronizes again so its generations use the updated policy. Validation uses
one sampled answer per question and a fixed evaluation seed across checkpoints
and training seeds. The R1 stopping string `</answer>` is retained for the
unmodified course grader. References are extracted after GSM8K's `####` marker
and never included in prompts.

### Preview the configuration without a GPU

This command uses only the Python standard library and reads the local data:

```sh
python3 -m cs336_alignment.train_grpo --dry-run --output-dir outputs/grpo-preview
```

Read `config.json` and `prompt_preview.json`. Every invocation needs a new or
empty output directory. The runner validates that microbatches contain complete
groups, because your current train step normalizes rewards inside microbatches.
It also requires temperature=1 and top-p=1: your on-policy loss scores the
unmodified policy, so a different sampling distribution would require further
algorithm changes. Generation length remains configurable.

### Run a small integration check on Modal

Authenticate once if needed:

```sh
uv run --no-project --with modal==1.4.2 modal setup
```

This starts a paid two-B200 job for two updates on small subsets. Its purpose
is to check CUDA, NCCL weight transfer, model loading, generation, and saving;
it is too small to assess learning:

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/train_grpo_modal.py \
  --run-name grpo-smoke --num-rollout-steps 2 \
  --extra-args "--n-train-examples 32 --n-val-examples 16 --rollout-batch-size 16 --gradient-accumulation-steps 2 --eval-every 1"
```

The image uses the locked GPU dependencies. One GPU trains the model in BF16
with gradient checkpointing; the other runs vLLM. The first invocation also
downloads the model and builds the inference engine. Later runs reuse the
image and persistent caches. The function has an eight-hour timeout and runs
seeds sequentially, using two GPUs at a time.

### Run the 50-step pilot

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/train_grpo_modal.py \
  --run-name grpo-pilot --seeds 0 --num-rollout-steps 50
```

Outputs download to `outputs/grpo/grpo-pilot/seed-0/`. Check the validation
reward at steps 0, 10, 20, 30, 40, and 50, and inspect the corresponding
validation rollout files for the same questions before and after training.
Training can start with near-zero rewards. Use the measured trajectory and
actual generations to decide whether to proceed; a lower policy-gradient loss
alone is not evidence of improved accuracy.

### Run the four-seed experiment

```sh
uv run --no-project --with modal==1.4.2 modal run scripts/train_grpo_modal.py \
  --run-name grpo-standard --seeds 0,1,2,3 --num-rollout-steps 200
```

The launcher runs each seed in a fresh subprocess to release CUDA/NCCL state
between runs. A run failure stops the experiment and preserves completed files
in the `cs336-a5-grpo` Modal Volume. Use a new `--run-name` for subsequent
experiments. You can run a remaining seed separately if a previous job failed.

All training-module flags are available through `--extra-args`, for example
`--extra-args "--learning-rate 5e-6 --save-every 50"`. Run
`python3 -m cs336_alignment.train_grpo --help` for model, prompt-file,
dataset, generation length, evaluation, and checkpoint options. Custom data
must be under the uploaded `data/gsm8k/` directory when using this Modal image;
custom prompts can be placed in `cs336_alignment/prompts/`. Paths passed to the
remote script are relative to `/root`, not local absolute paths.

The default allocation is `B200:2`, matching the course setup. To request a
different supported pair, set `CS336_GRPO_GPU`, for example
`CS336_GRPO_GPU=A100-80GB:2`. Runtime and memory headroom depend on the GPU;
validate a replacement with the small integration check first. See
[Modal GPU configuration](https://modal.com/docs/guide/gpu).

### Logs, checkpoints, and W&B

| File | Contents |
|---|---|
| `config.json` | Hyperparameters, exact prompt, dataset hashes, model commit, package versions, GPU types |
| `metrics.jsonl` | One record per update; step 0 contains initial validation metrics |
| `rollouts/train-step-*.jsonl` | Periodic training responses with rewards, reference answers, lengths, and the policy version that generated them |
| `rollouts/val-step-*.jsonl` | All validation generations at every evaluation, including step 0 and the final step |
| `summary.json` | Initial/final validation metrics and elapsed time |
| `status.json` | Running, complete, or failed status and completed-update count |
| `checkpoints/step-*/` | Model and tokenizer loadable with `from_pretrained` |
| `wandb/` | Offline W&B run files by default |

Required metrics include `train/loss`, `train/grad_norm_before_clip`,
`train/token_entropy`, `reward/mean_reward`, `reward/mean_format_reward`,
`val/mean_reward`, `val/mean_format_reward`, and `val/mean_response_tokens`.
Entropy excludes prompt/padding positions. Lengths include both re-tokenized
response-text length and the number of tokens reported by vLLM, which may also
count an EOS token omitted from the text. The script also records answer reward,
truncation/empty-response rates, advantage statistics, and timing.

JSONL logging always runs. W&B defaults to offline mode, which does not require
an account or API key. Use `--wandb-mode disabled` to omit W&B entirely. For live
W&B charts, create a Modal secret containing `WANDB_API_KEY`, then set its name:

```sh
CS336_WANDB_SECRET=wandb uv run --no-project --with modal==1.4.2 modal run \
  scripts/train_grpo_modal.py --run-name grpo-online --wandb-mode online
```

Offline runs can also be downloaded and uploaded later with `wandb sync`.
See [W&B offline logging](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues).

The vLLM launcher disables Uvicorn HTTP access logs (including successful
`/reset_prefix_cache` requests) and sets Uvicorn's log level to `warning`.
This retains HTTP server warnings/errors; vLLM's own logging level is controlled
separately by `VLLM_LOGGING_LEVEL`. See the
[vLLM 0.19.1 server logging options](https://docs.vllm.ai/en/v0.19.1/api/vllm/entrypoints/openai/cli_args/).

Logs and generations download automatically after the job completes. Large
checkpoints and offline W&B files stay in the Volume. Fetch them explicitly:

```sh
uv run --no-project --with modal==1.4.2 modal volume get cs336-a5-grpo \
  runs/grpo-standard/seed-0/checkpoints/step-0200 outputs/grpo-checkpoint-seed-0
```

For a disconnected or failed job, inspect and recover its directory:

```sh
uv run --no-project --with modal==1.4.2 modal volume ls cs336-a5-grpo runs/grpo-standard
uv run --no-project --with modal==1.4.2 modal volume get cs336-a5-grpo \
  runs/grpo-standard outputs/grpo-recovered
```

The final model is always saved; `--save-every N` additionally saves periodic
checkpoints. Checkpoints contain model/tokenizer weights, not optimizer and RNG
state, so exact interrupted-run resume is not implemented. Partial logs remain
useful for diagnosis but should not be mixed into a completed four-seed result.

### Produce the required plots

Plotting is local and needs only matplotlib beyond the standard library:

```sh
uv run --no-project --with matplotlib python -m cs336_alignment.plot_grpo \
  --runs outputs/grpo/grpo-standard/seed-* --output-dir outputs/grpo/grpo-standard/plots
```

For the 50-update pilot already downloaded to this repository:

```sh
uv run --no-project --with matplotlib python -m cs336_alignment.plot_grpo \
  --runs outputs/grpo/grpo-pilot/seed-0 --output-dir outputs/grpo/grpo-pilot/plots \
  --title "GRPO pilot (50 updates)"
```

Each figure has two panels: completed optimizer updates on the left and measured
wall-clock seconds on the right. The files are saved locally as 200-DPI PNGs and
vector PDFs; use `--dpi 300` for higher-resolution PNGs. Outputs include:

- All eight required metrics in separate figures.
- `reward_comparison`, `format_comparison`, and `length_comparison`, overlaying
  training rollouts and validation in each pair of panels.
- `overview`, combining reward, format compliance, and response-token entropy.
- `summary.json`, with final per-seed values, means, and sample standard deviations.

Curves show the recorded values without smoothing. For multiple seeds, faint
lines show individual seeds and the main line/band shows the mean ±1 sample
standard deviation. Time-axis aggregate points use mean elapsed time at matching
updates; the band compares seeds at that update, not at a fixed time budget.
The script rejects incomplete runs, duplicate seeds, mismatched settings, or
inconsistent evaluation schedules. A single seed has no variance band.

Time starts at the training loop, including rollouts, updates, and validation,
but excluding model/server startup. The existing pilot's step-0 evaluation lacks
a timestamp and appears only on the step axis. New runs also record its elapsed
time. Training rewards describe the rollouts used for an update; validation
evaluates the model after the indicated number of updates. Policy-gradient loss
is a different objective from cross-entropy and need not decrease monotonically;
use validation reward and saved generations to assess progress.

Compare the final mean validation reward to the assignment's 25% target, and
write your commentary from the resulting plots and actual saved rollouts.
The script does not guarantee that target or invent conclusions about learning.
Because the provided GSM8K test file serves as validation here, treat the result
as the course experiment, not an untouched held-out test after tuning.

### Run on your own two-GPU machine

```sh
uv sync --frozen --extra gpu
uv run --frozen --extra gpu python -m cs336_alignment.train_grpo \
  --seed 0 --output-dir outputs/grpo/local-seed-0
```

The implementation uses the repository's
[vLLM 0.19.1 NCCL weight-transfer API](https://docs.vllm.ai/en/v0.19.1/api/vllm/distributed/weight_transfer/nccl_engine/);
avoid upgrading vLLM independently of the lockfile. The log-probability,
normalization, gradient accumulation, and optimizer update use your existing
functions. Small integration fixes move tensors onto the policy GPU, detach
the returned logging loss, and give empty responses zero masked loss.

CPU verification (no model downloads or GPUs) includes your original update
snapshot and the training-loop tests with a tiny model and simulated generation:

```sh
uv run --no-sync pytest tests/test_grpo.py::test_grpo_train_step_standard_on_policy \
  tests/test_grpo_metadata.py tests/test_train_grpo.py -q
```

