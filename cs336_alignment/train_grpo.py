"""Section 4.3: standard on-policy GRPO on GSM8K (one training GPU, one vLLM GPU).

Preview without ML dependencies: python -m cs336_alignment.train_grpo --dry-run
See README.md for the 50-step pilot and four-seed Modal experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path
from time import perf_counter

from cs336_alignment.prompting_baselines import MODEL_ID, REPO_ROOT, load_examples


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt-file", type=Path, default=REPO_ROOT / "cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--reward-function", choices=("r1_zero", "question_only"), default="r1_zero")
    parser.add_argument("--train-data", type=Path, default=REPO_ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-data", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo/seed-0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1729)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256, help="Number of responses; one optimizer step per rollout batch")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=0, help="Periodic model checkpoints; 0 saves only the final model")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--attention-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-group", default="standard-on-policy")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args):
    for name in (
        "n_train_examples", "n_val_examples", "num_rollout_steps", "rollout_batch_size",
        "group_size", "gradient_accumulation_steps", "eval_every", "eval_batch_size",
        "log_rollouts_every", "sampling_max_tokens", "max_model_len",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.group_size < 2:
        raise ValueError("group_size must be at least 2 for the sample standard deviation")
    if args.rollout_batch_size % args.gradient_accumulation_steps:
        raise ValueError("rollout_batch_size must be divisible by gradient_accumulation_steps")
    microbatch = args.rollout_batch_size // args.gradient_accumulation_steps
    if microbatch % args.group_size:
        raise ValueError("Each microbatch must contain complete groups: microbatch_size must be divisible by group_size")
    if args.n_train_examples < args.rollout_batch_size // args.group_size:
        raise ValueError("The training subset must contain at least one batch of prompts")
    if args.save_every < 0 or args.seed < 0 or args.eval_seed < 0:
        raise ValueError("save_every and seeds must be nonnegative")
    for name in ("learning_rate", "max_grad_norm", "advantage_eps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < args.vllm_memory_utilization < 1:
        raise ValueError("vllm_memory_utilization must be between 0 and 1")
    # The implemented loss scores the original policy, without sampling corrections.
    if args.sampling_temperature != 1.0 or args.sampling_top_p != 1.0:
        raise ValueError("This strict on-policy implementation requires temperature=1 and top_p=1")
    if not args.policy_device.startswith("cuda:") or not args.policy_device[5:].isdigit():
        raise ValueError("policy-device must be an indexed CUDA device, e.g. cuda:0")
    if args.vllm_gpu < 0 or args.vllm_gpu == int(args.policy_device[5:]):
        raise ValueError("Training and vLLM must use different GPUs")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rollouts(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_run(args):
    validate_args(args)
    train = load_examples(args.train_data, args.n_train_examples)
    val = load_examples(args.val_data, args.n_val_examples)
    if len(train) != args.n_train_examples or len(val) != args.n_val_examples:
        raise ValueError("Requested dataset subset is larger than the supplied file")
    template = args.prompt_file.read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError("The prompt template must contain {question}")
    preview = template.format(question=train[0]["question"])
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("torch", "transformers", "vllm", "wandb", "einops"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prompt_template": template,
        "train_data_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "val_data_sha256": hashlib.sha256(args.val_data.read_bytes()).hexdigest(),
        "subset_selection": "first N examples; fixed validation subset across seeds",
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.0},
        "algorithm": {"baseline": "mean", "advantage_normalizer": "std", "std_correction": 1,
                      "importance_reweighting_method": "none", "loss_normalization": "sequence"},
        "versions": versions,
    }
    write_json(args.output_dir / "config.json", manifest)
    write_json(args.output_dir / "prompt_preview.json", {
        "prompt": preview, "ground_truth": train[0]["ground_truth"],
        "prompts_per_batch": args.rollout_batch_size // args.group_size,
        "responses_per_batch": args.rollout_batch_size,
        "microbatch_size": args.rollout_batch_size // args.gradient_accumulation_steps,
        "train_sampling": sampling_params(args, args.group_size, args.seed),
        "validation_sampling": sampling_params(args, 1, args.eval_seed),
    })
    return train, val, template, manifest


class PromptStream:
    """Shuffle questions each epoch while keeping each response group contiguous."""

    def __init__(self, examples, seed):
        self.examples = examples
        self.rng = random.Random(seed)
        self.order = []

    def take(self, count):
        result = []
        while len(result) < count:
            if not self.order:
                self.order = list(range(len(self.examples)))
                self.rng.shuffle(self.order)
            result.append(self.examples[self.order.pop()])
        return result


def sampling_params(args, n, seed):
    return {
        "n": n, "temperature": args.sampling_temperature, "top_p": args.sampling_top_p,
        "top_k": -1, "min_p": 0.0, "max_tokens": args.sampling_max_tokens,
        "seed": seed,
        "stop": ["</answer>"] if args.reward_function == "r1_zero" else [],
        "include_stop_str_in_output": args.reward_function == "r1_zero",
    }


def collect_rollouts(server, examples, template, params, reward_fn, tokenizer, batch_size, policy_step):
    prompts = [template.format(question=row["question"]) for row in examples]
    completions = server.generate_completions(prompts, params, batch_size=batch_size)
    expanded = [(row, prompt) for row, prompt in zip(examples, prompts, strict=True) for _ in range(params["n"])]
    if len(completions) != len(expanded):
        raise RuntimeError(f"Expected {len(expanded)} completions, received {len(completions)}")
    records = []
    for index, ((row, prompt), completion) in enumerate(zip(expanded, completions, strict=True)):
        scores = reward_fn(completion.text, row["ground_truth"])
        records.append({
            "example_id": row["example_id"], "sample_index": index % params["n"],
            "policy_step": policy_step, "question": row["question"], "prompt": prompt,
            "ground_truth": row["ground_truth"], "response": completion.text,
            "generated_tokens": len(completion.token_ids),
            "response_tokens": len(tokenizer.encode(completion.text, add_special_tokens=False)),
            "finish_reason": completion.finish_reason,
            **{key: float(scores[key]) for key in ("reward", "format_reward", "answer_reward")},
        })
    return records


def rollout_metrics(records, prefix):
    n = len(records)
    return {
        f"{prefix}/mean_{key}": sum(record[key] for record in records) / n
        for key in ("reward", "format_reward", "answer_reward", "response_tokens", "generated_tokens")
    } | {
        f"{prefix}/length_limit_fraction": sum(record["finish_reason"] == "length" for record in records) / n,
        f"{prefix}/empty_response_fraction": sum(record["response_tokens"] == 0 for record in records) / n,
    }


class ExperimentLogger:
    def __init__(self, args, manifest):
        self.path = args.output_dir / "metrics.jsonl"
        self.run = None
        if args.wandb_mode != "disabled":
            import wandb
            self.run = wandb.init(
                project=args.wandb_project, group=args.wandb_group,
                name=f"{args.output_dir.parent.name}-seed-{args.seed}",
                config=manifest, mode=args.wandb_mode, dir=str(args.output_dir),
            )

    def log(self, step, metrics, samples=()):
        values = {key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                  for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Nonfinite metrics at step {step}: {values}")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, **values}, allow_nan=False) + "\n")
        if self.run is not None:
            payload = dict(values)
            if samples:
                import wandb
                columns = ["policy_step", "question", "ground_truth", "response", "reward"]
                payload["rollouts"] = wandb.Table(columns=columns, data=[[row[key] for key in columns] for row in samples])
            self.run.log(payload, step=step)
        concise = {key: round(value, 5) for key, value in values.items() if key in {
            "train/loss", "train/grad_norm_before_clip", "train/token_entropy",
            "reward/mean_reward", "val/mean_reward", "val/mean_format_reward",
        }}
        print(json.dumps({"step": step, **concise}), flush=True)

    def close(self, failed=False):
        if self.run is not None:
            self.run.finish(exit_code=int(failed))


def save_checkpoint(model, tokenizer, directory, step):
    destination = directory / "checkpoints" / f"step-{step:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    write_json(destination / "step.json", {"step": step, "contents": "model and tokenizer; no optimizer/RNG resume state"})


def run_training_loop(args, model, tokenizer, optimizer, server, train, val, template, reward_fn, logger):
    """Synchronous on-policy loop, also exercised with a tiny CPU model in tests."""
    from cs336_alignment.grpo import grpo_train_step

    stream = PromptStream(train, args.seed)
    start_time = perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def evaluate(step):
        # Evaluation must see the weights AFTER the most recent optimizer update.
        server.sync_policy_weights(model)
        records = collect_rollouts(
            server, val, template, sampling_params(args, 1, args.eval_seed),
            reward_fn, tokenizer, args.eval_batch_size, policy_step=step,
        )
        write_rollouts(args.output_dir / "rollouts" / f"val-step-{step:04d}.jsonl", records)
        return rollout_metrics(records, "val"), records

    initial_metrics, initial_records = evaluate(0)
    logger.log(0, initial_metrics | {"time/elapsed_seconds": perf_counter() - start_time}, initial_records[:8])
    final_metrics = initial_metrics
    for step in range(1, args.num_rollout_steps + 1):
        step_started = perf_counter()
        server.sync_policy_weights(model)
        examples = stream.take(args.rollout_batch_size // args.group_size)
        records = collect_rollouts(
            server, examples, template, sampling_params(args, args.group_size, args.seed + step),
            reward_fn, tokenizer, batch_size=None, policy_step=step - 1,
        )
        log_rollouts = step == 1 or step % args.log_rollouts_every == 0 or step == args.num_rollout_steps
        if log_rollouts:
            write_rollouts(args.output_dir / "rollouts" / f"train-step-{step:04d}.jsonl", records)
        generation_seconds = perf_counter() - step_started

        # Reuse the scores already computed above; keep the train-step interface.
        score_cache = {(row["response"], row["ground_truth"]): {
            key: row[key] for key in ("reward", "format_reward", "answer_reward")
        } for row in records}
        try:
            loss, metadata = grpo_train_step(
                model, tokenizer, optimizer, args.gradient_accumulation_steps, args.max_grad_norm,
                lambda response, truth: score_cache[(response, truth)],
                [row["prompt"] for row in records], [row["response"] for row in records],
                [row["ground_truth"] for row in records], args.group_size,
                baseline="mean", advantage_eps=args.advantage_eps, advantage_normalizer="std",
                importance_reweighting_method="none", loss_normalization="sequence",
            )
        except Exception:
            write_rollouts(args.output_dir / "rollouts" / f"failed-train-step-{step:04d}.jsonl", records)
            raise
        metrics = dict(metadata) | rollout_metrics(records, "rollout")
        metrics["train/loss"] = loss.detach()
        metrics["time/rollout_seconds"] = generation_seconds
        metrics["time/update_seconds"] = perf_counter() - step_started - generation_seconds
        samples = records[:8] if log_rollouts else []
        if step % args.eval_every == 0 or step == args.num_rollout_steps:
            final_metrics, eval_records = evaluate(step)
            metrics.update(final_metrics)
            samples = eval_records[:8]
        metrics["time/elapsed_seconds"] = perf_counter() - start_time
        logger.log(step, metrics, samples)
        write_json(args.output_dir / "status.json", {"status": "running", "completed_steps": step})
        if (args.save_every and step % args.save_every == 0) or step == args.num_rollout_steps:
            save_checkpoint(model, tokenizer, args.output_dir, step)
    summary = {
        "seed": args.seed, "completed_steps": args.num_rollout_steps,
        "initial_validation": initial_metrics, "final_validation": final_metrics,
        "elapsed_seconds": perf_counter() - start_time,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    args = make_parser().parse_args(argv)
    train, val, template, manifest = prepare_run(args)
    if args.dry_run:
        write_json(args.output_dir / "status.json", {"status": "dry_run"})
        print(f"Validated configuration and datasets. Preview: {args.output_dir / 'prompt_preview.json'}")
        return

    server = logger = None
    failed = True
    write_json(args.output_dir / "status.json", {"status": "initializing", "completed_steps": 0})
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
        from cs336_alignment.vllm_utils import VLLMServer

        required_gpus = max(int(args.policy_device[5:]), args.vllm_gpu) + 1
        if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
            raise RuntimeError("This experiment needs two CUDA GPUs. Use the Modal launcher, or --dry-run for a CPU preview.")
        set_seed(args.seed)
        torch.cuda.set_device(args.policy_device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        prompt_ids = tokenizer([template.format(question=row["question"]) for row in train + val], add_special_tokens=False)["input_ids"]
        if max(map(len, prompt_ids)) + args.sampling_max_tokens > args.max_model_len:
            raise ValueError("Some prompts plus sampling-max-tokens exceed max-model-len; increase max-model-len")
        server = VLLMServer(
            model_id=args.model, gpu=args.vllm_gpu, port=args.vllm_port, seed=args.seed,
            gpu_memory_utilization=args.vllm_memory_utilization,
            max_model_len=args.max_model_len, startup_timeout=1200,
        )
        server.start()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map=args.policy_device, torch_dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
        )
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
        server.init_weight_sync(args.policy_device)
        manifest["resolved_model_commit"] = getattr(model.config, "_commit_hash", None)
        manifest["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        write_json(args.output_dir / "config.json", manifest)
        logger = ExperimentLogger(args, manifest)
        reward_fn = r1_zero_reward_fn if args.reward_function == "r1_zero" else question_only_reward_fn
        run_training_loop(args, model, tokenizer, optimizer, server, train, val, template, reward_fn, logger)
        write_json(args.output_dir / "status.json", {"status": "complete", "completed_steps": args.num_rollout_steps})
        failed = False
    except BaseException as error:
        status = json.loads((args.output_dir / "status.json").read_text())
        write_json(args.output_dir / "status.json", {**status, "status": "failed", "error": repr(error)})
        raise
    finally:
        try:
            if server is not None:
                server.stop()
        finally:
            if logger is not None:
                logger.close(failed=failed)


if __name__ == "__main__":
    main()
