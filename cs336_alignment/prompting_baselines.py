"""Section 3.4: evaluate the three supplied GSM8K prompts with vLLM.

Run locally with ``python -m cs336_alignment.prompting_baselines --help``;
for a GPU on Modal, see scripts/prompting_baselines_modal.py and README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "allenai/OLMo-2-0425-1B"
PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}
CATEGORIES = {
    (1.0, 1.0): "correct",
    (1.0, 0.0): "formatted_incorrect",
    (0.0, 0.0): "format_failure",
}


def load_examples(path: Path, limit: int = 0) -> list[dict]:
    """Keep reference rationales out of prompts and retain stable dataset row IDs."""
    if limit < 0:
        raise ValueError("limit must be nonnegative (0 means the whole dataset)")
    examples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question, answer = row["question"], row["answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"{path}:{line_number}: question and answer must be strings")
            if "####" not in answer or not answer.rsplit("####", 1)[1].strip():
                raise ValueError(f"{path}:{line_number}: missing GSM8K final answer")
            examples.append({
                "example_id": line_number - 1,
                "question": question,
                "ground_truth": answer.rsplit("####", 1)[1].strip(),
                "reference_solution": answer,
            })
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def sampling_config(prompt_name: str, seed: int) -> dict:
    if prompt_name not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt: {prompt_name}")
    is_r1 = prompt_name != "question_only"
    return {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 512,
        "seed": seed,
        "stop": ["</answer>"] if is_r1 else [],
        "include_stop_str_in_output": is_r1,
    }


def evaluate_prompt(engine, sampling_params, examples, template, reward_fn, output_path, batch_size):
    """Persist each batch; never alter response text before using the course grader."""
    records = []
    with output_path.open("x", encoding="utf-8") as sink:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [template.format(question=example["question"]) for example in batch]
            outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs than prompts")
            for example, prompt, output in zip(batch, prompts, outputs, strict=True):
                if output.prompt != prompt or len(output.outputs) != 1:
                    raise RuntimeError("Expected one completion per prompt, in input order")
                completion = output.outputs[0]
                record = {
                    **example,
                    "prompt": prompt,
                    "response": completion.text,
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
                try:
                    rewards = reward_fn(completion.text, example["ground_truth"])
                    category = CATEGORIES[(rewards["format_reward"], rewards["answer_reward"])]
                    if rewards["reward"] != rewards["answer_reward"]:
                        raise ValueError("Expected total reward to equal answer reward")
                except Exception as error:
                    # Preserve the generation for debugging without silently counting
                    # a grader exception as an incorrect mathematical answer.
                    sink.write(json.dumps({**record, "grading_error": repr(error)}, ensure_ascii=False) + "\n")
                    sink.flush()
                    raise
                record.update(rewards)
                record["category"] = category
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            sink.flush()
            print(f"{output_path.stem}: {len(records)}/{len(examples)}", flush=True)
    return records


def summarize(prompt_name: str, records: list[dict]) -> dict:
    counts = Counter(record["category"] for record in records)
    n = len(records)
    return {
        "prompt": prompt_name,
        "n": n,
        **{category: counts[category] for category in CATEGORIES.values()},
        "accuracy": counts["correct"] / n,
        "format_rate": (counts["correct"] + counts["formatted_incorrect"]) / n,
        "length_limit_rate": sum(r["finish_reason"] == "length" for r in records) / n,
        "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / n,
    }


def fenced(text: str) -> str:
    # A model can itself emit Markdown fences; keep the review document intact.
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}\n"


def write_reports(output_dir: Path, records_by_prompt: dict, seed: int, review_count: int):
    summaries = [summarize(name, records) for name, records in records_by_prompt.items()]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    review = [
        "# Prompting baseline review\n",
        "Accuracy below is the unmodified course grader's accuracy. Inspect the responses "
        "to distinguish mathematical errors from parsing failures. Human judgments are blank.\n",
        "| Prompt | N | Correct | Formatted incorrect | Format failure | Accuracy | Format rate | Length limit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        review.append(
            f"| {row['prompt']} | {row['n']} | {row['correct']} | {row['formatted_incorrect']} | "
            f"{row['format_failure']} | {row['accuracy']:.2%} | {row['format_rate']:.2%} | "
            f"{row['length_limit_rate']:.2%} |"
        )
    annotation_rows = []
    for name, records in records_by_prompt.items():
        review.append(f"\n## {name}\n")
        for category in CATEGORIES.values():
            candidates = [r for r in records if r["category"] == category]
            selected = random.Random(f"{seed}:{name}:{category}").sample(
                candidates, min(review_count, len(candidates))
            )
            review.append(f"\n### {category}: inspecting {len(selected)} of {len(candidates)}\n")
            for record in selected:
                review.extend([
                    f"\n#### Dataset row {record['example_id']}\n",
                    f"Reference answer: {record['ground_truth']}  \n"
                    f"Finish reason: {record['finish_reason']}\n",
                    "Prompt:\n" + fenced(record["prompt"]),
                    "Response:\n" + fenced(record["response"]),
                    "Human-correct? ___  Parsing failure? ___  Behavior / notes: ___\n",
                ])
                annotation_rows.append({
                    "prompt_name": name,
                    "example_id": record["example_id"],
                    "category": category,
                    "ground_truth": record["ground_truth"],
                    "question": record["question"],
                    "response": record["response"],
                    "human_correct": "",
                    "parsing_failure": "",
                    "notes": "",
                })
    (output_dir / "review.md").write_text("\n".join(review), encoding="utf-8")
    with (output_dir / "annotations.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    print(json.dumps(summaries, indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPT_FILES), default=list(PROMPT_FILES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="First N examples; 0 evaluates the complete file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--review-count", type=int, default=10, help="Examples per prompt/category for human review")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and save prompt previews without a GPU")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.limit < 0 or args.batch_size < 1 or args.review_count < 1:
        parser.error("limit must be >= 0; batch-size and review-count must be >= 1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    if len(set(args.prompts)) != len(args.prompts):
        parser.error("Each prompt may be selected only once")
    examples = load_examples(args.data, args.limit)
    templates = {
        name: (REPO_ROOT / "cs336_alignment/prompts" / PROMPT_FILES[name]).read_text(encoding="utf-8")
        for name in args.prompts
    }
    previews = {name: template.format(question=examples[0]["question"]) for name, template in templates.items()}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}; choose a new run directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("vllm", "torch", "transformers", "math-verify"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "num_examples": len(examples),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "templates": templates,
        "sampling": {name: sampling_config(name, args.seed) for name in args.prompts},
        "package_versions": versions,
        "grader": "course reward functions with fast=True, unchanged response text",
        "status": "dry_run" if args.dry_run else "running",
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "prompt_preview.json").write_text(json.dumps(previews, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Validated {len(examples)} examples and {len(templates)} prompts: {args.output_dir}")
        return

    started = perf_counter()
    try:
        from vllm import LLM, SamplingParams
        from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

        # Offline vLLM evaluates on one GPU and lets us explicitly set every
        # sampling option without inheriting a model's generation_config.json.
        engine = LLM(
            model=args.model,
            revision=args.revision,
            tokenizer_revision=args.revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            generation_config="vllm",
        )
        records_by_prompt = {}
        for name, template in templates.items():
            records_by_prompt[name] = evaluate_prompt(
                engine, SamplingParams(**sampling_config(name, args.seed)), examples, template,
                question_only_reward_fn if name == "question_only" else r1_zero_reward_fn,
                args.output_dir / f"{name}.jsonl", args.batch_size,
            )
            write_reports(args.output_dir, records_by_prompt, args.seed, args.review_count)
        config["status"] = "complete"
    except BaseException as error:
        config["status"] = "failed"
        config["error"] = repr(error)
        raise
    finally:
        config["elapsed_seconds"] = perf_counter() - started
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
