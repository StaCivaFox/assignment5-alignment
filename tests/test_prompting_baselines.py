"""CPU checks for evaluation bookkeeping, using the real course graders."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cs336_alignment import prompting_baselines as baseline
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


class FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return [
            SimpleNamespace(
                prompt=prompt,
                outputs=[SimpleNamespace(text=next(self.responses), token_ids=[1, 2], finish_reason="stop")],
            )
            for prompt in prompts
        ]


class PromptingBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data.jsonl"
        rows = [{"question": f"Question {i}?", "answer": "Reference reasoning.\n#### 72"} for i in range(3)]
        self.data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_reference_extraction_and_real_templates(self):
        examples = baseline.load_examples(self.data, limit=2)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["ground_truth"], "72")
        self.assertEqual(examples[1]["example_id"], 1)
        for filename in baseline.PROMPT_FILES.values():
            template = (baseline.REPO_ROOT / "cs336_alignment/prompts" / filename).read_text()
            rendered = template.format(question=examples[0]["question"])
            self.assertIn("Question 0?", rendered)
            self.assertNotIn("Reference reasoning", rendered)
            self.assertNotIn("{question}", rendered)

    def test_missing_reference_answer_fails(self):
        self.data.write_text(json.dumps({"question": "Q", "answer": "No delimiter"}))
        with self.assertRaisesRegex(ValueError, "missing GSM8K"):
            baseline.load_examples(self.data)

    def test_sampling_stop_rules(self):
        for name in baseline.PROMPT_FILES:
            config = baseline.sampling_config(name, 7)
            self.assertEqual((config["temperature"], config["top_p"], config["max_tokens"]), (1.0, 1.0, 512))
            self.assertEqual(config["n"], 1)
            self.assertEqual(config["seed"], 7)
            self.assertEqual(config["stop"], [] if name == "question_only" else ["</answer>"])
            self.assertEqual(config["include_stop_str_in_output"], name != "question_only")

    def test_real_r1_grader_categories_batch_alignment_and_reports(self):
        responses = [
            "Reasoning </think> <answer> 72 </answer>",
            "Reasoning </think> <answer> 73 </answer>",
            "The answer is 72.",
        ]
        path = self.root / "r1_zero.jsonl"
        records = baseline.evaluate_prompt(
            FakeEngine(responses), None, baseline.load_examples(self.data),
            "Question: {question}\nAssistant: <think>", r1_zero_reward_fn, path, batch_size=2,
        )
        saved = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["response"] for r in saved], responses)
        self.assertEqual([r["example_id"] for r in saved], [0, 1, 2])
        self.assertEqual([r["category"] for r in saved], list(baseline.CATEGORIES.values()))
        summary = baseline.summarize("r1_zero", records)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["format_rate"], 2 / 3)
        self.assertEqual(sum(summary[key] for key in baseline.CATEGORIES.values()), 3)
        baseline.write_reports(self.root, {"r1_zero": records}, seed=0, review_count=10)
        with (self.root / "annotations.csv").open() as source:
            annotations = list(csv.DictReader(source))
        self.assertEqual(len(annotations), 3)
        self.assertTrue(all(row["human_correct"] == "" for row in annotations))
        self.assertIn("inspecting 1 of 1", (self.root / "review.md").read_text())

    def test_question_only_uses_boxed_answer(self):
        records = baseline.evaluate_prompt(
            FakeEngine([r"Final answer: \boxed{72}", r"\boxed{73}", "<answer>72</answer>"]),
            None, baseline.load_examples(self.data), "{question}", question_only_reward_fn,
            self.root / "question_only.jsonl", batch_size=3,
        )
        self.assertEqual([r["category"] for r in records], list(baseline.CATEGORIES.values()))

    def test_grading_failure_keeps_raw_response(self):
        def broken_grader(response, answer):
            raise RuntimeError("grader failed")

        path = self.root / "failed.jsonl"
        with self.assertRaisesRegex(RuntimeError, "grader failed"):
            baseline.evaluate_prompt(
                FakeEngine(["raw text"]), None, baseline.load_examples(self.data, 1),
                "{question}", broken_grader, path, batch_size=1,
            )
        record = json.loads(path.read_text())
        self.assertEqual(record["response"], "raw text")
        self.assertIn("grading_error", record)
        self.assertNotIn("reward", record)

    def test_dry_run_and_overwrite_protection(self):
        destination = self.root / "preview"
        argv = ["--data", str(self.data), "--output-dir", str(destination), "--dry-run"]
        baseline.main(argv)
        config = json.loads((destination / "config.json").read_text())
        self.assertEqual(config["status"], "dry_run")
        self.assertEqual(config["num_examples"], 3)
        self.assertEqual(len(json.loads((destination / "prompt_preview.json").read_text())), 3)
        with self.assertRaises(SystemExit):
            baseline.main(argv)


if __name__ == "__main__":
    unittest.main()
