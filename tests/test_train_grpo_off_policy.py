"""Exercise stale-policy reuse and experiment schedules without GPUs or downloads."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cs336_alignment import grpo, grpo_utils, train_grpo
from cs336_alignment.plot_grpo import load_runs, plot_series


@pytest.mark.parametrize("experiment,method,clip", [
    ("offpolicy_naive", "none", None), ("offpolicy_noclip", "noclip", None),
    ("offpolicy_clip", "grpo", .2), ("offpolicy_gspo", "gspo", 3e-4),
])
def test_off_policy_defaults(experiment, method, clip):
    args = train_grpo.make_parser().parse_args(["--experiment", experiment])
    train_grpo.validate_args(args)
    assert args.rollout_batch_size == 256
    assert args.train_batch_size == 8
    assert args.gradient_accumulation_steps == 1
    assert args.num_rollout_steps * (args.rollout_batch_size // args.train_batch_size) == 6400
    assert train_grpo.algorithm_settings(args)["importance_reweighting_method"] == method
    assert args.cliprange == clip


@pytest.mark.parametrize("experiment", [
    "offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo",
])
def test_loop_reuses_frozen_rollout_policy(
    tmp_path, tiny_train_model, tokenizer, monkeypatch, experiment,
):
    args = train_grpo.make_parser().parse_args([
        "--experiment", experiment, "--output-dir", str(tmp_path),
        "--num-rollout-steps", "2", "--rollout-batch-size", "4", "--train-batch-size", "2",
        "--group-size", "2", "--n-train-examples", "4", "--n-val-examples", "3",
        "--eval-every", "1", "--wandb-mode", "disabled",
    ])
    train_grpo.validate_args(args)
    examples = [{"example_id": i, "question": question, "ground_truth": "42"}
                for i, question in enumerate(["Hello", "This is", "Hello", "This is"])]
    model = tiny_train_model
    events, old_captures, updates, reward_calls = [], [], [], []
    before = [parameter.detach().clone() for parameter in model.parameters()]

    def reward_fn(response, truth):
        reward_calls.append(response)
        reward = float(response == "world")
        return {"reward": reward, "format_reward": 1., "answer_reward": reward}

    class FakeServer:
        def sync_policy_weights(self, policy):
            events.append("sync")
            self.synced = [p.detach().clone() for p in policy.parameters()]

        def generate_completions(self, prompts, params, batch_size=None):
            for actual, synced in zip(model.parameters(), self.synced, strict=True):
                torch.testing.assert_close(actual, synced, rtol=0, atol=0)
            events.append("train" if params["n"] == 2 else "val")
            responses = ["world", "another test"] * len(prompts) if params["n"] == 2 else ["world", "a test", "test"]
            return [SimpleNamespace(text=text, token_ids=tokenizer.encode(text), finish_reason="stop") for text in responses]

    original_capture = train_grpo.capture_old_log_probs

    def capture(policy, tokenizer, records, microbatch_size):
        events.append("capture")
        assert len(updates) in (0, 2)  # before any update of this rollout batch
        frozen = original_capture(policy, tokenizer, records, microbatch_size)
        assert frozen.device.type == "cpu" and not frozen.requires_grad
        old_captures.append(frozen.clone())
        return frozen

    monkeypatch.setattr(train_grpo, "capture_old_log_probs", capture)
    original_step = grpo.grpo_train_step

    def observe_step(*positional, **kwargs):
        events.append("update")
        assert positional[3] == 1  # accumulation steps per optimizer update
        assert len(positional[6]) == 2  # one complete group
        assert positional[6][0] == positional[6][1]
        assert kwargs["importance_reweighting_method"] == train_grpo.EXPERIMENTS[experiment]
        assert kwargs["cliprange"] == args.cliprange
        old = kwargs["old_log_probs"]
        if experiment == "offpolicy_naive":
            assert old is None
        else:
            start = (len(updates) % 2) * 2
            torch.testing.assert_close(old, old_captures[-1][start:start + 2], rtol=0, atol=0)
        result = original_step(*positional, **kwargs)
        updates.append(True)
        return result

    monkeypatch.setattr(grpo, "grpo_train_step", observe_step)
    logger = train_grpo.ExperimentLogger(args, {})
    summary = train_grpo.run_training_loop(
        args, model, tokenizer, torch.optim.SGD(model.parameters(), lr=.01), FakeServer(),
        examples, examples[:3], "{question}", reward_fn, logger,
    )
    capture_events = [] if experiment == "offpolicy_naive" else ["capture"]
    assert events == ["sync", "val"] + ["sync", "train", *capture_events, "update", "update", "sync", "val"] * 2
    assert len(updates) == 4
    assert len(old_captures) == (0 if experiment == "offpolicy_naive" else 2)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    assert len(reward_calls) == 2 * 4 + 3 * 3  # cached grading during training
    assert summary["completed_steps"] == 4
    assert summary["completed_rollout_steps"] == 2
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["step"] for row in rows if "val/mean_reward" in row] == [0, 2, 4]
    assert [row["progress/rollout_step"] for row in rows] == [0, 1, 1, 2, 2]
    assert [row["progress/policy_lag_updates"] for row in rows[1:]] == [0, 1, 0, 1]
    assert [row["progress/behavior_policy_step"] for row in rows[1:]] == [0, 0, 2, 2]
    assert all("policy/clip_fraction" in row for row in rows[1:])
    assert rows[2]["time/rollout_seconds"] == rows[4]["time/rollout_seconds"] == 0
    assert rows[2]["time/old_log_probs_seconds"] == rows[4]["time/old_log_probs_seconds"] == 0
    assert all(a["time/elapsed_seconds"] < b["time/elapsed_seconds"] for a, b in zip(rows, rows[1:]))
    records = [json.loads(line) for line in (tmp_path / "rollouts/train-step-0002.jsonl").read_text().splitlines()]
    assert all(record["policy_step"] == 2 for record in records)
    assert (tmp_path / "rollouts/val-step-0004.jsonl").exists()
    assert (tmp_path / "checkpoints/step-0004/model.safetensors").exists()


def test_old_probabilities_align_with_training_tokenization(tiny_train_model, tokenizer):
    records = [{"prompt": "Hello", "response": "world"},
               {"prompt": "This is", "response": "another test"}]
    batch = grpo_utils.tokenize_prompt_and_output(
        [row["prompt"] for row in records], [row["response"] for row in records], tokenizer,
    )
    original = copy.deepcopy(tiny_train_model)
    original.eval()
    with torch.no_grad():
        expected = grpo_utils.get_response_log_probs(original, batch["input_ids"], batch["labels"])["log_probs"]
    actual = train_grpo.capture_old_log_probs(tiny_train_model, tokenizer, records, 1)
    torch.testing.assert_close(actual, expected)
    assert tiny_train_model.training
    assert actual.dtype == torch.float32 and actual.grad_fn is None
    assert all(parameter.grad is None for parameter in tiny_train_model.parameters())


def test_low_precision_logits_produce_float32_log_probs():
    logits = torch.tensor([[[1.01, 2.03, -.71]]], dtype=torch.bfloat16)
    model = lambda input_ids: SimpleNamespace(logits=logits)
    result = grpo_utils.get_response_log_probs(model, torch.tensor([[0]]), torch.tensor([[1]]))
    assert result["log_probs"].dtype == torch.float32
    torch.testing.assert_close(result["log_probs"], logits.float().log_softmax(-1)[:, :, 1])


@pytest.mark.parametrize("flags", [
    ["--train-batch-size", "7"], ["--train-batch-size", "512"],
    ["--gradient-accumulation-steps", "3"], ["--cliprange", "-1"],
])
def test_invalid_off_policy_configuration(flags):
    args = train_grpo.make_parser().parse_args(["--experiment", "offpolicy_clip", *flags])
    with pytest.raises(ValueError):
        train_grpo.validate_args(args)


def test_rollout_axis_uses_completed_batches():
    rows = [{"step": 0, "val/mean_reward": 0., "time/elapsed_seconds": 1.}]
    for step in range(1, 5):
        rows.append({"step": step, "progress/rollout_step": (step + 1) // 2,
                     "progress/update_in_rollout": 1 + (step - 1) % 2,
                     "progress/updates_per_rollout": 2, "train/loss": step,
                     "time/elapsed_seconds": 10. * step})
    runs = [{"rows": rows}]
    assert plot_series(runs, "train/loss", step_axis="rollout")[2] == [1, 2]
    assert plot_series(runs, "train/loss", step_axis="rollout")[3] == [2, 4]
    assert plot_series(runs, "train/loss", wall_time=True, step_axis="rollout")[2] == [20, 40]
    assert plot_series(runs, "train/loss")[2] == [1, 2, 3, 4]


def test_plot_rejects_different_clipranges(tmp_path):
    directories = []
    for seed, clip in enumerate([.2, .1]):
        args = train_grpo.make_parser().parse_args([
            "--experiment", "offpolicy_clip", "--seed", str(seed), "--cliprange", str(clip),
            "--output-dir", str(tmp_path / f"seed-{seed}"),
        ])
        train_grpo.prepare_run(args)
        train_grpo.write_json(args.output_dir / "status.json", {"status": "complete"})
        (args.output_dir / "metrics.jsonl").write_text('{"step": 0}\n')
        directories.append(args.output_dir)
    with pytest.raises(ValueError, match="off-policy settings"):
        load_runs(directories)
