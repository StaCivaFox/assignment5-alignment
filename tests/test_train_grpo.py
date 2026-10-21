import json
from types import SimpleNamespace

import pytest
import torch

from cs336_alignment import grpo, train_grpo, vllm_utils
from cs336_alignment.plot_grpo import metric_series, plot_series


def small_args(tmp_path):
    args = train_grpo.make_parser().parse_args([
        "--output-dir", str(tmp_path / "run"), "--num-rollout-steps", "2",
        "--n-train-examples", "4", "--n-val-examples", "3",
        "--rollout-batch-size", "4", "--group-size", "2",
        "--gradient-accumulation-steps", "2", "--eval-every", "1",
        "--wandb-mode", "disabled",
    ])
    train_grpo.validate_args(args)
    args.output_dir.mkdir()
    return args


def test_loop_updates_and_evaluates_current_weights(tmp_path, tiny_train_model, tokenizer):
    args = small_args(tmp_path)
    examples = [{"example_id": i, "question": question, "ground_truth": "42"}
                for i, question in enumerate(["Hello", "This is", "Hello", "This is"])]
    initial_parameters = [p.detach().clone() for p in tiny_train_model.parameters()]
    reward_calls = []

    def reward_fn(response, truth):
        reward_calls.append((response, truth))
        score = float(response == "world")
        return {"reward": score, "format_reward": 1.0, "answer_reward": score}

    class FakeServer:
        """Fixed test responses; verifies that generation never uses stale weights."""
        def __init__(self):
            self.events = []
            self.synced = None

        def sync_policy_weights(self, model):
            self.events.append("sync")
            self.synced = [p.detach().clone() for p in model.parameters()]

        def generate_completions(self, prompts, params, batch_size=None):
            self.events.append("train" if params["n"] == 2 else "val")
            assert self.synced is not None
            for actual, synced in zip(tiny_train_model.parameters(), self.synced):
                torch.testing.assert_close(actual, synced, rtol=0, atol=0)
            assert params["stop"] == ["</answer>"]
            assert params["include_stop_str_in_output"]
            responses = ["world", "test"] * len(prompts) if params["n"] == 2 else ["world", "a test", "another test"][:len(prompts)]
            return [SimpleNamespace(text=text, token_ids=tokenizer.encode(text, add_special_tokens=False),
                                    finish_reason="stop") for text in responses]

    server = FakeServer()
    logger = train_grpo.ExperimentLogger(args, {})
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    summary = train_grpo.run_training_loop(
        args, tiny_train_model, tokenizer, optimizer, server,
        examples, examples[:3], "{question}", reward_fn, logger,
    )
    assert server.events == ["sync", "val", "sync", "train", "sync", "val", "sync", "train", "sync", "val"]
    assert any(not torch.equal(before, after) for before, after in zip(initial_parameters, tiny_train_model.parameters()))
    assert all(p.grad is None for p in tiny_train_model.parameters())
    assert len(reward_calls) == 2 * 4 + 3 * 3  # Scores reused inside the train step.
    assert summary["final_validation"]["val/mean_reward"] == pytest.approx(1 / 3)
    assert summary["final_validation"]["val/mean_response_tokens"] == pytest.approx(5 / 3)
    rows = [json.loads(line) for line in (args.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    elapsed = [row["time/elapsed_seconds"] for row in rows]
    assert 0 < elapsed[0] < elapsed[1] < elapsed[2]
    assert rows[1]["reward/mean_reward"] == 0.5
    assert rows[1]["reward/mean_format_reward"] == 1
    assert "train/token_entropy" in rows[1] and "train/grad_norm_before_clip" in rows[1]
    train_records = [json.loads(line) for line in (args.output_dir / "rollouts/train-step-0001.jsonl").read_text().splitlines()]
    assert [row["sample_index"] for row in train_records] == [0, 1, 0, 1]
    assert train_records[0]["prompt"] == train_records[1]["prompt"]
    assert all(row["policy_step"] == 0 for row in train_records)
    assert (args.output_dir / "rollouts/val-step-0000.jsonl").exists()
    assert (args.output_dir / "rollouts/val-step-0002.jsonl").exists()
    assert (args.output_dir / "checkpoints/step-0002/model.safetensors").exists()


@pytest.mark.parametrize("updates", [
    {"group_size": 1}, {"gradient_accumulation_steps": 3},
    {"group_size": 3}, {"sampling_temperature": 0.7}, {"sampling_top_p": 0.95},
])
def test_invalid_batch_and_sampling_configuration(tmp_path, updates):
    args = small_args(tmp_path)
    for key, value in updates.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        train_grpo.validate_args(args)


def test_empty_response_has_zero_loss_and_gradient():
    losses = torch.tensor([[1., 3.], [7., 9.]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    loss = grpo.aggregate_loss_across_microbatch(losses, mask)
    assert loss.item() == 1.
    loss.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[0.25, 0.25], [0., 0.]]))


def test_vllm_request_preserves_sampling_and_order(monkeypatch):
    requests = []

    def fake_http(method, url, payload, timeout):
        requests.append(payload)
        return {"choices": [{"index": i, "text": str(i), "token_ids": [i], "finish_reason": "stop"}
                            for i in (3, 1, 2, 0)]}

    monkeypatch.setattr(vllm_utils, "_http_json", fake_http)
    outputs = vllm_utils.generate_completions("http://fake", "test-model", ["a", "b"], {
        "n": 2, "temperature": 1., "top_p": 0.9, "max_tokens": 32, "seed": 1,
        "stop": ["</answer>"], "include_stop_str_in_output": True,
    })
    assert [output.text for output in outputs] == ["0", "1", "2", "3"]
    assert requests[0]["top_p"] == 0.9
    assert requests[0]["add_special_tokens"] is False
    assert requests[0]["include_stop_str_in_output"] is True


def test_plot_means_and_seed_variance():
    runs = [{"rows": [{"step": 0, "val/mean_reward": a}, {"step": 10, "val/mean_reward": b}]}
            for a, b in [(0., 0.2), (0.1, 0.4)]]
    steps, values, means, stds = metric_series(runs, "val/mean_reward")
    assert steps == [0, 10]
    assert means == pytest.approx([0.05, 0.3])
    assert stds == pytest.approx([0.1 / 2**0.5, 0.2 / 2**0.5])
    runs[1]["rows"].pop()
    with pytest.raises(ValueError, match="step coverage"):
        metric_series(runs, "val/mean_reward")


def test_time_plot_uses_measured_times_and_matches_updates_across_seeds():
    runs = [{"rows": [
        {"step": 0, "val/mean_reward": 0.0},  # Older logs did not time the baseline.
        {"step": 1, "val/mean_reward": 0.2, "time/elapsed_seconds": 12.0},
        {"step": 2, "val/mean_reward": 0.4, "time/elapsed_seconds": 20.0},
    ]}, {"rows": [
        {"step": 0, "val/mean_reward": 0.1, "time/elapsed_seconds": 2.0},
        {"step": 1, "val/mean_reward": 0.4, "time/elapsed_seconds": 16.0},
        {"step": 2, "val/mean_reward": 0.8, "time/elapsed_seconds": 40.0},
    ]}]
    xs, values, mean_x, means, stds = plot_series(runs, "val/mean_reward", wall_time=True)
    assert xs == [[12.0, 20.0], [16.0, 40.0]]
    assert values == [[0.2, 0.4], [0.4, 0.8]]
    assert mean_x == [14.0, 30.0]
    assert means == pytest.approx([0.3, 0.6])
    assert stds == pytest.approx([0.2 / 2**0.5, 0.4 / 2**0.5])
    assert plot_series(runs, "val/mean_reward")[2] == [0, 1, 2]


def test_server_respects_container_gpu_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return object()

    monkeypatch.setattr(vllm_utils.subprocess, "Popen", fake_popen)
    vllm_utils.start_server("test-model", "127.0.0.1", 8000, 1, 0, "auto", "ERROR", max_model_len=4096)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    command = captured["command"]
    assert command[command.index("--generation-config") + 1] == "vllm"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert "--disable-uvicorn-access-log" in command
    assert command[command.index("--uvicorn-log-level") + 1] == "warning"
