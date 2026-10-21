"""Check train-step logging, which the assignment snapshots do not inspect."""

import copy

import torch

from cs336_alignment import grpo, grpo_utils
from .test_grpo import _train_step_inputs, _train_step_optimizer, _train_step_reward_fn


def test_train_step_metadata(tiny_train_model, tokenizer, monkeypatch):
    prompts, responses, truths, _ = _train_step_inputs(tokenizer)
    tokenized = grpo_utils.tokenize_prompt_and_output(prompts, responses, tokenizer)
    mask = tokenized["response_mask"]
    with torch.no_grad():
        outputs = grpo_utils.get_response_log_probs(
            tiny_train_model, tokenized["input_ids"], tokenized["labels"],
            return_token_entropy=True,
        )
        expected_entropy = outputs["token_entropy"][mask].mean()

    observed_norms = {}
    original_clip = grpo_utils.gradient_clipping

    def observe_clipping(parameters, max_norm):
        parameters = list(parameters)

        def total_norm():
            return torch.cat([p.grad.detach().flatten() for p in parameters if p.grad is not None]).norm()

        observed_norms["before"] = total_norm()
        original_clip(parameters, max_norm)
        observed_norms["after"] = total_norm()

    monkeypatch.setattr(grpo_utils, "gradient_clipping", observe_clipping)
    loss, metadata = grpo.grpo_train_step(
        tiny_train_model, tokenizer, _train_step_optimizer(tiny_train_model),
        gradient_accumulation_steps=2, max_grad_norm=1e-3,
        reward_fn=_train_step_reward_fn, repeated_prompts=prompts,
        rollout_responses=responses, repeated_ground_truths=truths, group_size=2,
    )

    torch.testing.assert_close(metadata["train/loss"], loss.detach())
    torch.testing.assert_close(metadata["train/token_entropy"], expected_entropy)
    torch.testing.assert_close(metadata["train/grad_norm_before_clip"], observed_norms["before"])
    assert observed_norms["before"] > 1e-3
    assert observed_norms["after"] <= 1e-3
    torch.testing.assert_close(metadata["reward/mean_reward"], torch.tensor(0.375))
    torch.testing.assert_close(metadata["reward/mean_format_reward"], torch.tensor(0.5))
    torch.testing.assert_close(metadata["reward/mean_answer_reward"], torch.tensor(0.375))
    # The two microbatches have different means: averaging their stds is wrong.
    torch.testing.assert_close(
        metadata["advantage/std_reward"],
        torch.tensor([1., 0., 0.5, 0.]).std(correction=0),
    )
    assert metadata["advantage/min_reward"] == 0
    assert metadata["advantage/max_reward"] == 1
    assert metadata["train/num_response_tokens"] == 6
    assert metadata["train/mean_response_length"] == 1.5
    assert metadata["train/num_sequences"] == 4
    assert metadata["train/num_microbatches"] == 2
    for value in metadata.values():
        if isinstance(value, torch.Tensor):
            assert value.ndim == 0
            assert not value.requires_grad
            assert value.grad_fn is None
    assert all(p.grad is None for p in tiny_train_model.parameters())


def test_metadata_is_independent_of_microbatch_count(tiny_train_model, tokenizer, monkeypatch):
    prompts, responses, truths, _ = _train_step_inputs(tokenizer)
    original_loss = grpo.compute_policy_gradient_loss

    def loss_with_extra_metric(*args, **kwargs):
        loss, metadata = original_loss(*args, **kwargs)
        # A future variant's scalar mean should flow through automatically.
        metadata["example_variant_mean"] = args[0].square().mean()
        return loss, metadata

    monkeypatch.setattr(grpo, "compute_policy_gradient_loss", loss_with_extra_metric)
    runs = []
    for accumulation_steps in (1, 2):
        model = copy.deepcopy(tiny_train_model)
        loss, metadata = grpo.grpo_train_step(
            model, tokenizer, _train_step_optimizer(model),
            gradient_accumulation_steps=accumulation_steps, max_grad_norm=1.,
            reward_fn=_train_step_reward_fn, repeated_prompts=prompts,
            rollout_responses=responses, repeated_ground_truths=truths, group_size=2,
        )
        runs.append((model, loss, metadata))

    torch.testing.assert_close(runs[0][1], runs[1][1], rtol=1e-4, atol=1e-6)
    assert "policy/example_variant_mean" in runs[0][2]
    assert runs[0][2].keys() == runs[1][2].keys()
    for key in runs[0][2]:
        if key != "train/num_microbatches":
            torch.testing.assert_close(
                torch.as_tensor(runs[0][2][key]), torch.as_tensor(runs[1][2][key]),
                rtol=1e-4, atol=1e-6,
            )
    for p1, p2 in zip(runs[0][0].parameters(), runs[1][0].parameters(), strict=True):
        torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-6)
