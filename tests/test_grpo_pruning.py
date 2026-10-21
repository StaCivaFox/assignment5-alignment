"""Pruning must save forward passes without changing the full-batch update."""

import copy
import math

import pytest
import torch

from cs336_alignment import grpo, grpo_utils


def rollout_batch():
    prompts = ["Hello"] * 4 + ["This is"] * 4 + ["Question"] * 4
    responses = [
        "world", "test", "a", "another",  # all wrong
        "world test", "a test", "another test", "test world",  # mixed
        "world world", "a a", "test test", "another another",  # all correct
    ]
    rewards = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.])
    scores = dict(zip(responses, rewards.tolist(), strict=True))

    def reward_fn(response, truth):
        value = scores[response]
        return {"reward": value, "format_reward": value, "answer_reward": value}

    return prompts, responses, rewards, reward_fn


@pytest.mark.parametrize("baseline,normalizer", [
    ("mean", "std"), ("mean", "none"), ("none", "none"), ("mean", "mean"),
    ("none", "std"), ("none", "mean"),
])
@pytest.mark.parametrize("loss_normalization", ["sequence", "constant"])
def test_pruned_update_matches_unpruned_batch(
    tiny_train_model, tokenizer, monkeypatch, baseline, normalizer, loss_normalization,
):
    prompts, responses, rewards, reward_fn = rollout_batch()
    tokenized = grpo_utils.tokenize_prompt_and_output(prompts, responses, tokenizer)
    grouped = rewards.reshape(-1, 4)
    means = grouped.mean(-1, keepdim=True)
    stds = grouped.std(-1, keepdim=True)
    advantages = grouped - (means if baseline == "mean" else 0)
    if normalizer == "std":
        advantages = advantages / (stds + 1e-6)
    elif normalizer == "mean":
        advantages = advantages / (means + 1e-6)
    advantages = advantages.flatten()
    retained = advantages.nonzero(as_tuple=True)[0]

    # Independent full-batch objective; no pruning or gradient accumulation.
    reference = copy.deepcopy(tiny_train_model)
    outputs = grpo_utils.get_response_log_probs(
        reference, tokenized["input_ids"], tokenized["labels"], return_token_entropy=True,
    )
    mask = tokenized["response_mask"]
    per_sequence = (-advantages[:, None] * outputs["log_probs"]).masked_fill(~mask, 0).sum(-1)
    if loss_normalization == "sequence":
        expected_loss = (per_sequence / mask.sum(-1).clamp_min(1)).mean()
    else:
        expected_loss = per_sequence.sum() / (len(prompts) * 8)
    expected_loss.backward()
    expected_gradients = [p.grad.detach().clone() for p in reference.parameters()]
    expected_entropy = outputs["token_entropy"][retained][mask[retained]].mean()

    original_forward = grpo_utils.get_response_log_probs
    calls = []

    def observe_forward(model, input_ids, labels, **kwargs):
        calls.append(input_ids.detach().clone())
        return original_forward(model, input_ids, labels, **kwargs)

    monkeypatch.setattr(grpo_utils, "get_response_log_probs", observe_forward)
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=1e-3)
    original_step = optimizer.step

    def check_gradients():
        for parameter, expected in zip(tiny_train_model.parameters(), expected_gradients, strict=True):
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=1e-5)
        return original_step()

    monkeypatch.setattr(optimizer, "step", check_gradients)
    loss, metadata = grpo.grpo_train_step(
        tiny_train_model, tokenizer, optimizer,
        gradient_accumulation_steps=4, max_grad_norm=None,
        reward_fn=reward_fn, repeated_prompts=prompts,
        rollout_responses=responses, repeated_ground_truths=["42"] * len(prompts),
        group_size=4, baseline=baseline, advantage_normalizer=normalizer,
        loss_normalization=loss_normalization, normalization_constant=len(prompts) * 8,
    )
    # Microbatch size 3 splits groups of 4; pruning leaves a shorter final batch.
    assert [len(batch) for batch in calls] == [
        min(3, len(retained) - start) for start in range(0, len(retained), 3)
    ]
    torch.testing.assert_close(torch.cat(calls), tokenized["input_ids"][retained])
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=2e-4, atol=1e-5)
    assert metadata["train/num_microbatches"] == math.ceil(len(retained) / 3)
    assert metadata["train/num_retained_sequences"] == len(retained)
    assert metadata["train/fraction_pruned_sequences"] == pytest.approx(1 - len(retained) / len(prompts))
    assert metadata["train/num_response_tokens"] == mask.sum()
    assert metadata["train/num_entropy_tokens"] == mask[retained].sum()
    torch.testing.assert_close(metadata["train/token_entropy"], expected_entropy)
    torch.testing.assert_close(metadata["reward/mean_reward"], rewards.mean())
    torch.testing.assert_close(metadata["policy/fraction_zero_weights"], (advantages == 0).float().mean())
    torch.testing.assert_close(metadata["advantage/mean_group_std"], stds.mean())
    torch.testing.assert_close(metadata["advantage/fraction_zero_std_groups"], torch.tensor(2 / 3))
    assert all(parameter.grad is None for parameter in tiny_train_model.parameters())


@pytest.mark.parametrize("baseline,normalizer,reward", [
    ("mean", "std", 0.), ("mean", "std", 1.), ("mean", "none", 0.),
    ("mean", "mean", 0.), ("none", "none", 0.),
])
def test_all_pruned_batch_preserves_zero_gradient_optimizer_step(
    tiny_train_model, tokenizer, monkeypatch, baseline, normalizer, reward,
):
    model = tiny_train_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    # Populate momentum: an optimizer step with zero gradients is not a no-op.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.1)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for parameter in reference.parameters():
        parameter.grad = torch.zeros_like(parameter)
    reference_optimizer.step()

    def forbidden_forward(*args, **kwargs):
        pytest.fail("zero-advantage sequences must not be forwarded")

    monkeypatch.setattr(grpo_utils, "get_response_log_probs", forbidden_forward)

    def reward_fn(response, truth):
        return {"reward": reward, "format_reward": reward, "answer_reward": reward}

    loss, metadata = grpo.grpo_train_step(
        model, tokenizer, optimizer, gradient_accumulation_steps=2, max_grad_norm=1.,
        reward_fn=reward_fn, repeated_prompts=["Hello"] * 4,
        rollout_responses=["world"] * 4, repeated_ground_truths=["42"] * 4,
        group_size=2, baseline=baseline, advantage_normalizer=normalizer,
        loss_normalization="constant", normalization_constant=32,
    )
    assert loss == 0
    assert metadata["train/grad_norm_before_clip"] == 0
    assert metadata["train/num_microbatches"] == 0
    assert metadata["train/fraction_pruned_sequences"] == 1
    assert metadata["train/token_entropy"] == 0
    assert metadata["train/num_entropy_tokens"] == 0
    assert metadata["train/num_response_tokens"] == 4
    assert all(math.isfinite(float(value)) for value in metadata.values())
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
