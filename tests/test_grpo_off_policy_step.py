"""Check off-policy updates against an unpruned, full-batch objective."""

import copy

import pytest
import torch

from cs336_alignment import grpo, grpo_utils
from .test_grpo_pruning import rollout_batch


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
@pytest.mark.parametrize("normalization", ["sequence", "constant"])
@pytest.mark.parametrize("baseline", ["mean", "none"])
def test_off_policy_pruning_and_accumulation(
    tiny_train_model, tokenizer, monkeypatch, method, normalization, baseline,
):
    prompts, responses, rewards, _ = rollout_batch()
    # Different response lengths make token-weighted clip fractions observable.
    responses = [response + " world" * (i % 3) for i, response in enumerate(responses)]
    scores = dict(zip(responses, rewards.tolist(), strict=True))

    def reward_fn(response, truth):
        reward = scores[response]
        return {"reward": reward, "format_reward": reward, "answer_reward": reward}

    batch = grpo_utils.tokenize_prompt_and_output(prompts, responses, tokenizer)
    mask = batch["response_mask"]
    n, length = mask.shape
    grouped = rewards.reshape(-1, 4)
    advantages = (grouped - grouped.mean(-1, keepdim=True)).flatten() if baseline == "mean" else rewards
    retained = advantages.nonzero(as_tuple=True)[0]
    reference = copy.deepcopy(tiny_train_model)
    log_probs = grpo_utils.get_response_log_probs(reference, batch["input_ids"], batch["labels"])["log_probs"]
    log_ratios = torch.tensor([.1, -.1, .3, -.3, .4, -.4, .4, .1, -.4, .4, .1, -.1])[:, None].expand(n, length)
    # Give prompt/padding positions different ratios: GSPO must ignore them.
    log_ratios = log_ratios.masked_fill(~mask, 2.)
    old_values = log_probs.detach() - log_ratios
    # Real rollout batches can be padded wider than a subsequent training batch.
    old = torch.cat([old_values, torch.full((n, 2), float("nan"))], dim=1).requires_grad_()
    old_before = old.detach().clone()

    # Build the objective directly, without the policy-loss or aggregation helper.
    differences = log_probs - old_values
    if method == "gspo":
        ratios = (differences.masked_fill(~mask, 0).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)).exp()
    else:
        ratios = differences.exp()
    if method == "noclip":
        objective = advantages[:, None] * ratios
    else:
        objective = advantages[:, None] * torch.where(
            advantages[:, None] >= 0, ratios.clamp_max(1.2), ratios.clamp_min(.8),
        )
    per_sequence = -objective.expand_as(mask).masked_fill(~mask, 0).sum(-1)
    expected_loss = (
        (per_sequence / mask.sum(-1)).mean()
        if normalization == "sequence" else per_sequence.sum() / (n * 8)
    )
    expected_loss.backward()
    expected_gradients = [p.grad.clone() for p in reference.parameters()]

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=1e-3)
    original_step = optimizer.step
    step_calls = []

    def check_gradients():
        step_calls.append(True)
        for actual, expected in zip(tiny_train_model.parameters(), expected_gradients, strict=True):
            torch.testing.assert_close(actual.grad, expected, rtol=2e-4, atol=1e-6)
        original_step()

    monkeypatch.setattr(optimizer, "step", check_gradients)
    original_loss = grpo.compute_policy_gradient_loss
    received_old = []

    def observe_loss(*args, **kwargs):
        received_old.append(kwargs["old_log_probs"].clone())
        assert not kwargs["old_log_probs"].requires_grad
        assert kwargs["importance_reweighting_method"] == method
        assert kwargs["cliprange"] == .2
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(grpo, "compute_policy_gradient_loss", observe_loss)
    loss, metadata = grpo.grpo_train_step(
        tiny_train_model, tokenizer, optimizer, gradient_accumulation_steps=4,
        max_grad_norm=None, reward_fn=reward_fn, repeated_prompts=prompts,
        rollout_responses=responses, repeated_ground_truths=["42"] * n,
        group_size=4, baseline=baseline, advantage_normalizer="none",
        importance_reweighting_method=method, old_log_probs=old, cliprange=.2,
        loss_normalization=normalization, normalization_constant=n * 8,
    )
    assert len(step_calls) == 1
    assert [len(chunk) for chunk in received_old] == [3, len(retained) - 3]
    torch.testing.assert_close(torch.cat(received_old), old_values[retained])
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(old, old_before, equal_nan=True)
    assert old.grad is None
    assert all(p.grad is None for p in tiny_train_model.parameters())
    if method != "noclip":
        with torch.no_grad():
            clipped = ((advantages[:, None] > 0) & (ratios > 1.2)) | ((advantages[:, None] < 0) & (ratios < .8))
            valid = mask if method == "grpo" else mask.any(-1, keepdim=True)
            valid = valid & (advantages[:, None] != 0)
            assert metadata["policy/clip_count"] == (clipped & valid).sum()
            assert metadata["policy/clip_total"] == valid.sum()
            torch.testing.assert_close(metadata["policy/clip_fraction"], (clipped & valid).sum() / valid.sum())


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_all_pruned_off_policy_batch(tiny_train_model, tokenizer, monkeypatch, method):
    def forbidden_forward(*args, **kwargs):
        pytest.fail("zero-advantage responses must not be forwarded")

    monkeypatch.setattr(grpo_utils, "get_response_log_probs", forbidden_forward)
    old = torch.full((4, 1), -2., requires_grad=True)
    loss, metadata = grpo.grpo_train_step(
        tiny_train_model, tokenizer, torch.optim.SGD(tiny_train_model.parameters(), lr=.001),
        gradient_accumulation_steps=2, max_grad_norm=1.,
        reward_fn=lambda response, truth: {"reward": 0., "format_reward": 0., "answer_reward": 0.},
        repeated_prompts=["Hello"] * 4, rollout_responses=["world"] * 4,
        repeated_ground_truths=["42"] * 4, group_size=2,
        importance_reweighting_method=method, old_log_probs=old, cliprange=.2,
    )
    assert loss == 0
    assert old.grad is None
    assert metadata["train/num_microbatches"] == 0
    if method != "noclip":
        assert metadata["policy/clip_fraction"] == 0
        assert metadata["policy/clip_total"] == 0


@pytest.mark.parametrize("updates,message", [
    ({"old_log_probs": None}, "old_log_probs"),
    ({"cliprange": None}, "cliprange"),
    ({"cliprange": -.1}, "cliprange"),
    ({"old_log_probs": torch.zeros(3, 1)}, "every original sequence"),
    ({"old_log_probs": torch.zeros(4, 0)}, "every original sequence"),
])
def test_invalid_off_policy_inputs(tiny_train_model, tokenizer, updates, message):
    args = dict(
        model=tiny_train_model, tokenizer=tokenizer,
        optimizer=torch.optim.SGD(tiny_train_model.parameters(), lr=.001),
        gradient_accumulation_steps=2, max_grad_norm=1.,
        reward_fn=lambda response, truth: {"reward": 0., "format_reward": 0., "answer_reward": 0.},
        repeated_prompts=["Hello"] * 4, rollout_responses=["world"] * 4,
        repeated_ground_truths=["42"] * 4, group_size=2,
        importance_reweighting_method="grpo", old_log_probs=torch.zeros(4, 1), cliprange=.2,
    )
    args.update(updates)
    with pytest.raises(ValueError, match=message):
        grpo.grpo_train_step(**args)
