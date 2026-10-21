from collections.abc import Callable
from typing import Literal
import torch
from einops import einsum, rearrange
from transformers import PreTrainedTokenizer, PreTrainedModel
from cs336_alignment import grpo_utils

def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    answer_rewards = []
    for response, truth in zip(rollout_responses, repeated_ground_truths, strict=True):
        scores = reward_fn(response, truth)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])
    raw_rewards = torch.tensor(rewards)
    format_rewards_mean = torch.tensor(format_rewards).mean().item()
    answer_rewards_mean = torch.tensor(answer_rewards).mean().item()
    metadata = {}
    metadata["mean_reward"] = raw_rewards.mean().item()
    metadata["mean_format_reward"] = format_rewards_mean
    metadata["mean_answer_reward"] = answer_rewards_mean
    return (raw_rewards, metadata)

def compute_group_normalized_rewards(
        raw_rewards: torch.Tensor, 
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    # Reshape raw_rewards to (B, G)
    grouped = rearrange(raw_rewards, "(B G) -> B G", G = group_size)
    batch_size = grouped.shape[0]
    if baseline == "none":
        group_mean = torch.zeros(batch_size, 1)
    elif baseline == "mean":
        group_mean = grouped.mean(dim=-1, keepdim=True) # (B, 1)
    else:
        raise NotImplementedError
    if advantage_normalizer == "std":
        group_std = grouped.std(dim=-1, correction=1, keepdim=True) # (B, 1)
    elif advantage_normalizer == "none":
        group_std = torch.ones(batch_size, 1)
    elif advantage_normalizer == "mean":
        group_std = group_mean
    else:
        raise NotImplementedError
    normalized = (grouped - group_mean) / (group_std + advantage_eps) # (B, G)
    # Flatten normalized rewards
    flatten = rearrange(normalized, "B G -> (B G)")

    # Logging statistics only; use population std to describe this batch.
    with torch.no_grad():
        metadata = {
            "mean_reward": raw_rewards.mean().item(),
            "std_reward": raw_rewards.std(correction=0).item(),
            "min_reward": raw_rewards.min().item(),
            "max_reward": raw_rewards.max().item(),
            "mean_group_std": group_std.mean().item(),
            # Identical rewards within a group give no relative learning signal.
            "fraction_zero_std_groups": (group_std == 0).float().mean().item(),
            "mean_advantage": flatten.mean().item(),
            "std_advantage": flatten.std(correction=0).item(),
        }
    return flatten, metadata

def compute_policy_gradient_loss_on_policy(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = raw_rewards_or_advantages
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1) # (B, 1)
    gradient_loss = - weights.detach() * policy_log_probs # (B, L)

    # Logging only: weights may be raw rewards or normalized advantages.
    with torch.no_grad():
        metadata = {
            "mean_weight": weights.mean(),
            "std_weight": weights.std(correction=0),
            "fraction_positive_weights": (weights > 0).float().mean(),
            "fraction_negative_weights": (weights < 0).float().mean(),
            "fraction_zero_weights": (weights == 0).float().mean(),
            # These include every position; the training objective is masked later.
            "mean_unmasked_log_prob": policy_log_probs.mean(),
            "mean_unmasked_policy_gradient_loss": gradient_loss.mean(),
        }
    return gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    # Filter out not-response tokens that we shouldn't aggregate loss for
    response_len = mask.sum(dim=-1) # (B)
    masked = torch.masked_fill(per_token_policy_gradient_loss, ~mask, 0.0) # (B, L)
    if loss_normalization == "constant":
        return masked.sum() / normalization_constant
    elif loss_normalization == "sequence":
        # An empty generated response contributes zero, without dividing by zero.
        seq_ave_policy_gradient_loss = masked.sum(dim=-1) / response_len.clamp_min(1) # (B)
        batch_ave_policy_gradient_loss = seq_ave_policy_gradient_loss.mean() # scaler tensor
        return batch_ave_policy_gradient_loss
    else:
        raise NotImplementedError

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Take one update and return detached logging metrics.

    Helper metrics use reward/, advantage/, and policy/ prefixes. Scalar means
    and fractions are weighted by microbatch size; standard deviations and
    extrema are computed over the full batch. Future variants can return new
    scalar means/fractions in their helper metadata; other reductions should be
    added explicitly below. Token entropy is averaged over response tokens.
    """
    metric_totals = {}
    metric_counts = {}
    reward_batches = []
    advantage_batches = []
    entropy_sum = None
    num_response_tokens = None
    num_microbatches = 0

    def accumulate_means(prefix, statistics, count):
        # These statistics need full-batch reductions, not averages of scalars.
        full_batch_statistics = {
            "std_reward", "min_reward", "max_reward", "std_advantage", "std_weight",
        }
        for name, value in statistics.items():
            if name in full_batch_statistics:
                continue
            key = f"{prefix}/{name}"
            value = torch.as_tensor(value).detach().float()
            metric_totals[key] = metric_totals.get(key, 0) + value * count
            metric_counts[key] = metric_counts.get(key, 0) + count

    # Tokenize all inputs and outputs
    tokenized_result = grpo_utils.tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    device = next(model.parameters()).device
    input_ids, labels, response_mask = (tokenized_result[k].to(device) for k in ("input_ids", "labels", "response_mask"))
    input_len = len(repeated_prompts)
    microbatch_size = input_len // gradient_accumulation_steps
    returned_loss = torch.tensor(0.0, device=device)
    # For each microbatch, compute loss and accumulate gradient
    for i in range(0, input_len, microbatch_size):
        inputs_microbatch = input_ids[i:i + microbatch_size] # (microbatch_size, seq_len)
        labels_microbatch = labels[i:i + microbatch_size]
        # Make one forward pass and calculate log probs
        policy_outputs = grpo_utils.get_response_log_probs(
            model, inputs_microbatch, labels_microbatch, return_token_entropy=True,
        )
        log_probs_microbatch = policy_outputs["log_probs"]
        token_entropy = policy_outputs.pop("token_entropy").detach()
        # compute policy loss
        raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses[i:i + microbatch_size], repeated_ground_truths[i:i + microbatch_size])
        normalized_rewards, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
        normalized_rewards = normalized_rewards.to(device)
        per_token_gradient_loss, policy_metadata = compute_policy_gradient_loss_on_policy(normalized_rewards, log_probs_microbatch, importance_reweighting_method, old_log_probs, cliprange, response_mask[i:i + microbatch_size])
        # Logging is independent of the loss normalization/reweighting variant.
        with torch.no_grad():
            count = len(inputs_microbatch)
            accumulate_means("reward", reward_metadata, count)
            accumulate_means("advantage", advantage_metadata, count)
            accumulate_means("policy", policy_metadata, count)
            reward_batches.append(raw_rewards.detach())
            advantage_batches.append(normalized_rewards.detach())
            mask = response_mask[i:i + microbatch_size]
            microbatch_entropy_sum = token_entropy.masked_select(mask).sum(dtype=torch.float32)
            microbatch_token_count = mask.sum()
            entropy_sum = microbatch_entropy_sum if entropy_sum is None else entropy_sum + microbatch_entropy_sum
            num_response_tokens = microbatch_token_count if num_response_tokens is None else num_response_tokens + microbatch_token_count
            num_microbatches += 1
        loss = aggregate_loss_across_microbatch(per_token_gradient_loss, response_mask[i:i + microbatch_size], loss_normalization, normalization_constant)
        # Reweight loss so it can accumulate across microbatches
        loss = loss * (microbatch_size / input_len)
        # Backward pass
        loss.backward()
        returned_loss += loss.detach()
    # Observe the accumulated norm before the existing clipping/update logic.
    with torch.no_grad():
        grad_norms = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.stack(grad_norms).norm() if grad_norms else torch.tensor(0.0)
    # Clip the gradients
    if max_grad_norm is not None:
        grpo_utils.gradient_clipping(model.parameters(), max_grad_norm)
    # Update weights once across the entire batch
    optimizer.step()
    # Zero gradients once across entire batch.
    optimizer.zero_grad()
    with torch.no_grad():
        metadata = {key: total / metric_counts[key] for key, total in metric_totals.items()}
        all_rewards = torch.cat(reward_batches)
        all_advantages = torch.cat(advantage_batches)
        metadata.update({
            "advantage/std_reward": all_rewards.std(correction=0),
            "advantage/min_reward": all_rewards.min(),
            "advantage/max_reward": all_rewards.max(),
            "advantage/std_advantage": all_advantages.std(correction=0),
            "policy/std_weight": all_advantages.std(correction=0),
            "train/loss": returned_loss.detach().clone(),
            "train/grad_norm_before_clip": grad_norm,
            "train/token_entropy": entropy_sum / num_response_tokens.clamp_min(1),
            "train/num_response_tokens": num_response_tokens,
            "train/mean_response_length": num_response_tokens / input_len,
            "train/num_sequences": float(input_len),
            "train/num_microbatches": float(num_microbatches),
        })
    return returned_loss, metadata
