from collections.abc import Callable
import math
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
    group_mean = grouped.mean(dim=-1, keepdim=True)
    # Keep the actual reward statistics separate from the chosen normalizer.
    group_std = grouped.std(dim=-1, correction=1, keepdim=True) if group_size > 1 else torch.zeros_like(group_mean)
    if baseline == "none":
        baseline_value = torch.zeros_like(group_mean)
    elif baseline == "mean":
        baseline_value = group_mean
    else:
        raise NotImplementedError
    if advantage_normalizer == "std":
        normalizer = group_std + advantage_eps
    elif advantage_normalizer == "none":
        normalizer = torch.ones_like(group_mean)
    elif advantage_normalizer == "mean":
        normalizer = group_mean + advantage_eps
    else:
        raise NotImplementedError
    normalized = (grouped - baseline_value) / normalizer # (B, G)
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

def _validate_reweighting_arguments(method, old_log_probs, cliprange):
    if method not in ("none", "noclip", "grpo", "gspo"):
        raise ValueError(f"unknown importance reweighting method: {method}")
    if method != "none" and old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy importance reweighting")
    if method in ("grpo", "gspo") and (
        cliprange is None or not math.isfinite(cliprange) or cliprange < 0
    ):
        raise ValueError("clipped importance reweighting requires a finite, nonnegative cliprange")


def compute_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_reweighting_arguments(importance_reweighting_method, old_log_probs, cliprange)
    if importance_reweighting_method != "none" and old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs must have the same shape as policy_log_probs")
    if importance_reweighting_method == "gspo" and response_mask is None:
        raise ValueError("response_mask is required for GSPO")
    weights = raw_rewards_or_advantages
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1) # (B, 1)
    if importance_reweighting_method == "none":
        gradient_loss = - weights.detach() * policy_log_probs # (B, L)
    elif importance_reweighting_method == "noclip":
        reweight_factor = torch.exp(policy_log_probs - old_log_probs.detach())
        gradient_loss = - weights.detach() * reweight_factor
    elif importance_reweighting_method == "grpo":
        reweight_factor = torch.exp(policy_log_probs - old_log_probs.detach()) # (B, L)
        clipped_reweight_factor = reweight_factor.clamp(1 - cliprange, 1 + cliprange)
        gradient_loss = - torch.minimum(
            weights.detach() * reweight_factor,
            weights.detach() * clipped_reweight_factor,
        )
    elif importance_reweighting_method == "gspo":
        log_ratio = policy_log_probs - old_log_probs.detach()
        masked_log_ratio = torch.masked_fill(log_ratio, ~response_mask, 0.0) # (B, L)
        response_len = response_mask.sum(dim=-1, keepdim=True) # (B, 1)
        reweight_factor = torch.exp(masked_log_ratio.sum(dim=-1, keepdim=True) / response_len.clamp_min(1)).expand_as(log_ratio)
        clipped_reweight_factor = reweight_factor.clamp(1 - cliprange, 1 + cliprange)
        gradient_loss = - torch.minimum(
            weights.detach() * reweight_factor,
            weights.detach() * clipped_reweight_factor,
        )
    else:
        raise NotImplementedError


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
        if importance_reweighting_method in ("grpo", "gspo"):
            # Count the terms whose improving-direction gradient is clipped,
            # rather than every ratio outside the symmetric clipping interval.
            ratios = reweight_factor
            valid = torch.ones_like(policy_log_probs, dtype=torch.bool) if response_mask is None else response_mask
            if importance_reweighting_method == "gspo":
                ratios = ratios[:, :1]
                valid = valid.any(dim=-1, keepdim=True)
            clipped = ((weights > 0) & (ratios > 1 + cliprange)) | (
                (weights < 0) & (ratios < 1 - cliprange)
            )
            metadata["clip_count"] = (clipped & valid).sum()
            metadata["clip_total"] = valid.sum()
            metadata["clip_fraction"] = metadata["clip_count"] / metadata["clip_total"].clamp_min(1)
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
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("constant loss normalization requires a positive normalization_constant")
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
    """Take one on- or off-policy update, pruning zero-advantage responses.

    Advantages are computed over complete groups before pruning/batching.
    ``normalization_constant`` is the denominator for the entire original batch
    (typically original_batch_size * max_generation_length), not a microbatch.
    Neither normalization mode renormalizes over the retained responses.

    ``old_log_probs`` must be frozen probabilities from the rollout policy,
    aligned with the original unpruned batch and shifted token labels. Extra
    right-padding columns from a larger rollout batch are allowed. They are
    sliced along with the responses, never recomputed under the current model.
    The standard GSPO objective uses sequence loss normalization.

    Reward/advantage and policy weight metrics describe the original batch.
    Entropy and unmasked model-output metrics describe only forwarded responses;
    entropy is zero when no response tokens were forwarded, with coverage logged
    in train/num_entropy_tokens. Additional scalar helper means/fractions are
    averaged by the number of forwarded sequences.
    Clip fraction counts forwarded response tokens for GRPO and forwarded
    nonempty responses for GSPO; clip_total records that denominator.
    """
    input_len = len(repeated_prompts)
    if input_len == 0 or not (input_len == len(rollout_responses) == len(repeated_ground_truths)):
        raise ValueError("prompts, responses, and ground truths must have equal, nonzero lengths")
    if group_size <= 0 or input_len % group_size:
        raise ValueError("the batch must contain complete groups")
    if not 1 <= gradient_accumulation_steps <= input_len or input_len % gradient_accumulation_steps:
        raise ValueError("gradient_accumulation_steps must divide the original batch size")
    _validate_reweighting_arguments(importance_reweighting_method, old_log_probs, cliprange)
    if loss_normalization not in ("sequence", "constant"):
        raise ValueError(f"unknown loss normalization: {loss_normalization}")
    if loss_normalization == "constant" and (normalization_constant is None or normalization_constant <= 0):
        raise ValueError("constant loss normalization requires a positive normalization_constant")

    metric_totals = {}
    metric_counts = {}
    num_microbatches = 0

    def accumulate_means(prefix, statistics, count):
        # These statistics need full-batch reductions, not averages of scalars.
        full_batch_statistics = {
            "std_reward", "min_reward", "max_reward", "std_advantage", "std_weight",
        }
        for name, value in statistics.items():
            if name in full_batch_statistics or name == "clip_fraction":
                continue
            key = f"{prefix}/{name}"
            value = torch.as_tensor(value).detach().float()
            if name in ("clip_count", "clip_total"):
                # Sum counts, then compute the fraction once over the batch.
                metric_totals[key] = metric_totals.get(key, 0) + value
                metric_counts[key] = 1
                continue
            metric_totals[key] = metric_totals.get(key, 0) + value * count
            metric_counts[key] = metric_counts.get(key, 0) + count

    # Compute rewards/advantages before splitting: microbatches may cut groups.
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths,
    )
    normalized_rewards, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer,
    )
    retained_indices = normalized_rewards.nonzero(as_tuple=True)[0]
    num_retained = retained_indices.numel()
    accumulate_means("reward", reward_metadata, input_len)
    accumulate_means("advantage", advantage_metadata, input_len)

    # Keep the original mask for full-rollout length statistics. Only retained
    # microbatches are transferred to the model's device and forwarded.
    tokenized_result = grpo_utils.tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    device = next(model.parameters()).device
    input_ids, labels, response_mask = (tokenized_result[k] for k in ("input_ids", "labels", "response_mask"))
    if importance_reweighting_method != "none":
        if old_log_probs.ndim != 2 or old_log_probs.shape[0] != input_len or old_log_probs.shape[1] < labels.shape[1]:
            raise ValueError("old_log_probs must cover every original sequence and shifted token position")
        old_log_probs = old_log_probs.detach()[:, :labels.shape[1]]
    microbatch_size = input_len // gradient_accumulation_steps
    returned_loss = torch.tensor(0.0, device=device)
    entropy_sum = torch.tensor(0.0, device=device)
    num_entropy_tokens = torch.tensor(0, device=device)
    optimizer.zero_grad(set_to_none=True)
    # For each microbatch, compute loss and accumulate gradient
    for i in range(0, num_retained, microbatch_size):
        indices = retained_indices[i:i + microbatch_size]
        inputs_microbatch = input_ids[indices].to(device)
        labels_microbatch = labels[indices].to(device)
        mask = response_mask[indices].to(device)
        # Make one forward pass and calculate log probs
        policy_outputs = grpo_utils.get_response_log_probs(
            model, inputs_microbatch, labels_microbatch, return_token_entropy=True,
        )
        log_probs_microbatch = policy_outputs["log_probs"]
        token_entropy = policy_outputs.pop("token_entropy").detach()
        # compute policy loss
        advantages_microbatch = normalized_rewards[indices].to(device)
        old_log_probs_microbatch = None
        if importance_reweighting_method != "none":
            old_log_probs_microbatch = old_log_probs.index_select(
                0, indices.to(old_log_probs.device),
            ).to(device)
        per_token_gradient_loss, policy_metadata = compute_policy_gradient_loss(
            advantages_microbatch, log_probs_microbatch,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs_microbatch, cliprange=cliprange,
            response_mask=mask,
        )
        # Logging is independent of the loss normalization/reweighting variant.
        with torch.no_grad():
            count = len(inputs_microbatch)
            accumulate_means("policy", policy_metadata, count)
            entropy_sum += token_entropy.masked_select(mask).sum(dtype=torch.float32)
            num_entropy_tokens += mask.sum()
            num_microbatches += 1
        loss = aggregate_loss_across_microbatch(per_token_gradient_loss, mask, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            # Convert the microbatch mean to its contribution to the ORIGINAL
            # batch mean, including a possibly shorter final microbatch.
            loss = loss * (len(indices) / input_len)
        # Constant mode already divides each partial sum by the full-batch Z.
        # Backward pass
        loss.backward()
        returned_loss += loss.detach()
    if num_retained == 0:
        # Match an unpruned zero-gradient step, including Adam momentum/decay.
        # Skipping optimizer.step() or leaving grads None would change that update.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
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
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        metadata = {key: total / metric_counts[key] for key, total in metric_totals.items()}
        if importance_reweighting_method in ("grpo", "gspo"):
            # An entirely pruned batch has no observed clipping decisions.
            metadata.setdefault("policy/clip_count", torch.tensor(0., device=device))
            metadata.setdefault("policy/clip_total", torch.tensor(0., device=device))
            metadata["policy/clip_fraction"] = metadata["policy/clip_count"] / metadata["policy/clip_total"].clamp_min(1)
        all_rewards = raw_rewards.detach()
        all_advantages = normalized_rewards.detach()
        num_response_tokens = response_mask.sum()
        metadata.update({
            "advantage/std_reward": all_rewards.std(correction=0),
            "advantage/min_reward": all_rewards.min(),
            "advantage/max_reward": all_rewards.max(),
            "advantage/std_advantage": all_advantages.std(correction=0),
            "policy/std_weight": all_advantages.std(correction=0),
            "policy/mean_weight": all_advantages.mean(),
            "policy/fraction_positive_weights": (all_advantages > 0).float().mean(),
            "policy/fraction_negative_weights": (all_advantages < 0).float().mean(),
            "policy/fraction_zero_weights": (all_advantages == 0).float().mean(),
            "train/loss": returned_loss.detach().clone(),
            "train/grad_norm_before_clip": grad_norm,
            "train/token_entropy": entropy_sum / num_entropy_tokens.clamp_min(1),
            "train/num_entropy_tokens": num_entropy_tokens,
            "train/num_response_tokens": num_response_tokens,
            "train/mean_response_length": num_response_tokens / input_len,
            "train/num_sequences": float(input_len),
            "train/num_microbatches": float(num_microbatches),
            "train/num_retained_sequences": float(num_retained),
            "train/fraction_pruned_sequences": (input_len - num_retained) / input_len,
        })
    return returned_loss, metadata
