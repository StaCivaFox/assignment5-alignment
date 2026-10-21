import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    # Tokenize prompt and respond seperately, then concatenate
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False, padding=False, truncation=False)["input_ids"] # list[list[int]]
    response_ids = tokenizer(output_strs, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    combined = [
        prompt + response
        for prompt, response in zip(prompt_ids, response_ids, strict=True)
    ]
    resp_start_end_index = [
        (len(prompt) - 1, len(prompt) + len(response) - 2)
        for prompt, response in zip(prompt_ids, response_ids, strict=True)
    ] # list[tuple(int, int)]; marks shifted index in labels
    # Pad combined so that it can be used to create a tensor
    pad_id = tokenizer.pad_token_id
    lengths = [len(row) for row in combined]
    max_len = max(lengths)
    combined = [
        row + [pad_id] * (max_len - len(row))
        for row in combined
    ]
    batch = torch.tensor(combined, dtype=torch.long) # (batch_size, max_len)
    input_ids = batch[:, :-1]
    labels = batch[:, 1:] # (batch_size, max_len - 1)
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (start, end) in enumerate(resp_start_end_index):
        response_mask[i, start:end + 1] = True
    res = {}
    res["input_ids"] = input_ids
    res["labels"] = labels
    res["response_mask"] = response_mask
    return res

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    # Make one forward pass through model
    logits = model(input_ids).logits # (B, L, V)
    # Convert scores to log probabilities
    all_log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (B, L, V)
    res = {}
    if return_token_entropy:
        # Entropy is a logging metric, so avoid retaining another gradient graph.
        with torch.no_grad():
            probs = torch.exp(all_log_probs)
            entropy = -(probs * all_log_probs).sum(dim=-1)
        res["token_entropy"] = entropy
    # Specifically select log probs corresponding to labels token ids for each sequence
    log_probs = torch.gather(all_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B, L)
    res["log_probs"] = log_probs
    return res

def gradient_clipping(params, max_norm):
    eps = 1e-6
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        norm += torch.square(g).sum()
    norm = torch.sqrt(norm)
    clip_factor = min(max_norm / (norm + eps), 1)
    for g in grads:
        g *= clip_factor

    




