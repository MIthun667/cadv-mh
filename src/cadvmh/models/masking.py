from __future__ import annotations

import math

import torch


def _stable_logistic_noise(reference: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    first = torch.rand_like(reference).clamp(epsilon, 1 - epsilon)
    second = torch.rand_like(reference).clamp(epsilon, 1 - epsilon)
    return (-torch.log(-torch.log(first))) - (-torch.log(-torch.log(second)))


def binary_concrete(
    logits: torch.Tensor,
    eligible_mask: torch.Tensor,
    temperature: float,
    hard: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft = torch.sigmoid((logits + _stable_logistic_noise(logits, epsilon)) / temperature)
    soft = soft * eligible_mask.to(soft.dtype)
    if not hard:
        return soft
    discrete = (soft >= 0.5).to(soft.dtype)
    return (discrete - soft).detach() + soft


def deterministic_selection(
    logits: torch.Tensor,
    eligible_mask: torch.Tensor,
    mode: str = "top_rho",
    rho: float = 0.2,
    threshold: float = 0.5,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    selected = torch.zeros_like(probabilities)
    for row in range(logits.size(0)):
        eligible = torch.where(eligible_mask[row])[0]
        if eligible.numel() == 0:
            continue
        max_selected = max(0, eligible.numel() - 1) if eligible.numel() > 1 else 0
        if mode == "top_rho":
            count = min(max_selected, max(1, math.ceil(float(rho) * eligible.numel())))
            if count:
                indices = eligible[torch.topk(probabilities[row, eligible], count).indices]
                selected[row, indices] = 1
        elif mode == "threshold":
            indices = eligible[probabilities[row, eligible] >= threshold]
            if indices.numel() > max_selected:
                indices = indices[torch.topk(probabilities[row, indices], max_selected).indices] if max_selected else indices[:0]
            selected[row, indices] = 1
        else:
            raise ValueError(f"Unknown deterministic selection mode: {mode}")
    return selected


def construct_masked_embeddings(
    original_embeddings: torch.Tensor,
    selection_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    mask_embedding: torch.Tensor,
    evidence_only: bool = False,
) -> torch.Tensor:
    selector = selection_mask.unsqueeze(-1).to(original_embeddings.dtype)
    eligible = eligible_mask.unsqueeze(-1).to(original_embeddings.dtype)
    mask_embedding = mask_embedding.view(1, 1, -1).to(original_embeddings.dtype)
    if evidence_only:
        mixed = selector * original_embeddings + (1 - selector) * mask_embedding
    else:
        mixed = (1 - selector) * original_embeddings + selector * mask_embedding
    return eligible * mixed + (1 - eligible) * original_embeddings
