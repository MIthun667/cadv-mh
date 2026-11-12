from __future__ import annotations


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


def budget_counts(eligible_counts: torch.Tensor, ratio: float) -> torch.Tensor:
    """Number of tokens to select per row: min(max(1, ceil(ratio*n)), n-1), and 0 when n <= 1.

    The 1e-6 offset is a defensive guard against float error (no misrounding was observed for common ratios).
    """
    n = eligible_counts.to(torch.float64)
    k = torch.ceil(float(ratio) * n - 1e-6).clamp(min=1)
    k = torch.minimum(k, n - 1).clamp(min=0)
    return k.to(torch.long)


def budgeted_topk(scores: torch.Tensor, eligible_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Hard 0/1 mask with exactly budget_counts(...) ones per row, over eligible tokens only.

    Ranking is on raw scores (logits), not sigmoid(scores): sigmoid saturates to exactly 1.0 in
    float32 and would turn ranking into index-order tie-breaking. Remaining exact ties resolve
    by earliest index (stable sort). No gradient flows through this function.
    """
    eligible = eligible_mask.bool()
    scores = scores.detach().masked_fill(~eligible, float("-inf"))
    order = scores.argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    positions = torch.arange(scores.size(1), device=scores.device).unsqueeze(0).expand_as(order)
    rank.scatter_(1, order, positions)
    k = budget_counts(eligible.sum(dim=1), ratio).unsqueeze(1)
    return ((rank < k) & eligible).to(scores.dtype)


def budgeted_straight_through(
    logits: torch.Tensor, eligible_mask: torch.Tensor, temperature: float, ratio: float, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training selection: forward = exact-budget hard top-k of perturbed logits, backward = binary-concrete.

    Returns (selection, soft). selection == hard in value and has d selection / d logits = d soft / d logits.
    Selecting every eligible token is impossible: sum(selection) = k <= n - 1.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    perturbed = logits + _stable_logistic_noise(logits, epsilon)
    hard = budgeted_topk(perturbed, eligible_mask, ratio)
    soft = torch.sigmoid(perturbed / temperature) * eligible_mask.to(logits.dtype)
    return hard.to(soft.dtype) + (soft - soft.detach()), soft  # parentheses: forward value is exactly `hard`


def deterministic_selection(
    logits: torch.Tensor,
    eligible_mask: torch.Tensor,
    mode: str = "top_rho",
    rho: float = 0.2,
    threshold: float = 0.5,
) -> torch.Tensor:
    if mode == "top_rho":
        return budgeted_topk(logits, eligible_mask, rho)
    if mode != "threshold":
        raise ValueError(f"Unknown deterministic selection mode: {mode}")
    eligible = eligible_mask.bool()
    probabilities = torch.sigmoid(logits)
    cap = budget_counts(eligible.sum(dim=1), 1.0).unsqueeze(1)  # n - 1 (at least one token left unmasked)
    candidate = torch.where(eligible & (probabilities >= threshold), logits, torch.full_like(logits, float("-inf")))
    order = candidate.argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(logits.size(1), device=logits.device).unsqueeze(0).expand_as(order))
    return ((rank < cap) & torch.isfinite(candidate)).to(logits.dtype)


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


def boundary_tie_fraction(scores: torch.Tensor, eligible_mask: torch.Tensor, ratio: float) -> float:
    """Fraction of rows where the k-th and (k+1)-th ranked eligible scores are exactly equal.

    A nonzero value means the selected set depends on tie-breaking order, not on the selector.
    """
    eligible = eligible_mask.bool()
    values = scores.detach().masked_fill(~eligible, float("-inf")).sort(dim=1, descending=True).values
    k = budget_counts(eligible.sum(dim=1), ratio)
    rows = torch.arange(scores.size(0), device=scores.device)
    n = eligible.sum(dim=1)
    valid = (k > 0) & (k < n)
    if not valid.any():
        return 0.0
    kth = values[rows[valid], k[valid] - 1]
    nxt = values[rows[valid], k[valid]]
    return float((kth == nxt).float().mean())
