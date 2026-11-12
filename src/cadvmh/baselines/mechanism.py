from __future__ import annotations

import torch


def random_fixed_budget_mask(eligible: torch.Tensor, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    mask = torch.zeros_like(eligible, dtype=torch.float32)
    for row in range(eligible.size(0)):
        indices = torch.where(eligible[row])[0]
        count = min(max(0, indices.numel() - 1), max(1, round(ratio * indices.numel()))) if indices.numel() else 0
        if count:
            chosen = indices[torch.randperm(indices.numel(), generator=generator, device=indices.device)[:count]]
            mask[row, chosen] = 1
    return mask
