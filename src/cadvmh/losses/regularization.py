from __future__ import annotations

import torch


def sparsity_loss(selection: torch.Tensor, eligible: torch.Tensor, target_ratio: float = 0.2) -> torch.Tensor:
    eligible_f = eligible.to(selection.dtype)
    ratios = (selection * eligible_f).sum(dim=1) / eligible_f.sum(dim=1).clamp_min(1)
    return (ratios - float(target_ratio)).abs().mean()


def continuity_loss(selection: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    adjacent = eligible[:, 1:] & eligible[:, :-1]
    differences = (selection[:, 1:] - selection[:, :-1]).abs()
    numerator = (differences * adjacent.to(differences.dtype)).sum(dim=1)
    denominator = adjacent.sum(dim=1)
    per_example = torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.zeros_like(numerator))
    return per_example.mean()


def selection_diagnostics(selection: torch.Tensor, eligible: torch.Tensor) -> dict[str, float]:
    counts = (selection * eligible.to(selection.dtype)).sum(dim=1)
    totals = eligible.sum(dim=1)
    ratios = counts / totals.clamp_min(1)
    return {
        "selection_ratio_mean": float(ratios.mean().detach()),
        "selection_ratio_std": float(ratios.std(unbiased=False).detach()),
        "selection_count_min": float(counts.min().detach()),
        "selection_count_max": float(counts.max().detach()),
        "empty_selection_fraction": float((counts <= 0).float().mean().detach()),
        "full_selection_fraction": float((counts >= totals).float().mean().detach()),
    }
