from __future__ import annotations

import torch
from torch import nn


class ContextEffectInteraction(nn.Module):
    def __init__(self, hidden_size: int, interaction_size: int | None = None, dropout: float = 0.1):
        super().__init__()
        interaction_size = interaction_size or hidden_size
        self.original = nn.Linear(hidden_size, interaction_size, bias=False)
        self.counterfactual = nn.Linear(hidden_size, interaction_size, bias=False)
        self.delta = nn.Linear(hidden_size, interaction_size, bias=False)
        self.product = nn.Linear(hidden_size, interaction_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(interaction_size))
        self.comparison = nn.Linear(5 * hidden_size, interaction_size)
        self.output = nn.Linear(interaction_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, original: torch.Tensor, counterfactual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = original - counterfactual
        comparison = torch.cat([original, counterfactual, delta, delta.abs(), original * counterfactual], dim=-1)
        interaction = torch.tanh(
            self.original(original)
            + self.counterfactual(counterfactual)
            + self.delta(delta)
            + self.product(original * counterfactual)
            + self.bias
        )
        effect = interaction + self.comparison(comparison)
        fused = self.norm(original + self.output(self.dropout(effect)))
        return fused, effect
