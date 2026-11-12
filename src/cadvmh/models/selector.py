from __future__ import annotations

import torch
from torch import nn


class ContextConditionedSelector(nn.Module):
    def __init__(self, hidden_size: int, selector_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * hidden_size, selector_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(selector_hidden, 1),
        )

    def forward(self, token_hidden: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        global_hidden = pooled.unsqueeze(1).expand_as(token_hidden)
        features = torch.cat(
            [token_hidden, global_hidden, token_hidden * global_hidden, (token_hidden - global_hidden).abs()], dim=-1
        )
        return self.network(features).squeeze(-1)
