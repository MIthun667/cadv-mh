from __future__ import annotations

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=dim)
    probabilities = probabilities * mask.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, include_special_tokens: bool = True):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.scorer = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.include_special_tokens = include_special_tokens

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor, special_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        valid = attention_mask.bool()
        if not self.include_special_tokens and special_mask is not None:
            valid = valid & ~special_mask.bool()
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        scores = self.scorer(torch.tanh(self.projection(self.dropout(hidden)))).squeeze(-1)
        weights = masked_softmax(scores, valid)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights
