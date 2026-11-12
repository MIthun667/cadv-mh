from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def effective_number_weights(counts: list[int] | np.ndarray, beta: float = 0.999) -> torch.Tensor:
    values = np.asarray(counts, dtype=np.float64)
    if (values <= 0).any():
        raise ValueError("Every configured class must be present in training data")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if beta == 0:
        weights = np.ones_like(values)
    else:
        weights = (1 - beta) / -np.expm1(values * np.log(beta))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    weight = class_weights.to(logits.device, logits.dtype) if class_weights is not None else None
    return F.cross_entropy(logits, labels, weight=weight)
