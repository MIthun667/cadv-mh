from __future__ import annotations

import torch
import torch.nn.functional as F


def necessity_loss(
    original_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    detach_original: bool = True,
) -> torch.Tensor:
    original = F.softmax(original_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    counterfactual = F.softmax(counterfactual_logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    if detach_original:
        original = original.detach()
    return F.relu(float(margin) - original + counterfactual).mean()
