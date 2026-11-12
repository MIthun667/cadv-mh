from __future__ import annotations

import torch

from .classification import weighted_cross_entropy


def sufficiency_loss(evidence_logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_cross_entropy(evidence_logits, labels, class_weights)
