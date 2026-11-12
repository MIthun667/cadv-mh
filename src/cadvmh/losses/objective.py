from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cadvmh.models.cadvmh import CADVMHOutput

from .classification import weighted_cross_entropy
from .consistency import teacher_student_kl
from .necessity import necessity_loss
from .regularization import continuity_loss, sparsity_loss
from .sufficiency import sufficiency_loss


@dataclass
class LossWeights:
    classification: float = 1.0
    original: float = 0.5
    necessity: float = 0.5
    sufficiency: float = 0.5
    sparsity: float = 0.05
    continuity: float = 0.05
    consistency: float = 0.1


class CADVMHLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights | None = None,
        class_weights: torch.Tensor | None = None,
        necessity_margin: float = 0.2,
        selection_ratio: float = 0.2,
    ):
        super().__init__()
        self.weights = weights or LossWeights()
        self.register_buffer("class_weights", class_weights)
        self.necessity_margin = necessity_margin
        self.selection_ratio = selection_ratio

    def forward(
        self,
        output: CADVMHOutput,
        labels: torch.Tensor,
        augmented_logits: torch.Tensor | None = None,
        stage: int = 3,
    ) -> dict[str, torch.Tensor]:
        zero = output.original_logits.sum() * 0
        components = {
            "classification": weighted_cross_entropy(output.final_logits, labels, self.class_weights),
            "original": weighted_cross_entropy(output.original_logits, labels, self.class_weights),
            "necessity": zero,
            "sufficiency": zero,
            "sparsity": zero,
            "continuity": zero,
            "consistency": zero,
        }
        if stage > 1:
            if self.weights.necessity and output.counterfactual_logits is not None:
                components["necessity"] = necessity_loss(
                    output.original_logits, output.counterfactual_logits, labels, self.necessity_margin, True
                )
            if self.weights.sufficiency and output.evidence_logits is not None:
                components["sufficiency"] = sufficiency_loss(output.evidence_logits, labels, self.class_weights)
            if output.selection_mask is not None and output.valid_token_mask is not None:
                if self.weights.sparsity:
                    components["sparsity"] = sparsity_loss(output.selection_mask, output.valid_token_mask, self.selection_ratio)
                if self.weights.continuity:
                    components["continuity"] = continuity_loss(output.selection_mask, output.valid_token_mask)
            if self.weights.consistency and augmented_logits is not None:
                components["consistency"] = teacher_student_kl(output.original_logits, augmented_logits)
        if stage == 1:
            total = components["original"]
        elif stage == 2:
            total = components["original"]
            for name in ("necessity", "sufficiency", "sparsity", "continuity"):
                total = total + getattr(self.weights, name) * components[name]
        else:
            total = sum(getattr(self.weights, name) * value for name, value in components.items())
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite CADV-MH loss")
        return {"total": total, **components}
