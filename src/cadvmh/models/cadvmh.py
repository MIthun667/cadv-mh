from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoder import SharedEncoder
from .heads import ClassificationHead
from .interaction import ContextEffectInteraction
from .masking import binary_concrete, budgeted_straight_through, construct_masked_embeddings, deterministic_selection
from .pooling import AttentionPooling
from .selector import ContextConditionedSelector


@dataclass
class CADVMHOutput:
    final_logits: torch.Tensor
    original_logits: torch.Tensor
    counterfactual_logits: torch.Tensor | None = None
    evidence_logits: torch.Tensor | None = None
    original_representation: torch.Tensor | None = None
    counterfactual_representation: torch.Tensor | None = None
    effect_representation: torch.Tensor | None = None
    selector_logits: torch.Tensor | None = None
    selector_probabilities: torch.Tensor | None = None
    selection_mask: torch.Tensor | None = None
    soft_selection: torch.Tensor | None = None
    budget_enforced: bool = False  # selection has exactly the budgeted count by construction
    valid_token_mask: torch.Tensor | None = None
    pooling_attention: torch.Tensor | None = None


class CADVMH(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_name: str | None = None,
        encoder: nn.Module | None = None,
        selector_hidden_dim: int = 256,
        interaction_dim: int | None = None,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = SharedEncoder(encoder_name, encoder, gradient_checkpointing)
        hidden = self.encoder.hidden_size
        self.pooling = AttentionPooling(hidden, dropout)
        self.selector = ContextConditionedSelector(hidden, selector_hidden_dim, dropout)
        self.auxiliary_head = ClassificationHead(hidden, num_classes, dropout)
        self.interaction = ContextEffectInteraction(hidden, interaction_dim, dropout)
        self.final_head = ClassificationHead(hidden, num_classes, dropout)
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_token_mask: torch.Tensor,
        mask_token_id: int,
        *,
        stage: int = 3,
        temperature: float = 1.0,
        selection_mode: str = "budgeted_st",
        selection_ratio: float = 0.2,
        selection_threshold: float = 0.5,
        enable_evidence: bool = True,
        return_auxiliary: bool = True,
        fusion_mode: str = "full",
        selector_logits_override: torch.Tensor | None = None,
    ) -> CADVMHOutput:
        """selection_mode (training): "budgeted_st" (repaired default: exact-budget hard top-k forward,
        binary-concrete backward), or legacy "soft" / "straight_through". When the module is in eval mode
        (or selection_mode is "top_rho"/"threshold") deterministic selection is used.
        selector_logits_override replaces the learned selector scores (random-rationale controls)."""
        eligible = attention_mask.bool() & ~special_token_mask.bool()
        position_ids = self.encoder.position_ids_from_input_ids(input_ids)
        original_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        original, pool_attention = self.pooling(original_hidden, attention_mask)
        original_logits = self.auxiliary_head(original)
        if stage == 1:
            return CADVMHOutput(
                final_logits=original_logits,
                original_logits=original_logits,
                original_representation=original if return_auxiliary else None,
                valid_token_mask=eligible if return_auxiliary else None,
                pooling_attention=pool_attention if return_auxiliary else None,
            )

        # Selection (noise, ranking, ties) is done in float32: under fp16 autocast the logits would be half
        # precision, whose ~1e-3 resolution creates spurious ties and degrades the Gumbel noise near 1.
        selector_logits = self.selector(original_hidden, original).float()
        if selector_logits_override is not None:
            selector_logits = selector_logits_override.to(selector_logits.dtype)
        soft_selection = None
        budget_enforced = True
        if self.training and selection_mode in {"budgeted_st", "budgeted_st_sigmoid"}:
            # "budgeted_st_sigmoid" = surrogate of the first pilot (kept for provenance); default uses the boundary surrogate.
            selection, soft_selection = budgeted_straight_through(
                selector_logits, eligible, temperature, selection_ratio,
                surrogate="sigmoid" if selection_mode == "budgeted_st_sigmoid" else "boundary",
            )
        elif self.training and selection_mode in {"soft", "straight_through"}:
            selection = binary_concrete(selector_logits, eligible, temperature, hard=selection_mode == "straight_through")
            soft_selection = selection
            budget_enforced = False
        else:
            deterministic_mode = "threshold" if selection_mode == "threshold" else "top_rho"
            budget_enforced = deterministic_mode == "top_rho"
            selection = deterministic_selection(selector_logits, eligible, deterministic_mode, selection_ratio, selection_threshold)

        word_embeddings = self.encoder.word_embeddings(input_ids)
        mask_embedding = self.encoder.mask_embedding(mask_token_id)
        counterfactual_embeddings = construct_masked_embeddings(word_embeddings, selection, eligible, mask_embedding, False)
        counter_hidden = self.encoder(inputs_embeds=counterfactual_embeddings, attention_mask=attention_mask, position_ids=position_ids)
        counterfactual, _ = self.pooling(counter_hidden, attention_mask)
        counter_logits = self.auxiliary_head(counterfactual)

        evidence_logits = None
        if enable_evidence:
            evidence_embeddings = construct_masked_embeddings(word_embeddings, selection, eligible, mask_embedding, True)
            evidence_hidden = self.encoder(inputs_embeds=evidence_embeddings, attention_mask=attention_mask, position_ids=position_ids)
            evidence, _ = self.pooling(evidence_hidden, attention_mask)
            evidence_logits = self.auxiliary_head(evidence)

        if fusion_mode == "full":
            fused, effect = self.interaction(original, counterfactual)
            final_logits = self.final_head(fused)
        elif fusion_mode == "mean":
            fused, effect = (original + counterfactual) / 2, original - counterfactual
            final_logits = self.final_head(fused)
        elif fusion_mode == "difference":
            fused, effect = original - counterfactual, original - counterfactual
            final_logits = self.final_head(fused)
        elif fusion_mode == "evidence":
            if evidence_logits is None:
                raise ValueError("evidence fusion requires the evidence-only branch")
            fused, effect, final_logits = original, original - counterfactual, evidence_logits
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        return CADVMHOutput(
            final_logits=final_logits,
            original_logits=original_logits,
            counterfactual_logits=counter_logits,
            evidence_logits=evidence_logits,
            original_representation=original if return_auxiliary else None,
            counterfactual_representation=counterfactual if return_auxiliary else None,
            effect_representation=effect if return_auxiliary else None,
            selector_logits=selector_logits if return_auxiliary else None,
            selector_probabilities=torch.sigmoid(selector_logits) if return_auxiliary else None,
            selection_mask=selection if return_auxiliary else None,
            soft_selection=soft_selection if return_auxiliary else None,
            budget_enforced=budget_enforced,
            valid_token_mask=eligible if return_auxiliary else None,
            pooling_attention=pool_attention if return_auxiliary else None,
        )
