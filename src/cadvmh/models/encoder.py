from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class SharedEncoder(nn.Module):
    def __init__(self, name: str | None = None, encoder: nn.Module | None = None, gradient_checkpointing: bool = False):
        super().__init__()
        if encoder is None and name is None:
            raise ValueError("Either an encoder name or an encoder instance is required")
        self.model = encoder if encoder is not None else AutoModel.from_pretrained(name)
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def mask_embedding(self, mask_token_id: int) -> torch.Tensor:
        weight = self.model.get_input_embeddings().weight
        return weight[mask_token_id]

    def position_ids_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        padding_idx = getattr(self.model.config, "pad_token_id", None)
        if getattr(self.model.config, "model_type", "") == "roberta" and padding_idx is not None:
            mask = input_ids.ne(padding_idx).int()
            return torch.cumsum(mask, dim=1).to(input_ids.dtype) * mask + padding_idx
        return torch.arange(input_ids.size(1), device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0).expand_as(input_ids)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)
        return output.last_hidden_state

    def config_dict(self) -> dict[str, Any]:
        return self.model.config.to_dict()
