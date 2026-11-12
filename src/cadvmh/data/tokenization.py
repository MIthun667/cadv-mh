from __future__ import annotations

from typing import Any

import torch


def special_token_mask(input_ids: torch.Tensor, tokenizer: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    masks = []
    for row in input_ids.detach().cpu().tolist():
        masks.append(tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
    special = torch.tensor(masks, dtype=torch.bool, device=input_ids.device)
    return special | ~attention_mask.bool()


def tokenize_batch(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=getattr(tokenizer, "is_fast", False),
    )
