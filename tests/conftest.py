import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from cadvmh.models import CADVMH


@pytest.fixture
def tiny_model():
    config = RobertaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return CADVMH(num_classes=5, encoder=RobertaModel(config), selector_hidden_dim=16, interaction_dim=24, dropout=0.0)


@pytest.fixture
def tiny_batch():
    ids = torch.tensor([[0, 5, 6, 7, 2, 1], [0, 8, 9, 10, 2, 1]])
    attention = ids.ne(1).long()
    special = ids.eq(0) | ids.eq(1) | ids.eq(2)
    return {"input_ids": ids, "attention_mask": attention, "special_token_mask": special, "labels": torch.tensor([0, 3])}
