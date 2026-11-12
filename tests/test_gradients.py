import torch

from cadvmh.losses.objective import CADVMHLoss
from cadvmh.training.stages import configure_stage


def test_stage2_selector_receives_mask_gradient(tiny_model, tiny_batch):
    configure_stage(tiny_model, 2)
    tiny_model.train()
    output = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=2)
    losses = CADVMHLoss()(output, tiny_batch["labels"], stage=2)
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.selector.parameters())
    assert all(p.grad is None for p in tiny_model.encoder.parameters())


def test_stage3_interaction_and_encoder_receive_gradients(tiny_model, tiny_batch):
    configure_stage(tiny_model, 3)
    output = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=3)
    CADVMHLoss()(output, tiny_batch["labels"], stage=3)["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.interaction.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.encoder.parameters())
