import torch

from cadvmh.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_predictions(tmp_path, tiny_model, tiny_batch):
    tiny_model.eval()
    args = {k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}
    before = tiny_model(**args, mask_token_id=4, stage=3, selection_mode="top_rho").final_logits.detach()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, tiny_model, stage=3, epoch=1, config={"x": 1}, label_mapping={"a": 0}, split_fingerprint="abc")
    for parameter in tiny_model.parameters():
        parameter.data.add_(1)
    payload = load_checkpoint(path, tiny_model, restore_rng=False)
    tiny_model.eval()
    after = tiny_model(**args, mask_token_id=4, stage=3, selection_mode="top_rho").final_logits.detach()
    assert torch.allclose(before, after)
    assert payload["label_mapping"] == {"a": 0}
