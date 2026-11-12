import torch


def test_stage_outputs_and_shared_encoder(tiny_model, tiny_batch):
    output = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=3)
    assert output.final_logits.shape == (2, 5)
    assert output.original_logits.shape == (2, 5)
    assert output.counterfactual_logits.shape == (2, 5)
    assert output.evidence_logits.shape == (2, 5)
    assert output.selection_mask.shape == tiny_batch["input_ids"].shape
    assert output.effect_representation.shape == (2, 24)
    assert sum(1 for _ in tiny_model.modules() if type(_) is type(tiny_model.encoder.model)) == 1


def test_pooling_weights_are_normalized(tiny_model, tiny_batch):
    tiny_model.eval()
    output = tiny_model(**{k: tiny_batch[k] for k in ("input_ids", "attention_mask", "special_token_mask")}, mask_token_id=4, stage=1)
    assert torch.allclose(output.pooling_attention.sum(1), torch.ones(2), atol=1e-6)
    assert torch.equal(output.pooling_attention[tiny_batch["attention_mask"] == 0], torch.zeros(2))


def test_input_ids_and_unmodified_embeddings_match(tiny_model, tiny_batch):
    tiny_model.eval()
    with torch.no_grad():
        ids_hidden = tiny_model.encoder(input_ids=tiny_batch["input_ids"], attention_mask=tiny_batch["attention_mask"])
        embeddings = tiny_model.encoder.word_embeddings(tiny_batch["input_ids"])
        positions = tiny_model.encoder.position_ids_from_input_ids(tiny_batch["input_ids"])
        embed_hidden = tiny_model.encoder(inputs_embeds=embeddings, attention_mask=tiny_batch["attention_mask"], position_ids=positions)
    assert torch.allclose(ids_hidden, embed_hidden, atol=1e-5, rtol=1e-5)
