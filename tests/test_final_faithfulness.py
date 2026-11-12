import torch

from cadvmh.evaluation.faithfulness import final_mechanism_faithfulness
from cadvmh.models.masking import budgeted_topk

KEYS = ("input_ids", "attention_mask", "special_token_mask")


def _batch(tiny_batch):
    return {k: tiny_batch[k] for k in (*KEYS, "labels")}


def test_final_faithfulness_reruns_the_whole_model_on_edited_ids(tiny_model, tiny_batch):
    tiny_model.eval()
    ratio = 0.5
    result = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, ratio, generator=torch.Generator().manual_seed(0))
    # manual recomputation with the deployed (final) head
    ids = tiny_batch["input_ids"]
    ref = tiny_model(**{k: tiny_batch[k] for k in KEYS}, mask_token_id=4, stage=3, selection_mode="top_rho", selection_ratio=ratio)
    eligible = ref.valid_token_mask.bool()
    chosen = ref.selection_mask.bool()
    edited = torch.where(chosen, torch.full_like(ids, 4), ids)
    cf = tiny_model(edited, tiny_batch["attention_mask"], tiny_batch["special_token_mask"], 4, stage=3, selection_mode="top_rho", selection_ratio=ratio)
    p0, p1 = torch.softmax(ref.final_logits, -1), torch.softmax(cf.final_logits, -1)
    yhat = p0.argmax(-1)
    expected = (p0.gather(1, yhat[:, None]) - p1.gather(1, yhat[:, None])).squeeze(1)
    assert torch.allclose(result["predicted_comprehensiveness"], expected, atol=1e-6)
    assert torch.allclose(result["selection_ratio"], chosen.sum(1) / eligible.sum(1))
    for key in ("predicted_sufficiency", "gold_comprehensiveness", "gold_sufficiency",
                "predicted_random_comprehensiveness", "predicted_random_sufficiency"):
        assert result[key].shape == (2,) and (result[key].abs() <= 1).all()


def test_final_faithfulness_uses_final_head_not_auxiliary_head(tiny_model, tiny_batch):
    tiny_model.eval()
    kwargs = dict(generator=torch.Generator().manual_seed(0))
    # Non-degenerate final head regardless of the random tiny-model init (the fixture is unseeded).
    torch.manual_seed(0)
    with torch.no_grad():
        for p in tiny_model.final_head.parameters():
            p.normal_(0, 2.0)
    before = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, 0.5, **kwargs)["predicted_comprehensiveness"]
    with torch.no_grad():  # perturbing only the auxiliary head must not change a final-mechanism metric
        for p in tiny_model.auxiliary_head.parameters():
            p.add_(5.0)
    after = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, 0.5, **kwargs)["predicted_comprehensiveness"]
    assert torch.allclose(before, after)
    with torch.no_grad():
        for p in tiny_model.final_head.parameters():
            p.normal_(0, 2.0)
    changed = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, 0.5, **kwargs)["predicted_comprehensiveness"]
    assert not torch.allclose(before, changed)


def test_random_control_same_budget_deterministic_and_training_mode_restored(tiny_model, tiny_batch):
    tiny_model.train()
    a = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, 0.5, generator=torch.Generator().manual_seed(3))
    assert tiny_model.training
    tiny_model.eval()
    b = final_mechanism_faithfulness(tiny_model, _batch(tiny_batch), 4, 0.5, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(a["predicted_random_comprehensiveness"], b["predicted_random_comprehensiveness"], atol=1e-5)
    eligible = tiny_batch["attention_mask"].bool() & ~tiny_batch["special_token_mask"].bool()
    scores = torch.rand(2, 6, generator=torch.Generator().manual_seed(3))
    assert budgeted_topk(scores, eligible, 0.5).sum() == b["selection_ratio"].mul(eligible.sum(1)).sum().round()
