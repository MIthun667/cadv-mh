import torch

from cadvmh.models.masking import binary_concrete, construct_masked_embeddings, deterministic_selection


def test_special_tokens_never_selected():
    logits = torch.full((1, 5), 20.0)
    eligible = torch.tensor([[False, True, True, False, False]])
    soft = binary_concrete(logits, eligible, 1.0)
    assert torch.equal(soft[~eligible], torch.zeros_like(soft[~eligible]))
    hard = deterministic_selection(logits, eligible, rho=1.0)
    assert hard.sum() == 1  # at least one content token remains


def test_counterfactual_and_evidence_embeddings():
    original = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    selection = torch.tensor([[0.0, 1.0, 0.0]])
    eligible = torch.tensor([[False, True, False]])
    mask = torch.full((4,), -1.0)
    cf = construct_masked_embeddings(original, selection, eligible, mask, False)
    ev = construct_masked_embeddings(original, selection, eligible, mask, True)
    assert torch.equal(cf[:, 1], mask.view(1, -1))
    assert torch.equal(ev[:, 1], original[:, 1])
    assert torch.equal(cf[:, 0], original[:, 0])
