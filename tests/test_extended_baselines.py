import torch

from cadvmh.baselines.neural import BiLSTMClassifier, SparseRationalizer


def test_bilstm_baseline_shapes():
    model = BiLSTMClassifier(
        vocab_size=20, embedding_dim=8, hidden_size=4, num_classes=3
    )
    inputs = torch.tensor([[2, 3, 0], [4, 5, 6]])
    mask = inputs.ne(0).float()
    assert model(inputs, mask).shape == (2, 3)


def test_sparse_rationalizer_shapes_and_padding_mask():
    model = SparseRationalizer(
        vocab_size=20, embedding_dim=8, hidden_size=4, num_classes=3
    )
    inputs = torch.tensor([[2, 3, 0], [4, 5, 6]])
    mask = inputs.ne(0).float()
    logits, selection = model(inputs, mask)
    assert logits.shape == (2, 3)
    assert selection.shape == inputs.shape
    assert selection[0, 2].item() == 0

    views = model.rationale_views(inputs, mask)
    assert views["original_logits"].shape == (2, 3)
    assert views["evidence_logits"].shape == (2, 3)
    assert views["counterfactual_logits"].shape == (2, 3)
    assert torch.equal(views["selection"] * (1 - mask), torch.zeros_like(mask))
