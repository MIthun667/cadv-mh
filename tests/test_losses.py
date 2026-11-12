import torch

from cadvmh.losses.classification import effective_number_weights
from cadvmh.losses.necessity import necessity_loss
from cadvmh.losses.regularization import continuity_loss, sparsity_loss


def test_effective_weights_are_finite_and_normalized():
    weights = effective_number_weights([10, 100, 1000], 0.999)
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_necessity_margin_behavior():
    labels = torch.tensor([0])
    good_original = torch.tensor([[5.0, 0.0]])
    low_counterfactual = torch.tensor([[0.0, 5.0]])
    assert necessity_loss(good_original, low_counterfactual, labels, margin=0.2) == 0
    assert necessity_loss(low_counterfactual, good_original, labels, margin=0.2) > 0


def test_regularization_ignores_padding_and_special_boundaries():
    selection = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    eligible = torch.tensor([[False, True, False, False]])
    assert continuity_loss(selection, eligible) == 0
    assert torch.isclose(sparsity_loss(selection, eligible, 1.0), torch.tensor(0.0))
