import torch

from cadvmh.models.selector import ContextConditionedSelector


def test_selector_shape_and_context_dependence():
    selector = ContextConditionedSelector(8, 4, 0.0)
    tokens = torch.randn(2, 5, 8)
    pooled = torch.randn(2, 8)
    assert selector(tokens, pooled).shape == (2, 5)
