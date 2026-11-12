import torch

from cadvmh.data.tokenization import special_token_mask


class DummyTokenizer:
    def get_special_tokens_mask(self, row, already_has_special_tokens=True):
        return [int(token in {0, 1, 2}) for token in row]


def test_special_and_padding_mask():
    ids = torch.tensor([[0, 4, 2, 1]])
    attention = torch.tensor([[1, 1, 1, 0]])
    mask = special_token_mask(ids, DummyTokenizer(), attention)
    assert mask.tolist() == [[True, False, True, True]]
