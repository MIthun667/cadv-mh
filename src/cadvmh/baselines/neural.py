from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output, _ = self.lstm(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


class CNNBiLSTMClassifier(BiLSTMClassifier):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int
    ):
        super().__init__(vocab_size, embedding_dim, hidden_size, num_classes)
        self.convolution = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size=3, padding=1
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        convolved = torch.relu(self.convolution(embedded.transpose(1, 2))).transpose(
            1, 2
        )
        output, _ = self.lstm(convolved)
        mask = attention_mask.unsqueeze(-1).to(output.dtype)
        return self.classifier((output * mask).sum(1) / mask.sum(1).clamp_min(1))


class SparseRationalizer(nn.Module):
    """Generator-predictor baseline with straight-through sparse token masks."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_size: int, num_classes: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.generator = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.selector = nn.Linear(2 * hidden_size, 1)
        self.predictor = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def _predict(self, embedded, selection):
        selected = embedded * selection.unsqueeze(-1)
        predicted, _ = self.predictor(selected)
        denominator = selection.sum(1, keepdim=True).clamp_min(1.0)
        pooled = (predicted * selection.unsqueeze(-1)).sum(1) / denominator
        return self.classifier(pooled)

    def select(self, input_ids, attention_mask, *, hard: bool = False):
        embedded = self.embedding(input_ids)
        generated, _ = self.generator(embedded)
        probabilities = (
            torch.sigmoid(self.selector(generated).squeeze(-1)) * attention_mask
        )
        if hard:
            selection = (probabilities >= 0.5).to(probabilities.dtype)
        else:
            binary = (probabilities >= 0.5).to(probabilities.dtype)
            selection = binary.detach() - probabilities.detach() + probabilities
        return embedded, selection, probabilities

    def forward(self, input_ids, attention_mask, *, hard: bool = False):
        embedded, selection, probabilities = self.select(
            input_ids, attention_mask, hard=hard
        )
        return self._predict(embedded, selection), probabilities

    def rationale_views(self, input_ids, attention_mask):
        embedded, selection, probabilities = self.select(
            input_ids, attention_mask, hard=True
        )
        eligible = attention_mask.to(embedded.dtype)
        return {
            "original_logits": self._predict(embedded, eligible),
            "evidence_logits": self._predict(embedded, selection),
            "counterfactual_logits": self._predict(embedded, eligible - selection),
            "selection": selection,
            "selector_probabilities": probabilities,
        }
