from __future__ import annotations

from torch import nn
from transformers import AutoModelForSequenceClassification


def build_transformer_classifier(checkpoint: str, num_classes: int) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes)
