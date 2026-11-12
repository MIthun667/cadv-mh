import numpy as np

from cadvmh.evaluation.metrics import classification_metrics
from cadvmh.evaluation.faithfulness import masked_rationale_text
from cadvmh.evaluation.statistics import paired_group_bootstrap


def test_metrics_and_group_bootstrap():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    metrics = classification_metrics(y, probabilities, ["a", "b"])
    assert metrics["macro_f1"] == 1.0
    comparison = paired_group_bootstrap(
        y, y, np.array([1, 1, 0, 0]), np.array([0, 1, 2, 3]), resamples=20
    )
    assert comparison["mean_difference"] > 0


def test_rationale_text_views_preserve_selected_and_complement_tokens():
    text = "I feel very anxious"
    tokens = [
        {"start": 0, "end": 1, "selected": False},
        {"start": 2, "end": 6, "selected": False},
        {"start": 7, "end": 11, "selected": False},
        {"start": 12, "end": 19, "selected": True},
    ]
    assert (
        masked_rationale_text(text, tokens, keep_selected=True)
        == "[MASK] [MASK] [MASK] anxious"
    )
    assert (
        masked_rationale_text(text, tokens, keep_selected=False) == "I feel very [MASK]"
    )
