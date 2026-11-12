from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    boundaries = np.linspace(0, 1, bins + 1)
    score = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        member = (confidence > lower) & (confidence <= upper)
        if member.any():
            score += member.mean() * abs(correct[member].mean() - confidence[member].mean())
    return float(score)


def classification_metrics(y_true, probabilities, label_names: list[str]) -> dict[str, object]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    prediction = probabilities.argmax(axis=1)
    labels = np.arange(len(label_names))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, prediction, labels=labels, zero_division=0)
    binary = np.eye(len(label_names), dtype=int)[y_true]
    per_class_ap = {}
    for index, name in enumerate(label_names):
        per_class_ap[name] = float(average_precision_score(binary[:, index], probabilities[:, index])) if binary[:, index].sum() else float("nan")
    one_hot = np.eye(len(label_names))[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "negative_log_likelihood": float(log_loss(y_true, probabilities, labels=labels)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(y_true, probabilities),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]), "auprc": per_class_ap[name]}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
        "classification_report": classification_report(y_true, prediction, labels=labels, target_names=label_names, zero_division=0, output_dict=True),
    }
