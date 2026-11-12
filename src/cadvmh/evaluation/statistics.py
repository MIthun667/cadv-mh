from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def paired_group_bootstrap(
    y_true,
    prediction_a,
    prediction_b,
    groups,
    resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    if seed != 42:
        raise ValueError("CADV-MH statistical evaluation is restricted to seed 42")
    y_true, prediction_a, prediction_b, groups = map(np.asarray, (y_true, prediction_a, prediction_b, groups))
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.where(groups == group)[0] for group in sampled])
        a = f1_score(y_true[indices], prediction_a[indices], average="macro", zero_division=0)
        b = f1_score(y_true[indices], prediction_b[indices], average="macro", zero_division=0)
        differences.append(a - b)
    alpha = (1 - confidence) / 2
    return {
        "mean_difference": float(np.mean(differences)),
        "lower": float(np.quantile(differences, alpha)),
        "upper": float(np.quantile(differences, 1 - alpha)),
        "resamples": resamples,
    }
