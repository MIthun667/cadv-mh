from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_logistic(seed: int = 42, max_features: int = 100_000) -> Pipeline:
    if seed != 42:
        raise ValueError("CADV-MH baseline training is restricted to seed 42")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def build_tfidf_svm(max_features: int = 100_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)),
        ("classifier", LinearSVC(class_weight="balanced")),
    ])
