from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .preprocessing import canonicalize_for_duplicates, stable_text_id


@dataclass(frozen=True)
class DuplicateAudit:
    rows: int
    exact_groups: int
    duplicate_rows: int
    conflicting_groups: int
    conflicting_rows: int


def assign_exact_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, DuplicateAudit]:
    result = frame.copy()
    result["canonical_text"] = result["text"].map(canonicalize_for_duplicates)
    result["group_id"] = result["canonical_text"].map(lambda text: "exact:" + stable_text_id(text)[:24])
    sizes = result.groupby("group_id").size()
    label_counts = result.groupby("group_id")["label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    audit = DuplicateAudit(
        rows=len(result),
        exact_groups=int(sizes.size),
        duplicate_rows=int((sizes - 1).clip(lower=0).sum()),
        conflicting_groups=len(conflicts),
        conflicting_rows=int(result["group_id"].isin(conflicts).sum()),
    )
    result["label_conflict"] = result["group_id"].isin(conflicts)
    return result, audit


def deduplicate_representatives(frame: pd.DataFrame, exclude_conflicts: bool = True) -> pd.DataFrame:
    source = frame.loc[~frame["label_conflict"]] if exclude_conflicts else frame
    return source.sort_values("sample_id").drop_duplicates("group_id", keep="first").reset_index(drop=True)


def word_shingle_jaccard(left: str, right: str, n: int = 3) -> float:
    def shingles(text: str) -> set[tuple[str, ...]]:
        tokens = text.split()
        if len(tokens) < n:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    a, b = shingles(left), shingles(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def audit_candidate_pairs(frame: pd.DataFrame, pairs: Iterable[tuple[int, int]], threshold: float = 0.92) -> list[dict[str, object]]:
    """Auditable near-duplicate verifier; candidate generation is intentionally external."""
    findings = []
    for left, right in pairs:
        score = word_shingle_jaccard(frame.iloc[left]["canonical_text"], frame.iloc[right]["canonical_text"])
        if score >= threshold:
            findings.append({"left": frame.iloc[left]["sample_id"], "right": frame.iloc[right]["sample_id"], "score": score})
    return findings


def find_near_duplicate_pairs(
    frame: pd.DataFrame,
    char_ngram: int = 5,
    threshold: float = 0.92,
    max_neighbors: int = 10,
    max_records: int = 100_000,
) -> list[dict[str, object]]:
    """Sparse character-ngram cosine candidate search with exact pair reporting."""
    if len(frame) > max_records:
        raise ValueError(f"Near-duplicate search capped at {max_records} records; partition or raise the audited limit")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(char_ngram, char_ngram), min_df=2, dtype=np.float32)
    matrix = vectorizer.fit_transform(frame["canonical_text"])
    neighbors = NearestNeighbors(n_neighbors=min(max_neighbors, len(frame)), metric="cosine", algorithm="brute").fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    findings = []
    seen = set()
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, right in zip(row_distances, row_indices):
            pair = tuple(sorted((left, int(right))))
            if left == right or pair in seen or 1 - float(distance) < threshold:
                continue
            seen.add(pair)
            findings.append({
                "left": frame.iloc[pair[0]]["sample_id"],
                "right": frame.iloc[pair[1]]["sample_id"],
                "cosine_similarity": 1 - float(distance),
                "same_label": frame.iloc[pair[0]]["label"] == frame.iloc[pair[1]]["label"],
            })
    return findings
