import pandas as pd

from cadvmh.data.deduplication import assign_exact_groups, deduplicate_representatives, find_near_duplicate_pairs
from cadvmh.data.splitting import group_stratified_split, verify_split_disjointness


def test_duplicate_and_conflict_detection():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"], "text": ["Hello  world", "hello world", "hello world"], "label": ["x", "x", "y"]})
    grouped, audit = assign_exact_groups(frame)
    assert audit.conflicting_groups == 1
    assert deduplicate_representatives(grouped).empty


def test_group_split_is_deterministic_and_disjoint():
    rows = []
    for label in ["a", "b"]:
        for index in range(40):
            rows.append({"sample_id": f"{label}{index}", "text": f"{label} text {index}", "label": label})
    grouped, _ = assign_exact_groups(pd.DataFrame(rows))
    first = group_stratified_split(grouped, 42)
    second = group_stratified_split(grouped, 42)
    assert first[["sample_id", "partition"]].equals(second[["sample_id", "partition"]])
    verify_split_disjointness(first)


def test_auditable_near_duplicate_candidates():
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "canonical_text": ["i feel very hopeless today", "i feel very hopeless today!", "the weather is sunny"],
        "label": ["x", "x", "y"],
    })
    pairs = find_near_duplicate_pairs(frame, char_ngram=3, threshold=0.8, max_neighbors=3)
    assert any({pair["left"], pair["right"]} == {"a", "b"} for pair in pairs)
