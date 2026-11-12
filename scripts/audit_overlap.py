#!/usr/bin/env python
from __future__ import annotations

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.data.deduplication import assign_exact_groups
from cadvmh.data.loader import load_dataset
from cadvmh.utils.config import load_config


def loaded(name):
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    return assign_exact_groups(frame)[0]


def main():
    left, right = loaded("dataset_a"), loaded("dataset_b")
    overlap = left[["canonical_text", "sample_id", "label"]].merge(
        right[["canonical_text", "sample_id", "label"]], on="canonical_text", suffixes=("_a", "_b")
    )
    report = {
        "unique_overlapping_canonical_texts": int(overlap.canonical_text.nunique()),
        "overlap_pairs": len(overlap),
        "label_pairs": overlap.groupby(["label_a", "label_b"]).size().rename("count").reset_index().to_dict("records"),
        "target_labels_used_for_scientific_overlap_audit": True,
        "used_for_model_selection": False,
    }
    destination = ROOT / "experiments" / "cross_dataset" / "overlap_audit.json"
    json_write(destination, report)
    print(
        "Cross-dataset overlap audit complete: "
        f"{report['unique_overlapping_canonical_texts']:,} shared canonical texts."
    )


if __name__ == "__main__":
    main()
