#!/usr/bin/env python
from __future__ import annotations

import argparse

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.data.deduplication import assign_exact_groups
from cadvmh.data.loader import load_dataset
from cadvmh.data.splitting import dataset_fingerprint
from cadvmh.utils.config import load_config


def audit(config_path: str) -> dict:
    config = load_config(config_path)
    path = resolve_dataset_path(config)
    config["dataset"]["path"] = str(path)
    frame, validation = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    grouped, duplicate = assign_exact_groups(frame)
    lengths = grouped["text"].str.split().str.len()
    return {
        "dataset": config["dataset"]["name"],
        "path": str(path),
        "sha256": dataset_fingerprint(path),
        "validation": validation.__dict__,
        "duplicates": duplicate.__dict__,
        "class_counts": grouped["label"].value_counts().sort_index().to_dict(),
        "unique_groups_by_class": grouped.groupby("label")["group_id"].nunique().sort_index().to_dict(),
        "text_words": {str(k): float(v) for k, v in lengths.describe(percentiles=[0.1, 0.5, 0.9, 0.99]).to_dict().items()},
        "conflicting_group_examples": grouped.loc[grouped.label_conflict, ["group_id", "label"]].drop_duplicates().head(50).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = args.config or str(ROOT / "configs" / f"{args.dataset}.yaml")
    result = audit(config)
    destination = ROOT / "experiments" / args.dataset / f"{args.dataset}_audit.json"
    json_write(destination, result)
    print(
        f"{args.dataset}: audit complete "
        f"({result['validation']['rows_valid']:,} valid rows, "
        f"{result['duplicates']['exact_groups']:,} exact-text groups)."
    )


if __name__ == "__main__":
    main()
