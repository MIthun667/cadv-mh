#!/usr/bin/env python
from __future__ import annotations

import argparse

from _common import ROOT, manifest_path, resolve_dataset_path
from cadvmh.data.deduplication import assign_exact_groups, deduplicate_representatives
from cadvmh.data.loader import load_dataset
from cadvmh.data.splitting import dataset_fingerprint, group_stratified_split, save_manifest
from cadvmh.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config or str(ROOT / "configs" / f"{args.dataset}.yaml")
    config = load_config(config_path)
    source = resolve_dataset_path(config)
    config["dataset"]["path"] = str(source)
    frame, summary = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    grouped, audit = assign_exact_groups(frame)
    primary = deduplicate_representatives(grouped, exclude_conflicts=True)
    manifest = group_stratified_split(primary, int(config["data"]["split_seed"]))
    counts = manifest.partition.value_counts()
    print(
        f"{args.dataset}: prepared {len(manifest):,} deduplicated records "
        f"(train {counts.get('train', 0):,}, validation {counts.get('validation', 0):,}, "
        f"test {counts.get('test', 0):,})."
    )
    if args.dry_run:
        return
    destination = manifest_path(config)
    save_manifest(
        manifest,
        destination,
        {
            "dataset": args.dataset,
            "dataset_sha256": dataset_fingerprint(source),
            "split_seed": config["data"]["split_seed"],
            "encoding": frame.attrs.get("encoding"),
            "c1_controls_removed": frame.attrs.get("c1_controls_in_raw"),
            "deduplication": "canonical exact groups; conflicting groups excluded; one representative per group",
            "actual_partition_counts": manifest.partition.value_counts().to_dict(),
        },
    )
    print(f"Split manifest created: {destination}")


if __name__ == "__main__":
    main()
