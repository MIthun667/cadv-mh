#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys

from _common import ROOT, run_root
from cadvmh.utils.config import load_config


MODELS = [
    ("original_roberta", "baselines/original_roberta/seed_42", "full"),
    ("A0", "ablations/A0/seed_42", "full"),
    ("A1", "ablations/A1/seed_42", "evidence"),
    ("A2", "ablations/A2/seed_42", "mean"),
    ("A3", "ablations/A3/seed_42", "mean"),
    ("A4", "ablations/A4/seed_42", "mean"),
    ("A5", "ablations/A5/seed_42", "full"),
    ("A6", "ablations/A6/seed_42", "full"),
    ("A7", "ablations/A7/seed_42", "full"),
    ("A8", "ablations/A8/seed_42", "full"),
    ("CADV-MH", "cadvmh/seed_42", "full"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every completed seed-42 neural checkpoint without overwriting metrics."
    )
    parser.add_argument(
        "--dataset", choices=["dataset_a", "dataset_b", "both"], default="both"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = ["dataset_a", "dataset_b"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        for name, relative, fusion in MODELS:
            run_dir = run_root(load_config(ROOT / "configs" / f"{dataset}.yaml")) / dataset / relative
            checkpoint = run_dir / "best.pt"
            output = run_dir / "test_metrics.json"
            prefix = f"{dataset}/{name}"
            if output.exists():
                print(f"{prefix}: evaluation already complete; skipping.")
                continue
            if not checkpoint.exists():
                print(f"{prefix}: no completed checkpoint; skipping.")
                continue
            command = [
                sys.executable,
                "scripts/evaluate.py",
                "--dataset",
                dataset,
                "--checkpoint",
                str(checkpoint),
                "--partition",
                "test",
                "--fusion-mode",
                fusion,
                "--output",
                str(output),
            ]
            print(f"{prefix}: evaluating test split with {fusion} fusion.")
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
