#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess

from _common import ROOT


ABLATIONS = {f"A{i}": ["--ablation", f"A{i}"] for i in range(9)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for ablation in args.ablations:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown or not-yet-certified ablation: {ablation}")
        seed = args.seed
        extra = ABLATIONS[ablation]
        command = ["python", "scripts/train.py", "--dataset", args.dataset, "--seed", str(seed), "--output-dir", f"experiments/{args.dataset}/ablations/{ablation}/seed_{seed}", *extra]
        if args.execute:
            print(f"{args.dataset}: starting ablation {ablation} with seed {seed}.")
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print(" ".join(command))
    if args.leave_one_out:
        for component in ("necessity", "sufficiency", "sparsity", "continuity", "consistency"):
            seed = args.seed
            command = ["python", "scripts/train.py", "--dataset", args.dataset, "--seed", str(seed), "--stage", "all", "--disable-loss", component, "--output-dir", f"experiments/{args.dataset}/ablations/without_{component}/seed_{seed}"]
            if args.execute:
                print(f"{args.dataset}: starting leave-one-out run without {component}.")
                subprocess.run(command, cwd=ROOT, check=True)
            else:
                print(" ".join(command))


if __name__ == "__main__":
    main()
