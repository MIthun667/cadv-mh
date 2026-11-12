#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess

from _common import ROOT, run_root
from cadvmh.training.ablations import ABLATIONS as SPECS
from cadvmh.utils.config import load_config


ABLATIONS = {name: ["--ablation", name] for name in SPECS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    for ablation in args.ablations:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown or not-yet-certified ablation: {ablation}")
        seed = args.seed
        extra = ABLATIONS[ablation]
        base = run_root(config).relative_to(ROOT)
        # A8 is the canonical full model: it is trained once, into the canonical run directory.
        target = f"{args.dataset}/cadvmh/seed_{seed}" if SPECS[ablation].alias_of_canonical_run else f"{args.dataset}/ablations/{ablation}/seed_{seed}"
        command = ["python", "scripts/train.py", "--dataset", args.dataset, "--seed", str(seed), "--output-dir", f"{base}/{target}", *extra]
        if args.execute:
            print(f"{args.dataset}: starting ablation {ablation} with seed {seed}.")
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print(" ".join(command))
    if args.leave_one_out:
        for component in ("necessity", "sufficiency", "continuity", "consistency"):  # sparsity is structural under budgeted_st
            seed = args.seed
            command = ["python", "scripts/train.py", "--dataset", args.dataset, "--seed", str(seed), "--stage", "all", "--disable-loss", component, "--output-dir", f"{run_root(config).relative_to(ROOT)}/{args.dataset}/ablations/without_{component}/seed_{seed}"]
            if args.execute:
                print(f"{args.dataset}: starting leave-one-out run without {component}.")
                subprocess.run(command, cwd=ROOT, check=True)
            else:
                print(" ".join(command))


if __name__ == "__main__":
    main()
