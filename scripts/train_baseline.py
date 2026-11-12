#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from _common import ROOT, json_write, manifest_path, resolve_dataset_path, run_root
from cadvmh.baselines.classical import build_tfidf_logistic, build_tfidf_svm
from cadvmh.data.loader import load_dataset
from cadvmh.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--baseline", choices=["tfidf_logistic", "tfidf_svm", "original_roberta"], required=True)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    manifest = pd.read_csv(manifest_path(config))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one")
    if args.baseline == "original_roberta":
        command = ["python", "scripts/train.py", "--dataset", args.dataset, "--stage", "1", "--seed", str(args.seed), "--output-dir", f"{config['project']['output_root']}/{args.dataset}/baselines/original_roberta/seed_{args.seed}"]
        print(f"{args.dataset}: starting original-view RoBERTa baseline with seed {args.seed}.")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
        return
    train, validation, test = (frame.loc[frame.partition == name] for name in ("train", "validation", "test"))
    print(
        f"{args.dataset}: {args.baseline} baseline; "
        f"train {len(train):,}, validation {len(validation):,}, test {len(test):,}."
    )
    if args.dry_run:
        return
    model = build_tfidf_logistic(args.seed) if args.baseline == "tfidf_logistic" else build_tfidf_svm()
    model.fit(train.text, train.label)
    predictions = model.predict(test.text)
    result = {
        "dataset": args.dataset,
        "baseline": args.baseline,
        "seed": args.seed,
        "macro_f1": f1_score(test.label, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(test.label, predictions, labels=config["dataset"]["labels"], zero_division=0, output_dict=True),
    }
    output = run_root(config) / args.dataset / "baselines" / args.baseline / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / "model.joblib")
    json_write(output / "metrics.json", result)
    print(f"{args.baseline} complete. Test macro-F1: {result['macro_f1']:.4f}.")


if __name__ == "__main__":
    main()
