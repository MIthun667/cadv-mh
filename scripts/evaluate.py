#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.evaluation.metrics import classification_metrics
from cadvmh.models import CADVMH
from cadvmh.training.checkpoints import load_checkpoint
from cadvmh.utils.config import load_config
from train import TextDataset, collator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--partition", default="test", choices=["validation", "test"])
    parser.add_argument("--output")
    parser.add_argument("--fusion-mode", default="full", choices=["full", "mean", "difference", "evidence"])
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    manifest = pd.read_csv(ROOT / "prepared" / args.dataset / "primary_split.csv")
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one")
    frame = frame.loc[frame.partition == args.partition]
    labels = config["dataset"]["labels"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"], use_fast=True)
    model = CADVMH(len(labels), config["model"]["encoder_name"], selector_hidden_dim=config["model"]["selector_hidden_dim"], interaction_dim=config["model"]["interaction_dim"])
    payload = load_checkpoint(args.checkpoint, model, restore_rng=False)
    trained_seed = payload.get("config", {}).get("project", {}).get("seed")
    if trained_seed != 42:
        raise ValueError(f"Only seed 42 checkpoints may be evaluated; checkpoint seed is {trained_seed!r}")
    checkpoint_stage = int(payload["stage"])
    del payload
    gc.collect()
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    loader = DataLoader(TextDataset(frame, labels), batch_size=config["training"]["microbatch_size"], shuffle=False, collate_fn=collator(tokenizer, config["model"]["max_length"]))
    probabilities, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.dataset}", unit="batch", dynamic_ncols=True):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["special_token_mask"].to(device),
                tokenizer.mask_token_id,
                stage=checkpoint_stage,
                selection_mode="top_rho",
                enable_evidence=args.fusion_mode == "evidence",
                return_auxiliary=False,
                fusion_mode=args.fusion_mode,
            )
            probabilities.append(torch.softmax(output.final_logits, -1).cpu().numpy())
            targets.extend(batch["labels"].tolist())
    metrics = classification_metrics(np.asarray(targets), np.concatenate(probabilities), labels)
    destination = args.output or str(ROOT / "experiments" / args.dataset / "evaluation" / f"{args.partition}_metrics.json")
    json_write(ROOT / destination if not str(destination).startswith("/") else __import__('pathlib').Path(destination), metrics)
    print(f"Evaluation complete. Macro-F1: {metrics['macro_f1']:.4f}.")
    print(f"Metrics saved to: {destination}")


if __name__ == "__main__":
    main()
