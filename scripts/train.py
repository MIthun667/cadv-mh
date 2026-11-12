#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.data.preprocessing import conservative_augment
from cadvmh.data.tokenization import special_token_mask
from cadvmh.losses.classification import effective_number_weights
from cadvmh.losses.objective import CADVMHLoss, LossWeights
from cadvmh.models import CADVMH
from cadvmh.training.checkpoints import load_checkpoint, save_checkpoint
from cadvmh.training.optimizer import build_optimizer
from cadvmh.training.seeds import seed_everything
from cadvmh.training.stages import configure_stage
from cadvmh.training.trainer import Trainer
from cadvmh.utils.config import load_config
from cadvmh.utils.hardware import git_report, hardware_report
from cadvmh.utils.logging import append_jsonl


class TextDataset(Dataset):
    def __init__(self, frame, labels):
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = {label: index for index, label in enumerate(labels)}
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {"text": row.text, "labels": self.label_to_id[row.label], "sample_id": row.sample_id}


def collator(tokenizer, max_length):
    def collate(rows):
        texts = [row["text"] for row in rows]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["special_token_mask"] = special_token_mask(encoded["input_ids"], tokenizer, encoded["attention_mask"])
        augmented = tokenizer([conservative_augment(text) for text in texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["augmented_input_ids"] = augmented["input_ids"]
        encoded["augmented_attention_mask"] = augmented["attention_mask"]
        encoded["augmented_special_token_mask"] = special_token_mask(augmented["input_ids"], tokenizer, augmented["attention_mask"])
        encoded["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        encoded["sample_id"] = [row["sample_id"] for row in rows]
        return encoded
    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--model", default="cadvmh", choices=["cadvmh"])
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--config")
    parser.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--ablation", choices=[f"A{i}" for i in range(9)])
    parser.add_argument("--disable-loss", action="append", choices=["necessity", "sufficiency", "sparsity", "continuity", "consistency"])
    parser.add_argument("--fusion-mode", choices=["full", "mean", "difference", "evidence"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, help="Smoke-test limit; omit for full training")
    parser.add_argument("--max-eval-batches", type=int, help="Smoke-test limit; omit for full evaluation")
    args = parser.parse_args()
    config_path = args.config or str(ROOT / "configs" / f"{args.dataset}.yaml")
    config = load_config(config_path)
    config["project"]["seed"] = args.seed
    ablation_modes = {
        "A0": (1, "full", {"necessity": 0, "sufficiency": 0, "sparsity": 0, "continuity": 0, "consistency": 0}),
        "A1": (None, "evidence", {"necessity": 0, "sufficiency": 1, "continuity": 0, "consistency": 0}),
        "A2": (None, "mean", {"necessity": 0, "sufficiency": 0, "continuity": 0, "consistency": 0}),
        "A3": (None, "mean", {"sufficiency": 0, "continuity": 0, "consistency": 0}),
        "A4": (None, "mean", {"continuity": 0, "consistency": 0}),
        "A5": (None, "full", {"continuity": 0, "consistency": 0}),
        "A6": (None, "full", {"consistency": 0}),
        "A7": (None, "full", {}),
        "A8": (None, "full", {}),
    }
    forced_stage, fusion_mode = None, "full"
    if args.ablation:
        forced_stage, fusion_mode, changes = ablation_modes[args.ablation]
        config["loss"].update(changes)
    for loss_name in args.disable_loss or []:
        config["loss"][loss_name] = 0.0
    if args.fusion_mode:
        fusion_mode = args.fusion_mode
    source = resolve_dataset_path(config)
    manifest_path = ROOT / "prepared" / args.dataset / "primary_split.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepare the immutable split first: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    config["dataset"]["path"] = str(source)
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise RuntimeError("Current dataset does not match the split manifest")
    labels = config["dataset"]["labels"]
    counts = frame.loc[frame.partition == "train", "label"].value_counts().reindex(labels, fill_value=0).tolist()
    summary = {"dataset": args.dataset, "rows": len(frame), "partitions": frame.partition.value_counts().to_dict(), "train_class_counts": dict(zip(labels, counts)), "config": config_path}
    if args.dry_run:
        partitions = summary["partitions"]
        print(
            f"Dry run ready for {args.dataset}: {summary['rows']:,} records; "
            f"train {partitions.get('train', 0):,}, validation {partitions.get('validation', 0):,}, "
            f"test {partitions.get('test', 0):,}; seed {args.seed}."
        )
        return
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"], use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Selected tokenizer has no mask token")
    model = CADVMH(
        len(labels), config["model"]["encoder_name"],
        selector_hidden_dim=config["model"]["selector_hidden_dim"],
        interaction_dim=config["model"]["interaction_dim"],
        dropout=config["model"]["selector_dropout"],
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
    )
    class_weights = effective_number_weights(counts, config["training"]["beta"])
    loss_weight_keys = {"classification", "original", "necessity", "sufficiency", "sparsity", "continuity", "consistency"}
    objective = CADVMHLoss(LossWeights(**{key: value for key, value in config["loss"].items() if key in loss_weight_keys}), class_weights)
    # Non-weight loss settings are not LossWeights fields.
    objective.necessity_margin = config["loss"]["necessity_margin"]
    objective.selection_ratio = config["model"]["selection_ratio"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments" / args.dataset / "cadvmh" / f"seed_{args.seed}"
    if run_dir.exists() and not args.resume:
        existing_entries = {path.name for path in run_dir.iterdir()}
        reusable_failed_run_entries = {"run_metadata.json"}
        if existing_entries - reusable_failed_run_entries:
            raise FileExistsError(f"Run directory contains training artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    json_write(run_dir / "run_metadata.json", summary | {"hardware": hardware_report(), "git": git_report(str(ROOT)), "seed": args.seed})
    loaders = {}
    for partition, shuffle in [("train", True), ("validation", False)]:
        subset = frame.loc[frame.partition == partition]
        loaders[partition] = DataLoader(TextDataset(subset, labels), batch_size=config["training"]["microbatch_size"], shuffle=shuffle, collate_fn=collator(tokenizer, config["model"]["max_length"]))
    stages = [forced_stage] if forced_stage else ([1, 2, 3] if args.stage == "all" else [int(args.stage)])
    split_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    final_stage = stages[-1]
    for stage in stages:
        configure_stage(model, stage)
        optimizer = build_optimizer(model, config["training"]["stage3_encoder_learning_rate"], config["training"]["selector_learning_rate"], config["training"]["module_learning_rate"], config["training"]["weight_decay"])
        trainer = Trainer(model, objective, optimizer, device, config["training"]["gradient_accumulation"], config["training"]["gradient_clip"], config["training"]["fp16"], fusion_mode)
        if args.resume:
            resumed = load_checkpoint(args.resume, model, optimizer=optimizer, scaler=trainer.scaler)
            if int(resumed["stage"]) != stage:
                raise ValueError(f"Resume checkpoint stage {resumed['stage']} does not match requested stage {stage}")
            args.resume = None
        epochs = config["training"][f"stage{stage}_epochs"]
        stage_best = float("-inf")
        stage_best_epoch = 0
        stage_best_state = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            temperature = config["model"]["temperature_initial"] if stage < 2 or args.ablation == "A7" else max(config["model"]["temperature_final"], config["model"]["temperature_initial"] - (epoch - 1) * 0.1)
            record = {
                "stage": stage,
                "epoch": epoch,
                **trainer.train_epoch(loaders["train"], stage, temperature, tokenizer.mask_token_id, args.max_train_batches),
                **trainer.evaluate(loaders["validation"], stage, tokenizer.mask_token_id, args.max_eval_batches),
            }
            append_jsonl(run_dir / "history.jsonl", record)
            if record["validation_macro_f1"] > stage_best:
                stage_best = record["validation_macro_f1"]
                stage_best_epoch = epoch
                stage_best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"Stage {stage}, epoch {epoch}: "
                f"train macro-F1 {record['train_macro_f1']:.4f}, "
                f"validation macro-F1 {record['validation_macro_f1']:.4f}, "
                f"validation loss {record['validation_loss']:.4f}, "
                f"runtime {record['runtime_seconds'] / 60:.1f} min."
            )
            if stale_epochs >= config["training"]["patience"]:
                break
        if stage_best_state is None:
            raise RuntimeError(f"Stage {stage} completed without a valid best model state")
        model.load_state_dict(stage_best_state)
        del stage_best_state
        if stage == final_stage:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                stage=stage,
                epoch=stage_best_epoch,
                config=config,
                label_mapping={v: i for i, v in enumerate(labels)},
                split_fingerprint=split_fingerprint,
            )
        print(f"Stage {stage} complete. Best validation macro-F1: {stage_best:.4f}.")
    print(f"Training complete. Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
