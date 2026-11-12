#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.baselines.neural import BiLSTMClassifier, SparseRationalizer
from cadvmh.data.loader import load_dataset
from cadvmh.evaluation.metrics import classification_metrics
from cadvmh.training.seeds import seed_everything
from cadvmh.utils.config import load_config
from cadvmh.utils.hardware import hardware_report


MODELS = {
    "majority": None,
    "bilstm": None,
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "mental_roberta": "mental/mental-roberta-base",
    "sparse_rationalizer": None,
}


def load_partitions(dataset: str):
    config = load_config(ROOT / "configs" / f"{dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    manifest = pd.read_csv(ROOT / "prepared" / dataset / "primary_split.csv")
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one")
    return config, frame


class WordDataset(Dataset):
    def __init__(self, frame, vocabulary, label_to_id, max_length):
        self.rows = frame.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        tokens = str(row.text).lower().split()[: self.max_length]
        ids = [self.vocabulary.get(token, 1) for token in tokens] or [1]
        return torch.tensor(ids), self.label_to_id[row.label]


def word_collator(rows):
    lengths = [len(ids) for ids, _ in rows]
    input_ids = torch.zeros(len(rows), max(lengths), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for index, (ids, _) in enumerate(rows):
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor([label for _, label in rows])}


class TransformerDataset(Dataset):
    def __init__(self, frame, label_to_id):
        self.rows = frame.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return str(row.text), self.label_to_id[row.label]


def transformer_collator(tokenizer, max_length):
    def collate(rows):
        encoded = tokenizer([text for text, _ in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([label for _, label in rows])
        return encoded
    return collate


def predict(model, loader, device, rationale=False, max_batches=None):
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            if rationale:
                logits, _ = model(**inputs, hard=True)
            elif isinstance(model, BiLSTMClassifier):
                logits = model(**inputs)
            else:
                logits = model(**inputs).logits
            probabilities.append(torch.softmax(logits, -1).cpu().numpy())
            targets.extend(labels.cpu().tolist())
    return np.asarray(targets), np.concatenate(probabilities)


def train_neural(model, train_loader, validation_loader, device, epochs, learning_rate, patience, rationale, sparsity, continuity, max_batches, label_names):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                if rationale:
                    logits, selection = model(**inputs)
                    eligible = inputs["attention_mask"]
                    sparse_loss = (selection * eligible).sum() / eligible.sum().clamp_min(1)
                    differences = torch.abs(selection[:, 1:] - selection[:, :-1])
                    boundaries = eligible[:, 1:] * eligible[:, :-1]
                    continuity_loss = (differences * boundaries).sum() / boundaries.sum().clamp_min(1)
                    loss = F.cross_entropy(logits, labels) + sparsity * sparse_loss + continuity * continuity_loss
                elif isinstance(model, BiLSTMClassifier):
                    loss = F.cross_entropy(model(**inputs), labels)
                else:
                    loss = model(**inputs, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        targets, probabilities = predict(model, validation_loader, device, rationale, max_batches)
        macro_f1 = classification_metrics(targets, probabilities, label_names)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": macro_f1})
        print(f"Epoch {epoch}: loss {np.mean(losses):.4f}; validation macro-F1 {macro_f1:.4f}.")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--baseline", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, frame = load_partitions(args.dataset)
    labels = config["dataset"]["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    partitions = {name: frame.loc[frame.partition == name] for name in ("train", "validation", "test")}
    output = ROOT / "experiments" / args.dataset / "extended_baselines" / args.baseline / f"seed_{args.seed}"
    print(f"{args.dataset}: {args.baseline}; " + ", ".join(f"{key} {len(value):,}" for key, value in partitions.items()))
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite baseline output: {output}")
    output.mkdir(parents=True)
    seed_everything(args.seed)
    if args.baseline == "majority":
        majority = partitions["train"].label.value_counts().idxmax()
        targets = partitions["test"].label.map(label_to_id).to_numpy()
        predictions = np.full(len(targets), label_to_id[majority])
        probabilities = np.eye(len(labels))[predictions]
        metrics = classification_metrics(targets, probabilities, labels)
        json_write(output / "test_metrics.json", metrics | {"majority_label": majority})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rationale = args.baseline == "sparse_rationalizer"
    if args.baseline in {"bilstm", "sparse_rationalizer"}:
        counts = Counter(token for text in partitions["train"].text for token in str(text).lower().split())
        vocabulary = {token: index + 2 for index, (token, _) in enumerate(counts.most_common(50000))}
        datasets = {name: WordDataset(part, vocabulary, label_to_id, config["model"]["max_length"]) for name, part in partitions.items()}
        loaders = {name: DataLoader(data, batch_size=32, shuffle=name == "train", collate_fn=word_collator) for name, data in datasets.items()}
        model_class = SparseRationalizer if rationale else BiLSTMClassifier
        model = model_class(len(vocabulary) + 2, 200, 128, len(labels)).to(device)
        learning_rate = 1e-3
        (output / "vocabulary.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    else:
        checkpoint = MODELS[args.baseline]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        datasets = {name: TransformerDataset(part, label_to_id) for name, part in partitions.items()}
        loaders = {
            name: DataLoader(data, batch_size=config["training"]["microbatch_size"], shuffle=name == "train", collate_fn=transformer_collator(tokenizer, config["model"]["max_length"]))
            for name, data in datasets.items()
        }
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels)).to(device)
        model.config.id2label = {index: label for label, index in label_to_id.items()}
        model.config.label2id = label_to_id
        learning_rate = config["training"]["encoder_learning_rate"]

    history = train_neural(model, loaders["train"], loaders["validation"], device, args.epochs, learning_rate, config["training"]["patience"], rationale, 0.05, 0.05, args.max_train_batches, labels)
    targets, probabilities = predict(model, loaders["test"], device, rationale, args.max_eval_batches)
    json_write(output / "test_metrics.json", classification_metrics(targets, probabilities, labels))
    json_write(output / "run_metadata.json", {"dataset": args.dataset, "baseline": args.baseline, "seed": args.seed, "hardware": hardware_report(), "history": history})
    if args.baseline in {"bert_base", "roberta_base", "mental_roberta"}:
        model.save_pretrained(output / "model")
        tokenizer.save_pretrained(output / "model")
    else:
        torch.save(model.state_dict(), output / "model.pt")
    print(f"Completed {args.baseline}; test macro-F1 {classification_metrics(targets, probabilities, labels)['macro_f1']:.4f}.")


if __name__ == "__main__":
    main()
