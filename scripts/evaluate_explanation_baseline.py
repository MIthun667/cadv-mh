#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.utils.config import load_config
from train_extended_baseline import TransformerDataset, transformer_collator


def eligible_tokens(input_ids, attention_mask, tokenizer):
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special |= input_ids.eq(token_id)
    return attention_mask.bool() & ~special


def top_budget(scores, eligible, ratio):
    selection = torch.zeros_like(scores, dtype=torch.bool)
    for row in range(scores.size(0)):
        indices = torch.where(eligible[row])[0]
        if indices.numel():
            count = max(1, round(float(ratio) * indices.numel()))
            chosen = indices[
                torch.topk(scores[row, indices], min(count, indices.numel())).indices
            ]
            selection[row, chosen] = True
    return selection


def attention_scores(model, batch):
    output = model(**batch, output_attentions=True)
    if not output.attentions:
        raise RuntimeError(
            "The model returned no attention tensors. Load the explanation model "
            "with attn_implementation='eager'."
        )
    attentions = output.attentions[-1].mean(1)
    return output.logits, attentions[:, 0, :]


def integrated_gradient_scores(model, batch, tokenizer, steps):
    embedding = model.get_input_embeddings()
    inputs = embedding(batch["input_ids"]).detach()
    baseline_ids = batch["input_ids"].clone()
    eligible = eligible_tokens(batch["input_ids"], batch["attention_mask"], tokenizer)
    baseline_ids[eligible] = tokenizer.mask_token_id
    baseline = embedding(baseline_ids).detach()
    with torch.no_grad():
        logits = model(**batch).logits
        classes = logits.argmax(-1)
    accumulated = torch.zeros_like(inputs)
    for alpha in torch.linspace(0, 1, steps, device=inputs.device):
        interpolated = (
            (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        )
        step_logits = model(
            inputs_embeds=interpolated, attention_mask=batch["attention_mask"]
        ).logits
        gradients = torch.autograd.grad(
            step_logits.gather(1, classes[:, None]).sum(), interpolated
        )[0]
        accumulated += gradients.detach()
    attribution = (inputs - baseline) * accumulated / steps
    return logits, attribution.norm(p=2, dim=-1)


def masked_probabilities(model, batch, selection, eligible, tokenizer):
    original = model(**batch).logits
    counterfactual_ids = batch["input_ids"].clone()
    counterfactual_ids[selection] = tokenizer.mask_token_id
    evidence_ids = batch["input_ids"].clone()
    evidence_ids[eligible & ~selection] = tokenizer.mask_token_id
    counterfactual = model(
        input_ids=counterfactual_ids, attention_mask=batch["attention_mask"]
    ).logits
    evidence = model(
        input_ids=evidence_ids, attention_mask=batch["attention_mask"]
    ).logits
    return [torch.softmax(value, -1) for value in (original, counterfactual, evidence)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument(
        "--method", choices=["attention", "integrated_gradients"], required=True
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--selection-ratio", type=float, default=0.20)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--output", help="Optional metrics path; useful for bounded smoke checks"
    )
    args = parser.parse_args()
    if not 0 < args.selection_ratio <= 1:
        raise ValueError("--selection-ratio must be in (0, 1]")
    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else ROOT
        / "experiments"
        / args.dataset
        / "extended_baselines"
        / "roberta_base"
        / "seed_42"
        / "model"
    )
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    manifest = pd.read_csv(ROOT / "prepared" / args.dataset / "primary_split.csv")
    frame = frame.merge(
        manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one"
    )
    frame = frame.loc[frame.partition == "test"]
    label_to_id = {
        label: index for index, label in enumerate(config["dataset"]["labels"])
    }
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "Explanation evaluation requires a tokenizer with a mask token"
        )
    model_kwargs = (
        {"attn_implementation": "eager"} if args.method == "attention" else {}
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, **model_kwargs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    loader = DataLoader(
        TransformerDataset(frame, label_to_id),
        batch_size=config["training"]["microbatch_size"],
        shuffle=False,
        collate_fn=transformer_collator(tokenizer, config["model"]["max_length"]),
    )
    comprehensiveness, sufficiency, ratios = [], [], []
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        eligible = eligible_tokens(
            batch["input_ids"], batch["attention_mask"], tokenizer
        )
        if args.method == "attention":
            with torch.no_grad():
                _, scores = attention_scores(model, batch)
        else:
            scores = integrated_gradient_scores(model, batch, tokenizer, args.ig_steps)[
                1
            ]
        selection = top_budget(scores, eligible, args.selection_ratio)
        with torch.no_grad():
            original, counterfactual, evidence = masked_probabilities(
                model, batch, selection, eligible, tokenizer
            )
            classes = original.argmax(-1)
            indices = classes[:, None]
            comprehensiveness.extend(
                (original.gather(1, indices) - counterfactual.gather(1, indices))
                .squeeze(1)
                .cpu()
                .tolist()
            )
            sufficiency.extend(
                (original.gather(1, indices) - evidence.gather(1, indices))
                .squeeze(1)
                .cpu()
                .tolist()
            )
            ratios.extend(
                (selection.sum(1) / eligible.sum(1).clamp_min(1)).cpu().tolist()
            )
    result = {
        "dataset": args.dataset,
        "method": args.method,
        "model_dir": str(model_dir),
        "examples": len(comprehensiveness),
        "selection_ratio_target": args.selection_ratio,
        "comprehensiveness_mean": float(np.mean(comprehensiveness)),
        "sufficiency_mean": float(np.mean(sufficiency)),
        "selection_ratio_mean": float(np.mean(ratios)),
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "experiments"
        / args.dataset
        / "explanation_baselines"
        / args.method
        / "seed_42"
        / "metrics.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite explanation output: {output}")
    json_write(output, result)
    print(f"{args.method} complete on {len(comprehensiveness):,} examples: {output}")


if __name__ == "__main__":
    main()
