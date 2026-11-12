#!/usr/bin/env python
"""Validation/train-batch probe of selector logit scale and straight-through gradient magnitude (no training)."""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from transformers import AutoTokenizer

from _common import ROOT, manifest_path, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.data.tokenization import special_token_mask
from cadvmh.losses.objective import CADVMHLoss, LossWeights
from cadvmh.models import CADVMH
from cadvmh.training.stages import configure_stage, set_stage_train_mode
from cadvmh.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_b")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--modes", nargs="+", default=["budgeted_st_sigmoid", "budgeted_st"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"])
    manifest = pd.read_csv(manifest_path(config))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id")
    texts = frame.loc[frame.partition == "validation"].head(64)
    labels = torch.tensor([1 if l == "adhd" else 0 for l in texts.label])
    tok = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])
    enc = tok(list(texts.text), padding=True, truncation=True, max_length=128, return_tensors="pt")
    special = special_token_mask(enc["input_ids"], tok, enc["attention_mask"])
    dev = torch.device("cuda")
    batch = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev), "special_token_mask": special.to(dev)}
    labels = labels.to(dev)
    eligible = batch["attention_mask"].bool() & ~batch["special_token_mask"].bool()
    report = {}
    for pair in args.checkpoints:
        name, path = pair.split("=")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        m = payload["config"]["model"]
        model = CADVMH(2, m["encoder_name"], selector_hidden_dim=m["selector_hidden_dim"], interaction_dim=m["interaction_dim"], dropout=m["dropout"])
        model.load_state_dict(payload["model"])
        model.to(dev).eval()
        entry = {}
        with torch.no_grad():
            hid = model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], position_ids=model.encoder.position_ids_from_input_ids(batch["input_ids"]))
            pooled, _ = model.pooling(hid, batch["attention_mask"])
            logits = model.selector(hid, pooled)[eligible]
            sel = model.selector
            g = pooled.unsqueeze(1).expand_as(hid)
            blocks = [hid, g, hid * g, (hid - g).abs()]
            entry["hidden_abs_mean"] = float(hid[eligible].abs().mean())
            entry["block_abs_mean"] = [float(b[eligible].abs().mean()) for b in blocks]
            entry["logit_mean"], entry["logit_std_global"] = float(logits.mean()), float(logits.std())
            per_example = model.selector(hid, pooled).masked_fill(~eligible, float("nan"))
            entry["logit_within_example_std_mean"] = float(torch.nanmean(torch.stack([torch.tensor(r[~torch.isnan(r)].std().item()) for r in per_example.cpu()])))
            ablated = {}
            for i, label in enumerate(["token", "pooled", "product", "abs_diff"]):
                kept = [b if j != i else torch.zeros_like(b) for j, b in enumerate(blocks)]
                ablated[f"logit_mean_without_{label}_block"] = float(sel.network(torch.cat(kept, -1))[eligible].mean())
            entry.update(ablated)
            p = torch.sigmoid(logits)
            entry["mean_sigmoid_slope_p(1-p)"] = float((p * (1 - p)).mean())
        # straight-through gradient magnitude on the selector under each surrogate
        configure_stage(model, 2)
        grads = {}
        for mode in args.modes:
            set_stage_train_mode(model, 2)
            sums, means, stds = [], [], []
            captured = {}
            handle = model.selector.register_forward_hook(lambda mod, inp, out: (out.retain_grad(), captured.__setitem__("l", out)) and None)
            for seed in range(4):
                torch.manual_seed(seed)
                model.zero_grad()
                out = model(**batch, mask_token_id=tok.mask_token_id, stage=2, temperature=1.0, selection_mode=mode, selection_ratio=0.2, enable_evidence=True)
                loss = CADVMHLoss(LossWeights(sparsity=0.0))(out, labels, stage=2)["total"]
                loss.backward()
                sums.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.selector.parameters() if p.grad is not None))))
                g = captured["l"].grad[eligible]
                means.append(float(g.mean())); stds.append(float(g.std()))
            handle.remove()
            # dL/dlogit per token: a mean far from 0 relative to the std means a common-mode push on all logits
            grads[mode] = {"selector_grad_norm_mean": sum(sums) / len(sums), "per_seed": sums,
                           "dL_dlogit_token_mean": sum(means) / 4, "dL_dlogit_token_std": sum(stds) / 4,
                           "common_mode_ratio_abs_mean_over_std": sum(abs(m) / max(sd, 1e-12) for m, sd in zip(means, stds)) / 4,
                           "sum_over_tokens_per_example_mean": None}
        entry["selector_gradient_norm"] = grads
        report[name] = entry
        del model
        torch.cuda.empty_cache()
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
