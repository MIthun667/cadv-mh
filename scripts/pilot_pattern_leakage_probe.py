#!/usr/bin/env python
"""Does the *pattern* of selected positions (not the kept tokens) carry the label? Validation only.

Evidence-only inputs are rebuilt on token ids with the model's own selection and three fillings of the selected positions:
  kept     : the true tokens (the normal evidence view)
  random   : random vocabulary tokens at the selected positions
  foreign  : the tokens found at the same positions in another example of the batch (shift by one)
and, separately, the model's selection computed on example i is applied to a different example j ("swapped mask").
If accuracy stays high with random/foreign fillings, the label is encoded by which positions are selected.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from _common import ROOT, json_write, manifest_path, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.models import CADVMH
from cadvmh.utils.config import load_config
from pilot_diagnostics import Rows, collator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"])
    manifest = pd.read_csv(manifest_path(config))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id")
    frame = frame.loc[frame.partition == "validation"]
    labels = config["dataset"]["labels"]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m = payload["config"]["model"]
    tok = AutoTokenizer.from_pretrained(m["encoder_name"])
    model = CADVMH(len(labels), m["encoder_name"], selector_hidden_dim=m["selector_hidden_dim"], interaction_dim=m["interaction_dim"], dropout=m["dropout"])
    model.load_state_dict(payload["model"])
    dev = torch.device("cuda")
    model.eval().to(dev)
    loader = DataLoader(Rows(frame, labels), batch_size=32, shuffle=False, collate_fn=collator(tok, m["max_length"]))
    gen = torch.Generator().manual_seed(0)
    mask_id, vocab = tok.mask_token_id, model.encoder.model.config.vocab_size
    acc = {k: [] for k in ("kept", "random_fill", "foreign_fill", "swapped_mask_foreign_text", "all_masked")}
    ptrue = {k: [] for k in acc}
    with torch.no_grad():
        for raw in loader:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            ids, y = b["input_ids"], b["labels"]
            out = model(ids, b["attention_mask"], b["special_token_mask"], mask_id, stage=3, selection_mode="top_rho", selection_ratio=0.2, enable_evidence=False)
            elig = out.valid_token_mask.bool()
            sel = out.selection_mask.bool() & elig
            def run(x):
                o = model(x, b["attention_mask"], b["special_token_mask"], mask_id, stage=1).original_logits
                return o.argmax(-1).eq(y), torch.softmax(o, -1).gather(1, y[:, None]).squeeze(1)
            masked = torch.full_like(ids, mask_id)
            variants = {
                "kept": torch.where(elig & ~sel, masked, ids),
                "random_fill": torch.where(sel, torch.randint(4, vocab - 2, ids.shape, generator=gen).to(dev), torch.where(elig, masked, ids)),
                "foreign_fill": torch.where(sel, torch.roll(ids, 1, 0), torch.where(elig, masked, ids)),
                "swapped_mask_foreign_text": torch.where(torch.roll(sel, 1, 0) & elig, torch.roll(ids, 0, 0), torch.where(elig, masked, ids)),
                "all_masked": torch.where(elig, masked, ids),
            }
            for k, x in variants.items():
                c, p = run(x)
                acc[k] += c.tolist(); ptrue[k] += p.tolist()
    report = {k: {"aux_head_accuracy": float(np.mean(v)), "aux_head_mean_p_true": float(np.mean(ptrue[k])), "n": len(v)} for k, v in acc.items()}
    report["note"] = "All variants keep ~20% of positions; chance accuracy ~0.5. 'swapped_mask_foreign_text' keeps another example's selected positions of THIS text."
    json_write(__import__("pathlib").Path(args.output), report)
    for k, v in report.items():
        if isinstance(v, dict): print(f"{k:28s} acc {v['aux_head_accuracy']:.4f}  p_true {v['aux_head_mean_p_true']:.4f}")


if __name__ == "__main__":
    main()
