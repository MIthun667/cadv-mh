#!/usr/bin/env python
"""CPU-only toy check (synthetic data, tiny random RoBERTa): legacy soft masks vs repaired budgeted straight-through.

Label = which of 5 "signal" tokens appears (exactly one, at a random position) among noise tokens, so the true
rationale is known. This is a mechanism check, NOT evidence about real data or final model quality.
"""
from __future__ import annotations

import json

import torch
from transformers import RobertaConfig, RobertaModel

from _common import ROOT  # noqa: F401  (adds src to sys.path)
from cadvmh.losses.objective import CADVMHLoss, LossWeights
from cadvmh.models import CADVMH
from cadvmh.training.optimizer import build_optimizer
from cadvmh.training.stages import configure_stage, set_stage_train_mode

SEQ, MASK_ID = 20, 4


def make_data(n, generator):
    ids = torch.randint(20, 60, (n, SEQ), generator=generator)
    ids[:, 0], ids[:, -1] = 0, 2
    labels = torch.randint(0, 5, (n,), generator=generator)
    positions = torch.randint(1, SEQ - 1, (n,), generator=generator)
    ids[torch.arange(n), positions] = 10 + labels
    attention = torch.ones_like(ids)
    special = (ids == 0) | (ids == 2)
    return dict(input_ids=ids, attention_mask=attention, special_token_mask=special, labels=labels, signal=positions)


def build():
    cfg = RobertaConfig(vocab_size=64, hidden_size=48, num_hidden_layers=2, num_attention_heads=4, intermediate_size=96,
                        max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return CADVMH(5, encoder=RobertaModel(cfg), selector_hidden_dim=32, interaction_dim=48, dropout=0.0)


def run(mode, seed=0, steps1=300, steps2=300, weights=None, selector_lr=5e-3):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train, test = make_data(512, gen), make_data(256, torch.Generator().manual_seed(99))
    model = build()
    keys = ("input_ids", "attention_mask", "special_token_mask")
    objective = CADVMHLoss(weights or LossWeights(), selection_ratio=0.2)
    configure_stage(model, 1); set_stage_train_mode(model, 1)
    opt = build_optimizer(model, 1e-3, 1e-3, 1e-3, 0.0)
    for step in range(steps1):
        idx = torch.randint(0, 512, (32,), generator=gen)
        out = model(*(train[k][idx] for k in keys), MASK_ID, stage=1)
        loss = objective(out, train["labels"][idx], stage=1)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    configure_stage(model, 2); set_stage_train_mode(model, 2)
    opt = build_optimizer(model, 1e-3, selector_lr, 1e-3, 0.0)
    for step in range(steps2):
        idx = torch.randint(0, 512, (32,), generator=gen)
        out = model(*(train[k][idx] for k in keys), MASK_ID, stage=2, temperature=1.0, selection_mode=mode, selection_ratio=0.2)
        losses = objective(out, train["labels"][idx], stage=2)
        opt.zero_grad(); losses["total"].backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(*(test[k] for k in keys), MASK_ID, stage=2, selection_mode="top_rho", selection_ratio=0.2)
        eligible = out.valid_token_mask.float()
        hard_ratio = float((out.selection_mask * eligible).sum() / eligible.sum())
        hit = float(out.selection_mask[torch.arange(256), test["signal"]].mean())  # signal token inside the rationale
        probs = lambda lg: torch.softmax(lg, -1).gather(1, test["labels"][:, None]).squeeze(1)
        comp = float((probs(out.original_logits) - probs(out.counterfactual_logits)).mean())
        suff = float((probs(out.original_logits) - probs(out.evidence_logits)).mean())
        # training-regime ratio: what the training-time selection looked like on the test inputs
        model.train()
        tr = model(*(test[k] for k in keys), MASK_ID, stage=2, temperature=1.0, selection_mode=mode, selection_ratio=0.2)
        train_ratio = float((tr.selection_mask.detach() * eligible).sum() / eligible.sum())
        sat = float(((out.selector_logits.abs() > 7) & out.valid_token_mask).sum() / out.valid_token_mask.sum())
    return dict(mode=mode, train_regime_selected_ratio=round(train_ratio, 3), eval_hard_ratio=round(hard_ratio, 3),
                signal_token_in_rationale=round(hit, 3), chance_level=0.2, aux_comprehensiveness=round(comp, 3),
                aux_sufficiency=round(suff, 3), selector_saturation=round(sat, 3),
                final_train_sparsity_loss=round(float(losses["sparsity"]), 3), final_train_necessity=round(float(losses["necessity"]), 3),
                final_train_sufficiency=round(float(losses["sufficiency"]), 3))


if __name__ == "__main__":
    results = [run(m) for m in ("soft", "budgeted_st")]
    print(json.dumps(results, indent=2))
