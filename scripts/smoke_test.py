#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
from transformers import RobertaConfig, RobertaModel

from _common import ROOT, json_write
from cadvmh.losses.objective import CADVMHLoss
from cadvmh.models import CADVMH
from cadvmh.training.checkpoints import load_checkpoint, save_checkpoint
from cadvmh.training.optimizer import build_optimizer
from cadvmh.training.seeds import seed_everything
from cadvmh.training.stages import configure_stage


def batch(device, batch_size=4, length=32):
    ids = torch.randint(5, 100, (batch_size, length), device=device)
    ids[:, 0], ids[:, -1] = 0, 2
    attention = torch.ones_like(ids)
    special = ids.eq(0) | ids.eq(1) | ids.eq(2)
    labels = torch.arange(batch_size, device=device) % 4
    return ids, attention, special, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(ROOT / "docs" / "gpu_smoke_results.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(42)
    config = RobertaConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    model = CADVMH(4, encoder=RobertaModel(config), selector_hidden_dim=32, interaction_dim=48, dropout=0.0).to(device)
    objective = CADVMHLoss().to(device)
    ids, attention, special, labels = batch(device)
    results = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "stages": {}}
    for stage in (1, 2, 3):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        state = configure_stage(model, stage)
        optimizer = build_optimizer(model)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(ids, attention, special, 4, stage=stage, temperature=0.7, enable_evidence=stage > 1)
            losses = objective(output, labels, stage=stage)
        losses["total"].backward()
        gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        results["stages"][str(stage)] = {
            "loss": float(losses["total"].detach()),
            "finite": bool(torch.isfinite(losses["total"]).item() and gradients_finite),
            "trainable_parameters": state["trainable"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "selector_gradient": bool(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.selector.parameters())),
        }
    model.eval()
    with torch.no_grad():
        prediction_before = model(ids, attention, special, 4, stage=3, selection_mode="top_rho").final_logits
    checkpoint = ROOT / "experiments" / "smoke" / "tiny_checkpoint.pt"
    save_checkpoint(checkpoint, model, stage=3, epoch=1, config={"synthetic": True}, label_mapping={str(i): i for i in range(4)}, split_fingerprint="synthetic")
    load_checkpoint(checkpoint, model, restore_rng=False)
    model.eval()
    with torch.no_grad():
        restored = model(ids, attention, special, 4, stage=3, selection_mode="top_rho")
    results["checkpoint_prediction_match"] = bool(torch.allclose(prediction_before, restored.final_logits))
    results["rationale_selection_ratio"] = float((restored.selection_mask * restored.valid_token_mask).sum() / restored.valid_token_mask.sum())
    json_write(__import__('pathlib').Path(args.output), results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
