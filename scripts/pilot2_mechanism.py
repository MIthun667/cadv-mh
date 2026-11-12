#!/usr/bin/env python
"""Mask-hypersensitivity mechanism analysis (validation split + a train subsample; never the test split).

For each checkpoint name=path, reports, separately for predicted OCD and predicted ADHD:
  * hinge activity / gradient magnitude of the three candidate necessity objectives (A absolute, B prob-contrast, C log-prob contrast)
  * generic mask response: drop of the predicted-class probability under random masks vs random-word replacement vs learned masks
  * mask-fraction curves, fully-masked prediction share
  * auxiliary-head vs final-head agreement (stage 3)
  * variance of the random-mask term across draws
  * train-subsample vs validation behaviour of the counterfactual branch (overfitting)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from _common import ROOT, manifest_path, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.models import CADVMH
from cadvmh.models.masking import budgeted_topk
from cadvmh.utils.config import load_config
from pilot_diagnostics import Rows, ci, collator

FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60)
DRAWS = 4
MARGIN, MARGIN_LOG = 0.2, math.log(2.0)


def necessity_terms(z_orig, z_learned, z_random, y):
    """Per-example values of the three candidate objectives and the gradient norm wrt the learned counterfactual logits."""
    z_learned = z_learned.detach().clone().requires_grad_(True)
    z_random = z_random.detach().clone().requires_grad_(True)
    po = F.softmax(z_orig, -1).gather(1, y[:, None]).squeeze(1).detach()
    pl = F.softmax(z_learned, -1).gather(1, y[:, None]).squeeze(1)
    pr = F.softmax(z_random, -1).gather(1, y[:, None]).squeeze(1)
    ll = F.log_softmax(z_learned, -1).gather(1, y[:, None]).squeeze(1)
    lr = F.log_softmax(z_random, -1).gather(1, y[:, None]).squeeze(1)
    objectives = {"A_absolute": F.relu(MARGIN - po + pl), "B_prob_contrast": F.relu(MARGIN - (pr - pl)),
                  "C_logprob_contrast": F.relu(MARGIN_LOG - (lr - ll))}
    out = {}
    for name, loss in objectives.items():
        grad, = torch.autograd.grad(loss.sum(), z_learned, retain_graph=True)
        out[name] = {"loss": loss.detach(), "active": (loss.detach() > 0).float(), "grad_norm": grad.norm(dim=1)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_b")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-subsample", type=int, default=2000)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"])
    manifest = pd.read_csv(manifest_path(config))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id")
    labels = config["dataset"]["labels"]
    dev = torch.device("cuda")
    report = {}
    for pair in args.checkpoints:
        name, path = pair.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stage, m = int(payload["stage"]), payload["config"]["model"]
        tok = AutoTokenizer.from_pretrained(m["encoder_name"])
        model = CADVMH(len(labels), m["encoder_name"], selector_hidden_dim=m["selector_hidden_dim"], interaction_dim=m["interaction_dim"], dropout=m["dropout"])
        model.load_state_dict(payload["model"])
        model.eval().to(dev)
        mask_id, vocab, rho = tok.mask_token_id, model.encoder.model.config.vocab_size, m["selection_ratio"]
        gen = torch.Generator().manual_seed(0)
        report[name] = {"stage": stage}
        for split in ("validation", "train"):
            part = frame.loc[frame.partition == split]
            if split == "train":
                part = part.sample(args.train_subsample, random_state=0)
            loader = DataLoader(Rows(part, labels), batch_size=32, shuffle=False, collate_fn=collator(tok, m["max_length"]))
            R = {k: [] for k in ["y", "pred", "aux_pred", "po", "pl", "pr", "pr_word", "pr_var", "fin_o", "fin_rand", "fin_word", "fin_learned_edit",
                                 "allmask_pred", "A_active", "B_active", "C_active", "A_g", "B_g", "C_g", "learned_drop_ge", "rand_drop_ge", "pl_sat", "pr_sat"]}
            curve = {f: [] for f in FRACTIONS}
            with torch.no_grad():
                for raw in loader:
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
                    ids, y = b["input_ids"], b["labels"]
                    elig = b["attention_mask"].bool() & ~b["special_token_mask"].bool()

                    def run(x, st=stage, **kw):
                        return model(x, b["attention_mask"], b["special_token_mask"], mask_id, stage=st, selection_mode="top_rho", selection_ratio=rho, **kw)

                    def aux_logits(x):
                        return run(x, 1).original_logits

                    base = run(ids, enable_evidence=False)
                    primary = base.final_logits if stage >= 3 else base.original_logits
                    pred = primary.argmax(-1)
                    R["y"] += y.tolist(); R["pred"] += pred.tolist(); R["aux_pred"] += base.original_logits.argmax(-1).tolist()
                    pc = lambda lg: F.softmax(lg.float(), -1).gather(1, pred[:, None]).squeeze(1)  # noqa: E731
                    R["po"].append(pc(base.original_logits).cpu())
                    R["fin_o"].append(pc(primary).cpu())
                    am = aux_logits(torch.where(elig, torch.full_like(ids, mask_id), ids))
                    R["allmask_pred"] += am.argmax(-1).tolist()
                    word = lambda sel: torch.where(sel.bool(), torch.randint(4, vocab - 2, ids.shape, generator=gen).to(dev), ids)  # noqa: E731
                    rand_sel = [budgeted_topk(torch.rand(ids.shape, generator=gen).to(dev), elig, rho) for _ in range(DRAWS)]
                    prs = torch.stack([pc(aux_logits(torch.where(s.bool(), torch.full_like(ids, mask_id), ids))) for s in rand_sel])
                    R["pr"].append(prs.mean(0).cpu()); R["pr_var"].append(prs.std(0).cpu())
                    R["pr_word"].append(pc(aux_logits(word(rand_sel[0]))).cpu())
                    if stage >= 3:
                        R["fin_rand"].append(pc(run(torch.where(rand_sel[0].bool(), torch.full_like(ids, mask_id), ids), enable_evidence=False).final_logits).cpu())
                        R["fin_word"].append(pc(run(word(rand_sel[0]), enable_evidence=False).final_logits).cpu())
                    for f in FRACTIONS:
                        s = budgeted_topk(torch.rand(ids.shape, generator=gen).to(dev), elig, f)
                        x = torch.where(s.bool(), torch.full_like(ids, mask_id), ids)
                        lg = run(x, enable_evidence=False)
                        curve[f].append((pc(primary) - pc(lg.final_logits if stage >= 3 else lg.original_logits)).cpu())
                    if stage >= 2:
                        learned = base.selection_mask.bool() & elig
                        R["pl"].append(pc(base.counterfactual_logits).cpu())
                        if stage >= 3:
                            R["fin_learned_edit"].append(pc(run(torch.where(learned, torch.full_like(ids, mask_id), ids), enable_evidence=False).final_logits).cpu())
                        z_rand = aux_logits(torch.where(rand_sel[0].bool(), torch.full_like(ids, mask_id), ids))
                        # objectives are evaluated on the TRUE label, as in training
                        with torch.enable_grad():
                            terms = necessity_terms(base.original_logits, base.counterfactual_logits, z_rand, y)
                        for key, tag in (("A_absolute", "A"), ("B_prob_contrast", "B"), ("C_logprob_contrast", "C")):
                            R[f"{tag}_active"].append(terms[key]["active"].cpu()); R[f"{tag}_g"].append(terms[key]["grad_norm"].cpu())
                        pt = lambda lg: F.softmax(lg.float(), -1).gather(1, y[:, None]).squeeze(1)  # noqa: E731
                        R["learned_drop_ge"].append(((pt(base.original_logits) - pt(base.counterfactual_logits)) >= MARGIN).float().cpu())
                        R["rand_drop_ge"].append(((pt(base.original_logits) - pt(z_rand)) >= MARGIN).float().cpu())
                        R["pl_sat"].append(((pt(base.counterfactual_logits) < 0.01) | (pt(base.counterfactual_logits) > 0.99)).float().cpu())
                        R["pr_sat"].append(((pt(z_rand) < 0.01) | (pt(z_rand) > 0.99)).float().cpu())
            cat = lambda k: torch.cat(R[k]).numpy() if R[k] else None  # noqa: E731
            pred = np.asarray(R["pred"])
            entry = {"n": int(len(pred)), "accuracy": float((pred == np.asarray(R["y"])).mean()),
                     "fully_masked_predicted_ADHD_share": float((np.asarray(R["allmask_pred"]) == labels.index("adhd")).mean()),
                     "aux_vs_primary_prediction_agreement": float((np.asarray(R["aux_pred"]) == pred).mean())}
            for cls in (None, "ocd", "adhd"):
                sel = np.ones(len(pred), bool) if cls is None else pred == labels.index(cls)
                key = "all" if cls is None else f"predicted_{cls}"
                d = {"n": int(sel.sum()), "mask_fraction_drop_primary_head": {str(f): ci(torch.cat(curve[f]).numpy()[sel]) for f in FRACTIONS}}
                po, pr, prw = cat("po")[sel], cat("pr")[sel], cat("pr_word")[sel]
                d["aux_random_mask_drop"] = ci(po - pr)
                d["aux_random_word_replacement_drop"] = ci(po - prw)
                d["aux_random_drop_std_across_draws_mean"] = float(cat("pr_var")[sel].mean())
                if cat("fin_rand") is not None:
                    fo = cat("fin_o")[sel]
                    d["final_random_mask_drop"] = ci(fo - cat("fin_rand")[sel])
                    d["final_random_word_replacement_drop"] = ci(fo - cat("fin_word")[sel])
                    d["final_learned_mask_drop"] = ci(fo - cat("fin_learned_edit")[sel])
                    d["final_learned_minus_random_drop"] = ci(cat("fin_o")[sel] - cat("fin_learned_edit")[sel] - (fo - cat("fin_rand")[sel]))
                if stage >= 2:
                    d["aux_learned_mask_drop"] = ci(po - cat("pl")[sel])
                    d["aux_learned_minus_random_drop"] = ci(po - cat("pl")[sel] - (po - pr))
                    d["hinge_active_fraction"] = {t: float(cat(f"{t}_active")[sel].mean()) for t in "ABC"}
                    d["mean_grad_norm_wrt_learned_cf_logits"] = {t: float(cat(f"{t}_g")[sel].mean()) for t in "ABC"}
                    d["learned_drop_ge_0.2_fraction"] = float(cat("learned_drop_ge")[sel].mean())
                    d["random_drop_ge_0.2_fraction"] = float(cat("rand_drop_ge")[sel].mean())
                    d["cf_prob_saturated_fraction_learned_random"] = [float(cat("pl_sat")[sel].mean()), float(cat("pr_sat")[sel].mean())]
                entry[key] = d
            report[name][split] = entry
        del model
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(report, indent=2))
    print("written", args.output)


if __name__ == "__main__":
    main()
