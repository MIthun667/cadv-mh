#!/usr/bin/env python
"""Validation-only selector / explanation / shortcut diagnostics for a pilot checkpoint.

Never touches the test split (the partition argument only accepts "validation").
Works for stage-1 (A0), stage-2 (selector only) and stage-3 (full) checkpoints.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from _common import ROOT, json_write, manifest_path, resolve_dataset_path
from cadvmh.data.loader import load_dataset
from cadvmh.data.tokenization import special_token_mask
from cadvmh.evaluation.faithfulness import final_mechanism_faithfulness
from cadvmh.evaluation.metrics import classification_metrics
from cadvmh.models import CADVMH
from cadvmh.models.masking import boundary_tie_fraction, budgeted_topk
from cadvmh.utils.config import load_config

KEYWORD = re.compile(r"\b(adhd|add|ocd)\b", re.IGNORECASE)  # explicit condition names (label leakage probe)
BUDGETS = (0.05, 0.10, 0.20, 0.40, 0.60)


class Rows(Dataset):
    def __init__(self, frame, labels):
        self.frame, self.map = frame.reset_index(drop=True), {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        return r.text, self.map[r.label], r.sample_id


def collator(tokenizer, max_length):
    def collate(rows):
        texts = [r[0] for r in rows]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        keyword = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
        for b, text in enumerate(texts):
            spans = [m.span() for m in KEYWORD.finditer(text)]
            if spans:
                for j, (left, right) in enumerate(offsets[b].tolist()):
                    if right > left and any(left < e and right > s for s, e in spans):
                        keyword[b, j] = True
        enc["special_token_mask"] = special_token_mask(enc["input_ids"], tokenizer, enc["attention_mask"])
        enc["keyword_mask"] = keyword
        enc["labels"] = torch.tensor([r[1] for r in rows])
        enc["sample_ids"] = [r[2] for r in rows]
        return enc
    return collate


def ci(values, seed=0, boots=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (boots, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": int(len(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", default="validation", choices=["validation"])  # test split deliberately unavailable
    parser.add_argument("--fusion-mode", default="full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    out_dir = ROOT / args.output_dir if not args.output_dir.startswith("/") else __import__("pathlib").Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"])
    manifest = pd.read_csv(manifest_path(config))
    frame = frame.merge(manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one")
    frame = frame.loc[frame.partition == args.partition]
    if args.max_examples:
        frame = frame.head(args.max_examples)
    labels = config["dataset"]["labels"]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage, mcfg = int(payload["stage"]), payload["config"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["encoder_name"], use_fast=True)
    model = CADVMH(len(labels), mcfg["encoder_name"], selector_hidden_dim=mcfg["selector_hidden_dim"], interaction_dim=mcfg["interaction_dim"], dropout=mcfg["dropout"])
    model.load_state_dict(payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    rho, mask_id = mcfg["selection_ratio"], tokenizer.mask_token_id
    loader = DataLoader(Rows(frame, labels), batch_size=args.batch_size, shuffle=False, collate_fn=collator(tokenizer, mcfg["max_length"]))
    gen = torch.Generator().manual_seed(0)
    has_selector = stage >= 2

    def fwd(batch, ids=None, **kw):
        kw.setdefault("selection_mode", "top_rho")
        kw.setdefault("selection_ratio", rho)
        kw.setdefault("fusion_mode", args.fusion_mode)
        return model(ids if ids is not None else batch["input_ids"], batch["attention_mask"], batch["special_token_mask"], mask_id, stage=kw.pop("stage", stage), **kw)

    def true_p(logits, y):
        return torch.softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1)

    rec = {k: [] for k in ["label", "sample_id", "p_final", "p_aux", "aux_o", "aux_cf", "aux_ev", "rnd_cf", "rnd_ev", "first_cf", "first_ev",
                           "last_cf", "last_ev", "ratio", "n_elig", "logit_mean", "logit_std", "sat", "tie", "kw_tokens", "kw_selected",
                           "allmask_p_true", "allmask_pred", "kw_masked_correct", "kw_orig_correct", "has_kw", "rand_mask_comp_stage1"]}
    curves = {f"{kind}_{b}": [] for kind in ("learned", "random") for b in BUDGETS}
    final_faith = {}
    sel_rel_pos, elig_rel_pos = [], []
    all_logits = []

    with torch.no_grad():
        for raw in tqdm(loader, desc=f"diagnostics {args.checkpoint}"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            y = batch["labels"]
            eligible = batch["attention_mask"].bool() & ~batch["special_token_mask"].bool()
            ids = batch["input_ids"]
            out = fwd(batch, enable_evidence=has_selector)
            rec["label"] += y.tolist(); rec["sample_id"] += batch["sample_ids"]
            primary = out.final_logits if stage >= 3 else out.original_logits
            rec["p_final"].append(torch.softmax(primary.float(), -1).cpu().numpy())
            rec["p_aux"].append(torch.softmax(out.original_logits.float(), -1).cpu().numpy())
            rec["aux_o"] += true_p(out.original_logits, y).tolist()
            # fully masked input and keyword ablation use the auxiliary/primary heads on edited ids
            allmask_ids = torch.where(eligible, torch.full_like(ids, mask_id), ids)
            am = fwd(batch, allmask_ids, stage=1, enable_evidence=False)
            rec["allmask_p_true"] += true_p(am.original_logits, y).tolist(); rec["allmask_pred"] += am.original_logits.argmax(-1).tolist()
            kw = batch["keyword_mask"] & eligible
            kw_ids = torch.where(kw, torch.full_like(ids, mask_id), ids)
            km = fwd(batch, kw_ids, enable_evidence=False)
            kp = (km.final_logits if stage >= 3 else km.original_logits).argmax(-1)
            rec["kw_masked_correct"] += kp.eq(y).tolist(); rec["kw_orig_correct"] += primary.argmax(-1).eq(y).tolist(); rec["has_kw"] += kw.any(1).tolist()
            if not has_selector:  # stage-1 / A0: random-mask sensitivity of the plain classifier
                rnd = budgeted_topk(torch.rand(eligible.shape, generator=gen).to(device), eligible, rho).bool()
                masked = fwd(batch, torch.where(rnd, torch.full_like(ids, mask_id), ids), stage=1, enable_evidence=False)
                rec["rand_mask_comp_stage1"] += (true_p(out.original_logits, y) - true_p(masked.original_logits, y)).tolist()
                continue
            sel = out.selection_mask.bool() & eligible
            rec["aux_cf"] += true_p(out.counterfactual_logits, y).tolist(); rec["aux_ev"] += true_p(out.evidence_logits, y).tolist()
            logits = out.selector_logits.float()
            n = eligible.sum(1)
            rec["n_elig"] += n.tolist(); rec["ratio"] += (sel.sum(1) / n.clamp_min(1)).tolist()
            ml = logits.masked_fill(~eligible, 0.0)
            mean = ml.sum(1) / n.clamp_min(1)
            rec["logit_mean"] += mean.tolist()
            rec["logit_std"] += (((logits - mean[:, None]) ** 2).masked_fill(~eligible, 0).sum(1) / n.clamp_min(1)).sqrt().tolist()
            rec["sat"] += (((logits.abs() > 7) & eligible).sum(1) / n.clamp_min(1)).tolist()
            for b in range(logits.size(0)):
                rec["tie"].append(boundary_tie_fraction(logits[b:b + 1], eligible[b:b + 1], rho))
                idx = torch.where(eligible[b])[0]
                if len(idx) > 1:
                    elig_rel_pos += (torch.arange(len(idx)) / (len(idx) - 1)).tolist()
                    sel_rel_pos += [i / (len(idx) - 1) for i, t in enumerate(sel[b, idx].tolist()) if t]
            all_logits.append(logits[eligible].cpu())
            rec["kw_tokens"] += kw.sum(1).tolist(); rec["kw_selected"] += (kw & sel).sum(1).tolist()
            # controls at the same budget: random and positional rationales
            pos = torch.arange(logits.size(1), device=device).float().expand_as(logits)
            for tag, override in (("rnd", torch.rand(logits.shape, generator=gen).to(device)), ("first", -pos), ("last", pos)):
                c = fwd(batch, enable_evidence=True, selector_logits_override=override)
                rec[f"{tag}_cf"] += true_p(c.counterfactual_logits, y).tolist(); rec[f"{tag}_ev"] += true_p(c.evidence_logits, y).tolist()
            # comprehensiveness vs budget, learned vs random
            rand_scores = torch.rand(logits.shape, generator=gen).to(device)
            for b in BUDGETS:
                l = fwd(batch, selection_ratio=b, enable_evidence=False)
                r = fwd(batch, selection_ratio=b, enable_evidence=False, selector_logits_override=rand_scores)
                curves[f"learned_{b}"] += (true_p(out.original_logits, y) - true_p(l.counterfactual_logits, y)).tolist()
                curves[f"random_{b}"] += (true_p(out.original_logits, y) - true_p(r.counterfactual_logits, y)).tolist()
            if stage >= 3:
                f = final_mechanism_faithfulness(model, {k: batch[k] for k in ("input_ids", "attention_mask", "special_token_mask", "labels")}, mask_id, rho, args.fusion_mode, gen)
                for k, v in f.items():
                    final_faith.setdefault(k, []).extend(v.cpu().tolist())

    probs, targets = np.concatenate(rec["p_final"]), np.asarray(rec["label"])
    aux_probs = np.concatenate(rec["p_aux"])
    report = {"checkpoint": args.checkpoint, "checkpoint_stage": stage, "partition": args.partition, "examples": int(len(targets)), "selection_ratio_target": rho,
              "primary_head": "final" if stage >= 3 else "auxiliary",
              "classification": {k: v for k, v in classification_metrics(targets, probs, labels).items() if k not in ("classification_report",)},
              "auxiliary_head_classification": {k: classification_metrics(targets, aux_probs, labels)[k] for k in ("accuracy", "macro_f1", "ece", "negative_log_likelihood")}}
    A = lambda k: np.asarray(rec[k])  # noqa: E731
    pred = probs.argmax(1)
    am_pred = A("allmask_pred")
    report["fully_masked_input"] = {"mean_p_true_class": float(A("allmask_p_true").mean()), "predicted_class_share": {labels[i]: float((am_pred == i).mean()) for i in range(len(labels))}}
    has = A("has_kw").astype(bool)
    report["keyword_leakage_probe"] = {
        "fraction_with_explicit_condition_name": float(has.mean()),
        "accuracy_original_on_those": float(A("kw_orig_correct")[has].mean()) if has.any() else None,
        "accuracy_with_keyword_masked_on_those": float(A("kw_masked_correct")[has].mean()) if has.any() else None,
        "accuracy_on_examples_without_keyword": float(A("kw_orig_correct")[~has].mean()) if (~has).any() else None}
    if not has_selector:
        report["random_mask_comprehensiveness_plain_classifier"] = ci(A("rand_mask_comp_stage1"))
    else:
        comp, suff = A("aux_o") - A("aux_cf"), A("aux_o") - A("aux_ev")
        score = A("aux_ev") - A("aux_cf")
        L = torch.cat(all_logits)
        report["selector"] = {
            "selection_ratio": ci(A("ratio")), "eligible_tokens_per_example_mean": float(A("n_elig").mean()),
            "logit_quantiles_q01_q25_q50_q75_q99": [float(x) for x in np.quantile(L.numpy(), [.01, .25, .5, .75, .99])],
            "logit_within_example_std_mean": float(A("logit_std").mean()), "logit_example_mean_mean": float(A("logit_mean").mean()),
            "saturation_fraction_abs_logit_gt_7": float(A("sat").mean()),
            "fraction_tokens_sigmoid_gt_0.999_or_lt_0.001": float(((L.abs() > 6.9)).float().mean()),
            "boundary_tie_fraction": float(A("tie").mean()),
            "selected_relative_position_histogram_5bins": (np.histogram(sel_rel_pos, bins=5, range=(0, 1))[0] / max(1, len(sel_rel_pos))).tolist(),
            "eligible_relative_position_histogram_5bins": (np.histogram(elig_rel_pos, bins=5, range=(0, 1))[0] / max(1, len(elig_rel_pos))).tolist(),
            "selected_relative_position_mean": float(np.mean(sel_rel_pos))}
        report["aux_head_mechanism"] = {
            "p_true_original": ci(A("aux_o")), "p_true_counterfactual": ci(A("aux_cf")), "p_true_evidence": ci(A("aux_ev")),
            "necessity_comprehensiveness": ci(comp), "sufficiency_gap": ci(suff), "evidence_minus_counterfactual_score": ci(score),
            "fraction_cf_prob_drop_ge_0.2": float((comp >= 0.2).mean())}
        controls = {}
        for tag in ("rnd", "first", "last"):
            c, s = A("aux_o") - A(f"{tag}_cf"), A("aux_o") - A(f"{tag}_ev")
            controls[tag] = {"comprehensiveness": ci(c), "sufficiency_gap": ci(s), "score": ci(A(f"{tag}_ev") - A(f"{tag}_cf")),
                             "learned_minus_control_comprehensiveness": ci(comp - c), "learned_minus_control_score": ci(score - (A(f"{tag}_ev") - A(f"{tag}_cf")))}
        report["controls_same_budget"] = controls
        report["comprehensiveness_vs_budget"] = {str(b): {"learned": float(np.mean(curves[f"learned_{b}"])), "random": float(np.mean(curves[f"random_{b}"])),
                                                         "diff_ci": ci(np.asarray(curves[f"learned_{b}"]) - np.asarray(curves[f"random_{b}"]))} for b in BUDGETS}
        kwt, kws = A("kw_tokens"), A("kw_selected")
        report["selector_keyword_attention"] = {"keyword_tokens_total": int(kwt.sum()), "selected_keyword_tokens": int(kws.sum()),
                                                "fraction_of_keyword_tokens_selected": float(kws.sum() / max(1, kwt.sum())), "expected_if_random": float(A("ratio").mean())}
        if final_faith:
            F = {k: np.asarray(v) for k, v in final_faith.items()}
            report["final_head_faithfulness"] = {
                tag: {m: ci(F[f"{tag}_{m}"]) for m in ("comprehensiveness", "sufficiency", "random_comprehensiveness", "random_sufficiency")} |
                     {"learned_minus_random_comprehensiveness": ci(F[f"{tag}_comprehensiveness"] - F[f"{tag}_random_comprehensiveness"]),
                      "learned_minus_random_sufficiency": ci(F[f"{tag}_sufficiency"] - F[f"{tag}_random_sufficiency"])} for tag in ("predicted", "gold")}
            report["final_head_faithfulness"]["selection_ratio"] = float(F["selection_ratio"].mean())
    json_write(out_dir / "selector_diagnostics.json", report)
    table = pd.DataFrame({"sample_id": rec["sample_id"], "label": [labels[i] for i in targets], "pred": [labels[i] for i in pred], **{f"p_{l}": probs[:, i] for i, l in enumerate(labels)},
                          "aux_p_true_original": rec["aux_o"]})
    if has_selector:
        table["aux_p_true_counterfactual"], table["aux_p_true_evidence"], table["selection_ratio"] = rec["aux_cf"], rec["aux_ev"], rec["ratio"]
        for tag in ("rnd", "first", "last"):
            table[f"aux_p_true_cf_{tag}"] = rec[f"{tag}_cf"]
        for k, v in final_faith.items():
            table[f"final_{k}"] = v
    table.to_csv(out_dir / "validation_predictions.csv", index=False)
    print(f"Diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
