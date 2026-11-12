#!/usr/bin/env python
"""Applies the PRE-REGISTERED acceptance criteria (docs/pilot2_preregistration.md) to pilot variants. Validation only.

Thresholds are constants below; they were committed before any variant was trained and must not be edited afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ---- PRE-REGISTERED THRESHOLDS (do not edit after training) ------------------------------------------------------
GAP_MAX = {"ocd": 0.20, "adhd": 0.10}      # C1: final-head 20%-random-mask drop minus A0's, per predicted class
ADV_MIN_LOWER = 0.02                       # C2: lower 95% CI of learned-minus-random final comprehensiveness, per predicted class
F1_NONINFERIORITY = -0.02                  # C4: lower 95% CI bound of (F1_variant - F1_A0)
ECE_MARGIN, NLL_MARGIN = 0.02, 0.10        # C5
RATIO_TOL, TIE_MAX = 0.02, 0.05            # C6
KEYWORD_DROP_MARGIN = 0.05                 # C7
PATTERN_ACC_MAX = 0.60                     # C8
# -------------------------------------------------------------------------------------------------------------------


def boot_ci(x, seed=0, boots=4000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (boots, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(run_dir: Path, mech_name: str, mech_file: Path):
    diag = json.loads((run_dir / "selector_diagnostics.json").read_text())
    pred = pd.read_csv(run_dir / "validation_predictions.csv")
    mech = json.loads(mech_file.read_text())[mech_name]["validation"]
    pattern = json.loads((run_dir / "pattern_leakage_probe.json").read_text()) if (run_dir / "pattern_leakage_probe.json").exists() else None
    return diag, pred, mech, pattern


def evaluate(name, v, base):
    diag, pred, mech, pattern = v
    bdiag, bpred, bmech, _ = base
    out = {"name": name, "criteria": {}, "values": {}}
    y = pred.label.values
    boots = np.random.default_rng(1).integers(0, len(y), (3000, len(y)))
    diffs = [f1_score(y[i], pred.pred.values[i], average="macro") - f1_score(y[i], bpred.pred.values[i], average="macro") for i in boots[:1500]]
    f1_lo = float(np.percentile(diffs, 2.5))
    out["values"]["macro_f1"] = diag["classification"]["macro_f1"]
    out["values"]["macro_f1_minus_A0"] = [float(np.mean(diffs)), f1_lo, float(np.percentile(diffs, 97.5))]
    for k in ("ece", "negative_log_likelihood", "multiclass_brier"):
        out["values"][k] = diag["classification"][k]
    C = out["criteria"]
    # C1 hypersensitivity gap per predicted class
    gaps = {}
    for cls in ("ocd", "adhd"):
        gaps[cls] = mech[f"predicted_{cls}"]["mask_fraction_drop_primary_head"]["0.2"]["mean"] - bmech[f"predicted_{cls}"]["mask_fraction_drop_primary_head"]["0.2"]["mean"]
    out["values"]["hypersensitivity_gap_vs_A0"] = gaps
    C["C1_mask_hypersensitivity_gap"] = all(gaps[c] <= GAP_MAX[c] for c in gaps)
    # C2/C3 per-class final-head advantages
    adv = {}
    for cls in ("ocd", "adhd", "all"):
        part = pred if cls == "all" else pred[pred.pred == cls]
        comp = part.final_predicted_comprehensiveness - part.final_predicted_random_comprehensiveness
        suff = part.final_predicted_sufficiency - part.final_predicted_random_sufficiency
        adv[cls] = {"n": int(len(part)), "comprehensiveness_learned_minus_random": boot_ci(comp), "sufficiency_learned_minus_random": boot_ci(suff),
                    "learned_comprehensiveness": float(part.final_predicted_comprehensiveness.mean()), "random_comprehensiveness": float(part.final_predicted_random_comprehensiveness.mean())}
    out["values"]["final_head_advantage"] = adv
    C["C2_comprehensiveness_advantage_both_classes"] = all(adv[c]["comprehensiveness_learned_minus_random"][1] > ADV_MIN_LOWER for c in ("ocd", "adhd"))
    C["C3_sufficiency_advantage_all_classes"] = all(adv[c]["sufficiency_learned_minus_random"][2] < 0 for c in ("ocd", "adhd", "all"))
    C["C4_f1_noninferior_to_A0"] = f1_lo >= F1_NONINFERIORITY
    C["C5_calibration_vs_A0"] = (diag["classification"]["ece"] <= bdiag["classification"]["ece"] + ECE_MARGIN and
                                 diag["classification"]["negative_log_likelihood"] <= bdiag["classification"]["negative_log_likelihood"] + NLL_MARGIN)
    sel = diag["selector"]
    aux_adv = {c: mech[f"predicted_{c}"]["aux_learned_minus_random_drop"]["ci95"][0] for c in ("ocd", "adhd")}
    out["values"]["aux_learned_minus_random_lower_ci"] = aux_adv
    C["C6_selector_health"] = (abs(sel["selection_ratio"]["mean"] - 0.2) <= RATIO_TOL and sel["boundary_tie_fraction"] < TIE_MAX and all(v > 0 for v in aux_adv.values()))
    kw = diag["keyword_leakage_probe"]; bkw = bdiag["keyword_leakage_probe"]
    drop = kw["accuracy_original_on_those"] - kw["accuracy_with_keyword_masked_on_those"]
    bdrop = bkw["accuracy_original_on_those"] - bkw["accuracy_with_keyword_masked_on_those"]
    out["values"]["condition_name_masking_accuracy_drop"] = {"variant": drop, "A0": bdrop}
    C["C7_condition_name_sensitivity"] = drop <= bdrop + KEYWORD_DROP_MARGIN
    if pattern:
        out["values"]["pattern_leakage_accuracy"] = {k: pattern[k]["aux_head_accuracy"] for k in ("random_fill", "foreign_fill")}
        C["C8_no_position_pattern_leakage"] = max(pattern["random_fill"]["aux_head_accuracy"], pattern["foreign_fill"]["aux_head_accuracy"]) <= PATTERN_ACC_MAX
    else:
        C["C8_no_position_pattern_leakage"] = None
    out["descriptive"] = {"selector_logit_quantiles": sel["logit_quantiles_q01_q25_q50_q75_q99"], "selector_within_example_logit_std": sel["logit_within_example_std_mean"],
                          "selector_saturation_abs_gt_7": sel["saturation_fraction_abs_logit_gt_7"], "selection_ratio": sel["selection_ratio"]["mean"],
                          "rationale_stability_jaccard": diag.get("rationale_stability_jaccard_dropout", {}).get("mean"),
                          "contiguity_runs_per_token": diag.get("mask_contiguity_runs_per_selected_token"),
                          "condition_name_tokens_selected_fraction": diag["selector_keyword_attention"]["fraction_of_keyword_tokens_selected"],
                          "aux_vs_final_agreement": mech["aux_vs_primary_prediction_agreement"],
                          "final_random_mask_drop_by_class": {c: mech[f"predicted_{c}"].get("final_random_mask_drop", {}).get("mean") for c in ("ocd", "adhd")},
                          "aux_random_mask_drop_by_class": {c: mech[f"predicted_{c}"]["aux_random_mask_drop"]["mean"] for c in ("ocd", "adhd")}}
    out["all_criteria_pass"] = all(v is True for v in C.values())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--baseline-mechanism", required=True, help="json file")
    parser.add_argument("--baseline-name", default="A0")
    parser.add_argument("--variant", action="append", required=True, help="name=diagnostics_dir=mechanism_json=mechanism_key")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load(Path(args.baseline_dir), args.baseline_name, Path(args.baseline_mechanism))
    results = []
    for spec in args.variant:
        name, d, mj, key = spec.split("=")
        results.append(evaluate(name, load(Path(d), key, Path(mj)), base))
    Path(args.output).write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"\n{r['name']}: ALL PASS = {r['all_criteria_pass']}")
        for k, v in r["criteria"].items():
            print(f"   {k:48s} {v}")


if __name__ == "__main__":
    main()
