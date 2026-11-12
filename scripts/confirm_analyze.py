#!/usr/bin/env python
"""Confirmatory analysis for the Dataset B multi-seed study. Validation only.

Decision rules are the constants below; they implement docs/confirmatory_protocol.md section 6 and were committed
before any new-seed result was examined. Do not edit after results are seen.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

# ---- FROZEN DECISION CONSTANTS ----------------------------------------------------------------------------------
SEEDS = (42, 43, 44, 45, 46)
ALPHA_ONE_SIDED = 0.05
F1_DELTA, F1_SEED_FLOOR = 0.015, 0.03          # N
E1_MEAN_MIN = 0.03                              # E1 (comprehensiveness advantage, each predicted class)
E2_MEAN_MAX = -0.02                             # E2 (sufficiency advantage)
R_MIN_SEEDS = 4                                 # R
ECE_MARGIN, NLL_MARGIN = 0.02, 0.10             # K
GAP_MEAN_MAX = {"ocd": 0.20, "adhd": 0.10}      # H
GAP_SEED_MAX = {"ocd": 0.30, "adhd": 0.15}      # H
RATIO_TOL, TIE_MAX = 0.02, 0.05                 # S
SATURATION_FLAG = 0.95                          # S (descriptive flag)
KEYWORD_MARGIN, PATTERN_MAX = 0.05, 0.60        # L
# -----------------------------------------------------------------------------------------------------------------

MODELS = ("A0", "B", "V1")
CLASSES = ("ocd", "adhd")


def boot_ci(x, seed=0, boots=4000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (boots, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def seed_stats(d):
    """Seed-level inference for paired per-seed values d (unit = training seed)."""
    d = np.asarray(d, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = float(stats.t.ppf(1 - ALPHA_ONE_SIDED, n - 1))
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    perm = (flips * d).mean(1)
    return {"values": d.tolist(), "mean": mean, "sd": sd, "lower_1s95": mean - t * se, "upper_1s95": mean + t * se,
            "ci90_two_sided": [mean - t * se, mean + t * se], "perm_p_mean_gt0": float((perm >= mean - 1e-12).mean()),
            "perm_p_mean_lt0": float((perm <= mean + 1e-12).mean())}


def load(run: Path, name: str):
    diag = json.loads((run / f"diagnostics_{name}" / "selector_diagnostics.json").read_text())
    pred = pd.read_csv(run / f"diagnostics_{name}" / "validation_predictions.csv")
    mech = json.loads((run / f"mechanism_{name}.json").read_text())[name]["validation"]
    pat_p = run / f"diagnostics_{name}" / "pattern_leakage_probe.json"
    return {"diag": diag, "pred": pred, "mech": mech, "pattern": json.loads(pat_p.read_text()) if pat_p.exists() else None}


def per_seed_metrics(m):
    d, p, mech = m["diag"], m["pred"], m["mech"]
    c = d["classification"]
    out = {"macro_f1": c["macro_f1"], "accuracy": c["accuracy"], "nll": c["negative_log_likelihood"], "ece": c["ece"], "brier": c["multiclass_brier"],
           "precision_ocd": c["per_class"]["ocd"]["precision"], "recall_ocd": c["per_class"]["ocd"]["recall"],
           "precision_adhd": c["per_class"]["adhd"]["precision"], "recall_adhd": c["per_class"]["adhd"]["recall"],
           "fully_masked_adhd_share": mech["fully_masked_predicted_ADHD_share"],
           "gap_source_drop_ocd": mech["predicted_ocd"]["mask_fraction_drop_primary_head"]["0.2"]["mean"],
           "gap_source_drop_adhd": mech["predicted_adhd"]["mask_fraction_drop_primary_head"]["0.2"]["mean"]}
    kw = d["keyword_leakage_probe"]
    out["condition_name_drop"] = kw["accuracy_original_on_those"] - kw["accuracy_with_keyword_masked_on_those"]
    if "final_predicted_comprehensiveness" in p:
        for cls in CLASSES:
            part = p[p.pred == cls]
            comp = part.final_predicted_comprehensiveness - part.final_predicted_random_comprehensiveness
            suff = part.final_predicted_sufficiency - part.final_predicted_random_sufficiency
            out[f"comp_adv_{cls}"], out[f"comp_adv_{cls}_boot"] = float(comp.mean()), boot_ci(comp)
            out[f"suff_adv_{cls}"], out[f"suff_adv_{cls}_boot"] = float(suff.mean()), boot_ci(suff)
            out[f"learned_comp_{cls}"] = float(part.final_predicted_comprehensiveness.mean())
            out[f"random_comp_{cls}"] = float(part.final_predicted_random_comprehensiveness.mean())
            out[f"learned_suff_{cls}"] = float(part.final_predicted_sufficiency.mean())
            out[f"random_suff_{cls}"] = float(part.final_predicted_random_sufficiency.mean())
            out[f"n_pred_{cls}"] = int(len(part))
            aux = mech[f"predicted_{cls}"]["aux_learned_minus_random_drop"]["ci95"]
            out[f"aux_adv_lower_{cls}"] = aux[0]
        sel = d["selector"]
        out.update({"selection_ratio": sel["selection_ratio"]["mean"], "tie_fraction": sel["boundary_tie_fraction"], "saturation": sel["saturation_fraction_abs_logit_gt_7"],
                    "logit_quantiles_q01_q25_q50_q75_q99": sel["logit_quantiles_q01_q25_q50_q75_q99"], "logit_within_example_std": sel["logit_within_example_std_mean"],
                    "stability_jaccard": d["rationale_stability_jaccard_dropout"]["mean"], "contiguity": d["mask_contiguity_runs_per_selected_token"]["learned"],
                    "contiguity_random": d["mask_contiguity_runs_per_selected_token"]["random"],
                    "kw_tokens_selected_fraction": d["selector_keyword_attention"]["fraction_of_keyword_tokens_selected"]})
        if m["pattern"]:
            out["pattern_random_fill"] = m["pattern"]["random_fill"]["aux_head_accuracy"]
            out["pattern_foreign_fill"] = m["pattern"]["foreign_fill"]["aux_head_accuracy"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments_v2/confirm/dataset_b")
    ap.add_argument("--out-json", default="experiments_v2/confirm/dataset_b/confirm_results.json")
    ap.add_argument("--out-md", default="experiments_v2/confirm/dataset_b/confirm_tables.md")
    ap.add_argument("--figdir", default="docs/figures")
    args = ap.parse_args()
    root = Path(args.root)
    runs = {s: {n: load(root / f"seed_{s}", n) for n in MODELS} for s in SEEDS}
    M = {s: {n: per_seed_metrics(runs[s][n]) for n in MODELS} for s in SEEDS}
    # per-seed paired F1 bootstrap (example-level), B vs A0 and V1 vs A0
    f1_boot = {}
    for s in SEEDS:
        y = runs[s]["A0"]["pred"].label.values
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(y), (1500, len(y)))
        base = runs[s]["A0"]["pred"].pred.values
        for n in ("B", "V1"):
            assert (runs[s][n]["pred"].sample_id.values == runs[s]["A0"]["pred"].sample_id.values).all()
            v = runs[s][n]["pred"].pred.values
            d = [f1_score(y[i], v[i], average="macro") - f1_score(y[i], base[i], average="macro") for i in idx]
            f1_boot[(s, n)] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    def vec(model, key):
        return np.array([M[s][model][key] for s in SEEDS], float)

    def contrast(a, b, key):
        return vec(a, key) - vec(b, key)

    keys = ["macro_f1", "accuracy", "nll", "ece", "brier", "precision_ocd", "recall_ocd", "precision_adhd", "recall_adhd", "condition_name_drop", "fully_masked_adhd_share",
            "gap_source_drop_ocd", "gap_source_drop_adhd"]
    sel_keys = ["comp_adv_ocd", "comp_adv_adhd", "suff_adv_ocd", "suff_adv_adhd", "learned_comp_ocd", "random_comp_ocd", "learned_comp_adhd", "random_comp_adhd",
                "selection_ratio", "tie_fraction", "saturation", "logit_within_example_std", "stability_jaccard", "contiguity", "kw_tokens_selected_fraction",
                "aux_adv_lower_ocd", "aux_adv_lower_adhd", "pattern_random_fill", "pattern_foreign_fill"]
    res = {"per_seed": {str(s): {n: {k: v for k, v in M[s][n].items()} for n in MODELS} for s in SEEDS}, "f1_paired_bootstrap_example_level": {f"{s}_{n}": v for (s, n), v in f1_boot.items()},
           "model_summary": {}, "contrasts": {}}
    for n in MODELS:
        res["model_summary"][n] = {k: seed_stats(vec(n, k)) for k in keys + (sel_keys if n != "A0" else [])}
    for a, b in (("B", "A0"), ("V1", "A0"), ("B", "V1")):
        res["contrasts"][f"{a}-{b}"] = {k: seed_stats(contrast(a, b, k)) for k in keys + (sel_keys if "A0" not in (a, b) else [])
                                       if not (b == "A0" and k in sel_keys)}
    gaps = {n: {c: vec(n, f"gap_source_drop_{c}") - vec("A0", f"gap_source_drop_{c}") for c in CLASSES} for n in ("B", "V1")}
    res["hypersensitivity_gap_vs_A0"] = {n: {c: seed_stats(gaps[n][c]) for c in CLASSES} for n in gaps}

    # ---- decision rules ------------------------------------------------------------------------------------
    dF1 = contrast("B", "A0", "macro_f1")
    N = seed_stats(dF1)
    crit, detail = {}, {}
    crit["N_f1_noninferiority"] = bool(N["lower_1s95"] > -F1_DELTA and dF1.min() >= -F1_SEED_FLOOR)
    detail["N_f1_noninferiority"] = {"delta_f1": N, "margin": F1_DELTA, "seed_floor": F1_SEED_FLOOR}
    E1, E2 = {}, {}
    for c in CLASSES:
        E1[c] = seed_stats(vec("B", f"comp_adv_{c}")); E2[c] = seed_stats(vec("B", f"suff_adv_{c}"))
    crit["E1_comprehensiveness_advantage"] = all(E1[c]["mean"] >= E1_MEAN_MIN and E1[c]["lower_1s95"] > 0 for c in CLASSES)
    crit["E2_sufficiency_advantage"] = all(E2[c]["mean"] <= E2_MEAN_MAX and E2[c]["upper_1s95"] < 0 for c in CLASSES)
    detail["E1_comprehensiveness_advantage"], detail["E2_sufficiency_advantage"] = E1, E2
    ok_seed = {}
    for s in SEEDS:
        m = M[s]["B"]
        ok_seed[s] = all(m[f"comp_adv_{c}_boot"][1] > 0 and m[f"suff_adv_{c}_boot"][2] < 0 for c in CLASSES)
    crit["R_replication"] = sum(ok_seed.values()) >= R_MIN_SEEDS
    detail["R_replication"] = {"seeds_passing": {str(s): v for s, v in ok_seed.items()}, "n_passing": int(sum(ok_seed.values())), "required": R_MIN_SEEDS}
    Mc = seed_stats(contrast("B", "V1", "comp_adv_adhd"))
    crit["M_necessity_adds_value_adhd"] = bool(Mc["mean"] > 0 and Mc["lower_1s95"] > 0)
    detail["M_necessity_adds_value_adhd"] = Mc
    Ke, Kn = seed_stats(contrast("B", "A0", "ece")), seed_stats(contrast("B", "A0", "nll"))
    crit["K_calibration"] = bool(Ke["upper_1s95"] <= ECE_MARGIN and Kn["upper_1s95"] <= NLL_MARGIN)
    detail["K_calibration"] = {"ece_diff": Ke, "nll_diff": Kn}
    crit["H_mask_hypersensitivity"] = all(gaps["B"][c].mean() <= GAP_MEAN_MAX[c] and gaps["B"][c].max() <= GAP_SEED_MAX[c] for c in CLASSES)
    detail["H_mask_hypersensitivity"] = {c: seed_stats(gaps["B"][c]) for c in CLASSES}
    s_ok = {}
    for s in SEEDS:
        m = M[s]["B"]
        s_ok[s] = bool(abs(m["selection_ratio"] - 0.2) <= RATIO_TOL and m["tie_fraction"] < TIE_MAX and all(m[f"aux_adv_lower_{c}"] > 0 for c in CLASSES))
    crit["S_selector_health"] = all(s_ok.values())
    detail["S_selector_health"] = {"per_seed": {str(s): v for s, v in s_ok.items()}, "saturation_flag_gt_0.95": {str(s): M[s]["B"]["saturation"] > SATURATION_FLAG for s in SEEDS}}
    Kw = seed_stats(contrast("B", "A0", "condition_name_drop"))
    pat = max(max(M[s]["B"]["pattern_random_fill"], M[s]["B"]["pattern_foreign_fill"]) for s in SEEDS)
    crit["L_shortcuts"] = bool(Kw["upper_1s95"] <= KEYWORD_MARGIN and pat <= PATTERN_MAX)
    detail["L_shortcuts"] = {"condition_name_drop_diff": Kw, "max_pattern_leakage_accuracy": pat}
    res["criteria"], res["criteria_detail"] = crit, detail
    res["verdict"] = "GO" if all(crit.values()) else "NO-GO"
    res["class_balanced_advantage_replicates"] = bool(crit["E1_comprehensiveness_advantage"] and crit["E2_sufficiency_advantage"] and crit["R_replication"])
    Path(args.out_json).write_text(json.dumps(res, indent=2, default=float))

    # ---- markdown tables -----------------------------------------------------------------------------------
    L = []
    def row(label, v, fmt="{:.4f}"):
        return f"| {label} | " + " | ".join(fmt.format(x) for x in v) + f" | {fmt.format(np.mean(v))} | {fmt.format(np.std(v, ddof=1))} |"
    hdr = "| | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean | SD |\n|---|" + "---|" * (len(SEEDS) + 2)
    for n in MODELS:
        L += [f"\n### {n} per-seed (validation)\n", hdr]
        for k in keys + (sel_keys if n != "A0" else []):
            L.append(row(k, vec(n, k)))
    L += ["\n### Seed-level paired contrasts (mean, SD, 90% two-sided t-interval = one-sided 95% bounds, sign-flip p)\n", "| contrast | metric | per-seed | mean | SD | 90% t-interval | p(mean>0) | p(mean<0) |", "|---|---|---|---|---|---|---|---|"]
    for cn, cd in res["contrasts"].items():
        for k, st in cd.items():
            L.append(f"| {cn} | {k} | {', '.join(f'{x:+.4f}' for x in st['values'])} | {st['mean']:+.4f} | {st['sd']:.4f} | [{st['ci90_two_sided'][0]:+.4f}, {st['ci90_two_sided'][1]:+.4f}] | {st['perm_p_mean_gt0']:.3f} | {st['perm_p_mean_lt0']:.3f} |")
    L += ["\n### Per-seed example-level bootstrap intervals (95%) for B, final-head learned minus random\n", "| seed | comp OCD | comp ADHD | suff OCD | suff ADHD | all four exclude 0 correctly |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        m = M[s]["B"]
        f = lambda b: f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]"  # noqa: E731
        L.append(f"| {s} | {f(m['comp_adv_ocd_boot'])} | {f(m['comp_adv_adhd_boot'])} | {f(m['suff_adv_ocd_boot'])} | {f(m['suff_adv_adhd_boot'])} | {ok_seed[s]} |")
    L += ["\n### Paired example-level bootstrap of Macro-F1 difference vs A0 (per seed)\n", "| seed | B - A0 | V1 - A0 |", "|---|---|---|"]
    for s in SEEDS:
        g = lambda b: f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]"  # noqa: E731
        L.append(f"| {s} | {g(f1_boot[(s, 'B')])} | {g(f1_boot[(s, 'V1')])} |")
    L += ["\n### Decision rules\n", "| rule | pass |", "|---|---|"] + [f"| {k} | {v} |" for k, v in crit.items()] + [f"\n**Verdict: {res['verdict']}**; class-balanced advantage replicates: {res['class_balanced_advantage_replicates']}"]
    Path(args.out_md).write_text("\n".join(L))

    # ---- figures ---------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"A0": "#666666", "B": "#1f77b4", "V1": "#d95f02"}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, (k, t) in zip(ax[:2], (("comp_adv_ocd", "Comprehensiveness advantage, predicted OCD"), ("comp_adv_adhd", "Comprehensiveness advantage, predicted ADHD"))):
        for j, n in enumerate(("B", "V1")):
            a.scatter([j] * len(SEEDS), vec(n, k), color=colors[n], zorder=3); a.hlines(vec(n, k).mean(), j - .25, j + .25, color=colors[n])
        a.axhline(0, color="k", lw=.6); a.axhline(E1_MEAN_MIN, color="g", ls="--", lw=.8); a.set_xticks([0, 1]); a.set_xticklabels(["B", "V1"]); a.set_title(t, fontsize=9)
    for a, (k, t) in zip(ax[2:], (("suff_adv_ocd", "Sufficiency advantage, OCD"), ("suff_adv_adhd", "Sufficiency advantage, ADHD"))):
        for j, n in enumerate(("B", "V1")):
            a.scatter([j] * len(SEEDS), vec(n, k), color=colors[n], zorder=3); a.hlines(vec(n, k).mean(), j - .25, j + .25, color=colors[n])
        a.axhline(0, color="k", lw=.6); a.axhline(E2_MEAN_MAX, color="g", ls="--", lw=.8); a.set_xticks([0, 1]); a.set_xticklabels(["B", "V1"]); a.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "confirm_rationale_advantage.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, (k, t) in zip(ax, (("macro_f1", "Macro-F1"), ("ece", "ECE"), ("nll", "NLL"), ("gap_source_drop_ocd", "20% random-mask drop, predicted OCD"))):
        for j, n in enumerate(MODELS):
            a.scatter([j] * len(SEEDS), vec(n, k), color=colors[n], zorder=3); a.hlines(vec(n, k).mean(), j - .25, j + .25, color=colors[n])
        a.set_xticks(range(3)); a.set_xticklabels(MODELS); a.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "confirm_performance_calibration.png", dpi=140); plt.close(fig)
    print(json.dumps({"verdict": res["verdict"], "criteria": crit, "class_balanced_advantage_replicates": res["class_balanced_advantage_replicates"]}, indent=1))


if __name__ == "__main__":
    main()
