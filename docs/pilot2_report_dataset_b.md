# Pilot 2 report: counterfactual-objective investigation (Dataset B, validation only)

Branch `repair/selector-and-eval-fixes`. Criteria and thresholds were committed in `1db6212` before any variant
training and were not edited afterwards (`scripts/pilot2_compare.py`, `docs/pilot2_preregistration.md`).
Seed 42, identical v2 train/validation partitions, same stage-1 checkpoint (sha256 prefix `dcc352492268e21f`),
same encoder/selector/fusion/budget (rho 0.2), same epochs. Only the necessity term differs. The test partition
was never used. Original artifacts (`experiments/`, `experiments_v2/pilot/`) are untouched; outputs are in
`experiments_v2/pilot2/dataset_b/`. Variants were trained concurrently on one GPU (affects wall-clock only).
Single seed, n = 2397 validation examples: all differences below are single-seed, validation-only evidence.

## 1. Diagnosed root causes of mask hypersensitivity (reference objective A, `mechanism_baseline_objective.json`)

| Candidate cause | Verdict | Evidence |
|---|---|---|
| Hinge saturation | Consequence, not cause | At stage 3 the absolute hinge is active for 13% (OCD) / 14% (ADHD) of validation examples (2% / 4% on train), mean grad norm 0.045. It is satisfied because random masks already satisfy it. |
| Generic learned response to masks | **Primary cause** | Random same-budget masks alone give a drop >= 0.2 for 81% / 88% of stage-3 examples (stage 2: 11% / 4%). Aux random-mask drop 0.40 / 0.44 vs A0 0.08 / 0.04. |
| Class-specific mask prior | Latent in A0, amplified ~8x | Fully masked input is classified ADHD 100% (also in A0). Final-head predicted-OCD drop under 20% random masks: 0.63 (A) vs 0.08 (A0). Predicted ADHD about 0.05 flat. |
| Aux vs final head disagreement | Contributing | Predictions agree 99.2%, but predicted-ADHD random-mask drop is 0.44 on aux and 0.06 on final; necessity trains only the aux head's views. |
| Counterfactual-branch overfitting | Minor | Train vs validation aux drops 0.55 vs 0.53 (OCD), 0.44 vs 0.43 (ADHD). |
| Selector logit saturation | Not the cause | Stage 2 learns (learned-minus-random aux comprehensiveness +0.280 OCD / +0.075 ADHD); the loss of the advantage happens in stage 3 when the encoder is trained against the hinge. |

## 2. Mathematical comparison (details in `docs/necessity_objectives.md`)

Notation: p_l, p_r, p_o = true-class probability under learned mask, same-budget random mask, original input.

- **A** `relu(m - p_o + p_l)`, p_o detached. Cheap, but it is satisfied by *any* mask-induced drop, so the
  encoder can reduce p for every masked input (trivial solution: generic mask sensitivity), after which the
  gradient vanishes. Gradient to the logits only flows while active (measured norm 0.045 at the reference checkpoint).
- **B** `relu(m - (p_r - p_l))`, m = 0.2. The random-mask drop cancels, so generic mask sensitivity earns nothing;
  only *selection-specific* damage is rewarded. Gradient flows through both views; active 82-99%, norm 0.28-0.34.
  Trivial solutions: p_r raised (the model becomes mask-robust to random masks, a benign direction), or p_l lowered.
  Variance: per-example std of p_r across random draws 0.03-0.11, one draw per step (unbiased, noisy).
  Overhead: one extra encoder pass.
- **C** `relu(ln2 - (log p_r - log p_l))`. Same logic in log space; the gradient is amplified when p_l is small
  (norm 0.65-0.71), so it keeps pushing p_l toward 0 (unbounded log-penalty), which can favor an extreme,
  overconfident wrong-class response and degrade calibration. Same overhead as B.
- None of A/B/C acts on the final head directly: they train the auxiliary head's views, so final-head
  faithfulness is only inherited through the shared encoder (the aux/final gap above).
- B and C were not assumed superior; V1 (no necessity term) was run as a causal control.

## 3. Controlled results (validation, n = 2397, A0 = no counterfactual terms, same matched epochs)

| Metric | A0 | A (absolute) | B (prob contrast) | C (log-prob contrast) | V1 (no necessity) |
|---|---|---|---|---|---|
| Macro-F1 | 0.8873 | 0.8890 | 0.8819 | 0.8890 | 0.8878 |
| paired dF1 vs A0 [95% CI] | - | +0.0016 [-0.0085, +0.0129] | -0.0053 [-0.0159, +0.0058] | +0.0015 [-0.0096, +0.0126] | +0.0003 [-0.0099, +0.0104] |
| NLL | 0.4157 | 0.4636 | 0.3962 | 0.5928 | 0.3072 |
| ECE | 0.0794 | 0.0928 | 0.0810 | 0.1005 | 0.0520 |
| Brier | 0.1919 | 0.2020 | 0.1971 | 0.2084 | 0.1741 |
| Hypersensitivity gap vs A0 (OCD / ADHD) | 0 | +0.55 / +0.01 | **+0.02 / -0.01** | +0.70 / +0.08 | -0.01 / -0.01 |
| Selector saturation (|logit|>7) | - | 0.52 | 0.86 | 0.55 | 0.47 |
| Rationale stability (Jaccard, dropout) | - | 0.49 | 0.49 | 0.39 | 0.46 |
| Contiguity (runs/selected token; random 0.82) | - | 0.57 | 0.65 | 0.73 | 0.65 |
| Selection ratio | - | 0.2125 | 0.2125 | 0.2125 | 0.2125 |
| Boundary tie fraction | - | 0.003 | 0.005 | 0.000 | 0.002 |

## 4. Per-class final-head faithfulness (learned minus same-budget random, 95% CI; predicted class)

| | Comprehensiveness OCD | Comprehensiveness ADHD | Sufficiency OCD | Sufficiency ADHD (lower is better) |
|---|---|---|---|---|
| A | +0.201 [0.187, 0.215] | +0.003 [-0.002, 0.007] | -0.708 [-0.730, -0.686] | +0.034 [0.019, 0.049] |
| B | +0.446 [0.422, 0.468] | **+0.050 [0.038, 0.062]** | -0.489 [-0.513, -0.465] | **-0.028 [-0.043, -0.014]** |
| C | +0.152 [0.134, 0.170] | +0.807 [0.789, 0.825] | -0.060 [-0.076, -0.043] | -0.864 [-0.881, -0.846] |
| V1 | +0.372 [0.352, 0.393] | -0.004 [-0.015, 0.007] | -0.510 [-0.531, -0.489] | +0.011 [-0.002, 0.024] |

For B, the final-head drop under 20% *random* masks is 0.096 (predicted OCD) / 0.038 (predicted ADHD), versus
0.528 / 0.080 under the *learned* mask. C's asymmetry has flipped sign (OCD advantage small, ADHD large), which
together with its calibration and condition-name-sensitivity failures indicates a different class-specific
shortcut rather than better explanations.

## 5. Pre-registered criteria (unchanged thresholds)

| Criterion | A | B | C | V1 |
|---|---|---|---|---|
| C1 hypersensitivity gap (<=0.20 OCD, <=0.10 ADHD) | fail | **pass** | fail (0.70) | pass |
| C2 comprehensiveness advantage lower CI > 0.02, both classes | fail | **pass** | pass | fail |
| C3 sufficiency advantage (upper CI < 0), OCD/ADHD/overall | fail | **pass** | pass | fail |
| C4 dF1 lower CI >= -0.02 | pass | pass | pass | pass |
| C5 ECE <= A0+0.02 and NLL <= A0+0.10 | pass | pass | fail | pass |
| C6 ratio, ties, aux learned>random both classes | fail | pass | pass | fail |
| C7 condition-name masking drop <= A0 + 0.05 | fail | pass (0.197 vs 0.181) | fail (0.279) | pass |
| C8 pattern-leakage accuracy <= 0.60 | pass | pass (0.547 / 0.513) | pass | pass |

**Only B passes all eight.** C fails C1, C5, C7. V1 confirms that removing the necessity term removes the
hypersensitivity (so the term is what causes it), but also removes the ADHD explanation advantage.

## 6. Masking-shortcut diagnostics

- B removes the generic mask response (gap +0.02 / -0.01) but **does not** remove A0's class prior: a fully
  masked input is still classified ADHD 100%. This is a property of the encoder from stage 1, not of the objective.
- Pattern-leakage probe: random-fill 0.547, foreign-fill 0.513 for B (threshold 0.60), no evidence that the
  position pattern alone carries the label. Learned masks are more contiguous than random (0.65 vs 0.82 runs per token).
- Condition-name masking drop for B (0.197) is slightly above A0 (0.181), within the pre-registered tolerance:
  lexical dependence on the condition name remains and is not reduced by any objective.

## 7. Limitations

1. One seed; the B-vs-A0 dF1 CI spans zero (-0.016 to +0.006). The B point estimate is the lowest F1 of all runs.
   F1 was not used for selection, but noninferiority at the -0.02 bound is only established for this seed/split.
2. B's predicted-ADHD advantage is real by the criterion but small (+0.050 comprehensiveness, -0.028 sufficiency);
   the OCD advantage is large. The explanations are far from symmetric.
3. V1 has the best calibration (ECE 0.052, NLL 0.307, Brier 0.174). B's calibration is only equal to A0, so B
   buys faithfulness, not calibration. The ECE comparison is mildly in V1's favor and is unexplained.
4. B has 86% selector saturation (|logit|>7) versus ~50% elsewhere; tie rate is still < 0.5%, stability is unchanged,
   but this warrants a check across seeds.
5. Necessity objectives still train only the aux head's views; final-head faithfulness is indirect. Margins were fixed
   by derivation (0.2, ln 2) and not tuned; no other hyperparameter was varied.
6. Masked-input interventions measure model dependence on tokens, not causal effect on the label; no causal claim is made.
   Fully masked-input bias, lexical (condition-name) dependence and out-of-distribution `<mask>` inputs remain.
7. Variants shared a GPU; the random control uses one draw per step.

## 8. Recommendation

**Adopt contrastive probability-margin necessity (B) as a justified replacement for the absolute hinge, conditionally**:
it is the only candidate that passes every pre-registered criterion, it fixes the diagnosed root cause (generic mask
response, gap ~0), and it is mathematically motivated and bounded. C is rejected (hypersensitivity, calibration, condition-name
sensitivity). The original objective A is rejected (fails C1, C2, C3, C6, C7).
This is **not** a claim that B is a solved explanation objective: evidence is single-seed, validation-only, the ADHD
advantage is small, and the class prior and lexical dependence remain. The next step that the evidence supports is
a multi-seed confirmation of B vs A0 vs V1 on Dataset B (not run here, per the constraints), with the same criteria; if
B's ADHD advantage or F1 non-inferiority does not replicate, the verdict should revert to NO-GO / redesign
(e.g., a final-head-level faithfulness term, which is a new loss and was out of scope).

Files: `experiments_v2/pilot2/dataset_b/compare_variants.json`, `compare_reference_A.json`,
`mechanism_*.json`, `diagnostics_*`, `*/history.jsonl`, `logs/`.
