# Counterfactual necessity: root causes and candidate objectives

Notation. `z^o, z^l, z^r` are auxiliary-head logits for the original input, the learned-mask counterfactual and a
same-budget random-mask counterfactual; `p = softmax(z)`; `y` the gold label; `m = 0.2`; `mu = ln 2`.

## 1. Diagnosed root causes (validation, `experiments_v2/pilot2/dataset_b/mechanism_baseline_objective.json`)
| Hypothesis | Evidence | Verdict |
|---|---|---|
| Necessity hinge saturation | Stage 3: hinge active for 13% (OCD) / 14% (ADHD) of validation and 2% / 4% of train examples; mean gradient norm 0.045 (val), 0.005-0.009 (train). Stage 2 (frozen encoder): active 50% / 83%. | Confirmed, but as a **consequence** |
| Generic learned response to masks | Random same-budget masks alone already satisfy the hinge (`p_o-p_r >= 0.2`) for 81% / 88% of stage-3 examples (stage 2: 11% / 4%). Aux random-mask drop 0.40 / 0.44 (A0: 0.08 / 0.04). Random-word replacement also hurts (0.26 / 0.37), but `<mask>` hurts more. | **Primary cause** (emerges during stage 3, when the encoder is trained) |
| Class-specific mask prior | A fully masked input is ADHD in 100% of cases (A0 also). Final head, predicted OCD: drop 0.15 at 5% masking, 0.63 at 20%; predicted ADHD: 0.05-0.07 at every fraction. A0, OCD: 0.03 / 0.08 at 5% / 20%. Final head's OCD mask-drop is twice its random-word drop (0.64 vs 0.32); ADHD none. | Confirmed: latent in A0, amplified ~8x in CADV-MH |
| Auxiliary vs final-head disagreement | Predictions agree (99.2%) but not their mask response: predicted ADHD, random-mask drop aux 0.44 vs final 0.06. The necessity term trains only the aux head's views. | Confirmed: the objective does not constrain the final head directly |
| Overfitting of the counterfactual branch | Train vs validation aux drops 0.55 vs 0.53 (OCD), 0.44 vs 0.43 (ADHD); hinge inactive more often on train (98% vs 87%). | Minor |
| Selector logit saturation | Stage 2 learns well (learned-minus-random aux +0.28 OCD / +0.075 ADHD) with a comparable saturated-logit fraction; the margin disappears only after stage 3 (+0.133 / **-0.009**), when the generic response appears. | Not the cause |

Mechanism: in stage 2 the encoder is frozen so necessity can only be reduced by choosing informative tokens. In stage 3 the
encoder is trainable and the absolute hinge is cheapest to satisfy by making the shared representation collapse for any
masked input. After that the hinge is inactive (no gradient) and the selector no longer receives a differential signal.

## 2. Candidates
**A, absolute hinge (current).** `L_A = [m - p^o_y + p^l_y]_+`, `p^o` detached.
`dL_A/dz^l_j = 1[p^l_y > p^o_y - m] * p^l_y (delta_jy - p^l_j)`. Zero when inactive, and scaled by `p^l_y` (-> 0 as the true-class
probability collapses). Trivial minimiser: any map "contains >= k mask tokens -> low `p_y`", independent of the selector (measured).

**B, probability contrast.** `L_B = [m - (p^r_y - p^l_y)]_+`, gradients through both views.
`dL_B/dz^l_j = 1[act] p^l_y (delta_jy - p^l_j)`, `dL_B/dz^r_j = -1[act] p^r_y (delta_jy - p^r_j)`, active iff `p^r_y - p^l_y < m`.
A generic response gives `p^r = p^l`, so the loss equals `m` (fully active): the gradient lowers `p^l`, raises `p^r`.
It therefore directly penalises hypersensitivity (the second term trains robustness to random masks). Same softmax factors
as A: vanishes when both probabilities saturate near 0 or 1. Trivial minimisers: (i) a head that recognises the learned mask by its
shape (learned masks are more contiguous: 0.60 vs 0.82 runs per selected token) and answers wrongly only for those; (ii) memorisation
of training labels. Both are tested (C8, train-vs-validation).

**C, log-probability contrast.** `L_C = [mu - (log p^r_y - log p^l_y)]_+`, `mu = ln 2`.
`dL_C/dz^l = 1[act] (e_y - p^l)` (norm ~ `(1-p^l_y)*sqrt(2)` in the binary case, **does not vanish as `p^l_y -> 0`**),
`dL_C/dz^r = -1[act] (e_y - p^r)`, which vanishes only as `p^r_y -> 1`. Bounded because the hinge stops at margin `mu`. Hazards: heavy
tails when `p^r_y` is small (`Var[log p^r] ~ Var[p^r]/p^r_y^2`) and a stronger push on the random view, which can dominate.
Measured at the stage-3 reference checkpoint: active fraction 82-99% (B) and 94-99% (C), mean gradient norm 0.28-0.34 (B), 0.65-0.71 (C), against 0.045 (A).

## 3. Common properties
- **Variance:** one random mask per step. Across 4 draws the per-example std of `p^r` is 0.03-0.11 (mean), small against the margin 0.2 in
  the aggregate but non-negligible per example; C inherits larger variance where `p^r` is small.
- **Cost:** one extra encoder forward and backward in stage 3 (5 vs 4 passes, about +25% step time); stage 2 one extra forward without backward.
- **Relation to final-head faithfulness:** all three act on the auxiliary head's views; the final-head metric re-runs the whole model on
  edited ids. The objectives reach the final head only through the shared encoder/pooling, so improvement in the auxiliary head does not
  imply improvement in the final head (measured disagreement above). C1-C2 are therefore judged on the final head.
- **What cannot be fixed by these objectives:** A0 itself maps masked inputs to ADHD (fully masked: 100%) and its OCD predictions degrade with
  mask fraction (0.32 at 60%). Criteria compare against A0, not against zero.
- No causal claim: masked-input effects describe the model's behaviour under an intervention on its inputs.
