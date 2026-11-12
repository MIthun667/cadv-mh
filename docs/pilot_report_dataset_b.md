# Dataset B pilot report (seed 42, validation split only)

Branch `repair/selector-and-eval-fixes`. Dataset B v2 split (`prepared/dataset_b/primary_split_v2.csv`, strict UTF-8):
train 16,780 / validation 2,397 / test 4,795. **The test split was never loaded**: `pilot_diagnostics.py` accepts only
`--partition validation`, and training reads train/validation only. GPU: RTX 5090 (31 GB free), fp16 AMP.
All artifacts: `experiments_v2/pilot/dataset_b/` (checkpoints are git-ignored).

## Runs
| Run | Dir | Notes |
|---|---|---|
| Stage 1 (shared by all) | `A0_seed_42/stage1_best.pt` is A0's; pilot stage 1 = `seed_42_r2` init | 2 epochs, selected epoch 1 |
| Attempt 1 (surrogate `sigmoid(s/tau)`) | `seed_42_attempt1_rerun` | stage 2 only; **failed** (below). Original run lost, see "Incident" |
| Attempt 2 (boundary surrogate, `R1b`) | `seed_42_r2` | stage 2 and stage 3 complete |
| Matched-epoch A0 | `A0_seed_42` | stage 1 only, up to 7 epochs, patience 2 |

Incident: a history rewrite I ran to remove accidentally committed checkpoints also deleted attempt 1's tracked files from
disk. I regenerated attempt 1 with the same seed and the legacy surrogate; it reproduces the pre-loss numbers exactly
(selector score -0.2765 / -0.2803, validation loss 0.7691 / 0.7712).

## Code changes made during the pilot (all logged; none touches architecture or loss weights)
1. Logging only: per-stage checkpoints, `--init-from`, per-component validation losses, validation-only diagnostics.
2. Selector logits computed in float32 (fp16 resolution creates spurious ties).
3. **Straight-through surrogate** changed (`docs/repair_plan.md` R1b) after attempt 1 failed. Mathematical justification there.
Gate 1 as written in `docs/pilot_protocol.md` ("selector score > 0") was wrong: random selection scores about -0.20 because
evidence keeps 20% of tokens and the counterfactual 80%. The right criterion is learned minus random, used below.

## Training progression
Stage 1 (both datasets share the code path): val macro-F1 0.8698 -> 0.8692; train 0.838 -> 0.908. Selected epoch 1.

Stage 2, attempt 1 (legacy surrogate) vs attempt 2 (boundary), validation after the selected epoch:
| | attempt 1 | attempt 2 |
|---|---|---|
| selection ratio (target 0.20) | 0.2125 | 0.2125 |
| logit mean / within-example std | +11.3 / 0.73 | +0.7 / 5.5 |
| fraction \|logit\| > 7 | 0.983 | 0.589 |
| boundary-tie fraction | 0 | 0.0008 |
| aux comprehensiveness learned vs random | 0.0265 vs 0.0269 | **0.1743 vs 0.0269** |
| aux sufficiency gap (lower better) | 0.303 | 0.080 |
| evidence-minus-counterfactual score | -0.277 | +0.094 (random -0.203) |
| necessity (hinge) train / val | 0.176 / 0.191 | 0.146 / 0.143 |
| sufficiency CE train / val | 0.601 / 0.698 | 0.403 / 0.387 |

Stage 3, attempt 2 (temperature 0.77 -> 0.30 over epochs; patience 2; best epoch 4):
| ep | train F1 | val F1 | val loss | val cls | val necessity | val suff CE | val cont | val cons | aux comp learned / random | skipped AMP |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.913 | 0.880 | 0.749 | 0.356 | 0.073 | 0.367 | 0.281 | 0.008 | 0.304 / 0.130 | 1 |
| 2 | 0.939 | 0.883 | 0.826 | 0.379 | 0.071 | 0.426 | 0.276 | 0.014 | 0.350 / 0.192 | 0 |
| 3 | 0.958 | 0.879 | 0.821 | 0.383 | 0.069 | 0.426 | 0.271 | 0.016 | 0.385 / 0.246 | 1 |
| **4** | 0.973 | **0.889** | 0.944 | 0.463 | 0.073 | 0.430 | 0.235 | 0.017 | 0.387 / 0.333 | 1 |
| 5 | 0.979 | 0.879 | 1.418 | 0.696 | 0.088 | 0.723 | 0.227 | 0.019 | 0.381 / 0.311 | 1 |

## Rationale behaviour (stage 3 selected checkpoint, validation n = 2,397)
- Ratio 0.2125 (target 0.20; the excess is `ceil` and the one-token minimum on short texts, not the selector). Ties 0.003.
- **Position bias: none.** Selected relative-position histogram [0.22, 0.20, 0.21, 0.18, 0.19] vs eligible [0.20, 0.20, 0.20, 0.20, 0.21].
  First-k and last-k controls are much weaker than learned (aux comprehensiveness 0.166 / 0.155 vs 0.387).
- Logit scale keeps growing (within-example std 28, q99 = 139). Harmless to ranking and to the offset-invariant surrogate, but
  the surrogate becomes nearly a step function (gradient only near the boundary).
- **Lexical selection:** 24.95% of validation texts contain an explicit condition name (adhd/add/ocd). The selector chose 85% of those
  tokens after stage 2 and 37% after stage 3 (random expectation 21%).

## Learned vs same-budget random (auxiliary head, 95% bootstrap CI over examples)
| | stage 2 | stage 3 |
|---|---|---|
| comprehensiveness learned | 0.174 [0.161, 0.188] | 0.387 [0.374, 0.399] |
| random | 0.027 | 0.333 |
| learned minus random | +0.147 [0.135, 0.160] | **+0.053** [0.048, 0.059] |
| sufficiency gap learned | 0.080 | 0.026 |
| random | 0.230 | 0.312 |
Budget curve at 0.05/0.10/0.20/0.40/0.60 (learned, random), stage 3: (0.175, 0.159), (0.287, 0.254), (0.387, 0.331), (0.475, 0.343), (0.556, 0.300).

## Validation classification (model-selection split)
| | CADV-MH final head | A0 (matched epochs) |
|---|---|---|
| accuracy / macro-F1 | 0.8890 / 0.8890 | 0.8874 / 0.8873 |
| ECE / NLL / Brier | 0.0928 / 0.4636 / 0.2020 | 0.0794 / 0.4157 / 0.1919 |
| per-class accuracy (ocd / adhd) | 0.901 / 0.877 | 0.877 / 0.897 |
| predictions with p > 0.99 | 81.6% | 74.4% |
Paired: CADV-only correct 85, A0-only correct 81, exact McNemar p = 0.82; macro-F1 difference +0.0017, bootstrap 95% CI [-0.0088, +0.0121].
A0 epochs (val F1): 0.870, 0.869, **0.887**, 0.876, 0.872. Both models overfit (train F1 0.98; validation loss rises).

## Final-head faithfulness (whole model re-run on edited ids, predicted class)
| | learned | same-budget random | learned minus random |
|---|---|---|---|
| comprehensiveness | 0.460 [0.443, 0.475] | 0.356 [0.342, 0.370] | +0.104 [0.096, 0.113] |
| sufficiency (lower better) | 0.046 [0.038, 0.053] | 0.392 | -0.346 [-0.366, -0.327] |
Gold-class: comprehensiveness 0.355 vs random 0.265; sufficiency 0.019 vs 0.294.
**By predicted class:** predicted-OCD comprehensiveness 0.837 (random 0.637, n = predicted OCD); predicted-ADHD 0.063 (random **0.061**).

## Shortcut and bias evidence
1. **Mask hypersensitivity (confirmed).** Random 20% masking lowers the final-head predicted-class probability by 0.356 for CADV-MH but by only
   0.035 for the plain A0 classifier. The random control rose across training (aux 0.03 at stage 2 -> 0.33 at epoch 4) while the
   learned-minus-random margin shrank (0.147 -> 0.053). The necessity term trains the network to lose confidence under `<mask>`.
2. **Class-asymmetric mask prior (confirmed).** A fully masked input is classified ADHD 94% (CADV stage 2) and 100% (stage 3 and A0).
   Comprehensiveness is large only for OCD predictions; for ADHD predictions learned equals random (0.063 vs 0.061). Reported
   comprehensiveness is therefore largely "masking flips OCD to ADHD", not evidence-specific.
3. **Position-pattern leakage (tested, rejected).** Evidence input with selected positions filled by random or foreign tokens:
   accuracy 0.534 / 0.531 (chance ~0.5) vs 0.875 with the true tokens. The label is not encoded in which positions are chosen.
4. **Lexical label leakage in the data (confirmed).** On the 24.95% of validation texts with an explicit condition name, accuracy is 99.2% (CADV) / 99.0% (A0),
   dropping to 74.8% / 80.9% when those tokens are masked; texts without one: 85.5% / 85.3%.
5. **Overfitting (confirmed).** Train F1 0.98 vs validation 0.88; validation loss 0.75 -> 1.42 over stage 3; selection by F1 alone
   picked a checkpoint (epoch 4) whose validation loss is 0.944.

## Confirmed remaining problems
1. No classification benefit over the matched A0 baseline and worse calibration (ECE 0.093 vs 0.079).
2. Comprehensiveness is dominated by a mask-induced class prior; the learned-minus-random margin is small and class-specific.
3. Heavy overfitting and a selection rule blind to validation loss/calibration.
4. Dataset B is partly solvable by the condition names; the selector initially concentrates on them.
5. Realized ratio 0.2125 > 0.20 (rounding on short texts).
6. Single seed; no test evaluation by design.

## Proposed (not implemented) change for the mask-cue problem
Replace the necessity hinge by a *contrastive* one that is zero if a same-budget random mask hurts as much as the learned mask:
`L_nec = relu(m - (p_cf^rand(y) - p_cf^learned(y)))`, where `p_cf^rand` is the true-class probability after masking a random set of the same size.
Justification: the current `relu(m - p_o + p_cf^learned)` is minimized by any change that lowers `p_cf` under masks, including a
generic response to `<mask>` tokens (measured above); the contrastive form removes that incentive because a generic response
lowers `p_cf^rand` equally. Cost: one extra encoder pass per step (a fourth view), i.e. an architecture/loss change that needs sign-off.
Also proposed: select stage-3 checkpoints on validation macro-F1 with a validation-loss or ECE tie-break, and use the final-head
random-control margin per predicted class as the faithfulness report.

## Recommendation: NO-GO for the full experiments
The selector no longer collapses and demonstrably beats random and positional controls, and the label is not leaked through
selection positions. But the faithfulness evidence is inflated by a measured mask-hypersensitivity shortcut, there is no
predictive gain over a matched baseline, and calibration is worse. A GO for the multi-seed campaign should wait for a second
pilot that (a) adopts the contrastive necessity (or another control that makes the random-mask margin the headline metric),
(b) shows a learned-minus-random margin that holds for both predicted classes on validation, and (c) keeps validation loss/ECE
no worse than A0.
