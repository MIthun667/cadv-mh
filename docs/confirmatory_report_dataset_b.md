# Dataset B multi-seed confirmation of Variant B: report

**Verdict: NO-GO.** The class-balanced rationale advantage of Variant B (random-mask contrastive probability-margin
necessity) does **not** replicate. Seed 42 (the pilot seed) was the most favourable of the three completed seeds.
Validation only; the test partition was never read. Branch `repair/selector-and-eval-fixes`.
Protocol (frozen before training): `docs/confirmatory_protocol.md`. Analysis: `scripts/confirm_analyze.py`.
Full tables: `experiments_v2/confirm/dataset_b/confirm_tables.md`, `confirm_results.json`. Figures: `docs/figures/confirm_*.png`.

## 1. Protocol deviations (read first)
1. **3 of the 5 planned seeds** (42, 43, 44) were completed. The user stopped the run after seed 44 finished; seed 45 was
   partially trained (stage 1 and A0 started) and is excluded, with its partial artifacts left in place; seed 46 never started.
   `SEEDS` in the analysis script was changed to (42, 43, 44) and the replication rule R changed from "at least 4 of 5 seeds" to "all 3 seeds"
   (stricter than 80%). Both edits were made **before any aggregate result was read**; no other threshold was touched.
2. With 3 seeds the seed-level t-quantile is 2.920 (df = 2), not 2.132, and the smallest possible sign-flip p is 1/8 = 0.125.
   Seed-level bounds are therefore wide. I separate failures caused by this lack of power from failures that are not (section 5).
3. Seed 42 was rerun rather than reused (as protocol). The rerun reproduced the pilot to four decimals for all three models
   (Macro-F1 A0 0.8873, B 0.8819, V1 0.8878), so training is deterministic at fixed seed and the pilot numbers are valid for that seed.
4. Rationale token positions were added to the validation prediction table (evaluation only; no numbers changed).
5. The three pilot `COMPLETE` markers were left as they were.

## 2. Predictive performance and calibration (validation n = 2397; per seed 42 / 43 / 44)
| | A0 | B | V1 |
|---|---|---|---|
| Macro-F1 | .8873 / .8890 / .8828 (mean .8864) | .8819 / .8932 / .8893 (.8881) | .8878 / .8864 / .8882 (.8874) |
| NLL | .416 / .501 / .577 | .396 / .548 / .746 | .307 / .523 / .547 |
| ECE | .079 / .093 / .098 | .081 / .095 / .106 | .052 / .096 / .094 |
| Brier | .192 / .204 / .211 | .197 / .198 / .214 | .174 / .206 / .204 |

Accuracy, per-class precision and recall are in the tables file. Paired seed-level contrasts (mean, 90% t-interval):
B−A0 Macro-F1 +0.0018 [−0.0089, +0.0124]; V1−A0 +0.0011 [−0.0057, +0.0079]; B−V1 +0.0007 [−0.010, +0.011].
**Non-inferiority (rule N) passes**: lower bound −0.0089 is above −0.015 and the worst single seed is −0.0054.
B−A0 ECE +0.0038 [−0.0020, +0.0095]; NLL +0.065 [−0.096, +0.227]. NLL worsens markedly in seed 44 (+0.17). The seed-42 pilot
calibration advantage of V1 (ECE 0.052) did not replicate (0.096, 0.094 in seeds 43/44).

## 3. Rationale faithfulness, final head, learned minus same-budget random, by predicted class
| | B comp OCD | B comp ADHD | B suff OCD | B suff ADHD |
|---|---|---|---|---|
| seed 42 | +0.446 | +0.050 | −0.489 | −0.028 |
| seed 43 | +0.324 | +0.450 | −0.016 (CI includes 0) | −0.826 |
| seed 44 | +0.166 | +0.833 | −0.047 | −0.907 |
| mean (SD) | +0.312 (0.140) | +0.444 (0.391) | −0.184 (0.265) | −0.587 (0.486) |

V1 (no necessity term): comprehensiveness advantage OCD +0.372 / +0.647 / +0.460, **ADHD −0.004 / −0.002 / −0.024** (no advantage in any seed);
sufficiency advantage OCD −0.51 / −0.73 / −0.57, ADHD +0.011 / +0.021 / +0.012 (wrong sign).

* The ADHD advantage of B is positive in every seed, but its size varies 17-fold (0.05 to 0.83), so the pilot's "+0.05" was not typical.
* The OCD advantage of B shrinks monotonically across seeds (0.45, 0.32, 0.17) because the **random** control becomes extremely damaging
  for predicted OCD (random comprehensiveness 0.08, 0.46, 0.77). That is generic mask hypersensitivity, not a better rationale.
* Replication rule R: only seed 42 and 44 have all four per-seed example-level intervals on the correct side of zero; seed 43 fails (OCD sufficiency). R requires 3 of 3: **fail**.
* B−V1 predicted-ADHD comprehensiveness: +0.054, +0.452, +0.857 (mean +0.454, lower bound −0.22; positive in all 3 seeds but its size is not reproducible).
  The necessity term is the only difference between B and V1, so it does produce ADHD-class advantage, but of seed-dependent magnitude. B−V1 OCD advantage is negative on average (−0.18).

## 4. Masking-shortcut diagnostics
* **Mask hypersensitivity is back in B.** Final-head 20% random-mask drop for predicted OCD, B minus A0: +0.017 / +0.371 / +0.712
  (threshold 0.20 mean, 0.30 any seed). Predicted ADHD gap is small (−0.011, −0.022, −0.021). V1−A0 OCD gap: −0.006, +0.012, +0.004.
  So the pilot conclusion that B removes the generic mask response does not hold in seeds 43 and 44. V1 never shows it, so in these seeds the
  hypersensitivity is attributable to the contrastive necessity term as implemented (and the pilot-2 mechanism analysis, which blamed the *absolute* hinge, was incomplete).
* **Fully masked input.** A0 classifies it ADHD in seed 42 but OCD in seeds 43 and 44 (the class prior is itself seed-dependent), while B and V1 classify it ADHD 100% in every seed.
  The ADHD default in B/V1 therefore appears to come from the stage-2/3 pipeline, not from the necessity objective; mechanism untested.
* **Condition-name masking:** accuracy drop B−A0 = +0.017, +0.065, +0.114 (mean +0.065, upper bound +0.147 vs limit +0.05): **fail**, rising with seed. V1−A0 is small (+0.006).
* **Pattern-position leakage:** B foreign-fill accuracy reaches 0.648 in seed 44 (limit 0.60): **fail** in that seed.
* **Selector health (rule S):** ratio 0.2125, tie fraction ≤ 0.005, aux learned>random for both classes in all seeds: pass. Saturation (|logit| > 7): B 0.86 / 0.84 / 0.50 vs V1 0.47 / 0.45 / 0.48; within-example logit std 14.8 / 12.3 / 10.8. Rationale stability (Jaccard) B 0.49 / 0.41 / 0.51.

## 5. Decision rules
| Rule | Result | Cause of failure |
|---|---|---|
| N Macro-F1 non-inferiority (δ = 0.015) | **pass** | |
| E1 comprehensiveness advantage ≥ 0.03, lower bound > 0, both classes | fail | ADHD lower bound −0.22: low power from large seed variance (the ADHD mean 0.44 is large) |
| E2 sufficiency advantage ≤ −0.02, upper bound < 0, both classes | fail | OCD mean −0.18 but SD 0.27; seed 43 and 44 OCD sufficiency advantage near 0 (−0.016, −0.047): partly substantive |
| R replication in all 3 seeds | fail | seed 43 OCD sufficiency interval includes 0 (substantive) |
| M B−V1 ADHD comprehensiveness, lower bound > 0 | fail | mean +0.454 but lower bound −0.22: power-limited |
| K calibration (ECE ≤ +0.02, NLL ≤ +0.10) | fail | ECE passes (upper +0.0095); NLL upper bound +0.227 (seed 44 +0.17) |
| H hypersensitivity (OCD ≤ 0.20 mean, 0.30 any seed) | fail | **substantive**: seed gaps 0.02, 0.37, 0.71 |
| S selector health | **pass** | |
| L condition-name and pattern leakage | fail | **substantive**: condition-name drop growth; pattern leakage 0.648 in seed 44 |

Failures of E1 and M are power-limited (df = 2) and would not by themselves be conclusive. H, R and L failures are driven by individual seeds
outside hard limits and do not depend on the t-interval. No claim of significance is made for any contrast; with 3 seeds the minimum attainable permutation p is 0.125.

**Does Variant B's class-balanced rationale advantage replicate? No.** B has a positive ADHD advantage in all three seeds, but its size varies from 0.05 to 0.83,
while the OCD advantage decays and its sufficiency advantage vanishes in seed 43. Large increases in random-mask damage for predicted OCD (generic hypersensitivity) account
for much of the apparent ADHD asymmetry shift.

## 6. Limitations
Three seeds only; seed-level intervals are very wide. Single dataset and single split. Validation examples are shared across seeds, so seed-level intervals ignore validation sampling.
Faithfulness uses masked-input interventions on the model, not causal effects on labels; the `<mask>` token is out of distribution. Explanations remain highly dependent on the condition names.
B and V1 share a stage-1 checkpoint per seed, so they are paired, but A0 is trained from scratch (selection among up to 7 epochs). The mechanistic explanations in section 4 are hypotheses; they were not tested.

## 7. Recommendation (no further automatic tuning)
**Do not adopt Variant B.** Retain A0 as the classifier reference. V1 (no necessity term) is the better-behaved explainer on this evidence: stable OCD advantage, no hypersensitivity,
same F1, comparable calibration, but **no** ADHD advantage, so it does not solve the original problem either.
Scientifically justified next step (a new, separately preregistered study, not a continuation of tuning): (i) test the diagnosed mechanism directly by blocking the necessity gradient from
the encoder (selector-only update for the counterfactual branch), because the failure appears when the encoder can reshape its response to `<mask>` under B; (ii) add a final-head-level
faithfulness measure to the objective or model selection, since necessity trains only the auxiliary head's views; (iii) use selection rules that include hypersensitivity, not only
Macro-F1. Run it on seeds 42-46 with the same criteria and report NO-GO if hypersensitivity or ADHD asymmetry persists.
