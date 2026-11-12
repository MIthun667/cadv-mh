# Confirmatory protocol: Dataset B multi-seed evaluation of Variant B (FROZEN before any new-seed training)

Committed before any seed other than the pilot's has been trained. The decision rules below (section 6) are
implemented as constants in `scripts/confirm_analyze.py` and must not be edited after new-seed results are seen.
Any deviation is to be listed as a protocol deviation in the report, not silently applied.

## 1. Question
Does Variant B (random-mask contrastive probability-margin necessity) consistently improve counterfactual
rationale faithfulness over same-budget random selection, in **both** predicted classes, without materially degrading
classification or calibration relative to A0 and without reintroducing generic mask hypersensitivity?

## 2. Design
Dataset B only; manifest `prepared/dataset_b/primary_split_v2.csv` (sha256 in
`experiments_v2/confirm/dataset_b/provenance_pre_training.json`; 16,780 / 2,397 / 4,795 train / validation / test).
The test partition is not read by any script in this protocol. Data split is fixed; only the training seed varies.
Seeds 42, 43, 44, 45, 46 (seed controls initialisation of non-pretrained modules, dropout, data order, selector noise,
random-control draws during training).

| Model | Definition |
|---|---|
| A0 | original-view RoBERTa classifier (stage 1 only), matched encoder epochs (stage1+stage3 = 7 max), from scratch per seed |
| B | stage 1 (2 ep) -> stage 2 (2 ep) -> stage 3 (5 ep), `necessity_form: contrastive_prob`, margin 0.2 |
| V1 | identical to B with `loss.necessity = 0` |

B and V1 share the seed's stage-1 checkpoint (stage 1 contains no necessity term, so it is identical by construction).
All configs, budget (rho 0.2, exact top-k), optimiser settings, schedules, patience, evaluation procedure are those of
`configs/dataset_b.yaml`/`configs/pilot2/*.yaml` at the committed revision. No hyperparameter is tuned.

**Seed 42 is rerun, not reused.** The pilot A0 was trained at revision 27d0c97 and the pilot-2 B/V1 runs started from a
stage-1 checkpoint produced earlier by a different invocation; bitwise configuration equivalence cannot be demonstrated
without rerunning. The rerun costs one seed and doubles as a determinism check against the pilot numbers (reported, not used for decisions). Pilot artifacts stay untouched.

## 3. Checkpoint selection (validation only, identical for all models, unchanged from the pilot)
Stage 1 and A0: epoch with the best validation macro-F1 of the primary head, patience 2. Stage 2: best
`validation_selector_score` (frozen aux head, hard budget). Stage 3: best validation macro-F1 of the final head, patience 2.
A0 has up to 7 candidate epochs versus 5 for B/V1, which favours A0 slightly; this is conservative for the B-vs-A0 comparison and is reported.

## 4. Metrics (per seed, validation n = 2397)
Macro-F1, accuracy, per-class precision and recall, NLL, ECE (15 bins as implemented), multiclass Brier; final-head
comprehensiveness and sufficiency on the predicted class for learned and same-budget random masks, learned-minus-random,
separately for predicted OCD and ADHD; generic mask hypersensitivity (final-head drop under 20% random `<mask>` minus A0's, by predicted class);
selector logit quantiles, within-example spread, saturation, tie fraction, selection ratio; rationale stability (Jaccard under dropout);
mask contiguity; condition-name masking accuracy drop; pattern-leakage probe; fully-masked ADHD share. Random-control seeds: the
diagnostic scripts use fixed generators (`torch.Generator().manual_seed(0)`, bootstrap seeds 0/1) identical for every model and seed.

## 5. Statistical analysis
Two sources of uncertainty are kept apart.
* **Example-level (within seed):** paired bootstrap over validation examples (conditional on that trained model); used for per-seed intervals.
* **Seed-level (between training runs):** the unit of inference is the seed. For a paired contrast d_s (e.g. B minus A0) over the 5 seeds I report the
  mean, SD, and the one-sided 95% t-bound (df = 4, t = 2.132), i.e. the 90% two-sided t-interval, plus the exact paired sign-flip permutation p
  (min attainable one-sided p = 1/32 = 0.031) and the per-seed values. Validation examples are shared across seeds, so seed-level
  intervals do not capture validation-sampling uncertainty; that is why the per-seed example-level replication requirement (R) exists.
  Pooling examples across seeds is not used for inference (non-independent). No claim of significance rests on overlap of individual intervals.

## 6. Decision rules (all must hold for GO)
Write d = B - A0 unless stated. "Lower/upper bound" = one-sided 95% t-bound over seeds.

* **N (Macro-F1 non-inferiority).** Margin delta = 0.015 Macro-F1. Rationale: on this validation set A0's epoch-to-epoch macro-F1 spans 0.869-0.887 (0.018),
  so a loss of 1.5 points is of the order of ordinary training noise while still being ~36 extra errors among 2,397 posts; a larger loss would not be
  acceptable for a classifier whose purpose is screening-style text triage. Pass iff lower bound of mean(dF1) > -0.015 **and** no single seed has dF1 < -0.03.
  A CI that merely contains zero is not sufficient.
* **E1 (comprehensiveness advantage).** For each predicted class: seed-mean of (learned - random) final-head comprehensiveness >= 0.03 and its lower bound > 0.
* **E2 (sufficiency advantage).** For each predicted class: seed-mean of (learned - random) sufficiency <= -0.02 and upper bound < 0.
* **R (replication).** In at least 4 of 5 seeds, for **both** predicted classes: the per-seed example-level bootstrap 95% interval for comprehensiveness advantage lies entirely above 0 and for
  sufficiency advantage entirely below 0. "Class-balanced rationale advantage replicates" is defined as E1 and E2 and R all passing.
* **M (necessity term adds value).** Paired B minus V1 on predicted-ADHD comprehensiveness advantage: seed-mean > 0 and lower bound > 0 (V1 lacked the ADHD advantage in the pilot). If this fails the necessity term is not shown to be what delivers the advantage and the verdict is NO-GO for B as proposed.
* **K (calibration).** Upper bound of mean(ECE_B - ECE_A0) <= 0.02 and upper bound of mean(NLL_B - NLL_A0) <= 0.10. Brier is reported, not gated.
* **H (mask hypersensitivity).** Seed-mean gap vs A0 (B minus A0 final-head 20%-random-mask drop, per predicted class) <= 0.20 (OCD) and <= 0.10 (ADHD), and no seed exceeds 0.30 (OCD) / 0.15 (ADHD).
* **S (selector health).** In every seed: |ratio - 0.2| <= 0.02, tie fraction < 0.05, aux learned-minus-random lower CI > 0 in both predicted classes. Saturation, stability, logit distributions are descriptive (not gated); a saturation fraction > 0.95 in any seed is flagged.
* **L (shortcuts).** Condition-name masking drop: upper bound of mean(drop_B - drop_A0) <= 0.05. Pattern-leakage random/foreign-fill accuracy <= 0.60 in every seed.

Verdict: **GO** iff N, E1, E2, R, M, K, H, S, L all pass. Otherwise **NO-GO**; failures are diagnosed and a next action is recommended, but no further tuning or reruns are done automatically. V1 and the pilot results are never used to select the winner; macro-F1 and aggregate comprehensiveness alone are never sufficient.
If a reproducible critical defect appears (crash, NaN, evaluation inconsistency), execution stops and the defect is diagnosed before continuing.

## 7. Provenance
Per run: config snapshots, run metadata (git revision, hardware), history, best checkpoints (stored in `experiments_v2/confirm/`, git-ignored), validation predictions with per-example
rationale-edit faithfulness records (`validation_predictions.csv`), diagnostic JSONs, logs. Small result files are committed; checkpoints are not.
