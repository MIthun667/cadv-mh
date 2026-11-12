# Pilot 2 pre-registration (committed before any variant was trained)

Question: does a counterfactual-necessity objective exist, within the current architecture, that removes the measured
mask hypersensitivity without losing predictive quality or calibration, and gives a learned-minus-random explanation
advantage in **both** predicted classes?

## Fixed protocol
Dataset B v2 split, seed 42, validation only (test never read). Identical stage-1 checkpoint
(`stage1_best.pt`, sha256 prefix `dcc352492268e21f`), identical stage-2 (2 epochs) and stage-3 (5 epochs, patience 2)
settings, identical learning rates, loss weights (necessity 0.5 unless stated), margins, temperature schedule and batch
size. Only `loss.necessity_form` (or `loss.necessity` for V1) differs. Stage-3 checkpoint = best validation macro-F1
(the protocol of the first pilot; no outcome-dependent selection). Each variant runs stage 2 and stage 3 from the shared
stage-1 checkpoint. Reference runs: **A** = absolute hinge (`seed_42_r2`, already trained under the same protocol),
**A0** = matched-epoch plain classifier (`A0_seed_42`).

Candidates: **B** `contrastive_prob` (margin 0.2) and **C** `contrastive_logprob` (margin ln 2 = 0.693: the learned mask must
at least halve the true-class probability relative to a same-budget random mask). **V1** (necessity weight 0) is a causal
diagnostic of whether the necessity term creates the generic mask response; it is reported against the same criteria but
is not an adoptable "contrastive" candidate. Margins were fixed here, by derivation, not tuned.

## Acceptance criteria (all must hold for a candidate to be adopted; thresholds are constants in `scripts/pilot2_compare.py`)
- C1 Mask-hypersensitivity gap: final-head predicted-class probability drop under a random 20% mask, minus A0's drop, is
  <= 0.20 for predicted OCD and <= 0.10 for predicted ADHD (A: 0.56 and 0.01).
- C2 Final-head comprehensiveness (predicted class), learned minus same-budget random: lower 95% bootstrap bound > 0.02 in
  **both** predicted classes (A: OCD +0.20, ADHD +0.004 [-0.001, 0.008]).
- C3 Final-head sufficiency, learned minus random: upper 95% bound < 0 for OCD, ADHD and overall.
- C4 Predictive non-inferiority: lower 95% bound of paired-bootstrap (macro-F1 variant minus A0) >= -0.02.
- C5 Calibration: ECE <= ECE(A0) + 0.02 and NLL <= NLL(A0) + 0.10.
- C6 Selector health: realized ratio within 0.02 of 0.20, boundary-tie fraction < 0.05, auxiliary learned-minus-random
  comprehensiveness lower 95% bound > 0 in both predicted classes.
- C7 Condition-name sensitivity: accuracy drop when explicit condition names are masked (examples that contain them)
  <= A0's drop + 0.05.
- C8 No position-pattern leakage: aux accuracy of evidence inputs with selected positions filled by random/foreign tokens
  <= 0.60 (guards against a head that distinguishes learned from random masks by mask shape; learned masks are more
  contiguous than random ones: 0.60 vs 0.82 runs per selected token).

Reported but not criteria: selector logit distribution, rationale stability (Jaccard under dropout), selected-condition-name
fraction, auxiliary-vs-final agreement, per-class numbers, hypersensitivity curves.

Decision rule. Adopt a candidate only if C1-C8 all pass. If none pass: NO-GO on the contrastive family; recommendation then
follows from V1 and the mechanism diagnostics (retain, or redesign). A candidate is never chosen by macro-F1 or aggregate
comprehensiveness. All results are single-seed: passing the criteria is evidence for a second, multi-seed confirmation,
not a claim of reliability. Masked-input effects are model-behaviour interventions, not causal claims about the text.
