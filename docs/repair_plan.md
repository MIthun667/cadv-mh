# CADV-MH repair plan (branch `repair/selector-and-eval-fixes`)

Scope: repair implementation defects found in the independent audit. The architecture
(shared encoder, attention pooling, context-conditioned selector, counterfactual and
evidence views, context-effect interaction, residual+LayerNorm fusion, final head, seven
losses, three stages) is unchanged. Historical artifacts under `experiments/` and
`prepared/` are not modified. New runs write to `experiments_v2/` and new split manifests
are written next to, not over, the old ones. No test-set number was used for any decision
below.

Baseline snapshot: commit `3a8baeb` on branch `baseline-pre-repair`.

Notation: `n` eligible tokens, budget `rho=0.2`, `k = min(max(1, ceil(rho n - 1e-6)), n-1)`.

## R1. Selector collapse and train/eval mask mismatch (audit C1, C2, C4)

Root cause. Training used the soft relaxation `m_i = sigmoid((l_i + g_i)/tau)` with no
budget. The counterfactual is `(1-m)e + m e_mask` and the evidence view is
`m e + (1-m) e_mask`. Selecting everything (`m = 1`) gives a counterfactual of all
`<mask>` (necessity satisfied) and an evidence view equal to the full text (sufficiency
satisfied). The weighted sparsity term (0.05 * 0.8) was far weaker than that incentive
and its gradient vanishes once the sigmoid saturates. Evaluation then used a hard top-20%
mask the model never saw. Ranking used `sigmoid(l)`, which is exactly 1.0 in float32 for
large logits, so top-k degenerated to index-order tie-breaking.

Fix (smallest change that removes the degenerate optimum). Training now uses the same
budgeted hard selection as evaluation, with a straight-through gradient:

    s_i = l_i + g_i^1 - g_i^2                (Gumbel/logistic perturbation, train only)
    h   = TopK_k(s)                          (hard, exactly k tokens, no gradient)
    p_i = sigmoid(s_i / tau) * eligible_i    (relaxation, carries the gradient)
    m   = h + p - stopgrad(p)                (forward value h, backward d m / d l = d p / d l)

Why this is justified. Forward values are exactly the discrete masks that evaluation
uses, so there is no regime mismatch. `m` is the standard straight-through estimator of
the hard top-k operator with the binary-concrete surrogate (Bengio et al. 2013; Maddison
et al. 2017). Because `sum(h) = k < n` by construction, a view with every token selected
is unreachable: necessity requires removing at most k tokens to lower the true-class
probability, and sufficiency requires classifying from at most k tokens. Neither can be
satisfied by a degenerate "select all" or "select none" solution.

Sparsity term (revised after a CPU experiment). Under a hard budget `mean(h)` is constant, so a sparsity
term can only act through the relaxation `p`, and there it is harmful: `|mean p - rho|` pushes every logit
down, which saturates `sigmoid` and removes the straight-through gradient (`dp/dl = p(1-p)/tau -> 0`).
I first implemented it as relaxation calibration; `scripts/cpu_toy_collapse_demo.py` showed it prevented the
selector from learning a known rationale (signal token selected 0% of the time, below the 20% chance rate),
while necessity+sufficiency alone with the hard budget learned it (100%). The term is therefore skipped
whenever the budget is enforced exactly (`output.budget_enforced`), `loss.sparsity` must be 0 in configs with
`selection_mode: budgeted_st` (validated), and the sparsity "ablation" no longer exists in the repaired
setup. This is a deliberate deviation: the seventh loss becomes a structural constraint. It remains in code
for the legacy soft modes. Ranking uses the perturbed logits `s`, not `sigmoid`, so saturation cannot create
ties; residual exact ties resolve deterministically (stable sort, earliest index) and are counted.

Trade-off. Alternative (keep soft masks, raise the sparsity weight) preserves the
train/eval mismatch and still has `m = 1` as a feasible optimum; rejected. A second
alternative (hard-concrete with L0 penalty) changes the parameterization more than needed.

The legacy `soft`, `straight_through` (threshold at 0.5) and `threshold` modes are kept
for reproducing historical runs but are not used by the repaired configuration.

Budget rule. `k = min(max(1, ceil(rho n - 1e-6)), n-1)` is now one shared function used by the model,
the validation diagnostics and the attention/IG baselines (which previously used `round(rho n)` with a
different minimum, so budgets differed: realized ratio 0.212/0.224 for CADV-MH vs ~0.200 for baselines).
The 1e-6 offset is purely defensive against float error; I checked ratios {0.1,0.15,0.2,0.25,0.3,0.4} for
n < 400 and found no case where plain `ceil` misrounds, so this is not a repaired defect. The excess in the
old CADV-MH ratio came from `ceil` and the minimum of one token on short texts.

## R2. Stage-2 checkpoint selection (audit C3)

Root cause. In stage 2 only the selector trains, so the final head and interaction are at
random initialization. The stage-2 "validation macro-F1" (0.296 / 0.113, chance) therefore
selected the epoch by noise.

Fix. Stage 2 is validated with the frozen auxiliary head under the hard budgeted mask:

    S_sel = mean_x [ p_ev(y|x) - p_cf(y|x) ]   (validation split only)

`p_ev` is the true-class probability on the evidence-only view and `p_cf` on the
counterfactual view. `S_sel = comprehensiveness - sufficiency`; it is bounded in
[-1, 1], is maximized only if the kept tokens preserve the label and the removed tokens
destroy it, and cannot be gamed by selection size because the budget is fixed.
Training F1 in stages 1-2 is computed from the auxiliary head. Stage 3 keeps validation
macro-F1 of the final head. Each epoch also logs comprehensiveness, sufficiency,
random-rationale comprehensiveness (same budget), selection ratio, selector saturation and
tie fraction as diagnostics (not used for selection). Frozen modules run in eval mode
during stage 2 so that dropout noise does not perturb the frozen encoder/head.

## R3. Encoding (audit C7)

Root cause. `base.yaml` forced `latin1` on both files. Dataset B is valid UTF-8, so 31.4%
of posts became mojibake (e.g. the right-single-quote sequence turned into three Latin-1
characters). Dataset A is not valid UTF-8 and has no valid multibyte UTF-8 sequences
(only ~800 stray high bytes, 559 of them C1 control characters, consistent with stripped
emoji).

Policy. Encoding is per dataset and explicit. Dataset B: `utf-8` strict. Dataset A:
`latin-1` (lossless byte mapping, the only single-byte codec that decodes every byte) plus
removal of U+0080-U+009F control characters, with the count reported. The loader raises if
more than 0.5% of texts match a mojibake pattern, so this class of error cannot pass
silently. Texts change, so `sample_id` hashes change; new manifests are written as
`primary_split_v2.csv` (old manifests untouched) and referenced from the dataset config.

## R4. Learning rate and temperature (audit C8, C9)

`train.py` passed `stage3_encoder_learning_rate` to every stage, so `encoder_learning_rate`
never applied. Stage 1 now uses `encoder_learning_rate`, stage 3 uses
`stage3_encoder_learning_rate`; stage 2 has a frozen encoder.

Temperature restarted at 1.0 each stage and never reached `temperature_final`. It now
anneals linearly over the combined stage 2+3 epoch index `g = 1..G`:
`tau_g = tau_0 + (tau_f - tau_0)(g-1)/(G-1)`, ending exactly at `tau_f`. Fixed-temperature
ablation A7 holds `tau_0`. Under straight-through selection tau only sharpens the
surrogate gradient and noise scale; it no longer changes forward masks.

## R5. Ablations (audit C10 and A0 confound)

Single source of truth: `src/cadvmh/training/ablations.py`. Each ablation specifies
stages, fusion mode, loss weight overrides and temperature policy; `train.py` reads it and
the YAMLs in `configs/ablations/` are generated from / tested against it. The evidence
branch is computed only if sufficiency is weighted or the fusion mode needs it
(previously always on in stages > 1). A0 is trained with matched encoder epochs
(`stage1 + stage3`) so the A0 vs full comparison is not confounded by 2 vs 7 encoder
epochs. A8 is declared an alias of the canonical full run, as `run.sh` already assumed.

## R6. Gradient accumulation (audit C12)

Loss is divided by the actual size of each accumulation window, so a final partial window
is not down-weighted.

## R7. Configuration fidelity

`validate_config` rejects unknown keys; unused keys (`comparison_mode`, `pooling_dropout`,
`residual_dropout`, `data.encoding`, `near_duplicate` stub) are removed or wired; the
orphan `configs/model.yaml` and `configs/training.yaml` (not inherited by anything and
contradicting `base.yaml`) are removed (recoverable from the baseline commit).

## R8. Explanation evaluation (audit C5, C6)

Faithfulness is now measured on the deployed mechanism. For rationale `R` chosen by the
model's own selector at budget `rho`, the full model (selector, both views, interaction,
final head) is re-run on the externally masked token ids:

    Comp_final = p_final(yhat | x) - p_final(yhat | x with R -> <mask>)
    Suff_final = p_final(yhat | x) - p_final(yhat | x with complement -> <mask>)

`yhat` is the model's predicted class, matching the baselines; the gold-class variant is
logged alongside. A random-rationale control at the same budget is always reported. The
auxiliary-head within-pipeline numbers remain available but are labelled as mechanism
diagnostics. Attention and IG baseline evaluation now use the same budget function.

## Verification boundary

Unit tests on a tiny random RoBERTa (CPU) check shapes, masks, budgets, gradients, stage
freezing, schedules, loss math and the no-collapse property. They do not establish that
the repaired model trains well; that requires the pilot procedure in the final report.
