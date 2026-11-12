# Pilot protocol (CPU checks done; GPU pilot not yet run)

Purpose: verify that the repaired implementation trains stably and produces non-degenerate, faithful
rationales **before** any full campaign. Pilots use the validation split only. The test split is not opened.
Nothing below may be tuned against test results.

## 0. Preconditions (CPU, already passing)
`CUDA_VISIBLE_DEVICES="" python -m pytest -q` (58 tests) and `python scripts/train.py --dataset dataset_{a,b} --dry-run`.

## 1. Data
`python scripts/prepare_data.py --dataset dataset_{a,b}` creates `primary_split_v2.csv` (done; refuses to overwrite).
Inspect the printed encoding/mojibake report; the loader aborts if >0.5% of texts look like mojibake.

## 2. Dataset B pilot, one seed (about 4 h on the RTX 2060 for all stages; Dataset B is the collapse case)
`python scripts/train.py --dataset dataset_b --output-dir experiments_v2/pilot/dataset_b/seed_42 --stage all`
(keeps the real data; do NOT pass `--max-train-batches` for the gate decisions). Monitor `history.jsonl`.

### Gate 1 (after stage 2, validation only; abort and debug if any fails)
- `validation_selection_ratio` within 0.01 of 0.20.
- `validation_boundary_tie_fraction` < 0.05.
- `validation_selector_score` > 0 and clearly above stage-2 epoch 0 (evaluate once before training for the reference).
- `validation_comprehensiveness_aux` exceeds `validation_random_comprehensiveness_aux` by > 0.05.
- `validation_selector_saturation` < 0.95 (a nonzero fraction is expected; 1.0 means collapse).

### Gate 2 (after stage 3)
- Final-head validation macro-F1 within 1 point of the A0 matched-epoch run (below).
- Gate-1 diagnostics still hold in every stage-3 epoch (no drift into saturation).
- Training loss components finite; `skipped_amp_steps` small (< 1% of steps).

## 3. Controls (same split, validation only)
1. A0 with matched encoder epochs: `python scripts/run_ablations.py --dataset dataset_b --ablations A0 --execute`.
2. Faithfulness on validation (add `--partition validation` support or use a bounded `--max-examples` run of
   `scripts/analyze_rationales.py`): headline `final_predicted_comprehensiveness` must exceed
   `final_predicted_random_comprehensiveness` by a margin that is stable across 3 bootstrap resamples; otherwise the
   effect is a mask-count cue, not selector quality.
3. Attention and IG baselines re-evaluated with the repaired budget function and the same model class rule.

## 4. Dataset A pilot
Repeat steps 2 and 3 on `dataset_a`. Report suicidal-class results with intervals only (validation n = 36).

## 5. Decide
Proceed to the full multi-seed campaign only if Gate 1 and Gate 2 pass on both datasets. Before any multi-seed
run, relax the research-policy restriction `RESEARCH_SEED = 42` in `src/cadvmh/training/seeds.py` and the
`choices=[42]` argparse limits in the scripts (deliberately left unchanged here), then use at least 5 seeds.
