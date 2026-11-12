# CADV-MH

Research implementation of the Counterfactual Affective Dual-View Transformer for mental-health text classification. It is a text-label research system, not a clinical diagnosis or individual risk-assessment tool.

## Current verified scope

- Independent Dataset A (five-class) and Dataset B (binary OCD vs ADHD, subreddit labels) pipelines.
- Canonical exact-duplicate audits and immutable group-disjoint splits.
- Optional sparse character-ngram near-duplicate candidate search.
- One shared configurable Hugging Face encoder for every view.
- Context-conditioned selection with exact-budget hard top-k masks (straight-through binary-concrete gradient in training, identical hard selection at evaluation), counterfactual and evidence-only views.
- Necessity, sufficiency, continuity, classification, and conservative consistency objectives (sparsity is a structural budget constraint under the repaired selector).
- Three-stage freezing and optimization policies.
- Classical baselines, original-view RoBERTa baseline path, progressive ablation runner, evaluation metrics, faithfulness functions, and paired group bootstrap.

`roberta-base` is not bundled. The first real run will download the checkpoint unless it is placed in the Hugging Face cache in advance.

## Installation

```bash
cd CADV-MH
python -m pip install -e .
```

## Data audits and immutable splits

```bash
python scripts/audit_data.py --dataset dataset_a
python scripts/audit_data.py --dataset dataset_b
python scripts/audit_overlap.py
python scripts/prepare_data.py --dataset dataset_a
python scripts/prepare_data.py --dataset dataset_b
```

Split creation refuses to overwrite an existing manifest.

## Repair phase

See `docs/repair_plan.md` (defects, mathematics, deviations) and `docs/pilot_protocol.md` (the only recommended next
step). Repaired runs write to `experiments_v2/`; `experiments/` holds the historical, pre-repair artifacts and must not
be modified. New split manifests are `prepared/<dataset>/primary_split_v2.csv`.

## Safe readiness checks

```bash
pytest
python scripts/train.py --dataset dataset_a --dry-run
python scripts/train.py --dataset dataset_b --dry-run
python scripts/smoke_test.py --device cuda
```

## Full training commands (not executed during implementation)

```bash
python scripts/train.py --dataset dataset_a --model cadvmh --seed 42 --stage all
python scripts/train.py --dataset dataset_b --model cadvmh --seed 42 --stage all
```

Dataset B is sourced reproducibly from Hugging Face. Fetch it before auditing
or preparing its immutable split:

```bash
python scripts/fetch_dataset_b.py
python scripts/audit_data.py --dataset dataset_b
python scripts/prepare_data.py --dataset dataset_b
```

## Baselines and ablations

```bash
python scripts/train_baseline.py --dataset dataset_a --baseline tfidf_svm --seed 42
python scripts/train_baseline.py --dataset dataset_b --baseline tfidf_logistic --seed 42
python scripts/train_baseline.py --dataset dataset_a --baseline original_roberta --seed 42
python scripts/run_ablations.py --dataset dataset_a --seed 42
python scripts/run_ablations.py --dataset dataset_b --seed 42
```

Omit `--execute` from the ablation runner to print commands without launching them. Add it only when the campaign is explicitly authorized.

### Extended publication baselines

The publication baselines are isolated under `experiments/<dataset>/extended_baselines`
and do not alter CADV-MH training or checkpoints:

```bash
python scripts/train_extended_baseline.py --dataset dataset_b --baseline majority --seed 42
python scripts/train_extended_baseline.py --dataset dataset_b --baseline bilstm --seed 42
python scripts/train_extended_baseline.py --dataset dataset_b --baseline bert_base --seed 42
python scripts/train_extended_baseline.py --dataset dataset_b --baseline roberta_base --seed 42
python scripts/train_extended_baseline.py --dataset dataset_b --baseline mental_roberta --seed 42
python scripts/train_extended_baseline.py --dataset dataset_b --baseline sparse_rationalizer --seed 42
```

After the isolated `roberta_base` baseline is complete, evaluate matched-budget
attention and Integrated Gradients rationales:

```bash
python scripts/evaluate_explanation_baseline.py --dataset dataset_b --method attention
python scripts/evaluate_explanation_baseline.py --dataset dataset_b --method integrated_gradients
```

All extended runners refuse to overwrite existing results. Use `--dry-run` or
the bounded batch flags for readiness checks without starting full experiments.

## Evaluation

```bash
python scripts/evaluate.py --dataset dataset_a --checkpoint experiments/dataset_a/cadvmh/seed_42/best.pt
python scripts/evaluate.py --dataset dataset_b --checkpoint experiments/dataset_b/cadvmh/seed_42/best.pt
python scripts/generate_report.py --dataset dataset_a
python scripts/generate_report.py --dataset dataset_b
```

## Interpretability analysis

After a complete CADV-MH checkpoint exists, generate the qualitative rationale
table, original/evidence/counterfactual probability plot, faithfulness-sparsity
plot, and class-wise rationale plot with:

```bash
python scripts/analyze_rationales.py --dataset dataset_a
python scripts/analyze_rationales.py --dataset dataset_b
```

The command uses deterministic top-`rho` masks and exports PNG, PDF, HTML, CSV,
JSON, and JSONL artifacts under `experiments/<dataset>/interpretability/`. If an
extended sparse-rationalizer checkpoint exists, it is added automatically to
the faithfulness-sparsity comparison. Qualitative examples are selected by
transparent heuristics and must be manually reviewed before publication.

See `docs/experiment_protocol.md` and `docs/reproducibility.md` before real experiments.

## Complete seed-42 campaign

To run every configured experiment for both datasets sequentially:

```bash
./run.sh --yes
```

This runs the classical and original-view RoBERTa baselines, A0–A8, all five
leave-one-loss-out experiments, final A8 test evaluation, and report generation.
It performs a full output-directory preflight and refuses to overwrite any
existing experiment run. The script is intentionally restricted to seed 42.
