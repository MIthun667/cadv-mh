#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SEED=42
DATASETS=(dataset_a dataset_b)
ABLATIONS=(A0 A1 A2 A3 A4 A5 A6 A7 A8)
LEAVE_ONE_OUT=(necessity sufficiency sparsity continuity consistency)

usage() {
  echo "Usage: ./run.sh --yes"
  echo
  echo "Runs the complete CADV-MH seed-42 campaign for Dataset A and Dataset B:"
  echo "  - data audits and immutable split preparation"
  echo "  - TF-IDF logistic-regression and linear-SVM baselines"
  echo "  - original-view RoBERTa baseline"
  echo "  - progressive ablations A0 through A8"
  echo "  - five leave-one-loss-out ablations"
  echo "  - A8 test-set evaluation and result-table generation"
  echo
  echo "This is an expensive sequential training campaign. It never uses a seed other than 42."
}

if [[ "${1:-}" != "--yes" ]]; then
  usage
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_restartable_training_target() {
  local target="$1"
  if [[ ! -d "$target" ]]; then
    return
  fi

  local unexpected
  unexpected="$(find "$target" -mindepth 1 -maxdepth 1 -type f ! -name 'run_metadata.json' -printf '%f\n')"
  if [[ -n "$unexpected" ]]; then
    echo "Incomplete training artifacts found in: $target" >&2
    echo "Resume or archive that run before continuing." >&2
    exit 1
  fi
}

neural_run_complete() {
  local target="$1"
  [[ -f "${target}/best.pt" || -f "${target}/test_metrics.json" ]]
}

ablation_complete() {
  local dataset="$1"
  local ablation="$2"
  local target="experiments/${dataset}/ablations/${ablation}/seed_${SEED}"

  if neural_run_complete "$target"; then
    return 0
  fi

  # A8 is the complete CADV-MH configuration. The canonical proposed-model
  # run is therefore the Dataset A A8 result and must not be trained twice.
  if [[ "$dataset" == "dataset_a" && "$ablation" == "A8" ]]; then
    neural_run_complete "experiments/dataset_a/cadvmh/seed_${SEED}"
    return
  fi

  return 1
}

require_file "../Data/data_2/final_balanced_5class_dataset.csv"
require_file "../Data/data_1/mental_health_reddit_posts.csv"

for dataset in "${DATASETS[@]}"; do
  python scripts/audit_data.py --dataset "$dataset"

  manifest="prepared/${dataset}/primary_split.csv"
  if [[ ! -f "$manifest" ]]; then
    python scripts/prepare_data.py --dataset "$dataset"
  else
    echo "${dataset}: using the existing immutable split manifest."
  fi
done

python scripts/audit_overlap.py

# Preflight incomplete training destinations. Completed runs are intentionally
# skipped below, making the campaign safe to restart after interruption.
for dataset in "${DATASETS[@]}"; do
  if ! neural_run_complete "experiments/${dataset}/baselines/original_roberta/seed_${SEED}"; then
    require_restartable_training_target "experiments/${dataset}/baselines/original_roberta/seed_${SEED}"
  fi

  for ablation in "${ABLATIONS[@]}"; do
    if ! ablation_complete "$dataset" "$ablation"; then
      require_restartable_training_target "experiments/${dataset}/ablations/${ablation}/seed_${SEED}"
    fi
  done

  for component in "${LEAVE_ONE_OUT[@]}"; do
    if ! neural_run_complete "experiments/${dataset}/ablations/without_${component}/seed_${SEED}"; then
      require_restartable_training_target "experiments/${dataset}/ablations/without_${component}/seed_${SEED}"
    fi
  done
done

for dataset in "${DATASETS[@]}"; do
  echo "Starting complete seed-${SEED} campaign for ${dataset}"

  if [[ -f "experiments/${dataset}/baselines/tfidf_logistic/seed_${SEED}/metrics.json" ]]; then
    echo "${dataset}: TF-IDF logistic baseline already complete; skipping."
  else
    python scripts/train_baseline.py \
      --dataset "$dataset" \
      --baseline tfidf_logistic \
      --seed "$SEED"
  fi

  if [[ -f "experiments/${dataset}/baselines/tfidf_svm/seed_${SEED}/metrics.json" ]]; then
    echo "${dataset}: TF-IDF SVM baseline already complete; skipping."
  else
    python scripts/train_baseline.py \
      --dataset "$dataset" \
      --baseline tfidf_svm \
      --seed "$SEED"
  fi

  if neural_run_complete "experiments/${dataset}/baselines/original_roberta/seed_${SEED}"; then
    echo "${dataset}: original-view RoBERTa baseline already complete; skipping."
  else
    python scripts/train_baseline.py \
      --dataset "$dataset" \
      --baseline original_roberta \
      --seed "$SEED"
  fi

  for ablation in "${ABLATIONS[@]}"; do
    if ablation_complete "$dataset" "$ablation"; then
      echo "${dataset}: ablation ${ablation} already complete; skipping."
    else
      python scripts/run_ablations.py \
        --dataset "$dataset" \
        --seed "$SEED" \
        --ablations "$ablation" \
        --execute
    fi
  done

  for component in "${LEAVE_ONE_OUT[@]}"; do
    if neural_run_complete "experiments/${dataset}/ablations/without_${component}/seed_${SEED}"; then
      echo "${dataset}: leave-one-out run without ${component} already complete; skipping."
    else
      python scripts/train.py \
        --dataset "$dataset" \
        --model cadvmh \
        --seed "$SEED" \
        --stage all \
        --disable-loss "$component" \
        --output-dir "experiments/${dataset}/ablations/without_${component}/seed_${SEED}"
    fi
  done

  final_checkpoint="experiments/${dataset}/ablations/A8/seed_${SEED}/best.pt"
  if [[ "$dataset" == "dataset_a" && ! -f "$final_checkpoint" ]]; then
    final_checkpoint="experiments/dataset_a/cadvmh/seed_${SEED}/best.pt"
  fi

  python scripts/evaluate.py \
    --dataset "$dataset" \
    --checkpoint "$final_checkpoint" \
    --partition test \
    --output "experiments/${dataset}/ablations/A8/seed_${SEED}/test_metrics.json"

  python scripts/generate_report.py --dataset "$dataset"
done

echo "Complete seed-${SEED} campaign finished."
