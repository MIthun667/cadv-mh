#!/usr/bin/env bash
# usage: run_confirm_seed.sh SEED   (Dataset B confirmation; validation only; protocol: docs/confirmatory_protocol.md)
set -euo pipefail
S="$1"
R=experiments_v2/confirm/dataset_b/seed_$S
mkdir -p "$R/logs"
post() {  # NAME CKPT : diagnostics, pattern probe, mechanism analysis
  local N="$1" C="$2"
  python scripts/pilot_diagnostics.py --dataset dataset_b --checkpoint "$C" --output-dir "$R/diagnostics_$N" > "$R/logs/${N}_diag.log" 2>&1
  python scripts/pilot_pattern_leakage_probe.py --checkpoint "$C" --output "$R/diagnostics_$N/pattern_leakage_probe.json" > "$R/logs/${N}_pattern.log" 2>&1
  python scripts/pilot2_mechanism.py --checkpoints "$N=$C" --output "$R/mechanism_$N.json" > "$R/logs/${N}_mechanism.log" 2>&1
}
a0() {
  python scripts/train.py --dataset dataset_b --seed "$S" --ablation A0 --output-dir "$R/A0" > "$R/logs/A0_train.log" 2>&1
  post A0 "$R/A0/best.pt"; echo done > "$R/A0/COMPLETE"
}
variant() {  # NAME CONFIG
  local N="$1" CFG="$2"
  python scripts/train.py --dataset dataset_b --seed "$S" --config "$CFG" --stage 2 --init-from "$R/shared/stage1_best.pt" --output-dir "$R/$N" > "$R/logs/${N}_stage2.log" 2>&1
  python scripts/train.py --dataset dataset_b --seed "$S" --config "$CFG" --stage 3 --init-from "$R/$N/stage2_best.pt" --output-dir "$R/$N" > "$R/logs/${N}_stage3.log" 2>&1
  post "$N" "$R/$N/best.pt"; echo done > "$R/$N/COMPLETE"
}
a0 & PA=$!
python scripts/train.py --dataset dataset_b --seed "$S" --stage 1 --output-dir "$R/shared" > "$R/logs/stage1.log" 2>&1
variant B configs/pilot2/B_contrastive_prob.yaml & PB=$!
variant V1 configs/pilot2/V1_no_necessity.yaml & PV=$!
wait $PA; wait $PB; wait $PV
echo done > "$R/COMPLETE"
