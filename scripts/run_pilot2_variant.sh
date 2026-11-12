#!/usr/bin/env bash
# usage: run_pilot2_variant.sh NAME CONFIG   (validation-only; stages 2 and 3 from the shared stage-1 checkpoint)
set -euo pipefail
NAME="$1"; CONFIG="$2"
B=experiments_v2/pilot2/dataset_b
D=$B/$NAME
mkdir -p "$D" "$B/logs"
python scripts/train.py --dataset dataset_b --config "$CONFIG" --stage 2 --init-from "$B/stage1_best.pt" --output-dir "$D" > "$B/logs/${NAME}_stage2.log" 2>&1
python scripts/train.py --dataset dataset_b --config "$CONFIG" --stage 3 --init-from "$D/stage2_best.pt" --output-dir "$D" > "$B/logs/${NAME}_stage3.log" 2>&1
python scripts/pilot_diagnostics.py --dataset dataset_b --checkpoint "$D/best.pt" --output-dir "$B/diagnostics_$NAME" > "$B/logs/${NAME}_diag.log" 2>&1
python scripts/pilot_pattern_leakage_probe.py --checkpoint "$D/best.pt" --output "$B/diagnostics_$NAME/pattern_leakage_probe.json" > "$B/logs/${NAME}_pattern.log" 2>&1
python scripts/pilot2_mechanism.py --checkpoints "$NAME=$D/best.pt" --output "$B/mechanism_$NAME.json" > "$B/logs/${NAME}_mechanism.log" 2>&1
echo done > "$D/COMPLETE"
