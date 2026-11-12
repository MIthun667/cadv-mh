#!/usr/bin/env bash
# Sequential seeds. Aborts if training/evaluation code or configs differ from the frozen revision after freezing.
set -uo pipefail
FROZEN="$1"
for S in 42 43 44 45 46; do
  [ -z "$(git diff "$FROZEN" -- src configs scripts/train.py scripts/pilot_diagnostics.py scripts/pilot_pattern_leakage_probe.py scripts/pilot2_mechanism.py scripts/run_confirm_seed.sh)" ] || { echo "REVISION CHANGED; stop"; exit 2; }
  [ -f experiments_v2/confirm/dataset_b/seed_$S/COMPLETE ] && continue
  bash scripts/run_confirm_seed.sh "$S" || { echo "seed $S FAILED"; exit 1; }
  echo "seed $S complete $(date)"
done
echo ALL_DONE
