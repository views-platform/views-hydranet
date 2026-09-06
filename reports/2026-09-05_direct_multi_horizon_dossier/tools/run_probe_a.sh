#!/usr/bin/env bash
# PROBE-A (#324): does removing self-feedback at INFERENCE beat the clamped control?
#
# `hold_last_real` + `--freeze cell` reproduces direct multi-horizon's INFERENCE semantics on an
# already-trained artifact: state held, input constant at the origin's real field, nothing the
# model emits ever fed back. If it does not beat the clamped control, the epic's remaining claim
# is TRAINING-time alignment only.
#
# ⚠️ Driven through run_realism_arms.py, NOT realism_arm_entry.py directly. The first attempt
# called the entry per-arm and lost every cube but the last: the pipeline names the prediction
# dir after the ARTIFACT, not the arm, so each arm silently overwrote its predecessor. The
# driver scores each arm and deletes its cube before the next runs, and refuses to start on a
# leftover prediction dir. That guard exists because this exact mistake was made before.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
RUN="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"
OUT="$HYD/reports/2026-09-05_direct_multi_horizon_dossier/results"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
mkdir -p "$OUT"

declare -A ART=(
  [fortytwo]=calibration_model_20260818_221401.pt
  [fortythree]=calibration_model_20260821_045948.pt
  [fortyfour]=calibration_model_20260821_082106.pt
  [fortyfive]=calibration_model_20260821_120116.pt
)

for seed in fortytwo fortythree fortyfour fortyfive; do
  # Batch 1: the clamped control and the clamped probe — both need --freeze cell.
  if [ ! -f "$OUT/done_${seed}_clamped" ]; then
    echo "=== $seed clamped (identity, hold_last_real) $(date +%H:%M:%S) ==="
    "$PY" "$RUN" --model "fullzero_${seed}" --artifact "${ART[$seed]}" \
      --arms identity,hold_last_real --freeze cell \
      --out "$OUT" --tag "probea_${seed}_clamped" > "$OUT/log_${seed}_clamped.txt" 2>&1
    rc=$?; echo "  exit=$rc"; [ $rc -eq 0 ] && touch "$OUT/done_${seed}_clamped"
  fi
  # Batch 2: the probe WITHOUT the clamp, to separate the two effects.
  if [ ! -f "$OUT/done_${seed}_free" ]; then
    echo "=== $seed unclamped (hold_last_real) $(date +%H:%M:%S) ==="
    "$PY" "$RUN" --model "fullzero_${seed}" --artifact "${ART[$seed]}" \
      --arms hold_last_real \
      --out "$OUT" --tag "probea_${seed}_free" > "$OUT/log_${seed}_free.txt" 2>&1
    rc=$?; echo "  exit=$rc"; [ $rc -eq 0 ] && touch "$OUT/done_${seed}_free"
  fi
done
echo "PROBE-A COMPLETE $(date)" > "$OUT/PROBE_A_DONE"
