#!/usr/bin/env bash
# confirm_cell_4seeds.sh — EXP-04: does the cell anchor hold at FOUR seeds?
#
# M38 measured +0.036 AP@h18 on TWO seeds (42, 43) with a paired origin-block CI on seed 43 only.
# This is the last experiment before shipping, so it exists to turn that into shipping evidence:
# all four seeds, a paired CI on each.
#
# EMIT ONLY. No training. The four `fullzero_*` artifacts already exist; `freeze_recurrent` is an
# inference-time setting on InferenceOrchestrator, not a config key and not a trained behaviour.
#
# Per seed: emit `none`, emit `cell`, keep both cubes, run the paired bootstrap, delete the cubes.
# Cubes are ~2.5 GB each and never coexist across seeds (C-154, the 37 GB scar).
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-22_state_freeze_l300_dossier"; RES="$D/results"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

SEEDS=(
  "fullzero_fortytwo|calibration_model_20260818_221401.pt"
  "fullzero_fortythree|calibration_model_20260821_045948.pt"
  "fullzero_fortyfour|calibration_model_20260821_082106.pt"
  "fullzero_fortyfive|calibration_model_20260821_120116.pt"
)

mkdir -p "$RES"; rm -f "$RES/CONFIRM4_DONE"
exec 8>"$RES/.confirm4.lock"; flock -n 8 || { echo "another confirm run holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] confirm4: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

log "=== EXP-04 · cell anchor at 4 seeds · HEAD $(cd "$HYD" && git rev-parse --short HEAD) ==="
FAILED=""
for spec in "${SEEDS[@]}"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  [ -f "$MODELS/$M/artifacts/$ART" ] || { log "SKIP $M — artifact missing"; FAILED="$FAILED $M"; continue; }
  if [ -s "$RES/paired_ci_${M}.json" ]; then log "SKIP $M — CI already computed"; continue; fi

  GEN="$MODELS/$M/data/generated"
  rm -rf "$GEN"/paired_none "$GEN"/paired_cell 2>/dev/null   # a half-finished seed must not be reused
  t0=$(date +%s)
  log "--- $M: emitting none + cell (cubes kept) ---"
  MODEL="$M" ARTIFACT="$ART" PAIR_ARMS="none cell" \
    bash "$D/tools/reemit_for_paired_ci.sh" >> "$RES/confirm4.log" 2>&1
  if [ $? -ne 0 ]; then log "$M FAILED at emit"; FAILED="$FAILED $M"; continue; fi

  log "--- $M: paired origin-block CI ---"
  $CENV python "$D/tools/paired_ci.py" --model "$M" --h 18 \
      --out "$RES/paired_ci_${M}.json" >> "$RES/confirm4.log" 2>&1
  rc=$?
  rm -rf "$GEN"/paired_none "$GEN"/paired_cell 2>/dev/null   # free ~5 GB before the next seed
  if [ $rc -ne 0 ]; then log "$M FAILED at bootstrap rc=$rc"; FAILED="$FAILED $M"; continue; fi
  log "$M OK in $(( ($(date +%s)-t0)/60 )) min"
done

[ -z "$FAILED" ] || { log "FAILED:$FAILED"; exit 1; }
touch "$RES/CONFIRM4_DONE"
log "=== EXP-04 DONE ==="
