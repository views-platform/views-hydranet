#!/usr/bin/env bash
# run_all_seeds.sh — the other three eps=0 L=300 seeds, so M34 stops being an n=1 claim.
#
# The ledger's own standard, in the section M34 edits: "No result here is multi-seed or
# multi-vehicle. Positive findings at n=1 have historically evaporated on proper runs." M8 was
# just demoted (#280) on exactly that count, so M34 does not get a pass on it for being good news.
#
# Emit-only on existing weights, ~7 min per seed. Serial by design: run_persistence_ref.sh takes
# a flock, and two concurrent emits would both write to the same model's data/generated tree.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-21_persistence_reference_dossier"; RES="$D/results"

# seed 43 is already done (EXP-01/02)
SEEDS=(
  "fullzero_fortytwo|calibration_model_20260818_221401.pt"
  "fullzero_fortyfour|calibration_model_20260821_082106.pt"
  "fullzero_fortyfive|calibration_model_20260821_120116.pt"
)

log(){ echo "[$(date '+%F %T')] all_seeds: $*" | tee -a "$RES/run.log" >&2; }
log "=== $((${#SEEDS[@]})) remaining seeds ==="
FAILED=""
for spec in "${SEEDS[@]}"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  if [ -s "$RES/score_persistence_ref_$M.csv" ]; then log "SKIP $M — already scored"; continue; fi
  log "--- $M ---"
  bash "$D/tools/run_persistence_ref.sh" "$M" "$ART"
  rc=$?
  [ $rc -eq 0 ] && log "$M OK" || { log "$M FAILED rc=$rc"; FAILED="$FAILED $M"; }
done
[ -z "$FAILED" ] || { log "FAILED:$FAILED"; exit 1; }
touch "$RES/ALL_SEEDS_DONE"
log "=== all seeds done ==="
