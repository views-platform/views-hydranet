#!/usr/bin/env bash
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
RUNNER="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"
MODELS="$HYD/../views-models/models"
export WANDB_MODE=offline WANDB_SILENT=true
log(){ echo "[$(date '+%F %T')] $*" >> "$RES/score.log"; return 0; }
for spec in "ssdetached_fortytwo|calibration_model_20260903_141102.pt" \
            "ssattached_fortytwo|calibration_model_20260903_160027.pt"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  [ -s "$RES/score_${M}_identity.csv" ] && { log "SKIP $M"; continue; }
  rm -rf "$MODELS/$M"/data/generated/predictions_* "$MODELS/$M"/data/generated/_pf_staging 2>/dev/null
  log "--- emit $M ---"
  timeout -k 120 5400 "$PY" "$RUNNER" --model "$M" --artifact "$ART" --arms identity \
    --body-mean-dump --tag identity --out "$RES" >> "$RES/emit_${M}.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && log "$M OK" || log "$M FAILED rc=$rc"
  rm -rf "$MODELS/$M"/data/generated/predictions_* "$MODELS/$M"/data/generated/_pf_staging 2>/dev/null
done
touch "$RES/SCORED"
log "=== scoring done ==="
