#!/usr/bin/env bash
# run_st_bias.sh — EXP-BIAS (#308 Phase 1) on both trained SCREEN-3 arms.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
M="$HYD/../views-models/models"
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results/stbias"
CENV="conda run --no-capture-output -n views-hydranet-env"
export WANDB_MODE=offline WANDB_SILENT=true
mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] $*" >> "$RES/run.log"; return 0; }
N=${N:-256}; STEPS=${STEPS:-1,3,5}
log "=== EXP-BIAS: n_draws=$N steps=$STEPS ==="
for ARM in screendetached_fortytwo screenattached_fortytwo; do
  ART=$(ls -t "$M/$ARM"/artifacts/*.pt 2>/dev/null | head -1)
  [ -z "$ART" ] && { log "!! $ARM has no artifact — skipped"; continue; }
  log "--- $ARM <- $(basename "$ART")"
  ( cd "$M/$ARM" && timeout -k 60 5400 $CENV python "$D/tools/st_bias_entry.py" \
      --model-dir "$M/$ARM" --artifact "$(basename "$ART")" \
      --out "$RES/stbias_${ARM}.json" --n-draws "$N" --steps "$STEPS" ) \
    >> "$RES/${ARM}.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && log "$ARM OK" || log "$ARM FAILED rc=$rc"
  grep -h "^step " "$RES/${ARM}.log" | sed "s/^/  $ARM  /" >> "$RES/run.log"
done
log "=== done ==="
touch "$RES/STBIAS_DONE"
