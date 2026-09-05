#!/usr/bin/env bash
# score_arms.sh — free-running emit + score for the SCREEN arms.
#
# REWRITTEN 2026-09-04. The previous version hardcoded `ssdetached_fortytwo` and
# `ssattached_fortytwo` together with two literal artifact filenames — and those artifacts were
# the ones from the VOID first run, the inert-treatment pair that trained to byte-identical
# weights. Running it unchanged would have emitted, scored and reported the wrong models entirely,
# with nothing in the output saying so. Arms now come from the environment and artifacts are
# resolved from disk, and both are ECHOED into the log so the scored pair is always on the record.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results/screen3"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
RUNNER="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"
MODELS="$HYD/../views-models/models"
export WANDB_MODE=offline WANDB_SILENT=true
mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] $*" >> "$RES/score.log"; return 0; }

ARMS=(${SCREEN_ARMS:-screendetached_fortytwo screenattached_fortytwo})
log "=== scoring arms: ${ARMS[*]} ==="

for M in "${ARMS[@]}"; do
  # Resolve the artifact instead of trusting a literal. Newest wins, and it is logged, because an
  # arm directory can carry stale .pt files from an earlier run of the same name.
  ART=$(ls -t "$MODELS/$M"/artifacts/*.pt 2>/dev/null | head -1)
  if [ -z "$ART" ]; then
    log "!! $M has NO .pt artifact — BRANCH 0 (the arm did not train). Not scored, not estimated."
    echo "$M" >> "$RES/NO_ARTIFACT"
    continue
  fi
  ART=$(basename "$ART")
  log "$M -> $ART"
  [ -s "$RES/score_${M}_identity.csv" ] && { log "SKIP $M (already scored)"; continue; }
  rm -rf "$MODELS/$M"/data/generated/predictions_* "$MODELS/$M"/data/generated/_pf_staging 2>/dev/null
  timeout -k 120 5400 "$PY" "$RUNNER" --model "$M" --artifact "$ART" --arms identity \
    --body-mean-dump --tag identity --out "$RES" >> "$RES/emit_${M}.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && log "$M emit OK" || log "$M emit FAILED rc=$rc"
  rm -rf "$MODELS/$M"/data/generated/predictions_* "$MODELS/$M"/data/generated/_pf_staging 2>/dev/null
done
touch "$RES/SCORED"
log "=== emit done — applying the pre-registered rule ==="
"$PY" "$D/tools/read_screen.py" "${ARMS[@]}" >> "$RES/score.log" 2>&1
