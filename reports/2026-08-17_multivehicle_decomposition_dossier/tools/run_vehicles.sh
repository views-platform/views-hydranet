#!/usr/bin/env bash
# run_vehicles.sh — the placement decomposition on the vehicles the floor gate admitted.
#
# Inference-only: five arms per vehicle on its existing artifact, ~10 min each. The control is each
# vehicle's preserved production cube, already scored, so no control is re-run.
#
# Reuses run_realism_arms.py unchanged (leftover refusal, disk preflight, score-then-delete, manifest,
# per-arm sentinel). Resumable: an arm whose score CSV exists is skipped.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-17_multivehicle_decomposition_dossier"; RES="$D/results"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
HORIZONS=1,6,12,18,24,30,36

# vehicle:artifact_timestamp — pinned from partition_audit.json, sha-verified before each vehicle
VEHICLES=(
  "purple_alien:20260813_062540:89135913e0ad"
  "blue_stranger:20260813_042946:2b51feda86ab"
  "blazing_meteor:20260812_232850:a9c7136234c2"
)
ARMS=(use_real spatial_scramble occurrence_real_magnitude_model occurrence_model_magnitude_real thin:0.75)

mkdir -p "$RES"
rm -f "$RES/VEHICLES_DONE"          # a stale sentinel turns "not finished" into "finished"
exec 9>"$RES/.lock"; flock -n 9 || { echo "another run holds the lock"; exit 11; }
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$RES/run.log"; }

log "=== ${#VEHICLES[@]} vehicles x ${#ARMS[@]} arms ==="
for spec in "${VEHICLES[@]}"; do
  IFS=: read -r M TS SHA <<<"$spec"
  VV="$MODELS/$M"; GEN="$VV/data/generated"; ART="calibration_model_${TS}.pt"

  [ -d "$VV" ] || { log "SKIP $M — model dir missing"; continue; }
  got=$(sha256sum "$VV/artifacts/$ART" 2>/dev/null | cut -c1-12)
  [ "$got" = "$SHA" ] || { log "SKIP $M — artifact sha $got != audit's $SHA"; continue; }
  n=$(ls -d "$GEN"/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { log "SKIP $M — $n leftover cube(s); two arms' cubes would mix"; continue; }

  for arm in "${ARMS[@]}"; do
    label="${arm/:/_}"
    if [ -s "$RES/score_${M}_${label}.csv" ]; then log "SKIP $M/$arm — scored"; continue; fi
    free=$(df -BG "$VV" | tail -1 | awk '{print $4}' | tr -d 'G')
    [ "$free" -ge 20 ] || { log "ABORT — ${free}G free"; break 2; }

    log "--- $M / $arm ---"
    t0=$(date +%s)
    $CENV python "$RT/run_realism_arms.py" --model "$M" --artifact "$ART" \
        --arms "$arm" --tag "$label" --out "$RES" >> "$RES/${M}_${label}.log" 2>&1
    rc=$?
    # the driver names outputs score_<model>_<label>.csv; confirm it landed
    if [ $rc -eq 0 ] && [ -s "$RES/score_${M}_${label}.csv" ]; then
      log "$M/$arm OK in $(( ($(date +%s)-t0)/60 )) min"
    else
      log "$M/$arm FAILED rc=$rc (continuing)"
      tail -3 "$RES/${M}_${label}.log" >> "$RES/run.log" 2>/dev/null
    fi
    for d in "$GEN"/predictions_*; do
      [ -e "$d" ] && { log "  cleaning leftover $(basename "$d")"; rm -rf "$d"; }
    done
  done
done

log "=== COMPLETE ==="
touch "$RES/VEHICLES_DONE"
