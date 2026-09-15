#!/usr/bin/env bash
# run_wave2.sh — attribution: roll ONE live driver per step, see which one the forecast follows.
# Same hardening as run_wave1.sh (see 03_harness_and_invariants.md §B). Ordered most-informative
# first, so a window that runs out still answers the main question.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-03_drivers_and_dynamics_dossier"
RES="$D/results"; OVN="$RES/overnight2"
MODELS="$HYD/../views-models/models"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
RUNNER="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"
export WANDB_MODE=offline WANDB_SILENT=true
ARM_TIMEOUT=${ARM_TIMEOUT:-3600}; STALL_MAX=${STALL_MAX:-900}; DEADLINE_S=${DEADLINE_S:-7200}
N_ORIGINS=13; START=$(date +%s)
mkdir -p "$OVN"
log(){ echo "[$(date '+%F %T')] $*" >> "$OVN/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$OVN/PHASE"; log "PHASE: $*"; }
exec 9>"$OVN/.lock"; flock -n 9 || { log "locked — exiting"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$OVN/repo.head"
log "=== wave2 attribution | timeout ${ARM_TIMEOUT}s deadline ${DEADLINE_S}s ==="
( while :; do date '+%s %F %T' > "$OVN/HEARTBEAT"; sleep 30; done ) & HB=$!
trap 'kill $HB 2>/dev/null; echo "ended=$(date "+%F %T")" > "$OVN/RUN_COMPLETE"; log "=== RUN_COMPLETE ==="' EXIT

# cell first (the hypothesised driver), then input (its rival), then hidden. Seed 42 then 43.
SPECS=( "fullzero_fortytwo|calibration_model_20260818_221401.pt|cell:90"
        "fullzero_fortytwo|calibration_model_20260818_221401.pt|input:90"
        "fullzero_fortytwo|calibration_model_20260818_221401.pt|hidden:90"
        "fullzero_fortythree|calibration_model_20260821_045948.pt|cell:90"
        "fullzero_fortythree|calibration_model_20260821_045948.pt|input:90"
        "fullzero_fortythree|calibration_model_20260821_045948.pt|hidden:90" )

cleanup(){ local gen="$MODELS/$1/data/generated" d
  for d in "$gen"/predictions_*; do [ -d "$d" ] && { log "  cleaning $(basename "$d")"; rm -rf "$d"; }; done
  rm -rf "$gen/_pf_staging"; }
guard(){ local m="$1"
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null || { log "GUARD FAIL: no CUDA"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) return 1;; esac; [ "$u" -lt 3000 ] || { log "GUARD FAIL: ${u}MiB foreign"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$OVN/repo.head")" ] || { log "GUARD FAIL: HEAD moved"; return 1; }
  [ -z "$(ls -d "$MODELS/$m"/data/generated/predictions_* 2>/dev/null)" ] || return 1
  [ ! -d "$MODELS/$m/data/generated/_pf_staging" ] || return 1; return 0; }

OK=0; FAILED=""; CONSEC=0
for spec in "${SPECS[@]}"; do
  M="${spec%%|*}"; rest="${spec#*|}"; ART="${rest%%|*}"; PSR="${rest##*|}"
  LABEL="identity_psr${PSR/:/}"; SENT="$RES/${M}_${LABEL}_DONE"; DUMP="$RES/bodymean_${M}_${LABEL}"
  [ -f "$SENT" ] && { log "SKIP $M/$PSR"; continue; }
  E=$(( $(date +%s) - START )); [ "$E" -gt "$DEADLINE_S" ] && { phase "DEADLINE at ${E}s"; break; }
  cleanup "$M"
  guard "$M" || { FAILED="$FAILED ${M}/${PSR}:guard"; CONSEC=$((CONSEC+1)); [ $CONSEC -ge 3 ] && break; continue; }
  rm -f "$RES/score_${M}_${LABEL}.csv"; rm -rf "$DUMP"
  phase "ARM $M / $PSR (elapsed ${E}s)"
  timeout --signal=TERM --kill-after=120 "$ARM_TIMEOUT" \
    "$PY" "$RUNNER" --model "$M" --artifact "$ART" --arms identity \
      --per-step-roll "$PSR" --body-mean-dump --tag "$LABEL" --out "$RES" \
      >> "$OVN/arm_${M}_${LABEL}.log" 2>&1 < /dev/null & APID=$!
  ( while kill -0 $APID 2>/dev/null; do n1=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l); sleep "$STALL_MAX"
      kill -0 $APID 2>/dev/null || break
      n2=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l)
      [ "$n1" = "$n2" ] && [ "$n2" -lt "$N_ORIGINS" ] && { echo "$(date '+%F %T') STALLED $M/$PSR at $n2" >> "$OVN/ANOMALIES.txt"; kill -TERM $APID; break; }
    done ) & WD=$!
  wait $APID; RC=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null
  NPZ=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l)
  if [ "$RC" -eq 0 ] && [ -f "$SENT" ] && [ "$NPZ" -eq "$N_ORIGINS" ]; then
    log "$M/$PSR OK"; OK=$((OK+1)); CONSEC=0
  else log "$M/$PSR FAILED rc=$RC npz=$NPZ"; FAILED="$FAILED ${M}/${PSR}:rc${RC}"; CONSEC=$((CONSEC+1))
    [ $CONSEC -ge 3 ] && { phase "ABORT: 3 consecutive"; cleanup "$M"; break; }; fi
  cleanup "$M"
done
log "=== $OK ok | failed:${FAILED:- none} ==="
