#!/usr/bin/env bash
# run_wave1.sh — 4 freeze arms x 4 seeds, emit-only, fully unattended.
#
# Shaped on 2026-08-17_vehicle_replication_dossier/tools/overnight_run.sh and
# 2026-08-21_persistence_reference_dossier/tools/run_all_seeds.sh. Every property below exists
# because something already went wrong without it; the register entry or measurement is named.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-03_drivers_and_dynamics_dossier"
RES="$D/results"; OVN="$RES/overnight"
MODELS="$HYD/../views-models/models"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
RUNNER="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"

# C-163: wandb run.finish() has DNS-hung for 38 min on this box. overnight_run.sh set exactly this
# and drove six arms of THIS driver through it successfully on 2026-08-17.
export WANDB_MODE=offline WANDB_SILENT=true

# 3600, not 1800: the `none` arm is MEASURED at 2127.8 s on this exact model+driver (manifest,
# 2026-09-02). A 30-min ceiling would have killed it and left a cube, taking the seed's other
# three arms with it via the "refusing to start" guard.
ARM_TIMEOUT=${ARM_TIMEOUT:-3600}
STALL_MAX=${STALL_MAX:-900}          # no new origin dumped in 15 min => stuck (CPU grind / hang)
DEADLINE_S=${DEADLINE_S:-23400}      # 6.5 h: stop STARTING arms, so the finisher runs inside 8 h
N_ORIGINS=13

SEEDS=(
  "fullzero_fortytwo|calibration_model_20260818_221401.pt"
  "fullzero_fortythree|calibration_model_20260821_045948.pt"
  "fullzero_fortyfour|calibration_model_20260821_082106.pt"
  "fullzero_fortyfive|calibration_model_20260821_120116.pt"
)
ARMS=(none hidden cell all)   # `none` => omit --freeze entirely; it has no such choice

mkdir -p "$OVN"
START=$(date +%s)
log(){ echo "[$(date '+%F %T')] $*" >> "$OVN/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$OVN/PHASE"; log "PHASE: $*"; }

exec 9>"$OVN/.lock"
flock -n 9 || { log "another wave1 holds the lock — exiting"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$OVN/repo.head"
log "=== wave1: ${#SEEDS[@]} seeds x ${#ARMS[@]} arms | timeout ${ARM_TIMEOUT}s stall ${STALL_MAX}s deadline ${DEADLINE_S}s ==="
log "repo HEAD $(cat "$OVN/repo.head")"

( while :; do date '+%s %F %T' > "$OVN/HEARTBEAT"; sleep 30; done ) & HB=$!
finish(){
  kill $HB 2>/dev/null
  phase "FINISHING"
  "$PY" "$D/tools/verify_wave1.py" >> "$OVN/run.log" 2>&1
  echo "ended=$(date '+%F %T') elapsed_min=$(( ($(date +%s) - START) / 60 ))" > "$OVN/RUN_COMPLETE"
  log "=== RUN_COMPLETE ==="
}
trap finish EXIT

label_for(){ [ "$1" = none ] && echo identity || echo "identity_freeze$1"; }

# Runs after EVERY arm, pass or fail. _pf_staging is the one nothing else cleans: pipeline-core
# removes it only on the success path (model.py:1596), it is indexed by sequential origin_i so a
# stale tree is shape-compatible with the next arm's, and _prediction_dirs globs only predictions_*
# so no guard in this repo can see it.
cleanup(){
  local m="$1" gen="$MODELS/$1/data/generated" d
  for d in "$gen"/predictions_*; do
    [ -d "$d" ] || continue
    log "  cleaning $(basename "$d") in $m"; rm -rf "$d"
  done
  rm -rf "$gen/_pf_staging"
}

guard(){
  local m="$1"
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    || { log "GUARD FAIL: CUDA unavailable — refusing a silent CPU grind (C-163)"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) log "GUARD FAIL: nvidia-smi unreadable"; return 1;; esac
  [ "$u" -lt 3000 ] || { log "GUARD FAIL: ${u} MiB already on the GPU (ollama serve is running)"; return 1; }
  local f; f=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "${f:-}" in ''|*[!0-9]*) log "GUARD FAIL: df unreadable"; return 1;; esac
  [ "$f" -ge 25 ] || { log "GUARD FAIL: ${f}G free, need 25"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$OVN/repo.head")" ] \
    || { log "GUARD FAIL: repo HEAD moved mid-run — arms would not be comparable"; return 1; }
  [ -z "$(ls -d "$MODELS/$m"/data/generated/predictions_* 2>/dev/null)" ] || { log "GUARD FAIL: stale pred dir"; return 1; }
  [ ! -d "$MODELS/$m/data/generated/_pf_staging" ] || { log "GUARD FAIL: stale _pf_staging"; return 1; }
  return 0
}

FAILED=""; OK=0; CONSEC=0
for spec in "${SEEDS[@]}"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  for ARM in "${ARMS[@]}"; do
    LABEL=$(label_for "$ARM")
    SENT="$RES/${M}_${LABEL}_DONE"      # written by run_realism_arms.py AFTER scoring AND cube delete
    DUMP="$RES/bodymean_${M}_${LABEL}"

    if [ -f "$SENT" ]; then log "SKIP $M/$LABEL — sentinel present"; continue; fi
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt "$DEADLINE_S" ]; then phase "DEADLINE at ${ELAPSED}s — no new arms"; break 2; fi

    cleanup "$M"
    if ! guard "$M"; then FAILED="$FAILED ${M}/${ARM}:guard"; CONSEC=$((CONSEC+1))
      [ "$CONSEC" -ge 3 ] && { phase "ABORT: 3 consecutive failures — systemic"; break 2; }; continue; fi

    # A crashed earlier attempt leaves evidence that reads as this attempt's.
    rm -f "$RES/score_${M}_${LABEL}.csv" "$RES/fedfield_${M}_${LABEL}.csv"; rm -rf "$DUMP"

    phase "ARM $M/$ARM (elapsed ${ELAPSED}s)"
    FREEZE=(); [ "$ARM" = none ] || FREEZE=(--freeze "$ARM")
    timeout --signal=TERM --kill-after=120 "$ARM_TIMEOUT" \
      "$PY" "$RUNNER" --model "$M" --artifact "$ART" --arms identity \
        "${FREEZE[@]}" --body-mean-dump --tag "$LABEL" --out "$RES" \
        >> "$OVN/arm_${M}_${LABEL}.log" 2>&1 < /dev/null &
    APID=$!

    # Progress watchdog. The runner's own stdout does NOT grow during an arm (it writes the child's
    # log only after the child exits), so the live signal is the dump directory: one npz per origin.
    ( while kill -0 $APID 2>/dev/null; do
        n1=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l); sleep "$STALL_MAX"
        kill -0 $APID 2>/dev/null || break
        n2=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l)
        if [ "$n1" = "$n2" ] && [ "$n2" -lt "$N_ORIGINS" ]; then
          echo "$(date '+%F %T') STALLED $M/$LABEL at ${n2}/${N_ORIGINS} origins" >> "$OVN/ANOMALIES.txt"
          kill -TERM $APID 2>/dev/null; break
        fi
      done ) & WD=$!
    wait $APID; RC=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null

    NPZ=$(ls "$DUMP"/*.npz 2>/dev/null | wc -l)
    if [ "$RC" -eq 0 ] && [ -f "$SENT" ] && [ "$NPZ" -eq "$N_ORIGINS" ]; then
      log "$M/$LABEL OK (${NPZ} origins)"; OK=$((OK+1)); CONSEC=0
    else
      case "$RC" in
        124|137) R="timeout";;
        *) R="rc$RC";;
      esac
      [ "$NPZ" -ne "$N_ORIGINS" ] && R="$R,npz${NPZ}/${N_ORIGINS}"
      [ -f "$SENT" ] || R="$R,nosentinel"
      log "$M/$LABEL FAILED ($R)"; FAILED="$FAILED ${M}/${ARM}:${R}"; CONSEC=$((CONSEC+1))
      [ "$CONSEC" -ge 3 ] && { phase "ABORT: 3 consecutive failures — systemic"; cleanup "$M"; break 2; }
    fi
    cleanup "$M"
    "$PY" "$D/tools/verify_wave1.py" >> "$OVN/run.log" 2>&1
  done
done
log "=== $OK ok | failed:${FAILED:- none} ==="
