#!/usr/bin/env bash
# run_screen.sh — the two BPTT-SA screen arms, trained back to back, unattended.
#
# Same hardening as the drivers dossier's wave launchers (03_harness §B there): no -e, one
# invocation per arm, per-arm timeout because a hang eats the window silently, CUDA and
# foreign-GPU gates because a wedged context becomes a CPU grind that looks like slow progress
# (C-163), stale-artifact cleanup so one crash does not cascade, HEAD guard, heartbeat/phase,
# and a summary that names failures so a partial run is legible.
#
# Training, not emit: the timeout is 5 h per arm against an expected 2-3 h.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results"; OVN="$RES/overnight"
CENV="conda run --no-capture-output -n views-hydranet-env"
export WANDB_MODE=offline WANDB_SILENT=true          # C-163: wandb.finish() has DNS-hung 38 min here
ARM_TIMEOUT=${ARM_TIMEOUT:-18000}                    # 5 h
ARMS=(ssdetached_fortytwo ssattached_fortytwo)       # detached FIRST: it reproduces the known result
mkdir -p "$OVN"; START=$(date +%s)
log(){ echo "[$(date '+%F %T')] $*" >> "$OVN/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$OVN/PHASE"; log "PHASE: $*"; }
exec 9>"$OVN/.lock"; flock -n 9 || { log "locked — exiting"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$OVN/repo.head"
log "=== BPTT-SA screen: ${#ARMS[@]} training arms, timeout ${ARM_TIMEOUT}s ==="
( while :; do date '+%s %F %T' > "$OVN/HEARTBEAT"; sleep 30; done ) & HB=$!
trap 'kill $HB 2>/dev/null; echo "ended=$(date "+%F %T") elapsed_min=$(( ($(date +%s)-START)/60 ))" > "$OVN/RUN_COMPLETE"; log "=== RUN_COMPLETE ==="' EXIT

guard(){ local m="$1"
  $CENV python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    || { log "GUARD FAIL: no CUDA — refusing a silent CPU grind (C-163)"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) log "GUARD FAIL: nvidia-smi unreadable"; return 1;; esac
  [ "$u" -lt 3000 ] || { log "GUARD FAIL: ${u} MiB already on the GPU"; return 1; }
  local f; f=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "${f:-}" in ''|*[!0-9]*) return 1;; esac
  [ "$f" -ge 25 ] || { log "GUARD FAIL: ${f}G free"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$OVN/repo.head")" ] \
    || { log "GUARD FAIL: repo HEAD moved mid-run — the arms would not be comparable"; return 1; }
  return 0; }

OK=0; FAILED=""
for ARM in "${ARMS[@]}"; do
  A="$MODELS/$ARM"
  if ls "$A"/artifacts/*.pt >/dev/null 2>&1; then log "SKIP $ARM — artifact already present"; continue; fi
  guard "$ARM" || { FAILED="$FAILED ${ARM}:guard"; continue; }
  rm -rf "$A"/data/generated/predictions_* "$A"/data/generated/_pf_staging 2>/dev/null
  phase "TRAIN $ARM (elapsed $(( ($(date +%s)-START)/60 )) min)"
  ( cd "$A" && timeout -k 120 "$ARM_TIMEOUT" $CENV python main.py -r calibration -t -e -sa ) \
      >> "$OVN/arm_${ARM}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && ls "$A"/artifacts/*.pt >/dev/null 2>&1; then
    log "$ARM OK ($(ls "$A"/artifacts/*.pt | wc -l) artifact)"; OK=$((OK+1))
  else
    case $rc in 124|137) R=timeout;; *) R="rc$rc";; esac
    log "$ARM FAILED ($R)"; FAILED="$FAILED ${ARM}:${R}"
  fi
done
log "=== $OK ok | failed:${FAILED:- none} ==="
{ echo "# BPTT-SA screen — run summary"; echo
  echo "- finished: $(date '+%F %T')"; echo "- elapsed: $(( ($(date +%s)-START)/60 )) min"
  echo "- arms trained: $OK   FAILED: ${FAILED:- none}"; echo
  for ARM in "${ARMS[@]}"; do
    echo "## $ARM"
    ls "$MODELS/$ARM"/artifacts/*.pt 2>/dev/null | sed 's|.*/|- artifact: |' || echo "- no artifact"
  done; } > "$RES/SCREEN_SUMMARY.md"
