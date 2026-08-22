#!/usr/bin/env bash
# run_decay_dial.sh — is the cell freeze a switch or a dial?
#
# EXP-01/02 measured the two ENDPOINTS: weight 0 (`none`, 0.3318 at h18) and weight 1 (`cell`,
# 0.3709). The freeze recovers 23% of the oracle gap and leaves 77% open. If the cell degrades
# progressively rather than catastrophically, an intermediate pull should beat BOTH endpoints —
# and if the response is monotone instead, the dial is a switch and full freeze is the answer.
#
# Three interior points on seed 43, ~6 min each. Both endpoints are already measured on this exact
# vehicle, so this sweep buys the SHAPE for ~20 minutes and no training.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-22_state_freeze_l300_dossier"; RES="$D/results"
FT="$HYD/reports/2026-08-15_state_freeze_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
M=fullzero_fortythree
ART=calibration_model_20260821_045948.pt
ARMS="${DIAL_ARMS:-cell@0.25,cell@0.5,cell@0.75}"

mkdir -p "$RES"; rm -f "$RES/DIAL_DONE"
exec 8>"$RES/.dial.lock"; flock -n 8 || { echo "another dial run holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] dial: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
log "=== decay dial · HEAD ${HEAD_SHA:0:12} · $M · $ARMS ==="
free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
case "$free" in ''|*[!0-9]*) log "ABORT — cannot read df"; exit 13;; esac
[ "$free" -ge 25 ] || { log "ABORT — ${free}G free"; exit 13; }
n=$(ls -d "$MODELS/$M"/data/generated/predictions_* 2>/dev/null | wc -l)
[ "$n" -eq 0 ] || { log "ABORT — $n leftover cube(s) would be mixed in"; exit 3; }

t0=$(date +%s)
timeout -k 120 21600 $CENV python "$FT/run_freeze_arms.py" \
    --model "$M" --artifact "$ART" --arms "$ARMS" --out "$RES" >> "$RES/dial.log" 2>&1
rc=$?
for d in "$MODELS/$M"/data/generated/predictions_*; do
  [ -e "$d" ] && { log "  cleaning leftover $(basename "$d")"; rm -rf "$d"; }
done
[ $rc -eq 0 ] || { log "FAILED rc=$rc"; exit 4; }
log "OK in $(( ($(date +%s)-t0)/60 )) min"
touch "$RES/DIAL_DONE"
log "=== DONE ==="
