#!/usr/bin/env bash
# run_freeze_l300.sh — does freezing recurrent state still help at 300 lessons?
#
# M8 says freezing recurrent state recovers gate AP h18 0.0070 -> 0.0912. It was measured on
# `truncated_smoke`: 40 lessons, ONE seed. M28 now classifies that vehicle as a smoke test with no
# skill at any horizon, and the pre-registered violet_visitor confirmation was never run — which is
# why M8 is the primary suspect in #280.
#
# The number that makes this worth an overnight: M8's RECOVERED value (0.0912) is 3.6x BELOW what a
# 300-lesson model scores free-running with no intervention at all (0.3298, M34). So we do not know
# whether state-freezing helps a model that actually works, does nothing, or hurts.
#
# Emit-only on existing weights. No training, no new code — run_freeze_arms.py already takes any
# model+artifact.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-22_state_freeze_l300_dossier"; RES="$D/results"
FT="$HYD/reports/2026-08-15_state_freeze_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

# two seeds, so a difference is not one draw. Both are eps=0 L=300 arms whose free-running AP is
# already published (M34): seed 43 h18 0.3318, seed 42 h18 0.3298.
SEEDS=(
  "fullzero_fortythree|calibration_model_20260821_045948.pt"
  "fullzero_fortytwo|calibration_model_20260818_221401.pt"
)
ARMS=none,hidden,cell,all

mkdir -p "$RES"; rm -f "$RES/FREEZE_DONE"
exec 6>"$RES/.freeze.lock"; flock -n 6 || { echo "another run holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] freeze300: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
log "=== state-freeze at L=300 · HEAD ${HEAD_SHA:0:12} · ${#SEEDS[@]} seeds x 4 arms ==="
FAILED=""
for spec in "${SEEDS[@]}"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "$free" in ''|*[!0-9]*) log "ABORT — cannot read df"; exit 13;; esac
  [ "$free" -ge 25 ] || { log "ABORT — ${free}G free"; exit 13; }
  [ -f "$MODELS/$M/artifacts/$ART" ] || { log "SKIP $M — artifact missing"; FAILED="$FAILED $M"; continue; }
  n=$(ls -d "$MODELS/$M"/data/generated/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { log "ABORT — $n leftover cube(s) in $M"; exit 3; }

  log "--- $M ($ART) · arms $ARMS ---"
  t0=$(date +%s)
  timeout -k 120 21600 $CENV python "$FT/run_freeze_arms.py" \
      --model "$M" --artifact "$ART" --arms "$ARMS" --out "$RES" >> "$RES/${M}_freeze.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && log "$M OK in $(( ($(date +%s)-t0)/60 )) min" || { log "$M FAILED rc=$rc"; FAILED="$FAILED $M"; }
  for d in "$MODELS/$M"/data/generated/predictions_*; do
    [ -e "$d" ] && { log "  cleaning leftover $(basename "$d")"; rm -rf "$d"; }
  done
done
echo "$HEAD_SHA" > "$RES/repo.head"
[ -z "$FAILED" ] || { log "FAILED:$FAILED"; exit 1; }
touch "$RES/FREEZE_DONE"
log "=== DONE ==="
