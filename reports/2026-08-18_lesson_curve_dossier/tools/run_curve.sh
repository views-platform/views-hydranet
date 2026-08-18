#!/usr/bin/env bash
# run_curve.sh — the lesson curve, staged so that every prefix is a complete result.
#
# Stage order is not arbitrary. The noise floor comes FIRST, because a one-seed lesson point cannot be
# read without it: without sigma_seed a difference of 0.10 in retention is neither an effect nor a null,
# it is a number. Then each lesson point completes with its own oracle before the next begins, so an
# interruption never leaves a control without its ceiling.
#
# Resumable at arm granularity (each runner skips an arm whose score CSV exists). There is no
# mid-training checkpoint, so an interrupted arm restarts from zero — which is why the long ones run
# last and why the stage gates are real.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-18_lesson_curve_dossier"; RES="$D/results"; T="$D/tools"
SSD="$HYD/reports/2026-08-17_ss_retention_dossier"; SSRES="$SSD/results"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

BUDGET_HOURS="${BUDGET_HOURS:-26}"
STAGE4_MIN_HOURS=8            # a 900-lesson arm is ~6.2 h and indivisible; do not start one late
START=$(date +%s)

mkdir -p "$RES"
rm -f "$RES/CURVE_COMPLETE"   # a stale sentinel turns "not finished" into "finished"
exec 9>"$RES/.curve.lock"; flock -n 9 || { echo "another curve run holds the lock"; exit 11; }
# stderr, not stdout: `label=$(build ...)` captures this function's callers, and a log line
# landing in a model directory name is the kind of bug that only shows up at 3am.
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$RES/run.log" >&2; }
elapsed_h(){ echo "scale=2; ($(date +%s) - $START) / 3600" | bc; }
remaining_h(){ echo "scale=2; $BUDGET_HOURS - ($(date +%s) - $START) / 3600" | bc; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
echo "$HEAD_SHA" > "$RES/curve.head"
log "=== lesson curve · HEAD ${HEAD_SHA:0:12} · budget ${BUDGET_HOURS}h ==="

assert_head(){
  now=$(cd "$HYD" && git rev-parse HEAD)
  [ "$now" = "$HEAD_SHA" ] || { log "ABORT — HEAD moved ${HEAD_SHA:0:12} -> ${now:0:12}; arms would not be comparable"; exit 12; }
}
assert_disk(){
  free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$free" -ge 25 ] || { log "ABORT — ${free}G free; the oracle step refuses below 25"; exit 13; }
}
verdict_state(){ $CENV python "$T/verify_curve.py" >> "$RES/run.log" 2>&1
  $CENV python -c "import json,sys;print(json.load(open('$RES/curve_state.json'))['state'])" 2>/dev/null; }

build(){  # build <lessons> <seed> -> echoes the label
  local L=$1 S=$2 label
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$SSD/tools')
from make_ss_arm import arm_label; print(arm_label(lessons=$L, eps=0.0, seed=$S))" 2>/dev/null)
  [ -n "$label" ] || { log "ABORT — no legal arm name for lessons=$L seed=$S"; return 1; }
  if [ ! -d "$MODELS/$label" ]; then
    log "building $label (lessons=$L seed=$S)"
    $CENV python "$SSD/tools/make_ss_arm.py" --lessons "$L" --eps 0.0 --seed "$S" \
      >> "$RES/run.log" 2>&1 || { log "ABORT — make_ss_arm refused $label"; return 1; }
  fi
  echo "$label"
}

# =============================================================== stage 1 — the noise floor
log "--- stage 1: sigma_seed at L=160, and the anchor's ceiling ---"
assert_head; assert_disk

# 1a. the L=160 oracle. Its control already exists in the SS dossier (longzero_fortytwo, gate PASS),
#     so only the ceiling is missing. Called directly rather than through run_lesson_arm.sh, which
#     would not find that control in THIS dossier and would retrain a model we already have.
if [ -s "$RES/score_longzero_fortytwo_use_real.csv" ]; then
  log "SKIP anchor oracle — already scored"
else
  ART=$($CENV python -c "
import json;print(json.load(open('$SSRES/arm_longzero_fortytwo.json'))['artifact'])" 2>/dev/null)
  if [ -z "$ART" ]; then
    log "ABORT — cannot name longzero_fortytwo's artifact; the anchor has no ceiling"; exit 14
  fi
  n=$(ls -d "$MODELS/longzero_fortytwo"/data/generated/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { log "ABORT — $n leftover cube in longzero_fortytwo"; exit 15; }
  log "anchor oracle (use_real) on $ART"
  t0=$(date +%s)
  timeout -k 120 5400 $CENV python "$RT/run_realism_arms.py" --model longzero_fortytwo \
    --artifact "$ART" --arms use_real --tag oracle --out "$RES" \
    >> "$RES/longzero_fortytwo_oracle.log" 2>&1
  if [ -s "$RES/score_longzero_fortytwo_use_real.csv" ]; then
    log "anchor oracle OK in $(( ($(date +%s)-t0)/60 )) min"
  else
    log "anchor oracle FAILED — F7 and the decomposition cannot be computed"
  fi
  for d in "$MODELS/longzero_fortytwo"/data/generated/predictions_*; do
    [ -e "$d" ] && rm -rf "$d"
  done
fi

# 1b. three more seeds at L=160. These are arms 1-3 of the parked SS sweep, run through ITS
#     unmodified runner into ITS results directory, so run_sweep.sh will skip them later (SCOPE.md).
for s in 43 44 45; do
  assert_head
  label=$(build 160 "$s") || continue
  if [ -s "$SSRES/score_${label}.csv" ]; then log "SKIP $label — already scored"; continue; fi
  log "seed arm $label (elapsed $(elapsed_h)h)"
  bash "$SSD/tools/run_arm.sh" "$label" >> "$RES/run.log" 2>&1
  [ -s "$SSRES/score_${label}.csv" ] && log "$label OK" || log "$label FAILED (continuing)"
done

STATE=$(verdict_state)
log "stage 1 verdict: $STATE"
if [ "$STATE" = "G1-STOP" ]; then
  log "=== G1 FIRED: seed variance swamps the effect. Stopping — this IS the result. ==="
  touch "$RES/CURVE_COMPLETE"; exit 0
fi

# ================================================================ stages 2 and 3 — the curve
for L in 300 600; do
  assert_head; assert_disk
  label=$(build "$L" 42) || break
  if [ -s "$RES/score_${label}.csv" ] && [ -s "$RES/score_${label}_use_real.csv" ]; then
    log "SKIP $label — control and oracle both scored"
  else
    log "--- L=$L: $label (elapsed $(elapsed_h)h, remaining $(remaining_h)h) ---"
    bash "$T/run_lesson_arm.sh" "$label" --gate >> "$RES/run.log" 2>&1
    [ -s "$RES/score_${label}.csv" ] && log "$label control OK" || log "$label control FAILED"
  fi
  STATE=$(verdict_state); log "after L=$L: $STATE"
done

# =========================================================== stage 4 — the pre-registered branch
REM=$(remaining_h)
if [ "$(echo "$REM < $STAGE4_MIN_HOURS" | bc)" -eq 1 ]; then
  log "stage 4 SKIPPED — ${REM}h remaining, below the ${STAGE4_MIN_HOURS}h a 900-lesson arm needs"
else
  case "$STATE" in
    RISING)
      log "--- stage 4 (RISING branch): L=900, where does it stop? ---"
      assert_head; assert_disk
      label=$(build 900 42) && bash "$T/run_lesson_arm.sh" "$label" --gate >> "$RES/run.log" 2>&1
      ;;
    *)
      log "--- stage 4 ($STATE branch): two more seeds at L=300, harden the null ---"
      for s in 43 44; do
        assert_head; assert_disk
        label=$(build 300 "$s") || continue
        bash "$T/run_lesson_arm.sh" "$label" >> "$RES/run.log" 2>&1
      done
      ;;
  esac
  STATE=$(verdict_state)
fi

log "=== COMPLETE after $(elapsed_h)h · final state: $STATE ==="
log "read results/VERDICT.md — the state is its first line, read it before the numbers"
touch "$RES/CURVE_COMPLETE"
