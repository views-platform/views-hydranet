#!/usr/bin/env bash
# run_wave1.sh — 4 freeze arms x 4 seeds, emit-only, fully unattended.
#
# Shape copied from 2026-08-21_persistence_reference_dossier/tools/run_all_seeds.sh, the proven
# unattended launcher in this repo. Every property below is there because something already went
# wrong without it:
#
#   `set -uo pipefail` and NOT -e ... one arm failing must not abort the other fifteen.
#   ONE run_realism_arms.py per arm ... it raises SystemExit on any arm failure, so a 4-arm
#                                      invocation loses all four (run_realism_arms.py:94..245).
#   idempotent skip ................... a partial night resumes instead of restarting.
#   per-arm timeout ................... a HANG is worse than a crash: it eats the night silently.
#   clean pred dirs before each arm ... a crashed arm leaves predictions_* behind, and the next arm
#                                      then refuses to start ("expected exactly one new prediction
#                                      dir"), cascading one failure into all the rest.
#   NO --keep-cubes ................... C-321: it silently disables the multi-arm contamination
#                                      guard, and arms overwrite each other's cubes.
#   global deadline ................... stop LAUNCHING new arms near the end of the window rather
#                                      than being killed mid-arm.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-03_drivers_and_dynamics_dossier"
RES="$D/results"
MODELS="$HYD/../views-models/models"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
RUNNER="$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py"

ARM_TIMEOUT=${ARM_TIMEOUT:-1800}          # 30 min; observed 430-800 s on a free GPU
DEADLINE_S=${DEADLINE_S:-23400}           # 6.5 h: stop starting new arms after this
START=$(date +%s)

SEEDS=(
  "fullzero_fortytwo|calibration_model_20260818_221401.pt"
  "fullzero_fortythree|calibration_model_20260821_045948.pt"
  "fullzero_fortyfour|calibration_model_20260821_082106.pt"
  "fullzero_fortyfive|calibration_model_20260821_120116.pt"
)
ARMS=(none hidden cell all)

mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] wave1: $*" | tee -a "$RES/wave1.log" >&2; }
log "=== ${#SEEDS[@]} seeds x ${#ARMS[@]} arms, timeout ${ARM_TIMEOUT}s, deadline ${DEADLINE_S}s ==="

FAILED=""; DONE=0; SKIPPED=0
for spec in "${SEEDS[@]}"; do
  M="${spec%%|*}"; ART="${spec##*|}"
  for ARM in "${ARMS[@]}"; do
    if [ "$ARM" = "none" ]; then LABEL="identity"; FREEZE=(); else LABEL="identity_freeze${ARM}"; FREEZE=(--freeze "$ARM"); fi
    SCORE="$RES/score_${M}_${LABEL}.csv"

    if [ -s "$SCORE" ]; then log "SKIP $M/$ARM — already scored"; SKIPPED=$((SKIPPED+1)); continue; fi
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt "$DEADLINE_S" ]; then log "DEADLINE reached at ${ELAPSED}s — not starting $M/$ARM"; continue; fi

    # A crashed predecessor leaves this behind and every later arm then refuses to start.
    for stale in "$MODELS/$M/data/generated"/predictions_*; do
      [ -e "$stale" ] || continue
      log "  cleaning stale $(basename "$stale") in $M"; rm -rf "$stale"
    done

    log "--- $M / $ARM (elapsed ${ELAPSED}s) ---"
    timeout --signal=TERM --kill-after=60 "$ARM_TIMEOUT" \
      "$PY" "$RUNNER" --model "$M" --artifact "$ART" --arms identity \
        "${FREEZE[@]}" --body-mean-dump --tag drivers --out "$RES" \
        >> "$RES/wave1_${M}_${ARM}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s "$SCORE" ]; then
      log "$M/$ARM OK"; DONE=$((DONE+1))
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      log "$M/$ARM TIMED OUT after ${ARM_TIMEOUT}s"; FAILED="$FAILED ${M}/${ARM}:timeout"
    else
      log "$M/$ARM FAILED rc=$rc"; FAILED="$FAILED ${M}/${ARM}:rc$rc"
    fi
  done
done

log "=== done: $DONE ran, $SKIPPED skipped, failed:${FAILED:- none} ==="
{
  echo "# Wave 1 run summary"
  echo
  echo "- finished: $(date '+%F %T')"
  echo "- elapsed: $(( ($(date +%s) - START) / 60 )) min"
  echo "- arms run: $DONE   skipped: $SKIPPED"
  echo "- FAILED: ${FAILED:- none}"
  echo
  echo "## score CSVs present"
  for f in "$RES"/score_*.csv; do [ -e "$f" ] && echo "- $(basename "$f") ($(wc -l < "$f") rows)"; done
  echo
  echo "## body-mean dumps present"
  for d in "$RES"/bodymean_*; do [ -d "$d" ] && echo "- $(basename "$d") ($(ls "$d" | wc -l) origins)"; done
} > "$RES/WAVE1_SUMMARY.md"
touch "$RES/WAVE1_DONE"
log "wrote WAVE1_SUMMARY.md"
