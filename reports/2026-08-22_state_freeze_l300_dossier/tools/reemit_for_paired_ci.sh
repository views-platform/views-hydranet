#!/usr/bin/env bash
# reemit_for_paired_ci.sh — re-emit `none` and `cell` and KEEP both cubes.
#
# A paired origin-block CI on (cell - none) needs BOTH arms' cubes at once: one origin draw per
# replicate must score both arms on the same resampled cell set, which is the whole point of
# pairing. run_freeze_arms.py is score-then-delete, so the overnight run left nothing to bootstrap.
#
# The pipeline names the prediction dir after the ARTIFACT, so both arms write the SAME path. This
# runs one arm, MOVES its cube aside under an arm-specific name, then runs the next — which is why
# --keep-cubes alone is not enough and the driver's own refusal-on-leftover guard stays satisfied.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-22_state_freeze_l300_dossier"; RES="$D/results"
FT="$HYD/reports/2026-08-15_state_freeze_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
# EXP-04 (2026-08-31): parameterised so the same script serves all four seeds. Defaults are the
# original seed-43 pair, so any earlier invocation is unchanged.
M="${MODEL:-fullzero_fortythree}"
ART="${ARTIFACT:-calibration_model_20260821_045948.pt}"
# Which two arms to pair. Defaults reproduce the cell-vs-none interval (EXP-02); override to
# compare two INTERIOR points, where the arms are far more correlated and the cell-vs-none MDE
# is the wrong yardstick entirely.
PAIR_ARMS="${PAIR_ARMS:-none cell}"
GEN="$MODELS/$M/data/generated"

mkdir -p "$RES"; rm -f "$RES/PAIRED_CUBES_READY"
exec 7>"$RES/.reemit.lock"; flock -n 7 || { echo "another re-emit holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] reemit: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

# PAIR_ARMS is arbitrary-length and every cube is KEPT, so one up-front check is not enough: a
# four-arm pairing would fill the disk mid-run, which is the C-154 scar this preflight exists for.
# ~2.5 GB per cube, checked before EVERY arm.
n_arms=$(set -- $PAIR_ARMS; echo $#)
need=$(( n_arms * 3 + 10 ))
free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
case "$free" in ''|*[!0-9]*) log "ABORT — cannot read df"; exit 13;; esac
[ "$free" -ge "$need" ] || { log "ABORT — ${free}G free, need ${need}G for $n_arms kept cube(s)"; exit 13; }

for ARM in $PAIR_ARMS; do
  DEST="$GEN/paired_${ARM//@/_}"
  if [ -d "$DEST" ]; then log "SKIP $ARM — cube already at $(basename "$DEST")"; continue; fi
  n=$(ls -d "$GEN"/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { log "ABORT — $n leftover predictions_* dir(s); would be mixed in"; exit 3; }
  free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$free" -ge 8 ] || { log "ABORT — ${free}G free before $ARM; a kept cube needs ~2.5G"; exit 13; }
  log "--- emitting $ARM (keeping cubes) ---"
  t0=$(date +%s)
  timeout -k 120 7200 $CENV python "$FT/run_freeze_arms.py" --model "$M" --artifact "$ART" \
      --arms "$ARM" --keep-cubes --out "$RES" >> "$RES/reemit.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "ABORT — $ARM failed rc=$rc"; exit 4; }
  SRC=$(ls -d "$GEN"/predictions_* 2>/dev/null | head -1)
  [ -n "$SRC" ] || { log "ABORT — no cube after $ARM"; exit 5; }
  mv "$SRC" "$DEST" || { log "ABORT — could not move $SRC"; exit 6; }
  log "$ARM OK in $(( ($(date +%s)-t0)/60 )) min → $(basename "$DEST")"
done
touch "$RES/PAIRED_CUBES_READY"
log "=== both cubes ready for the paired bootstrap ==="
