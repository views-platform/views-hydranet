#!/usr/bin/env bash
# run_placement_probe.sh — WHY did scheduled sampling make the rollout worse?
#
# The sweep's own data produced a result nobody predicted: SS largely FIXED the zero collapse
# (act_ratio at h18 rose 0.0093 -> 0.0875, 9.4x; at h36, 28x) and rollout AP fell at every horizon.
# So "the model just needs to answer" (M15) is wrong on its own — answering in the WRONG PLACES is
# worse than staying quiet, which is M12 restated and consistent with I-A putting 90-95% of the gap
# on placement.
#
# This probe is inference-only on two frozen artifacts that differ in exactly one config key:
#   fullzero_fortytwo  L=300 seed 42 eps=0.0   AP h18 0.3298
#   fullhalf_fortytwo  L=300 seed 42 eps=0.5   AP h18 0.3064
#
# Three arms per model, ~10 min each, no training:
#   occurrence_real_magnitude_model  hand it PERFECT occurrence, keep its own magnitudes
#   spatial_scramble                 perfect marginals and magnitudes, LOCATIONS permuted
#   thin:0.75                        keep a quarter of the true events, correctly placed
#
# The control and oracle for both already exist and are not re-run.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-17_ss_retention_dossier"; RES="$D/results"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

MODELS_UNDER_TEST=("fullzero_fortytwo" "fullhalf_fortytwo")
ARMS=(occurrence_real_magnitude_model spatial_scramble thin:0.75)

mkdir -p "$RES"; rm -f "$RES/PROBE_DONE"
exec 6>"$RES/.probe.lock"; flock -n 6 || { echo "another probe holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] probe: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
log "=== placement probe · HEAD ${HEAD_SHA:0:12} · ${#MODELS_UNDER_TEST[@]} models x ${#ARMS[@]} arms ==="

DONE=""; FAILED=""
for M in "${MODELS_UNDER_TEST[@]}"; do
  now=$(cd "$HYD" && git rev-parse HEAD)
  [ "$now" = "$HEAD_SHA" ] || { log "ABORT — HEAD moved"; exit 12; }
  free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "$free" in ''|*[!0-9]*) log "ABORT — cannot read df"; exit 13;; esac
  [ "$free" -ge 25 ] || { log "ABORT — ${free}G free"; exit 13; }

  ART=$(ls -t "$MODELS/$M"/artifacts/*.pt 2>/dev/null | head -1 | xargs -r basename)
  [ -n "$ART" ] || { log "SKIP $M — no artifact"; FAILED="$FAILED $M"; continue; }

  for arm in "${ARMS[@]}"; do
    label="${arm/:/_}"
    if [ -s "$RES/score_${M}_${label}.csv" ]; then log "SKIP $M/$arm — scored"; DONE="$DONE $M/$arm"; continue; fi
    n=$(ls -d "$MODELS/$M"/data/generated/predictions_* 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] || { log "ABORT — $n leftover cube(s) in $M"; exit 3; }

    log "--- $M / $arm (on $ART) ---"
    t0=$(date +%s)
    timeout -k 120 14400 $CENV python "$RT/run_realism_arms.py" --model "$M" --artifact "$ART" \
        --arms "$arm" --tag probe --out "$RES" >> "$RES/${M}_${label}_probe.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s "$RES/score_${M}_${label}.csv" ]; then
      log "$M/$arm OK in $(( ($(date +%s)-t0)/60 )) min"; DONE="$DONE $M/$arm"
    else
      log "$M/$arm FAILED rc=$rc"; FAILED="$FAILED $M/$arm"
    fi
    for d in "$MODELS/$M"/data/generated/predictions_*; do
      [ -e "$d" ] && { log "  cleaning leftover $(basename "$d")"; rm -rf "$d"; }
    done
  done
done

$CENV python "$D/tools/probe_report.py" >> "$RES/run.log" 2>&1
log "=== PROBE COMPLETE · done:${DONE:- none} · failed:${FAILED:- none} — read results/PROBE.md ==="
printf 'done:%s\nfailed:%s\n' "${DONE:- none}" "${FAILED:- none}" > "$RES/PROBE_DONE"
