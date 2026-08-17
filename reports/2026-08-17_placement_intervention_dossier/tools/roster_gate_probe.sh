#!/usr/bin/env bash
# roster_gate_probe.sh — the gate-structure probe across the rest of the roster.
#
# Question: is the quiet-gate diagnosis (EXP-02 — the gate keeps its shape and loses its nerve,
# committing to 2 cells where reality has 116) a property of violet_visitor, or of the model family?
# Everything expensive we might build next rests on the answer, and today has twice shown that a
# result from one vehicle can be the vehicle's property rather than the model's.
#
# Artifacts are pinned from reports/2026-08-15_rollout_ruler_trust_dossier/results/partition_audit.json
# and their sha256 verified on disk before each run. bright_starship carries TWO calibration artifacts
# and the audit says the run used the NEWER one — defaulting to "the only one" or "the first one" would
# have silently probed a different model.
#
# Resumable: an arm whose gate CSV already exists is skipped. Sequential — one heavy job at a time.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
DOS="$HYD/reports/2026-08-17_placement_intervention_dossier"
RES="$DOS/results"; OVN="$RES/roster"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

# model:artifact_timestamp:expected_sha_prefix   (from partition_audit.json)
ARMS=(
  "purple_alien:20260813_062540:89135913e0ad"
  "blazing_meteor:20260812_232850:a9c7136234c2"
  "bright_starship:20260813_174320:61bc5973f885"
  "pink_pirate:20260813_025117:300ca131854c"
  "blue_stranger:20260813_042946:2b51feda86ab"
)

mkdir -p "$OVN"
# Clear the terminal sentinel at START. Leaving a previous run's ROSTER_DONE in place makes any
# watcher report "complete" the instant a rerun begins — which it did at 11:10 on 2026-08-17, and
# the rerun was still on its first model. A stale sentinel is worse than no sentinel: it turns
# "not finished" into "finished", which is the one direction that must never happen silently.
rm -f "$OVN/ROSTER_DONE"
LOG="$OVN/roster.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== roster gate probe: ${#ARMS[@]} models ==="
for spec in "${ARMS[@]}"; do
  IFS=: read -r m ts sha <<<"$spec"
  VV="$MODELS/$m"; GEN="$VV/data/generated"; ART="calibration_model_${ts}.pt"

  [ -s "$RES/gate_${m}_identity.csv" ] && { say "SKIP $m — gate CSV exists"; continue; }

  # --- guards, each one earned by a scar ---
  [ -d "$VV" ] || { say "SKIP $m — model dir missing"; continue; }
  got=$(sha256sum "$VV/artifacts/$ART" 2>/dev/null | cut -c1-12)
  [ "$got" = "$sha" ] || { say "SKIP $m — artifact sha $got != audit's $sha"; continue; }
  n=$(ls -d "$GEN"/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { say "SKIP $m — $n leftover prediction dir(s); would mix two models' cubes"; continue; }
  free=$(df -BG "$VV" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$free" -ge 20 ] || { say "ABORT — ${free}G free"; break; }

  say "--- $m ($ART) ---"
  t0=$(date +%s)
  $CENV python "$RT/run_realism_arms.py" --model "$m" --artifact "$ART" \
      --arms identity --gate-probe --tag gateprobe --out "$RES" \
      >> "$OVN/${m}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$RES/gate_${m}_identity.csv" ]; then
    say "$m OK in $(( ($(date +%s)-t0)/60 )) min"
  else
    say "$m FAILED rc=$rc (see $OVN/${m}.log)"
    tail -3 "$OVN/${m}.log" >> "$LOG" 2>/dev/null
  fi
  # bounded cleanup: only the dir this arm would have written
  for d in "$GEN"/predictions_*; do
    [ -e "$d" ] && { say "  cleaning $(basename "$d")"; rm -rf "$d"; }
  done
done
say "=== roster gate probe COMPLETE ==="
touch "$OVN/ROSTER_DONE"
