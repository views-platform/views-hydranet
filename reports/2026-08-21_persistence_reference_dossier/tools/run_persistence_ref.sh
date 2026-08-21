#!/usr/bin/env bash
# run_persistence_ref.sh — does a 300-lesson model beat persistence on AP?
#
# Step 1 of the decision table in 00_README.md. Re-emits ONE eps=0 L=300 control to recover the
# support set (score-then-delete removed the original cubes; _support_keys reads identifiers.npz
# from the prediction dirs, so persistence cannot be built from truth alone), then scores the arm
# and persistence together on that identical support.
#
# --keep-cubes is deliberate and load-bearing: the cube is the support, and it is also the raw
# material for any matched-S work the decision table selects next. It is deleted by this script
# only after BOTH scores are on disk.
#
# The re-emit is a falsifier, not just a means: eps=0 arms are bit-reproducible (M22), so
# `identity` must reproduce the archived control exactly. If it does not, the vehicle is not what
# we think it is and no persistence number from it is worth reading.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-21_persistence_reference_dossier"; RES="$D/results"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
V2T="$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"

# One seed per invocation. Defaults reproduce EXP-01 so the original command still works.
M="${1:-fullzero_fortythree}"
ART="${2:-calibration_model_20260821_045948.pt}"
HORIZONS=1,6,12,18,24,30,36
[ -n "$M" ] && [ -n "$ART" ] || { echo "usage: $0 [<model> <artifact.pt>]"; exit 2; }

mkdir -p "$RES/identifiers"; rm -f "$RES/PERSIST_DONE_$M"
exec 6>"$RES/.persist.lock"; flock -n 6 || { echo "another run holds the lock"; exit 11; }
log(){ _m="[$(date '+%F %T')] persist: $*"; echo "$_m" >> "$RES/run.log"; [ -t 2 ] && echo "$_m" >&2; return 0; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
log "=== persistence re-reference · HEAD ${HEAD_SHA:0:12} · $M / $ART ==="

free=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
case "$free" in ''|*[!0-9]*) log "ABORT — cannot read df"; exit 13;; esac
[ "$free" -ge 25 ] || { log "ABORT — ${free}G free"; exit 13; }
[ -f "$MODELS/$M/artifacts/$ART" ] || { log "ABORT — artifact missing"; exit 14; }
# The pipeline names the cube dir after the ARTIFACT, which is what makes both branches below
# safe: a dir matching this artifact's stem can only be the cube we emitted for it, while any
# other name is a stray from a different artifact and must still be refused. Without the stem
# check, "resume" would be a bypass of the guard rather than a use of it.
EXPECT="$MODELS/$M/data/generated/predictions_$(echo "$ART" | sed 's/^calibration_model_/calibration_/; s/\.pt$//')"
STRAY=0
for d in "$MODELS/$M"/data/generated/predictions_*; do
  [ -e "$d" ] || continue
  [ "$d" = "$EXPECT" ] || { log "ABORT — stray cube $(basename "$d") would be mixed in"; STRAY=1; }
done
[ "$STRAY" -eq 0 ] || exit 3

if [ -d "$EXPECT" ]; then
  log "RESUME — cube for $ART already on disk, skipping the emit"
else
  log "--- emitting identity (free-running control) on $ART ---"
  t0=$(date +%s)
  timeout -k 120 7200 $CENV python "$RT/run_realism_arms.py" --model "$M" --artifact "$ART" \
      --arms identity --keep-cubes --tag persistref --out "$RES" >> "$RES/emit.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "ABORT — emit failed rc=$rc (cube kept for re-score)"; exit 4; }
  log "emit OK in $(( ($(date +%s)-t0)/60 )) min"
fi

PRED=$(ls -d "$MODELS/$M"/data/generated/predictions_* 2>/dev/null | head -1)
[ -n "$PRED" ] || { log "ABORT — no cube after emit"; exit 5; }
log "cube: $(basename "$PRED")"

log "--- scoring arm + persistence on ONE support ---"
$CENV python "$V2T/score_v2_horizons.py" \
    "$M|$PRED|lr_{t}_best|by_{t}_best" \
    --targets=sb --horizons="$HORIZONS" --persistence \
    --out="$RES/score_persistence_ref_$M.csv" >> "$RES/score.log" 2>&1
rc=$?
[ $rc -eq 0 ] && [ -s "$RES/score_persistence_ref_$M.csv" ] || {
    log "ABORT — scoring failed rc=$rc (cube KEPT at $PRED for re-score)"; exit 6; }
log "scored OK"

# Preserve the SUPPORT, not one arbitrary origin's copy of it. The first version of this line
# was `cp "$PRED"/origin_*/lr_*/identifiers.npz "$RES/"`, which collapsed 13 origins onto one
# filename and destroyed exactly the thing it was trying to keep — support is the set of
# (origin, unit) pairs present at EVERY horizon, so it cannot be rebuilt from one origin.
mkdir -p "$RES/identifiers"
for od in "$PRED"/origin_*; do
  [ -d "$od" ] || continue
  src=$(ls "$od"/lr_*/identifiers.npz 2>/dev/null | head -1)
  [ -n "$src" ] && cp "$src" "$RES/identifiers/$(basename "$od").npz"
done
nid=$(ls "$RES/identifiers"/*.npz 2>/dev/null | wc -l)
[ "$nid" -gt 0 ] || { log "ABORT — no identifiers preserved; keeping cube"; exit 7; }
log "preserved $nid origin identifier file(s) — support is reconstructible without a re-emit"
rm -rf "$PRED"; log "cube deleted after both scores landed"
echo "$HEAD_SHA" > "$RES/repo.head"
touch "$RES/PERSIST_DONE_$M"
log "=== DONE ==="
