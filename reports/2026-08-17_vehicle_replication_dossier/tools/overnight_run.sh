#!/usr/bin/env bash
# overnight_run.sh — the five replication arms on violet_visitor, detached and resumable.
#
# Launched via `setsid` so it survives the assistant idling and the terminal going away: plain
# background jobs get reaped, a session leader does not. Never run this in a foreground shell you
# intend to close.
#
# The unit of resumption is the ARM, and the token is results/<model>_<label>_DONE, written by
# run_realism_arms.py only after eval succeeded, scoring succeeded AND the cube was deleted. The
# score CSV is NOT the token: it is written before deletion, so it would mark an arm done that
# left a 2.5 GB contaminant behind.
#
# Liveness grammar, so that in the morning silence is never ambiguous:
#   RUN_COMPLETE = terminal   HEARTBEAT = alive   PHASE = progressing
# Fresh heartbeat + stale phase = hung. Both stale = killed/rebooted. Neither = never launched.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
DOS="$HYD/reports/2026-08-17_vehicle_replication_dossier"
TOOLS="$DOS/tools"; RES="$DOS/results"; OVN="$RES/overnight"
V2T="$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools"
REALISM_TOOLS="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
MODEL=violet_visitor; VV="$MODELS/$MODEL"; GEN="$VV/data/generated"
ART=calibration_model_20260812_191742.pt
ART_SHA=909f44c0096ee6b5c675f65539fcc31977bbe3ea6bb52438f2ed7b165237ddcb
PREDNAME=predictions_calibration_20260812_191742
CENV="conda run --no-capture-output -n views-hydranet-env"

ARMS=(use_real spatial_scramble occurrence_real_magnitude_model occurrence_model_magnitude_real thin:0.75)
[ $# -gt 0 ] && ARMS=("$@")
ARM_TIMEOUT=3600          # ~2.3x the 26-min mean, above the 36-min worst case
MIN_FREE_GB=27
START=$(date +%s)
NEW_ARM_DEADLINE=$((START + 4*3600))   # start no NEW arm after T+4h

mkdir -p "$OVN"
exec 9>"$OVN/.lock"; flock -n 9 || { echo "another overnight run holds the lock — refusing"; exit 11; }
RLOG="$OVN/run.log"
say(){ echo "[$(date '+%F %T')] $*" >> "$RLOG"; }
phase(){ echo "$(date '+%F %T')  $*" > "$OVN/PHASE"; say "PHASE: $*"; }

# eval-only. No -re (OOMs this box), no wandb network.
unset ALLOW_RE_REPORT
export WANDB_MODE=offline WANDB_SILENT=true

( while :; do
    date '+%s %F %T' > "$OVN/HEARTBEAT"
    f=$(df -BG "$VV" | tail -1 | awk '{print $4}' | tr -d 'G')
    [ "$f" -lt 12 ] && { echo "DISK CRITICAL ${f}G" >> "$RLOG"; touch "$OVN/ABORT_DISK"; }
    sleep 30
  done ) & HB=$!

finish(){
  rc=$?
  kill $HB 2>/dev/null
  phase "EXIT rc=$rc"
  ( cd "$VV/configs" && md5sum -c --quiet "$OVN/config_floor.md5" ) 2>/dev/null \
    || { say "!! CONFIG DRIFT — restoring floor"; cp "$OVN/config_floor"/*.py "$VV/configs/" 2>/dev/null; \
         echo "config drifted and was restored" >> "$OVN/ANOMALIES.txt"; }
  $CENV python "$TOOLS/overnight_verify.py" >> "$RLOG" 2>&1
  { echo "rc=$rc"; echo "ended=$(date '+%F %T')"; echo "elapsed_min=$(( ($(date +%s)-START)/60 ))"; } > "$OVN/STATUS.txt"
  echo "run finished rc=$rc at $(date '+%F %T')" > "$OVN/RUN_COMPLETE"
}
trap finish EXIT
trap 'phase "SIGNAL received"; exit 130' INT TERM HUP

# Bounded deletion: ONLY the exact expected dir name. A blanket glob-rm is how a reference cube
# dies. Anything else present means another process writes here — abort rather than guess.
cleanup_cube(){
  local d
  for d in "$GEN"/predictions_*; do
    [ -e "$d" ] || continue
    if [ "$(basename "$d")" = "$PREDNAME" ]; then
      say "cleaning partial cube $d"; rm -rf "$d"
    else
      say "!! REFUSING to delete unexpected dir $d — aborting"
      echo "unexpected prediction dir $d" >> "$OVN/ANOMALIES.txt"; return 2
    fi
  done
  return 0
}

guard(){
  [ -f "$OVN/ABORT_DISK" ] && { say "GUARD FAIL: disk critical"; return 1; }
  [ -d "$VV" ] || { say "GUARD FAIL: model dir missing $VV"; return 1; }   # the guard the driver dropped
  local f; f=$(df -BG "$VV" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$f" -ge "$MIN_FREE_GB" ] || { say "GUARD FAIL: ${f}G free < ${MIN_FREE_GB}G"; return 1; }
  [ -z "$(ls -d "$GEN"/predictions_* 2>/dev/null)" ] || { say "GUARD FAIL: cube exists before arm"; return 1; }
  [ "$(sha256sum "$VV/artifacts/$ART" | awk '{print $1}')" = "$ART_SHA" ] \
    || { say "GUARD FAIL: ARTIFACT CHANGED mid-run"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$OVN/repo.head")" ] \
    || { say "GUARD FAIL: repo HEAD moved mid-run — arms not comparable"; return 1; }
  ( cd "$VV/configs" && md5sum -c --quiet "$OVN/config_floor.md5" ) 2>/dev/null \
    || { say "GUARD FAIL: config drifted"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${u:-9999}" -lt 3000 ] || { say "GUARD FAIL: GPU ${u} MiB used by a foreign process"; return 1; }
  return 0
}

classify(){
  local label="$1"
  local rc="$2"
  local alog="$OVN/arm_${label}.log"
  local cause
  if   grep -qsE 'CUDA out of memory|OutOfMemoryError' "$alog"; then cause="GPU OOM"
  elif [ "$rc" -eq 124 ]; then cause="TIMEOUT (>${ARM_TIMEOUT}s)"
  elif [ "$rc" -eq 137 ]; then cause="SIGKILL / host OOM"
  elif grep -qs 'refusing to start arm' "$alog"; then cause="LEFTOVER CUBE (contamination guard fired)"
  elif grep -qs 'IndexError' "$alog"; then cause="SCORER ARG FORM (--targets)"
  elif grep -qs 'recorded NO fed-field statistics' "$alog"; then cause="TRANSFORM NEVER RAN"
  else cause="OTHER rc=$rc"; fi
  { echo "arm   : $label"; echo "rc    : $rc"; echo "cause : $cause"; echo "when  : $(date '+%F %T')"
    echo "--- last 60 lines ---"; tail -60 "$alog" 2>/dev/null; } > "$OVN/FAIL_${label}.txt"
  say "ARM $label FAILED: $cause (see FAIL_${label}.txt)"
}

run_one(){
  # NB: separate `local` statements. A single `local a=.. b=${a}` leaves the later name unbound
  # under `set -u` — it cost this run its first launch at 02:53.
  local arm="$1"
  local label="${1/:/_}"
  local sentinel="$RES/${MODEL}_${label}_DONE"
  [ -f "$sentinel" ] && { say "SKIP $arm — sentinel exists"; return 0; }
  # A prior crashed attempt can leave a stale CSV that the verifier would read as this attempt's
  # evidence. Remove them before retrying.
  rm -f "$RES/score_${MODEL}_${label}.csv" "$RES/fedfield_${MODEL}_${label}.csv"
  [ $(date +%s) -ge $NEW_ARM_DEADLINE ] && { say "DEADLINE — not starting $arm"; return 20; }
  guard || { cleanup_cube; return 12; }
  phase "ARM $arm"
  local t0=$(date +%s)
  timeout -k 120 "$ARM_TIMEOUT" $CENV python "$REALISM_TOOLS/run_realism_arms.py" \
      --model "$MODEL" --artifact "$ART" --arms "$arm" --tag "$label" --out "$RES" \
      >> "$OVN/arm_${label}.log" 2>&1
  local rc=$?
  pkill -f 'realism_arm_entry.py' 2>/dev/null   # a timeout kills the driver, not its GPU child
  if [ $rc -eq 0 ] && [ -f "$sentinel" ]; then
    say "ARM $arm OK in $(( ($(date +%s)-t0)/60 )) min"
    cleanup_cube
    $CENV python "$TOOLS/overnight_verify.py" >> "$RLOG" 2>&1
    return 0
  fi
  [ $rc -eq 0 ] && say "!! $arm returned 0 but wrote no sentinel — treating as FAILURE"
  classify "$label" "$rc"
  cleanup_cube
  $CENV python "$TOOLS/overnight_verify.py" >> "$RLOG" 2>&1
  return 1
}

phase "START (${#ARMS[@]} arms)"
( cd "$HYD" && git rev-parse HEAD ) > "$OVN/repo.head"
mkdir -p "$OVN/config_floor"; cp "$VV"/configs/*.py "$OVN/config_floor/"
( cd "$VV/configs" && md5sum *.py ) > "$OVN/config_floor.md5"

consec=0
for arm in "${ARMS[@]}"; do
  if run_one "$arm"; then consec=0; else
    rc=$?
    [ $rc -eq 20 ] && { phase "DEADLINE — stopping with arms remaining"; break; }
    consec=$((consec+1))
    [ $consec -ge 2 ] && { phase "ABORT: 2 consecutive failures — systemic"; break; }
  fi
done
phase "VERIFY"
exit 0
