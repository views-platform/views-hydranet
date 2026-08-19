#!/usr/bin/env bash
# run_queue.sh <lessons>:<seed>[:resume[@<train-head>]] ...  — run arms, one at a time, in order.
#
#   run_queue.sh 40:47                      a fresh 40-lesson arm at seed 47
#   run_queue.sh 300:42:resume@eb388f9      emit from the saved artifact trained at that commit
#   run_queue.sh 160:47 600:42              two arms, in that order
#
# This REPLACES run_curve.sh + run_followup{,2,3,4}.sh. Those grew by accretion — each written while
# an arm was running, each patching the previous one's blind spot — and the accretion was itself the
# root cause of a day of failures (06_harness_audit.md). Three defects disappear by construction
# rather than by fix:
#
#   * No `pgrep` anywhere. The old drivers detected "is an arm running?" by matching command-line
#     strings, which matched unrelated shells (a status command containing "main.py -r calibration"
#     read as a live arm) and could not see a SIBLING driver between arms, so two schedulers could
#     run concurrently. One sequential queue holding one lock cannot race with itself.
#   * No stale-state reads. verify_curve's exit code is checked, so a crashed verifier stops the
#     queue instead of leaving the previous stage's verdict to be acted on.
#   * No lying sentinel. QUEUE_DONE records what actually completed; a skipped or failed arm is
#     named, not silently absorbed.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-18_lesson_curve_dossier"
RES="${RES_DIR:-$D/results}"; T="$D/tools"
SSD="$HYD/reports/2026-08-17_ss_retention_dossier"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
DISK_WAIT_MAX_MIN="${DISK_WAIT_MAX_MIN:-720}"

[ $# -gt 0 ] || { echo "usage: run_queue.sh <lessons>:<seed>[:resume[@head]] ..." >&2; exit 64; }

mkdir -p "$RES"
rm -f "$RES/QUEUE_DONE"
exec 7>"$RES/.queue.lock"
flock -n 7 || { echo "another queue holds the lock — refusing to run two schedulers"; exit 11; }
# the queue owns run.log, so it tees; it does NOT also write to stderr (see run_lesson_arm.sh)
log(){ echo "[$(date '+%F %T')] queue: $*" | tee -a "$RES/run.log"; }

HEAD_SHA=$(cd "$HYD" && git rev-parse HEAD)
log "=== queue start · HEAD ${HEAD_SHA:0:12} · $# arm(s): $* ==="

wait_for_disk(){
  local waited=0 free
  while :; do
    free=$(df -BG "$MODELS" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    # an unparseable df must not silently skip the guard (it used to: [ "" -lt 25 ] errors non-zero)
    case "$free" in ''|*[!0-9]*) log "ABORT — cannot read free space from df"; return 1;; esac
    [ "$free" -ge "$MIN_FREE_GB" ] && break
    [ "$waited" -eq 0 ] && log "PAUSED — ${free}G free, need ${MIN_FREE_GB}G; re-checking every 10 min"
    sleep 600; waited=$((waited + 10))
    if [ "$waited" -ge "$DISK_WAIT_MAX_MIN" ]; then
      log "ABORT — ${DISK_WAIT_MAX_MIN} min under ${MIN_FREE_GB}G free"; return 1
    fi
  done
  [ "$waited" -gt 0 ] && log "resumed after ${waited} min waiting for disk"
  return 0
}

assert_head(){
  local now; now=$(cd "$HYD" && git rev-parse HEAD)
  [ "$now" = "$HEAD_SHA" ] && return 0
  log "ABORT — HEAD moved ${HEAD_SHA:0:12} -> ${now:0:12}; arms would not be comparable (F6)"
  return 1
}

# Build the arm if absent; if present, PROVE it matches before reusing it. Reusing a directory by
# name alone is how an old configuration gets silently relabelled.
ensure_arm(){
  local L=$1 S=$2 label got
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$SSD/tools')
from make_ss_arm import arm_label; print(arm_label(lessons=$L, eps=0.0, seed=$S))" 2>/dev/null)
  [ -n "$label" ] || { log "ABORT — no legal arm name for lessons=$L seed=$S"; return 1; }
  if [ -d "$MODELS/$label" ]; then
    got=$($CENV python -c "
import ast,sys
from pathlib import Path
t=(Path('$MODELS/$label')/'configs/config_hyperparameters.py').read_text()
ast.parse(t); ns={}; exec(compile(t,'cfg','exec'),ns); hp=ns['get_hp_config']()
print(f\"{hp['total_lessons']}:{hp['torch_seed']}:{hp['ss_epsilon_max']}\")" 2>/dev/null)
    if [ "$got" != "$L:$S:0.0" ]; then
      log "ABORT — $label exists but its config reads '$got', wanted '$L:$S:0.0'"
      return 1
    fi
  else
    log "building $label (lessons=$L seed=$S)"
    $CENV python "$SSD/tools/make_ss_arm.py" --lessons "$L" --eps 0.0 --seed "$S" \
      >>"$RES/run.log" 2>&1 || { log "ABORT — make_ss_arm refused $label"; return 1; }
  fi
  echo "$label"
}

DONE=""; FAILED=""
for spec in "$@"; do
  IFS=: read -r L S MODE <<<"$spec"
  case "$L:$S" in *[!0-9]*:*|*:*[!0-9]*) log "SKIP '$spec' — malformed"; FAILED="$FAILED $spec"; continue;; esac

  assert_head   || { FAILED="$FAILED $spec"; break; }
  wait_for_disk || { FAILED="$FAILED $spec"; break; }

  label=$(ensure_arm "$L" "$S") || { FAILED="$FAILED $spec"; continue; }

  if [ -s "$RES/score_${label}.csv" ] && [ -s "$RES/score_${label}_use_real.csv" ]; then
    log "SKIP $label — control and oracle both already scored"
    DONE="$DONE $label"
    continue
  fi

  ARGS=(--gate)
  if [ "${MODE%%@*}" = "resume" ]; then
    ARGS+=(--resume)
    [ "${MODE#*@}" != "$MODE" ] && ARGS+=(--train-head "${MODE#*@}")
    # a killed emit leaves a partial cube and a staging dir; run_lesson_arm refuses to start on them
    rm -rf "$MODELS/$label"/data/generated/predictions_* \
           "$MODELS/$label"/data/generated/_pf_staging 2>/dev/null
  fi

  log "--- $label (L=$L seed=$S ${MODE:-fresh}) ---"
  bash "$T/run_lesson_arm.sh" "$label" "${ARGS[@]}" >>"$RES/run.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then log "$label OK"; DONE="$DONE $label"
  else                   log "$label FAILED rc=$rc"; FAILED="$FAILED $label"; fi

  # verify after every arm so the verdict is never stale; a crashed verifier stops the queue
  $CENV python "$T/verify_curve.py" >>"$RES/run.log" 2>&1
  if [ $? -ne 0 ]; then
    log "ABORT — verify_curve failed; not starting another arm on an unknown verdict"
    FAILED="$FAILED verify"; break
  fi
done

log "=== queue end · completed:${DONE:- none} · failed/skipped:${FAILED:- none} ==="
printf 'completed:%s\nfailed:%s\n' "${DONE:- none}" "${FAILED:- none}" > "$RES/QUEUE_DONE"
[ -z "$FAILED" ]
