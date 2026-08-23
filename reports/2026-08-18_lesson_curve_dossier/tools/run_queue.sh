#!/usr/bin/env bash
# run_queue.sh <lessons>:<seed>:<eps>[:resume[@<train-head>]] ...  — arms, one at a time, in order.
#
#   run_queue.sh 40:47:0.0                     a fresh 40-lesson arm, seed 47, no scheduled sampling
#   run_queue.sh 300:42:0.5                    scheduled sampling at eps_max=0.5
#   run_queue.sh 300:42:0.0:resume@eb388f9     emit from the artifact trained at that commit
#   run_queue.sh 160:47:0.0 600:42:0.0         two arms, in that order
#
# eps is REQUIRED, never defaulted. C-259 and C-300 both trace to a scheduled-sampling parameter
# that was implicit: four shipped roster models trained with an unset `ss_feedback` that silently
# became 'mean'. The dose is exactly the value that must be stated out loud every time.
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
#   * No stale-state reads. The verifier's exit code is checked, so a crashed verifier stops the
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
# Which verifier owns these results. RES_DIR and VERIFIER travel together: pointing the queue at
# another experiment's results while silently running THIS experiment's verifier over them is the
# same class as the C-259 implicit default. If you redirect one you must state the other.
VERIFIER="${VERIFIER:-}"
if [ -z "$VERIFIER" ]; then
  if [ -n "${RES_DIR:-}" ]; then
    echo "run_queue.sh: RES_DIR is set but VERIFIER is not — refusing to run the lesson-curve" >&2
    echo "              verifier over another experiment's results. Set VERIFIER explicitly." >&2
    exit 65
  fi
  VERIFIER="$T/verify_curve.py"
fi
[ -f "$VERIFIER" ] || { echo "run_queue.sh: no such verifier: $VERIFIER" >&2; exit 66; }
MIN_FREE_GB="${MIN_FREE_GB:-25}"
DISK_WAIT_MAX_MIN="${DISK_WAIT_MAX_MIN:-720}"

[ $# -gt 0 ] || { echo "usage: run_queue.sh <lessons>:<seed>:<eps>[:resume[@head]] ..." >&2; exit 64; }

mkdir -p "$RES"
rm -f "$RES/QUEUE_DONE"
exec 7>"$RES/.queue.lock"
flock -n 7 || { echo "another queue holds the lock — refusing to run two schedulers"; exit 11; }
# NEVER stdout. `ensure_arm` returns the arm label on stdout and the caller captures it, so a log
# line written to stdout is captured AS THE ARM NAME. That happened on 2026-08-20: this function
# tee'd to stdout, `label=$(ensure_arm ...)` swallowed "queue: building longzero_fortyseven", and the
# 160-seed-47 arm was built and then never run. Introduced by me while fixing DOUBLE logging in
# run_lesson_arm.sh — a cosmetic fix in one file that broke a working thing in another.
log(){
  _m="[$(date '+%F %T')] queue: $*"
  echo "$_m" >> "$RES/run.log"
  [ -t 2 ] && echo "$_m" >&2
  return 0
}

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
# Sets ARM_LABEL rather than echoing it. Command substitution around a function that also logs is
# the trap above; removing the substitution removes the whole class, not just today's instance.
ARM_LABEL=""
# Which builder makes an arm, and where it lives. Defaults reproduce the SS/lesson-curve behaviour
# exactly. The ITF pilot (#287) sets ARM_MODULE=make_itf_arm + ARM_TOOLS=<itf dossier>/tools rather
# than adding a second scheduler — keeping two schedulers is what caused the 2026-08-19 audit.
ARM_MODULE="${ARM_MODULE:-make_ss_arm}"
ARM_TOOLS="${ARM_TOOLS:-$SSD/tools}"

ensure_arm(){
  local L=$1 S=$2 E=$3 label got want
  ARM_LABEL=""
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$ARM_TOOLS')
from $ARM_MODULE import arm_label; print(arm_label(lessons=$L, eps=$E, seed=$S))" 2>/dev/null)
  [ -n "$label" ] || { log "ABORT — no legal arm name for lessons=$L seed=$S"; return 1; }
  if [ -d "$MODELS/$label" ]; then
    # ss_reverse is part of the arm's IDENTITY, not decoration: an SS arm and an ITF arm at the
    # same lessons/seed/eps both read "300:42:0.5", so without it this check would accept a
    # decreasing-TF arm where an increasing-TF one was asked for — the two the pilot compares.
    got=$($CENV python -c "
import ast,sys
from pathlib import Path
t=(Path('$MODELS/$label')/'configs/config_hyperparameters.py').read_text()
ast.parse(t); ns={}; exec(compile(t,'cfg','exec'),ns); hp=ns['get_hp_config']()
print(f\"{hp['total_lessons']}:{hp['torch_seed']}:{hp['ss_epsilon_max']}:{bool(hp.get('ss_reverse', False))}\")" 2>/dev/null)
    want="$L:$S:$E:$([ "$ARM_MODULE" = "make_itf_arm" ] && echo True || echo False)"
    if [ "$got" != "$want" ]; then
      log "ABORT — $label exists but its config reads '$got', wanted '$want'"
      return 1
    fi
  else
    log "building $label (lessons=$L seed=$S eps=$E)"
    $CENV python "$ARM_TOOLS/$ARM_MODULE.py" --lessons "$L" --eps "$E" --seed "$S" \
      >>"$RES/run.log" 2>&1 || { log "ABORT — $ARM_MODULE refused $label"; return 1; }
  fi
  # a label must look like a pipeline-legal model name; anything else means something upstream
  # leaked into it (see the log() note) and must stop the arm, not name it
  case "$label" in
    [a-z]*_[a-z]*) ;;
    *) log "ABORT — computed arm label is not a legal model name: '$label'"; return 1 ;;
  esac
  ARM_LABEL="$label"
}

DONE=""; FAILED=""
for spec in "$@"; do
  IFS=: read -r L S E MODE <<<"$spec"
  case "$L:$S" in *[!0-9]*:*|*:*[!0-9]*) log "SKIP '$spec' — malformed"; FAILED="$FAILED $spec"; continue;; esac
  case "$E" in ''|*[!0-9.]*) log "SKIP '$spec' — eps missing or malformed; it is required"; FAILED="$FAILED $spec"; continue;; esac

  assert_head   || { FAILED="$FAILED $spec"; break; }
  wait_for_disk || { FAILED="$FAILED $spec"; break; }

  ensure_arm "$L" "$S" "$E" || { FAILED="$FAILED $spec"; continue; }
  label="$ARM_LABEL"

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

  log "--- $label (L=$L seed=$S eps=$E ${MODE:-fresh}) ---"
  bash "$T/run_lesson_arm.sh" "$label" "${ARGS[@]}" >>"$RES/run.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then log "$label OK"; DONE="$DONE $label"
  else                   log "$label FAILED rc=$rc"; FAILED="$FAILED $label"; fi

  # verify after every arm so the verdict is never stale; a crashed verifier stops the queue
  $CENV python "$VERIFIER" >>"$RES/run.log" 2>&1
  if [ $? -ne 0 ]; then
    log "ABORT — $(basename "$VERIFIER") failed; not starting another arm on an unknown verdict"
    FAILED="$FAILED verify"; break
  fi
done

log "=== queue end · completed:${DONE:- none} · failed/skipped:${FAILED:- none} ==="
printf 'completed:%s\nfailed:%s\n' "${DONE:- none}" "${FAILED:- none}" > "$RES/QUEUE_DONE"
[ -z "$FAILED" ]
