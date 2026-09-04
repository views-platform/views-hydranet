#!/usr/bin/env bash
# run_gradtraj.sh — the GRAD-TRAJ probe: does the gradient CREEP or does it JUMP?
#
# SCREEN-2 established that connecting the BPTT-SA gradient wire kills training (NaN gradients in
# enc_conv0.weight at lesson 48) while cutting it trains 300 clean lessons. It did NOT establish
# why, and three static tests failed to reproduce it (no growth with sequence length on a toy head
# or on the real architecture at 31 steps; no surrogate overflow). So the instability is emergent
# — it develops over lessons — and the cheapest thing that separates the two worlds it could be is
# the gradient-norm TRAJECTORY over the ~48 lessons before it dies:
#
#   CREEP  — the norm climbs lesson over lesson, well above the control's. A real explosion.
#            Fixable in principle (clip the feedback path, GTF alpha) and worth more GPU.
#   JUMP   — the norm sits on the control's curve and then goes non-finite in one lesson. That is
#            a numerical BUG in the straight-through path, and no stabiliser should be bought
#            until it is found.
#
# The engine already has the instrument: config `trajectory_log_path` writes per-lesson
# (lesson, raw_grad_norm PRE-clip, loss_reg, loss_cls, gate_logit_mean). It is opt-in and
# observational — read-only forward hooks, no change to the math, the RNG or the optimiser.
#
# Both arms run as THROWAWAY CLONES (traj{attached,detached}_fortytwo). Re-running in the original
# directories would either destroy ssdetached_fortytwo's good 300-lesson artifact or leave its
# config no longer describing the run that produced it.
#
# THE CONTROL IS NOT OPTIONAL. A rising gradient norm early in training is ordinary; without
# ssdetached's curve over the same lessons there is no way to call a rise abnormal. This is the
# same reason F3 fired on a known-good control in the silence-vs-fade dossier (C-320).
#
# Hardening copied from run_screen.sh, for the reasons recorded there: no -e, CUDA and foreign-GPU
# gates (C-163), disk gate, HEAD guard, per-arm timeout, stall watchdog, heartbeat/PHASE.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results/gradtraj"
CENV="conda run --no-capture-output -n views-hydranet-env"
export WANDB_MODE=offline WANDB_SILENT=true          # C-163: wandb.finish() has DNS-hung 38 min here

# Measured, not estimated: arm B reached its lesson-48 crash in 15.6 min (run.log, 22:44 -> 22:59).
# The caps are generous multiples of that so a working arm is never killed by an estimate — the
# Wave-1 lesson. The stall watchdog catches a real hang instead.
ARM_TIMEOUT=${ARM_TIMEOUT:-5400}                     # 90 min per arm vs ~16-25 min expected
STALL_MAX=${STALL_MAX:-900}                          # 15 min of no log growth = stuck
ATT_MAX_LESSONS=${ATT_MAX_LESSONS:-80}               # attached should die ~48; 80 = it did not
DET_MAX_LESSONS=${DET_MAX_LESSONS:-60}               # control only needs to span the crash lesson

# attached FIRST: it carries the result, and if it fails to reproduce the crash that is the
# headline and the control can be skipped.
ARMS=(trajattached_fortytwo trajdetached_fortytwo)
declare -A CAP=( [trajattached_fortytwo]=$ATT_MAX_LESSONS [trajdetached_fortytwo]=$DET_MAX_LESSONS )
declare -A CSV=( [trajattached_fortytwo]="$RES/traj_attached.csv" \
                 [trajdetached_fortytwo]="$RES/traj_detached.csv" )

mkdir -p "$RES"; START=$(date +%s)
log(){ echo "[$(date '+%F %T')] $*" >> "$RES/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$RES/PHASE"; log "PHASE: $*"; }
exec 9>"$RES/.lock"; flock -n 9 || { log "locked — exiting"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$RES/repo.head"
log "=== GRAD-TRAJ probe: ${#ARMS[@]} arms, timeout ${ARM_TIMEOUT}s ==="
( while :; do date '+%s %F %T' > "$RES/HEARTBEAT"; sleep 30; done ) & HB=$!
trap 'kill $HB 2>/dev/null; echo "ended=$(date "+%F %T") elapsed_min=$(( ($(date +%s)-START)/60 ))" > "$RES/RUN_COMPLETE"; log "=== RUN_COMPLETE ==="' EXIT

guard(){
  $CENV python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    || { log "GUARD FAIL: no CUDA — refusing a silent CPU grind (C-163)"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) log "GUARD FAIL: nvidia-smi unreadable"; return 1;; esac
  [ "$u" -lt 3000 ] || { log "GUARD FAIL: ${u} MiB already on the GPU"; return 1; }
  local f; f=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "${f:-}" in ''|*[!0-9]*) return 1;; esac
  [ "$f" -ge 25 ] || { log "GUARD FAIL: ${f}G free"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$RES/repo.head")" ] \
    || { log "GUARD FAIL: repo HEAD moved mid-run — the arms would not be comparable"; return 1; }
  return 0; }

# The knob must be live on THIS clone, not on the arm it was cloned from (C-324). Two seconds.
phase "POTENCY PRE-FLIGHT"
if ! $CENV python "$D/tools/preflight_potency.py" "${ARMS[@]}" >> "$RES/run.log" 2>&1; then
  log "ABORT: potency pre-flight FAILED — the knob cannot act on these clones (C-324)."
  exit 12
fi
log "potency pre-flight PASSED"

OK=0; FAILED=""
for ARM in "${ARMS[@]}"; do
  A="$MODELS/$ARM"; C="${CSV[$ARM]}"; LIM="${CAP[$ARM]}"
  guard || { FAILED="$FAILED ${ARM}:guard"; continue; }
  rm -f "$C"
  phase "TRAIN $ARM (cap ${LIM} lessons, elapsed $(( ($(date +%s)-START)/60 )) min)"
  LOG="$RES/arm_${ARM}.log"; : > "$LOG"
  # -t only. No -e/-sa: this probe reads the trajectory CSV, never an artifact, and the eval +
  # BN-recal tail is most of arm A's 184 min.
  ( cd "$A" && timeout -k 120 "$ARM_TIMEOUT" $CENV python main.py -r calibration -t ) \
      >> "$LOG" 2>&1 &
  APID=$!

  # LESSON CAP. The control must not train 300 lessons to give 60 rows, and the attached arm must
  # not run for 90 min if it fails to reproduce the crash. Kill on row count, not on wall clock.
  ( while kill -0 $APID 2>/dev/null; do
      sleep 20
      n=$(wc -l < "$C" 2>/dev/null || echo 0)          # includes the header row
      if [ "${n:-0}" -gt "$LIM" ]; then
        echo "$(date '+%F %T') CAP $ARM at $((n-1)) lessons — killing, this is intended" >> "$RES/run.log"
        touch "$RES/${ARM}.capped"
        kill -TERM $APID 2>/dev/null; break
      fi
    done ) & CAPW=$!
  # Stall watchdog, independent of the cap: a hang produces neither log growth nor CSV rows.
  ( while kill -0 $APID 2>/dev/null; do
      s1=$(stat -c %s "$LOG" 2>/dev/null || echo 0); sleep "$STALL_MAX"
      kill -0 $APID 2>/dev/null || break
      s2=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
      [ "$s1" = "$s2" ] && { echo "$(date '+%F %T') STALLED $ARM at ${s2}B" >> "$RES/ANOMALIES.txt"
                             kill -TERM $APID 2>/dev/null; break; }
    done ) & WD=$!
  wait $APID; rc=$?
  kill $CAPW $WD 2>/dev/null; wait $CAPW $WD 2>/dev/null

  rows=$(( $(wc -l < "$C" 2>/dev/null || echo 1) - 1 ))
  if [ -f "$RES/${ARM}.capped" ]; then
    log "$ARM CAPPED at ${rows} lessons (intended)"; OK=$((OK+1))
  elif [ $rc -ne 0 ] && grep -q "FATAL GRADIENT EXPLOSION" "$LOG"; then
    log "$ARM CRASHED as expected at lesson ${rows} — trajectory captured"; OK=$((OK+1))
  elif [ $rc -eq 0 ]; then
    log "$ARM finished rc0 with ${rows} lessons"; OK=$((OK+1))
  else
    case $rc in 124|137) R=timeout;; *) R="rc$rc";; esac
    log "$ARM FAILED ($R) at ${rows} lessons"; FAILED="$FAILED ${ARM}:${R}"
  fi
  # Whatever happened, the CSV is the deliverable. Say plainly if it is empty.
  [ "${rows:-0}" -ge 2 ] || { log "!! $ARM produced ${rows} trajectory rows — nothing to read"
                              echo "$(date '+%F %T') EMPTY TRAJECTORY $ARM" >> "$RES/ANOMALIES.txt"; }
done

phase "READOUT"
$CENV python "$D/tools/read_gradtraj.py" >> "$RES/run.log" 2>&1 \
  || log "!! readout failed — the CSVs are still on disk and can be read by hand"
log "=== $OK ok | failed:${FAILED:- none} ==="
