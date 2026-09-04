#!/usr/bin/env bash
# run_screen.sh — the two BPTT-SA screen arms, trained back to back, unattended.
#
# Same hardening as the drivers dossier's wave launchers (03_harness §B there): no -e, one
# invocation per arm, per-arm timeout because a hang eats the window silently, CUDA and
# foreign-GPU gates because a wedged context becomes a CPU grind that looks like slow progress
# (C-163), stale-artifact cleanup so one crash does not cascade, HEAD guard, heartbeat/phase,
# and a summary that names failures so a partial run is legible.
#
# Training, not emit: the timeout is 5 h per arm against an expected 2-3 h.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
D="$HYD/reports/2026-09-03_bptt_sa_dossier"; RES="$D/results"; OVN="$RES/overnight"
CENV="conda run --no-capture-output -n views-hydranet-env"
export WANDB_MODE=offline WANDB_SILENT=true          # C-163: wandb.finish() has DNS-hung 38 min here
# 12.5 h, matching run_lesson_arm.sh's max(150*L, 21600) for L=300. Deliberately generous: Wave 1
# taught that a timeout set from an ESTIMATE kills working arms (a 30-min ceiling against a
# measured 35-min arm), and a killed arm leaves artefacts that cascade. A real hang is caught fast
# by the stall watchdog instead, which is the right division of labour.
ARM_TIMEOUT=${ARM_TIMEOUT:-45000}
STALL_MAX=${STALL_MAX:-1800}                         # 30 min of no log growth = stuck
ARMS=(ssdetached_fortytwo ssattached_fortytwo)       # detached FIRST: it reproduces the known result
mkdir -p "$OVN"; START=$(date +%s)
log(){ echo "[$(date '+%F %T')] $*" >> "$OVN/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$OVN/PHASE"; log "PHASE: $*"; }
exec 9>"$OVN/.lock"; flock -n 9 || { log "locked — exiting"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$OVN/repo.head"
log "=== BPTT-SA screen: ${#ARMS[@]} training arms, timeout ${ARM_TIMEOUT}s ==="
( while :; do date '+%s %F %T' > "$OVN/HEARTBEAT"; sleep 30; done ) & HB=$!
trap 'kill $HB 2>/dev/null; echo "ended=$(date "+%F %T") elapsed_min=$(( ($(date +%s)-START)/60 ))" > "$OVN/RUN_COMPLETE"; log "=== RUN_COMPLETE ==="' EXIT

# `$APID` is the SUBSHELL; killing it leaves `timeout -> conda run -> python` alive beneath it.
# Measured in run_gradtraj.sh 2026-09-04: an arm reported killed went on training for 7 more
# lessons on the GPU. This watchdog has never fired (no STALLED line in any ANOMALIES.txt), so no
# past result is affected -- but if it ever did, it would log a kill and leave the process running,
# and the next arm's <3000 MiB GPU guard would fail loud for a reason nobody would connect to this.
kill_tree(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
             kill -TERM "$p" 2>/dev/null; }

guard(){ local m="$1"
  $CENV python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    || { log "GUARD FAIL: no CUDA — refusing a silent CPU grind (C-163)"; return 1; }
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) log "GUARD FAIL: nvidia-smi unreadable"; return 1;; esac
  [ "$u" -lt 3000 ] || { log "GUARD FAIL: ${u} MiB already on the GPU"; return 1; }
  local f; f=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
  case "${f:-}" in ''|*[!0-9]*) return 1;; esac
  [ "$f" -ge 25 ] || { log "GUARD FAIL: ${f}G free"; return 1; }
  [ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$OVN/repo.head")" ] \
    || { log "GUARD FAIL: repo HEAD moved mid-run — the arms would not be comparable"; return 1; }
  return 0; }

# C-324 LAUNCH PRECONDITION. #308's first run burned 276 min training two arms to byte-identical
# weights because the treatment was INERT on the family + sampled-feedback path C-259 forces
# production to use. Two seconds here would have caught it. A gate with a demonstrated ability to
# fire, not decoration: it reproduces that failure exactly as off=0.0, on=0.0 -> INERT.
phase "POTENCY PRE-FLIGHT"
if ! $CENV python "$D/tools/preflight_potency.py" "${ARMS[@]}" >> "$OVN/run.log" 2>&1; then
  log "ABORT: potency pre-flight FAILED — the knob cannot act on this configuration."
  log "       Any result would be a fact about the harness, not the hypothesis (C-324)."
  exit 12
fi
log "potency pre-flight PASSED"

OK=0; FAILED=""
for ARM in "${ARMS[@]}"; do
  A="$MODELS/$ARM"
  if ls "$A"/artifacts/*.pt >/dev/null 2>&1; then log "SKIP $ARM — artifact already present"; continue; fi
  guard "$ARM" || { FAILED="$FAILED ${ARM}:guard"; continue; }
  rm -rf "$A"/data/generated/predictions_* "$A"/data/generated/_pf_staging 2>/dev/null
  phase "TRAIN $ARM (elapsed $(( ($(date +%s)-START)/60 )) min)"
  LOG="$OVN/arm_${ARM}.log"; : > "$LOG"
  ( cd "$A" && timeout -k 120 "$ARM_TIMEOUT" $CENV python main.py -r calibration -t -e -sa ) \
      >> "$LOG" 2>&1 &
  APID=$!
  # Training writes a tqdm line per lesson, so the log grows continuously. 30 min of silence at
  # L=300 (~30 s/lesson) means stuck, not slow.
  ( while kill -0 $APID 2>/dev/null; do
      s1=$(stat -c %s "$LOG" 2>/dev/null || echo 0); sleep "$STALL_MAX"
      kill -0 $APID 2>/dev/null || break
      s2=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
      [ "$s1" = "$s2" ] && { echo "$(date '+%F %T') STALLED $ARM at ${s2}B" >> "$OVN/ANOMALIES.txt"
                             kill_tree $APID; break; }
    done ) & WD=$!
  wait $APID; rc=$?; kill $WD 2>/dev/null; wait $WD 2>/dev/null
  if [ $rc -eq 0 ] && ls "$A"/artifacts/*.pt >/dev/null 2>&1; then
    log "$ARM OK ($(ls "$A"/artifacts/*.pt | wc -l) artifact)"; OK=$((OK+1))
  else
    case $rc in 124|137) R=timeout;; *) R="rc$rc";; esac
    log "$ARM FAILED ($R)"; FAILED="$FAILED ${ARM}:${R}"
  fi
done
# POST-CONDITION. The signature of "the treatment did nothing" must be distinguishable from the
# signature of "the treatment did nothing USEFUL" -- that is C-324's whole principle. Identical
# weights mean the screen is void again, and this must be known BEFORE any score is read.
if [ "$OK" -eq 2 ]; then
  phase "WEIGHT-HASH CHECK"
  $CENV python - "${ARMS[@]}" >> "$OVN/run.log" 2>&1 <<'PYEOF'
import hashlib, sys, glob, torch
hashes = {}
for m in sys.argv[1:]:
    f = sorted(glob.glob(f"/home/simon/Documents/scripts/views_platform/views-models/models/{m}/artifacts/*.pt"))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(sd[k].detach().cpu().numpy().tobytes())
    hashes[m] = h.hexdigest()
    print(f"weight-hash {m}: {hashes[m][:24]}")
if len(set(hashes.values())) == 1:
    print("VOID: the arms trained to IDENTICAL weights — the treatment was inert again (C-324). "
          "Do NOT read the scores; they would describe the harness.")
    sys.exit(3)
print("weights differ — the arms are genuinely distinct")
PYEOF
  if [ $? -ne 0 ]; then
    log "!! WEIGHT-HASH CHECK FAILED — arms are identical. SCREEN IS VOID; scores must not be read."
    echo "$(date '+%F %T') VOID: identical trained weights" >> "$OVN/ANOMALIES.txt"
  else
    log "weight-hash check PASSED — the arms are genuinely distinct"
  fi
fi
log "=== $OK ok | failed:${FAILED:- none} ==="
{ echo "# BPTT-SA screen — run summary"; echo
  echo "- finished: $(date '+%F %T')"; echo "- elapsed: $(( ($(date +%s)-START)/60 )) min"
  echo "- arms trained: $OK   FAILED: ${FAILED:- none}"; echo
  for ARM in "${ARMS[@]}"; do
    echo "## $ARM"
    ls "$MODELS/$ARM"/artifacts/*.pt 2>/dev/null | sed 's|.*/|- artifact: |' || echo "- no artifact"
  done; } > "$RES/SCREEN_SUMMARY.md"
