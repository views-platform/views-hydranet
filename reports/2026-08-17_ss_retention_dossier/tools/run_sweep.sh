#!/usr/bin/env bash
# run_sweep.sh — the 7 remaining SS arms, unattended and resumable. See ../LAUNCH.md.
#
# The seed-42 eps=0 control is already trained and gated; this runs the rest and then emits VERDICT.md.
# An arm that fails is logged and skipped rather than stopping the sweep — but the verdict reports how
# many arms actually completed, and refuses to claim a result it cannot support.
set -uo pipefail

HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-17_ss_retention_dossier"; RES="$D/results"
T="$D/tools"
CENV="conda run --no-capture-output -n views-hydranet-env"
PINNED_MD5=6d5714d5ceda147ed16f53143abe7e37   # matches 05_analysis_plan.md §5

mkdir -p "$RES"
exec 9>"$RES/.sweep.lock"; flock -n 9 || { echo "another sweep holds the lock"; exit 11; }
log(){ echo "[$(date '+%F %T')] SWEEP: $*" | tee -a "$RES/run.log"; }

# ---- the gate handshake: refuse to run treatment arms without a valid licence -------------------
GATE=$(ls "$RES"/FLOORGATE_longzero_fortytwo_PASS 2>/dev/null | head -1)
[ -n "$GATE" ] || { log "ABORT — no FLOORGATE_..._PASS. The vehicle is not licensed."; exit 2; }
grep -q "$PINNED_MD5" "$GATE" || {
  log "ABORT — the gate token's threshold md5 does not match the pre-registration ($PINNED_MD5)."
  log "        A threshold was changed after a control was seen; the licence is void."
  exit 3
}
log "gate licence OK ($(basename "$GATE"))"
( cd "$HYD" && git rev-parse HEAD ) > "$RES/sweep.head"
log "HEAD $(cut -c1-8 < "$RES/sweep.head")"

run_one(){          # $1=lessons $2=eps $3=seed
  local L="$1" E="$2" S="$3"
  local label; label=$($CENV python -c "
import sys; sys.path.insert(0,'$T')
from make_ss_arm import arm_label
print(arm_label(lessons=$L, eps=$E, seed=$S))" 2>/dev/null | tail -1)
  [ -n "$label" ] || { log "ABORT — could not derive a label for ($L,$E,$S)"; return 9; }

  if [ -s "$RES/score_${label}.csv" ]; then log "SKIP $label — already scored"; return 0; fi

  if [ ! -d "/home/simon/Documents/scripts/views_platform/views-models/models/$label" ]; then
    $CENV python "$T/make_ss_arm.py" --lessons "$L" --eps "$E" --seed "$S" >> "$RES/run.log" 2>&1 \
      || { log "ARM BUILD FAILED $label"; return 8; }
    log "built $label"
  fi
  bash "$T/run_arm.sh" "$label" >> "$RES/run.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && log "$label OK" || log "$label FAILED rc=$rc (continuing)"
  return 0
}

log "=== 7 arms: 3 x eps=0 (seeds 43,44,45) + 4 x eps=0.5 (seeds 42-45) ==="
for s in 43 44 45;    do run_one 160 0.0 "$s"; done
for s in 42 43 44 45; do run_one 160 0.5 "$s"; done

log "=== verify + verdict ==="
$CENV python "$T/verify_sweep.py" >> "$RES/run.log" 2>&1
log "VERDICT: $(head -1 "$RES/VERDICT.md" 2>/dev/null)"
touch "$RES/SWEEP_COMPLETE"
log "SWEEP COMPLETE"
