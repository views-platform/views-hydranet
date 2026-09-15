#!/usr/bin/env bash
# smoke.sh — the REAL gate: every architecture trains and emits for 2 lessons before any 300-lesson
# arm is allowed to start.
#
# This is also the only honest memory measurement in the programme. `preflight.py` profiles a single
# forward/backward on one 32x32 window; a /falsify audit measured that check's margin at x292 and
# showed a 64x-wider architecture passing it silently. Here the real pipeline allocates for real,
# and **peak footprint does not depend on lesson count** — so a 2-lesson arm measures a 300-lesson
# arm exactly. It also exercises what no unit test reaches: the emit path, the scorer's view of the
# cube, BatchNorm recalibration, and the data pipeline.
#
# Writes results/smoke.json and results/SMOKE_OK. launch_bakeoff.sh refuses without the sentinel.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-24_architecture_bakeoff_dossier"
RES="$D/results"; MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
HEADROOM_FRAC=${HEADROOM_FRAC:-0.80}
SEED=42
mkdir -p "$RES"; : > "$RES/smoke.jsonl"
export WANDB_MODE=offline WANDB_SILENT=true
unset ALLOW_RE_REPORT

fail=0
for ARCH in AntiAliasedPool DynamicTopSkip FiLMSkip ShallowPool DualStream WideMemory; do
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$D/tools')
from make_arch_arm import arm_label; print(arm_label(lessons=2, eps=0.0, seed=$SEED, variant='$ARCH'))" 2>/dev/null)
  [ -n "$label" ] || { echo "SMOKE $ARCH: no legal label"; fail=1; continue; }

  if [ ! -d "$MODELS/$label" ]; then
    $CENV python "$D/tools/make_arch_arm.py" --lessons 2 --eps 0.0 --seed $SEED --variant "$ARCH" \
      >>"$RES/smoke.log" 2>&1 || { echo "SMOKE $ARCH: builder refused"; fail=1; continue; }
  fi
  A="$MODELS/$label"
  rm -rf "$A"/data/generated/predictions_* "$A"/data/generated/_pf_staging 2>/dev/null

  # poll GPU memory for the whole run; the box is otherwise idle so total-used is this process
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 2; done ) \
      > "$RES/.mem_$label" 2>/dev/null &
  poll=$!
  start=$(date +%s)
  ( cd "$A" && timeout -k 60 3600 $CENV python main.py -r calibration -t -e -sa ) \
      >>"$RES/smoke_${label}.log" 2>&1
  rc=$?
  kill $poll 2>/dev/null; wait $poll 2>/dev/null
  dur=$(( $(date +%s) - start ))
  peak=$(sort -n "$RES/.mem_$label" 2>/dev/null | tail -1); rm -f "$RES/.mem_$label"
  cube=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | head -1)

  ok=1
  [ "$rc" -eq 0 ] || { echo "SMOKE $ARCH: train+emit rc=$rc"; ok=0; }
  [ -n "$cube" ] || { echo "SMOKE $ARCH: no prediction cube emitted"; ok=0; }
  if [ -n "$peak" ] && [ "$peak" -gt "$(python3 -c "print(int($GPU_TOTAL_MIB*$HEADROOM_FRAC))")" ]; then
    echo "SMOKE $ARCH: peak ${peak} MiB exceeds ${HEADROOM_FRAC} of ${GPU_TOTAL_MIB} MiB"; ok=0
  fi
  printf '{"arch":"%s","label":"%s","rc":%d,"peak_mib":%s,"seconds":%d,"cube":%s,"ok":%d}\n' \
    "$ARCH" "$label" "$rc" "${peak:-null}" "$dur" "$([ -n "$cube" ] && echo true || echo false)" "$ok" \
    >> "$RES/smoke.jsonl"
  printf "  %-16s rc=%-3s peak=%-6s MiB  %4ds  cube=%s  %s\n" \
    "$ARCH" "$rc" "${peak:-?}" "$dur" "$([ -n "$cube" ] && echo yes || echo NO)" \
    "$([ "$ok" -eq 1 ] && echo OK || echo FAIL)"
  [ "$ok" -eq 1 ] || fail=1
  rm -rf "$cube" 2>/dev/null   # smoke cubes are never scored; do not let them accumulate
done

if [ "$fail" -eq 0 ]; then
  cp "$RES/smoke.jsonl" "$RES/SMOKE_OK"; echo "SMOKE OK — sentinel written"
else
  rm -f "$RES/SMOKE_OK"; echo "SMOKE FAILED — no sentinel; the queue will refuse to launch"
fi
exit $fail
