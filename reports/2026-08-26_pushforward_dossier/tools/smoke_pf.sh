#!/usr/bin/env bash
# smoke_pf.sh — the gate before any scored pushforward arm.
#
# Descends from the architecture bake-off's smoke.sh, and inherits its central claim: peak GPU
# footprint does NOT depend on lesson count, so a 2-lesson arm measures a 300-lesson arm exactly.
# It is also the only honest cost measurement — the CPU-side probe in the audit said x1.76, but
# that ran one window in isolation, not the real pipeline with its emit path, BN recalibration and
# data plumbing.
#
# Runs the treatment AND its control, because the number that matters is the RATIO, and a cost
# multiplier measured against yesterday's timing on a different machine state is not a measurement.
#
# Writes results/smoke_pf.jsonl and results/SMOKE_PF_OK.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-26_pushforward_dossier"
RES="$D/results"; MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
HEADROOM_FRAC=${HEADROOM_FRAC:-0.80}
SEED=42
mkdir -p "$RES"; : > "$RES/smoke_pf.jsonl"
export WANDB_MODE=offline WANDB_SILENT=true

fail=0
for VARIANT in 0.0 0.1; do
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$D/tools')
from make_pf_arm import arm_label; print(arm_label(lessons=2, eps=0.0, seed=$SEED, variant='$VARIANT'))" 2>/dev/null)
  [ -n "$label" ] || { echo "SMOKE $VARIANT: no legal label"; fail=1; continue; }

  if [ ! -d "$MODELS/$label" ]; then
    $CENV python "$D/tools/make_pf_arm.py" --lessons 2 --eps 0.0 --seed $SEED --variant "$VARIANT" \
      >>"$RES/smoke_pf.log" 2>&1 || { echo "SMOKE $VARIANT: builder refused"; fail=1; continue; }
  fi
  A="$MODELS/$label"
  rm -rf "$A"/data/generated/predictions_* "$A"/data/generated/_pf_staging 2>/dev/null

  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 2; done ) \
      > "$RES/.mem_$label" 2>/dev/null &
  poll=$!
  start=$(date +%s)
  ( cd "$A" && timeout -k 60 5400 $CENV python main.py -r calibration -t -e -sa ) \
      >>"$RES/smoke_${label}.log" 2>&1
  rc=$?
  kill $poll 2>/dev/null; wait $poll 2>/dev/null
  dur=$(( $(date +%s) - start ))
  peak=$(sort -n "$RES/.mem_$label" 2>/dev/null | tail -1); rm -f "$RES/.mem_$label"
  cube=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | head -1)

  ok=1
  [ "$rc" -eq 0 ] || { echo "SMOKE $VARIANT: train+emit rc=$rc"; ok=0; }
  [ -n "$cube" ] || { echo "SMOKE $VARIANT: no prediction cube emitted"; ok=0; }
  if [ -n "$peak" ] && [ "$peak" -gt "$(python3 -c "print(int($GPU_TOTAL_MIB*$HEADROOM_FRAC))")" ]; then
    echo "SMOKE $VARIANT: peak ${peak} MiB exceeds ${HEADROOM_FRAC} of ${GPU_TOTAL_MIB} MiB"; ok=0
  fi
  printf '{"variant":"%s","label":"%s","rc":%d,"peak_mib":%s,"seconds":%d,"cube":%s,"ok":%d}\n' \
    "$VARIANT" "$label" "$rc" "${peak:-null}" "$dur" "$([ -n "$cube" ] && echo true || echo false)" "$ok" \
    >> "$RES/smoke_pf.jsonl"
  printf "  %-6s %-24s rc=%-3s peak=%-6s MiB  %5ds  cube=%s  %s\n" \
    "$VARIANT" "$label" "$rc" "${peak:-?}" "$dur" "$([ -n "$cube" ] && echo yes || echo NO)" \
    "$([ "$ok" -eq 1 ] && echo OK || echo FAIL)"
  [ "$ok" -eq 1 ] || fail=1
  rm -rf "$cube" 2>/dev/null   # smoke cubes are never scored
done

if [ "$fail" -eq 0 ]; then
  cp "$RES/smoke_pf.jsonl" "$RES/SMOKE_PF_OK"
  $CENV python - <<'PY'
import json, pathlib
rows = [json.loads(l) for l in
        pathlib.Path("/home/simon/Documents/scripts/views_platform/views-hydranet/"
                     "reports/2026-08-26_pushforward_dossier/results/smoke_pf.jsonl").read_text().splitlines()]
by = {r["variant"]: r for r in rows}
if "0.0" in by and "0.1" in by:
    c, t = by["0.0"], by["0.1"]
    print(f"  RATIO  time x{t['seconds']/max(c['seconds'],1):.2f}   "
          f"peak x{t['peak_mib']/max(c['peak_mib'],1):.2f}  "
          f"({c['peak_mib']} -> {t['peak_mib']} MiB)")
    print(f"  PROJECTED at 300 lessons: control 1.82 h/arm -> "
          f"treatment {1.82*t['seconds']/max(c['seconds'],1):.2f} h/arm")
PY
  echo "SMOKE_PF OK — sentinel written"
else
  rm -f "$RES/SMOKE_PF_OK"; echo "SMOKE_PF FAILED — no sentinel"
fi
exit $fail
