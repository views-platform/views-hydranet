#!/usr/bin/env bash
# smoke_exp02.sh — the REAL gate for EXP-02 (candidate 7). A faithful sibling of the EXP-01
# dossier's `smoke.sh`, changed in exactly two places: the architecture list and the dossier dir.
#
# WHY A SIBLING AND NOT AN EDIT. `smoke.sh:26` carries a hard-coded six-candidate list and writes
# EXP-01's `SMOKE_OK`. Editing it would alter an instrument that produced a committed experiment,
# and would overwrite that experiment's sentinel. PREREG.md section 10b.
#
# WHAT IT MEASURES, in its parent's words: "the only honest memory measurement in the programme.
# `preflight.py` profiles a single forward/backward on one 32x32 window; a /falsify audit measured
# that check's margin at x292 and showed a 64x-wider architecture passing it silently. Here the real
# pipeline allocates for real, and peak footprint does not depend on lesson count — so a 2-lesson
# arm measures a 300-lesson arm exactly. It also exercises what no unit test reaches: the emit path,
# the scorer's view of the cube, BatchNorm recalibration, and the data pipeline."
#
# THE CONTROL IS AN ARM HERE. `HydraBNUNet06_LSTM4` is smoked too, because EXP-02 re-runs it rather
# than pairing against August's `fullzero_*` scores — those model directories are gone and their
# code version differs. PREREG.md section 10a.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-10-30_upsample_arm_dossier"
BAKEOFF="$HYD/reports/2026-08-24_architecture_bakeoff_dossier"   # the shared builder lives there
RES="$D/results"; MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
CENV="conda run --no-capture-output -n views-hydranet-env"
GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
HEADROOM_FRAC=${HEADROOM_FRAC:-0.80}
SEED=42
mkdir -p "$RES"; : > "$RES/smoke.jsonl"
export WANDB_MODE=offline WANDB_SILENT=true
unset ALLOW_RE_REPORT

# ⛔ REVERTED 2026-10-30. This block previously carried `export PYTHONPATH="$HYD"` and a
# `SMOKE_REFETCH=1` escape that dropped `-sa`. Both were bypasses of gates that had correctly
# refused, and both are gone:
#
#   * PYTHONPATH pointed the run at the working tree because the candidate does not exist in the
#     INSTALLED wheel. ⛔ The honest reading of that refusal is that the candidate is not released
#     yet — which is the production step, not an obstacle. All six EXP-01 candidates were in the
#     published package BEFORE their arms ran. Candidate 7 waits for 0.1.8.
#   * SMOKE_REFETCH dropped `--saved` to get past a C-61 StaleCacheError. ⛔ That refusal is
#     load-bearing: "without this check the cache would have been served and the run would have
#     used data that does not match its own configuration, with no error anywhere." A refetch is a
#     deliberate, logged decision with a data-vintage caveat, not a flag on a gate.

# ⛔ The control label is DERIVED from the shared builder, never retyped — make_arch_arm.py's
# `--control` is experiment-specific and its default (`fullzero_<seed>`) no longer exists on disk.
ctl_label() {
  $CENV python -c "
import sys; sys.path.insert(0,'$BAKEOFF/tools')
from make_arch_arm import arm_label; print(arm_label(lessons=2, eps=0.0, seed=$1, variant='$CTL'))" 2>/dev/null
}

# ⛔⛔ THE ARCHES COME FROM G-F'S SENTINEL, NOT FROM THIS LINE. `for ARCH in HydraBNUNet06_LSTM4
# UpsampleConv2` stood here until 2026-10-30 — one of four hand-typed copies of a decision
# `gf_phase_signature.py` makes and, at the time, only printed. A /falsify guard audit measured
# the cost: one word changed in the launcher and four arms trained, scored and were never
# identity-checked. Smoking an architecture the queue will not run is the same defect pointed the
# other way: the gate would pass on a model nobody trains.
SEL="$RES/SELECTED_ARM"
[ -f "$SEL" ] || { echo "SMOKE REFUSED: no SELECTED_ARM sentinel at $SEL. The arm is chosen by \
gf_phase_signature.py under PREREG section 2, not typed into this file — run G-F first."; exit 1; }
read -r CTL ARM < <(python3 -c "
import json
d = json.load(open('$SEL'))
print(d['control'], d['selected'])" 2>/dev/null) || true
[ -n "${CTL:-}" ] && [ -n "${ARM:-}" ] || { echo "SMOKE REFUSED: $SEL is unreadable or has no \
'control'/'selected' keys. Re-run gf_phase_signature.py rather than editing it by hand."; exit 1; }
echo "smoking the control and the SELECTED arm, read from the sentinel: $CTL, $ARM"

fail=0
for ARCH in "$CTL" "$ARM"; do
  label=$($CENV python -c "
import sys; sys.path.insert(0,'$BAKEOFF/tools')
from make_arch_arm import arm_label; print(arm_label(lessons=2, eps=0.0, seed=$SEED, variant='$ARCH'))" 2>/dev/null)
  [ -n "$label" ] || { echo "SMOKE $ARCH: no legal label"; fail=1; continue; }

  if [ ! -d "$MODELS/$label" ]; then
    $CENV python "$BAKEOFF/tools/make_arch_arm.py" --lessons 2 --eps 0.0 --seed $SEED \
      --variant "$ARCH" --control "$(ctl_label $SEED)" >>"$RES/smoke.log" 2>&1 \
      || { echo "SMOKE $ARCH: builder refused"; fail=1; continue; }
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
  printf "  %-22s rc=%-3s peak=%-6s MiB  %4ds  cube=%s  %s\n" \
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
