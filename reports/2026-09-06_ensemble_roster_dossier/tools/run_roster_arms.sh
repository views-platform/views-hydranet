#!/usr/bin/env bash
# 16 emits: 8 roster models x {soft_gate, threshold_gate}, cell clamp ON. No retraining.
# Plus 4 clamp-OFF control arms on 2 models (Q1: does the clamp transfer to THESE models?).
#
# Driven arm-by-arm with score-then-delete, because the pipeline names the prediction dir after the
# ARTIFACT, not the arm — two arms of one model write the same path and the second silently
# overwrites the first. That mistake cost an hour of GPU earlier today.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
OUT="$HYD/reports/2026-09-06_ensemble_roster_dossier/results"
ENTRY="$HYD/reports/2026-09-06_ensemble_roster_dossier/tools/roster_arm_entry.py"
V2T="$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
mkdir -p "$OUT"

declare -A ART=(
  [violet_visitor]=calibration_model_20260812_191742.pt
  [bright_starship]=calibration_model_20260813_174320.pt
  [bold_comet]=calibration_model_20260812_215145.pt
  [blazing_meteor]=calibration_model_20260812_232850.pt
  [heavy_freighter]=calibration_model_20260813_010047.pt
  [pink_pirate]=calibration_model_20260813_025117.pt
  [blue_stranger]=calibration_model_20260813_042946.pt
  [purple_alien]=calibration_model_20260813_062540.pt
)
ROSTER="violet_visitor bright_starship bold_comet blazing_meteor heavy_freighter pink_pirate blue_stranger purple_alien"
# Q1 control: one nb, one mixture_nb, emitted clamp-OFF as well.
CLAMP_CONTROL="violet_visitor pink_pirate"

run_arm() {
  local m="$1" comp="$2" fz="$3" tag="${1}_${2}_${3}"
  [ -f "$OUT/done_${tag}" ] && { echo "SKIP $tag"; return 0; }
  local MD="$MODELS/$m"
  # Refuse on a leftover prediction dir rather than score an ambiguous one.
  local before; before=$(ls -d "$MD"/data/generated/predictions_* 2>/dev/null | wc -l)
  if [ "$before" -ne 0 ]; then echo "GUARD FAIL $tag: stale prediction dir"; return 1; fi
  echo "=== $tag $(date +%H:%M:%S) ==="
  ( cd "$MD" && "$PY" "$ENTRY" --model-dir "$MD" --artifact "${ART[$m]}" \
      --composition "$comp" --freeze "$fz" ) > "$OUT/log_${tag}.txt" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  EMIT FAILED rc=$rc"; return 1; fi
  local pd; pd=$(ls -d "$MD"/data/generated/predictions_* 2>/dev/null | head -1)
  if [ -z "$pd" ]; then echo "  no prediction dir produced"; return 1; fi
  "$PY" "$V2T/score_v2_horizons.py" "${tag}|${pd}|lr_{t}_best|by_{t}_best" \
      --targets=sb,ns,os --horizons=1,3,6,12,18,24,30,36 \
      --out="$OUT/score_${tag}.csv" >> "$OUT/log_${tag}.txt" 2>&1
  local sc=$?
  if [ $sc -eq 0 ]; then
    cp -r "$pd" "$OUT/cube_${tag}" 2>/dev/null   # kept for the pooled-ensemble and error-correlation work
    touch "$OUT/done_${tag}"; echo "  scored"
  else
    echo "  SCORING FAILED"
  fi
  rm -rf "$pd"
}

for m in $ROSTER; do
  for comp in soft_gate threshold_gate; do run_arm "$m" "$comp" cell; done
done
for m in $CLAMP_CONTROL; do
  for comp in soft_gate threshold_gate; do run_arm "$m" "$comp" none; done
done
echo "ROSTER ARMS COMPLETE $(date)" > "$OUT/ROSTER_DONE"
