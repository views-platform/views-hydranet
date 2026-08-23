#!/usr/bin/env bash
# run_capture.sh — drive capture_regimes.py for both seeds.
#
# Exists because `ModelPathManager` resolves the model root from the CURRENT WORKING DIRECTORY, so
# the tool must run from views-models — which in turn means every path handed to it has to be
# absolute. Passing a repo-relative tool path silently produced "can't open file" and zero output.
set -euo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
TOOL="$HYD/reports/2026-08-23_state_range_dossier/tools/capture_regimes.py"
RES="$HYD/reports/2026-08-23_state_range_dossier/results"
CENV="conda run -n views-hydranet-env"
[[ -f "$TOOL" ]] || { echo "tool not found: $TOOL" >&2; exit 1; }
mkdir -p "$RES"
cd /home/simon/Documents/scripts/views_platform/views-models
for spec in \
  "fortytwo:fullzero_fortytwo:calibration_model_20260818_221401.pt" \
  "fortythree:fullzero_fortythree:calibration_model_20260821_045948.pt"
do
  L=${spec%%:*}; rest=${spec#*:}; DIR=${rest%%:*}; ART=${rest#*:}
  echo "=== seed $L ==="
  $CENV python "$TOOL" --model-dir "$MODELS/$DIR" --artifact "$ART" \
      --out "$RES" --seed-label "$L" "$@" > "$RES/capture_$L.log" 2>&1
  grep -E "F3:|R1 ratio|HARD STOP" "$RES/capture_$L.log" | sed 's/.*INFO - //' || true
  ls -la "$RES"/r1_state_"$L"_*.pt "$RES"/r2_state_"$L".pt 2>/dev/null | awk '{printf "  %10d  %s\n",$5,$9}'
done
