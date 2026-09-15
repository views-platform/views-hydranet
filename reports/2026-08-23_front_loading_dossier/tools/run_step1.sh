#!/usr/bin/env bash
# Runs step1_decompose.py for all four L=300 seeds. Absolute paths throughout: ModelPathManager
# resolves the model root from CWD, so the tool must run from views-models.
set -euo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
TOOL="$HYD/reports/2026-08-23_front_loading_dossier/tools/step1_decompose.py"
RES="$HYD/reports/2026-08-23_front_loading_dossier/results"
mkdir -p "$RES"; cd /home/simon/Documents/scripts/views_platform/views-models
for spec in \
  "fortytwo:fullzero_fortytwo" "fortythree:fullzero_fortythree" \
  "fortyfour:fullzero_fortyfour" "fortyfive:fullzero_fortyfive"
do
  L=${spec%%:*}; DIR=${spec#*:}
  ART=$(basename "$(ls "$MODELS/$DIR"/artifacts/*.pt | head -1)")
  conda run -n views-hydranet-env python "$TOOL" --model-dir "$MODELS/$DIR" --artifact "$ART" \
    --out "$RES" --seed-label "$L" --origin 335 --period 371 "$@" 2>&1 | grep -E "^$L:" || \
    echo "$L: FAILED (see $RES/$L.log)"
done
