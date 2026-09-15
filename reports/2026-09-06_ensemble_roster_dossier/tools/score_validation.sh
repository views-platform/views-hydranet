#!/usr/bin/env bash
# Score every completed validation run that has not been scored yet. Idempotent; safe to re-run.
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
D="$HYD/reports/2026-09-06_ensemble_roster_dossier"
OUT="$D/results/validation/scores"; mkdir -p "$OUT"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
TRUTH="$D/results/v2_truth_validation.parquet"   # months 505-552, built 2026-09-07

for f in "$D"/results/validation/done_*; do
  [ -e "$f" ] || continue
  m=$(basename "$f" | sed 's/^done_//')
  [ -f "$OUT/score_$m.csv" ] && continue
  pd=$(ls -d "$MODELS/$m"/data/generated/predictions_validation_* 2>/dev/null | head -1)
  if [ -z "$pd" ]; then echo "  $m: no prediction dir (already cleaned?)"; continue; fi
  echo "scoring $m"
  "$PY" "$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py" \
     "$m|$pd|lr_{t}_best|by_{t}_best" --targets=sb,ns,os --horizons=1,6,12,18,24,36 \
     --truth="$TRUTH" --out="$OUT/score_$m.csv" > "$OUT/log_$m.txt" 2>&1 \
     && echo "  ok" || echo "  SCORING FAILED (see $OUT/log_$m.txt)"
done
