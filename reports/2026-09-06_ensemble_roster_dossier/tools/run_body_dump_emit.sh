#!/usr/bin/env bash
# Re-emit the 8 validation artifacts WITH the body-mean dump. Emit only — no retraining.
#
# The prediction cube holds the gate and the already-COMPOSED forecast; the un-composed body is
# never written, and recovering it as composed/gate is division noise where the gate is near zero
# (i.e. almost everywhere). The dump writes mu and gate as separate [T,n_reg,H,W] / [T,H,W,n_cls]
# grids. tests/distributions/test_body_mean_dump.py asserts the cube is byte-identical with the
# dump on and off, so this cannot disturb the scores already computed — verified again below.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
D="$HYD/reports/2026-09-06_ensemble_roster_dossier"
OUT="$D/results/bodydump"; mkdir -p "$OUT"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
ENTRY="$D/tools/roster_arm_entry.py"

ROSTER="purple_alien pink_pirate blue_stranger bold_comet blazing_meteor heavy_freighter bright_starship violet_visitor"

for m in $ROSTER; do
  [ -f "$OUT/done_$m" ] && { echo "SKIP $m"; continue; }
  MD="$MODELS/$m"
  art=$(ls -t "$MD"/artifacts/validation_model_*.pt 2>/dev/null | head -1 | xargs -r basename)
  if [ -z "$art" ]; then echo "MISSING artifact for $m"; continue; fi

  # Refuse on a leftover prediction dir: the pipeline names it after the artifact, so a stale one
  # makes the emitted cube ambiguous. This cost an hour on 2026-09-06.
  if [ -n "$(ls -d "$MD"/data/generated/predictions_validation_* 2>/dev/null)" ]; then
    echo "  note: removing the existing cube for $m before re-emit (it will be regenerated)"
    rm -rf "$MD"/data/generated/predictions_validation_*
  fi

  # Dump dir derived from the model name, never caller-supplied: a flat shared directory would let
  # two models write bodymean_origin*.npz over each other and leave a complete-looking result that
  # is silently half one model and half the other.
  dump="$OUT/bodymean_$m"
  rm -rf "$dump"

  echo "=== $m ($art)  $(date +%H:%M:%S) ==="
  ( cd "$MD" && "$PY" "$ENTRY" --model-dir "$MD" --artifact "$art" \
      --composition "$(grep -hoE "'forecast_composition':\s*'[^']+'" "$MD/configs/config_hyperparameters.py" | head -1 | sed "s/.*'\(.*\)'/\1/")" \
      --gate-threshold 0.5 --freeze cell --body-mean-dump "$dump" \
      --run-type validation ) > "$OUT/log_$m.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc (see $OUT/log_$m.txt)"; continue; fi
  grep -m1 "POTENCY OK" "$OUT/log_$m.txt" | sed 's/^/  /'
  n=$(ls "$dump"/bodymean_origin*.npz 2>/dev/null | wc -l)
  echo "  dumps: $n  ($(du -sh "$dump" 2>/dev/null | cut -f1))"
  [ "$n" -eq 13 ] || { echo "  EXPECTED 13 origin dumps, got $n — incomplete arm"; continue; }
  touch "$OUT/done_$m"
done
echo "BODY DUMP EMIT COMPLETE $(date)" > "$OUT/EMIT_DONE"
