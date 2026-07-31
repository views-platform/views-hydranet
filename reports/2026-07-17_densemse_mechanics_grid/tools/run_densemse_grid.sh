#!/usr/bin/env bash
# DENSE-MSE MECHANICS GRID (2026-07-17) — all-cell plain point body.
# 8 runs: act{softplus,relu} x space{log=mse, count=count_mean} x seed{42,43}. Log-space first (safe),
# count-space last (count_mean landmine — expected to diverge). STEALTH patch + trap-restore + per-run
# timeout + best-effort auto-score. Order chosen so the informative arms finish first.
set -uo pipefail

VV=/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor
CFG=$VV/configs/config_hyperparameters.py
GEN=$VV/data/generated
PARQ=$VV/data/raw/calibration_viewser_df.parquet
SCRATCH=/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/faf1c146-3d1e-4018-bf86-28d563362784/scratchpad
HN=/home/simon/Documents/scripts/views_platform/views-hydranet
DIAG=$HN/reports/plots/diagnostics
FLOOR_BAK=$SCRATCH/floor_config_backup.py
MAN=$SCRATCH/dm_manifest.txt
EXPECT_FLOOR_MD5=6c28bdb1390fc413d43b2d74d87251f8
RUN_TIMEOUT=3000   # 50 min/run — a 40L TRAIN-ONLY run is ~32 min; guards a count_mean hang
# TRAIN-ONLY (no -e): the 36-step rollout eval blooms to inf for the alive all-cell 'standard' body
# (C-113). Per user scope (step-0/in-sample; feedback-clamp rejected), read the body IN-SAMPLE from the
# training forensics + the saved artifact. No rollout, no clamp.

GOT=$(md5sum "$CFG" | awk '{print $1}')
if [ "$GOT" != "$EXPECT_FLOOR_MD5" ]; then
  echo "ABORT: config not the expected floor (got $GOT)." | tee "$MAN"; exit 3
fi
cp "$CFG" "$FLOOR_BAK"
restore() {
  cp "$FLOOR_BAK" "$CFG"
  R=$(md5sum "$CFG" | awk '{print $1}')
  echo "[restore floor] $(date) md5=$R $([ "$R" = "$EXPECT_FLOOR_MD5" ] && echo OK || echo MISMATCH!)" | tee -a "$MAN"
}
trap restore EXIT

echo "=== DENSE-MSE GRID started $(date) ===" | tee "$MAN"
cd "$VV"
# log-space (safe) first, count-space (landmine) last
ARMS="sp_log_s42 sp_log_s43 relu_log_s42 relu_log_s43 sp_count_s42 sp_count_s43 relu_count_s42 relu_count_s43"
for ARM in $ARMS; do
  cp "$SCRATCH/config_dm_${ARM}.py" "$CFG"
  LOG=$SCRATCH/dm_${ARM}.log
  echo "=== START $ARM $(date) ===" | tee -a "$MAN"
  timeout $RUN_TIMEOUT conda run -n views-hydranet-env python main.py -r calibration -t >"$LOG" 2>&1
  EX=$?
  ART=$(ls -t "$VV"/artifacts/calibration_model_*.pt 2>/dev/null | head -1)
  DIAGDIR=$(ls -td "$DIAG"/* 2>/dev/null | head -1)
  echo "=== DONE $ARM exit=$EX $(date) ===" | tee -a "$MAN"
  echo "    artifact: $ART" | tee -a "$MAN"
  echo "    diagnostics: $DIAGDIR" | tee -a "$MAN"
done
echo "=== DENSE-MSE GRID COMPLETE $(date) ===" | tee -a "$MAN"
