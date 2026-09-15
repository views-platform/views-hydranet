#!/usr/bin/env bash
# stage_a.sh — the cheap-vehicle probe. One 40L nb control at HEAD, then the floor gate.
# Decides whether the sweep costs ~8.4 h (40L) or ~26 h (160L) — or whether no vehicle exists.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-17_ss_retention_dossier"; RES="$D/results"
A=/home/simon/Documents/scripts/views_platform/views-models/models/shortzero_fortytwo
CENV="conda run --no-capture-output -n views-hydranet-env"
unset ALLOW_RE_REPORT
export WANDB_MODE=offline WANDB_SILENT=true
echo "[$(date '+%F %T')] STAGE A start" | tee -a "$RES/stage_a.log"
cd "$A" || exit 2
timeout -k 120 7200 $CENV python main.py -r calibration -t -e -sa >> "$RES/stage_a_run.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] train+emit rc=$rc" | tee -a "$RES/stage_a.log"
touch "$RES/STAGE_A_RUN_DONE"
exit $rc
