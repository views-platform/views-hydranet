#!/usr/bin/env bash
# summarise_freeze.sh — assemble the freeze table the moment the run finishes.
#
# setsid-detached finisher on a sentinel, per the standing infra rule: background jobs are reaped
# when the assistant idles, setsid daemons are not. Writes results/SUMMARY.md so the answer is on
# disk in the morning whether or not anyone is watching.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
RES="$HYD/reports/2026-08-22_state_freeze_l300_dossier/results"
for _ in $(seq 1 720); do            # 6 h ceiling; the run's own timeout is 6 h per seed
  [ -f "$RES/FREEZE_DONE" ] && break
  pgrep -f run_freeze_l300.sh >/dev/null 2>&1 || break
  sleep 30
done
conda run --no-capture-output -n views-hydranet-env python \
  "$HYD/reports/2026-08-22_state_freeze_l300_dossier/tools/freeze_table.py" \
  --results "$RES" --out "$RES/SUMMARY.md" > "$RES/summarise.log" 2>&1
