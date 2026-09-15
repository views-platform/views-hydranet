#!/usr/bin/env bash
# summarise_freeze.sh — assemble the freeze table the moment the run finishes.
#
# setsid-detached finisher on a sentinel, per the standing infra rule: background jobs are reaped
# when the assistant idles, setsid daemons are not. Writes results/SUMMARY.md so the answer is on
# disk in the morning whether or not anyone is watching.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
RES="$HYD/reports/2026-08-22_state_freeze_l300_dossier/results"
# The SENTINEL is the only safe wait condition. An earlier version also broke out when
# `pgrep run_freeze_l300.sh` found nothing — which loses the start-up race if this finisher is
# armed before the setsid driver has exec'd, and then assembles SUMMARY.md from the PREVIOUS run's
# score CSVs with no sign it is stale. Give the driver a grace period, then wait on the sentinel
# alone and refuse to write anything if it never appears.
GRACE=20
for _ in $(seq 1 720); do            # 6 h ceiling; the run's own timeout is 6 h per seed
  [ -f "$RES/FREEZE_DONE" ] && break
  if [ "$GRACE" -gt 0 ]; then GRACE=$((GRACE-1)); fi
  sleep 30
done
[ -f "$RES/FREEZE_DONE" ] || {
  echo "summarise_freeze: FREEZE_DONE never appeared — refusing to assemble a possibly stale summary" \
    >> "$RES/summarise.log"
  exit 1
}
conda run --no-capture-output -n views-hydranet-env python \
  "$HYD/reports/2026-08-22_state_freeze_l300_dossier/tools/freeze_table.py" \
  --results "$RES" --out "$RES/SUMMARY.md" > "$RES/summarise.log" 2>&1
