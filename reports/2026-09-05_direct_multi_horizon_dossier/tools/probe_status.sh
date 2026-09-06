#!/usr/bin/env bash
# Compact PROBE-A status — the arm logs are progress-bar heavy, so never cat them.
D=/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-09-05_direct_multi_horizon_dossier
echo "done arms : $(ls $D/results/done_* 2>/dev/null | wc -l) / 12"
echo "running   : $(pgrep -f run_probe_a.sh >/dev/null && echo yes || echo NO)"
grep -E "^===|exit=" $D/results/driver.log 2>/dev/null | tail -4
[ -f "$D/results/PROBE_A_DONE" ] && echo "*** COMPLETE ***"
