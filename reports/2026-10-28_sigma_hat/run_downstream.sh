#!/bin/bash
# RP-2 downstream chain: d4 cube -> B-1a (variogram + scoreboard) and B-1b (placement), per replicate.
#
# ⛔⛔ EVERY CALL PASSES --tag. Three reasons, all of them things that have already gone wrong here:
#   * d12_variogram.py writes d12_result{TAG}.npz and d12_fits{TAG}.csv UNCONDITIONALLY. Without a
#     tag it overwrites the control's curves -- and those are the irreplaceable output of a
#     checkpoint that NO LONGER EXISTS on disk. This is a standing never-do.
#   * body_scoreboard.py reads d4_result{TAG}.npz, d12_result{TAG}.npz and d12_fits{TAG}.csv.
#   * b1b_placement_baseline.py reads d4_result{TAG}.npz (--tag added 2026-10-28, default inert,
#     verified by diff against the pre-change output -- gate G-2b).
#
# Run AFTER all three d4 passes finish, not alongside them: each script prints its own measured
# wall time and peak RSS, and CPU contention would make those numbers a property of the schedule.
set -u
cd /home/simon/Documents/scripts/views_platform/views-hydranet
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
D=reports/2026-10-28_sigma_hat
U=reports/2026-10-12_underfire_diagnosis
G=reports/2026-10-26_body_optimisation_gap

for t in _s42 _s43 _s44; do
  if [ ! -f "$U/d4_result$t.npz" ]; then
    echo "⛔ ABORT: $U/d4_result$t.npz missing — d4 did not produce this replicate."
    exit 1
  fi
done

for t in _s42 _s43 _s44; do
  echo "=== $t d12_variogram  $(date -Is) ==="
  $PY $U/d12_variogram.py "--tag=$t"            > "$D/d12_variogram$t.out" 2>&1 || exit 1
  echo "=== $t body_scoreboard  $(date -Is) ==="
  $PY $U/body_scoreboard.py "--tag=$t"          > "$D/body_scoreboard$t.out" 2>&1 || exit 1
  echo "=== $t b1b_placement  $(date -Is) ==="
  $PY $G/b1b_placement_baseline.py "--tag=$t"   > "$D/b1b_placement_baseline$t.out" 2>&1 || exit 1
done
echo "DOWNSTREAM_DONE $(date -Is)" > "$D/downstream_sentinel.txt"
