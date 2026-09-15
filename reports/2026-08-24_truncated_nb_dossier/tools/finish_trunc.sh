#!/usr/bin/env bash
# finish_trunc.sh — final verify once the queue writes its sentinel.
#
# The queue already re-runs the verifier after every arm, so VERDICT.md is never stale. This
# exists for the one case that leaves: the LAST arm's verify crashing, which stops the queue but
# leaves the verdict written from the arm before it. Waits on the sentinel ONLY — never pgrep,
# which loses the start-up race and can assemble from a previous run (summarise_freeze.sh's scar).
set -uo pipefail
RES=/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-08-24_truncated_nb_dossier/results
TOOL=/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-08-24_truncated_nb_dossier/tools/verify_trunc.py
# C-310, recorded 2026-08-24: this 12 h ceiling was TOO SHORT. It was sized from a ~2 h/arm
# estimate; arms actually took ~2.4 h (137-192 min train+emit plus 47-56 min oracle, the truncated
# sampler being ~7x slower than nb), so it expired at ~11:34 while the queue ran until ~13:2x and
# the finisher REFUSED to assemble. That is the guard working — it wrote "QUEUE_DONE never
# appeared" rather than emitting a verdict built from three of four arms — but the estimate was
# wrong. Left at the as-run value so the record matches what executed; the rule for next time is
# in C-310: size from the MEASURED worst case x 2, not the expected case.
for _ in $(seq 1 1440); do            # 12 h ceiling — see C-310 above; too short in practice
  [ -f "$RES/QUEUE_DONE" ] && break
  sleep 30
done
[ -f "$RES/QUEUE_DONE" ] || {
  echo "finish_trunc: QUEUE_DONE never appeared — refusing to assemble a possibly stale verdict" \
    >> "$RES/finish.log"; exit 1; }
conda run --no-capture-output -n views-hydranet-env python "$TOOL" >> "$RES/finish.log" 2>&1
echo "finish_trunc: final verify rc=$? at $(date '+%F %T')" >> "$RES/finish.log"
