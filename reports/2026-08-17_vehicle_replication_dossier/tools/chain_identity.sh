#!/usr/bin/env bash
# chain_identity.sh — run the `identity` control AFTER the main batch, on today's code.
#
# Why this exists (found 02:58, after the batch had already launched):
# the preserved 2026-08-12 cubes were written at 19:18 on 2026-08-12, but three commits that touch
# the inference path landed AFTER them — notably a2eabeb (2026-08-14 19:28) "per-site LockedDropout:
# independent MC-dropout masks per layer". `violet_visitor` evaluates with `evaluation_mode:
# stochastic` and `dropout_rate: 0.15`, so MC-dropout is live at inference and per-site masks change
# the posterior draws. Using the preserved cubes as the control would therefore confound each
# transform's effect with a dropout change.
#
# So `identity` is run today, on today's code, and becomes THE control. The preserved cubes keep two
# jobs: they proved the shipped board reproduces bit-for-bit (F1, passed), and identity-today minus
# cubes-2026-08-12 now measures what those three commits did to the free-running path — a byproduct
# worth having.
#
# Chained rather than inserted because the batch was already running; the repo's own pattern is a
# sentinel poll (`while [ ! -f "$SENT" ]; do sleep 60; done`).
set -uo pipefail

DOS=/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-08-17_vehicle_replication_dossier
OVN="$DOS/results/overnight"
SENT="$OVN/RUN_COMPLETE"

echo "[$(date '+%F %T')] chain: waiting for the main batch sentinel $SENT" >> "$OVN/chain.log"
for _ in $(seq 1 400); do            # ~6.7 h ceiling; the batch is expected in ~2.2 h
  [ -f "$SENT" ] && break
  sleep 60
done
[ -f "$SENT" ] || { echo "[$(date '+%F %T')] chain: TIMED OUT waiting for the batch" >> "$OVN/chain.log"; exit 1; }

sleep 45                              # grace: let the batch's EXIT trap finish writing
rm -f "$SENT" "$OVN/STATUS.txt"       # the chained run writes its own terminal state
echo "[$(date '+%F %T')] chain: batch done, starting the identity control" >> "$OVN/chain.log"

exec bash "$DOS/tools/overnight_run.sh" identity
