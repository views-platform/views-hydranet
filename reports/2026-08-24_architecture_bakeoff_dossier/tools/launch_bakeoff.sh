#!/usr/bin/env bash
# launch_bakeoff.sh — the ONLY sanctioned way to start the 12-arm bake-off.
#
# It exists because a `/falsify guard` audit found two guards that nothing ran: `PREFLIGHT_OK` was
# written and read by nobody ("the launcher refuses without it" — there was no launcher), and
# `arm_postflight` had zero call sites. Both are now load-bearing here.
#
# Refuses unless, in order:
#   1. PREFLIGHT_OK exists and is FRESH (older than HEAD => the code moved since it was measured)
#   2. the working tree is clean and committed (run_queue aborts on any HEAD change mid-flight, F6)
#   3. ruff + the architecture/guard tests are green
#
# Then hands off to the shared scheduler. It does NOT reimplement one — two schedulers caused the
# 2026-08-19 audit.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-24_architecture_bakeoff_dossier"
RES="$D/results"
CENV="conda run --no-capture-output -n views-hydranet-env"

die(){ echo "LAUNCH REFUSED: $*" >&2; exit 1; }

[ -f "$RES/PREFLIGHT_OK" ] || die "no PREFLIGHT_OK sentinel. Run tools/preflight.py first."

# SMOKE_OK is the load-bearing one. preflight profiles a synthetic single window; smoke runs the
# REAL pipeline for 2 lessons per architecture, which is the only measurement of actual training
# footprint and the only check that exercises the emit path, the scorer's view of the cube, and
# BatchNorm recalibration. Footprint does not scale with lesson count, so 2 lessons measures 300.
[ -f "$RES/SMOKE_OK" ] || die "no SMOKE_OK sentinel. Run tools/smoke.sh first — it is the real gate."
smoke_epoch=$(stat -c %Y "$RES/SMOKE_OK")

# A preflight measured before the last commit describes code that no longer exists. Silent staleness
# is how a gate becomes decorative: it passes because it is old, not because it is true.
# Freshness is measured against the last commit that touched CODE, not any commit. A docs-only
# commit (the pre-registration, a README) cannot invalidate a smoke run, and forcing a 2.5 h
# re-smoke for one would be the kind of friction that gets a gate switched off. Anything under
# `views_hydranet/`, `scripts/`, or this dossier's `tools/` counts as code.
head_epoch=$(cd "$HYD" && git log -1 --format=%ct -- views_hydranet scripts   reports/2026-08-24_architecture_bakeoff_dossier/tools)
pf_epoch=$(stat -c %Y "$RES/PREFLIGHT_OK")
if [ "$pf_epoch" -lt "$head_epoch" ]; then
  die "PREFLIGHT_OK predates HEAD — re-run the preflight."
fi
if [ "$smoke_epoch" -lt "$head_epoch" ]; then
  die "SMOKE_OK predates HEAD — the architectures changed since they were smoked. Re-run smoke.sh."
fi

[ -z "$(cd "$HYD" && git status --porcelain)" ] || die "working tree is dirty; run_queue aborts on HEAD drift (F6). Commit first."

$CENV ruff check "$HYD" >/dev/null 2>&1 || die "ruff check is red."
$CENV python -m pytest -q "$HYD/tests/architectures" "$HYD/tests/test_arm_identity_check.py" \
  "$HYD/tests/test_arm_postflight.py" >/dev/null 2>&1 || die "the guard tests are red."

echo "preflight OK, tree clean, guards green — launching."
mkdir -p "$RES"
cd /home/simon/Documents/scripts/views_platform/views-models
# Candidate order is pre-registered (04_roadmap): the zero-parameter arm first, so a harness fault
# surfaces on the cheapest arm; capacity-adders last, being slowest and likeliest to hit memory.
exec setsid nohup env \
  RES_DIR="$RES" \
  VERIFIER="$D/tools/verify_bakeoff.py" \
  ARM_MODULE=make_arch_arm \
  ARM_TOOLS="$D/tools" \
  bash "$HYD/reports/2026-08-18_lesson_curve_dossier/tools/run_queue.sh" \
  300:42:0.0::AntiAliasedPool 300:43:0.0::AntiAliasedPool \
  300:42:0.0::WideMemory     300:43:0.0::WideMemory \
  300:42:0.0::DynamicTopSkip 300:43:0.0::DynamicTopSkip \
  300:42:0.0::FiLMSkip       300:43:0.0::FiLMSkip \
  300:42:0.0::ShallowPool    300:43:0.0::ShallowPool \
  300:42:0.0::DualStream     300:43:0.0::DualStream \
  > "$RES/launcher.log" 2>&1 < /dev/null &
