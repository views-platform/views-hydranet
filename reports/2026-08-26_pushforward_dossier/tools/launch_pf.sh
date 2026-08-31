#!/usr/bin/env bash
# launch_pf.sh — the ONLY sanctioned way to start the scored pushforward run (#289).
#
# Refuses unless, in order:
#   1. SMOKE_PF_OK exists and is FRESH (not older than the last commit touching what it measured)
#   2. the pre-registration is committed — no scored arm before 05_analysis_plan.md exists in git
#   3. the working tree is clean (run_queue aborts on HEAD drift mid-flight, F6)
#   4. ruff + the pushforward/guard tests are green
#
# The control-reuse gate (refullzero_fortytwo) already ran and PASSED under Amendment 1, so it is
# no longer in the queue; the queue would SKIP it anyway. Amendment 2: the tested weight is 0.01
# after w=0.1 came back VOID. Arm 1 is seed 42 and verify_pf.py halts the queue if F5 or F6 fires —
# that is the pre-committed stop rule, enforced by the harness rather than by my judgement.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-26_pushforward_dossier"
RES="$D/results"
CENV="conda run --no-capture-output -n views-hydranet-env"
die(){ echo "LAUNCH REFUSED: $*" >&2; exit 1; }

[ -f "$RES/SMOKE_PF_OK" ] || die "no SMOKE_PF_OK sentinel. Run tools/smoke_pf.sh — it is the gate."

# A sentinel older than the code it measured is decorative: it passes because it is old, not
# because it is true. Scope is deliberately narrow — widening it to make a refusal go away is the
# anti-pattern this exists to prevent. verify_pf.py and this launcher are excluded: both run
# strictly after the smoke and cannot change what it measured.
head_epoch=$(cd "$HYD" && git log -1 --format=%ct -- \
  views_hydranet scripts "$D/tools/make_pf_arm.py" "$D/tools/smoke_pf.sh" "$D/tools/train_time.py")
[ "$(stat -c %Y "$RES/SMOKE_PF_OK")" -ge "$head_epoch" ] \
  || die "SMOKE_PF_OK predates HEAD — the code moved since it was smoked. Re-run smoke_pf.sh."

(cd "$HYD" && git cat-file -e "HEAD:reports/2026-08-26_pushforward_dossier/05_analysis_plan.md" 2>/dev/null) \
  || die "05_analysis_plan.md is not committed. No scored arm runs before the plan is locked."

[ -z "$(cd "$HYD" && git status --porcelain)" ] || die "working tree is dirty; run_queue aborts on HEAD drift (F6). Commit first."

$CENV ruff check "$HYD" >/dev/null 2>&1 || die "ruff check is red."
$CENV python -m pytest -q "$HYD/tests/train" "$HYD/tests/test_arm_postflight.py" \
  "$HYD/tests/test_rollout_horizon_config.py" >/dev/null 2>&1 || die "the guard tests are red."

echo "smoke fresh, plan locked, tree clean, guards green — launching."
mkdir -p "$RES"
cd /home/simon/Documents/scripts/views_platform/views-models
exec setsid nohup env \
  RES_DIR="$RES" \
  VERIFIER="$D/tools/verify_pf.py" \
  ARM_MODULE=make_pf_arm \
  ARM_TOOLS="$D/tools" \
  bash "$HYD/reports/2026-08-18_lesson_curve_dossier/tools/run_queue.sh" \
  300:42:0.0::0.01 300:43:0.0::0.01 300:44:0.0::0.01 300:45:0.0::0.01 \
  > "$RES/launcher.log" 2>&1 < /dev/null &
