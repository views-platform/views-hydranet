#!/usr/bin/env bash
# launch_exp02.sh — the ONLY sanctioned way to start EXP-02 (candidate 7, four arms).
#
# A sibling of the EXP-01 dossier's `launch_bakeoff.sh`, which must not be edited: it carries that
# experiment's twelve specs and writes that experiment's sentinels. PREREG.md section 10b.
#
# Refuses unless, in order:
#   1. PREFLIGHT_OK and SMOKE_OK both exist -- the SAME gate set as EXP-01's launcher, deliberately
#   2. SMOKE_OK is FRESH (older than the last commit touching what it measured => the code moved
#      since it was measured)
#   3. the working tree is clean and committed (run_queue aborts on any HEAD change mid-flight, F6)
#   4. ruff + the architecture/guard tests are green
#
# Hands off to the SHARED scheduler. It does not reimplement one — two schedulers caused the
# 2026-08-19 audit.
#
# ⛔ NO DEVIATION from the parent's gate set. An earlier version of this file dropped
# PREFLIGHT_OK; see the comment at the check itself for why that was wrong.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-10-30_upsample_arm_dossier"
BAKEOFF="$HYD/reports/2026-08-24_architecture_bakeoff_dossier"
RES="$D/results"
CENV="conda run --no-capture-output -n views-hydranet-env"

die(){ echo "LAUNCH REFUSED: $*" >&2; exit 1; }

# ⛔ RESTORED 2026-10-30. This launcher previously required SMOKE_OK but NOT PREFLIGHT_OK, with a
# paragraph arguing preflight is the weaker check. ⛔ That was narrowing a gate set to make a
# refusal go away — the exact anti-pattern `launch_bakeoff.sh` warns about in its own comments —
# and the copy had therefore DRIFTED from its parent on day one, which is how two copies of a gate
# diverge while the weaker one still passes. The gate set here is now IDENTICAL to EXP-01's.
[ -f "$RES/PREFLIGHT_OK" ] || die "no PREFLIGHT_OK sentinel. Run tools/preflight_exp02.py first."
[ -f "$RES/SMOKE_OK" ] || die "no SMOKE_OK sentinel. Run tools/smoke_exp02.sh — it is the real gate."
smoke_epoch=$(stat -c %Y "$RES/SMOKE_OK")

# ⛔⛔ WHICH ARM RUNS IS READ FROM G-F'S SENTINEL, NOT TYPED HERE. The specs below carried
# `UpsampleConv2` as a literal until 2026-10-30 — one of four hand-typed copies of a decision
# `gf_phase_signature.py` makes and, at the time, only printed. A /falsify guard audit changed
# that one word and four arms trained, scored, and were NEVER identity-checked, because
# `verify_exp02.py`'s own arm list was an independent fifth copy. Gate F1 (PREREG.md:226) read as
# HAVING RUN when it had not. Re-running G-F is now the only way to change the arm.
[ -f "$RES/SELECTED_ARM" ] || die "no SELECTED_ARM sentinel. Run gf_phase_signature.py — PREREG \
section 2 says the arm is chosen by those numbers, and this launcher will not guess it."
read -r CTL ARM < <(python3 -c "
import json
d = json.load(open('$RES/SELECTED_ARM'))
print(d['control'], d['selected'])" 2>/dev/null) || true
[ -n "${CTL:-}" ] && [ -n "${ARM:-}" ] || die "SELECTED_ARM is unreadable or lacks \
'control'/'selected'. Re-run gf_phase_signature.py rather than editing the sentinel by hand."
sel_epoch=$(stat -c %Y "$RES/SELECTED_ARM")

# A sentinel measured before the last commit describes code that no longer exists. Silent staleness
# is how a gate becomes decorative: it passes because it is old, not because it is true. Scope is
# the model code, the guards and the arm builder — the things that could change WHAT SMOKE
# MEASURED. Deliberately excluded: this launcher and the verifier, which run strictly after; and
# docs/results, which are not executable. Widening this scope to make a refusal go away is exactly
# the anti-pattern the gate exists to prevent. If in doubt, re-run smoke.
head_epoch=$(cd "$HYD" && git log -1 --format=%ct -- \
  views_hydranet tests "$BAKEOFF/tools/make_arch_arm.py" \
  "$D/tools/smoke_exp02.sh" 2>/dev/null)
[ -n "$head_epoch" ] || die "cannot read the last commit touching the measured surface."
[ "$smoke_epoch" -ge "$head_epoch" ] || die \
  "SMOKE_OK is STALE: measured $(date -d @"$smoke_epoch" '+%F %T'), but the measured surface last \
changed $(date -d @"$head_epoch" '+%F %T'). Re-run tools/smoke_exp02.sh."
# The SELECTION is a measurement too, so the same staleness DISCIPLINE applies — but over its own
# surface, which is not SMOKE_OK's. The test is "would a change here move G-F's numbers?":
# `views_hydranet` and `tests` yes; `gf_phase_signature.py` itself yes. `smoke_exp02.sh` and
# `make_arch_arm.py` NO — they do not enter the gap computation at all, and SMOKE_OK's scope
# includes them because they determine what SMOKE measures.
#
# ⛔ THIS IS NOT THE SCOPE BEING WIDENED TO DODGE A REFUSAL, which is what the comment above warns
# against — it is NARROWER than SMOKE_OK's on the two files that cannot affect a gap, and it ADDS
# `gf_phase_signature.py`, which SMOKE_OK's scope omits and which is the one file that most
# obviously invalidates a selection. Re-run G-F and the refusal clears honestly.
sel_head_epoch=$(cd "$HYD" && git log -1 --format=%ct -- \
  views_hydranet tests "$D/gf_phase_signature.py" 2>/dev/null)
[ -n "$sel_head_epoch" ] || die "cannot read the last commit touching G-F's measured surface."
[ "$sel_epoch" -ge "$sel_head_epoch" ] || die \
  "SELECTED_ARM is STALE: written $(date -d @"$sel_epoch" '+%F %T'), but G-F's measured surface \
last changed $(date -d @"$sel_head_epoch" '+%F %T'). Re-run gf_phase_signature.py."

[ -z "$(cd "$HYD" && git status --porcelain)" ] || die \
  "working tree is dirty; run_queue aborts on HEAD drift (F6). Commit first."

$CENV ruff check "$HYD" >/dev/null 2>&1 || die "ruff check is red."
$CENV python -m pytest -q "$HYD/tests/architectures" "$HYD/tests/test_arm_identity_check.py" \
  "$HYD/tests/test_arm_postflight.py" >/dev/null 2>&1 || die "the guard tests are red."

echo "smoke fresh, selection fresh, tree clean, guards green — launching EXP-02 (4 arms)."
echo "  control: $CTL   arm: $ARM   (both read from SELECTED_ARM, neither typed here)"
mkdir -p "$RES"
cd /home/simon/Documents/scripts/views_platform/views-models
# Order is pre-registered (PREREG.md section 10d): the CONTROL first, both seeds, then the arm.
# EXP-01's rule was "the cheapest arm first, so a harness fault surfaces cheaply"; here there is a
# second reason — a control that fails to reproduce its own historical AP tells us the code moved
# since August, and that is worth knowing before two more runs. P-27.
exec setsid nohup env \
  RES_DIR="$RES" \
  VERIFIER="$D/tools/verify_exp02.py" \
  ARM_MODULE=make_arch_arm \
  ARM_TOOLS="$BAKEOFF/tools" \
  bash "$HYD/reports/2026-08-18_lesson_curve_dossier/tools/run_queue.sh" \
  "300:42:0.0::$CTL" "300:43:0.0::$CTL" \
  "300:42:0.0::$ARM" "300:43:0.0::$ARM" \
  > "$RES/launcher.log" 2>&1 < /dev/null &
