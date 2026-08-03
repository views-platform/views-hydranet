#!/usr/bin/env bash
# Score the datafactory white_ranger clone (light_strider, ConflictologyModel, africa_me_legacy) on
# the v2 truth — FRESH end-to-end (no stale data/artifacts), for the Tier-B "beats white_ranger" leg.
#   move stale raw+artifacts aside -> fresh pull+train -> eval (lodestar cube) -> score vs v2 truth -> save.
set -uo pipefail
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
LS=$MODELS/light_strider
HN=/home/simon/Documents/scripts/views_platform/views-hydranet
DOSS="$HN/reports/2026-07-28_datafactory_migration_dossier"
V2TRUTH="$DOSS/tools/v2_truth/calibration_datafactory_df.parquet"
LODE="$HN/reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py"
JT=/home/simon/.claude/jobs/318eac15/tmp
STALE="$JT/lightstrider_stale_$(date +%s)"
LOG="$JT/wr_v2.log"; RES="$DOSS/results/v2_baseline"
: > "$LOG"; log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
mkdir -p "$RES" "$STALE"

[ -f "$V2TRUTH" ] || { log "V2 TRUTH MISSING — ABORT"; exit 4; }

# ---- 1) move STALE data + artifacts aside (recoverable; nothing stale is relied on) ----
log "moving stale light_strider raw data + artifacts aside -> $STALE"
mkdir -p "$STALE/raw" "$STALE/artifacts" "$STALE/generated"
mv "$LS"/data/raw/*.parquet "$LS"/data/raw/*data_fetch_log* "$STALE/raw/" 2>/dev/null || true
mv "$LS"/artifacts/*.pkl "$STALE/artifacts/" 2>/dev/null || true
mv "$LS"/data/generated/predictions_* "$STALE/generated/" 2>/dev/null || true
log "stale raw remaining: $(ls "$LS"/data/raw/ 2>/dev/null | grep -c parquet) parquet ; artifacts: $(ls "$LS"/artifacts/ 2>/dev/null | grep -c pkl) pkl"

run(){ ( cd "$LS" && export WANDB_MODE=offline WANDB_SILENT=true && \
  timeout 2400 conda run -n views_pipeline python main.py "$@" >> "$LOG" 2>&1 ); }

# ---- 2) FRESH train (forces a fresh datafactory pull; no stale cache present) ----
log "=== FRESH TRAIN light_strider (calibration) ==="
run --run_type calibration --train; rc=$?; log "train rc=$rc"
[ $rc -eq 0 ] || { log "TRAIN FAILED — see $LOG"; exit 6; }
log "raw after fetch:"; ls -la "$LS"/data/raw/ 2>/dev/null | grep -vE "^total|^d" | tee -a "$LOG"

# ---- 3) EVAL -> lodestar cube ----
before=$(ls -d "$LS"/data/generated/predictions_calibration_* 2>/dev/null | sort)
log "=== EVAL light_strider (calibration) ==="
run --run_type calibration --evaluate --saved; rc=$?; log "eval rc=$rc"
after=$(ls -d "$LS"/data/generated/predictions_calibration_* 2>/dev/null | sort)
PRED=$(comm -13 <(echo "$before") <(echo "$after") | tail -1)
[ $rc -eq 0 ] && [ -n "$PRED" ] || { log "EVAL FAILED (no pred dir) — see $LOG"; exit 7; }
log "predictions dir: $PRED"
log "origins emitted: $(ls -d "$PRED"/origin_* 2>/dev/null | wc -l) ; first/last: $(ls -d "$PRED"/origin_* 2>/dev/null | sed 's/.*origin_//' | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
log "target subdirs in origin: $(ls -d "$PRED"/origin_*/ 2>/dev/null | head -1 | xargs -I{} ls {} 2>/dev/null | tr '\n' ' ')"

# ---- 4) SCORE on v2 truth (lodestar intersects to common support) ----
log "=== SCORE white_ranger (light_strider) vs v2 truth ==="
# white_ranger is COUNT-ONLY (climatology resample) -> empty by-template (P(conflict)=frac samples>0)
conda run -n views-hydranet-env python "$LODE" "$V2TRUTH" \
  "white_ranger|$PRED|lr_{t}_best|" --targets=sb,ns,os > "$RES/score_white_ranger.txt" 2>&1
grep -q "Traceback\|Error" "$RES/score_white_ranger.txt" && log "SCORE had errors — see $RES/score_white_ranger.txt"
echo "wr_pred_dir=$PRED" > "$RES/white_ranger_provenance.txt"
log "SAVED -> $RES/score_white_ranger.txt (stale backup at $STALE)"
log "=== score head ==="; sed -n '1,20p' "$RES/score_white_ranger.txt" | tee -a "$LOG"
log "WR-V2-DONE"
