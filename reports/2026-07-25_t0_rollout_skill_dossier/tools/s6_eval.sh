#!/usr/bin/env bash
# S6 (#199): bloom-verification eval. 18 arms = {gated_NB,th_gated_NB,ZINB}×{42,43,44}×{mean,sample}.
# Inference-only (no retrain). Stealth-safe (verify+trap-restore floor md5). Resumable (skip a scored
# arm). Score-after-each (the in-place-overwrite scar). Delete each cube after scoring (disk-safe),
# except KEEP the one passed as $2=keep (for the determinism spot-check).
#
# Usage:
#   s6_eval.sh <label|all> [keep]
#     label = <arm>_<seed>_<mode> (e.g. gated_NB_42_mean) to run ONE arm; 'all' runs the 18 (resumable).
set -uo pipefail
VV=/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor
CFG="$VV/configs/config_hyperparameters.py"
GEN="$VV/data/generated"
TRUTH="$VV/data/raw/calibration_viewser_df.parquet"
JT=/home/simon/.claude/jobs/318eac15/tmp
RES=/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-07-25_t0_rollout_skill_dossier/results/bloomverify
LOG="$JT/s6_eval.log"; BACKUP="$JT/floor.backup.s6.py"; MAN="$JT/s6_manifest.txt"
EXPECT=6c28bdb1390fc413d43b2d74d87251f8
WHICH="${1:-all}"; KEEP="${2:-}"
mkdir -p "$RES"
touch "$LOG" "$MAN"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

GOT=$(md5sum "$CFG" | awk '{print $1}')
[ "$GOT" = "$EXPECT" ] || { log "MD5 MISMATCH floor ($GOT != $EXPECT) — ABORT"; exit 3; }
log "floor md5 verified"; cp "$CFG" "$BACKUP"
trap 'cp "$BACKUP" "$CFG"; log "RESTORED floor ($(md5sum "$CFG" | awk "{print \$1}"))"' EXIT

# seed -> artifact timestamp, per family
nb_ts()   { case "$1" in 42) echo 20260727_033447;; 43) echo 20260727_035759;; 44) echo 20260727_042258;; esac; }
zinb_ts() { case "$1" in 42) echo 20260727_061041;; 43) echo 20260727_065917;; 44) echo 20260727_073712;; esac; }

run_arm () {
  local label="$1"                    # e.g. gated_NB_42_mean
  local mode="${label##*_}"           # mean | sample
  local rest="${label%_*}"            # gated_NB_42
  local seed="${rest##*_}"            # 42
  local arm="${rest%_*}"              # gated_NB | th_gated_NB | ZINB
  local out_csv="$RES/bloomverify_${label}.csv"
  if [ -f "$out_csv" ]; then log "SKIP $label (already scored)"; return 0; fi

  # disk preflight (the disk-full scar)
  local freeg; freeg=$(df -BG "$VV" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$freeg" -lt 15 ] && { log "DISK LOW (${freeg}G) — ABORT $label"; return 9; }

  local fam ts
  if [ "$arm" = "ZINB" ]; then fam=zinb; ts=$(zinb_ts "$seed"); else fam=nb; ts=$(nb_ts "$seed"); fi
  local art="calibration_model_${ts}.pt"
  local cfgsrc="$JT/cfg_s6_${label}.py"
  [ -f "$cfgsrc" ] || { log "[$label] MISSING config $cfgsrc"; return 4; }

  cp "$cfgsrc" "$CFG"
  local before after newdir
  before=$(ls -d "$GEN"/predictions_calibration_* 2>/dev/null | sort)
  log "=== EVAL $label  (arm=$arm fam=$fam seed=$seed mode=$mode art=$ts) ==="
  ( cd "$VV" && export WANDB_MODE=offline WANDB_SILENT=true && \
    timeout 3600 conda run -n views-hydranet-env python main.py --run_type calibration \
      --evaluate --saved --artifact_name "$art" >> "$LOG" 2>&1 )
  local rc=$?
  after=$(ls -d "$GEN"/predictions_calibration_* 2>/dev/null | sort)
  newdir=$(comm -13 <(echo "$before") <(echo "$after") | tail -1)
  log "eval $label rc=$rc newdir=$newdir"
  [ $rc -eq 0 ] && [ -n "$newdir" ] || { log "[$label] EVAL FAILED rc=$rc — no scoring"; return 6; }

  # score-after-each (composition already baked in-model → no thresh spec)
  log "--- score $label ---"
  conda run -n views-hydranet-env python "$JT/s6_score_one.py" "$label" "$newdir" "$TRUTH" "$RES" >> "$LOG" 2>&1
  local src=$?
  [ $src -eq 0 ] && [ -f "$out_csv" ] || { log "[$label] SCORE FAILED src=$src"; return 7; }
  echo "$label $ts $newdir" >> "$MAN"
  log "SCORED $label -> $out_csv"

  # free the cube unless it's the keep arm (determinism spot-check)
  if [ "$label" = "$KEEP" ]; then
    log "KEEP $label cube at $newdir (determinism spot-check)"
  else
    rm -rf "$newdir"; log "freed cube $newdir"
  fi
}

if [ "$WHICH" = "all" ]; then
  for arm in gated_NB th_gated_NB ZINB; do
    for seed in 42 43 44; do
      for mode in mean sample; do run_arm "${arm}_${seed}_${mode}"; done
    done
  done
  log "S6-ALL-DONE"
  log "scored: $(ls "$RES"/bloomverify_*.csv 2>/dev/null | wc -l)/18"
else
  run_arm "$WHICH"
  log "S6-ONE-DONE ($WHICH)"
fi
