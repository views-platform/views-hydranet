#!/usr/bin/env bash
# run_lesson_arm.sh <arm_dir_name> [--gate] — one point of the lesson curve, end to end.
#
# Copied from 2026-08-17_ss_retention_dossier/tools/run_arm.sh rather than parameterised, for two
# reasons: that sweep is parked ready-to-launch and must stay byte-untouched, and this script does a
# thing that one does not — it runs the ORACLE arm after the control, so a lesson point is never left
# half-measured. A stage that produces a decision must produce the decision, not the inputs to one.
#
# Two things differ beyond the oracle step:
#   * TRAIN_TIMEOUT scales with the arm's own total_lessons. run_arm.sh's flat 6 h would SIGKILL a
#     900-lesson run at ~60% complete, and there is no mid-training checkpoint (train_model.py:73
#     saves once, at the end) — a killed arm is lost entirely, not resumed.
#   * total_lessons is read from the arm's config, never from the shell. That is the 2026-08-14
#     mismatch class: a label and a config that disagreed, discovered after the run.
set -uo pipefail

ARM="${1:?usage: run_lesson_arm.sh <arm_dir_name> [--gate]}"
GATE="${2:-}"
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-18_lesson_curve_dossier"
# RES_DIR exists so the 2-lesson smoke can run the whole seam without writing a floor-limited
# arm into the real curve's results, where verify_curve.py would pick it up. Unset => the
# real directory, so the driver is unaffected.
RES="${RES_DIR:-$D/results}"
V2T="$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools"
RT="$HYD/reports/2026-08-16_feedback_realism_dossier/tools"
A="/home/simon/Documents/scripts/views_platform/views-models/models/$ARM"
CENV="conda run --no-capture-output -n views-hydranet-env"
HORIZONS=1,6,12,18,24,30,36

mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] $ARM: $*" | tee -a "$RES/run.log" >&2; }

unset ALLOW_RE_REPORT
export WANDB_MODE=offline WANDB_SILENT=true

[ -d "$A" ] || { log "ABORT — arm dir missing"; exit 2; }

# ---- the arm states its own length; the timeout follows from it -----------------------------
L=$($CENV python - "$A" <<'PY' 2>/dev/null
import ast, sys
from pathlib import Path
t = (Path(sys.argv[1]) / "configs/config_hyperparameters.py").read_text()
ast.parse(t); ns = {}; exec(compile(t, "cfg", "exec"), ns)
print(ns["get_hp_config"]()["total_lessons"])
PY
)
case "$L" in ''|*[!0-9]*) log "ABORT — could not read total_lessons from the arm's config"; exit 7;; esac
TRAIN_TIMEOUT=$(( 36 * L + 3600 )); [ "$TRAIN_TIMEOUT" -lt 21600 ] && TRAIN_TIMEOUT=21600
log "total_lessons=$L  TRAIN_TIMEOUT=${TRAIN_TIMEOUT}s ($(( TRAIN_TIMEOUT / 3600 ))h)"

# ---- control ---------------------------------------------------------------------------------
if [ -s "$RES/score_${ARM}.csv" ]; then
  log "SKIP control — already scored"
else
  n=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { log "ABORT — $n leftover cube(s); two arms' cubes would mix"; exit 3; }
  free=$(df -BG "$A" | tail -1 | awk '{print $4}' | tr -d 'G')
  [ "$free" -ge 25 ] || { log "ABORT — ${free}G free, the oracle step needs 25"; exit 4; }

  log "train+emit start (HEAD $(cd "$HYD" && git rev-parse --short HEAD))"
  t0=$(date +%s)
  ( cd "$A" && timeout -k 120 "$TRAIN_TIMEOUT" $CENV python main.py -r calibration -t -e -sa ) \
      >> "$RES/${ARM}_run.log" 2>&1
  rc=$?
  log "train+emit rc=$rc in $(( ($(date +%s)-t0)/60 )) min"
  # 124/137 from `timeout` mean the wall clock, not the model, ended the run — name it, because a
  # silent "failed" here reads as a modelling problem and sends the next person to the wrong place
  case $rc in
    0) ;;
    124|137) log "FAILED — TIMEOUT at ${TRAIN_TIMEOUT}s. The cost estimate was wrong, not the arm."; exit 8 ;;
    *) log "FAILED — see ${ARM}_run.log"; exit "$rc" ;;
  esac

  P=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | head -1)
  [ -n "$P" ] || { log "ABORT — no cube produced"; exit 5; }
  log "cube $(basename "$P") with $(ls -d "$P"/origin_* | wc -l) origins"

  $CENV python "$V2T/score_v2_horizons.py" "$ARM|$P|lr_{t}_best|by_{t}_best" \
      --targets=sb --horizons=$HORIZONS --out="$RES/score_${ARM}.csv" >> "$RES/${ARM}_score.log" 2>&1
  [ -s "$RES/score_${ARM}.csv" ] || { log "ABORT — scoring produced nothing"; exit 6; }

  $CENV python "$HYD/scripts/ap_block_bootstrap.py" --pred-dir "$P" --target sb \
      --horizons 1,18 --n-boot 200 --seed 0 --out "$RES/ap_ci_${ARM}.json" \
      >> "$RES/${ARM}_score.log" 2>&1
  # the co-primary endpoint gets its own paired interval — retention is a ratio on shared origins,
  # so propagating the two per-horizon intervals is not a valid construction
  $CENV python "$HYD/scripts/ap_block_bootstrap.py" --pred-dir "$P" --target sb \
      --ratio --num-h 18 --den-h 1 --n-boot 200 --seed 0 --out "$RES/ret_ci_${ARM}.json" \
      >> "$RES/${ARM}_score.log" 2>&1
  log "scored + bootstrapped (AP and retention)"

  # record the arm's identity from its own config, never from the shell
  $CENV python - "$A" "$ARM" "$P" > "$RES/arm_${ARM}.json" 2>>"$RES/${ARM}_score.log" <<'PY'
import ast, hashlib, json, subprocess, sys
from pathlib import Path
arm_dir, label, pred = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def res(p, fn):
    t = Path(p).read_text(); ast.parse(t); ns = {}; exec(compile(t, str(p), "exec"), ns); return ns[fn]()
hp = res(arm_dir / "configs/config_hyperparameters.py", "get_hp_config")
art = sorted((arm_dir / "artifacts").glob("*.pt"))[-1]
import torch
w = torch.load(art, map_location="cpu", weights_only=False)
sd = w.get("model_state_dict", w) if isinstance(w, dict) else {}
h = hashlib.sha256()
for k in sorted(sd):
    v = sd[k]
    if hasattr(v, "detach"):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({
    "label": label, "artifact": art.name, "pred_dir": Path(pred).name,
    "weight_sha256": h.hexdigest(),        # NOT the .pt file sha — see the nondeterminism postmortem
    "total_lessons": hp["total_lessons"], "ss_epsilon_max": hp["ss_epsilon_max"],
    "ss_feedback": hp.get("ss_feedback"), "torch_seed": hp["torch_seed"],
    "output_distribution": hp["output_distribution"], "forecast_composition": hp["forecast_composition"],
    "head": subprocess.run(["git","-C","/home/simon/Documents/scripts/views_platform/views-hydranet",
                            "rev-parse","HEAD"], capture_output=True, text=True).stdout.strip(),
}, indent=2))
PY
  rm -rf "$P"
  log "cube deleted"
fi

# ---- oracle: the same artifact, fed the real field ---------------------------------------------
# Inference-only. Without it a lesson point has a free-running number and no ceiling, so the
# ceiling/retention decomposition — the whole reason for two arms — cannot be computed for it.
if [ -s "$RES/score_${ARM}_use_real.csv" ]; then
  log "SKIP oracle — already scored"
else
  ART=$($CENV python - "$RES/arm_${ARM}.json" <<'PY' 2>/dev/null
import json, sys
print(json.load(open(sys.argv[1]))["artifact"])
PY
)
  if [ -z "$ART" ]; then
    log "ORACLE SKIPPED — no arm_${ARM}.json to name the artifact"
  else
    n=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | wc -l)
    if [ "$n" -ne 0 ]; then
      log "ORACLE SKIPPED — $n leftover cube(s)"
    else
      log "oracle (use_real) on $ART"
      t0=$(date +%s)
      timeout -k 120 5400 $CENV python "$RT/run_realism_arms.py" --model "$ARM" --artifact "$ART" \
          --arms use_real --tag oracle --out "$RES" >> "$RES/${ARM}_oracle.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ] && [ -s "$RES/score_${ARM}_use_real.csv" ]; then
        log "oracle OK in $(( ($(date +%s)-t0)/60 )) min"
      else
        log "oracle FAILED rc=$rc — the point has a control but no ceiling"
      fi
      for d in "$A"/data/generated/predictions_*; do
        [ -e "$d" ] && { log "  cleaning leftover $(basename "$d")"; rm -rf "$d"; }
      done
    fi
  fi
fi

# ---- gate ---------------------------------------------------------------------------------------
if [ "$GATE" = "--gate" ]; then
  $CENV python - "$ARM" "$RES" >> "$RES/run.log" 2>&1 <<'PY'
import csv, json, sys
HYD = "/home/simon/Documents/scripts/views_platform/views-hydranet"
sys.path.insert(0, HYD)
from scripts.floor_gate import floor_gate
arm, R = sys.argv[1], sys.argv[2]
row = {r["h"]: r for r in csv.DictReader(open(f"{R}/score_{arm}.csv")) if r["target"] == "sb"}
ci = json.load(open(f"{R}/ap_ci_{arm}.json"))
clim = {r["h"]: float(r["AP"]) for r in csv.DictReader(open(
    f"{HYD}/reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv"))
    if r["target"] == "sb" and r["model"] == "climatology"}
res = floor_gate(
    ap_control=float(row["18"]["AP"]), n_cells=int(row["18"]["N"]), n_event=int(row["18"]["n_event"]),
    horizon=18, target="sb", ap_control_h1=float(row["1"]["AP"]), ap_clim_h1=clim["1"],
    mde_ap=ci["18"]["mde"])
print(res.report())
print(f"retention: {float(row['18']['AP']) / float(row['1']['AP']):.4f}")
open(f"{R}/FLOORGATE_{arm}_{res['verdict']}", "w").write(res.report() + "\n")
PY
  log "GATE written: $(ls "$RES"/FLOORGATE_${ARM}_* 2>/dev/null | sed 's|.*/||')"
fi

touch "$RES/ARM_DONE_${ARM}"
log "ARM COMPLETE"
