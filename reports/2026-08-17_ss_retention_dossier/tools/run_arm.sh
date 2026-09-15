#!/usr/bin/env bash
# run_arm.sh <arm_dir_name> [--gate] — train, emit, score, and (optionally) gate one arm.
#
# The gate runs INSIDE this script. Stage A finished at 15:38 and sat idle until 17:21 because the
# runner only trained and the gate was a separate manual step — the result existed and nothing looked
# at it. A stage that produces a decision must produce the decision, not the inputs to one.
set -uo pipefail

ARM="${1:?usage: run_arm.sh <arm_dir_name> [--gate]}"
GATE="${2:-}"
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-08-17_ss_retention_dossier"; RES="$D/results"
V2T="$HYD/reports/2026-07-29_v2_scoreboard_dossier/tools"
A="/home/simon/Documents/scripts/views_platform/views-models/models/$ARM"
CENV="conda run --no-capture-output -n views-hydranet-env"
HORIZONS=1,6,12,18,24,30,36

mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] $ARM: $*" | tee -a "$RES/run.log"; }

unset ALLOW_RE_REPORT
export WANDB_MODE=offline WANDB_SILENT=true

[ -d "$A" ] || { log "ABORT — arm dir missing"; exit 2; }
[ -s "$RES/score_${ARM}.csv" ] && { log "SKIP — already scored"; exit 0; }
n=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | wc -l)
[ "$n" -eq 0 ] || { log "ABORT — $n leftover cube(s); two arms' cubes would mix"; exit 3; }
free=$(df -BG "$A" | tail -1 | awk '{print $4}' | tr -d 'G')
[ "$free" -ge 20 ] || { log "ABORT — ${free}G free"; exit 4; }

log "train+emit start (HEAD $(cd "$HYD" && git rev-parse --short HEAD))"
t0=$(date +%s)
( cd "$A" && timeout -k 120 21600 $CENV python main.py -r calibration -t -e -sa ) \
    >> "$RES/${ARM}_run.log" 2>&1
rc=$?
log "train+emit rc=$rc in $(( ($(date +%s)-t0)/60 )) min"
[ $rc -eq 0 ] || { log "FAILED — see ${ARM}_run.log"; exit $rc; }

P=$(ls -d "$A"/data/generated/predictions_* 2>/dev/null | head -1)
[ -n "$P" ] || { log "ABORT — no cube produced"; exit 5; }
log "cube $(basename "$P") with $(ls -d "$P"/origin_* | wc -l) origins"

$CENV python "$V2T/score_v2_horizons.py" "$ARM|$P|lr_{t}_best|by_{t}_best" \
    --targets=sb --horizons=$HORIZONS --out="$RES/score_${ARM}.csv" >> "$RES/${ARM}_score.log" 2>&1
[ -s "$RES/score_${ARM}.csv" ] || { log "ABORT — scoring produced nothing"; exit 6; }

$CENV python "$HYD/scripts/ap_block_bootstrap.py" --pred-dir "$P" --target sb \
    --horizons 1,18 --n-boot 200 --seed 0 --out "$RES/ap_ci_${ARM}.json" \
    >> "$RES/${ARM}_score.log" 2>&1
log "scored + bootstrapped"

# record the arm's identity from its own config, never from the shell (the 2026-08-14 mismatch class)
$CENV python - "$A" "$ARM" "$P" >> "$RES/arm_${ARM}.json" 2>>"$RES/${ARM}_score.log" <<'PY'
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

if [ "$GATE" = "--gate" ]; then
  $CENV python - "$ARM" >> "$RES/run.log" 2>&1 <<'PY'
import csv, json, sys
sys.path.insert(0, "/home/simon/Documents/scripts/views_platform/views-hydranet")
from scripts.floor_gate import floor_gate
arm = sys.argv[1]
R = "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-08-17_ss_retention_dossier/results"
row = {r["h"]: r for r in csv.DictReader(open(f"{R}/score_{arm}.csv")) if r["target"] == "sb"}
ci = json.load(open(f"{R}/ap_ci_{arm}.json"))
clim = {r["h"]: float(r["AP"]) for r in csv.DictReader(open(
    "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
    "2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv"))
    if r["target"] == "sb" and r["model"] == "climatology"}
res = floor_gate(
    ap_control=float(row["18"]["AP"]), n_cells=int(row["18"]["N"]), n_event=int(row["18"]["n_event"]),
    horizon=18, target="sb", ap_control_h1=float(row["1"]["AP"]), ap_clim_h1=clim["1"],
    mde_ap=ci["18"]["mde"])
print(res.report())
print(f"retention: {float(row['18']['AP']) / float(row['1']['AP']):.3f}")
open(f"{R}/FLOORGATE_{arm}_{res['verdict']}", "w").write(res.report() + "\n")
PY
  log "GATE written: $(ls "$RES"/FLOORGATE_${ARM}_* 2>/dev/null | sed 's|.*/||')"
fi

touch "$RES/ARM_DONE_${ARM}"
log "ARM COMPLETE"
