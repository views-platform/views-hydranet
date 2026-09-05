#!/usr/bin/env bash
# run_overnight.sh — Epic #311, S4 + S5 + S6 chained, unattended.
#
# One script because nobody is awake between the stages. Every gate is a hard exit with a named
# reason, so the morning's question is always "which gate fired and what was the number", never
# "what did it decide to do instead".
#
# GATE ORDER, and each one's provenance:
#   1. GPU wait      — the launcher must not start a silent CPU grind (C-163).
#   2. SMOKE         — 2 lessons on the noise arm's real config; it must train and write an artifact.
#   3. POTENCY       — C-324 (Tier 1) on the arm's OWN config, at a TRAINED checkpoint (C-325).
#   4. FLOOR (FG-A)  — C-299, on the CONTROL, BEFORE the treatment arm runs, at zero extra GPU.
#   5. WEIGHT-HASH   — identical weights mean the treatment was inert; VOID before any score is read.
#   6. VERDICT       — the rule locked in 47d66af, applied in code.
#
# kill_tree, not `kill -TERM $APID`: the latter signals the subshell and leaves the training process
# alive — measured in C-326, where an arm ran 7 lessons past a declared cap.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS="$HYD/../views-models/models"
D="$HYD/reports/2026-09-04_input_noise_dossier"; RES="$D/results/s5"
T="$D/tools"
CENV="conda run --no-capture-output -n views-hydranet-env"
export WANDB_MODE=offline WANDB_SILENT=true      # C-163: wandb.finish() has DNS-hung 38 min here
FLOOR=fullzero_fortytwo
CTRL=noisecontrol_fortytwo
NOISE=noisedropout_fortytwo
SMOKE=noisesmoke_fortytwo
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-16200}            # 4.5 h vs ~110 min measured
EMIT_TIMEOUT=${EMIT_TIMEOUT:-5400}
GPU_WAIT=${GPU_WAIT:-21600}                      # up to 6 h for the GPU to free
STALL_MAX=${STALL_MAX:-1800}

mkdir -p "$RES"; START=$(date +%s)
log(){ echo "[$(date '+%F %T')] $*" >> "$RES/run.log"; return 0; }
phase(){ echo "$(date '+%F %T') $*" > "$RES/PHASE"; log "PHASE: $*"; }
abort(){ echo "$(date '+%F %T') $*" > "$RES/ABORTED"; log "ABORT: $*"; exit "${2:-9}"; }
kill_tree(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
             kill -TERM "$p" 2>/dev/null; }

exec 9>"$RES/.lock"; flock -n 9 || { echo "locked"; exit 11; }
( cd "$HYD" && git rev-parse HEAD ) > "$RES/repo.head"
( while :; do date '+%s %F %T' > "$RES/HEARTBEAT"; sleep 30; done ) & HB=$!
trap 'kill $HB 2>/dev/null; echo "ended=$(date "+%F %T") elapsed_min=$(( ($(date +%s)-START)/60 ))" > "$RES/RUN_COMPLETE"; log "=== RUN_COMPLETE ==="' EXIT
log "=== S4+S5+S6 chain: control=$CTRL noise=$NOISE ==="

# ---- 1. wait for the GPU ---------------------------------------------------
phase "WAIT FOR GPU"
t0=$(date +%s)
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) u=999999;; esac
  [ "$u" -lt 3000 ] && { log "GPU free (${u} MiB)"; break; }
  [ $(( $(date +%s) - t0 )) -gt "$GPU_WAIT" ] && abort "GPU still busy (${u} MiB) after ${GPU_WAIT}s" 10
  sleep 60
done
$CENV python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
  || abort "no CUDA — refusing a silent CPU grind (C-163)" 10
f=$(df -BG "$MODELS" | tail -1 | awk '{print $4}' | tr -d 'G')
case "${f:-}" in ''|*[!0-9]*) f=0;; esac
[ "$f" -ge 30 ] || abort "only ${f}G free on the models volume" 10
[ "$(cd "$HYD" && git rev-parse HEAD)" = "$(cat "$RES/repo.head")" ] \
  || abort "repo HEAD moved before launch — the arms would not be comparable" 10

# ---- 2. build the arms -----------------------------------------------------
phase "BUILD ARMS"
for spec in "control:$CTRL:" "noise:$NOISE:" "noise:$SMOKE:--lessons 2"; do
  IFS=: read -r which label extra <<< "$spec"
  [ -d "$MODELS/$label" ] && { log "$label exists — reusing"; continue; }
  # shellcheck disable=SC2086
  $CENV python "$T/make_noise_arm.py" "$which" --label "$label" $extra >> "$RES/run.log" 2>&1 \
    || abort "arm build failed for $label" 12
  log "built $label"
done

# ---- 3. SMOKE --------------------------------------------------------------
phase "SMOKE (2 lessons, $SMOKE)"
if ! ls "$MODELS/$SMOKE"/artifacts/*.pt >/dev/null 2>&1; then
  # -t train, -sa use the locally cached parquet. NOT -s: that is --sweep, and a sweep
  # deliberately does not save artifacts, so every arm would train and write nothing.
  ( cd "$MODELS/$SMOKE" && timeout -k 120 3600 $CENV python main.py -r calibration -t -sa ) \
      >> "$RES/smoke.log" 2>&1
  ls "$MODELS/$SMOKE"/artifacts/*.pt >/dev/null 2>&1 \
    || abort "SMOKE failed — the noise arm's config does not train. See results/s5/smoke.log" 13
fi
log "SMOKE OK"

# ---- 4. POTENCY, on the arm's own config at a TRAINED checkpoint ------------
phase "POTENCY GATE (C-324 + C-325)"
FLOOR_ART=$(ls -t "$MODELS/$FLOOR"/artifacts/*.pt 2>/dev/null | head -1)
[ -n "$FLOOR_ART" ] || abort "no trained floor artifact to gate against" 14
( cd "$MODELS/$NOISE" && $CENV python "$T/preflight_input_noise.py" \
    --model-dir "$MODELS/$NOISE" --artifact-path "$FLOOR_ART" ) >> "$RES/potency.log" 2>&1 \
  || abort "POTENCY FAILED — the knob is inert on the arm's own config at a trained checkpoint. No GPU spent on a run whose null would be a fact about the harness (C-324)." 14
grep -h "^POTENT" "$RES/potency.log" | tail -1 >> "$RES/run.log"
log "POTENCY PASS"

# ---- helpers ---------------------------------------------------------------
train_arm(){ local ARM=$1
  ls "$MODELS/$ARM"/artifacts/*.pt >/dev/null 2>&1 && { log "$ARM already trained — skipping"; return 0; }
  phase "TRAIN $ARM (elapsed $(( ($(date +%s)-START)/60 )) min)"
  local LOG="$RES/train_${ARM}.log"; : > "$LOG"
  ( cd "$MODELS/$ARM" && timeout -k 120 "$TRAIN_TIMEOUT" $CENV python main.py -r calibration -t -sa ) \
      >> "$LOG" 2>&1 & local P=$!
  ( while kill -0 $P 2>/dev/null; do
      s1=$(stat -c %s "$LOG" 2>/dev/null || echo 0); sleep "$STALL_MAX"
      kill -0 $P 2>/dev/null || break
      s2=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
      [ "$s1" = "$s2" ] && { echo "$(date '+%F %T') STALLED $ARM" >> "$RES/ANOMALIES.txt"; kill_tree $P; break; }
    done ) & local W=$!
  wait $P; local rc=$?; kill_tree $W 2>/dev/null; wait $W 2>/dev/null
  kill_tree $P; sleep 3
  ls "$MODELS/$ARM"/artifacts/*.pt >/dev/null 2>&1 || return 1
  log "$ARM trained (rc=$rc)"; return 0
}
score_arm(){ local ARM=$1
  [ -s "$RES/score_${ARM}_s5.csv" ] && { log "$ARM already scored"; return 0; }
  phase "EMIT+SCORE $ARM"
  local ART; ART=$(ls -t "$MODELS/$ARM"/artifacts/*.pt | head -1)
  rm -rf "$MODELS/$ARM"/data/generated/predictions_* 2>/dev/null
  timeout -k 120 "$EMIT_TIMEOUT" $CENV python \
    "$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py" \
    --model "$ARM" --artifact "$(basename "$ART")" --arms identity --tag s5 --out "$RES" \
    >> "$RES/emit_${ARM}.log" 2>&1
  local rc=$?
  rm -rf "$MODELS/$ARM"/data/generated/predictions_* 2>/dev/null
  [ -s "$RES/score_${ARM}_s5.csv" ] || return 1
  log "$ARM scored (rc=$rc)"; return 0
}

# ---- 5. CONTROL first, then its floor gate, THEN the treatment -------------
train_arm "$CTRL" || abort "control arm did not train — BRANCH 0: the screen is VOID, not negative" 20
score_arm "$CTRL" || abort "control arm produced no score — BRANCH 0: VOID, not negative" 20

phase "FLOOR GATE on the control (C-299)"
$CENV python "$T/check_floor.py" --score-csv "$RES/score_${CTRL}_s5.csv" >> "$RES/floor_gate.log" 2>&1
fg=$?
cat "$RES/floor_gate.log" >> "$RES/run.log"
[ $fg -eq 1 ] && abort "FG-A FAILED on the control — it does not rank above chance. The vehicle cannot carry this experiment; VOID before the treatment arm runs (C-299)." 21
[ $fg -eq 0 ] || abort "floor gate could not be evaluated (rc=$fg)" 21
log "FLOOR GATE: FG-A PASS"

train_arm "$NOISE" || abort "noise arm did not train — BRANCH 0: the screen is VOID, not negative" 22

# ---- 6. weight hash, before any score is read ------------------------------
phase "WEIGHT-HASH CHECK"
$CENV python - "$CTRL" "$NOISE" >> "$RES/run.log" 2>&1 <<'PYEOF'
import glob, hashlib, sys, torch
M = "/home/simon/Documents/scripts/views_platform/views-models/models"
h = {}
for m in sys.argv[1:]:
    f = sorted(glob.glob(f"{M}/{m}/artifacts/*.pt"))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    d = hashlib.sha256()
    for k in sorted(sd):
        d.update(sd[k].detach().cpu().numpy().tobytes())
    h[m] = d.hexdigest()
    print(f"weight-hash {m}: {h[m][:24]}")
if len(set(h.values())) == 1:
    print("VOID: identical trained weights — the treatment was inert (C-324). Do NOT read scores.")
    sys.exit(3)
print("weights differ — the arms are genuinely distinct")
PYEOF
[ $? -eq 0 ] || abort "WEIGHT-HASH: the arms trained to IDENTICAL weights. The treatment was inert (C-324); the screen is VOID and the scores must not be read." 23
log "weight-hash PASS"

score_arm "$NOISE" || abort "noise arm produced no score — BRANCH 0: VOID, not negative" 24

# ---- 7. verdict ------------------------------------------------------------
phase "VERDICT"
$CENV python "$T/read_verdict.py" "$CTRL" "$NOISE" > "$RES/VERDICT.txt" 2>&1
cat "$RES/VERDICT.txt" >> "$RES/run.log"
{ echo "# Epic #311 / S5 — run summary"; echo
  echo "- finished: $(date '+%F %T')"; echo "- elapsed: $(( ($(date +%s)-START)/60 )) min"; echo
  echo '```'; cat "$RES/VERDICT.txt"; echo '```'; echo
  echo "## Floor gate (control)"; echo '```'; cat "$RES/floor_gate.log"; echo '```'; } \
  > "$RES/SUMMARY.md"
log "=== DONE ==="
touch "$RES/S5_DONE"
