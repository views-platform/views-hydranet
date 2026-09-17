#!/usr/bin/env bash
# PyPI smoke (2026-09-15): the consumer path end to end. A FRESH env built the way each model's
# run.sh builds it — `pip install -r requirements.txt`, which pulls views-hydranet~=0.1.0 from
# PyPI — then eval-only on the existing validation artifact, saved data, four models.
# 2026-09-16 09:50 relaunch ON THE GPU: the fresh env had pulled torch 2.14+cu130, which the 535 driver
# cannot run, so the first pass trained on CPU (violet_visitor: 6 h 46 m; kept as log_violet_visitor_CPU.txt).
# torch 2.6.0+cu124 installed into the env by hand before this relaunch.
# TRAINING + evaluating (user asked for a real rerun, 2026-09-15 evening). A new artifact gets a
# new timestamp; the 2026-09-07/08 roster artifacts are not touched. ~3 h per model on the RTX 4070.
set +u
OUT="$(cd "$(dirname "$0")" && pwd)"
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
# NOT views-models/envs/views-hydranet: on this machine that is a SYMLINK to the dev env
# (~/anaconda3/envs/views-hydranet-env, editable checkout). The first attempt of this script went
# through it, pip-upgraded the dev env, and the wheel guard below caught the checkout import.
ENV=/tmp/claude-1000/pypi-smoke-env
eval "$(conda shell.bash hook)"
log() { echo "$(date +%F' '%T)  $*" | tee -a "$OUT/SMOKE.log"; }

log "=== env: creating $ENV from PyPI (the model run.sh path) ==="
[ -d "$ENV" ] || conda create --prefix "$ENV" python=3.11 -y >> "$OUT/env.log" 2>&1 || { log "ENV CREATE FAILED"; exit 1; }
conda activate "$ENV"
"$ENV/bin/python" -m pip install -r "$MODELS/violet_visitor/requirements.txt" >> "$OUT/env.log" 2>&1 || { log "PIP INSTALL FAILED (see env.log)"; exit 1; }
"$ENV/bin/python" - <<'PY' 2>&1 | tee -a "$OUT/SMOKE.log"
import importlib.metadata as md, views_hydranet
print(f"  views-hydranet {md.version('views-hydranet')} from {views_hydranet.__file__}")
from views_hydranet import HydranetManager
print("  from views_hydranet import HydranetManager: OK")
PY
case "$("$ENV/bin/python" -c 'import views_hydranet,sys; print("site-packages" in views_hydranet.__file__)')" in
  True) log "env: PyPI wheel confirmed (site-packages)";;
  *) log "env: NOT the wheel — aborting"; exit 1;;
esac

for m in bold_comet purple_alien heavy_freighter violet_visitor; do
  MD="$MODELS/$m"
  log "=== $m  train + evaluate  (artifacts before: $(ls "$MD"/artifacts/validation_model_*.pt | wc -l)) ==="
  ( cd "$MD" && "$ENV/bin/python" main.py --run_type validation --train --evaluate --saved ) \
      > "$OUT/log_$m.txt" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    log "  $m  OK  $(grep -m1 -oE 'CLAMPED[^\"]{0,60}|evolves freely' "$OUT/log_$m.txt")"
    touch "$OUT/done_$m"
  else
    log "  $m  FAILED rc=$rc  last line: $(tail -1 "$OUT/log_$m.txt" | cut -c1-160)"
    touch "$OUT/failed_$m"
  fi
done
log "=== SMOKE COMPLETE: $(ls "$OUT"/done_* 2>/dev/null | wc -l) ok, $(ls "$OUT"/failed_* 2>/dev/null | wc -l) failed ==="
touch "$OUT/SMOKE_DONE"
