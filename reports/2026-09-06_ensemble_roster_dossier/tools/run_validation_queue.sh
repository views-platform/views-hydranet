#!/usr/bin/env bash
# The 9 validation runs: light_strider (K3 comparator) first, then the 8 roster models.
# 300 lessons each, validation partition, sequential. ~30 h.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
MODELS=/home/simon/Documents/scripts/views_platform/views-models/models
OUT="$HYD/reports/2026-09-06_ensemble_roster_dossier/results/validation"
PY=/home/simon/anaconda3/envs/views-hydranet-env/bin/python
# light_strider is a views-baseline model, NOT a HydraNet: it imports `views_baseline`,
# which is not installed in the hydranet env. Each family has its own environment under
# views-models/envs/ and its run.sh names it. The wrong interpreter fails instantly with
# ModuleNotFoundError -- which is exactly how this was found, on the first queued run.
PY_BASELINE=/home/simon/Documents/scripts/views_platform/views-models/envs/views-baseline/bin/python
interp_for() { case "$1" in light_strider|white_ranger) echo "$PY_BASELINE";; *) echo "$PY";; esac; }
mkdir -p "$OUT"

# Comparator FIRST: without it on the same partition, K3 has nothing to compare against and every
# roster verdict is VOID rather than a pass.
QUEUE="light_strider purple_alien pink_pirate blue_stranger bold_comet blazing_meteor heavy_freighter bright_starship violet_visitor"

for m in $QUEUE; do
  [ -f "$OUT/done_$m" ] && { echo "SKIP $m"; continue; }
  MD="$MODELS/$m"
  [ -d "$MD" ] || { echo "MISSING $m"; continue; }

  # Pre-flight: a config that cannot construct must fail in seconds, not after three hours.
  if ! "$PY" - "$m" <<'PYEOF'
import importlib.util, sys
from pathlib import Path
m = sys.argv[1]
root = Path("/home/simon/Documents/scripts/views_platform/views-models/models") / m / "configs"
parts = {"run_type": "validation"}
for stem in ("config_hyperparameters", "config_meta", "config_deployment"):
    p = root / f"{stem}.py"
    if not p.exists():
        continue
    s = importlib.util.spec_from_file_location(stem, p)
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
    g = [n for n in dir(mod) if n.startswith("get_")]
    parts.update(getattr(mod, g[0])())
if parts.get("model", "").startswith("Hydra"):
    from views_hydranet.utils.config_initializer import ConfigInitializer
    ConfigInitializer(parts).get_config()
print(f"preflight OK: {m}")
PYEOF
  then echo "PREFLIGHT FAILED $m — skipping"; continue; fi

  # Refuse on a leftover prediction dir: the pipeline names it after the artifact, so a stale one
  # makes the scored cube ambiguous. This cost an hour on 2026-09-06.
  if [ -n "$(ls -d "$MD"/data/generated/predictions_* 2>/dev/null)" ]; then
    echo "GUARD FAIL $m: stale prediction dir"; continue
  fi

  echo "=== $m  $(date +%F' '%H:%M:%S) ==="
  ( cd "$MD" && "$(interp_for "$m")" main.py --run_type validation --train --evaluate ) \
      > "$OUT/log_$m.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAILED rc=$rc"; continue; fi
  # Evidence the clamp was live (added 2026-09-07 precisely so this can be checked).
  grep -m1 -E "CLAMPED|evolves freely" "$OUT/log_$m.txt" | sed 's/^/  /'
  touch "$OUT/done_$m"; echo "  done $(date +%H:%M:%S)"

  # Cooldown. This is a LAPTOP RTX 4070 and the queue runs it flat out for hours. On 2026-09-07
  # the third consecutive model was 45% slower than the first two through its middle lessons --
  # same config, same lesson count, identical pace at lesson 50 -- because the GPU had reached
  # 86 C and the driver clamped the SM clock to 735 MHz against a 3105 MHz maximum
  # (SW Thermal Slowdown: Active). A throttled run at 24% of rated speed is slower than pausing
  # and then running at full speed, so this gap is expected to REDUCE total wall time, not add to
  # it. It also stops each model inheriting the previous one's accumulated heat, which is what
  # made the arms non-comparable on timing.
  cool_until=$(( $(date +%s) + 600 ))
  echo "  cooling: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -> waiting up to 10 min for <75C"
  while [ "$(date +%s)" -lt "$cool_until" ]; do
    tC=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -dc '0-9')
    [ -n "$tC" ] && [ "$tC" -lt 75 ] && break
    sleep 30
  done
  echo "  resumed at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader), $(date +%H:%M:%S)"
done
echo "VALIDATION QUEUE COMPLETE $(date)" > "$OUT/QUEUE_DONE"
