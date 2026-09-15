#!/usr/bin/env bash
# emit_s1.sh — one emit of the eps=0 control vehicle, cubes RETAINED, for the S1 error measurement.
#
# --keep-cubes is required (S1 needs per-cell predictions, and the normal path deletes them) and is
# safe here for exactly one reason: C-321 records that --keep-cubes SKIPS the multi-arm contamination
# guard, because every arm writes the same artifact-named prediction path. With a single arm there is
# nothing to contaminate. The single-arm precondition is asserted rather than assumed.
set -uo pipefail
HYD=/home/simon/Documents/scripts/views_platform/views-hydranet
D="$HYD/reports/2026-09-04_input_noise_dossier"; RES="$D/results/s1"
ARM=fullzero_fortytwo   # eps=0.0, 300 lessons, seed 42 — the shape S5's control takes
M="$HYD/../views-models/models"
export WANDB_MODE=offline WANDB_SILENT=true
mkdir -p "$RES"
ART=$(ls -t "$M/$ARM"/artifacts/*.pt 2>/dev/null | head -1)
[ -z "$ART" ] && { echo "no artifact for $ARM" >&2; exit 2; }
echo "[$(date '+%F %T')] emit $ARM <- $(basename "$ART") (cubes retained)" | tee "$RES/emit.log"
conda run --no-capture-output -n views-hydranet-env python \
  "$HYD/reports/2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py" \
  --model "$ARM" --artifact "$(basename "$ART")" --arms identity \
  --keep-cubes --tag s1 --out "$RES" >> "$RES/emit.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] emit rc=$rc" >> "$RES/emit.log"
ls -d "$M/$ARM"/data/generated/predictions_* 2>/dev/null | head -1 > "$RES/CUBE_DIR" || true
touch "$RES/EMIT_DONE"
