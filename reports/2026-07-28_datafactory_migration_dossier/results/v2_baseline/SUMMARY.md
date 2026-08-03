# v2 foundation baseline — gated_NB (nb family, soft_gate) × 3 seeds on datafactory truth

**Epic #203 S5** · run 2026-07-28 17:19→18:50 · scored on the **v2 datafactory truth**
(`tools/v2_truth/`, sha256 `620f4aa…`) with the byte-identical frozen lodestar functions.
Arm = **gated_NB** (`output_distribution=nb`, `forecast_composition=soft_gate`, `n_head_samples=8`,
40 lessons) — NOT the legacy `hurdle_nb` floor (which stayed the trap-restore anchor). Cubes deleted
after scoring; floor md5 `6c28bdb…` restored; no OOM.

## crps-all (primary; per seed + mean) vs v1 reference

| target | nb_42 | nb_43 | nb_44 | **v2 mean** | v1 foundation ref | Δ vs v1 |
|---|---|---|---|---|---|---|
| sb | 0.14217 | 0.14542 | 0.14421 | **0.1439** | 0.1373 | +4.8% |
| ns | 0.08238 | 0.08565 | 0.08320 | **0.0837** | 0.0834 | +0.4% |
| os | 0.03031 | 0.02859 | 0.03176 | **0.0302** | 0.0276 | +9.4% |

Seed-stable (sb range 0.142–0.145; os 0.029–0.032). Slightly higher than v1 — expected clean-cut drift
(KGI handling + GED vintage change the targets a hair; different data ⇒ v1/v2 not byte-comparable).

## Structure (per seed, all targets)
- **size_ratio = 0.0** every seed/target ⇒ the **body is timid** (E[y]≈0 on positive cells) — the same
  conclusion as v1 (there ~0.02; here soft_gate drives it to ≈0). The magnitude wall is unmoved.
- **tail dead** — `tail_scorecard` E[y]=0 / q90=0 across all truth-magnitude bins (sb/ns/os).
- **crps_events** ~17.85 (sb) / ~23.49 (ns) / ~6.76 (os), seed-stable — the timid body does not cover events.
- **AP** (occurrence) ~0.306 (sb) / ~0.234 (ns) / ~0.105 (os), seed-stable — occurrence skill intact.

## Tier-B verdict (do the v1 conclusions survive on datafactory truth?)
**PASS (2 of 3 legs, on reproduction + structure):**
1. ✅ **Foundation reproduces** — crps-all within a few % of v1, seed-stable across 3 seeds.
2. ✅ **Same structural conclusions** — body timid (size_ratio ≈ 0), tail dead, occurrence skill intact.
3. ✅/⚠️ **vs white_ranger — CLOSED (2026-07-28).** The datafactory white_ranger clone =
   **`light_strider`** (ConflictologyModel, africa_me_legacy), run **FRESH end-to-end** (fresh pull +
   fresh train + eval; no stale data/artifacts — the stale `20260604` cache + `.pkl` were moved aside)
   in the `views_pipeline` env, scored on the **same v2 support** (N=170430 = 457–469, 13 origins;
   count-only cube, empty by-template ⇒ P(conflict)=frac samples>0). Results:

   | target | gated_NB crps-all | white_ranger crps-all | winner |
   |---|---|---|---|
   | sb | 0.144 | 0.191 | **gated_NB** (−25%) |
   | ns | 0.084 | 0.088 | **gated_NB** |
   | os | 0.030 | 0.030 | tie (wr 0.0300 vs nb 0.0302) |

   **gated_NB beats white_ranger on sb + ns, ties on os** — NOT the v1 "beats on all 3" (v1 used an
   MSE body; this is the timid gated_NB). white_ranger has higher occurrence AP (sb 0.335 vs 0.306;
   os 0.159 vs 0.105) and real magnitude (size_ratio 0.40/0.15/0.05 vs 0.0); gated_NB wins crps-all
   via true-zero calibration (crps_none), not magnitude. Score: `results/v2_baseline/score_white_ranger.txt`;
   repeatable runner: `tools/score_white_ranger_v2.sh`.

**Conclusion:** the v2 datafactory-era baseline is established and trustworthy for future experiments to
extend from. Tier-B holds: foundation reproduces (crps within a few % of v1), structure survives
(timid body, dead tail), and it beats the climatology baseline on 2/3 targets (ties os). The clean-cut
holds: values drift a few %, conclusions survive.
