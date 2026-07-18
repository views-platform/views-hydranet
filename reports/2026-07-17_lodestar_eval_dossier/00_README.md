# Lodestar Evaluation — dossier spine

**Opened:** 2026-07-17 · **Status:** ✅ FOUNDATION COMPLETE — 12-cell grid scored vs baseline on the frozen ruler (07). Headline: all-cell+MSE gated forecast beats white_ranger on crps-all (all 3 targets); pos_weight 2 best-calibrated; body timid (next lever). Terminology is LOCKED to `reports/GLOSSARY.md` (gate, body, bulk, tail, gated forecast, all-cell,
MSE, crps-all/events/none, size-ratio, pos-mcr, AP, Brier, the baseline). If a word here isn't in the
glossary, that's a bug.

## 1. Purpose — build the FOUNDATION
Establish, on one frozen ruler, exactly where our **gated forecast** stands against the **baseline** — so we
can improve from solid ground instead of shifting numbers. This exists because earlier comparisons were
silently wrong (different months) and the vocabulary kept drifting; the ruler + the locked glossary fix both.

## 2. The three questions this grid answers
1. **Is the gate calibrated, and at which pos_weight?** — sweep pos_weight {2, 3, 4, 5}; read **AP** + **Brier**.
2. **How much does the body wobble across seeds?** — 3 seeds; read the body scores per seed (spread visible).
3. **How does the gated forecast compare to the baseline?** — whole grid vs **white_ranger**, same ruler.

## 3. The model under test (held fixed across the grid)
- **gated forecast** = gate × body.
- **body: all-cell**, loss **MSE**, softplus head, **BatchNorm fix on**.
- Config that expresses this: `output_distribution='hurdle_shrinkage'` (gated compose) + **no**
  `hurdle_threshold` (all-cell body) + `loss_reg='mse'` + `reg_activation='softplus'`.
- Swept: **pos_weight {2,3,4,5} × seed {42,43,44} = 12 runs**.

## 4. The models in the table
| row | model | what it is |
|---|---|---|
| **baseline** | white_ranger | climatology (resample each cell's own history); PGM; all 3 targets |
| **the grid** | violet_visitor × 12 | gated forecast, all-cell body, MSE, softplus, BN-fix; pos_weight 2/3/4/5 × 3 seeds |

## 5. The ruler (FROZEN — `tools/lodestar_score.py`)
T=0 · identical months (457–469) · identical cells (intersection) · one truth. **Gate:** AP + Brier.
**Body:** crps-all, crps-events, crps-none, size-ratio (+ pos-mcr to add). Self-test + end-to-end validated.
Do not change mid-analysis.

## 6. Document index
| # | doc | status |
|---|---|---|
| 00 | README (this spine) | live |
| 02 | design (frame + the three questions) | rewritten for the all-cell grid |
| 03 | harness & invariants (the frozen ruler) | live |
| 05 | analysis plan (pre-registration) | rewritten for the all-cell grid |
| 07 | experiment log | open |
| tools/ | `lodestar_score.py` (frozen ruler) | done |
| results/ | scores + the lodestar table | to fill |

## 7. Status & next actions
**Status (2026-07-17):** ruler FROZEN + validated; white_ranger re-run + aligned (457–469); the earlier
positives-only grid was KILLED at the user's instruction; the plan is now the **all-cell + MSE gated forecast
grid**.
- [x] ruler built, self-tested, end-to-end validated, frozen
- [x] white_ranger re-run on current partition → aligned to 457–469
- [ ] 2-lesson smoke: all-cell body + gated forecast + MSE trains & evaluates **without the bloom**
- [ ] run the 12-cell grid (pos_weight {2,3,4,5} × seed {42,43,44})
- [ ] score all 12 + white_ranger on the frozen ruler → the lodestar table (answers the 3 questions)

## 8. Conventions
Locked glossary. Stealth rule for views-models (edit/restore in place; never commit/push). Dossier
git-tracked. Frozen ruler unchanged mid-analysis.
