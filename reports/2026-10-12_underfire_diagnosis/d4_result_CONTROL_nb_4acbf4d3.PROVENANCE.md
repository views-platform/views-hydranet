# `d4_result_CONTROL_nb_4acbf4d3.npz` — frozen 2026-10-21

A byte-identical copy of `d4_result.npz` as it stood at 2026-10-20 18:08, taken before the
`truncated_nb` re-baseline overwrites that fixed filename (`d4_mu_on_the_grid.py:109` hardcodes
`out_path`, with no `--out`).

| | |
|---|---|
| vehicle sha stored inside | `4acbf4d3c2b457b0827a84148606c55dab022e859257ca4e669e224edca6bc7f` |
| that vehicle | `calibration_model_20261012_033112.pt`, family **`nb`**, git `f99413e` (`2026-10-15_loss_balance_prereg.md:291`; registered as THE diagnosis vehicle at `2026-10-12_underfire_diagnosis_prereg.md:391`) |
| ⛔ that file | **DELETED — not on disk anywhere under `views-models`.** No file, no `.config.json`, no `.sha256` |
| copy verified | `sha256 cf13f2d7e165d7d3…` identical to the source, both 138,903,920 bytes |
| contents | `mu`, `theta`, `gate` at `(4, 49, 64818)`, months 421–469, 64,818 land cells |

## ⛔ Why this file cannot be regenerated

The checkpoint it was measured on no longer exists. Every baseline number in
`2026-10-21_body_scoreboard.md` §5 and §7 — B-1's scale profile, B-2's slopes, B-3's levels,
including the **24.7%** and **0.4562** that the structure-penalty prereg states its predictions
against — traces to this one file and to nothing else.

⇒ It is a **frozen historical record**, not a live comparator. The arm is re-baselined on the
surviving `truncated_nb` vehicle (`calibration_model_20261013_010739.pt`, sha `2a5efc2a67c08eac…`)
because that one can be re-measured and this one cannot.

⚠️ `reports/` is gitignored and this is 139 MB, so it is **not in git**. It exists on this disk only.
