# `d4_result_CONTROL_nb_4acbf4d3.npz` — frozen 2026-10-21

> ## ⛔⛔ CORRECTION, 2026-10-28 — THE SECTION BELOW TITLED "Why this file cannot be regenerated" IS WRONG
>
> **The checkpoint is on disk.** Verified by hash, not by reading a note:
>
> ```
> /home/simon/Documents/views_hydranet_vehicle_20261012/calibration_model_20261012_033112.pt
> sha256  4acbf4d3c2b457b0827a84148606c55dab022e859257ca4e669e224edca6bc7f
> ```
>
> — which is **exactly** the sha stored inside `d4_result.npz`. It has its `.config.json` sidecar
> (load-bearing for `load_vehicle`), its own `.sha256`, a `PROVENANCE` file, and the complete
> `run_20261012/` directory with the prediction parquets and draws. Sidecar: family **`nb`**,
> dropout **0.15**, 300 lessons, git `f99413e`, model `violet_visitor`, region `land`.
>
> ⚠️ **The original sentence is literally true and its conclusion is false.** It says *"not on disk
> anywhere under `views-models`"* — correct, the file is under `~/Documents/`. The heading above it
> then generalises that to *"cannot be regenerated"*, and everything downstream inherited the
> generalisation, not the scoped fact. ⇒ **The defect is a scoped observation written up as a
> general one**, which is the same shape as the *"origins 0–5"* error (route-dependent, stated as
> general) and the third instance of recording something as absent that is on disk (W-83, W-89).
>
> **What is now true:**
> - ⭐ `d4_result.npz` **can** be regenerated, and was: `l3b_three_target_cube.py`'s **G-2** asserts
>   bit identity of its `sb` slice against this file and so re-derives it from the vehicle.
> - ⭐ B-1a / B-1b / B-2 / B-3 **can** be re-measured on the actual published `nb` @ 0.15 vehicle,
>   so `2026-10-21_body_scoreboard.md`'s *"frozen historical record, not a live comparator"* is
>   withdrawn — it **is** a live comparator.
> - ⛔ **But σ̂ for `nb` @ 0.15 is still NOT available.** σ̂ needs ≥2 further seeds, and this is one
>   checkpoint at seed 42. ⇒ RP-2's transfer table has a **false row** (the published numbers are
>   not on a deleted checkpoint) and a **still-true limit** (no replicates exist in that cell).
>   Do not read this correction as discharging the σ̂ caveat. See **C-394**.
>
> The original text is kept below unedited, marked wrong rather than deleted, per the standing rule
> that a retraction keeps the words it retracts.

---

## ⛔ ORIGINAL TEXT, 2026-10-21 — SUPERSEDED IN PART, SEE THE CORRECTION ABOVE


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
