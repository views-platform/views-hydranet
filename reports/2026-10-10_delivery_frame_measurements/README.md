# Delivery-frame measurements — 2026-10-10

One script, `delivery_frame_audit.py`. It audits a **delivered selector-frame handover**: the
collapsed point surfaces a partner actually receives, rather than the sample cube our other
instruments take.

## Why it exists

§5.0(a) of `reports/2026-09-30_evaluation_learning_report.md` records the largest hole in the
report: **nothing in the instrument inventory scores the delivered product.** Every instrument in
Appendix A reads `origin_*/<target>/y_pred.npy` and returns AP, CRPS or Brier. The partner receives
collapsed frames. This is a first, partial answer — everything checkable **without truth**.

⛔ **It cannot score accuracy.** A handover folder carries no observed values, so there is no AP, no
CRPS and no calibration here. Those need the validation cubes, which Appendix C records as gone from
this machine.

## What it measures

| § | measures | why it is not trivial |
|---|---|---|
| a | NaN / negative / infinite values | the handover README says there is nothing to verify by hand; this is the positive statement instead |
| b | frame II and III ≥ frame I on every row | `mean = P(y>0)·E[y\|y>0]`, so a conditional mean below the mean would prove the frames came from **different draw sets** |
| c | total-mass ratios between frames, per model and target | ⚠️ sum-ratios, **not** the per-cell `ŷ/y` a delivery note may quote — that needs truth |
| d | per-step mass **and** step-to-step relative L1 change | separates "the marginal intensity settles" from "the field is frozen"; these are different claims and only the first was true |
| e | member distinctness on the union support | answers whether *N* delivered models are *N* forecasts |

## Two traps it was written to avoid

⛔ **Whole-field agreement metrics measure shared zeros.** On a ~99% zero field, "identical rows
between pairs" reads 99.5% because both members are zero nearly everywhere. The first version of (e)
reported exactly that and it was wrong — see the learning report's Appendix C. Section (e) therefore
restricts to the union support and prints the all-rows correlation **only** so it can be labelled
inflated. Prefer the firing histogram.

⛔ **Filenames are not shared across models.** Each model carries its own run timestamp, so files
must be matched by the trailing `_NN` origin index, not by name. Section (e) asserts the row order is
identical across models on the `month_id × priogrid_id` key before comparing row-wise.

## Run

```bash
conda run -n views-hydranet-env python \
    reports/2026-10-10_delivery_frame_measurements/delivery_frame_audit.py \
    --root ~/Documents/views_selector_frames_20260929
```

`--skip a b c` omits the two expensive full passes. `--target` selects the series for (d) and (e).

## Provenance

Every figure it produces for the 2026-09-29 handover (8 models × 3 frames × 13 origins, 312 parquet,
242,678,592 rows per frame) is quoted in the learning report's §7 **S7** row and its Appendix C
retraction entry. The `models` session independently reproduced (d) and (e) on the same columns;
agreement to ≤0.03 on churn and to the unit on the firing histogram and the mass ratios.
