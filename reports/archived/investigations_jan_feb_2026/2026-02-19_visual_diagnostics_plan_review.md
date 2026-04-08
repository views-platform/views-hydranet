# Plan Review: Visual Truth Engine (Diagnostics)

**Date:** 2026-02-19
**Reviewer:** Claude Code
**Plan Reviewed:** `2026-02-19_visual_diagnostics_plan.md`
**Context:** `2026-02-19_diagnosis_random_number_generator.md`

---

## Overall Verdict

The plan is well-conceived for one specific hypothesis but significantly under-scoped for the full diagnostic challenge. It will definitively answer the Spatial Scrambling question (Hypothesis 1), but it largely ignores the other three live fault lines. If the bug is Hypothesis 2 or 3, you could execute this entire plan and still have no answer.

---

## What the Plan Gets Right

**The Gradient Test is the plan's strongest idea.** Treating `row_idx`/`col_idx` as diagnostic features and expecting smooth gradients is elegant, cheap, and immediately falsifiable. This is exactly the right test for Hypothesis 1 and it is hard to misread — either you see a gradient or you see static.

**The Null Object pattern is correct.** A no-op class controlled by a config flag is the right architecture. It keeps probes in the production code path without polluting it.

**Stage 3 is correctly identified as the critical point.** `VolumeHandler.from_df` with numpy fancy indexing is the exact location where the coordinate system can break silently. The plan's instinct to focus there is right.

**The 6-stage coverage is comprehensive in scope.** The pipeline has genuine fault lines at each of those points, and it is correct to probe all of them rather than only the suspected one.

---

## What the Plan Misses or Gets Wrong

### 1. Hypothesis 2 (Scaling Failure) is invisible to heatmaps without context

The Stage 2 biopsy will show you a heatmap of scaled values. But if the scaler silently skipped `classification_targets`, the map will look *reasonable* — just in the wrong range. Without expected range annotations or summary statistics printed alongside the plot, you cannot diagnose this visually. The plan needs either:
- A numerical stats panel per biopsy (e.g., `μ=0.02, σ=1.1, range=[-3.4, 4.1]`)
- An explicit assertion: "After log1p+zscore, values must be in ~[-4, 4]. Flag if not."

### 2. Hypothesis 3 (Hardcoded Heads Conflict) is completely unaddressable by this plan

This is a pure config/architecture mismatch. No heatmap will reveal that `Head 1` is being trained against `Target 2`'s labels. The plan should include a config validation step — a pre-flight check that prints the alignment of `[regression_targets, classification_targets]` vs the model's hardcoded head structure. This is two lines of code with no IO overhead, and it either passes or it does not.

### 3. Hypothesis 5 (Sorting Vacuum) is not probed

The plan has no diagnostic for temporal order integrity. A simple check — verifying `month_id` is monotonically increasing in the DataFrame before `VolumeHandler` ingests it — would close this off. Not a visualization, just an assertion with a logged warning.

### 4. Stage 4 (Sampling) is underspecified

"Are the 32x32 windows capturing conflict or empty ocean?" is a good question but the plan offers no concrete answer for how to sample them. Which windows do you plot? Random? First N? The ones with highest conflict density? Without this, the probe is vague and hard to implement. Recommend: plot the first K windows sampled per epoch, overlaid with the conflict target channel.

### 5. No quantitative fallback

The plan is entirely visual. Visual inspection is unreliable for subtle scrambling (partial offset errors, off-by-one). Consider adding Moran's I or a simple spatial autocorrelation metric alongside Stage 3 plots. If spatial autocorrelation is near zero, scrambling is confirmed without ambiguity. This is one `scipy` call.

### 6. The `biopsy_dataframe` method is underspecified

How does this method know which columns map to spatial dimensions? The plan says "Adapts DF to the biopsy grid" without specifying the coordinate column names. This will create friction during implementation — the method needs to know `row_col`, `col_col`, and `time_col` names, either passed as arguments or read from config.

### 7. The 5-timestep choice is arbitrary and unexplained

"5 sequential months" — which 5? First 5 (cold start), last 5 (end of sequence), or a window around a conflict peak? For detecting temporal scrambling, the choice matters. The plan should specify this or make it configurable.

### 8. No before/after comparison framing

The plan shows stages in isolation. The most diagnostic view is a side-by-side: Stage 1 `sb_best` vs Stage 2 `sb_best` on the same color scale. This makes the effect of `log1p` immediately readable. As written, each stage generates independent plots with no shared reference frame.

---

## Structural Gaps in the Execution Plan

Section 4 (Execution Plan) lists 5 steps but step 5 ("Eyeball the `row_idx` maps") is the only forensic step, and it only covers Hypothesis 1. There is no corresponding forensic step for the other hypotheses. The execution plan should end with a decision matrix:

> If Stage 3 gradient is smooth → Hypothesis 1 is ruled out → check Stage 2 statistics → if range is anomalous → Hypothesis 2 confirmed → else run config pre-flight → etc.

---

## Priority Recommendations

Execute the plan as written — it will likely resolve Hypothesis 1 quickly. But **before or alongside implementation**, add:

1. **Config/architecture pre-flight validator** — closes Hypothesis 3, no IO cost.
2. **Sort-order assertion** in the DataFrame ingestion path — closes Hypothesis 5.
3. **Numerical stat overlays** on Stage 2 biopsies — makes Hypothesis 2 diagnosable.

The visual biopsy infrastructure is good. The gap is that it has been scoped to one hypothesis while the diagnosis document lists four live ones.
