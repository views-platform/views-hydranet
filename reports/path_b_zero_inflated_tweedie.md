# Path B: Zero-Inflated Tweedie Distribution (ZITD) Output Head — SUPERSEDED

**Status:** Superseded 2026-06-05 — absorbed into the distributional-head dossier.

The full Path B proposal (problem statement, Tweedie/ZITD math, architecture, NLL loss,
implementation plan, theory, empirical evidence, limitations, evaluation, references) now
lives as the **canonical design** at:

> **`reports/2026-06-05_distributional_head_dossier/02_design.md`**

That document preserves the original §1–§11 verbatim and adds **§0 — advances since 2026-05-27**:
the link to this session's C-113 diagnostics (ZITD structurally removes the `expm1` runaway),
the autoregressive-feedback treatment, MC-dropout coexistence (Kendall 2017), the chronic-vs-acute
framing relative to the C-111 bisect, and the Tweedie-density implementation blocker.

See the dossier index: `reports/2026-06-05_distributional_head_dossier/00_README.md`.

(Original date 2026-05-27. This stub retained so existing cross-references resolve.)
