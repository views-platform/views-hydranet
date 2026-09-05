# 04 — Roadmap

The phased sequence, its dependency graph, its STOP-gates and the per-story acceptance criteria live
in **GitHub, not here**: epic **#311**, tracking checklist **#320**, stories **#312–#319**.

Duplicating them into a markdown file would create a second copy that drifts — and *prose asserting
something the system does not actually do* is **C-303**, the most habitual defect in this repo's
register at **ten** recorded occurrences. One source of truth.

**Shape:** a chain, not a fan. Nothing parallelises usefully — S2's design has no basis until S1
measures, S3 cannot audit code that does not exist, and S5 must not spend GPU until S4 proves the knob
can act. The only work that can start early is S5's launcher and S6's readout, which should both be
written — and every verdict branch fired on synthetic fixtures — *before* any real data exists.
