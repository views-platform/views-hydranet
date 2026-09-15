# 06 — Glossary (program-local additions only)

The repo glossary (`reports/GLOSSARY.md`) governs. Terms this program introduces:

| term | meaning |
|---|---|
| **arm** | one (architecture × seed) 300-lesson model, its own directory in views-models, untracked there |
| **candidate** | one of the six new architectures; a candidate becomes two arms (seeds 42, 43) |
| **incumbent** | `HydraBNrecurrentUnet_06_LSTM4`, the current architecture; its arms are the existing `fullzero_fortytwo`/`fortythree` controls |
| **smoke** | a 2-lesson train+emit+score whose purpose is to prove the architecture *runs* and to measure its cost — **never a scored result** |
| **preflight** | the checks run once before the queue launches (`03` §D) |
| **postflight audit** | the per-arm check that the **setup** is intact — artifacts present, non-empty, consistent `N`/support, no NaN — as distinct from the verifier, which judges the **result** |
| **capacity confound** | candidates (5) and (6) add substantial parameters, so a win there mixes inductive bias with capacity; reported, never silently compared |
