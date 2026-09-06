# 06 — Glossary

Terms this program adds. Repo-wide vocabulary is `reports/GLOSSARY.md` and **that file governs** —
if a word is missing there, edit it; do not invent a synonym here.

| term | definition |
|---|---|
| **behaviour sequence** | Lamb's term: the sequence of chosen hidden states (and optionally output values) produced by one forward mode over a window. The discriminator's input. |
| **teacher-forced mode** | inputs clamped to ground truth — what our training loop does today. |
| **free-running mode** | inputs self-generated from the model's own emission — what deployment does. |
| **the cell / the long-term half** | `hl_1..hl_4`, the last four of the eight `% 8` splits of `h` in `HydraBNUNet06_LSTM4.forward`. Distinct from **the hidden half** (`hs_1..hs_4`). |
| **`C_f` / `C_t`** | Lamb's two generator terms: `C_f` changes only free-running behaviour; `C_t` also pulls teacher-forced behaviour toward it. |
| **stability gate** | this dossier's pre-registered check that the discriminator neither failed to learn (accuracy ≈ 0.5) nor won outright (≈ 1.0). Breaching it makes a run **VOID**, not negative. |
| **VOID** | a run whose result cannot be read against the hypothesis because a precondition failed. Distinct from a **null**, which is a real measurement of no effect. |
