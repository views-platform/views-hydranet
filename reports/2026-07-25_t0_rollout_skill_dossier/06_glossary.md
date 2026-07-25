# 06 — Glossary (program-specific; defers to the LOCKED reports/GLOSSARY.md)

New terms this program introduces. When one stabilizes, promote it into the LOCKED glossary (never invent a
synonym there; edit it).

- **horizon h** — steps ahead in the rollout; h=1 is the seed step (= the frozen-ruler T=0), h=36 the last.
- **free-running rollout** — AR rollout feeding back its own emitted prediction (deployed behavior; the
  bloom driver). The skill we actually get.
- **teacher-forced-oracle rollout** — AR rollout feeding back the realized truth each step; the intrinsic
  predictability ceiling given the features. Upper bound, not deployable.
- **skill crossover (h_x)** — the horizon where free-running crps-all rises above the climatology baseline;
  the usable rollout depth.
- **bloom-cost gap** — `crps_free(h) − crps_oracle(h)`; exposure-bias cost, separated from the ceiling.
  Large+growing ⇒ fixable bug; ≈0 ⇒ ceiling.
- **STABILITY ≠ SKILL** — bounded/sparse trajectory statistics ≠ accurate forecast; the reason this ruler
  exists.
