# 01 — Literature (scaffold + gaps-to-fetch)

Populate via the `library` skill (`/library search`, `/library find`, `/library cite`) before the
method-review. Anchors expected in-corpus; verify + cite, don't assert.

## Roles we need grounding for
- **Exposure bias / autoregressive rollout** — why free-running decays vs teacher-forced (the bloom's
  mechanism). Anchors: scheduled sampling (Bengio 2015), professor forcing, DeepAR (Salinas) ancestral
  sampling, GTF / pushforward. → grounds the free-running−oracle decomposition (`02.3`).
- **Forecast skill vs a climatology reference** — CRPSS / skill scores; when "beats climatology" is the
  right bar; horizon-decay of skill. Anchors: Gneiting & Raftery 2007 (proper scores), Lerch/Taillardat
  (CRPS decomposition). → grounds DQ1 / the crossover metric.
- **Predictability ceiling in conflict forecasting** — long-horizon limits; feature/world-model bound.
  Anchors: the VIEWS eval literature; our own amount-ceiling wall finding (internal). → grounds `03.F.2`.
- **JEPA / diffuse-mean-is-correct** — the user's framing that a multi-step predictive mean *should* be
  diffuse (video prediction blur). Anchors: LeCun JEPA, Kohl 2018 (probabilistic seg). → grounds the "is
  the bloom a bug or honest marginal uncertainty" question.

## Gaps to fetch
- [ ] A concrete CRPSS-vs-horizon precedent in count/spatial forecasting (metric shape for `02.3`).
- [ ] Teacher-forced-as-ceiling precedent (is realized-feedback the standard exposure-bias probe?).
