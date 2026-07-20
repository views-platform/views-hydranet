# ADR-066: Rename `output_distribution` to a readable `body` setting (and finally make ZINB nameable)

**Status:** Withdrawn — collapsed into ADR-067 (2026-07-20). The output-distribution subsystem and the
real `nb`/`zinb` families now live in **ADR-067**; the optional cosmetic field rename
(`output_distribution`→`body`) is deferred to **Epic B (#181)**. Retained here for reference only — not a
live proposal.
**Date:** 2026-07-19
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## Summary (read this first — self-contained)

The model's forecast for each map cell is built from two parts: a **gate** (the chance that *any*
violence happens there) and a **body** (the guess for *how many deaths* if it does). Which kind of body
the model uses is chosen today by one config setting, `output_distribution`, whose values are cryptic and
whose naming clashes with an unrelated setting. Worse, the one model we most want to try next — a
Negative-Binomial count body with a built-in spike of zeros — **has no value at all**, so we cannot even
write it down. This ADR renames the setting to `body`, gives its values plain readable names, and adds the
missing one. It is a rename-and-tidy, not a redesign: every existing model keeps behaving exactly as it
does now.

*(A short glossary of the technical terms is at the end.)*

---

## 1. The problem

The setting `output_distribution` picks which kind of body the model uses. Its current values are
`standard`, `hurdle_shrinkage`, `hurdle_nb`, `hurdle_lognormal`, and `dense_nb`. Three problems:

- **The names are cryptic.** You cannot tell what a value does by reading it. `hurdle_shrinkage` means
  "single-number body **with** a gate." `dense_nb` means "Negative-Binomial body **without** a gate." The
  value name tells you neither.
- **The model we want to try has no name.** There is no value for "Negative-Binomial body **with an
  explicit built-in spike of zeros**" — this is the model (**ZINB**) a sister project found works best
  (see the Background). Today you cannot express it.
- **One name collides with an unrelated setting.** `output_distribution='hurdle_shrinkage'` and the
  separate `loss_reg='shrinkage'` both say "shrinkage" but mean different things.

Note: a *fourth* thing — *which cells the body trains on* (all cells, or only the ones that had violence)
— used to be hidden in the `hurdle_` name prefix. It is already a separate setting, `body_mask`
(ADR-065), and stays separate. This ADR does not touch it.

## 2. The decision

Rename `output_distribution` to **`body`**, and give it plain values whose names say what they are:

| `body` value | what it is | replaces |
|---|---|---|
| `point` | a single-number guess, no gate | `standard` (the default) |
| `point_gated` | a single-number guess, times a gate | `hurdle_shrinkage` |
| `nb` | a Negative-Binomial count body, no gate (it makes its own zeros) | `dense_nb` |
| `nb_gated` | a Negative-Binomial body for the positive amount, times a gate | `hurdle_nb` |
| `zinb` | a Negative-Binomial body **plus a built-in spike of zeros**, no gate | *(new — had no name)* |
| `lognormal_gated` | a lognormal body, times a gate | `hurdle_lognormal` |

The `_gated` suffix means "there is a separate gate." A value with no suffix has no separate gate — the
body makes its own zeros (a count body puts probability on 0; `point` just regresses toward 0).

`zinb` has no `_gated` variant on purpose: a ZINB already contains its own zero mechanism, so a separate
gate would be doing the same job twice. That is why picking `zinb` is all you say — its self-zeroing is
part of what `zinb` *means*, not a second thing you set.

### This makes the next experiment a one-line change

The next question we want to answer is: *does adding a built-in zero-spike to a Negative-Binomial body
help?* The two models being compared now differ by **exactly one line**:

```python
# plain all-cell Negative Binomial
body      = "nb"
body_mask = "none"

# ZINB — the only change is this one line
body      = "zinb"     # <-- the single difference
body_mask = "none"
```

Today, `nb` exists (as `dense_nb`) but `zinb` cannot be written at all.

### Nothing existing breaks

Each old `output_distribution` value maps to exactly one new `body` value, with the model's behaviour kept
*bit-for-bit identical*:

| old `output_distribution` | new `body` |
|---|---|
| `standard` (default) | `point` |
| `hurdle_shrinkage` | `point_gated` |
| `hurdle_nb` | `nb_gated` |
| `hurdle_lognormal` | `lognormal_gated` |
| `dense_nb` | `nb` |
| *(had no name)* | `zinb` |

*(The `quantile` value is retained unchanged and is out of scope for this ADR.)*

### Guardrails

The config rejects bad input at load time, before any training starts (this repo's fail-loud rule,
ADR-008/009):

- An unknown `body` value is rejected, with the list of valid values.
- The old `output_distribution` key, if still present, is rejected with a message telling you the new
  `body` value to use instead (the same clean break we used when retiring `hurdle_threshold` in ADR-065).

## 3. Why this is better

- **You can read the model straight off the config** — the value names say what the body is and whether
  it has a gate.
- **The model we want to try becomes writable** — `zinb` finally exists, and the experiment that matters
  becomes a one-line change.
- **The name collision is gone** — there is no `hurdle_shrinkage` value anymore.

## 4. Consequences

**Benefits:** a config you can read; the ZINB experiment unblocked; a new count body is a new value in one
short list.

**Cost:** every place that sets `output_distribution` today (the main run config, the sweep scripts, the
tests) switches to the new `body` name and value. We soften this with a one-release **translation layer**
that accepts an old `output_distribution` value and maps it to the new `body` value (printing a
deprecation warning), then remove the old key. The change renames one field and adds one new value.

## 5. How we will know the refactor is correct

- **No behaviour change:** for each old value, the new `body` value must produce a **bit-for-bit
  identical forecast**. A test captures each old value's output *before* the change and re-checks it
  *after* (the same before/after safety net we used in ADR-065).
- **Guardrails work:** unknown values and the retired old key both raise an error.

## 6. How we will know a new *model* is actually good

Separate from the rename: when we later score a new count body (e.g. `zinb`), a sister project taught us
that an in-sample "win" can vanish under proper scrutiny. So the bar is:

- score on the **held-out validation years**, not the years the model was tuned on;
- repeat with **at least 3 random seeds** (a single seed's ranking is often just noise);
- run the calibration check on **only the ~1% of cells that actually had violence** — a check pooled over
  all cells is misleading, because the 99% easy zeros hide the real weakness;
- also use a **tail-weighted score** that emphasises the large death tolls, since that is where the
  decisions matter — not the average score alone.

## 7. How we will build it (for implementers)

- **Config** — `config_initializer.py`: rename the field to `body`; validate the value list; reject the
  old `output_distribution` key.
- **Forecast assembly** — `hydranet_inference.py`, the `_emit_magnitude` step: keep the existing
  per-value branches; add the `zinb` branch (the Negative-Binomial mean plus a zero-spike sized from the
  data).
- **Loss wiring** — `utils.py`, `choose_loss`: reuse the existing Negative-Binomial code; add the ZINB
  likelihood.
- **Migration** — copy the ADR-065 translation-layer + before/after-test approach.
- **First experiment this unblocks** — plain all-cell Negative Binomial (`nb`) **vs** ZINB (`zinb`) — the
  one-line change above — scored on the standard evaluation plus the harder bar in §6.

---

## Glossary (the technical terms used above)

- **gate** — the part of the model that predicts *whether* a cell has any violence (a probability).
- **body** — the part that predicts *how many* deaths.
- **point** — a single-number guess, with no spread.
- **Negative Binomial (`nb`)** — a probability distribution over whole-number counts (0, 1, 2, …).
  Naturally puts a lot of weight on 0, so it can model mostly-zero data without extra help.
- **ZINB (zero-inflated Negative Binomial)** — a Negative Binomial plus an explicit built-in spike of
  probability at exactly 0, to match data that is zero even more often than a plain Negative Binomial
  expects.
- **lognormal** — a distribution over positive continuous amounts, skewed so most values are small but a
  few are very large. It has no probability at exactly 0, so it always needs a gate.
- **gated** — the body is multiplied by a separate gate that decides whether the cell is on. A body with
  no gate makes its own zeros instead.
- **calibration check** — a test of whether the model's stated uncertainty is honest: e.g. do outcomes
  land above the "90% chance it's below this" line about 10% of the time?

## Background (why this came up now — optional)

- An **8-expert config-landscape review** (2026-07-19) flagged `output_distribution` as the worst
  offender because its values are cryptic and one collides with the `shrinkage` loss name.
- The **ZINB** target comes from the sibling **views-baseline** repo, which compared many distributions
  against the crude "just resample this cell's own past" baseline; ZINB was the one that reliably beat
  both plain Negative Binomial and that baseline on the lower-volume targets. **Caveat:** that study used
  a different setup — it fit a distribution to each cell's own history, whereas our model also uses
  neighbouring cells and other predictors. So it *points to* a promising direction rather than proving
  ZINB will win here. That is exactly why we make it a cheap one-line experiment.
- **Related ADRs:** ADR-065 (`body_mask` — which cells the body trains on, kept separate), ADR-008/009
  (reject bad config loudly), ADR-054/055/059 (the likelihood-based losses).
