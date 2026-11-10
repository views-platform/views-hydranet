# 07 — Experiment log (append-only)

**Serves:** the record half — every run against the ranking body, its outcome, and its verdict
against the falsifiers that were pre-committed before it ran.

⛔ **Negatives are first-class here.** This log is a Popperian record and a meta-evaluation corpus,
not a highlight reel. Every entry links its pre-registration and states the verdict **against the
pre-registered falsifiers**. Nothing is softened or omitted. A falsified entry gets the fuller
postmortem, linked, at the same prominence as a win.

---

## Status: **NO ENTRIES YET**

⚠️ Stated as a fact rather than left ambiguous. **Nothing has been run against the ranking body.**
The dossier was opened 2026-11-10 and `04_roadmap` phase 1 has not started.

⛔ This line exists because the precedent is bad: `reports/2026-11-04_delivered_gate_lookup/`
is fully pre-registered with a measured baseline and its log reads *"No entries yet"* — and a reader
cannot tell from that whether the experiment ran and failed or never ran at all. Here it is said
plainly: **it never ran.**

---

## Entry format

Newest at the bottom. ⛔ Never edit a past entry except to add a cross-link.

```markdown
## E-N · <date> · <what was run> · <⭐ CONFIRMS | ⛔ FALSIFIED | ⚠️ INCONCLUSIVE | VOID>

**Pre-registration:** <link + section refs>. Committed before the script existed.
**The one variable:** <what changed from the control, and nothing else>
**Driver / artifacts:** <script · artifact timestamp · results path · runtime and device>
**Read-out:** <the fast probe> → <the full table, estimator named on every line>
**Verdict against the pre-registered falsifiers:** <which fired / none> ⇒ <verdict>
**Decision:** <the next step, citing the pre-committed rule that selects it>
**⛔ What this does NOT establish**, stated here rather than left to be discovered:
```

**Status vocabulary:** `planned` · `running` · `⭐ CONFIRMS` · `⛔ FALSIFIED` ·
`⚠️ INCONCLUSIVE` · `VOID` (a pre-registered potency control failed ⇒ the instrument is refused,
and the arm is neither credited nor blamed).

⭐ **The `VOID` distinction is load-bearing** and was earned in #579: the `soft_gate` row was
correctly declared void by its own potency control rather than reported as a loss for B. Scoring
either way would credit the base rate to an instrument demonstrated not to work. Predictions voided
this way take a `V-` id in `reports/PREDICTION_LEDGER.md` and count in **neither** tally.

**Process failures get their own entry**, with the same prominence:
`## ⛔⛔ POSTMORTEM · <date> · A PROCESS FAILURE — <what was bypassed>`. No experiment need have run.
The log's value depends on it being honest about process.

---

## Linkage rule

Every entry links **both** directions: to the pre-registration it tests, and from that
pre-registration's decision rules back to here. ⚠️ `/rnd-dossier status` reports a log-integrity
failure for any pre-registration without a logged outcome, any outcome lacking a pre-registration
link, and any success-only drift.

⛔ **No entry may be written for a run whose criterion was declared after the run.** The correct
record in that case is a process-failure entry naming the omission — not a verdict.
