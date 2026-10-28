# RP-2 step 0 — the pipeline is bit-deterministic, and that REFUTES the readout that measured σ̂

**Date** 2026-10-28 · **Cost** $0, 24 s local CPU · **Script** `determinism_matrix.py`, committed
`ab3f122` **before** it ran (D-5) · **Output** `determinism_matrix.out` · **All five gates PASS**

**C-351 slots.** Population = the 6 staged `kestrel` checkpoints at 300 lessons ·
transform = raw float32 weight tensors, no normalisation · estimator = exact bitwise equality
(`torch.equal`), max |Δ| reported only for pairs that are not equal · not normalised.

---

## 0 · ⛔ The thing to say first: σ̂ was NOT unrun

`reports/RESEARCH_PROGRAMME.md` RP-2 says *"run-to-run σ̂ has never been computed"* and *"⭐ it is
unrun, not blocked"*. **That is wrong.** A full readout exists at
`/home/simon/Documents/kestrel_sigma_backup_20261021/READOUT.md`, dated **2026-10-21**, reporting
σ̂ for the gate-calibration slope `b` over seeds 42/43/44, plus M2/M3/M4 and a pre-registered
prediction.

⇒ **This is W-83's shape for the second time** — a quantity recorded as missing while the artifact
sat on this machine. ⚠️ **The mitigation that worked was not vigilance.** It was reading the
directory before planning the compute, which took four minutes and is now the first step of any
"this has never been measured" claim.

⚠️ **What IS still unrun:** σ̂ for **B-1a / B-1b** (the spatial-structure and placement statistics
#462/#463 ask for). The 2026-10-21 readout measures σ̂ of a **different quantity** — the gate
calibration slope `b`. Same word, different statistic, which is GLOSSARY §5e's whole subject.

## 1 · ⭐⭐⭐ The measurement

| | bit-identical tensors, of 190 |
|---|---|
| **amber ≡ silver ≡ silver_0_1_4** | **190 / 190**, all three pairs |
| **every other pair** (12 of them) | **15 / 190** |

⛔ **The 15 are not a partial agreement.** They are exactly the 15 BatchNorm `num_batches_tracked`
counters — `int64`, every one equal to **10410** — i.e. the three runs saw the same number of
batches. **0 of the 175 float tensors agree between any two seeds.** Verified by name and dtype,
not inferred from the count.

⇒ **Pipeline determinism is EXACT**, and across more than the readout claimed:

| the three identical checkpoints | git sha | views-hydranet | trained |
|---|---|---|---|
| `amber` | **73c3f42** | **0.1.2** | 2026-10-12 → 13 |
| `silver` | 2e73de3 | **0.1.3** | 2026-10-19 |
| `silver_0_1_4` | 2e73de3 | **0.1.4** | 2026-10-21 |

**Two different git shas, three library versions, three pods, three different days — and 190 of 190
weight tensors identical to the bit.**

## 2 · ⭐⭐ What that settles — RP-2's own falsification test, already answered

RP-2's hard question (the plan's S5) was: *is σ̂ seed variance, or pipeline non-determinism?* —
because views-models' own config header warns that identical seeds *"DEFLATE σ̂ … a manufactured
positive"*, and if rerun variance were comparable to across-seed spread then **the seed axis is
decorative and every one-run-per-arm result in this programme reads differently.**

**Answer: rerun variance is exactly ZERO.** ⇒ **σ̂ across seeds is pure seed sensitivity, with no
non-determinism term to subtract.** That is the favourable outcome and it is now measured rather
than hoped.

## 3 · ⛔⛔ And it REFUTES the readout's headline, which two committed documents disagreed about

`READOUT.md` §*The confound* argues the as-launched σ̂ is not a seed variance:

> *"By elimination the excess sits on the 0.1.3/0.1.4 boundary"* — and therefore reports the
> **same-version pair** `{43,44}` as the clean estimate: **0.0002 … 0.0196**, against the
> as-launched **0.0351 … 0.0932**.

`ARTIFACT_METADATA.md`, staged **later** in the same directory, says the opposite — that
`silver_0_1_4` is bit-identical to `silver` across that very boundary, and calls it *"the
single-variable test that killed the library-version hypothesis"*.

⛔ **Both cannot be true, and no gate in either document looks.** That is the living report's **L8**
class. Measured verdict:

| claim | verdict |
|---|---|
| ARTIFACT_METADATA — `silver_0_1_4 ≡ silver`, 0.1.3 vs 0.1.4 | ✅ **UPHELD**, 190/190 |
| READOUT — the excess sits on the 0.1.3/0.1.4 boundary | ⛔ **REFUTED** |

**The 0.1.3 → 0.1.4 boundary moves nothing, so there is nothing to eliminate onto it.** With rerun
variance at zero and hardware excluded, the only variable left between `silver` (42) and
`bronze`/`cobalt` (43/44) is **the seed**.

⇒ **The honest σ̂_b is the as-launched family — 0.0351 (n≥1000) / 0.0932 (all) / 0.0434 (√n), df=2.**
⛔ **The same-version pair is not a cleaner estimate; it is the same estimate with the outlying seed
dropped, and it understates the floor by 3× to 200×.** Any decision rule built on 0.0002–0.0196 is
built on the smaller of two numbers for a reason that does not hold.

⭐ **And the readout's own pre-registration decides this, not my preference.** It predicted: *if the
version boundary is real, seed-42-on-0.1.4 shows `bronze`/`cobalt`'s positive-throughout
signed-error pattern.* `silver_0_1_4` **is** seed-42-on-0.1.4, and it is bit-identical to `silver`,
which **crosses zero**. ⇒ The version hypothesis is dead **on its own declared terms**.

## 4 · ⭐⭐ One clean single-variable contrast falls out, and it is uncomfortable

With the version proven inert, `silver_0_1_4` vs `indigo` differs in **one recorded variable only**:
`weight_decay` 0.1 → 0.0. Both seed 42, both `2e73de3`, version inert.

| contrast | differs in | max |Δ| over 175 float tensors |
|---|---|---|
| `bronze` vs `cobalt` | **seed only** (43 vs 44) | **13.83** |
| `silver_0_1_4` vs `cobalt` | **seed only** (42 vs 44) | **13.86** |
| `silver_0_1_4` vs `bronze` | **seed only** (42 vs 43) | 9.68 |
| `silver_0_1_4` vs `indigo` | **`weight_decay` only**, 0.1 → 0.0 | **4.92** |

⇒ **Changing the seed perturbs the weights ~2.8× more than removing weight decay entirely.**

⛔ **Read this narrowly.** `max |Δ|` over parameters is **not** a skill statistic and a larger
parameter perturbation does not imply a larger effect on any forecast — a tiny weight change in the
right place can move a score more than a large one elsewhere. The claim is about **parameter space
only**. It is logged because it is a clean single-variable contrast against a clean seed contrast on
the same scale, and because it is the shape of the thing σ̂ exists to catch.

## 5 · What this hands RP-2 next

1. ⛔ **Amend `RESEARCH_PROGRAMME.md` RP-2** — it says σ̂ is unrun; σ̂_b is run and this is its
   corrected reading. What remains is **σ̂ for B-1a/B-1b**, a different statistic.
2. ⛔ **The decision rule must use 0.0351–0.0932, not 0.0002–0.0196**, and must carry the df=2
   caveat: 95% CI **[0.52 σ̂, 6.28 σ̂]**, a **12.1-fold** span, with the honest bar against one
   unreplicated arm at **4.97 σ̂**.
3. ⭐ **The determinism result makes the B-1a/B-1b σ̂ run cheaper and stronger than planned.** The
   plan's **G-3 mask-pairing gate** existed because `lock_entropy` sets
   `use_deterministic_algorithms(warn_only=True)`, so non-determinism *could* fall through and let
   draw noise masquerade as a seed effect. ⇒ **Bit-identical weights across three pods and three
   library versions is far stronger evidence than the mask-hash check G-3 proposed**, though it is
   evidence about **training**, not about the **scoring forward pass**, so G-3 is narrowed rather
   than dropped.
4. ⚠️ **The transfer limit is unchanged** — these are all `truncated_nb` @ dropout 0.15. The
   published B-1a/B-1b headline numbers are `nb` @ 0.15 on a **deleted** checkpoint, and the live
   `nb` control is @ 0.125. σ̂ discharges the noise-floor requirement for `truncated_nb` @ 0.15, and
   only provisionally — as a transported CV — for `nb`.

## 6 · ⚠️ What this does NOT establish

- **Not** that seed 42 is the outlier rather than 43/44 being the pair that happens to agree. n=3,
  df=2. Three points cannot identify which one is unusual.
- **Not** that determinism holds for the **scoring** path. Measured on training outputs (the saved
  weights). MC-dropout draws at score time are a separate question.
- **Not** that 0.1.2 → 0.1.4 is inert in general — only for **this config**, which is what the
  comparison covers.
- **Not** a σ̂ for anything #462/#463 asks for. That measurement has not run.
