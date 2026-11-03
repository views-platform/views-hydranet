# PREREG — EXP-03, four more seeds, written BEFORE they run

**2026-11-03.** Committed before any of the eight new arms exists.

⛔⛔ **This document does NOT change `PREREG.md` §6.** §6 is the verdict rule for `P-23`, it was
written before EXP-02's numbers existed, and §6 device 6 forbids changing it after them. Every
branch condition in §6 carries over **verbatim**. This document does exactly three things §6 could
not do for itself:

1. **Specifies** what §6's phrase *"both seeds"* means at n=6. §6 was written for n=2 and the
   phrase has no defined reading at n=6. ⇒ Pinned below as **all seeds, strictly** — the stringent
   reading, chosen so this specification cannot weaken the rule.
2. **Discharges** §6's own MIXED-branch order: *"names the seed count needed, **derived** from the
   observed spread rather than assumed."* EXP-02 returned MIXED, so that clause is live.
3. **Pre-registers** the new predictions, including one that predicts this experiment will not
   resolve `P-23` either.

⚠️ **The reason this document exists at all is a failure of mine, logged as `W-124`.** I first told
Simon that §6 was the wrong instrument and needed replacing with a mean endpoint. It is not: §6 is
stringent, not inadequate, and it reports MIXED because the h36 effect measures ≈ 0. The argument
ran toward the conclusion I wanted — a confirmable endpoint. **§6 stands unaltered.**

---

## 1 · What EXP-02 measured, and why it cannot answer its own question

ΔAP = AP(UpsampleConv2) − AP(its own-seed control), `sb`, **gate** AP, frozen
`lodestar_score.average_precision`, paired origin-block bootstrap over **13 origins**,
**842,634 cell-months** per endpoint, S = 16 on all four arms.

| seed | ΔAP @h1 | ΔAP @h36 |
|---|---|---|
| 42 | **+0.004890** | **−0.003549** |
| 43 | **+0.004700** | **+0.003405** |

⇒ **§6 verdict: MIXED** — *"seeds disagree at h36."* Not CONFIRMS (requires < 0 at h36 on both);
not FALSIFIES (requires ≥ 0 at h36 on both, and seed 42 is negative).

⭐⭐ **The asymmetry is the finding, and it needs no error bar to state.** At h1 the two seeds agree
to **4%**. At h36 they are on **opposite sides of zero**. Between-seed sd: **0.00013** at h1 against
**0.00492** at h36 — a factor of **38**.

⛔ **Why n=2 cannot name the seed count §6 asks for.** That h36 sd is estimated from **one degree of
freedom**. Its own 90% χ² interval is **[0.00251, 0.07842]** — a factor of **31**. Carried into a
power calculation at 90% confidence / 80% power for a mean effect of 0.005:

| σ used | seeds required |
|---|---|
| σ as measured (0.00492) | **6** |
| σ at the top of its own CI (0.07842) | **1,521** |

⇒ ⛔ **The data cannot distinguish these.** That is the honest discharge of §6's clause as of
EXP-02, and it is the entire justification for EXP-03. **EXP-03 buys σ at 5 dof instead of 1.**

---

## 2 · The intervention — seeds only. One variable, and it is n.

Eight new arms: seeds **44, 45, 46, 47** × {`HydraBNUNet06_LSTM4` (control), `UpsampleConv2`}, at
`total_lessons=300`, `ss_epsilon_max=0.0` — ⛔ **identical in every respect to EXP-02's four except
the seed.**

Binding, and each verified rather than asserted:

- ⛔ **One data pull.** `queryset_digest` **must** equal
  `b32a221b70e5255d3d85e683b483d9398ae08ed3b5e9d1be71c6e8f7aa13422c` on all **TWELVE** — the
  same scope correction AMENDMENT 1 made for diagnostics, which was not applied here
  either; a pooled σ̂ over two pulls is the `C-351` defect this document invokes twice. ✅
  EXP-02's four are verified identical (raw parquet md5 `d3636988b1085af74a08555c347ec5fe`
  on all four). And the raw
  parquet **md5 must match EXP-02's** `d3636988b1085af74a08555c347ec5fe`. Four arms on two pulls
  makes this a cross-population comparison (`C-351`), which is what `P-30`'s falsification already
  did to `P-27` once.
- ⛔ **S must equal 16 on all TWELVE** (AMENDMENT 1: the scored set is twelve, not eight),
  read off `y_pred.npy.shape[1]` in each arm's own cube, not
  off a manifest field. AP is quantised to S+1 rank levels; unequal S makes AP incomparable across
  arms (epic #263: *"the reference's S MUST equal the arms' cube width"*).
- ⛔ **`n_origins` must equal 13 on all TWELVE**, and each `upctwo` arm is differenced against the
  control **of its own seed**.
- ⛔ **No `h18` number appears anywhere in this document's verdict**, in either direction
  (§6 device 2, carried).

ℹ️ Cost, for the record: ~**$22** and ~6 h wall clock at EXP-02's measured rate (~$11 / 5h45m for
four arms, four pods in parallel).

---

## 3 · ⛔ THE VERDICT RULE AT n=6 — §6, specified, not amended

§6's three branches, **verbatim**, with the single substitution `both seeds → all six seeds`:

| branch | condition at n=6 |
|---|---|
| **CONFIRMS** | ΔAP > 0 at h1 **and** < 0 at h36, **all six seeds** |
| **FALSIFIES** | ΔAP **≥ 0 at h36 on all six seeds** |
| **MIXED** | seeds disagree at h36 |

⛔ **"All six", not "five of six", not "the mean".** The strict reading is chosen deliberately: a
specification written after seeing EXP-02's numbers must not be able to turn a MIXED into a
CONFIRMS, and only the strictest reading is immune to that charge. If achieved, unanimity at n=6
carries a binomial **p = 0.0156** on its own.

⚠️ **A mean test on ΔAP is registered here as SECONDARY and ⛔ CANNOT produce CONFIRMS.** It is
reported alongside, never instead, and the primary verdict is read off the table above. Stated now
so that preferring it later is visibly a change to this document.

---

## 4 · Pre-registered predictions

| id | prediction | threshold, declared now |
|---|---|---|
| **P-31** | the h36 between-seed sd, re-estimated at 5 dof, lands near its n=2 point estimate rather than anywhere in the 31× band — i.e. within a factor of ~1.6 either way of **0.00492** | σ̂(h36) ∈ **[0.0030, 0.0080]**, against an n=2 interval of [0.0025, 0.0784] |
| **P-32** | ⭐⭐ **THE ONE THAT MATTERS.** The h1/h36 sd asymmetry is **real and not an n=2 artifact** — the arm effect is near-deterministic at h1 and seed-dominated at h36 | **σ̂(h1) < σ̂(h36) / 3** |
| **P-33** | ΔAP @h1 is **positive on all six seeds** | sign test, all six |
| **P-34** | ⛔ **This experiment does NOT resolve `P-23`.** h36 signs are **not** unanimous ⇒ §6 returns **MIXED** a second time | sign test, stated as a failure in advance |
| **P-35** | the six-seed mean ΔAP @h36 is **not distinguishable from zero** | \|mean\| **<** its own between-seed 90% half-width |

⛔⛔ **`P-34` is declared deliberately, and it is the anti-rescue device of this document.** I am
predicting, before spending the money, that my own headline prediction will again fail to resolve.
⇒ If MIXED arrives I may **not** report it as a surprise, as *"under-powered therefore
inconclusive therefore still promising"*, or as grounds for a third seed purchase. Its measured
probability from EXP-02's own numbers is **~97%**: P(a seed lands negative at h36) = **0.506**.

---

## 5 · ⛔ The seed-count derivation, formula pre-committed

Reported from EXP-03's σ̂ with **no** post-hoc choice of estimator:

```
n_required = (z_0.95 + z_0.80)^2 * sigma_hat^2 / mu^2        # = 6.1826 * sigma^2 / mu^2
```

evaluated at **three** pre-named μ, and reported at σ̂ **and** at both ends of σ̂'s own χ²(5)
interval, so the answer is a range and is never quoted as a scalar
(`feedback_sweep_before_reporting_a_scalar`):

- **μ = 0.0050** — the effect size actually observed at h1
- **μ = 0.0035** — the larger of EXP-02's two h36 magnitudes
- **μ = 0.0400** — `P-24`'s own pre-registered threshold

⚠️ **And the pre-committed consequence of the answer.** If `n_required` at σ̂ exceeds **20 seeds**,
⛔ **the seed route is declared CLOSED on measurement** and no further seeds are bought — the
h36 endpoint is then not resolvable by replication at any affordable n, and that is the result, not
a reason to keep buying. ⚠️ What would reopen it: a change that reduces between-seed variance
itself, not more draws of it.

---

## 6 · What would make EXP-03 unreadable, named now

- ⛔ **A second data pull.** Any digest or md5 differing from §2's ⇒ the eight arms are not
  comparable to EXP-02's four and the pooled n=6 does not exist. **Verified per arm before scoring.**
- ⛔ **S ≠ 16 on any arm** ⇒ that arm's AP is on a different rank lattice. Excluded, and the
  exclusion reported.
- ⛔ **The gate not coming home** at both endpoints ⇒ verdict is **COULD NOT BE READ**, which is the
  honest outcome and **not** "the arm failed" (carried from the EXP-02 plan).
- ⚠️ **No guard enforces slicing the gate before the pod is destroyed.** It is prose in
  `runpod_run_guide.md:419-432` and an `xfail` stub. This remains the highest-consequence manual
  step in the programme.

---

## 7 · Devices carried forward from `PREREG.md` §6, unaltered

1. ⛔ An oracle-only / T=0-only improvement is a **FAIL**, declared in advance, and may not be
   reported as encouraging.
2. ⛔ No `h18` verdict, in either direction.
3. ⛔⛔ If the result is negative and I then wish to argue the test was inadequate, that argument is
   logged as a `W-` row against myself in `PREDICTION_LEDGER.md` **FIRST**. ⚠️ **This device has
   already fired once on this programme — `W-124`, 2026-11-03 — and it fired one message late.**
4. ⚠️ No branch above says **"closed"**, except §5's measured seed-route clause, which names in the
   same sentence what would reopen it.
5. ⛔ This document may not be changed after EXP-03's numbers exist.

---

## ⛔ AMENDMENT 1 — 2026-11-03, before any EXP-03 arm exists

⚠️ **Legitimate because EXP-03 has no numbers.** §7 device 5 forbids changing this document after
them; nothing has run. Recorded as an amendment rather than a silent edit.

### The defect this closes, found by the `models` session

§3 and §5 estimate σ at **5 dof** over **all six seeds**. ⇒ The scored set is not EXP-03's eight
arms. It is **twelve**: EXP-02's 42 and 43 plus EXP-03's 44–47. And **42/43 already ran with
`diagnostic_visualizations: False`** — verified four ways by `models`: the key in all four
`config_meta.py`, no `reports/plots` tree on any of the four pods, `visual_diagnostics.py:73`
returning False, and the Null Object path taken.

⛔ **So "run all eight EXP-03 arms the same way, with diagnostics on" reintroduces exactly the
mixture it was written to prevent** — four off-arms and eight on-arms inside one σ — displaced from
*within* EXP-03 to *across* the EXP-02/EXP-03 boundary. The argument is unchanged: if viz perturbs
the numerics by one RNG draw, the between-seed variance being bought is inflated by a difference
that tracks which experiment an arm came from, **and the failure looks exactly like the finding.**

⚠️ 42/43 cannot be re-run with diagnostics on at any sensible cost, so **they set the standard the
other ten must match**, not the reverse.

### ⛔ Binding, and it replaces the constraint as first stated

1. ⛔ **All twelve scored arms run `diagnostic_visualizations: False`.** Not because viz is
   suspected — the call sites in `training_engine.py` and `hydranet_inference.py` are plotting calls
   on already-computed tensors, guarded by `sample_idx == 0 and self.viz.active`, and
   `visual_diagnostics.py` makes no RNG call — but because **that is read off the code, not
   measured**, and this repo's pushforward audit already caught a diagnostic forward pass writing
   BatchNorm statistics. The exposure is removed rather than reasoned away.
2. ⭐ **Enforced mechanically, not remembered.** Before any ΔAP is computed,
   `diagnostic_visualizations` is read from **each of the twelve arms' own `config_meta.py`** and
   must be `False` in all twelve. ⇒ An arm built with `--diagnostics` that leaked into the set is
   caught by its own config, which `make_arch_arm.py::_verify` guarantees matches what was asked
   for. **This is the check that makes the probe below safe by construction.**
3. **The diagnostics-ON run is a probe, not a member.** One arm at seed 47 built with
   `--diagnostics`, run far enough to read `[Origin N/13]` timestamps for origins 1 and 2, then
   **killed, deleted, and rebuilt with diagnostics off** before it is ever scored. Cost ~2 origins,
   not an arm. Check 2 catches it if the rebuild is forgotten.
4. ⚠️ **What this does NOT buy.** Two origins of plots is a cost measurement and a first look, not a
   study. If a full diagnostics run is wanted to actually read, that is **one additional arm,
   deliberately outside the scored twelve and never entered into σ** (`models`' proposal, and the
   right shape for that case). ⛔ Not authorised here; it is a separate spend.

### ⚠️ One thing that stays unmeasured, stated so it is not mistaken for settled

Whether `diagnostic_visualizations: True` is numerically inert is **still unmeasured**. The cheap
test would be a diagnostics-on replica of an existing arm at the same seed, compared cube-to-cube —
but `_SEED_WORD` admits only 42–47, so a replica collides with the arm it replicates, and
`build` refuses to overwrite. ⚠️ And the test would be **asymmetric even if run**: identical cubes
would prove inertness, while differing cubes would prove nothing, because cross-pod cuDNN
nondeterminism is not excluded. ⇒ Not attempted. The constraint above does not depend on it.

---

## ⛔ AMENDMENT 2 — 2026-11-03, still before any EXP-03 arm exists

⚠️ Legitimate for the same reason as AMENDMENT 1: EXP-03 has no numbers. ⛔ Found by
`/code-review max` on the builder branch, which reviewed this document as well as the code.

### 1 · ⛔⛔ AMENDMENT 1 credited a check that did not exist. It exists now.

AMENDMENT 1 item 2 bound *"Before any ΔAP is computed, `diagnostic_visualizations` is read from
each of the twelve arms' own `config_meta.py` and must be `False` in all twelve"*, and item 3 then
leaned on it: *"Check 2 catches it if the rebuild is forgotten."* ⛔ **No code in either repository
implemented that sweep.** The only readers of the key were the seven dossier builders (at BUILD
time) and the hydranet runtime. `verify_exp02.py`, `verify_bakeoff.py` and `arm_identity_check.py`
do not contain the string at all. ⇒ The probe arm was safe by *memory*, which is what item 2 said
it was not.

⚠️ And `make_arch_arm.py::_verify` could never have been that check, for a reason worth stating:
it runs at **build** time, so it structurally cannot see an arm that leaked into the scored set or
one **reused** from an earlier build — `run_queue.sh:114` reuses on `[ -d ... ]` alone and
`arm_identity_check.py` reads only `config_hyperparameters.py`.

⇒ **The gate is `scripts/assert_diagnostics_off.py`**, committed with this amendment and covered by
16 tests. Invocation, binding before any ΔAP is computed::

    python scripts/assert_diagnostics_off.py --models-root ../views-models/models --expect 12 \
        ctlfullzero_fortytwo ctlfullzero_fortythree upctwofullzero_fortytwo \
        upctwofullzero_fortythree <the eight EXP-03 arms>

⭐ `--expect 12` is not decoration: a sweep over four clean arms is not evidence about twelve, and
without it the gate can be satisfied by naming fewer arms — the same scope error AMENDMENT 1 exists
to close, one level up. ⚠️ `--expect` also refuses **duplicate** arm names: naming one clean arm
twelve times satisfied the count and printed *"all 12 arms carry …=False"*, measured, which is the
same bypass one level further in.

⭐ It **parses** the literal rather than executing the config, and refuses an unverifiable value
instead of resolving it — because an absent key is exactly what `visual_diagnostics.py:73` treats
as off, so "nothing says True" is not the same as "this arm ran with it off".

⛔⛔ **And the first version of that parse was not sound, which is worth recording here rather than
only in the commit.** It walked every dict in the file, and two false-accepts were **measured**:

| shape | parse said | real value | consequence |
|---|---|---|---|
| `cfg = {K: False}` then `cfg[K] = True` | `False` | **`True`** | ⛔ the gate **PASSED** a tampered arm |
| `return {"inner": {K: True}}` | `True` | *absent* | ⛔ an arm asked for diagnostics shipped without |

The first is the direction that matters: a hand-edited arm passing the very gate written to catch
it. ⇒ The readback is now scoped to the dict `get_meta_config` **returns**, traces the returned
name back to its single assignment (the real configs are `meta_config = {…}` then `return
meta_config`, not a bare `return {literal}`), and refuses if anything could have mutated it after
construction — a subscript store, a `.update`, an augmented assignment, a decorator on the getter,
`**` unpacking, two assignments, two getters, or a non-literal value. ✅ Verified against all four
arms' real files plus the floor's: floor `True`, all four arms `False`.

⚠️ **The residual limit, stated rather than left implicit:** this is a static read. It cannot see
anything that changes the config at import time by a route the parse does not model. Every such
route it does not model refuses rather than guesses, so the failure direction is "refuses a valid
arm", not "passes an invalid one" — but that is a property of the cases enumerated above, not a
proof over all Python.
✅ Run against the four existing arms: **all four pass**, which is the first machine-checked
confirmation of AMENDMENT 1's "verified four ways".

### 2 · ⚠️ The probe's plots live inside the directory item 3 orders removed

AMENDMENT 1 item 3 says the seed-47 probe arm is *"killed, deleted, and rebuilt with diagnostics
off"*. ⛔ Its plots are written to `<arm>/reports/plots/diagnostics/<ts>/` — **inside the arm
directory** — and `build` refuses to overwrite an existing `dest`, so the rebuild requires deleting
it. ⇒ **Retrieval is sequenced BEFORE deletion, explicitly**, and this document already calls that
shape its highest-consequence manual step:

1. read the per-origin elapsed for origins 1–2 and kill the run
2. **pull the plots off the pod** (views-models `tools/podrun/stream_diagnostics.py`) and verify
   they are on local disk by content, not by count
3. only then delete the arm directory and rebuild with diagnostics off
4. `assert_diagnostics_off.py --expect 12` before any ΔAP

### 3 · ⚠️ Two figures withdrawn from §1 and from the builder's docstring

The emit baseline quoted as **"1.9–3.6 min/origin"** was relayed, described as parsed from
"`[Origin N/13]` timestamps in `exp02_arm.log`". ⛔ That log carries **no per-line timestamps** —
only tqdm bars with embedded elapsed fields — so the stated method cannot be what produced it, and
an independent parse here matched only 4–13 of 13 origins per arm and returned **1.25 min/origin**
for the arm quoted at 3.6, a ~3× disagreement. ⇒ **Withdrawn.** Both the "~28 min/origin" figure
and the "1.9–3.6" baseline are unverified, and the log that would settle either is gitignored, in
another repository, on destroyed pods. ⚠️ This does not change the probe plan — it is the reason
for it.
