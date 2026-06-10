# 02c — Expert Method Review: B1 pushforward first, or straight to B2 GTF?

**Date:** 2026-06-06 · **Skill:** `expert-method-review` · **Target:** `02_design.md §4.3` · **Chair:** simon (not seated)
**Trigger:** surfaced at the start of increment-2 implementation — ADR-056 scheduled sampling is already a proto-pushforward, so what does B1 add?

---

## 1. Target & decision under review
**One question:** is **B1 pushforward** the right *first* rung of the C-113 rollout-training fix, or should we go straight to **B2 GTF**?

**The decisive fact (verified this session):** scheduled sampling was **active** in the runs that exploded — `violet` config `ss_epsilon_max=0.25`, scheduled via `ss_mixer`. ADR-056 SS feeds `prev_pred.detach()` and scores the prediction-from-it ⇒ the **within-step operator gradient was already live** (last-step pushforward), at a 25% feed rate, **and the post-C-111 runaway happened anyway.** So B1 ("always-feed + an annealed stability term, gradient still last-step-only") is a *stronger dose of a mechanism that already failed at 25%*, not a new capability. **B2 GTF** is the genuinely new lever: un-detach + α-bound the **cross-step / through-time** gradient (scales every Jacobian by `(1−α)`).

**DGP/diagnosis to respect:** the io-gain probe found the free-running operator's **Jacobian gain `‖J‖₂ > 1`** with an **out-of-range attractor** the training trajectory never visits. That is a *dynamics* (product-of-Jacobians) property, not just an operating-point-distribution property.

## 2. Panel
| Seat | Why | Side |
|---|---|---|
| **Hochreiter** | through-time gradient / product-of-Jacobians; SS-failed is his kind of tell | B2 / dynamics |
| **DL-engineer** | cost/diff/memory — B1 is nearly free | B1-as-cheap-falsification |
| **Sutton** | bitter lesson; if the simple general method (train-on-own-output) failed, question the framing | reframe (maybe it's the link) |
| **Shi** | ConvLSTM nowcasting practice — how is multi-step rollout actually stabilised there | B2 (unrolled training) |

## 3. Library grounding
**Held & load-bearing:** `Brandstetter2022` (pushforward — note it trains at `A_# p_k`, *near the data*), `hess23a` (GTF — bounds the **Jacobian product** globally via `(1−α)`, Eq. 8), `NIPS-2015 scheduled-sampling` (the proto-pushforward we already ship), `Pascanu2013` (exploding/vanishing — clipping; the product-of-Jacobians), `MillerHardt2019` (stable RNN ≈ feed-forward — the expressiveness cost of contractivity), `Miyato2018` (spectral norm — a *direct* operator-norm bound), `Erichson2021`/`Chang2019`/`Arjovsky2016` (Lipschitz/antisymmetric/unitary — the architectural-stability family). **Gap → fetch:** `Mikhaeil2022` (chaotic-DS ill-posedness — load-bearing for whether GTF's α-bound is *needed* vs over-imported).

## 4. Independent critiques

### Hochreiter — *the product-of-Jacobians realist*
- **SS-failed is the diagnostic.** Last-step training optimises the one-step map at *visited* points; it does **not** control the product `∏ Jₖ` that governs multi-step divergence (Pascanu 2013). The runaway is that product growing — exactly what last-step (B1/SS) leaves untouched. **GTF's `(1−α)` scaling tames the product directly** (Hess Eq. 8). → **B2 is the on-target lever; B1 is more of the thing that already failed.**
- **But B2 reopens the wound:** un-detaching restores exploding/vanishing gradients — so α-bounding **and** gradient clipping (Pascanu) are mandatory, not optional.
- **The gain>1 is spectral** → a **spectral-norm / Lipschitz** constraint on the input→output operator (Miyato 2018 / Erichson 2021) is a *direct* complement the B1/B2/B3 ladder omits. Combine with training-level (belt-and-suspenders), since Hess argues architecture-alone is insufficient for chaotic targets.

### DL-engineer — *the throughput pragmatist*
- **B1 is nearly free** — it's turning up an existing knob (`ss_epsilon`→always) + adding a weighted term; flat memory in K. B2 is O(K) BPTT memory + α tuning + clipping.
- **So run B1 as a one-shot falsification, not a campaign.** Given SS@0.25 failed, the honest prior is low — but the cost to *learn* is one cheap run + the 30 s io-gain readout. If always-feed+term doesn't move the attractor in one run → escalate to B2 immediately. **Don't sink effort into B1; sink it into the B1→B2 gate.**
- Wants the dose quantified: SS was 25% feed; B1 is 100% + a term — is *that* delta expected to matter? Pre-register the expectation.

### Sutton — *the bitter-lesson contrarian*
- **The simple general method already ran and failed.** Training on the model's own outputs (SS) is the general move; at 25% it didn't prevent the runaway. "Add α-machinery" (B2) or "more SS knobs" (B1) are both *added structure* on a signal that didn't bite.
- **Question the framing:** if the dynamics diverge to an **out-of-range** attractor, maybe the problem is the **unbounded `expm1` output representation**, not the training signal. The **ZITD softplus link** (the *other* dossier) makes divergence *linear*, not exponential — a structural fix that doesn't need rollout-gradient cleverness. → this **pressure-tests the P4/C-129 "Axis B first" decision**: maybe ZITD (the link) should lead.
- Minimal-move alternative he'd still try before B2's α: **longer training windows / more compute** through the rollout.

### Shi — *the ConvLSTM nowcasting author*
- **Nowcasting stabilises multi-step rollout by unrolling the forecaster and training through it** (BPTT through the rollout) — i.e. **cross-step gradient (B2-flavoured)**, *not* last-step-only. The field precedent is against B1-as-sufficient.
- Practical: use **truncated BPTT** (K window) + clipping — exactly B2-with-TBPTT. This is well-trodden for spatiotemporal ConvLSTM.
- Reiterates the **blurring** caveat (multi-step optimisation → mean-hedging): applies to B1 and B2 alike; keep the calibration/sharpness readout (C-126).

## 5. Key disagreements
- **B1-first vs B2-first.** DL-engineer: B1 is so cheap, run it as a falsification gate. Hochreiter + Shi: the evidence (SS-failed + gain>1 product-of-Jacobians + nowcasting precedent) says the cross-step gradient (B2) is the actual missing ingredient — B1 is predicted to fail. *Merit:* both right — B1 is cheap *to falsify*, but its *success* prior is genuinely low; the resolution is to run it **as an explicitly-low-prior one-shot gate**, not as "the fix."
- **Training-signal vs representation (the deepest one).** Sutton: maybe neither B1 nor B2 is the fix — the unbounded `expm1` link is, and the **ZITD softplus head should lead** (reopens P4/C-129). vs the rest: rollout training and the link are complementary; fix the dynamics. *Merit:* Sutton's point is strong *because* the attractor is out-of-range (a link that makes divergence linear would defang it) — this is a real challenge to "Axis B first."
- **Architecture lever omitted.** Hochreiter: spectral-norm/Lipschitz directly bounds the diagnosed `‖J‖₂>1` and belongs in the ladder; the design treats it only as a passing mention in §3.

## 6. Synthesis & recommendation
1. **Reframe B1 (don't cancel it).** The SS-already-failed fact lowers B1's success prior sharply. `05`'s `F1 → B2` escalation is therefore the **expected path, not a fallback** — make that explicit: B1 is a **one-shot, low-prior falsification gate** (cheap: always-feed+term, one run + io-gain readout). If it doesn't move the attractor → B2 immediately. *Pre-register the low prior* so a B1 failure isn't a surprise or a sunk cost.
2. **Promote B2 GTF (+ clipping) to the likely-real fix** — and treat **spectral-norm/Lipschitz** (Miyato/Erichson) as a **peer lever / complement**, since the diagnosis is literally operator gain >1. Add it to the B1/B2/B3 ladder rather than a footnote.
3. **Re-open the P4/C-129 sequencing as a genuine question, not a settled "Axis B first."** Sutton's link argument + the out-of-range-attractor diagnosis are strong enough that "ZITD softplus first" deserves an honest comparison. *Strongest dissent to keep live.*
4. **Fetch `Mikhaeil2022`** before committing to B2's α-bound *as theory* (vs heuristic).

**Strongest dissent (carry live):** *the runaway may be an output-representation problem (unbounded `expm1`), in which case the ZITD softplus link — not rollout-gradient training — is the fix, and "Axis B first" (C-129) is wrong.*

## 7. Methodological risks (register-compatible — for `register-risk`)
| ID | Tier | Trigger | Location | Narrative |
|----|------|---------|----------|-----------|
| **RT-B1prior** | 3 | Running/scaling B1 as "the fix" without pre-registering that SS (proto-pushforward) already failed at ε=0.25 ⇒ B1's success prior is low | `02_design §4.3`, `05` | Last-step operator training already ran (SS) and didn't prevent the runaway; B1 is a stronger dose of the same. Treat as a one-shot low-prior falsification gate with automatic B2 escalation, not a campaign. |
| **RT-spectral** | 3 | Building the B1/B2/B3 ladder without the direct operator-norm lever | `02_design §4.2`, §3 | The diagnosis is `‖J‖₂>1`; spectral-norm/Lipschitz (Miyato/Erichson) bounds it directly and composes with B1/B2 — it's omitted from the ladder. |
| **RT-P4reopen** | 3 | Treating "Axis B first" (C-129) as settled when the out-of-range attractor implies the unbounded `expm1` link may be the real cause | `02_design §4.3`, C-129 | The ZITD softplus link makes divergence linear-not-exponential — a structural cure that may dominate rollout-gradient training. The P4 sequencing should be an explicit comparison, not an assumption. |
